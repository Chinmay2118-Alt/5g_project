"""
baselines/exact_minlp.py
Responsibility: global solution of P0 by spatial branch-and-bound (SCIP), small instances only.
Returns the best solution found plus a certified upper bound, so the optimality gap of every
other algorithm can be measured. Probabilistic QoS is not part of the model (see docs/BASELINES.md).
Paper mapping: Sec. XVII (P0)
"""

import copy
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import numpy as np

from sim.problem import BeamSolution
from sim.rates import as_assoc_matrix

LN2 = np.log(2.0)
MAX_REAL_BEAM_VARIABLES = 40
WORKER_MARGIN_S = 60.0  # model building and start-up, on top of SCIP's own time limit


def small_instance_config(config):
    """The config restricted to the small network named in its `exact` section."""
    small = copy.deepcopy(config)
    exact = config["exact"]
    for key in ("num_aps", "num_relays", "num_ues", "n_antennas_ap", "n_antennas_relay", "cluster_top_l", "max_streams_per_relay"):
        small["network"][key] = exact[key]
    small["blockage"]["frac_blocked_ues"] = exact["frac_blocked_ues"]
    return small


def _complex_var(model, name, bound):
    return (model.addVar(f"{name}_re", lb=-bound, ub=bound), model.addVar(f"{name}_im", lb=-bound, ub=bound))


def _quicksum_gain(terms):
    # Each term is (complex coefficient, (re var, im var)); returns Re and Im of coef * var.
    real = sum(c.real * re - c.imag * im for c, (re, im) in terms)
    imag = sum(c.real * im + c.imag * re for c, (re, im) in terms)
    return real, imag


def _add_rate(model, name, desired, interferers, noise_var, gamma_ub, bits_ub):
    """Adds bits <= log2(1 + gamma) with gamma (I + noise) <= |desired|^2 and I >= sum |interferer|^2.

    The inequalities are tight at any optimum since the objective increases with every rate.
    """
    gamma = model.addVar(f"gamma_{name}", lb=0.0, ub=gamma_ub)
    signal = model.addVar(f"signal_{name}", lb=0.0, ub=gamma_ub * noise_var)
    interference = model.addVar(f"interf_{name}", lb=0.0, ub=None)
    bits = model.addVar(f"bits_{name}", lb=0.0, ub=bits_ub)
    model.addCons(signal <= desired[0] * desired[0] + desired[1] * desired[1])
    if interferers:
        model.addCons(interference >= sum(re * re + im * im for re, im in interferers))
    model.addCons(gamma * (interference + noise_var) <= signal)
    from pyscipopt import log as scip_log
    model.addCons(bits * LN2 <= scip_log(1.0 + gamma))
    return bits


def solve_exact(instance, time_limit_s=None, gap_limit=None, optimize_association=None):
    """Solve P0 with beams, combiners, rates and (optionally) association as decision variables; tau is fixed.

    SCIP runs in its own Python process: PyTorch and SCIP each bundle an OpenMP runtime, and the
    process aborts if both are active. KMP_DUPLICATE_LIB_OK is documented as able to silently
    corrupt results, which a ground-truth solver cannot accept.
    """
    settings = instance.config.get("exact", {})
    time_limit_s = settings.get("time_limit_s", 300) if time_limit_s is None else time_limit_s
    with tempfile.TemporaryDirectory() as folder:
        request, response = Path(folder) / "request.pkl", Path(folder) / "response.pkl"
        request.write_bytes(pickle.dumps((instance, time_limit_s, gap_limit, optimize_association)))
        finished = subprocess.run(
            [sys.executable, "-m", "baselines.exact_minlp", str(request), str(response)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            timeout=float(time_limit_s) + WORKER_MARGIN_S,
        )
        if finished.returncode != 0:
            message = finished.stderr.strip().splitlines()[-1] if finished.stderr.strip() else f"exit code {finished.returncode}"
            if message.startswith("ValueError: "):
                raise ValueError(message.removeprefix("ValueError: "))
            raise RuntimeError(f"exact solver process failed: {message}")
        return pickle.loads(response.read_bytes())


def _solve_in_this_process(instance, time_limit_s, gap_limit, optimize_association):
    from pyscipopt import Model, log as scip_log

    scenario, channels, config = instance.scenario, instance.channels, instance.config
    network, power = config["network"], config["power"]
    settings = config.get("exact", {})
    gap_limit = settings.get("gap_limit", 0.01) if gap_limit is None else gap_limit
    optimize_association = settings.get("optimize_association", False) if optimize_association is None else optimize_association

    M, R, K = network["num_aps"], network["num_relays"], network["num_ues"]
    N_A, N_R = network["n_antennas_ap"], network["n_antennas_relay"]
    fixed_assoc = as_assoc_matrix(scenario.relay_assoc, R)
    num_real = 2 * (int(scenario.cluster_mask.sum()) * N_A + (R * K if optimize_association else int(fixed_assoc.sum())) * N_R)
    if num_real > MAX_REAL_BEAM_VARIABLES:
        raise ValueError(f"{num_real} real beam variables is too large for spatial branch-and-bound (limit {MAX_REAL_BEAM_VARIABLES})")

    tau = instance.tau
    P_a, P_r = power["p_max_ap"], power["p_max_relay"]
    sigma_ue, sigma_relay = scenario.noise_var_ue, scenario.noise_var_relay
    model = Model("rap_fedbeam_p0")
    model.hideOutput()
    model.setRealParam("limits/time", float(time_limit_s))
    model.setRealParam("limits/gap", float(gap_limit))

    # --- association ---
    if optimize_association:
        assoc_var = {(r, k): model.addVar(f"a_{r}_{k}", vtype="B") for r in range(R) for k in range(K)}
        for k in range(K):
            model.addCons(sum(assoc_var[r, k] for r in range(R)) <= network.get("l_max", 1))
        for r in range(R):
            model.addCons(sum(assoc_var[r, k] for k in range(K)) <= network.get("max_streams_per_relay", K))
        relay_pairs = [(r, k) for r in range(R) for k in range(K)]
    else:
        assoc_var = None
        relay_pairs = [(int(r), int(k)) for r, k in zip(*np.nonzero(fixed_assoc))]

    # --- beams ---
    # Each stream's common phase changes no |gain|, so fixing it removes a symmetry of the search.
    w, v = {}, {}
    for k in range(K):
        fixed_phase = False
        for m in range(M):
            if not scenario.cluster_mask[m, k]:
                continue
            for a in range(N_A):
                w[m, k, a] = _complex_var(model, f"w_{m}_{k}_{a}", np.sqrt(P_a))
                if not fixed_phase:
                    model.chgVarLb(w[m, k, a][0], 0.0)
                    model.chgVarLb(w[m, k, a][1], 0.0)
                    model.chgVarUb(w[m, k, a][1], 0.0)
                    fixed_phase = True
    for r, k in relay_pairs:
        for n in range(N_R):
            v[r, k, n] = _complex_var(model, f"v_{r}_{k}_{n}", np.sqrt(P_r))
        model.chgVarLb(v[r, k, 0][0], 0.0)
        model.chgVarLb(v[r, k, 0][1], 0.0)
        model.chgVarUb(v[r, k, 0][1], 0.0)
    for m in range(M):
        entries = [w[m, k, a] for k in range(K) for a in range(N_A) if (m, k, a) in w]
        if entries:
            model.addCons(sum(re * re + im * im for re, im in entries) <= P_a)
    for r in range(R):
        entries = [v[rel, k, n] for (rel, k, n) in v if rel == r]
        if entries:
            model.addCons(sum(re * re + im * im for re, im in entries) <= P_r)
    if optimize_association:
        for r, k in relay_pairs:
            model.addCons(sum(v[r, k, n][0] * v[r, k, n][0] + v[r, k, n][1] * v[r, k, n][1] for n in range(N_R)) <= assoc_var[r, k] * P_r)

    # --- combiners: unit norm with fixed phase; a single receive antenna needs no variable ---
    u = {}
    for r, k in relay_pairs:
        if N_R == 1:
            u[r, k, 0] = (1.0, 0.0)
        else:
            for n in range(N_R):
                u[r, k, n] = _complex_var(model, f"u_{r}_{k}_{n}", 1.0)
            model.chgVarLb(u[r, k, 0][0], 0.0)
            model.chgVarLb(u[r, k, 0][1], 0.0)
            model.chgVarUb(u[r, k, 0][1], 0.0)
            model.addCons(sum(u[r, k, n][0] * u[r, k, n][0] + u[r, k, n][1] * u[r, k, n][1] for n in range(N_R)) == 1.0)

    # --- rates ---
    rate = {k: model.addVar(f"r_{k}", lb=0.0) for k in range(K)}
    util = {k: model.addVar(f"t_{k}", lb=np.log(instance.epsilon_0), ub=None) for k in range(K)}
    relayable = {k: [r for r, j in relay_pairs if j == k] for k in range(K)}

    for k in range(K):
        gains = {}
        for j in range(K):
            terms = [(np.conj(channels.h_hat[m, k, a]), w[m, j, a]) for m in range(M) for a in range(N_A) if (m, j, a) in w]
            gains[j] = _quicksum_gain(terms) if terms else (0.0, 0.0)
        gamma_ub = sum(np.linalg.norm(channels.h_hat[m, k]) * np.sqrt(P_a) for m in range(M) if scenario.cluster_mask[m, k]) ** 2 / sigma_ue
        bits1 = _add_rate(model, f"1_{k}", gains[k], [gains[j] for j in range(K) if j != k], sigma_ue, gamma_ub, np.log2(1.0 + gamma_ub))
        destination = tau * bits1

        if relayable[k]:
            gains2 = {}
            for j in range(K):
                terms = [(np.conj(channels.g_hat[r, k, n]), v[r, j, n]) for r in relayable[j] for n in range(N_R)]
                gains2[j] = _quicksum_gain(terms) if terms else (0.0, 0.0)
            gamma_ub2 = (sum(np.linalg.norm(channels.g_hat[r, k]) for r in relayable[k]) * np.sqrt(P_r)) ** 2 / sigma_ue
            bits2 = _add_rate(model, f"2_{k}", gains2[k], [gains2[j] for j in range(K) if j != k], sigma_ue, gamma_ub2, np.log2(1.0 + gamma_ub2))
            destination = destination + (1.0 - tau) * bits2

            for r in relayable[k]:
                gains_ar = {}
                for j in range(K):
                    # u^H G w with both u and w variable: a bilinear form per receive antenna
                    real, imag = 0.0, 0.0
                    for n in range(N_R):
                        terms = [(channels.G_hat[m, r, n, a], w[m, j, a]) for m in range(M) for a in range(N_A) if (m, j, a) in w]
                        y_re, y_im = _quicksum_gain(terms) if terms else (0.0, 0.0)
                        u_re, u_im = u[r, k, n]
                        real = real + u_re * y_re + u_im * y_im
                        imag = imag + u_re * y_im - u_im * y_re
                    gains_ar[j] = (real, imag)
                gamma_ub_ar = sum(np.linalg.norm(channels.G_hat[m, r], 2) * np.sqrt(P_a) for m in range(M) if scenario.cluster_mask[m, k]) ** 2 / sigma_relay
                bits_ar = _add_rate(model, f"ar_{r}_{k}", gains_ar[k], [gains_ar[j] for j in range(K) if j != k], sigma_relay, gamma_ub_ar, np.log2(1.0 + gamma_ub_ar))
                if optimize_association:
                    big_m = tau * np.log2(1.0 + gamma_ub) + (1.0 - tau) * np.log2(1.0 + gamma_ub2)
                    model.addCons(rate[k] <= tau * bits_ar + big_m * (1 - assoc_var[r, k]))
                else:
                    model.addCons(rate[k] <= tau * bits_ar)

        model.addCons(rate[k] <= destination)
        model.addCons(util[k] <= scip_log(rate[k] + instance.epsilon_0))

    objective = sum(float(instance.ue_weights[k]) * util[k] for k in range(K))
    lambda_e = float(config.get("objective", {}).get("lambda_e", 0.0))
    if lambda_e > 0.0:
        ptx = model.addVar("ptx", lb=0.0)
        model.addCons(ptx >= sum((re * re + im * im) / power["pa_efficiency_ap"] for re, im in w.values())
                      + sum((re * re + im * im) / power["pa_efficiency_relay"] for re, im in v.values()))
        objective = objective - lambda_e * ptx
    model.setObjective(objective, "maximize")
    model.optimize()

    if model.getNSols() == 0:
        raise RuntimeError(f"SCIP found no feasible solution (status: {model.getStatus()})")
    solution = model.getBestSol()

    def complex_value(pair):
        return model.getSolVal(solution, pair[0]) + 1j * model.getSolVal(solution, pair[1])

    W = np.zeros((M, K, N_A), dtype=complex)
    V = np.zeros((R, K, N_R), dtype=complex)
    U = np.zeros((R, K, N_R), dtype=complex)
    for key, pair in w.items():
        W[key] = complex_value(pair)
    if optimize_association:
        chosen = np.zeros((R, K), dtype=int)
        for (r, k), var in assoc_var.items():
            chosen[r, k] = int(model.getSolVal(solution, var) > 0.5)
        if network.get("l_max", 1) > 1:
            assoc = chosen
        else:
            assoc = np.full(K, -1, dtype=int)
            for r, k in zip(*np.nonzero(chosen)):
                assoc[k] = r
    else:
        chosen = fixed_assoc
        assoc = np.asarray(scenario.relay_assoc).copy()
    for (r, k, n), pair in v.items():
        if chosen[r, k]:
            V[r, k, n] = complex_value(pair)
    for r, k in relay_pairs:
        if not chosen[r, k]:
            continue
        if N_R == 1:
            U[r, k, 0] = 1.0
        else:
            for n in range(N_R):
                U[r, k, n] = complex_value(u[r, k, n])
            U[r, k] /= max(np.linalg.norm(U[r, k]), 1e-12)
    # SCIP's feasibility tolerance can exceed a power budget slightly; pull each node back.
    for m in range(M):
        used = float(np.sum(np.abs(W[m]) ** 2))
        if used > P_a:
            W[m] *= np.sqrt(P_a / used)
    for r in range(R):
        used = float(np.sum(np.abs(V[r]) ** 2))
        if used > P_r:
            V[r] *= np.sqrt(P_r / used)

    info = {
        "status": model.getStatus(),
        "upper_bound": float(model.getDualbound()),
        "incumbent": float(model.getPrimalbound()),
        "gap": float(model.getGap()),
        "nodes": int(model.getNNodes()),
        "solve_seconds": float(model.getSolvingTime()),
    }
    return BeamSolution(W, V, U, tau, assoc, info=info)


if __name__ == "__main__":
    request_path, response_path = Path(sys.argv[1]), Path(sys.argv[2])
    arguments = pickle.loads(request_path.read_bytes())
    response_path.write_bytes(pickle.dumps(_solve_in_this_process(*arguments)))
