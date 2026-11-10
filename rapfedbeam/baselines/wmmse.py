"""
baselines/wmmse.py
Responsibility: the WMMSE family of baselines for P0 with fixed association.
  - centralized_wmmse: full CSI at the CPU, alternating optimization where each beam update is
    a convex program (WMMSE lower bound, SCA in the rates). Monotone in the PF utility.
  - local_wmmse: every node uses only its own CSI, with no FL and no coordination.
  - wmmse_node: closed-form single-node update, reusable by a distributed algorithm.
Paper mapping: Sec. XIV (DF rate), Sec. XVII (P0). Derivation: docs/WMMSE_DERIVATION.md
"""

import warnings

import numpy as np

from eval.metrics import pf_utility
from sim.problem import BeamSolution
from sim.rates import (ap_target_channels, build_beams_from_power_coeffs, compute_full_rates, mmse_combiners, mrt_beam_directions)
from baselines.heuristics import equal_power, relay_power_equal_or_fractional

LN2 = np.log(2.0)


def _objective(instance, W, V, U, tau, assoc):
    scenario = instance.scenario
    if not np.array_equal(assoc, scenario.relay_assoc):
        scenario = scenario.copy()
        scenario.relay_assoc = assoc
    # Algorithms may only see the estimates, so the true channels are swapped out here.
    channels = _estimate_view(instance.channels)
    rates, debug = compute_full_rates(scenario, channels, W, V, U, tau)
    power = instance.config["power"]
    lambda_e = float(instance.config.get("objective", {}).get("lambda_e", 0.0))
    utility = float(pf_utility(rates, instance.ue_weights, instance.epsilon_0))
    utility -= lambda_e * (np.sum(np.abs(W) ** 2) / power["pa_efficiency_ap"] + np.sum(np.abs(V) ** 2) / power["pa_efficiency_relay"])
    return utility, rates, debug


def _estimate_view(channels):
    view = type(channels)(channels.h_hat, channels.G_hat, channels.g_hat, channels.h_hat, channels.G_hat, channels.g_hat)
    return view


def _initial_beams(instance, assoc):
    scenario, channels, config = instance.scenario, instance.channels, instance.config
    network, power = config["network"], config["power"]
    targets = ap_target_channels(channels.h_hat, channels.G_hat, assoc)
    W = build_beams_from_power_coeffs(mrt_beam_directions(targets), equal_power(scenario.cluster_mask, power["p_max_ap"]), power["p_max_ap"])
    relay_coeffs = relay_power_equal_or_fractional(assoc, power["p_max_relay"], network["num_relays"])
    V = build_beams_from_power_coeffs(mrt_beam_directions(channels.g_hat), relay_coeffs, power["p_max_relay"])
    return W, V


class _Layout:
    """Maps the active beam entries (c_mk = 1 or a_rk = 1) to one complex vector z."""

    def __init__(self, instance, assoc):
        scenario, network = instance.scenario, instance.config["network"]
        self.num_ues = network["num_ues"]
        self.ap_vars = [(m, k, a) for m in range(network["num_aps"]) for k in range(self.num_ues)
                        for a in range(network["n_antennas_ap"]) if scenario.cluster_mask[m, k]]
        self.relay_vars = [(int(assoc[k]), k, n) for k in range(self.num_ues) if assoc[k] >= 0
                           for n in range(network["n_antennas_relay"])]
        self.size = len(self.ap_vars) + len(self.relay_vars)
        self.stream = np.array([var[1] for var in self.ap_vars + self.relay_vars], dtype=int)
        self.is_ap = np.array([True] * len(self.ap_vars) + [False] * len(self.relay_vars))

    def pack(self, W, V):
        return np.array([W[var] for var in self.ap_vars] + [V[var] for var in self.relay_vars], dtype=complex)

    def unpack(self, z, W_shape, V_shape):
        W, V = np.zeros(W_shape, dtype=complex), np.zeros(V_shape, dtype=complex)
        for i, var in enumerate(self.ap_vars):
            W[var] = z[i]
        for i, var in enumerate(self.relay_vars):
            V[var] = z[len(self.ap_vars) + i]
        return W, V

    def node_groups(self, instance):
        network = instance.config["network"]
        groups = [([i for i, var in enumerate(self.ap_vars) if var[0] == m], instance.config["power"]["p_max_ap"])
                  for m in range(network["num_aps"])]
        offset = len(self.ap_vars)
        groups += [([offset + i for i, var in enumerate(self.relay_vars) if var[0] == r], instance.config["power"]["p_max_relay"])
                   for r in range(network["num_relays"])]
        return [(indices, p_max) for indices, p_max in groups if indices]


def _link_bound(row, k, alpha, noise_term, xi, layout):
    """WMMSE bound on log2(1 + SINR) of stream k as const + f'x - ||Qx||^2, with z = x_re + i x_im.

    Here s_j = sum over stream j of row * z, and the MSE is
    e = |1 - alpha s_k|^2 + sum_{j != k} |alpha s_j|^2 + noise_term.
    The bound (ln xi + 1 - xi e) / ln 2 touches the true rate at the point xi was computed from.
    """
    const = (np.log(xi) + 1.0 - xi * (1.0 + noise_term)) / LN2
    desired = alpha * row * (layout.stream == k)
    f = (2.0 * xi / LN2) * np.concatenate([desired.real, -desired.imag])
    scale = abs(alpha) * np.sqrt(xi / LN2)
    rows = []
    for j in range(layout.num_ues):
        part = scale * row * (layout.stream == j)
        if np.any(part):
            rows.append(np.concatenate([part.real, -part.imag]))
            rows.append(np.concatenate([part.imag, part.real]))
    Q = np.array(rows) if rows else np.zeros((1, 2 * layout.size))
    return const, f, Q


def _ue_receiver(row, z, k, noise_var, layout):
    gains = np.array([np.sum((row * z)[layout.stream == j]) for j in range(layout.num_ues)])
    total = np.sum(np.abs(gains) ** 2) + noise_var
    a = gains[k] / total
    mse = 1.0 - (abs(gains[k]) ** 2) / total
    return np.conj(a), abs(a) ** 2 * noise_var, 1.0 / max(mse, 1e-12)


def _solve_beam_update(instance, layout, z, tau, assoc):
    """One convex update of all beams and rates; every other block is held at its MMSE value."""
    import cvxpy as cp

    scenario, channels, config = instance.scenario, instance.channels, instance.config
    network, power = config["network"], config["power"]
    # The step d = x - x0 is the variable: at high SINR xi is large, and written around x0 the
    # bound's large terms cancel in closed form instead of inside the solver.
    x0 = np.concatenate([z.real, z.imag])
    d = cp.Variable(2 * layout.size)
    x = x0 + d
    r = cp.Variable(layout.num_ues)
    constraints = []

    def bound_expression(const, f, Q):
        Qx0 = Q @ x0
        return const + f @ x0 - float(Qx0 @ Qx0) + (f - 2.0 * Q.T @ Qx0) @ d - cp.sum_squares(Q @ d)

    W_now, _ = layout.unpack(z, (network["num_aps"], layout.num_ues, network["n_antennas_ap"]), (network["num_relays"], layout.num_ues, network["n_antennas_relay"]))
    effective = np.einsum("mrna,mka->rnk", channels.G_hat, W_now)

    for k in range(layout.num_ues):
        row1 = np.array([np.conj(channels.h_hat[m, k, a]) for m, _, a in layout.ap_vars] + [0.0] * len(layout.relay_vars), dtype=complex)
        alpha, noise_term, xi = _ue_receiver(row1, z, k, scenario.noise_var_ue, layout)
        const, f, Q = _link_bound(row1, k, alpha, noise_term, xi, layout)
        const, f, Q = tau * const, tau * f, np.sqrt(tau) * Q

        relay_id = int(assoc[k])
        if relay_id >= 0:
            row2 = np.array([0.0] * len(layout.ap_vars) + [np.conj(channels.g_hat[rel, k, n]) for rel, _, n in layout.relay_vars], dtype=complex)
            alpha2, noise_term2, xi2 = _ue_receiver(row2, z, k, scenario.noise_var_ue, layout)
            const2, f2, Q2 = _link_bound(row2, k, alpha2, noise_term2, xi2, layout)
            const, f = const + (1.0 - tau) * const2, f + (1.0 - tau) * f2
            Q = np.vstack([Q, np.sqrt(1.0 - tau) * Q2])

            # Relay decoding: a full MMSE combiner c replaces the scalar receiver (alpha = 1).
            covariance = effective[relay_id] @ effective[relay_id].conj().T + scenario.noise_var_relay * np.eye(network["n_antennas_relay"])
            c = np.linalg.solve(covariance, effective[relay_id][:, k])
            mse_ar = 1.0 - float(np.real(np.conj(effective[relay_id][:, k]) @ c))
            row_ar = np.array([np.conj(c) @ channels.G_hat[m, relay_id, :, a] for m, _, a in layout.ap_vars] + [0.0] * len(layout.relay_vars), dtype=complex)
            const_ar, f_ar, Q_ar = _link_bound(row_ar, k, 1.0, float(np.sum(np.abs(c) ** 2)) * scenario.noise_var_relay, 1.0 / max(mse_ar, 1e-12), layout)
            constraints.append(r[k] <= bound_expression(tau * const_ar, tau * f_ar, np.sqrt(tau) * Q_ar))

        constraints.append(r[k] <= bound_expression(const, f, Q))

    for indices, p_max in layout.node_groups(instance):
        selected = np.array(indices)
        constraints.append(cp.sum_squares(x[np.concatenate([selected, layout.size + selected])]) <= p_max)

    objective = cp.sum(cp.multiply(instance.ue_weights, cp.log(r + instance.epsilon_0)))
    lambda_e = float(config.get("objective", {}).get("lambda_e", 0.0))
    if lambda_e > 0.0:
        ap_idx = np.flatnonzero(layout.is_ap)
        relay_idx = np.flatnonzero(~layout.is_ap)
        objective -= lambda_e / power["pa_efficiency_ap"] * cp.sum_squares(x[np.concatenate([ap_idx, layout.size + ap_idx])])
        if len(relay_idx):
            objective -= lambda_e / power["pa_efficiency_relay"] * cp.sum_squares(x[np.concatenate([relay_idx, layout.size + relay_idx])])

    problem = cp.Problem(cp.Maximize(objective), constraints)
    try:
        with warnings.catch_warnings():
            # An inaccurate step is still safe: the caller rejects any step that lowers the utility.
            warnings.filterwarnings("ignore", message="Solution may be inaccurate")
            problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return None
    if d.value is None or problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None
    x_new = x0 + d.value
    z_new = x_new[:layout.size] + 1j * x_new[layout.size:]
    # Solver tolerance can leave a node a hair over budget; pull it back so feasibility is exact.
    for indices, p_max in layout.node_groups(instance):
        used = float(np.sum(np.abs(z_new[indices]) ** 2))
        if used > p_max:
            z_new[indices] *= np.sqrt(p_max / used)
    return z_new


def _best_tau(instance, debug_rates, tau):
    """PF utility is concave in tau for fixed beams, so a refined grid search is enough."""
    phase = instance.config["phase"]
    gamma1, gamma2, gamma_ar, assoc = debug_rates
    grid = np.linspace(phase.get("tau_min", 0.1), phase.get("tau_max", 0.9), 121)
    candidates = np.append(grid, tau)
    best_tau, best_value = tau, -np.inf
    for candidate in candidates:
        rates = candidate * np.log2(1.0 + gamma1) + (1.0 - candidate) * np.log2(1.0 + gamma2)
        for k, relay_id in enumerate(assoc):
            if relay_id >= 0:
                rates[k] = min(rates[k], candidate * np.log2(1.0 + gamma_ar[relay_id, k]))
        value = float(pf_utility(rates, instance.ue_weights, instance.epsilon_0))
        if value > best_value:
            best_tau, best_value = float(candidate), value
    return best_tau


def centralized_wmmse(instance, use_relay=True, max_iters=None, tol=None, optimize_tau=None):
    """Centralized WMMSE-SCA/AO. Each accepted iterate improves the PF utility on the estimates."""
    scenario, config = instance.scenario, instance.config
    settings = config.get("wmmse", {})
    max_iters = settings.get("max_iters", 30) if max_iters is None else max_iters
    tol = settings.get("tol", 1e-3) if tol is None else tol
    optimize_tau = settings.get("optimize_tau", False) if optimize_tau is None else optimize_tau
    assoc = np.asarray(scenario.relay_assoc) if use_relay else np.full(config["network"]["num_ues"], -1)
    tau = instance.tau

    W, V = _initial_beams(instance, assoc)
    layout = _Layout(instance, assoc)
    U = mmse_combiners(instance.channels.G_hat, W, scenario.noise_var_relay)
    utility, _, _ = _objective(instance, W, V, U, tau, assoc)
    history = [utility]
    z = layout.pack(W, V)
    status = "max_iters"

    for _ in range(max_iters):
        z_new = _solve_beam_update(instance, layout, z, tau, assoc)
        if z_new is None:
            status = "solver_failed"
            break
        W_new, V_new = layout.unpack(z_new, W.shape, V.shape)
        U_new = mmse_combiners(instance.channels.G_hat, W_new, scenario.noise_var_relay)
        tau_new = tau
        utility_new, _, debug = _objective(instance, W_new, V_new, U_new, tau, assoc)
        if optimize_tau:
            tau_new = _best_tau(instance, (debug["gamma1"], debug["gamma2"], debug["gamma_AR"], assoc), tau)
            utility_new, _, _ = _objective(instance, W_new, V_new, U_new, tau_new, assoc)
        if utility_new <= utility:
            # Only solver tolerance can cause this; keep the better iterate.
            status = "converged"
            break
        improvement = utility_new - utility
        W, V, U, z, tau, utility = W_new, V_new, U_new, z_new, tau_new, utility_new
        history.append(utility)
        if improvement < tol:
            status = "converged"
            break

    return BeamSolution(W, V, U, tau, assoc, info={"history": history, "status": status})


def wmmse_node(b, noise_var, p_max, phase_scale, weights, epsilon_0, iters=100, power_price=0.0):
    """Closed-form WMMSE for one transmitter serving its users with rates phase_scale * log2(1 + SINR).

    Args:
        b:           complex [K', N], effective channel of each served user (gain is b^H w)
        noise_var:   receiver noise variance
        p_max:       the node's power budget
        phase_scale: real [K'], time share of each user's rate
        weights:     real [K'], omega_k
        power_price: lambda on ||w||^2 in the objective
    Returns beams, complex [K', N].
    """
    b = np.asarray(b, dtype=complex)
    num_users, num_antennas = b.shape
    if num_users == 0:
        return np.zeros((0, num_antennas), dtype=complex)
    norms = np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    w = b / norms * np.sqrt(p_max / num_users)
    phase_scale, weights = np.asarray(phase_scale, dtype=float), np.asarray(weights, dtype=float)

    def stats(beams):
        gains = b.conj() @ beams.T
        total = np.sum(np.abs(gains) ** 2, axis=1) + noise_var
        desired = np.diag(gains)
        a = desired / total
        mse = 1.0 - np.abs(desired) ** 2 / total
        return a, np.maximum(mse, 1e-12)

    def utility(beams):
        _, mse = stats(beams)
        return float(np.sum(weights * np.log(phase_scale * np.log2(1.0 / mse) + epsilon_0)) - power_price * np.sum(np.abs(beams) ** 2))

    best_w, best_value = w, utility(w)
    for _ in range(iters):
        a, mse = stats(w)
        xi = 1.0 / mse
        # PF weights are refreshed every step; held fixed for longer, a weighted sum-rate
        # update switches weak users off and the log utility collapses.
        pi = weights * phase_scale / (phase_scale * np.log2(1.0 / mse) + epsilon_0)
        A = np.einsum("i,ia,ib->ab", pi * xi * np.abs(a) ** 2, b, b.conj())
        rhs = (pi * xi * a)[:, None] * b
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-12 * max(eigvals.max(), 1e-300))
        projected = rhs @ eigvecs.conj()

        def beam_power(price):
            return float(np.sum(np.abs(projected) ** 2 / (eigvals + price) ** 2))

        low = power_price
        if beam_power(low) > p_max:
            high = max(low, 1e-12) + np.sqrt(np.sum(np.abs(rhs) ** 2) / p_max)
            for _ in range(80):
                middle = 0.5 * (low + high)
                if beam_power(middle) > p_max:
                    low = middle
                else:
                    high = middle
            low = high
        w = (projected / (eigvals + low)) @ eigvecs.T
        value = utility(w)
        if value > best_value:
            best_w, best_value = w, value
    return best_w


def local_wmmse(instance, use_relay=True):
    """Local-WMMSE / No-FL (working definition, see docs/BASELINES.md).

    AP m knows h_mk and G_mr, relay r knows G_mr and g_rk, and none of them exchange anything.
    An AP treats the stream of a relayed UE as a link towards that UE's relay, using its dominant
    receive direction since the true combiner is unknown to it. Each relay's combiner is the
    MMSE filter from the post-beamforming channels it measures on pilots.
    """
    scenario, channels, config = instance.scenario, instance.channels, instance.config
    network, power = config["network"], config["power"]
    settings = config.get("wmmse", {})
    iters = settings.get("local_iters", 100)
    assoc = np.asarray(scenario.relay_assoc) if use_relay else np.full(network["num_ues"], -1)
    tau = instance.tau
    lambda_e = float(config.get("objective", {}).get("lambda_e", 0.0))
    W = np.zeros_like(channels.h_hat)
    V = np.zeros_like(channels.g_hat)
    targets = ap_target_channels(channels.h_hat, channels.G_hat, assoc)

    for m in range(network["num_aps"]):
        served = np.flatnonzero(scenario.cluster_mask[m])
        if not len(served):
            continue
        W[m, served] = wmmse_node(targets[m, served], scenario.noise_var_ue, power["p_max_ap"], np.full(len(served), tau),
                                  instance.ue_weights[served], instance.epsilon_0, iters, lambda_e / power["pa_efficiency_ap"])

    for r in range(network["num_relays"]):
        served = np.flatnonzero(assoc == r)
        if len(served):
            V[r, served] = wmmse_node(channels.g_hat[r, served], scenario.noise_var_ue, power["p_max_relay"], np.full(len(served), 1.0 - tau),
                                      instance.ue_weights[served], instance.epsilon_0, iters, lambda_e / power["pa_efficiency_relay"])

    U = mmse_combiners(channels.G_hat, W, scenario.noise_var_relay)
    return BeamSolution(W, V, U, tau, assoc)
