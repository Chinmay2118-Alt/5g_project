"""
eval/metrics.py
Responsibility: compute the metrics used across all experiments E1-E6.
"""

import numpy as np

from sim.rates import as_assoc_matrix, compute_full_rates


def pf_utility(rates, weights=None, epsilon_0=1e-3):
    # sum_k omega_k log(r_k + epsilon_0), the first term of P0
    rates = np.asarray(rates, dtype=float)
    weights = np.ones(rates.shape[-1]) if weights is None else np.asarray(weights, dtype=float)
    return np.sum(weights * np.log(np.maximum(rates, 0.0) + epsilon_0), axis=-1)


def check_feasibility(instance, solution, tol=1e-4):
    """Raise ValueError unless the solution satisfies the constraints of P0 that are hard in v1."""
    scenario, power = instance.scenario, instance.config["power"]
    W, V = np.asarray(solution.W), np.asarray(solution.V)
    network = instance.config["network"]
    expected = (network["num_aps"], network["num_ues"], network["n_antennas_ap"])
    if W.shape != expected:
        raise ValueError(f"W has shape {W.shape}, expected {expected}")
    expected = (network["num_relays"], network["num_ues"], network["n_antennas_relay"])
    if V.shape != expected or np.asarray(solution.U).shape != expected:
        raise ValueError(f"V/U have shapes {V.shape}/{np.asarray(solution.U).shape}, expected {expected}")
    # NaN fails every comparison below, so it has to be caught explicitly.
    if not (np.all(np.isfinite(W)) and np.all(np.isfinite(V)) and np.all(np.isfinite(solution.U)) and np.isfinite(solution.tau)):
        raise ValueError("solution contains NaN or infinite values")
    if np.any(np.sum(np.abs(W) ** 2, axis=(1, 2)) > power["p_max_ap"] * (1.0 + tol)):
        raise ValueError("AP power constraint violated")
    if np.any(np.sum(np.abs(V) ** 2, axis=(1, 2)) > power["p_max_relay"] * (1.0 + tol)):
        raise ValueError("relay power constraint violated")
    if np.any(np.sum(np.abs(W), axis=-1)[~np.asarray(scenario.cluster_mask, dtype=bool)] > tol):
        raise ValueError("AP transmits to a UE outside its cluster (c_mk = 0 requires w_mk = 0)")
    active = as_assoc_matrix(solution.relay_assoc, V.shape[0]).astype(bool)
    if np.any(np.sum(np.abs(V), axis=-1)[~active] > tol):
        raise ValueError("relay transmits to a UE it is not associated with (a_rk = 0 requires v_rk = 0)")
    if np.any(active.sum(axis=1) > network.get("max_streams_per_relay", V.shape[1])):
        raise ValueError("relay capacity C_r^max violated")
    if np.any(active.sum(axis=0) > network.get("l_max", 1)):
        raise ValueError("cooperation order L_max violated")
    phase = instance.config["phase"]
    if not active.any():
        # No relayed UE means no Phase II: a single-phase network may use the whole interval (tau = 1).
        if not phase.get("tau_min", 0.0) - tol <= solution.tau <= 1.0 + tol:
            raise ValueError("tau outside [tau_min, 1]")
    elif not phase.get("tau_min", 0.0) - tol <= solution.tau <= phase.get("tau_max", 1.0) + tol:
        raise ValueError("tau outside [tau_min, tau_max]")
    norms = np.linalg.norm(np.asarray(solution.U), axis=-1)[active]
    if np.any((norms > tol) & (np.abs(norms - 1.0) > 1e-3)):
        raise ValueError("relay combiner does not have unit norm")


def transmit_power(instance, solution):
    # P_tx of paper Sec. XVI
    power = instance.config["power"]
    return float(np.sum(np.abs(solution.W) ** 2) / power["pa_efficiency_ap"] + np.sum(np.abs(solution.V) ** 2) / power["pa_efficiency_relay"])


def total_power(instance, solution):
    # P_tot = P_tx + P_circ, a relay is active (b_r = 1) when it serves at least one UE
    power = instance.config["power"]
    num_relays = instance.config["network"]["num_relays"]
    active_relays = int(as_assoc_matrix(solution.relay_assoc, num_relays).any(axis=1).sum())
    return transmit_power(instance, solution) + instance.config["network"]["num_aps"] * power["circuit_power_ap"] + active_relays * power["circuit_power_relay"]


def evaluate_solution(instance, solution):
    """The single scoring path: feasibility, then rates on the TRUE channels, whatever the algorithm saw."""
    check_feasibility(instance, solution)
    scenario = instance.scenario
    if np.shape(solution.relay_assoc) != np.shape(scenario.relay_assoc) or not np.array_equal(solution.relay_assoc, scenario.relay_assoc):
        scenario = scenario.copy()
        scenario.relay_assoc = np.asarray(solution.relay_assoc)
    rates, debug = compute_full_rates(scenario, instance.channels, solution.W, solution.V, solution.U, solution.tau)
    return {
        "rates": rates,
        "pf_utility": float(pf_utility(rates, instance.ue_weights, instance.epsilon_0)),
        "sum_rate": float(rates.sum()),
        "transmit_power": transmit_power(instance, solution),
        "total_power": total_power(instance, solution),
        "debug": debug,
    }


def evaluate_pf_utility_on_holdout(node_models, config, num_holdout_drops=100, scenario=None):
    from sim.channels import sample_full_channel_set
    from sim.scenario import generate_drop
    rng = np.random.default_rng(config.get("seed", 42) + 12345)
    all_rates = []

    # Lock the geography for the holdout set so personalized models are evaluated on their home topology
    eval_scenario = scenario if scenario is not None else generate_drop(config, rng)

    for _ in range(num_holdout_drops):
        # ONLY sample fast-fading channels on the fixed geography
        channels = sample_full_channel_set(eval_scenario, config, rng)
        if callable(node_models):
            rates = np.asarray(node_models(eval_scenario, channels))
        else:
            raise ValueError("node_models must be a callable evaluator in the reduced implementation")
        all_rates.append(rates)

    all_rates = np.asarray(all_rates)
    return float(np.mean(np.log(np.maximum(all_rates, 1e-6)).sum(axis=-1))), all_rates


def per_user_rate_cdf(all_rates):
    # PSEUDOCODE: flatten all_rates across drops and UEs, sort, return (sorted_rates, cdf_values)
    values = np.sort(np.asarray(all_rates).ravel())
    return values, np.arange(1, len(values) + 1) / max(len(values), 1)


def percentile_rate(all_rates, percentile=5):
    # PSEUDOCODE: return np.percentile(all_rates.flatten(), percentile)
    return float(np.percentile(np.asarray(all_rates), percentile))


def outage_probability(all_rates, r_min):
    # PSEUDOCODE: fraction of (drop, ue) pairs where rate < r_min
    # (reported as a DIAGNOSTIC only in v1, not enforced as a constraint -- see Sec. 8 limitations)
    return float(np.mean(np.asarray(all_rates) < r_min))


def jain_fairness(all_rates):
    # (sum r)^2 / (K sum r^2) per drop, averaged over drops
    rates = np.atleast_2d(np.asarray(all_rates, dtype=float))
    return float(np.mean(rates.sum(axis=1) ** 2 / np.maximum(rates.shape[1] * np.sum(rates ** 2, axis=1), 1e-12)))
