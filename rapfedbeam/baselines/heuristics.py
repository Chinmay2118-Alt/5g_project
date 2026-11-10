"""
baselines/heuristics.py
Responsibility: non-learned power allocation policies used as comparison points for Claim 2,
plus a no-relay baseline used for Claim 1.
"""

import copy
import numpy as np

from sim.problem import BeamSolution
from sim.rates import ap_target_channels, build_beams_from_power_coeffs, mmse_combiners, mrt_beam_directions, rzf_beam_directions


def equal_power(cluster_mask, p_max_ap):
    # PSEUDOCODE:
    # for each AP m: split p_max_ap equally across all UEs it serves (sum_k mask[m,k])
    # power_coeffs[m, k] = sqrt(p_max_ap / num_ues_served_by_m)  if cluster_mask[m,k] else 0
    # return power_coeffs   # shape [M, K]
    mask = np.asarray(cluster_mask, dtype=float)
    counts = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    return mask * np.sqrt(float(p_max_ap) / counts)


def fractional_power(beta_mk, cluster_mask, p_max_ap, alpha=0.5):
    # PSEUDOCODE:
    # power proportional to beta_mk^(-alpha) (weaker links get relatively more power)
    # normalize per-AP so total power == p_max_ap
    # return power_coeffs   # shape [M, K]
    beta = np.maximum(np.asarray(beta_mk, dtype=float), 1e-12)
    weights = np.asarray(cluster_mask, dtype=float) * beta ** (-alpha)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.sqrt(float(p_max_ap) * weights)


def relay_power_equal_or_fractional(relay_assoc, p_max_relay, num_relays=None, beta_rk=None, fractional=False, alpha=0.5):
    assoc = np.asarray(relay_assoc)
    if num_relays is None:
        num_relays = int(assoc.max()) + 1 if np.any(assoc >= 0) else 0
    result = np.zeros((num_relays, len(assoc)), dtype=float)
    for relay_id in range(num_relays):
        users = np.flatnonzero(assoc == relay_id)
        if not len(users):
            continue
        weights = np.ones(len(users))
        if fractional and beta_rk is not None:
            weights = np.maximum(np.asarray(beta_rk)[relay_id, users], 1e-12) ** (-alpha)
        weights /= weights.sum()
        result[relay_id, users] = np.sqrt(float(p_max_relay) * weights)
    return result


def no_relay_scenario(scenario):
    # PSEUDOCODE:
    # returns a copy of `scenario` with relay_assoc all set to -1
    # (used to run compute_full_rates with only Phase-I / direct-link contribution -> Claim 1 baseline)
    result = copy.deepcopy(scenario)
    result.relay_assoc = np.full_like(np.asarray(scenario.relay_assoc), -1)
    return result


def heuristic_solution(instance, allocation="equal", use_relay=True, precoder="mrt", tau=None):
    """Fixed beam directions from the channel estimates, power from a closed-form rule."""
    scenario, channels, config = instance.scenario, instance.channels, instance.config
    network, power = config["network"], config["power"]
    assoc = np.asarray(scenario.relay_assoc) if use_relay else np.full(network["num_ues"], -1)
    targets = ap_target_channels(channels.h_hat, channels.G_hat, assoc)
    if precoder == "rzf":
        directions_ap = rzf_beam_directions(targets, reg_eps=scenario.noise_var_ue / power["p_max_ap"], cluster_mask=scenario.cluster_mask)
    else:
        directions_ap = mrt_beam_directions(targets)
    if allocation == "fractional":
        ap_coeffs = fractional_power(10.0 ** (scenario.beta_ap_ue_db / 10.0), scenario.cluster_mask, power["p_max_ap"])
    else:
        ap_coeffs = equal_power(scenario.cluster_mask, power["p_max_ap"])
    W = build_beams_from_power_coeffs(directions_ap, ap_coeffs, power["p_max_ap"])
    relay_coeffs = relay_power_equal_or_fractional(
        assoc, power["p_max_relay"], network["num_relays"],
        10.0 ** (scenario.beta_relay_ue_db / 10.0), fractional=allocation == "fractional",
    )
    V = build_beams_from_power_coeffs(mrt_beam_directions(channels.g_hat), relay_coeffs, power["p_max_relay"])
    U = mmse_combiners(channels.G_hat, W, scenario.noise_var_relay)
    return BeamSolution(W, V, U, config["phase"]["tau_fixed"] if tau is None else tau, assoc)
