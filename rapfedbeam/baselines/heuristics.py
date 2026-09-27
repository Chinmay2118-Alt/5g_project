"""
baselines/heuristics.py
Responsibility: non-learned power allocation policies used as comparison points for Claim 2,
plus a no-relay baseline used for Claim 1.
"""

import copy
import numpy as np


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
