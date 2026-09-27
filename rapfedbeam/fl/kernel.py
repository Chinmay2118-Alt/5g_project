"""
fl/kernel.py
Responsibility: compute pairwise topology similarity S_ij and role-compatibility
coefficient kappa, used to build personalized aggregation weights alpha_ij.
Paper mapping: Sec. XXIV
"""

import numpy as np


def role_compatibility(role_i, role_j, kappa_ap_ap, kappa_relay_relay, kappa_ap_relay):
    # PSEUDOCODE:
    # if role_i == role_j == 'AP':      return kappa_ap_ap        # typically 1.0
    # if role_i == role_j == 'relay':   return kappa_relay_relay  # typically 1.0
    # else:                              return kappa_ap_relay     # swept in E5, e.g. 0.3
    left, right = role_i.lower(), role_j.lower()
    if left == right == "ap":
        return float(kappa_ap_ap)
    if left == right and left in ("relay", "r"):
        return float(kappa_relay_relay)
    return float(kappa_ap_relay)


def topology_kernel(s_i, s_j, role_i, role_j, sigma, kappa_config):
    # PSEUDOCODE:
    # kappa = role_compatibility(role_i, role_j, **kappa_config)
    # dist_sq = ||s_i - s_j||^2
    # S_ij = kappa * exp(-dist_sq / (2 * sigma**2))
    # return S_ij
    sigma = max(float(sigma), 1e-12)
    kappa = role_compatibility(role_i, role_j, **kappa_config)
    return kappa * np.exp(-np.sum((np.asarray(s_i) - np.asarray(s_j)) ** 2) / (2.0 * sigma**2))


def compute_kernel_matrix(signatures, roles, sigma, kappa_config):
    # PSEUDOCODE:
    # for every pair (i, j) of participating nodes:
    #     S[i, j] = topology_kernel(signatures[i], signatures[j], roles[i], roles[j], sigma, kappa_config)
    # return S   # shape [N, N], N = num participating nodes this round
    ids = list(signatures)
    matrix = np.empty((len(ids), len(ids)), dtype=float)
    for i, node_i in enumerate(ids):
        for j, node_j in enumerate(ids):
            matrix[i, j] = topology_kernel(signatures[node_i], signatures[node_j], roles[node_i], roles[node_j], sigma, kappa_config)
    return matrix


def compute_aggregation_weights(S, participating_ids):
    # PSEUDOCODE (paper Sec. XXIV, alpha_ij formula):
    # for each node i:
    #     alpha_i = S[i, participating_ids] / sum(S[i, participating_ids])   # normalize row
    # return alpha   # shape [N, N], row i = aggregation weights node i uses over all participants
    S = np.asarray(S, dtype=float)
    selected = list(participating_ids)
    weights = S[:, selected]
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
