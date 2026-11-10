"""
fl/communication.py
Responsibility: RAP-FedWMMSE's federated communication (Steps 12-15): event-triggered eligibility,
communication-budgeted client selection, Top-k compression with error feedback, and topology-
and role-aware personalized aggregation.
"""

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from fl.kernel import role_compatibility

BYTES_PER_FLOAT = 4
BYTES_PER_INDEX = 4


def urgency(model_drift, topology_drift, age, grad_norm, weights, age_max, grad_max, eps=1e-12):
    """Step 12: Gamma_i^FL = b1 d_theta + b2 d_H + b3 A / A_max + b4 ||grad|| / (G_max + eps)."""
    b1, b2, b3, b4 = weights
    return b1 * model_drift + b2 * topology_drift + b3 * age / age_max + b4 * grad_norm / (grad_max + eps)


def model_drift(theta, theta_last, eps=1e-12):
    return float(np.linalg.norm(theta - theta_last) / (np.linalg.norm(theta_last) + eps))


def select_clients(scores, eligible, sizes, budget):
    """Step 13: binary knapsack, max sum q_i Gamma_i s.t. q_i <= chi_i, sum q_i S_i <= B."""
    scores, sizes = np.asarray(scores, dtype=float), np.asarray(sizes, dtype=float)
    allowed = np.asarray(eligible, dtype=bool) & (sizes <= budget)
    if not np.any(allowed):
        return np.zeros(len(scores), dtype=bool)
    result = milp(-scores, integrality=np.ones_like(scores), bounds=Bounds(np.zeros_like(scores), allowed.astype(float)),
                  constraints=[LinearConstraint(sizes[None, :], 0.0, float(budget))])
    if not result.success:
        raise RuntimeError(f"client selection failed: {result.message}")
    return np.rint(result.x).astype(bool)


def top_k(update, residual, fraction):
    """Step 14: keep the largest fraction of (update + residual); return it and the new residual."""
    corrected = update + residual
    keep = max(1, int(math.ceil(fraction * corrected.size)))
    sent = np.zeros_like(corrected)
    index = np.argpartition(np.abs(corrected), -keep)[-keep:]
    sent[index] = corrected[index]
    return sent, corrected - sent


def compressed_bytes(num_params, fraction):
    return int(math.ceil(fraction * num_params)) * (BYTES_PER_FLOAT + BYTES_PER_INDEX)


def similarity(signature_i, signature_j, role_i, role_j, sigma, kappa_config):
    """Step 15: S_ij = kappa_rho(i)rho(j) exp(-||s_i - s_j||^2 / (2 sigma^2))."""
    kappa = role_compatibility(role_i, role_j, **kappa_config)
    return kappa * math.exp(-float(np.sum((signature_i - signature_j) ** 2)) / (2.0 * sigma ** 2))


def personalized_updates(signatures, roles, updates, selected, sigma, kappa_config, client_weights, eps=1e-12):
    """Step 15: Delta_bar_i = sum_{j in S_e} alpha_ij Delta_j, with
    alpha_ij = S_ij p_j / (sum_{l in S_e} S_il p_l + eps). Returns one update per target node."""
    nodes = list(signatures)
    chosen = [j for j in nodes if selected[j]]
    result = {}
    for i in nodes:
        weights = np.array([similarity(signatures[i], signatures[j], roles[i], roles[j], sigma, kappa_config) * client_weights[j] for j in chosen])
        alpha = weights / (weights.sum() + eps)
        result[i] = sum((a * updates[j] for a, j in zip(alpha, chosen)), np.zeros_like(next(iter(updates.values()))))
    return result
