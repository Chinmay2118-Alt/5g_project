"""
sim/association.py
Responsibility: slow-timescale relay-user association (RAP-FedWMMSE Step 1). Each pair (r, k) gets
the score Gamma_rk, and the capacitated weighted matching
    max sum a_rk Gamma_rk  s.t.  sum_r a_rk <= L_max,  sum_k a_rk <= C_r^max,  a_rk in {0, 1}
is solved exactly as a small binary program.
Paper mapping: Sec. XI (association), RAP-FedWMMSE Step 1
"""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

# Floor for an interference term with no interfering AP, in decades (-30 dB).
NO_INTERFERENCE_DECADES = -3.0


def _db(linear):
    return 10.0 * np.log10(np.maximum(linear, 1e-300))


def association_features(scenario):
    """Large-scale features of every relay-user pair, in decades (10 dB) relative to the direct link.

    Returns (beta_ru, beta_ar, interference), each of shape [R, K]:
      beta_ru       relay-to-user gain minus the UE's direct gain from its serving APs
      beta_ar       gain from the UE's serving APs to the relay, minus the same direct gain
      interference  gain from the other APs to the relay, relative to the serving APs' gain
    A positive beta_ru or beta_ar means that hop is stronger than the direct link.
    """
    cluster = np.asarray(scenario.cluster_mask, dtype=float)               # [M, K]
    direct_db = _db(np.sum(cluster * 10.0 ** (scenario.beta_ap_ue_db / 10.0), axis=0))   # [K]
    ap_relay = 10.0 ** (np.asarray(scenario.beta_ap_relay_db) / 10.0)      # [M, R]
    serving_db = _db(ap_relay.T @ cluster)                                  # [R, K]
    other = ap_relay.T @ (1.0 - cluster)                                    # [R, K]
    beta_ru = (np.asarray(scenario.beta_relay_ue_db) - direct_db[None, :]) / 10.0
    beta_ar = (serving_db - direct_db[None, :]) / 10.0
    interference = np.where(other > 0.0, (_db(other) - serving_db) / 10.0, NO_INTERFERENCE_DECADES)
    return beta_ru, beta_ar, interference


def relay_scores(scenario, config, qos_prices=None):
    """Gamma_rk = a1 beta_ru + a2 beta_ar - a3 I + a4 lambda_Q_k, shape [R, K]."""
    settings = config["association"]
    beta_ru, beta_ar, interference = association_features(scenario)
    prices = np.zeros(beta_ru.shape[1]) if qos_prices is None else np.asarray(qos_prices, dtype=float)
    return (settings["alpha_relay_ue"] * beta_ru + settings["alpha_ap_relay"] * beta_ar
            - settings["alpha_interference"] * interference + settings["alpha_qos"] * prices[None, :])


def solve_matching(scores, l_max, capacity, min_score=0.0):
    """Capacitated b-matching: binary a [R, K] maximizing the summed score of chosen pairs.

    Only pairs scoring above min_score can be chosen, since a non-positive pair cannot raise the sum.
    """
    scores = np.asarray(scores, dtype=float)
    num_relays, num_ues = scores.shape
    allowed = np.isfinite(scores) & (scores > min_score)
    if not np.any(allowed):
        return np.zeros_like(scores, dtype=int)
    cost = -np.where(allowed, scores, 0.0).ravel()
    per_ue = np.zeros((num_ues, num_relays * num_ues))
    per_relay = np.zeros((num_relays, num_relays * num_ues))
    for r in range(num_relays):
        for k in range(num_ues):
            per_ue[k, r * num_ues + k] = 1.0
            per_relay[r, r * num_ues + k] = 1.0
    constraints = [LinearConstraint(per_ue, 0.0, float(l_max)),
                   LinearConstraint(per_relay, 0.0, float(capacity))]
    upper = allowed.ravel().astype(float)
    result = milp(cost, integrality=np.ones_like(cost), bounds=Bounds(np.zeros_like(cost), upper), constraints=constraints)
    if not result.success:
        raise RuntimeError(f"association matching failed: {result.message}")
    return np.rint(result.x).reshape(num_relays, num_ues).astype(int)


def matrix_to_assoc(assoc_matrix):
    """a [R, K] with at most one relay per UE -> relay index per UE, -1 for none."""
    assoc_matrix = np.asarray(assoc_matrix)
    if np.any(assoc_matrix.sum(axis=0) > 1):
        raise NotImplementedError("the simulator stores one relay per UE, so L_max must be 1")
    assoc = np.full(assoc_matrix.shape[1], -1, dtype=int)
    relay_ids, ue_ids = np.nonzero(assoc_matrix)
    assoc[ue_ids] = relay_ids
    return assoc


def matched_association(scenario, config, qos_prices=None):
    """Step 1. A pair is eligible only if its relay-UE hop beats the direct link, relaxed by the UE's
    QoS price: the DF rate is limited by the weaker hop, so a strong AP-relay hop must not make up
    for a weak relay-UE hop in the additive score, and a UE that keeps missing R_min gets a relay
    more easily. Among eligible pairs, the matching maximizes Sir's summed score."""
    network, settings = config["network"], config["association"]
    scores = relay_scores(scenario, config, qos_prices)
    beta_ru, beta_ar, _ = association_features(scenario)
    prices = np.zeros(beta_ru.shape[1]) if qos_prices is None else np.asarray(qos_prices, dtype=float)
    eligible = beta_ru + settings["alpha_qos"] * prices[None, :] > 0.0
    if settings.get("eligibility", "relay_ue_hop") == "both_hops":
        eligible &= beta_ar > 0.0
    scores = np.where(eligible, scores, -np.inf)
    capacity = network.get("max_streams_per_relay", scores.shape[1])
    return matrix_to_assoc(solve_matching(scores, network.get("l_max", 1), capacity, config["association"].get("min_score", 0.0)))
