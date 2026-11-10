"""
sim/rates.py
Responsibility: turn (fixed beam directions x learned power coeffs) into complex beams,
then compute Phase-I SINR, Phase-I relay-decode SINR, Phase-II SINR, and end-to-end DF rate.
Paper mapping: Sec. VIII-XIV
"""

import numpy as np


def as_assoc_matrix(relay_assoc, num_relays):
    """Binary a_rk [R, K] from either form an association is stored in: a relay index per UE (-1 for
    none, enough when L_max = 1) or the matrix itself (needed when L_max > 1)."""
    relay_assoc = np.asarray(relay_assoc)
    if relay_assoc.ndim == 2:
        if relay_assoc.shape[0] != num_relays:
            raise ValueError(f"association matrix has {relay_assoc.shape[0]} rows, expected {num_relays}")
        return (relay_assoc > 0).astype(int)
    matrix = np.zeros((num_relays, len(relay_assoc)), dtype=int)
    for k, relay_id in enumerate(relay_assoc):
        if relay_id >= 0:
            matrix[int(relay_id), k] = 1
    return matrix


def relayed_ues(relay_assoc, num_relays):
    """Boolean [K]: UE k has at least one assisting relay."""
    return as_assoc_matrix(relay_assoc, num_relays).any(axis=0)


def build_beams_from_power_coeffs(beam_directions, power_coeffs, p_max):
    # PSEUDOCODE:
    # v1 simplification: beam DIRECTION is fixed (MRT or RZF), only power coeff is learned.
    # w_mk = power_coeffs[m, k] * beam_directions[m, k]   (already unit-norm directions)
    # (feasibility projection module enforces sum_k ||w_mk||^2 <= p_max_m; see models/projection.py)
    # return w   # shape [M, K, N_A] complex
    directions = np.asarray(beam_directions)
    coeffs = np.asarray(power_coeffs)
    if directions.shape[:2] != coeffs.shape:
        raise ValueError("beam directions must have shape [nodes, users, antennas]")
    return directions * coeffs[..., None]


def mrt_beam_directions(h_mk):
    # Received gain is h^H w, so the matched beam is w = h / ||h|| (giving h^H w = ||h||).
    # Works for any [nodes, users, antennas] channel, including relay-to-UE g_rk.
    h_mk = np.asarray(h_mk)
    norms = np.linalg.norm(h_mk, axis=-1, keepdims=True)
    return h_mk / np.maximum(norms, 1e-12)


def ap_target_channels(h_hat, G_hat, relay_assoc):
    # The channel each AP should steer stream k along, shape [M, K, N_A]: towards the UE, or for a
    # relayed UE towards its relay (dominant receive direction of G_mr), because r_k <= C_AR
    # makes the relay the receiver that must decode Phase I. With several relays (L_max > 1) every
    # one of them must decode, so the target is the sum of their unit-norm directions. Uses only
    # that AP's own CSI.
    targets = np.array(h_hat, dtype=np.complex128, copy=True)
    assoc = as_assoc_matrix(relay_assoc, G_hat.shape[1])
    for m in range(targets.shape[0]):
        for k in range(targets.shape[1]):
            relays = np.flatnonzero(assoc[:, k])
            if not len(relays):
                continue
            target = np.zeros(targets.shape[2], dtype=np.complex128)
            for relay_id in relays:
                left, _, _ = np.linalg.svd(G_hat[m, relay_id], full_matrices=False)
                direction = G_hat[m, relay_id].conj().T @ left[:, 0]
                target += direction / max(np.linalg.norm(direction), 1e-300) if len(relays) > 1 else direction
            targets[m, k] = target
    return targets


def rzf_beam_directions(h_mk, reg_eps, cluster_mask=None):
    # Regularized zero-forcing per AP, across the UEs that AP serves.
    # With H = h^H stacked in rows: w_dir = H^H (H H^H + reg_eps * I)^-1, columns unit norm.
    h_mk = np.asarray(h_mk)
    directions = np.zeros_like(h_mk, dtype=np.complex128)
    for m in range(h_mk.shape[0]):
        served = np.arange(h_mk.shape[1]) if cluster_mask is None else np.flatnonzero(cluster_mask[m])
        if not len(served):
            continue
        H = np.conj(h_mk[m, served])
        precoder = H.conj().T @ np.linalg.inv(H @ H.conj().T + reg_eps * np.eye(len(served)))
        norms = np.linalg.norm(precoder, axis=0, keepdims=True)
        directions[m, served] = (precoder / np.maximum(norms, 1e-12)).T
    return directions


def phase1_sinr_ue(h_mk, w, cluster_mask, noise_var_k):
    # PSEUDOCODE (paper Sec. IX):
    # for each UE k:
    #   desired   = | sum_{m in cluster(k)} h_mk^H w_mk |^2
    #   interf    = sum_{j != k} | sum_m h_mk^H w_mj |^2
    #   gamma1_k  = desired / (interf + noise_var_k)
    # return gamma1   # shape [K]
    h_mk, w = np.asarray(h_mk), np.asarray(w)
    cluster_mask = np.asarray(cluster_mask, dtype=bool)
    num_ues = h_mk.shape[1]
    gamma = np.zeros(num_ues, dtype=float)
    for k in range(num_ues):
        effective = np.einsum("ma,mja->mj", np.conj(h_mk[:, k, :]), w)
        desired = abs(np.sum(effective[cluster_mask[:, k], k])) ** 2
        interference = sum(abs(np.sum(effective[cluster_mask[:, j], j])) ** 2 for j in range(num_ues) if j != k)
        gamma[k] = desired / (interference + noise_var_k)
    return gamma


def phase1_relay_decode_sinr(G_mr, w, relay_combiner_u, assoc_relay_of_k, noise_var_r):
    # PSEUDOCODE (paper Sec. X):
    # for the relay r assigned to UE k:
    #   effective_channel = sum_m G_mr[:, m, :] @ w[m, k]      (project AP beams through G)
    #   z = u_rk^H @ effective_channel
    #   desired = |z|^2 ; interference from other streams j computed similarly
    #   gamma_AR_rk = desired / (interference + noise_var_r)
    # return gamma_AR   # shape [K] (only defined for blocked/relay-assisted UEs)
    G_mr, w, u = np.asarray(G_mr), np.asarray(w), np.asarray(relay_combiner_u)
    num_relays, num_ues = G_mr.shape[1], w.shape[1]
    assoc = as_assoc_matrix(assoc_relay_of_k, num_relays)
    gamma = np.zeros((num_relays, num_ues), dtype=float)
    for r in range(num_relays):
        for k in range(num_ues):
            if not assoc[r, k]:
                continue
            effective = np.einsum("n,mna->ma", np.conj(u[r, k]), G_mr[:, r, :, :])
            gains = np.einsum("ma,mja->mj", effective, w)
            desired = abs(np.sum(gains[:, k])) ** 2
            interference = sum(abs(np.sum(gains[:, j])) ** 2 for j in range(num_ues) if j != k)
            gamma[r, k] = desired / (interference + noise_var_r)
    return gamma


def relay_decode_capacity(gamma_AR, tau):
    # C_AR_rk = tau * log2(1 + gamma_AR_rk)
    return np.asarray(tau) * np.log2(1.0 + np.maximum(np.asarray(gamma_AR), 0.0))


def phase2_sinr_ue(g_rk, v, relay_assoc, noise_var_k):
    # PSEUDOCODE (paper Sec. XIII):
    # for each UE k assigned to relay r:
    #   desired = | g_rk^H v_rk |^2
    #   interf  = sum_{j != k} | g_rk^H v_rj |^2   (interference from relay serving other UEs)
    #   gamma2_k = desired / (interf + noise_var_k)
    # UEs with no relay assigned: gamma2_k = 0 (no Phase II contribution)
    # return gamma2   # shape [K]
    g_rk, v = np.asarray(g_rk), np.asarray(v)
    num_ues = g_rk.shape[1]
    relayed = relayed_ues(relay_assoc, g_rk.shape[0])
    gamma = np.zeros(num_ues, dtype=float)
    for k in range(num_ues):
        if not relayed[k]:
            continue
        # Per-stream gains add coherently across relays, as in the paper's gamma^(2) (L_max > 1 ready).
        gains = np.einsum("rn,rjn->j", np.conj(g_rk[:, k, :]), v)
        desired = abs(gains[k]) ** 2
        interference = sum(abs(gains[j]) ** 2 for j in range(num_ues) if j != k)
        gamma[k] = desired / (interference + noise_var_k)
    return gamma


def destination_capacity(gamma1, gamma2, tau):
    # C_D_k = tau * log2(1 + gamma1_k) + (1 - tau) * log2(1 + gamma2_k)
    return tau * np.log2(1.0 + np.maximum(gamma1, 0.0)) + (1.0 - tau) * np.log2(1.0 + np.maximum(gamma2, 0.0))


def end_to_end_rate(C_D, C_AR, relay_assoc, hard_min=True, soft_temp=None):
    # PSEUDOCODE (paper Sec. XIV):
    # for UE k with no relay (relay_assoc[k] == -1):
    #     r_k = C_D_k   (direct-link-only rate; Phase II term is already 0)
    # for UE k with a relay:
    #     if hard_min:
    #         r_k = min(C_D_k, C_AR[relay_assoc[k], k])
    #     else:  # soft-min for training gradients
    #         r_k = -soft_temp * logsumexp([-C_D_k / soft_temp, -C_AR_rk / soft_temp])
    # return r   # shape [K]
    C_D, C_AR = np.asarray(C_D), np.asarray(C_AR)
    assoc = as_assoc_matrix(relay_assoc, C_AR.shape[0])
    rates = C_D.copy()
    for k in range(len(rates)):
        relays = np.flatnonzero(assoc[:, k])
        if not len(relays):
            continue
        # Every assisting relay must decode the stream: r_k = min(C_D, min over r with a_rk = 1 of C_AR).
        relay_rates = C_AR[relays, k]
        if hard_min:
            rates[k] = min(C_D[k], float(relay_rates.min()))
        else:
            if soft_temp is None or soft_temp <= 0:
                raise ValueError("soft_temp must be positive for soft-min rates")
            values = np.concatenate([[-C_D[k] / soft_temp], -relay_rates / soft_temp])
            rates[k] = -soft_temp * (np.max(values) + np.log(np.exp(values - np.max(values)).sum()))
    return rates


def compute_full_rates(scenario, channels, w, v, u, tau, hard_min=True, soft_temp=None):
    # PSEUDOCODE: orchestrates the above into one call used by both training and eval
    # gamma1   = phase1_sinr_ue(...)
    # gamma_AR = phase1_relay_decode_sinr(...)
    # C_AR     = relay_decode_capacity(gamma_AR, tau)
    # gamma2   = phase2_sinr_ue(...)
    # C_D      = destination_capacity(gamma1, gamma2, tau)
    # r        = end_to_end_rate(C_D, C_AR, scenario.relay_assoc, hard_min, soft_temp)
    # return r, {debug: gamma1, gamma2, gamma_AR, C_D, C_AR}
    noise_k = scenario.noise_var_ue
    noise_r = scenario.noise_var_relay
    gamma1 = phase1_sinr_ue(channels.h_mk, w, scenario.cluster_mask, noise_k)
    gamma_ar = phase1_relay_decode_sinr(channels.G_mr, w, u, scenario.relay_assoc, noise_r)
    C_AR = relay_decode_capacity(gamma_ar, tau)
    gamma2 = phase2_sinr_ue(channels.g_rk, v, scenario.relay_assoc, noise_k)
    C_D = destination_capacity(gamma1, gamma2, tau)
    rates = end_to_end_rate(C_D, C_AR, scenario.relay_assoc, hard_min, soft_temp)
    return rates, {"gamma1": gamma1, "gamma_AR": gamma_ar, "gamma2": gamma2, "C_D": C_D, "C_AR": C_AR}


def mmse_combiners(G_hat, w, noise_var=1.0):
    """Return normalized per-relay/user MMSE combiners from effective AP-relay channel estimates.

    Args:
        G_hat:     AP-to-relay channel estimates, shape [M, R, N_R, N_A]
        w:         AP beamforming weights, shape [M, K, N_A]
        noise_var: noise variance at relay receive antennas
    """
    G_mr, w = np.asarray(G_hat), np.asarray(w)
    num_relays, num_ues, n_relay = G_mr.shape[1], w.shape[1], G_mr.shape[2]
    combiners = np.zeros((num_relays, num_ues, n_relay), dtype=np.complex128)
    # Effective channel: shape [R, N_R, K]; effective[r] = [N_R, K]
    effective = np.einsum("mrna,mka->rnk", G_mr, w)
    for r in range(num_relays):
        eff_r = effective[r]  # [N_R, K]
        # Covariance = Σ_k e_k e_k^H + noise * I, positive definite since noise_var > 0.
        covariance = eff_r @ eff_r.conj().T + noise_var * np.eye(n_relay)
        for k in range(num_ues):
            rhs = eff_r[:, k]
            try:
                vector = np.linalg.solve(covariance, rhs)
            except np.linalg.LinAlgError:
                vector = rhs.copy()          # fallback: matched filter
            if not np.all(np.isfinite(vector)):
                vector = rhs.copy()          # fallback on NaN/Inf from solver
            norm = float(np.linalg.norm(vector))
            if norm > 1e-12 and np.isfinite(norm):
                combiners[r, k] = vector / norm
            # else: leave combiner as zero (relay produces no useful signal for this UE)
    return combiners
