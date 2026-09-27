"""
sim/channels.py
Responsibility: sample h_mk, G_mr, g_rk with large-scale fading + small-scale fading,
apply blockage extra loss, optionally apply channel aging (rho model).
Paper mapping: Sec. VI (channel model), Sec. VII (channel aging / CSI error)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ChannelSet:
    h_mk: np.ndarray
    G_mr: np.ndarray
    g_rk: np.ndarray
    h_hat: np.ndarray
    G_hat: np.ndarray
    g_hat: np.ndarray


def large_scale_fading_db(distance, path_loss_exponent, shadow_std_db, rng):
    # PSEUDOCODE:
    # path_loss_db = path_loss_const + 10 * path_loss_exponent * log10(distance)
    # shadow_db = rng.normal(0, shadow_std_db)
    # return -(path_loss_db + shadow_db)   # beta in dB (negative = attenuation)
    distance = max(float(distance), 1.0)
    path_loss_db = -30.0 + 10.0 * path_loss_exponent * np.log10(distance)
    return -(path_loss_db + rng.normal(0.0, shadow_std_db))


def apply_blockage(beta_db, blocked_mask, extra_loss_db):
    # PSEUDOCODE:
    # beta_db[blocked_ue_indices] -= extra_loss_db
    # return beta_db
    result = np.array(beta_db, copy=True)
    result[..., np.asarray(blocked_mask, dtype=bool)] -= extra_loss_db
    return result


def small_scale_fading(shape, rng, correlated=True, corr_coeff=0.0):
    # PSEUDOCODE:
    # if not correlated: sample iid complex gaussian CN(0,1) of given shape
    # else: sample first tap iid, then blend with a correlation matrix / AR(1) process
    #       to get spatially/temporally correlated Rayleigh fading
    # return complex array of `shape`
    samples = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)
    if not correlated or corr_coeff == 0.0 or shape[-1] < 2:
        return samples
    result = samples.copy()
    for index in range(1, shape[-1]):
        result[..., index] = corr_coeff * result[..., index - 1] + np.sqrt(1 - corr_coeff**2) * samples[..., index]
    return result


def sample_channel_hmk(ap_positions, ue_positions, blocked_mask, n_antennas_ap, rng, config):
    # PSEUDOCODE:
    # beta_db = large_scale_fading_db(dist(ap, ue), ...)
    # beta_db = apply_blockage(beta_db, blocked_mask, config.blockage.extra_loss_db)
    # beta_lin = 10 ** (beta_db / 10)
    # small_scale = small_scale_fading((M, K, n_antennas_ap), rng)
    # h_mk = sqrt(beta_lin) * small_scale
    # return h_mk   # shape [M, K, N_A]
    distances = np.maximum(np.linalg.norm(ap_positions[:, None, :] - ue_positions[None, :, :], axis=-1), 1.0)
    beta_db = np.empty_like(distances)
    for m in range(len(ap_positions)):
        for k in range(len(ue_positions)):
            beta_db[m, k] = large_scale_fading_db(distances[m, k], config["simulation"]["path_loss_exponent"], config["simulation"]["shadow_std_db"], rng)
    beta_db = apply_blockage(beta_db, blocked_mask, config["blockage"]["extra_loss_db"])
    return np.sqrt(10.0 ** (beta_db / 10.0))[..., None] * small_scale_fading((len(ap_positions), len(ue_positions), n_antennas_ap), rng, corr_coeff=0.35)


def sample_channel_Gmr(ap_positions, relay_positions, n_antennas_ap, n_antennas_relay, rng, config):
    # PSEUDOCODE: same recipe as sample_channel_hmk but AP-to-relay geometry
    # return G_mr  # shape [M, R, N_R, N_A]
    distances = np.maximum(np.linalg.norm(ap_positions[:, None, :] - relay_positions[None, :, :], axis=-1), 1.0)
    beta_db = np.empty_like(distances)
    for m in range(len(ap_positions)):
        for r in range(len(relay_positions)):
            beta_db[m, r] = large_scale_fading_db(distances[m, r], config["simulation"]["path_loss_exponent"], config["simulation"]["shadow_std_db"], rng)
    return np.sqrt(10.0 ** (beta_db / 10.0))[:, :, None, None] * small_scale_fading((len(ap_positions), len(relay_positions), n_antennas_relay, n_antennas_ap), rng, corr_coeff=0.35)


def sample_channel_grk(relay_positions, ue_positions, n_antennas_relay, rng, config):
    # PSEUDOCODE: same recipe, relay-to-UE geometry
    # return g_rk  # shape [R, K, N_R]
    distances = np.maximum(np.linalg.norm(relay_positions[:, None, :] - ue_positions[None, :, :], axis=-1), 1.0)
    beta_db = np.empty_like(distances)
    for r in range(len(relay_positions)):
        for k in range(len(ue_positions)):
            beta_db[r, k] = large_scale_fading_db(distances[r, k], config["simulation"]["path_loss_exponent"], config["simulation"]["shadow_std_db"], rng)
    return np.sqrt(10.0 ** (beta_db / 10.0))[..., None] * small_scale_fading((len(relay_positions), len(ue_positions), n_antennas_relay), rng, corr_coeff=0.35)


def apply_aging(true_channel, rho, rng):
    # PSEUDOCODE (paper Sec. VII):
    # e = sample complex gaussian noise, same shape as true_channel, unit variance
    # h_hat = rho * true_channel + sqrt(1 - rho**2) * e
    # return h_hat   # the "estimate" fed to nodes; true_channel used only by sim for rates
    if rho >= 1.0:
        return np.array(true_channel, copy=True)
    noise = (rng.normal(size=true_channel.shape) + 1j * rng.normal(size=true_channel.shape)) / np.sqrt(2.0)
    return rho * true_channel + np.sqrt(max(1.0 - rho**2, 0.0)) * noise


def sample_full_channel_set(scenario, config, rng):
    # PSEUDOCODE:
    # h_mk  = sample_channel_hmk(...)
    # G_mr  = sample_channel_Gmr(...)
    # g_rk  = sample_channel_grk(...)
    # if config.aging.enabled:
    #     h_hat = apply_aging(h_mk, rho, rng); etc for G, g
    # else:
    #     h_hat = h_mk (treated as perfect CSI, stretch goal to add error)
    # return ChannelSet(h_mk, G_mr, g_rk, h_hat, G_hat, g_hat)
    network = config["network"]
    h_mk = sample_channel_hmk(scenario.ap_positions, scenario.ue_positions, scenario.blocked_mask, network["n_antennas_ap"], rng, config)
    G_mr = sample_channel_Gmr(scenario.ap_positions, scenario.relay_positions, network["n_antennas_ap"], network["n_antennas_relay"], rng, config)
    g_rk = sample_channel_grk(scenario.relay_positions, scenario.ue_positions, network["n_antennas_relay"], rng, config)
    aging = config.get("aging", {})
    if aging.get("enabled", False):
        h_hat = apply_aging(h_mk, aging.get("rho_h", 0.95), rng)
        G_hat = apply_aging(G_mr, aging.get("rho_G", 0.95), rng)
        g_hat = apply_aging(g_rk, aging.get("rho_g", 0.95), rng)
    else:
        h_hat, G_hat, g_hat = h_mk.copy(), G_mr.copy(), g_rk.copy()
    return ChannelSet(h_mk, G_mr, g_rk, h_hat, G_hat, g_hat)
