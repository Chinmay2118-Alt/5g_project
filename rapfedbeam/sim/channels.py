"""
sim/channels.py
Responsibility: sample h_mk, G_mr, g_rk from a drop's large-scale fading + small-scale fading,
optionally apply channel aging (rho model).
Paper mapping: Sec. VI (channel model), Sec. VII (channel aging / CSI error)

Units: every channel is normalized by the noise power of its receiver, so receiver noise
variance is 1 and p * |h|^2 is an SNR when p is in Watts.
"""

from dataclasses import dataclass
import numpy as np


# Receiver noise variance after channel normalization.
NORMALIZED_NOISE_VAR = 1.0


@dataclass
class ChannelSet:
    h_mk: np.ndarray
    G_mr: np.ndarray
    g_rk: np.ndarray
    h_hat: np.ndarray
    G_hat: np.ndarray
    g_hat: np.ndarray


def independent_streams(rng, count):
    # Draws a fixed amount from rng, so what each stream later consumes cannot shift the others.
    return [np.random.default_rng(seed) for seed in rng.integers(0, 2**63, size=count)]


def noise_power_w(config, noise_figure_key):
    link = config["link_budget"]
    noise_dbm = link["noise_psd_dbm_hz"] + 10.0 * np.log10(link["bandwidth_hz"]) + link[noise_figure_key]
    return 10.0 ** ((noise_dbm - 30.0) / 10.0)


def channel_amplitude(beta_db, noise_w):
    # sqrt(beta / sigma^2): large-scale amplitude in noise-normalized units
    return np.sqrt(10.0 ** (np.asarray(beta_db) / 10.0) / noise_w)


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


def sample_channel_hmk(beta_ap_ue_db, n_antennas_ap, noise_w, rng):
    # h_mk = sqrt(beta_mk / noise) * small_scale, shape [M, K, N_A]
    beta_db = np.asarray(beta_ap_ue_db)
    return channel_amplitude(beta_db, noise_w)[..., None] * small_scale_fading((*beta_db.shape, n_antennas_ap), rng, corr_coeff=0.35)


def sample_channel_Gmr(beta_ap_relay_db, n_antennas_ap, n_antennas_relay, noise_w, rng):
    # G_mr shape [M, R, N_R, N_A]
    beta_db = np.asarray(beta_ap_relay_db)
    return channel_amplitude(beta_db, noise_w)[:, :, None, None] * small_scale_fading((*beta_db.shape, n_antennas_relay, n_antennas_ap), rng, corr_coeff=0.35)


def sample_channel_grk(beta_relay_ue_db, n_antennas_relay, noise_w, rng):
    # g_rk shape [R, K, N_R]
    beta_db = np.asarray(beta_relay_ue_db)
    return channel_amplitude(beta_db, noise_w)[..., None] * small_scale_fading((*beta_db.shape, n_antennas_relay), rng, corr_coeff=0.35)


def apply_aging(channel_hat, rho, amplitude, rng):
    # Paper Sec. VII: h = rho * h_hat + sqrt(1 - rho**2) * e, with e drawn at the same
    # large-scale power as the link, so the true channel keeps the link's average power.
    if rho >= 1.0:
        return np.array(channel_hat, copy=True)
    shape = channel_hat.shape
    error = np.asarray(amplitude).reshape(shape[:2] + (1,) * (len(shape) - 2)) * small_scale_fading(shape, rng, corr_coeff=0.0)
    return rho * channel_hat + np.sqrt(max(1.0 - rho**2, 0.0)) * error


def sample_full_channel_set(scenario, config, rng):
    # The samples below are the estimates nodes hold; with aging on, the true channels
    # that rates are scored on are derived from them (paper Sec. VII).
    network = config["network"]
    noise_ue, noise_relay = noise_power_w(config, "noise_figure_ue_db"), noise_power_w(config, "noise_figure_relay_db")
    h_rng, G_rng, g_rng, h_age_rng, G_age_rng, g_age_rng = independent_streams(rng, 6)
    h_hat = sample_channel_hmk(scenario.beta_ap_ue_db, network["n_antennas_ap"], noise_ue, h_rng)
    G_hat = sample_channel_Gmr(scenario.beta_ap_relay_db, network["n_antennas_ap"], network["n_antennas_relay"], noise_relay, G_rng)
    g_hat = sample_channel_grk(scenario.beta_relay_ue_db, network["n_antennas_relay"], noise_ue, g_rng)
    aging = config.get("aging", {})
    if aging.get("enabled", False):
        h_mk = apply_aging(h_hat, aging.get("rho_h", 0.95), channel_amplitude(scenario.beta_ap_ue_db, noise_ue), h_age_rng)
        G_mr = apply_aging(G_hat, aging.get("rho_G", 0.95), channel_amplitude(scenario.beta_ap_relay_db, noise_relay), G_age_rng)
        g_rk = apply_aging(g_hat, aging.get("rho_g", 0.95), channel_amplitude(scenario.beta_relay_ue_db, noise_ue), g_age_rng)
    else:
        h_mk, G_mr, g_rk = h_hat.copy(), G_hat.copy(), g_hat.copy()
    return ChannelSet(h_mk, G_mr, g_rk, h_hat, G_hat, g_hat)
