"""
sim/scenario.py
Responsibility: place APs, relays, UEs; assign blockage; draw the drop's large-scale fading;
compute AP clustering.
Paper mapping: Sec. III (clustering), Sec. II (network sets M, R, K)
"""

from dataclasses import dataclass
import copy
import numpy as np

from sim.channels import NORMALIZED_NOISE_VAR, independent_streams


@dataclass
class Scenario:
    ap_positions: np.ndarray
    relay_positions: np.ndarray
    ue_positions: np.ndarray
    blocked_mask: np.ndarray
    cluster_mask: np.ndarray
    relay_assoc: np.ndarray
    beta_ap_ue_db: np.ndarray
    beta_relay_ue_db: np.ndarray
    beta_ap_relay_db: np.ndarray = None
    noise_var_ue: float = NORMALIZED_NOISE_VAR
    noise_var_relay: float = NORMALIZED_NOISE_VAR

    def copy(self):
        return copy.deepcopy(self)

# --- data structures ---
# Scenario:
#   ap_positions      : array [M, 2]
#   relay_positions    : array [R, 2]
#   ue_positions       : array [K, 2]
#   blocked_mask       : bool array [K]         -> True if UE behind blockage
#   cluster_mask       : bool array [M, K]      -> c_mk, 1 if AP m serves UE k
#   relay_assoc        : int array [K]          -> which relay serves UE k (-1 if none)
#   beta_*_db          : large-scale gains (path loss + shadowing + blockage), fixed for the drop


def sample_positions(num_nodes, area_size, rng):
    # PSEUDOCODE:
    # uniformly (or grid-jittered) sample num_nodes (x, y) points inside area_size
    # return array [num_nodes, 2]
    return rng.uniform(0.0, float(area_size), size=(num_nodes, 2))


def assign_blockage(ue_positions, frac_blocked, rng):
    num_blocked = int(round(len(ue_positions) * frac_blocked))
    mask = np.zeros(len(ue_positions), dtype=bool)
    if num_blocked:
        # Sort by X coordinate to force geographic clustering (non-IID environment)
        sorted_indices = np.argsort(ue_positions[:, 0])
        mask[sorted_indices[-num_blocked:]] = True
    return mask


def compute_ap_clustering(ap_positions, ue_positions, beta_db, top_l):
    # PSEUDOCODE:
    # for each UE k:
    #     rank APs by large-scale fading beta_mk (descending)
    #     select top_l APs -> set cluster_mask[m, k] = 1 for those
    # return cluster_mask [M, K]
    beta_db = np.asarray(beta_db)
    top_l = min(int(top_l), beta_db.shape[0])
    mask = np.zeros_like(beta_db, dtype=bool)
    for ue_idx in range(beta_db.shape[1]):
        mask[np.argsort(beta_db[:, ue_idx])[-top_l:], ue_idx] = True
    return mask


def assign_relay_association(relay_positions, ue_positions, blocked_mask, beta_relay_db, max_streams=None):
    # PSEUDOCODE:
    # for each blocked UE k:
    #     find relay r with strongest relay-to-UE large-scale fading beta_rk
    #     relay_assoc[k] = r
    # for unblocked UEs: relay_assoc[k] = -1 (no relay needed)
    # enforce L_max = 1 (already implied: one relay per UE)
    # enforce C_r^max: strongest links are placed first, a full relay is skipped
    # return relay_assoc [K]
    del relay_positions, ue_positions
    beta = np.asarray(beta_relay_db)
    assoc = np.full(len(blocked_mask), -1, dtype=int)
    load = np.zeros(beta.shape[0], dtype=int)
    blocked = np.flatnonzero(blocked_mask)
    for ue_idx in blocked[np.argsort(-beta[:, blocked].max(axis=0))]:
        for relay_id in np.argsort(-beta[:, ue_idx]):
            if max_streams is None or load[relay_id] < max_streams:
                assoc[ue_idx] = int(relay_id)
                load[relay_id] += 1
                break
    return assoc


def path_loss_gain_db(source, target, simulation):
    # Deterministic part of the large-scale gain, [len(source), len(target)]
    distances = np.maximum(np.linalg.norm(source[:, None, :] - target[None, :, :], axis=-1), simulation["min_distance_m"])
    return simulation["path_loss_ref_db"] - 10.0 * simulation["path_loss_exponent"] * np.log10(distances)


def default_association(scenario, config):
    """The association every algorithm starts from: Step 1's matching with zero QoS prices, or the
    earlier rule that gives each blocked UE its strongest relay."""
    network = config["network"]
    if config.get("association", {}).get("policy", "blocked_strongest") == "matching":
        from sim.association import matched_association
        return matched_association(scenario, config)
    return assign_relay_association(scenario.relay_positions, scenario.ue_positions, scenario.blocked_mask,
                                    scenario.beta_relay_ue_db, network.get("max_streams_per_relay"))


def generate_drop(config, rng, infra_rng=None):
    """A drop: UE positions and every large-scale gain. With infra_rng, the AP and relay positions and
    the AP-relay shadowing come from it instead, so drops can share one deployment."""
    network = config["network"]
    blockage = config["blockage"]
    simulation = config["simulation"]
    area_size = simulation["area_size"]
    top_l = network.get("cluster_top_l", min(4, network["num_aps"]))

    # One stream per quantity: changing R must not change where the APs and UEs are, and a sweep
    # over K keeps its first UEs, so sweep points differ only in what the sweep changes.
    ap_rng, relay_rng, ue_rng, *shadow_rngs = independent_streams(rng, 6)
    if infra_rng is not None:
        ap_rng, relay_rng, shadow_rngs[2] = independent_streams(infra_rng, 3)
    ap_positions = sample_positions(network["num_aps"], area_size, ap_rng)
    relay_positions = sample_positions(network["num_relays"], area_size, relay_rng)
    ue_positions = sample_positions(network["num_ues"], area_size, ue_rng)
    blocked_mask = assign_blockage(ue_positions, blockage["frac_blocked_ues"], rng)

    def large_scale_gain_db(source, target, shadow_rng):
        # Shadowing is drawn once here: it is a property of the drop, not of a coherence block.
        gain = path_loss_gain_db(source, target, simulation)
        return gain + shadow_rng.normal(0.0, simulation["shadow_std_db"], gain.shape[::-1]).T

    beta_ap_ue_db = large_scale_gain_db(ap_positions, ue_positions, shadow_rngs[0])
    beta_ap_ue_db[:, blocked_mask] -= blockage["extra_loss_db"]
    beta_relay_ue_db = large_scale_gain_db(relay_positions, ue_positions, shadow_rngs[1])
    beta_ap_relay_db = large_scale_gain_db(ap_positions, relay_positions, shadow_rngs[2])
    cluster_mask = compute_ap_clustering(ap_positions, ue_positions, beta_ap_ue_db, top_l)
    scenario = Scenario(ap_positions, relay_positions, ue_positions, blocked_mask, cluster_mask,
                        np.full(network["num_ues"], -1, dtype=int), beta_ap_ue_db, beta_relay_ue_db, beta_ap_relay_db)
    scenario.relay_assoc = default_association(scenario, config)
    return scenario


def evolve_drop(scenario, config, rng):
    """One slow-timescale epoch later: UEs take a random step, and their shadowing decorrelates with
    the distance moved. Infrastructure, blockage and AP-relay links do not change."""
    simulation, mobility = config["simulation"], config["mobility"]
    extra = np.where(scenario.blocked_mask, config["blockage"]["extra_loss_db"], 0.0)
    old = scenario.ue_positions
    new = np.clip(old + rng.normal(0.0, mobility["step_m"], old.shape), 0.0, simulation["area_size"])
    rho = np.exp(-np.linalg.norm(new - old, axis=1) / mobility["decorrelation_m"])   # [K]

    def moved(beta_db, source, blocked_loss):
        shadow = beta_db + blocked_loss - path_loss_gain_db(source, old, simulation)
        fresh = rng.normal(0.0, simulation["shadow_std_db"], shadow.shape)
        shadow = rho[None, :] * shadow + np.sqrt(1.0 - rho[None, :] ** 2) * fresh
        return path_loss_gain_db(source, new, simulation) + shadow - blocked_loss

    result = scenario.copy()
    result.ue_positions = new
    result.beta_ap_ue_db = moved(scenario.beta_ap_ue_db, scenario.ap_positions, extra[None, :])
    result.beta_relay_ue_db = moved(scenario.beta_relay_ue_db, scenario.relay_positions, 0.0)
    result.cluster_mask = compute_ap_clustering(result.ap_positions, new, result.beta_ap_ue_db, config["network"]["cluster_top_l"])
    result.relay_assoc = default_association(result, config)
    return result
