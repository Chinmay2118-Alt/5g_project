"""
sim/scenario.py
Responsibility: place APs, relays, UEs; assign blockage; compute AP clustering.
Paper mapping: Sec. III (clustering), Sec. II (network sets M, R, K)
"""

from dataclasses import dataclass
import copy
import numpy as np


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


def assign_relay_association(relay_positions, ue_positions, blocked_mask, beta_relay_db):
    # PSEUDOCODE:
    # for each blocked UE k:
    #     find relay r with strongest relay-to-UE large-scale fading beta_rk
    #     relay_assoc[k] = r
    # for unblocked UEs: relay_assoc[k] = -1 (no relay needed)
    # enforce L_max = 1 (already implied: one relay per UE)
    # return relay_assoc [K]
    del relay_positions, ue_positions
    assoc = np.full(len(blocked_mask), -1, dtype=int)
    for ue_idx in np.flatnonzero(blocked_mask):
        assoc[ue_idx] = int(np.argmax(np.asarray(beta_relay_db)[:, ue_idx]))
    return assoc


def generate_drop(config, rng):
    # PSEUDOCODE:
    # ap_pos = sample_positions(M, area, rng)
    # relay_pos = sample_positions(R, area, rng)
    # ue_pos = sample_positions(K, area, rng)
    # blocked = assign_blockage(ue_pos, config.blockage.frac_blocked_ues, rng)
    # cluster_mask = compute_ap_clustering(ap_pos, ue_pos, ..., top_l=...)
    # relay_assoc = assign_relay_association(relay_pos, ue_pos, blocked, ...)
    # return Scenario(ap_pos, relay_pos, ue_pos, blocked, cluster_mask, relay_assoc)
    network = config["network"]
    blockage = config["blockage"]
    simulation = config.get("simulation", {})
    area_size = simulation.get("area_size", 500.0)
    path_loss_exponent = simulation.get("path_loss_exponent", 3.2)
    shadow_std_db = simulation.get("shadow_std_db", 4.0)
    top_l = network.get("cluster_top_l", min(4, network["num_aps"]))

    ap_positions = sample_positions(network["num_aps"], area_size, rng)
    relay_positions = sample_positions(network["num_relays"], area_size, rng)
    ue_positions = sample_positions(network["num_ues"], area_size, rng)
    blocked_mask = assign_blockage(ue_positions, blockage["frac_blocked_ues"], rng)

    def path_loss_db(source, target):
        distances = np.maximum(np.linalg.norm(source[:, None, :] - target[None, :, :], axis=-1), 1.0)
        return -(-30.0 + 10.0 * path_loss_exponent * np.log10(distances) + rng.normal(0.0, shadow_std_db, distances.shape))

    beta_ap_ue_db = path_loss_db(ap_positions, ue_positions)
    beta_ap_ue_db[:, blocked_mask] -= blockage["extra_loss_db"]
    beta_relay_ue_db = path_loss_db(relay_positions, ue_positions)
    cluster_mask = compute_ap_clustering(ap_positions, ue_positions, beta_ap_ue_db, top_l)
    relay_assoc = assign_relay_association(relay_positions, ue_positions, blocked_mask, beta_relay_ue_db)
    return Scenario(ap_positions, relay_positions, ue_positions, blocked_mask, cluster_mask,
                    relay_assoc, beta_ap_ue_db, beta_relay_ue_db)
