"""
fl/signature.py
Responsibility: compute the slow-varying topology signature s_i per node, used by
the RBF topology kernel for personalized aggregation.
Paper mapping: Sec. XXIII
"""

import numpy as np


def compute_node_signature(node_id, role, beta_to_ues, interference_estimate, load_estimate):
    # PSEUDOCODE:
    # beta_norm = normalize(beta_to_ues)               # vector, one entry per UE (or padded/binned)
    # I_norm    = normalize(interference_estimate)      # scalar
    # L_norm    = normalize(load_estimate)               # scalar, e.g. num UEs served / capacity
    # role_flag = 1.0 if role == 'AP' else 0.0
    # s_i = concat([beta_norm, I_norm, L_norm, role_flag])
    # return s_i   # fixed-length vector, SAME length for AP and relay nodes
    beta = np.asarray(beta_to_ues, dtype=float)
    beta_norm = (beta - beta.mean()) / (beta.std() + 1e-8)
    interference = np.asarray(interference_estimate, dtype=float).reshape(-1)
    load = np.asarray(load_estimate, dtype=float).reshape(-1)
    return np.concatenate([beta_norm.ravel(), [(interference.mean() - 0.0) / (abs(interference).mean() + 1e-8)], [load.mean()], [1.0 if role.lower() == "ap" else 0.0]]).astype(np.float32)


def compute_all_signatures(scenario, channels, node_list):
    # PSEUDOCODE:
    # for each node i in node_list (APs then relays):
    #     beta_to_ues = <slice of large-scale fading relevant to this node>
    #     interference_estimate = <measured or estimated aggregate interference>
    #     load_estimate = <num UEs currently served by this node>
    #     s_i = compute_node_signature(...)
    # return {node_id: s_i for all nodes}
    signatures = {}
    beta_ap = np.asarray(getattr(scenario, "beta_ap_ue_db"))
    beta_relay = np.asarray(getattr(scenario, "beta_relay_ue_db"))
    for node_id in node_list:
        if str(node_id).upper().startswith("A") or (isinstance(node_id, int) and node_id < beta_ap.shape[0]):
            index = int(str(node_id).lstrip("APap") or node_id)
            signatures[node_id] = compute_node_signature(node_id, "AP", beta_ap[index], 0.0, scenario.cluster_mask[index].mean())
        else:
            index = int(str(node_id).lstrip("Rrelay"))
            signatures[node_id] = compute_node_signature(node_id, "relay", beta_relay[index], 0.0, (scenario.relay_assoc == index).mean())
    return signatures
