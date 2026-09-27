"""
fl/aggregate.py
Responsibility: three comparable aggregation modes applied to the BACKBONE parameters only
(adapters/heads never leave the node). This is the core mechanism for Claim 3.
Paper mapping: Sec. XXVI-XXVII
"""

import torch


def compute_deltas(node_backbones_before, node_backbones_after):
    # PSEUDOCODE:
    # for each node i: delta_theta[i] = state_dict_after[i] - state_dict_before[i]  (elementwise)
    # return {node_id: delta_theta}
    return {node_id: {key: node_backbones_after[node_id][key] - node_backbones_before[node_id][key] for key in node_backbones_before[node_id]} for node_id in node_backbones_before}


def aggregate_local_only(node_backbones_after, **kwargs):
    # PSEUDOCODE: no aggregation at all -- just return node_backbones_after unchanged
    # (this is the lower-bound baseline)
    return node_backbones_after


def aggregate_fedavg(deltas, participating_ids):
    # PSEUDOCODE:
    # avg_delta = mean(deltas[i] for i in participating_ids)   # elementwise average, uniform
    # new_state = {i: node_backbones_before[i] + avg_delta for i in participating_ids}
    # (SAME averaged update applied identically to every node -- no personalization)
    # return new_state
    ids = list(participating_ids)
    return {node_id: {key: node_backbones_before[node_id][key] + sum(deltas[j][key] for j in ids) / len(ids) for key in node_backbones_before[node_id]} for node_id in ids}


def aggregate_topology_kernel(deltas, alpha, node_backbones_before, rho_p, participating_ids):
    # PSEUDOCODE (paper Sec. XXVII):
    # for each node i in participating_ids:
    #     theta_hat_i = node_backbones_before[i] + sum_j( alpha[i, j] * deltas[j] )
    #     theta_new_i = (1 - rho_p) * node_backbones_before[i] + rho_p * theta_hat_i
    # return {i: theta_new_i for i in participating_ids}
    # NOTE: this is the ONLY mode where different nodes get genuinely different updates
    #       based on their topology similarity -- this is what Claim 3 is testing.
    ids = list(participating_ids)
    result = {}
    for row, node_id in enumerate(ids):
        result[node_id] = {key: (1.0 - rho_p) * node_backbones_before[node_id][key] + rho_p * (node_backbones_before[node_id][key] + sum(alpha[row, col] * deltas[other][key] for col, other in enumerate(ids))) for key in node_backbones_before[node_id]}
    return result


def run_aggregation_round(mode, node_backbones_before, node_backbones_after, alpha=None, rho_p=None):
    # PSEUDOCODE: dispatch to one of the three functions above based on `mode` string
    # ("local_only" | "fedavg" | "topology_kernel")
    if mode == "local_only":
        return aggregate_local_only(node_backbones_after)
    ids = list(node_backbones_before)
    if mode == "fedavg":
        return aggregate_fedavg(compute_deltas(node_backbones_before, node_backbones_after), ids)
    if mode == "topology_kernel":
        if alpha is None or rho_p is None:
            raise ValueError("topology_kernel aggregation requires alpha and rho_p")
        return aggregate_topology_kernel(compute_deltas(node_backbones_before, node_backbones_after), alpha, node_backbones_before, rho_p, ids)
    raise ValueError(f"unknown aggregation mode: {mode}")
