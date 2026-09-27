"""
fl/aggregate.py
Responsibility: three comparable aggregation modes applied to the BACKBONE parameters only
(adapters/heads never leave the node). This is the core mechanism for Claim 3.
Paper mapping: Sec. XXVI-XXVII
"""

import torch

def compute_deltas(node_backbones_before, node_backbones_after):
    return {
        node_id: {
            key: node_backbones_after[node_id][key] - node_backbones_before[node_id][key] 
            for key in node_backbones_before[node_id] if 'backbone' in key.lower()
        } 
        for node_id in node_backbones_before
    }

def aggregate_local_only(node_backbones_after, **kwargs):
    return node_backbones_after

def aggregate_fedavg(deltas, node_backbones_before, participating_ids):
    ids = list(participating_ids)
    result = {}
    for node_id in ids:
        result[node_id] = {}
        for key in node_backbones_before[node_id]:
            if 'backbone' in key.lower():
                result[node_id][key] = node_backbones_before[node_id][key] + sum(deltas[j][key] for j in ids) / len(ids)
            else:
                result[node_id][key] = node_backbones_before[node_id][key]
    return result

def aggregate_topology_kernel(deltas, alpha, node_backbones_before, rho_p, participating_ids):
    ids = list(participating_ids)
    result = {}
    for row, node_id in enumerate(ids):
        result[node_id] = {}
        for key in node_backbones_before[node_id]:
            if 'backbone' in key.lower():
                theta_hat = node_backbones_before[node_id][key] + sum(alpha[row, col] * deltas[other][key] for col, other in enumerate(ids))
                result[node_id][key] = (1.0 - rho_p) * node_backbones_before[node_id][key] + rho_p * theta_hat
            else:
                result[node_id][key] = node_backbones_before[node_id][key]
    return result

def run_aggregation_round(mode, node_backbones_before, node_backbones_after, alpha=None, rho_p=None):
    if mode == "local_only":
        return aggregate_local_only(node_backbones_after)
    ids = list(node_backbones_before)
    if mode == "fedavg":
        return aggregate_fedavg(compute_deltas(node_backbones_before, node_backbones_after), node_backbones_before, ids)
    if mode == "topology_kernel":
        if alpha is None or rho_p is None:
            raise ValueError("topology_kernel aggregation requires alpha and rho_p")
        return aggregate_topology_kernel(compute_deltas(node_backbones_before, node_backbones_after), alpha, node_backbones_before, rho_p, ids)
    raise ValueError(f"unknown aggregation mode: {mode}")
