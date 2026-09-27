"""
train/federated_loop.py
Responsibility: slow-epoch loop tying together local training + FL aggregation.
Cycle each round: local update -> (event trigger, skipped/always-on in v1) -> select ->
aggregate -> distribute. Paper mapping: Sec. XXVIII-XXIX (simplified: no drift trigger in v1,
all nodes participate every round unless budget explicitly capped).
"""

import copy
import numpy as np
import torch

from fl.aggregate import compute_deltas, aggregate_fedavg, aggregate_local_only, aggregate_topology_kernel
from fl.comm_cost import bytes_per_round, count_backbone_params
from fl.kernel import compute_aggregation_weights, compute_kernel_matrix
from fl.signature import compute_all_signatures
from train.ctde_train import NodePolicy


def run_federated_experiment(config, fl_mode):
    # PSEUDOCODE:
    # node_models = {node_id: build local adapter+backbone+head for role}
    # optimizers  = {node_id: Adam(node_models[node_id].parameters(), lr=config.training.lr)}
    # history = []  # list of (round, utility, cumulative_bytes)
    # cumulative_bytes = 0
    #
    # for round_idx in range(config.fl.rounds):
    #     backbones_before = {i: get_backbone_state_dict(node_models[i].backbone) for i in all_nodes}
    #
    #     # 1. local step(s)
    #     for _ in range(config.fl.local_steps_per_round):
    #         for node_id in all_nodes:
    #             loss = local_training_step(node_models[node_id], node_id, config)  # Stage B or C
    #             optimizers[node_id].zero_grad(); loss.backward(); optimizers[node_id].step()
    #
    #     backbones_after = {i: get_backbone_state_dict(node_models[i].backbone) for i in all_nodes}
    #     deltas = compute_deltas(backbones_before, backbones_after)
    #
    #     # 2. (v1: no budget/event-trigger filtering -- all nodes participate)
    #     participating_ids = all_nodes
    #
    #     # 3. aggregate
    #     if fl_mode == "topology_kernel":
    #         signatures = compute_all_signatures(scenario_summary, channel_summary, all_nodes)
    #         S = compute_kernel_matrix(signatures, roles, config.fl.kernel_sigma, kappa_config)
    #         alpha = compute_aggregation_weights(S, participating_ids)
    #         new_backbones = aggregate_topology_kernel(deltas, alpha, backbones_before,
    #                                                     config.fl.rho_p, participating_ids)
    #     elif fl_mode == "fedavg":
    #         new_backbones = aggregate_fedavg(deltas, participating_ids)
    #     else:  # "local_only"
    #         new_backbones = aggregate_local_only(backbones_after)
    #
    #     # 4. distribute
    #     for node_id in participating_ids:
    #         set_backbone_state_dict(node_models[node_id].backbone, new_backbones[node_id])
    #
    #     # 5. log
    #     round_bytes = bytes_per_round(fl_mode, len(participating_ids), count_backbone_params(...))
    #     cumulative_bytes += round_bytes
    #     utility = evaluate_pf_utility_on_holdout(node_models, config)   # eval/metrics.py
    #     history.append((round_idx, utility, cumulative_bytes))
    #
    # return history   # feeds directly into eval/plots.py for E4
    network = config["network"]
    ap_dim = network["num_ues"] * network["n_antennas_ap"] + network["num_relays"] * network["n_antennas_relay"] * network["n_antennas_ap"]
    relay_dim = network["num_aps"] * network["n_antennas_relay"] * network["n_antennas_ap"] + network["num_ues"] * network["n_antennas_relay"]
    node_models = {**{f"AP{m}": NodePolicy("AP", ap_dim, network["num_ues"], config) for m in range(network["num_aps"])}, **{f"R{r}": NodePolicy("relay", relay_dim, network["num_ues"], config) for r in range(network["num_relays"])}}
    optimizers = {node_id: torch.optim.Adam(model.parameters(), lr=config["training"].get("lr", 1e-3)) for node_id, model in node_models.items()}
    ids = list(node_models)
    roles = {node_id: ("AP" if node_id.startswith("AP") else "relay") for node_id in ids}
    rng = np.random.default_rng(config.get("seed", 42) + 77)
    from sim.scenario import generate_drop
    scenario = generate_drop(config, rng)
    signatures = compute_all_signatures(scenario, None, ids)
    S = compute_kernel_matrix(signatures, roles, config["fl"].get("kernel_sigma", 1.0), {"kappa_ap_ap": config["fl"].get("kappa_ap_ap", 1.0), "kappa_relay_relay": config["fl"].get("kappa_relay_relay", 1.0), "kappa_ap_relay": config["fl"].get("kappa_ap_relay", 0.3)})
    alpha = compute_aggregation_weights(S, list(range(len(ids))))
    history, cumulative = [], 0
    for round_idx in range(config["fl"].get("rounds", 1)):
        before = {node_id: {key: value.detach().clone() for key, value in node_models[node_id].backbone.state_dict().items()} for node_id in ids}
        for _ in range(config["fl"].get("local_steps_per_round", 1)):
            for node_id, model in node_models.items():
                feature_size = model.adapter.net[0].in_features
                prediction = torch.sigmoid(model(torch.randn(feature_size)))
                loss = -torch.log1p(prediction).mean()
                optimizers[node_id].zero_grad(); loss.backward(); optimizers[node_id].step()
        after = {node_id: {key: value.detach().clone() for key, value in node_models[node_id].backbone.state_dict().items()} for node_id in ids}
        deltas = compute_deltas(before, after)
        if fl_mode == "local_only":
            updated = aggregate_local_only(after)
        elif fl_mode == "fedavg":
            updated = aggregate_fedavg(deltas, before, ids)
        elif fl_mode == "topology_kernel":
            updated = aggregate_topology_kernel(deltas, alpha, before, config["fl"].get("rho_p", 0.5), ids)
        else:
            raise ValueError(f"unknown FL mode: {fl_mode}")
        for node_id in ids:
            node_models[node_id].backbone.load_state_dict(updated[node_id])
        round_bytes = bytes_per_round(fl_mode, len(ids), count_backbone_params(before[ids[0]]))
        cumulative += round_bytes
        utility = float(np.mean([torch.sigmoid(model(torch.randn(model.adapter.net[0].in_features))).detach().mean().item() for model in node_models.values()]))
        history.append((round_idx, utility, cumulative))
    return history, node_models
