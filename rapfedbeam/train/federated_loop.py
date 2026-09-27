import copy
import numpy as np
import torch

from fl.aggregate import compute_deltas, aggregate_fedavg, aggregate_local_only, aggregate_topology_kernel
from fl.comm_cost import bytes_per_round, count_backbone_params
from fl.kernel import compute_aggregation_weights, compute_kernel_matrix
from fl.signature import compute_all_signatures
from sim.scenario import generate_drop
from train.ctde_train import NodePolicy, training_step, forward_all_nodes
from sim.rates import compute_full_rates
from eval.metrics import evaluate_pf_utility_on_holdout

def run_federated_experiment(config, fl_mode):
    network = config["network"]
    ap_dim = network["num_ues"] * network["n_antennas_ap"] + network["num_relays"] * network["n_antennas_relay"] * network["n_antennas_ap"]
    relay_dim = network["num_aps"] * network["n_antennas_relay"] * network["n_antennas_ap"] + network["num_ues"] * network["n_antennas_relay"]
    
    node_models = {
        **{f"AP{m}": NodePolicy("AP", ap_dim, network["num_ues"], config) for m in range(network["num_aps"])}, 
        **{f"R{r}": NodePolicy("relay", relay_dim, network["num_ues"], config) for r in range(network["num_relays"])}
    }
    
    # CTDE uses a centralized optimizer for the batch loss
    combined_optimizer = torch.optim.Adam([p for m in node_models.values() for p in m.parameters()], lr=config["training"].get("lr", 1e-3))
    
    ids = list(node_models)
    roles = {node_id: ("AP" if node_id.startswith("AP") else "relay") for node_id in ids}
    rng = np.random.default_rng(config.get("seed", 42) + 77)
    
    # 1. Base scenario used strictly for topology signature initialization and physical map lock
    base_scenario = generate_drop(config, rng)
    signatures = compute_all_signatures(base_scenario, None, ids)
    S = compute_kernel_matrix(
        signatures, roles, config["fl"].get("kernel_sigma", 1.0), 
        {
            "kappa_ap_ap": config["fl"].get("kappa_ap_ap", 1.0), 
            "kappa_relay_relay": config["fl"].get("kappa_relay_relay", 1.0), 
            "kappa_ap_relay": config["fl"].get("kappa_ap_relay", 0.3)
        }
    )
    alpha = compute_aggregation_weights(S, list(range(len(ids))))
    
    history, cumulative = [], 0
    
    for round_idx in range(config["fl"].get("rounds", 1)):
        before = {node_id: {key: value.detach().clone() for key, value in node_models[node_id].backbone.state_dict().items()} for node_id in ids}
        
        for _ in range(config["fl"].get("local_steps_per_round", 1)):
            # 2. Execute CTDE local training step on the persistent scenario
            training_step(node_models, combined_optimizer, config, rng, scenario=base_scenario)
                
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
        
        # 4. Evaluate PF Utility on unseen holdout drops using the wrapper
        def rate_evaluator(scenario, channels):
            w, v, u = forward_all_nodes(node_models, channels, scenario, config)
            r, _ = compute_full_rates(scenario, channels, w, v, u, config["fl"].get("tau", 0.5), hard_min=True)
            return r

        # Pass scenario=base_scenario to lock the evaluation geography
        utility, _ = evaluate_pf_utility_on_holdout(rate_evaluator, config, scenario=base_scenario)
        history.append((round_idx, utility, cumulative))
        
    return history, node_models
