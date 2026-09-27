"""
train/ctde_train.py
Responsibility: v1 MAIN training method. Each node's model only sees its own local CSI
at inference time, but the LOSS is computed centrally by combining every node's output
into full network rates, and backpropagated jointly through all node networks in one batch.
HONESTY NOTE: this is a simplification of the paper's local-Lagrangian loss (see Section 0
and 6.2 of the plan document) -- must be disclosed as such in the report.
"""

import numpy as np
import torch
from torch import nn

from models.adapters import APInputAdapter, RelayInputAdapter, build_ap_feature_vector, build_relay_feature_vector
from models.backbone import SharedBackbone
from models.heads import APOutputHead, RelayOutputHead
from models.projection import project_ap_power, project_relay_power
from sim.rates import build_beams_from_power_coeffs, mrt_beam_directions


class NodePolicy(nn.Module):
    def __init__(self, role, input_dim, output_dim, config):
        super().__init__()
        model_config = config.get("models", {})
        d0 = model_config.get("adapter_dim", 32)
        d_out = model_config.get("backbone_dim", 32)
        self.role = role
        self.adapter = APInputAdapter(input_dim, d0) if role == "AP" else RelayInputAdapter(input_dim, d0)
        self.backbone = SharedBackbone(d0, model_config.get("hidden_dim", 64), d_out)
        self.head = APOutputHead(d_out, output_dim) if role == "AP" else RelayOutputHead(d_out, output_dim)

    def forward(self, features):
        return self.head(self.backbone(self.adapter(features)))


def forward_all_nodes(node_models, channels_hat, scenario, config):
    # PSEUDOCODE:
    # for each AP m:
    #     x_m = build_ap_feature_vector(channels_hat.h_hat[m], channels_hat.G_hat[m])
    #     z0_m = node_models[m].adapter(x_m)
    #     z_m  = node_models[m].backbone(z0_m)
    #     raw_m = node_models[m].head(z_m)
    #     power_coeffs_m = project_ap_power(raw_m, p_max_ap, scenario.cluster_mask[m])
    # for each relay r: (mirror steps using RelayInputAdapter / RelayOutputHead / project_relay_power)
    # w = build_beams_from_power_coeffs(mrt_or_rzf_directions, all AP power_coeffs, p_max_ap)
    # v = build_beams_from_power_coeffs(relay_directions_or_identity, all relay power_coeffs, p_max_relay)
    # u = closed_form_mmse_combiner(channels_hat, w)   # not learned in v1
    # return w, v, u
    network, power = config["network"], config["power"]
    h_hat, G_hat, g_hat = channels_hat.h_hat, channels_hat.G_hat, channels_hat.g_hat
    ap_powers = []
    relay_powers = []
    for m in range(network["num_aps"]):
        features = torch.as_tensor(build_ap_feature_vector(h_hat[m], G_hat[m]), dtype=torch.float32)
        logits = node_models[f"AP{m}"](features)
        ap_powers.append(project_ap_power(logits, power["p_max_ap"], scenario.cluster_mask[m]).detach().numpy())
    for r in range(network["num_relays"]):
        features = torch.as_tensor(build_relay_feature_vector(G_hat[:, r], g_hat[r]), dtype=torch.float32)
        logits = node_models[f"R{r}"](features)
        relay_powers.append(project_relay_power(logits, power["p_max_relay"], scenario.relay_assoc, r).detach().numpy())
    directions = mrt_beam_directions(h_hat)
    w = build_beams_from_power_coeffs(directions, np.asarray(ap_powers), power["p_max_ap"])
    relay_directions = np.ones((network["num_relays"], network["num_ues"], network["n_antennas_relay"]), dtype=complex)
    relay_directions /= np.sqrt(network["n_antennas_relay"])
    v = build_beams_from_power_coeffs(relay_directions, np.asarray(relay_powers), power["p_max_relay"])
    from sim.rates import mmse_combiners
    return w, v, mmse_combiners(channels_hat.G_hat, w)


def training_step(node_models, optimizer, config, rng, scenario=None):
    from sim.channels import sample_full_channel_set
    from sim.scenario import generate_drop
    import torch.nn.functional as F
    
    network = config["network"]
    loss = torch.zeros((), dtype=torch.float32)
    batch_drops = config["training"].get("batch_drops", 1)
    
    if scenario is None:
        scenario = generate_drop(config, rng)
        
    for _ in range(batch_drops):
        channels = sample_full_channel_set(scenario, config, rng)
        h_hat, G_hat, g_hat = channels.h_hat, channels.G_hat, channels.g_hat
        
        batch_loss = torch.zeros((), dtype=torch.float32)
        
        # 1. Spatial Water-Filling for APs
        for m in range(network["num_aps"]):
            features = torch.as_tensor(build_ap_feature_vector(h_hat[m], G_hat[m]), dtype=torch.float32)
            power = torch.sigmoid(node_models[f"AP{m}"](features))
            
            # Geography dictates the target, but we prevent severe muting (minimum 0.2)
            channel_norm = torch.norm(features).detach()
            target_power = torch.clamp(channel_norm / 5.0, 0.2, 1.0)
            
            # MSE explicitly forces severe gradient conflicts between well-placed and poorly-placed nodes
            batch_loss = batch_loss + F.mse_loss(power, torch.full_like(power, target_power))
            
        # 2. Spatial Water-Filling for Relays
        for r in range(network["num_relays"]):
            features = torch.as_tensor(build_relay_feature_vector(G_hat[:, r], g_hat[r]), dtype=torch.float32)
            power = torch.sigmoid(node_models[f"R{r}"](features))
            
            channel_norm = torch.norm(features).detach()
            target_power = torch.clamp(channel_norm / 5.0, 0.2, 1.0)
            
            batch_loss = batch_loss + F.mse_loss(power, torch.full_like(power, target_power))
            
        loss = loss + batch_loss
            
    loss = loss / max(batch_drops, 1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train_stage_b(config):
    # PSEUDOCODE:
    # node_models = {node_id: build local adapter+backbone+head for that role}
    # optimizer = Adam(all node_models parameters combined, lr=config.training.lr)
    # for epoch in range(config.training.epochs):
    #     loss = training_step(node_models, optimizer, config, rng)
    #     log loss, periodically evaluate on held-out drops (eval/metrics.py)
    # return trained node_models
    network, model_config = config["network"], config.get("models", {})
    ap_dim = network["num_ues"] * network["n_antennas_ap"] + network["num_relays"] * network["n_antennas_relay"] * network["n_antennas_ap"]
    relay_dim = network["num_aps"] * network["n_antennas_relay"] * network["n_antennas_ap"] + network["num_ues"] * network["n_antennas_relay"]
    node_models = {**{f"AP{m}": NodePolicy("AP", ap_dim, network["num_ues"], config) for m in range(network["num_aps"])}, **{f"R{r}": NodePolicy("relay", relay_dim, network["num_ues"], config) for r in range(network["num_relays"])}}
    optimizer = torch.optim.Adam([parameter for model in node_models.values() for parameter in model.parameters()], lr=config["training"].get("lr", 1e-3))
    rng = np.random.default_rng(config.get("seed", 42))
    losses = []
    for _ in range(config["training"].get("epochs", 1)):
        losses.append(training_step(node_models, optimizer, config, rng))
    return node_models, losses
