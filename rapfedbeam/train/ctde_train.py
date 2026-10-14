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
    """One CTDE training step: maximise PF utility via a differentiable signal-power proxy.

    The proxy replaces the non-differentiable numpy simulator with a MRT signal-amplitude
    approximation that IS differentiable through project_ap_power / project_relay_power:

        Phase-I proxy:  S1_k = (Σ_m cluster_mk · p_mk · ||h_hat_mk||)²
        Phase-II proxy: S2_k = (Σ_r assoc_rk  · p_rk · ||g_hat_rk||)²

    These are lower bounds on the single-user (zero-interference) SINRs under MRT/equal-direction
    beamforming.  The PF utility proxy maximised here is Σ_k log2(1 + S_k / noise + ε).

    Gradient clipping (max_norm=1.0) prevents parameter explosion.
    """
    from sim.channels import sample_full_channel_set
    from sim.scenario import generate_drop

    network, power_cfg = config["network"], config["power"]
    training_cfg = config["training"]
    batch_drops = training_cfg.get("batch_drops", 1)
    p_max_ap = power_cfg["p_max_ap"]
    p_max_relay = power_cfg["p_max_relay"]
    tau = config["phase"]["tau_fixed"]
    penalty_w = training_cfg.get("power_penalty_weight", 0.01)
    noise_var = 1e-3
    eps = 1e-6

    if scenario is None:
        scenario = generate_drop(config, rng)

    # Relay association mask (fixed per scenario) — shape [R, K]
    relay_mask = torch.zeros(network["num_relays"], network["num_ues"], dtype=torch.float32)
    for k, r_id in enumerate(scenario.relay_assoc):
        if r_id >= 0:
            relay_mask[r_id, k] = 1.0

    # AP cluster mask — shape [M, K]
    cluster_t = torch.as_tensor(scenario.cluster_mask.astype(np.float32))  # [M, K]

    all_params = [p for m in node_models.values() for p in m.parameters()]
    optimizer.zero_grad()
    cumulative_loss = 0.0

    for _ in range(batch_drops):
        channels = sample_full_channel_set(scenario, config, rng)
        h_hat = channels.h_hat  # [M, K, N_A]
        G_hat = channels.G_hat  # [M, R, N_R, N_A]
        g_hat = channels.g_hat  # [R, K, N_R]

        # --- Channel norms squared (no gradient needed) ---
        # h_norms_sq[m, k] = ||h_hat[m, k, :]||²
        h_norms_sq = torch.as_tensor(
            np.sum(np.abs(h_hat) ** 2, axis=-1).astype(np.float32)
        )  # [M, K]
        # g_norms_sq[r, k] = ||g_hat[r, k, :]||²
        g_norms_sq = torch.as_tensor(
            np.sum(np.abs(g_hat) ** 2, axis=-1).astype(np.float32)
        )  # [R, K]

        # --- AP forward pass (with gradient graph) ---
        ap_powers = []
        for m in range(network["num_aps"]):
            feat = torch.as_tensor(
                build_ap_feature_vector(h_hat[m], G_hat[m]), dtype=torch.float32
            )
            logits = node_models[f"AP{m}"](feat)
            ap_powers.append(project_ap_power(logits, p_max_ap, scenario.cluster_mask[m]))
            # ap_powers[m]: shape [K]

        # --- Relay forward pass (with gradient graph) ---
        relay_powers = []
        for r in range(network["num_relays"]):
            feat = torch.as_tensor(
                build_relay_feature_vector(G_hat[:, r], g_hat[r]), dtype=torch.float32
            )
            logits = node_models[f"R{r}"](feat)
            relay_powers.append(project_relay_power(logits, p_max_relay, scenario.relay_assoc, r))
            # relay_powers[r]: shape [K]

        # --- Differentiable Phase-I signal amplitude (MRT coherent gain proxy) ---
        # signal_amp_1[k] = Σ_m cluster_mk * p_mk * ||h_hat_mk||
        # Under MRT: h_hat_mk^H w_mk = p_mk * ||h_hat_mk||  (exactly)
        signal_amp_1 = torch.zeros(network["num_ues"])
        for m, p_m in enumerate(ap_powers):
            signal_amp_1 = signal_amp_1 + cluster_t[m] * p_m * torch.sqrt(h_norms_sq[m].clamp(min=0.0))

        # Phase-I capacity proxy: tau * log2(1 + S1_k / noise)
        phase1_cap = tau * torch.log2(1.0 + signal_amp_1.square() / noise_var + eps)

        # --- Differentiable Phase-II signal amplitude (relay DF proxy) ---
        # signal_amp_2[k] = Σ_r assoc_rk * p_rk * ||g_hat_rk||
        signal_amp_2 = torch.zeros(network["num_ues"])
        for r, p_r in enumerate(relay_powers):
            signal_amp_2 = signal_amp_2 + relay_mask[r] * p_r * torch.sqrt(g_norms_sq[r].clamp(min=0.0))

        # Phase-II capacity proxy: (1-tau) * log2(1 + S2_k / noise)
        phase2_cap = (1.0 - tau) * torch.log2(1.0 + signal_amp_2.square() / noise_var + eps)

        # --- Combined rate proxy (DF constraint: rate_k = min(C_D_k, C_AR_k)) ---
        # For non-relay UEs: rate = phase1_cap (phase2 term is zero)
        # For relay UEs: apply DF min (relay decode proxy ≈ phase1_cap from AP→relay link)
        rate_proxy = phase1_cap + phase2_cap

        # --- PF utility (sum of log rates proxy) ---
        # Use log1p(rate * scale) instead of log2(rate) to avoid 1/(tiny_rate) gradient explosion.
        # scale=1000 shifts the knee of log1p away from zero (log1p(0) gradient = 1, not 1/eps).
        rate_scale = 1000.0  # dimensionless; makes log1p gradient well-conditioned at SNR <<1
        pf_proxy = torch.log1p(rate_proxy * rate_scale).sum()

        # --- Power regularisation (encourage feasibility headroom) ---
        power_sq = (
            sum(p.square().sum() for p in ap_powers) +
            sum(p.square().sum() for p in relay_powers)
        )
        power_penalty = penalty_w * power_sq

        drop_loss = (-pf_proxy + power_penalty) / batch_drops

        # Backward PER DROP so each drop's gradient is clipped before the next accumulates.
        # This prevents one extreme-channel drop (with near-zero signal_amp, causing huge
        # 1/phase1_cap gradient) from polluting the accumulated gradient buffer.
        drop_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0 / batch_drops)
        cumulative_loss += float(drop_loss.detach())

    optimizer.step()
    return cumulative_loss



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
    node_models = {
        **{f"AP{m}": NodePolicy("AP", ap_dim, network["num_ues"], config) for m in range(network["num_aps"])},
        **{f"R{r}": NodePolicy("relay", relay_dim, network["num_ues"], config) for r in range(network["num_relays"])}
    }
    optimizer = torch.optim.Adam(
        [parameter for model in node_models.values() for parameter in model.parameters()],
        lr=config["training"].get("lr", 1e-3)
    )
    rng = np.random.default_rng(config.get("seed", 42))
    losses = []
    for _ in range(config["training"].get("epochs", 1)):
        losses.append(training_step(node_models, optimizer, config, rng))
    return node_models, losses
