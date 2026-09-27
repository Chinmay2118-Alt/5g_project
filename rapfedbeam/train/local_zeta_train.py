"""
train/local_zeta_train.py
Responsibility: STRETCH GOAL (Stage C). Each node computes its OWN loss using only its
own action plus a broadcast low-dimensional coordination vector zeta -- no node's training
step ever touches the full network state. This is the mechanism actually closer to the
paper's proposed method (Sec. XXVI), used to test whether local-CSI + zeta is sufficient.
"""

from dataclasses import dataclass
import numpy as np
import torch


def compute_zeta(scenario, channels, current_w, current_v, current_u, dual_prices):
    # PSEUDOCODE:
    # zeta = {
    #   'qos_dual_price_k'   : dual_prices.qos,          # per-UE QoS constraint price
    #   'df_dual_price_rk'   : dual_prices.df,           # per-relay-link decode constraint price
    #   'interference_I1_k'  : measured aggregate Phase-I interference per UE
    #   'interference_I2_k'  : measured aggregate Phase-II interference per UE
    #   'desired_signal_stats_k' : measured effective desired-signal power per UE
    # }
    # These are broadcast (small, scalar-per-UE quantities) -- NOT raw channel matrices.
    # return zeta
    del scenario, channels, current_w, current_v, current_u
    return {"qos": np.asarray(dual_prices.qos), "df": np.asarray(dual_prices.df), "I1": np.asarray(dual_prices.I1), "I2": np.asarray(dual_prices.I2), "desired": np.asarray(dual_prices.desired)}


def local_loss_ap(node_model, x_m, zeta_slice_for_served_ues, power_penalty_weight):
    # PSEUDOCODE:
    # power_coeffs_m = node_model.forward(x_m)   # this node's own adapter->backbone->head->proj
    # local_rate_proxy_k = <combine power_coeffs_m with zeta's desired/interference stats
    #                        to get an APPROXIMATE per-UE rate contribution, computable
    #                        WITHOUT other nodes' raw channels>
    # loss = -sum_k( qos_dual_price_k * local_rate_proxy_k ) + power_penalty_weight * ||power_coeffs_m||^2
    # return loss
    power = node_model(x_m)
    desired = torch.as_tensor(zeta_slice_for_served_ues["desired"], dtype=power.dtype, device=power.device)
    interference = torch.as_tensor(zeta_slice_for_served_ues["I1"], dtype=power.dtype, device=power.device)
    qos = torch.as_tensor(zeta_slice_for_served_ues["qos"], dtype=power.dtype, device=power.device)
    proxy_rate = torch.log1p(desired * power.square() / (interference + 1e-6))
    return -(qos * proxy_rate).sum() + power_penalty_weight * power.square().sum()


def local_loss_relay(node_model, x_r, zeta_slice_for_served_ues, power_penalty_weight):
    # PSEUDOCODE: mirrors local_loss_ap, but also includes the DF dual price term
    # penalizing this relay if its decode capacity C_AR_rk is the binding constraint
    power = node_model(x_r)
    desired = torch.as_tensor(zeta_slice_for_served_ues["desired"], dtype=power.dtype, device=power.device)
    interference = torch.as_tensor(zeta_slice_for_served_ues["I2"], dtype=power.dtype, device=power.device)
    qos = torch.as_tensor(zeta_slice_for_served_ues["qos"], dtype=power.dtype, device=power.device)
    df = torch.as_tensor(zeta_slice_for_served_ues["df"], dtype=power.dtype, device=power.device)
    proxy_rate = torch.log1p(desired * power.square() / (interference + 1e-6))
    return -(qos * proxy_rate).sum() - (df * proxy_rate).sum() + power_penalty_weight * power.square().sum()


def update_dual_prices(dual_prices, measured_constraint_violations, dual_step_size):
    # PSEUDOCODE (primal-dual update, standard Lagrangian ascent on the dual variables):
    # dual_prices.qos += dual_step_size * (R_min_k - measured_rate_k)   # clipped at 0
    # dual_prices.df  += dual_step_size * (measured_C_AR_violation)      # clipped at 0
    # return updated dual_prices
    updated = dual_prices
    updated.qos = np.maximum(0.0, updated.qos + dual_step_size * np.asarray(measured_constraint_violations["qos"]))
    updated.df = np.maximum(0.0, updated.df + dual_step_size * np.asarray(measured_constraint_violations["df"]))
    return updated


def train_stage_c(config):
    # PSEUDOCODE:
    # dual_prices = init_dual_prices()
    # node_models = {node_id: build local adapter+backbone+head}
    # optimizers = {node_id: Adam(node_models[node_id].parameters(), lr=config.training.lr)}
    # for epoch in range(config.training.epochs):
    #     scenario, channels = generate_drop + sample_channels
    #     zeta = compute_zeta(scenario, channels, ..., dual_prices)   # broadcast, read-only for nodes
    #     for each node i independently:
    #         loss_i = local_loss_ap_or_relay(node_models[i], x_i, zeta[served_ues(i)], ...)
    #         optimizers[i].zero_grad(); loss_i.backward(); optimizers[i].step()
    #     periodically: measure actual constraint violations, update_dual_prices(...)
    # return trained node_models
    @dataclass
    class DualPrices:
        qos: np.ndarray
        df: np.ndarray
        I1: np.ndarray
        I2: np.ndarray
        desired: np.ndarray
    return DualPrices(np.ones(config["network"]["num_ues"]), np.ones((config["network"]["num_relays"], config["network"]["num_ues"])), np.ones(config["network"]["num_ues"]), np.ones(config["network"]["num_ues"]), np.ones(config["network"]["num_ues"])), None
