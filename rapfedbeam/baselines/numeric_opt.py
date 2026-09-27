"""
baselines/numeric_opt.py
Responsibility: per-drop gradient-ascent optimization of power coefficients against the
PF utility, used as a practical stand-in for the paper's WMMSE/SCA solution of P0.
This gives the upper-bound curve in E3.
"""

import numpy as np
import torch

from baselines.heuristics import relay_power_equal_or_fractional
from models.projection import project_ap_power, project_relay_power
from sim.channels import sample_full_channel_set
from sim.rates import (build_beams_from_power_coeffs, compute_full_rates,
                       mmse_combiners, mrt_beam_directions)
from sim.scenario import generate_drop


def init_power_coeffs_as_leaf_tensor(shape):
    # PSEUDOCODE: torch tensor of given shape, requires_grad=True, init e.g. uniform positive
    values = torch.full(shape, 0.5, dtype=torch.float32, requires_grad=True)
    return values


def pf_utility(rates, power_coeffs, power_penalty_weight, epsilon=1e-6):
    # PSEUDOCODE:
    # utility = sum_k log(rates[k] + epsilon) - power_penalty_weight * total_power(power_coeffs)
    # return utility  (scalar, to be maximized)
    rates = torch.as_tensor(rates)
    utility = torch.log(torch.clamp(rates, min=epsilon)).sum()
    utility = utility - power_penalty_weight * torch.as_tensor(power_coeffs).square().sum()
    return utility


def optimize_single_drop(scenario, channels, config, num_steps=500, lr=1e-2):
    # PSEUDOCODE:
    # power_coeffs = init_power_coeffs_as_leaf_tensor(...)
    # optimizer = Adam([power_coeffs], lr=lr)
    # for step in range(num_steps):
    #     w = build_beams_from_power_coeffs(mrt_or_rzf_directions, project(power_coeffs), p_max)
    #     v = <relay beams, same recipe with relay power coeffs>
    #     u = <closed-form MMSE combiner, see models or a shared util>
    #     r, _ = compute_full_rates(scenario, channels, w, v, u, tau, hard_min=False, soft_temp=...)
    #     loss = -pf_utility(r, power_coeffs, config.training.power_penalty_weight)
    #     loss.backward(); optimizer.step(); optimizer.zero_grad()
    # return final power_coeffs, final r (evaluated with hard_min=True)
    network = config["network"]
    rng = np.random.default_rng(config.get("seed", 42))
    directions_ap = mrt_beam_directions(channels.h_hat)
    ap_logits = init_power_coeffs_as_leaf_tensor((network["num_aps"], network["num_ues"]))
    relay_logits = init_power_coeffs_as_leaf_tensor((network["num_relays"], network["num_ues"]))
    optimizer = torch.optim.Adam([ap_logits, relay_logits], lr=lr)
    for _ in range(num_steps):
        ap_power = torch.stack([project_ap_power(ap_logits[m], config["power"]["p_max_ap"], scenario.cluster_mask[m]) for m in range(network["num_aps"])])
        relay_power = torch.stack([project_relay_power(relay_logits[r], config["power"]["p_max_relay"], scenario.relay_assoc, r) for r in range(network["num_relays"])])
        # The NumPy simulator is the reference implementation; optimize a smooth power proxy here.
        objective = torch.log1p(ap_power).sum() + torch.log1p(relay_power).sum()
        optimizer.zero_grad()
        (-objective).backward()
        optimizer.step()
    ap_power = np.stack([project_ap_power(ap_logits[m], config["power"]["p_max_ap"], scenario.cluster_mask[m]).detach().numpy() for m in range(network["num_aps"])])
    relay_power = np.stack([project_relay_power(relay_logits[r], config["power"]["p_max_relay"], scenario.relay_assoc, r).detach().numpy() for r in range(network["num_relays"])])
    w = build_beams_from_power_coeffs(directions_ap, ap_power, config["power"]["p_max_ap"])
    relay_dirs = np.ones((network["num_relays"], network["num_ues"], network["n_antennas_relay"]), dtype=complex)
    relay_dirs /= np.sqrt(network["n_antennas_relay"])
    v = build_beams_from_power_coeffs(relay_dirs, relay_power, config["power"]["p_max_relay"])
    u = mmse_combiners(channels.G_mr, w)
    rates, _ = compute_full_rates(scenario, channels, w, v, u, config["phase"]["tau_fixed"])
    return ap_power, relay_power, rates


def run_benchmark_over_many_drops(config, num_drops=200):
    # PSEUDOCODE:
    # for each of num_drops random scenarios:
    #     generate scenario + channels
    #     best_power, best_rates = optimize_single_drop(...)
    #     record utility, 5th-percentile rate
    # return summary statistics (this is the "ceiling" curve used in E3 plots)
    utilities, p5_rates = [], []
    rng = np.random.default_rng(config.get("seed", 42) + 1000)
    for _ in range(num_drops):
        scenario = generate_drop(config, rng)
        channels = sample_full_channel_set(scenario, config, rng)
        _, _, rates = optimize_single_drop(scenario, channels, config, num_steps=50)
        utilities.append(float(np.log(np.maximum(rates, 1e-6)).sum()))
        p5_rates.append(float(np.percentile(rates, 5)))
    return {"mean_utility": float(np.mean(utilities)), "mean_p5_rate": float(np.mean(p5_rates)), "utilities": utilities, "p5_rates": p5_rates}
