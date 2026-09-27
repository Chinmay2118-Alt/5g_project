"""
eval/plots.py
Responsibility: generate every plot needed for E1-E6.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_rate_cdf(rate_distributions_dict, title, save_path):
    # PSEUDOCODE:
    # for label, rates in rate_distributions_dict.items():
    #     sorted_r, cdf = per_user_rate_cdf(rates)
    #     plt.plot(sorted_r, cdf, label=label)
    # plt.xlabel('per-user rate'); plt.ylabel('CDF'); plt.legend(); plt.title(title)
    # plt.savefig(save_path)
    for label, rates in rate_distributions_dict.items():
        sorted_rates, cdf = per_user_rate_cdf(rates)
        plt.plot(sorted_rates, cdf, label=label)
    plt.xlabel("per-user rate")
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_tau_sweep(tau_values, mean_rates, p5_rates, save_path):
    # PSEUDOCODE (E2): two lines vs tau -- mean rate and 5th-percentile rate
    fig, axis = plt.subplots()
    axis.plot(tau_values, mean_rates, marker="o", label="mean rate")
    axis.plot(tau_values, p5_rates, marker="o", label="5th percentile")
    axis.set(xlabel="tau", ylabel="rate")
    axis.legend(); fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


def plot_policy_vs_heuristics_bar(utilities_dict, save_path):
    # PSEUDOCODE (E3): bar chart -- equal-power, fractional-power, learned (Stage B), numeric-optimum
    fig, axis = plt.subplots()
    labels, values = list(utilities_dict), list(utilities_dict.values())
    axis.bar(labels, values); axis.set_ylabel("PF utility"); fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


def plot_utility_vs_round(history_dict, save_path):
    # PSEUDOCODE (E4a): for each fl_mode in {local_only, fedavg, topology_kernel}:
    #     plt.plot([h[0] for h in history_dict[fl_mode]], [h[1] for h in history_dict[fl_mode]], label=fl_mode)
    # plt.xlabel('round'); plt.ylabel('PF utility')
    fig, axis = plt.subplots()
    for label, history in history_dict.items():
        axis.plot([row[0] for row in history], [row[1] for row in history], label=label)
    axis.set(xlabel="round", ylabel="PF utility"); axis.legend(); fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


def plot_utility_vs_bytes(history_dict, save_path):
    # PSEUDOCODE (E4b): same as above but x-axis = cumulative_bytes (history[i][2])
    # -- this is the plot that most directly supports Claim 3
    fig, axis = plt.subplots()
    for label, history in history_dict.items():
        axis.plot([row[2] for row in history], [row[1] for row in history], label=label)
    axis.set(xlabel="cumulative bytes", ylabel="PF utility"); axis.legend(); fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


def plot_kappa_ablation(kappa_values, final_utilities, save_path):
    # PSEUDOCODE (E5): x = kappa_ap_relay in [0, 0.3, 0.6, 1.0], y = final utility after fixed rounds
    fig, axis = plt.subplots()
    axis.plot(kappa_values, final_utilities, marker="o")
    axis.set(xlabel="kappa_AP-relay", ylabel="final PF utility"); fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


def plot_stage_b_vs_stage_c(history_b, history_c, save_path):
    # PSEUDOCODE (E6, stretch): overlay utility-vs-round for both training methods
    fig, axis = plt.subplots()
    axis.plot([row[0] for row in history_b], [row[1] for row in history_b], label="Stage B CTDE")
    axis.plot([row[0] for row in history_c], [row[1] for row in history_c], label="Stage C local-zeta")
    axis.set(xlabel="round", ylabel="PF utility"); axis.legend(); fig.tight_layout(); fig.savefig(save_path); plt.close(fig)


def per_user_rate_cdf(rates):
    values = np.sort(np.asarray(rates).ravel())
    return values, np.arange(1, len(values) + 1) / max(len(values), 1)
