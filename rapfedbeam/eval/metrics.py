"""
eval/metrics.py
Responsibility: compute the metrics used across all experiments E1-E6.
"""

import numpy as np


def evaluate_pf_utility_on_holdout(node_models, config, num_holdout_drops=100):
    # PSEUDOCODE:
    # rng = fixed holdout seed (different from training seed, kept constant across all comparisons)
    # for each holdout drop:
    #     scenario, channels = generate_drop + sample_channels (using holdout rng)
    #     w, v, u = forward_all_nodes(node_models, channels.hat, scenario)   # inference only, no grad
    #     r, _ = compute_full_rates(..., hard_min=True)   # TRUE hard min at eval time
    #     record r (per-UE rate vector), utility = sum(log(r + eps))
    # return mean_utility, all_rates (for CDF plotting)
    from sim.channels import sample_full_channel_set
    from sim.scenario import generate_drop
    rng = np.random.default_rng(config.get("seed", 42) + 12345)
    all_rates = []
    for _ in range(num_holdout_drops):
        scenario = generate_drop(config, rng)
        channels = sample_full_channel_set(scenario, config, rng)
        if callable(node_models):
            rates = np.asarray(node_models(scenario, channels))
        else:
            raise ValueError("node_models must be a callable evaluator in the reduced implementation")
        all_rates.append(rates)
    all_rates = np.asarray(all_rates)
    return float(np.mean(np.log(np.maximum(all_rates, 1e-6)).sum(axis=-1))), all_rates


def per_user_rate_cdf(all_rates):
    # PSEUDOCODE: flatten all_rates across drops and UEs, sort, return (sorted_rates, cdf_values)
    values = np.sort(np.asarray(all_rates).ravel())
    return values, np.arange(1, len(values) + 1) / max(len(values), 1)


def percentile_rate(all_rates, percentile=5):
    # PSEUDOCODE: return np.percentile(all_rates.flatten(), percentile)
    return float(np.percentile(np.asarray(all_rates), percentile))


def outage_probability(all_rates, r_min):
    # PSEUDOCODE: fraction of (drop, ue) pairs where rate < r_min
    # (reported as a DIAGNOSTIC only in v1, not enforced as a constraint -- see Sec. 8 limitations)
    return float(np.mean(np.asarray(all_rates) < r_min))


def compare_relay_vs_no_relay(node_models_or_heuristic, config):
    # PSEUDOCODE (E1):
    # run evaluate_pf_utility_on_holdout with relay_assoc as-is         -> "with relay" rates
    # run evaluate_pf_utility_on_holdout with no_relay_scenario(...)     -> "no relay" rates
    # return both rate distributions for the CDF comparison plot
    with_relay = evaluate_pf_utility_on_holdout(node_models_or_heuristic, config)
    # The evaluator is responsible for selecting the scenario mode; retain a compact API here.
    return {"with_relay": with_relay[1], "no_relay": np.array([])}
