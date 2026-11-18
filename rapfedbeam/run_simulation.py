"""Run the reduced RAP-FedBeam physical-layer simulation."""

from pathlib import Path
import argparse
import numpy as np
import yaml

from baselines.heuristics import heuristic_solution
from eval.metrics import evaluate_solution, pf_utility
from eval.plots import plot_policy_vs_heuristics_bar, plot_rate_cdf, plot_tau_sweep
from sim.problem import make_instance


def evaluate_drop(config, seed, drop_idx, use_relay=True, allocation="equal", tau=None):
    instance = make_instance(config, seed, drop_idx)
    solution = heuristic_solution(instance, allocation, use_relay, tau=tau)
    metrics = evaluate_solution(instance, solution)
    return metrics["rates"], metrics["debug"], instance.scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--drops", type=int, default=20)
    parser.add_argument("--plot-dir", default="plots")
    args = parser.parse_args()

    with Path(args.config).open() as config_file:
        config = yaml.safe_load(config_file)
    base_seed = int(config.get("seed", 42))
    epsilon_0 = config["objective"]["epsilon_0"]

    results = {}
    for label, use_relay, allocation in (
        ("relay + equal power", True, "equal"),
        ("no relay + equal power", False, "equal"),
        ("relay + fractional power", True, "fractional"),
    ):
        rates = []
        for drop_idx in range(args.drops):
            drop_rates, _, _ = evaluate_drop(config, base_seed, drop_idx, use_relay, allocation)
            rates.append(drop_rates)
        values = np.asarray(rates)
        results[label] = values
        print(f"{label}: mean={values.mean():.4f} bit/s/Hz, p5={np.percentile(values, 5):.4f}, PF={pf_utility(values, epsilon_0=epsilon_0).mean():.4f}")

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_rate_cdf(results, "Relay and power-policy comparison", plot_dir / "rate_cdf.png")
    policy_utilities = {label: float(pf_utility(values, epsilon_0=epsilon_0).mean()) for label, values in results.items()}
    plot_policy_vs_heuristics_bar(policy_utilities, plot_dir / "policy_pf_utility.png")

    print("\nTau sweep, relay + equal power:")
    tau_values = config["phase"]["tau_sweep"]
    tau_means, tau_p5 = [], []
    for tau in tau_values:
        rates = [evaluate_drop(config, base_seed, i, True, "equal", tau)[0] for i in range(args.drops)]
        values = np.asarray(rates)
        tau_means.append(float(values.mean()))
        tau_p5.append(float(np.percentile(values, 5)))
        print(f"tau={tau:.1f}: mean={values.mean():.4f}, p5={np.percentile(values, 5):.4f}")
    plot_tau_sweep(tau_values, tau_means, tau_p5, plot_dir / "tau_sweep.png")

    improvement = results["relay + equal power"].mean() - results["no relay + equal power"].mean()
    print(f"\nRelay mean-rate improvement: {improvement:.4f} bit/s/Hz")
    print(f"Plots saved to: {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
