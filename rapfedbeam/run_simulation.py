"""Run the reduced RAP-FedBeam physical-layer simulation."""

from pathlib import Path
import argparse
import numpy as np
import yaml

from baselines.heuristics import equal_power, fractional_power, no_relay_scenario, relay_power_equal_or_fractional
from eval.plots import plot_policy_vs_heuristics_bar, plot_rate_cdf, plot_tau_sweep
from sim.channels import sample_full_channel_set
from sim.rates import build_beams_from_power_coeffs, compute_full_rates, mmse_combiners, mrt_beam_directions
from sim.scenario import generate_drop


def evaluate_drop(config, rng, use_relay=True, allocation="equal", tau=None):
    scenario = generate_drop(config, rng)
    channels = sample_full_channel_set(scenario, config, rng)
    if not use_relay:
        scenario = no_relay_scenario(scenario)

    directions_ap = mrt_beam_directions(channels.h_hat)
    if allocation == "fractional":
        ap_coeffs = fractional_power(
            10.0 ** (scenario.beta_ap_ue_db / 10.0),
            scenario.cluster_mask,
            config["power"]["p_max_ap"],
        )
    else:
        ap_coeffs = equal_power(scenario.cluster_mask, config["power"]["p_max_ap"])
    beams_ap = build_beams_from_power_coeffs(directions_ap, ap_coeffs, config["power"]["p_max_ap"])

    relay_coeffs = relay_power_equal_or_fractional(
        scenario.relay_assoc,
        config["power"]["p_max_relay"],
        config["network"]["num_relays"],
        10.0 ** (scenario.beta_relay_ue_db / 10.0),
        fractional=allocation == "fractional",
    )
    if not use_relay:
        relay_coeffs.fill(0.0)
    relay_directions = np.ones(
        (config["network"]["num_relays"], config["network"]["num_ues"], config["network"]["n_antennas_relay"]),
        dtype=complex,
    )
    relay_directions /= np.sqrt(config["network"]["n_antennas_relay"])
    beams_relay = build_beams_from_power_coeffs(relay_directions, relay_coeffs, config["power"]["p_max_relay"])
    combiners = mmse_combiners(channels.G_mr, beams_ap)
    rates, debug = compute_full_rates(
        scenario,
        channels,
        beams_ap,
        beams_relay,
        combiners,
        tau if tau is not None else config["phase"]["tau_fixed"],
    )
    return rates, debug, scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--drops", type=int, default=20)
    parser.add_argument("--plot-dir", default="plots")
    args = parser.parse_args()

    with Path(args.config).open() as config_file:
        config = yaml.safe_load(config_file)
    base_seed = int(config.get("seed", 42))

    results = {}
    for label, use_relay, allocation in (
        ("relay + equal power", True, "equal"),
        ("no relay + equal power", False, "equal"),
        ("relay + fractional power", True, "fractional"),
    ):
        rates = []
        for drop_idx in range(args.drops):
            drop_rates, _, _ = evaluate_drop(config, np.random.default_rng(base_seed + drop_idx), use_relay, allocation)
            rates.append(drop_rates)
        values = np.asarray(rates)
        results[label] = values
        print(f"{label}: mean={values.mean():.4f} bit/s/Hz, p5={np.percentile(values, 5):.4f}, PF={np.log(values + 1e-6).sum(axis=1).mean():.4f}")

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_rate_cdf(results, "Relay and power-policy comparison", plot_dir / "rate_cdf.png")
    policy_utilities = {label: float(np.log(values + 1e-6).sum(axis=1).mean()) for label, values in results.items()}
    plot_policy_vs_heuristics_bar(policy_utilities, plot_dir / "policy_pf_utility.png")

    print("\nTau sweep, relay + equal power:")
    tau_values = config["phase"]["tau_sweep"]
    tau_means, tau_p5 = [], []
    for tau in tau_values:
        rates = [evaluate_drop(config, np.random.default_rng(base_seed + i), True, "equal", tau)[0] for i in range(args.drops)]
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
