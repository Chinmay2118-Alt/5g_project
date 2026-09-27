"""Run a compact federated-learning experiment."""

from pathlib import Path
import argparse
import yaml

from train.federated_loop import run_federated_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-steps", type=int, default=None)
    parser.add_argument("--small", action="store_true", help="Use a small network for a quick smoke test")
    args = parser.parse_args()

    with Path(args.config).open() as config_file:
        config = yaml.safe_load(config_file)
        
    if args.rounds is not None:
        config["fl"]["rounds"] = args.rounds
    if args.local_steps is not None:
        config["fl"]["local_steps_per_round"] = args.local_steps

    if args.small:
        config["network"].update(num_aps=2, num_relays=1, num_ues=3)

    print(f"Running {config['fl']['rounds']} FL rounds with {config['fl']['local_steps_per_round']} local step(s) per round")
    for mode in ("local_only", "fedavg", "topology_kernel"):
        history, _ = run_federated_experiment(config, mode)
        first = history[0]
        last = history[-1]
        print(
            f"{mode}: utility {first[1]:.4f} -> {last[1]:.4f}, "
            f"communication={last[2]} bytes"
        )


if __name__ == "__main__":
    main()
