"""Run the comparison sweeps and draw the planned graphs."""

from pathlib import Path
import argparse
import os

# Parallelism comes from one worker process per drop; threads inside each would only compete.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import yaml

from eval.figures import draw_all
from experiments.algorithms import available_algorithms
from experiments.definitions import DEFAULT_ALGORITHMS, EXPERIMENTS
from experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS), help=f"any of: {', '.join(EXPERIMENTS)}")
    parser.add_argument("--algorithms", nargs="*", default=None, help="replaces each experiment's own list")
    parser.add_argument("--drops", type=int, default=50)
    parser.add_argument("--deployments", type=int, default=4, help="AP/relay layouts the drops are spread over; learning methods train once per layout")
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 1))
    parser.add_argument("--out", default="results")
    parser.add_argument("--plot-only", action="store_true", help="redraw graphs from saved results")
    args = parser.parse_args()

    with Path(args.config).open() as config_file:
        config = yaml.safe_load(config_file)

    if not args.plot_only:
        available = available_algorithms()
        for name in args.experiments:
            experiment = EXPERIMENTS[name]
            wanted = args.algorithms or experiment.get("algorithms", DEFAULT_ALGORITHMS)
            missing = [algorithm for algorithm in wanted if algorithm not in available]
            if missing:
                print(f"{name}: skipping unavailable algorithm(s): {', '.join(missing)}")
            selected = [algorithm for algorithm in wanted if algorithm in available]
            print(f"{name}: {len(selected)} algorithms, {args.drops} drops over {args.deployments} deployments", flush=True)
            run_experiment(experiment, config, selected, args.drops, int(config.get("seed", 42)), args.workers, args.out, args.deployments)

    drawn, skipped = draw_all(args.out, Path(args.out) / "figures")
    print(f"Drew: {', '.join(drawn) or 'none'}")
    print(f"Not drawn (no results yet): {', '.join(skipped) or 'none'}")


if __name__ == "__main__":
    main()
