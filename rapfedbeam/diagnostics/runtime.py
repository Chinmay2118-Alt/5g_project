"""
diagnostics/runtime.py
Responsibility: one clean timing run for Table II (Google Deep Review round 2). Every method solves
the same drops one after another in a single process with one thread, on an otherwise idle
machine, so all times come from the same conditions. RAP-FedWMMSE's per-drop time is split into
its warm-up (3 slow-timescale epochs) and the single-block inference that follows.

Usage, from rapfedbeam/:  python -m diagnostics.runtime
Result: results/diagnostics/runtime.json
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.algorithms import get_algorithm
from experiments.runner import apply_overrides, deployment_of
from sim.problem import make_instance
from train.rap_fedwmmse import RapFedWMMSETrainer

SEED, DROPS, DEPLOYMENTS, DEPLOYMENT = 42, 100, 10, 0


def _time(solve, instance, repeats=1):
    started = time.perf_counter()
    for _ in range(repeats):
        solve(instance)
    return (time.perf_counter() - started) / repeats


def main():
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    with open("configs/config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    drops = [d for d in range(DROPS) if deployment_of(d, DEPLOYMENTS) == DEPLOYMENT]
    instances = [make_instance(config, SEED, d, DEPLOYMENT) for d in drops]
    out = {"drops": drops, "seconds": {}}

    started = time.perf_counter()
    trainer = RapFedWMMSETrainer(config, SEED, DEPLOYMENT, "rap")
    trainer.run()
    out["rap_training_seconds"] = time.perf_counter() - started
    print(f"trained in {out['rap_training_seconds']:.0f} s", flush=True)

    full = [_time(trainer.solve, i) for i in instances]
    warmup_epochs = trainer.settings["warmup_epochs"]
    trainer.settings["warmup_epochs"] = 0
    single = [_time(trainer.solve, i) for i in instances]
    trainer.settings["warmup_epochs"] = warmup_epochs
    out["seconds"]["rap_fedwmmse"] = full
    out["seconds"]["rap_fedwmmse_single_block"] = single

    names = ["unfolded_wmmse", "classical_wmmse_layers", "centralized_wmmse", "centralized_no_relay",
             "local_wmmse", "rzf_equal", "mrt_equal"]
    for name in names:
        algorithm = get_algorithm(name)
        out["seconds"][name] = [_time(algorithm.solve, i) for i in instances]
        print(f"{name}: {np.mean(out['seconds'][name]):.3f} s", flush=True)
    deep = apply_overrides(config, {"fedwmmse.layers": 100})
    deep_instances = [make_instance(deep, SEED, d, DEPLOYMENT) for d in drops]
    out["seconds"]["classical_wmmse_layers_L100"] = [_time(get_algorithm("classical_wmmse_layers").solve, i) for i in deep_instances]

    Path("results/diagnostics").mkdir(parents=True, exist_ok=True)
    with open("results/diagnostics/runtime.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle)
    for name, values in out["seconds"].items():
        print(f"{name}: mean {np.mean(values):.3f} s, median {np.median(values):.3f} s")


if __name__ == "__main__":
    main()
