"""
diagnostics/round2.py
Responsibility: the extra diagnostics asked for by Google Deep Review round 2. They use the same
seed, drops and deployments as the final run, so every number pairs with results/*.json.

  ablations   untrained (relaxation 0.5) and undamped classical (relaxation 1) layers, at fixed
              tau = 0.5 with L = 6 and L = 100, and at L = 6 with the training-free slow
              timescale (Steps 1-2 and the warm-up)
  aging       RAP-FedWMMSE trained once on the default network (robust rho = 0.98) and evaluated
              zero-shot at every aging coefficient, against the per-rho retrained models of G08
  starvation  small exact-method network: what the trained model predicts on the starved drops,
              two inference-time checks (uniform initial power shares; step size capped at 1), and
              one retrained check without the QoS machinery (no outage loss, no prices)

Each also re-scores a configuration from the final run, as a determinism check.

Usage, from rapfedbeam/:  python -m diagnostics.round2 all --workers 11
Results: results/diagnostics/<name>.json
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.definitions import SAME_PROBLEM
from experiments.runner import _score, apply_overrides, deployment_of, run_point
from baselines.exact_minlp import small_instance_config
from train.rap_fedwmmse import RapFedWMMSETrainer, ap_features, relay_features
from train.unfolded_wmmse import LayerParams, RDTYPE, initial_beams, masked_softmax, unfold

SEED, DROPS, DEPLOYMENTS = 42, 100, 10
RHOS = (0.8, 0.85, 0.9, 0.95, 0.99)
STARVED = 0.02   # bit/s/Hz: a UE below this is counted as (nearly) starved
OUT = Path("results/diagnostics")


def _config():
    with open("configs/config.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _drops_of(deployment):
    return [d for d in range(DROPS) if deployment_of(d, DEPLOYMENTS) == deployment]


def _save(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"saved {OUT / name}.json", flush=True)


# --- ablations: no training, so run_point does everything ---

def ablations(config, workers):
    payload = {}
    for layers in (6, 100):
        cfg = apply_overrides(config, {"fedwmmse.layers": layers})
        names = ["unfolded_wmmse", "classical_wmmse_layers"]
        if layers == 6:
            names += ["unfolded_wmmse_slow", "classical_wmmse_slow"]
        results, _, _ = run_point(cfg, names, DROPS, SEED, workers, DEPLOYMENTS)
        payload[f"L{layers}"] = results
        print(f"  ablations L={layers} done", flush=True)
    _save("ablations", payload)


# --- aging: train once on the default network, evaluate at every rho ---

def _aging_task(args):
    config, deployment = args
    torch.set_num_threads(1)
    trainer = RapFedWMMSETrainer(config, SEED, deployment, "rap")
    trainer.run()
    records = {"default": [_score(trainer, config, SEED, d, deployment) for d in _drops_of(deployment)]}
    for rho in RHOS:
        cfg = apply_overrides(config, {"aging.enabled": True, "aging.rho_h": rho, "aging.rho_G": rho, "aging.rho_g": rho})
        records[str(rho)] = [_score(trainer, cfg, SEED, d, deployment) for d in _drops_of(deployment)]
    return records


def aging(config, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_aging_task, [(config, dep) for dep in range(DEPLOYMENTS)]))
    payload = {key: sorted((r for part in parts for r in part[key]), key=lambda r: r["drop"]) for key in parts[0]}
    _save("aging_zero_shot", payload)


# --- starvation on the small network ---

class _ProbeTrainer(RapFedWMMSETrainer):
    """RAP-FedWMMSE with an inference-time switch: 'uniform' sets every initial power share equal,
    'step_cap' caps the predicted step size at 1. It records what the head predicted for the last
    block it solved. Training is unchanged (mode None)."""

    mode = None

    def policy(self, net, coordination, layers=None):
        layers = self.settings["layers"] if layers is None else layers
        prices = torch.as_tensor(coordination.qos, dtype=RDTYPE).expand(net.batch, -1)
        zeta = torch.as_tensor(coordination.vector(), dtype=RDTYPE).expand(net.batch, -1)
        ap_out = [self.nodes[f"AP{m}"](ap_features(net, m, zeta)) for m in range(self.M)]
        relay_out = [self.nodes[f"R{r}"](relay_features(net, r, zeta)) for r in range(self.R)]
        stacked = [torch.stack([o[i] for o in ap_out], dim=1) for i in range(4)]
        relay_stacked = [torch.stack([o[i] for o in relay_out], dim=1) for i in range(4)]
        if self.mode == "step_cap":
            stacked[0], relay_stacked[0] = torch.clamp(stacked[0], max=1.0), torch.clamp(relay_stacked[0], max=1.0)
        if self.mode == "uniform":
            stacked[3], relay_stacked[3] = torch.zeros_like(stacked[3]), torch.zeros_like(relay_stacked[3])
        params = LayerParams(*stacked[:3], *relay_stacked[:3])
        W0, V0 = initial_beams(net, stacked[3], relay_stacked[3])
        self.probe = {
            "ap_shares": masked_softmax(stacked[3], net.cluster[None])[-1].tolist(),
            "relay_shares": masked_softmax(relay_stacked[3], net.assoc[None])[-1].tolist(),
            "ap_step": stacked[0][-1].tolist(), "ap_relaxation": stacked[1][-1].tolist(), "ap_reg": stacked[2][-1].tolist(),
            "relay_step": relay_stacked[0][-1].tolist(), "qos_prices": coordination.qos.tolist(),
        }
        duals = (prices.clone(), torch.as_tensor(coordination.df, dtype=RDTYPE).expand(net.batch, -1, -1).clone())
        return unfold(net, W0, V0, params, duals, layers, self.settings["softmin_kappa"], self.settings["eta_q"], self.settings["eta_d"])


def _starvation_task(args):
    config, deployment, retrain_without_qos = args
    torch.set_num_threads(1)
    cfg = config
    if retrain_without_qos:
        cfg = apply_overrides(config, {"fedwmmse.outage_weight": 0.0, "fedwmmse.eta_q": 0.0, "fedwmmse.eta_d": 0.0,
                                       "fedwmmse.eta_q_epoch": 0.0, "fedwmmse.eta_d_epoch": 0.0})
    trainer = _ProbeTrainer(cfg, SEED, deployment, "rap")
    trainer.run()
    modes = [None] if retrain_without_qos else [None, "uniform", "step_cap"]
    records = {}
    for mode in modes:
        trainer.mode = mode
        key = "no_qos" if retrain_without_qos else (mode or "trained")
        rows = []
        for d in _drops_of(deployment):
            row = _score(trainer, cfg, SEED, d, deployment)
            row["probe"] = trainer.probe
            rows.append(row)
        records[key] = rows
    return records


def starvation(config, workers):
    small = apply_overrides(small_instance_config(config), SAME_PROBLEM)
    tasks = [(small, dep, flag) for flag in (False, True) for dep in range(DEPLOYMENTS)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_starvation_task, tasks))
    payload = {}
    for part in parts:
        for key, rows in part.items():
            payload.setdefault(key, []).extend(rows)
    for key in payload:
        payload[key].sort(key=lambda r: r["drop"])
    _save("starvation", payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["ablations", "aging", "starvation", "all"])
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    config = _config()
    for name, run in (("ablations", ablations), ("aging", aging), ("starvation", starvation)):
        if args.which in (name, "all"):
            print(f"{name}...", flush=True)
            run(config, args.workers)


if __name__ == "__main__":
    main()
