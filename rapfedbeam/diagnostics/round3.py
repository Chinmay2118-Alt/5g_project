"""
diagnostics/round3.py
Responsibility: the extra diagnostics asked for by Google Deep Review round 3, on the same seed,
drops and deployments as the final run, so every number pairs with results/*.json.

  untrained   no training: the slow timescale split into tau alone vs. tau + priced association
              (default network); untrained layers with Steps 1-2 at K = 12 with L = 6 and 100;
              untrained layers with the warm-up on the small network (tau and association fixed);
              untrained layers with Steps 1-2 at every aging coefficient
  exact       the SCIP bound on the small network with the association fixed to Step 1's
  trained     RAP-FedWMMSE trained on the default network (a) as in the final run, with the
              predicted layer parameters logged; (b) without the QoS machinery (pure PF: no outage
              term, no prices); (c) without the outage term only, evaluated zero-shot at every
              aging coefficient

Usage, from rapfedbeam/:  python -m diagnostics.round3 all --workers 11
Results: results/diagnostics/round3_<name>.json
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import torch

from baselines.exact_minlp import small_instance_config
from diagnostics.round2 import DEPLOYMENTS, DROPS, RHOS, SEED, _config, _drops_of, _ProbeTrainer, _save
from experiments.definitions import SAME_PROBLEM
from experiments.runner import _score, apply_overrides, run_point

PURE_PF = {"fedwmmse.outage_weight": 0.0, "fedwmmse.eta_q": 0.0, "fedwmmse.eta_d": 0.0,
           "fedwmmse.eta_q_epoch": 0.0, "fedwmmse.eta_d_epoch": 0.0}
NO_OUTAGE = {"fedwmmse.outage_weight": 0.0}


def untrained(config, workers):
    payload = {}
    payload["default"], _, _ = run_point(config, ["unfolded_wmmse_tau_only", "unfolded_wmmse_slow"], DROPS, SEED, workers, DEPLOYMENTS)
    for layers in (6, 100):
        cfg = apply_overrides(config, {"network.num_ues": 12, "fedwmmse.layers": layers})
        payload[f"K12_L{layers}"], _, _ = run_point(cfg, ["unfolded_wmmse_slow"], DROPS, SEED, workers, DEPLOYMENTS)
    small = apply_overrides(small_instance_config(config), SAME_PROBLEM)
    payload["small"], _, _ = run_point(small, ["unfolded_wmmse_slow"], DROPS, SEED, workers, DEPLOYMENTS)
    _save("round3_untrained", payload)
    aging = {}
    for rho in RHOS:
        aged = apply_overrides(config, {"aging.enabled": True, "aging.rho_h": rho, "aging.rho_G": rho, "aging.rho_g": rho})
        results, _, _ = run_point(aged, ["unfolded_wmmse_slow"], DROPS, SEED, workers, DEPLOYMENTS)
        aging[str(rho)] = results["unfolded_wmmse_slow"]
    _save("round3_aging_untrained", aging)


def exact(config, workers):
    small = apply_overrides(small_instance_config(config), {**SAME_PROBLEM, "exact.optimize_association": False})
    payload, _, _ = run_point(small, ["exact_minlp"], DROPS, SEED, workers, DEPLOYMENTS)
    _save("round3_exact_fixed_association", payload)


def _trained_task(args):
    config, deployment, variant = args
    torch.set_num_threads(1)
    cfg = config if variant == "default" else apply_overrides(config, PURE_PF if variant == "pure_pf" else NO_OUTAGE)
    trainer = _ProbeTrainer(cfg, SEED, deployment, "rap")
    trainer.run()
    rows = []
    for d in _drops_of(deployment):
        row = _score(trainer, cfg, SEED, d, deployment)
        row["probe"] = trainer.probe
        rows.append(row)
    out = {variant: rows}
    if variant == "no_outage":
        for rho in RHOS:
            aged = apply_overrides(cfg, {"aging.enabled": True, "aging.rho_h": rho, "aging.rho_G": rho, "aging.rho_g": rho})
            out[f"no_outage_rho_{rho}"] = [_score(trainer, aged, SEED, d, deployment) for d in _drops_of(deployment)]
    return out


def trained(config, workers):
    tasks = [(config, dep, variant) for variant in ("default", "pure_pf", "no_outage") for dep in range(DEPLOYMENTS)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_trained_task, tasks))
    payload = {}
    for part in parts:
        for key, rows in part.items():
            payload.setdefault(key, []).extend(rows)
    for key in payload:
        payload[key].sort(key=lambda r: r["drop"])
    _save("round3_trained", payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["untrained", "exact", "trained", "all"])
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    config = _config()
    for name, run in (("untrained", untrained), ("exact", exact), ("trained", trained)):
        if args.which in (name, "all"):
            print(f"{name}...", flush=True)
            run(config, args.workers)


if __name__ == "__main__":
    main()
