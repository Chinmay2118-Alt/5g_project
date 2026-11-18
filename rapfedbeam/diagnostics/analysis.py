"""
diagnostics/analysis.py
Responsibility: recompute every paired statistic quoted in the paper from the saved results, so each
reported number can be traced to a command. Paired differences are over the same drops; the
interval is mean +- 1.96 * std / sqrt(n).

  main        Table II/III numbers: the gain decomposition, margins over the baselines (also at the
              best fixed tau), aging tail and outage vs. the no-relay benchmark, learning curves
  round2      diagnostics/round2.py: damped/undamped layers, L = 100, slow timescale on the untrained
              layers, zero-shot aging, small-network starvation
  round3      diagnostics/round3.py: tau-only split, K = 12 depth, small network with warm-up, SCIP with
              the association fixed, default-network learned parameters, pure-PF and no-outage training,
              untrained layers under aging
  runtime     diagnostics/runtime.py: the clean timing of Table II

Usage, from rapfedbeam/ after the runs:  python -m diagnostics.analysis all
Needs: results/*.json (run_experiments.py) and results/diagnostics/*.json (the diagnostics).
"""

import argparse
import json
from pathlib import Path

import numpy as np

RES = Path("results")
DIAG = RES / "diagnostics"
STARVED = 0.02


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def point(data, x):
    return next(p["results"] for p in data["points"] if p["x"] == x)


def by_drop(records, key="pf_utility"):
    return {r["drop"]: r[key] for r in records if "error" not in r}


def paired(a, b, label):
    common = sorted(set(a) & set(b))
    diff = np.array([a[d] - b[d] for d in common])
    half = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"  {label}: {diff.mean():+.3f} +- {half:.3f} (n={len(diff)}, wins {int(np.sum(diff > 0))})")


def summary(records, label):
    ok = [r for r in records if "error" not in r]
    pf = np.array([r["pf_utility"] for r in ok])
    rates = np.concatenate([r["rates"] for r in ok])
    starved = sum(min(r["rates"]) < STARVED for r in ok)
    print(f"  {label}: PF {pf.mean():.3f} +- {1.96 * pf.std(ddof=1) / np.sqrt(len(pf)):.3f}, "
          f"outage@0.5 {np.mean(rates < 0.5) * 100:.1f}%, p5 {np.percentile(rates, 5):.3f}, "
          f"starved {starved}, errors {len(records) - len(ok)}")


def same(a, b):
    keys = sorted(by_drop(a))
    return bool(np.allclose([by_drop(a)[k] for k in keys], [by_drop(b)[k] for k in keys]))


def main_results():
    base = point(load(RES / "baseline.json"), "default")
    layers6 = point(load(RES / "layers.json"), 6)
    tau = load(RES / "tau.json")
    print("== Table II ==")
    for name in ("centralized_wmmse", "centralized_no_relay", "rap_fedwmmse", "unfolded_wmmse", "local_wmmse", "rzf_equal", "mrt_equal"):
        summary(base[name], name)
    print("== Table III, order A, and margins (paired) ==")
    paired(by_drop(base["unfolded_wmmse"]), by_drop(base["local_wmmse"]), "coordination: untrained - Local")
    paired(by_drop(layers6["rap_fedwmmse"]), by_drop(layers6["unfolded_wmmse"]), "learning at tau 0.5, L = 6")
    paired(by_drop(base["rap_fedwmmse"]), by_drop(point(tau, 0.5)["rap_fedwmmse"]), "slow timescale: RAP - RAP at tau 0.5")
    paired(by_drop(base["rap_fedwmmse"]), by_drop(base["local_wmmse"]), "total: RAP - Local")
    paired(by_drop(base["rap_fedwmmse"]), by_drop(base["rzf_equal"]), "RAP - RZF")
    paired(by_drop(base["rap_fedwmmse"]), by_drop(base["unfolded_wmmse"]), "RAP - untrained")
    for name in ("local_wmmse", "rzf_equal"):
        paired(by_drop(base["rap_fedwmmse"]), by_drop(point(tau, 0.8)[name]), f"RAP - {name} at its best fixed tau 0.8")
    print("  RAP median adapted tau:", np.median([r["info"]["tau"] for r in base["rap_fedwmmse"]]))
    print("== Aging (G08): tail and outage vs. the no-relay benchmark ==")
    for p in load(RES / "aging.json")["points"]:
        r = p["results"]
        paired(by_drop(r["rap_fedwmmse"]), by_drop(r["centralized_wmmse"]), f"rho {p['x']}: RAP - centralized")
        paired(by_drop(r["rap_fedwmmse"]), by_drop(r["centralized_no_relay"]), f"rho {p['x']}: RAP - no relay")
        for name in ("rap_fedwmmse", "centralized_no_relay"):
            rates = np.concatenate([x["rates"] for x in r[name]])
            print(f"    {name}: p5 {np.percentile(rates, 5):.2f}, outage {np.mean(rates < 0.5) * 100:.1f}%")
    print("== Federated learning (G12) ==")
    history = load(RES / "baseline.json")["points"][0]["fit_history"]
    for name in ("rap_fedwmmse", "rap_fedwmmse_fedavg", "rap_fedwmmse_nofl"):
        runs = [h["history"] for h in history[name]]
        print(f"  {name}: epoch-1 PF {np.mean([h[0]['pf_utility'] for h in runs]):.2f}, "
              f"epoch-1 tau {sorted(set(round(h[0]['tau'], 2) for h in runs))}, "
              f"final MB {np.mean([h[-1]['cumulative_bytes'] for h in runs]) / 1e6:.2f}")
    paired(by_drop(base["rap_fedwmmse"]), by_drop(base["rap_fedwmmse_nofl"]), "personalized FL - no FL")


def round2():
    base = point(load(RES / "baseline.json"), "default")
    abl = load(DIAG / "ablations.json")
    l6, l100 = abl["L6"], abl["L100"]
    print("== Round 2: ablations ==")
    print("  determinism, untrained L6 == baseline:", same(l6["unfolded_wmmse"], base["unfolded_wmmse"]))
    for key, results in (("L6", l6), ("L100", l100)):
        for name, records in results.items():
            summary(records, f"{key} {name}")
    paired(by_drop(l6["unfolded_wmmse_slow"]), by_drop(l6["unfolded_wmmse"]), "order B slow: untrained+slow - untrained")
    paired(by_drop(base["rap_fedwmmse"]), by_drop(l6["unfolded_wmmse_slow"]), "order B learning: RAP - untrained+slow")
    paired(by_drop(l6["classical_wmmse_layers"]), by_drop(l6["unfolded_wmmse"]), "undamped - damped, L6")
    layers6 = point(load(RES / "layers.json"), 6)
    paired(by_drop(layers6["rap_fedwmmse"]), by_drop(l100["unfolded_wmmse"]), "trained L6 - damped untrained L100 (tau 0.5)")
    paired(by_drop(point(load(RES / "tau.json"), 0.5)["centralized_wmmse"]), by_drop(l100["unfolded_wmmse"]), "centralized - damped L100 (tau 0.5)")
    print("== Round 2: zero-shot aging ==")
    zero = load(DIAG / "aging_zero_shot.json")
    print("  determinism, default re-score == baseline RAP:", same(zero["default"], base["rap_fedwmmse"]))
    aging = load(RES / "aging.json")
    for rho in ("0.8", "0.85", "0.9", "0.95", "0.99"):
        p = point(aging, float(rho))
        paired(by_drop(zero[rho]), by_drop(p["rap_fedwmmse"]), f"rho {rho}: zero-shot - retrained")
        paired(by_drop(zero[rho]), by_drop(p["centralized_wmmse"]), f"rho {rho}: zero-shot - centralized")
    print("== Round 2: small-network starvation ==")
    starvation = load(DIAG / "starvation.json")
    gap = point(load(RES / "optimality_gap.json"), "small")
    print("  determinism, trained == G10 RAP:", same(starvation["trained"], gap["rap_fedwmmse"]))
    bound = {r["drop"]: r["info"]["upper_bound"] for r in gap["exact_minlp"] if "error" not in r}
    for key, records in starvation.items():
        starved = sum(min(r["rates"]) < STARVED for r in records)
        print(f"  {key}: starved {starved}, mean gap {np.mean([bound[r['drop']] - r['pf_utility'] for r in records]):.3f}")
    for name in ("unfolded_wmmse", "centralized_wmmse"):
        print(f"  {name}: starved {sum(min(r['rates']) < STARVED for r in gap[name])}")
    trained = starvation["trained"]
    hit = [r for r in trained if min(r["rates"]) < STARVED]
    rest = [r for r in trained if min(r["rates"]) >= STARVED]
    for label, group in (("starved", hit), ("other", rest)):
        print(f"  {label} ({len(group)}): step {np.mean([np.mean(r['probe']['ap_step']) for r in group]):.3f}, "
              f"relaxation {np.mean([np.mean(r['probe']['ap_relaxation']) for r in group]):.3f}, "
              f"lambda_reg {np.mean([np.mean(r['probe']['ap_reg']) for r in group]):.3f}")


def round3():
    base = point(load(RES / "baseline.json"), "default")
    u = load(DIAG / "round3_untrained.json")
    r2 = load(DIAG / "ablations.json")["L6"]
    print("== Round 3: slow timescale split ==")
    print("  determinism, untrained+slow == round 2:", same(u["default"]["unfolded_wmmse_slow"], r2["unfolded_wmmse_slow"]))
    summary(u["default"]["unfolded_wmmse_tau_only"], "untrained + tau only")
    summary(u["default"]["unfolded_wmmse_slow"], "untrained + full slow")
    paired(by_drop(u["default"]["unfolded_wmmse_tau_only"]), by_drop(base["unfolded_wmmse"]), "tau only - untrained")
    paired(by_drop(u["default"]["unfolded_wmmse_slow"]), by_drop(u["default"]["unfolded_wmmse_tau_only"]), "priced assoc + prices + warm-up")
    print("== Round 3: K = 12 depth ==")
    k12 = point(load(RES / "num_ues.json"), 12)
    for key in ("K12_L6", "K12_L100"):
        summary(u[key]["unfolded_wmmse_slow"], f"untrained+slow {key}")
    paired(by_drop(u["K12_L100"]["unfolded_wmmse_slow"]), by_drop(k12["rap_fedwmmse"]), "untrained+slow L100 - RAP, K12")
    paired(by_drop(k12["centralized_wmmse"]), by_drop(u["K12_L100"]["unfolded_wmmse_slow"]), "centralized - untrained+slow L100, K12")
    print("== Round 3: small network ==")
    gap = point(load(RES / "optimality_gap.json"), "small")
    print(f"  untrained + warm-up: starved {sum(min(r['rates']) < STARVED for r in u['small']['unfolded_wmmse_slow'])}")
    joint = {r["drop"]: r["info"]["upper_bound"] for r in gap["exact_minlp"] if "error" not in r}
    fixed_records = load(DIAG / "round3_exact_fixed_association.json")["exact_minlp"]
    fixed = {r["drop"]: r["info"]["upper_bound"] for r in fixed_records if "error" not in r}
    statuses = [r["info"].get("status") for r in fixed_records if "error" not in r]
    print("  fixed-association SCIP statuses:", {s: statuses.count(s) for s in set(statuses)})
    cost = np.array([joint[d] - fixed[d] for d in fixed])
    print(f"  association cost: mean {cost.mean():.3f}, max {cost.max():.3f}, > 0.01 on {int(np.sum(cost > 0.01))} drops")
    for name in ("centralized_wmmse", "unfolded_wmmse", "rap_fedwmmse", "local_wmmse", "mrt_equal"):
        g = np.array([fixed[r["drop"]] - r["pf_utility"] for r in gap[name] if r["drop"] in fixed])
        print(f"  gap to fixed-association bound, {name}: {g.mean():.3f} (exceeds it on {int(np.sum(g < -1e-3))})")
    print("== Round 3: trained variants ==")
    t = load(DIAG / "round3_trained.json")
    print("  determinism, default == baseline RAP:", same(t["default"], base["rap_fedwmmse"]))
    probe = t["default"]
    steps = np.array([r["probe"]["ap_step"] for r in probe])
    relax = np.array([np.mean(r["probe"]["ap_relaxation"]) for r in probe])
    print(f"  learned (default network): AP step {steps.mean():.3f} (per layer {np.round(steps.mean(axis=(0, 1)), 3)}), "
          f"relaxation {relax.mean():.3f}, effective {np.mean(steps.mean(axis=(1, 2)) * relax):.3f}, "
          f"lambda_reg {np.mean([np.mean(r['probe']['ap_reg']) for r in probe]):.3f}")
    summary(t["pure_pf"], "pure PF")
    summary(t["no_outage"], "no outage term")
    paired(by_drop(t["pure_pf"]), by_drop(base["rap_fedwmmse"]), "pure PF - RAP")
    paired(by_drop(t["no_outage"]), by_drop(base["rap_fedwmmse"]), "no outage - RAP")
    print("== Round 3: where the stale-CSI advantage comes from ==")
    zero = load(DIAG / "aging_zero_shot.json")
    untrained_aging = load(DIAG / "round3_aging_untrained.json")
    aging = load(RES / "aging.json")
    for rho in ("0.8", "0.85", "0.9", "0.95", "0.99"):
        p = point(aging, float(rho))
        paired(by_drop(t[f"no_outage_rho_{rho}"]), by_drop(zero[rho]), f"rho {rho}: no-outage - with-outage (zero-shot)")
        paired(by_drop(untrained_aging[rho]), by_drop(p["rap_fedwmmse"]), f"rho {rho}: untrained+slow - RAP")
        paired(by_drop(untrained_aging[rho]), by_drop(p["centralized_wmmse"]), f"rho {rho}: untrained+slow - centralized")


def runtime():
    data = load(DIAG / "runtime.json")
    print(f"== Clean timing (drops {data['drops']}) ==")
    for name, seconds in data["seconds"].items():
        print(f"  {name}: mean {np.mean(seconds):.3f} s, median {np.median(seconds):.3f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["main", "round2", "round3", "runtime", "all"])
    args = parser.parse_args()
    for name, run in (("main", main_results), ("round2", round2), ("round3", round3), ("runtime", runtime)):
        if args.which in (name, "all"):
            run()


if __name__ == "__main__":
    main()
