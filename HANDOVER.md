# RAP-FedWMMSE handover

Status as of 10 November 2026, for Group 12 and anyone picking up this work.

## Where things stand

- **Complete.** Everything is committed on branch `kushagra`, with `main` merged in:
  - Sir's algorithm, RAP-FedWMMSE (Steps 1–15 of his document);
  - every baseline he asked for, and the exact MINLP;
  - the 14 figures, in `rapfedbeam/figures/`.
- **Final results:** 100 drops over 10 deployments, 12 experiments, 0 failed drops. 85 tests pass.
- **Not yet done:**
  1. Push the branch and open a pull request into `main`.
  2. Send [`rapfedbeam/docs/REPORT_FOR_SIR.md`](rapfedbeam/docs/REPORT_FOR_SIR.md) to Sir. It lists six
     interpretation points he should confirm, and the results that do not support our hypotheses.

## Background

- **The paper:** RAP-FedBeam, on personalized federated beam optimization for relay-assisted cell-free massive MIMO.
- **Sir's direction, 15 Oct:** a model-driven personalized federated algorithm based on WMMSE, with no dataset.
  - Baselines: Centralized WMMSE-SCA/AO, Local-WMMSE / No-FL, and an exact method (MINLP by spatial
    branch-and-bound, small networks only).
  - Compare 5–6 algorithms across 12–14 graphs.
- **Sir's main algorithm:** RAP-FedWMMSE (`5g_group_12_solution.pdf`).

## What the final session changed

**Fixes**
- **No-relay baseline:** it now uses τ = 1, since a network without relays has no Phase II. Before, it was
  held to τ ≤ 0.8, which flattered the relays.
- **Exact method:** it now optimizes the binary association too, so it solves a true MINLP.
- **Crash fix:** a numerically singular beam update crashed training on some drops. It is now fixed.
- **Training data:** training now uses a fresh UE drop each epoch. With a single moving drop, the model overfit it.

**Sir's instructions now covered in full**
- **Step 3:** each node's input carries all of ζ(t): QoS prices, DF prices, measured interference, rate statistics.
- **L_max > 1:** cooperative multi-relay forwarding works everywhere: rates, association, both WMMSE baselines,
  the exact MINLP and RAP-FedWMMSE.
- **Energy:** the λ_E·P_tot term is exercised in its own sweep (G13).
- **Fronthaul (paper Sec. XXXI):** the fronthaul and control-link constraint is enforced in Step 13's client
  selection (G14).

**Figures**
- Regrouped to 14. G03, G12 and G14 have two panels each.
- The layers figure shows every algorithm as a reference line.

**Tried and reverted, because they did not help**
- A minimum initial power share.
- A per-stream power-share head. Both targeted the small-network starvation.

## Results

See [`rapfedbeam/README.md`](rapfedbeam/README.md) for the table and
[`rapfedbeam/docs/REPORT_FOR_SIR.md`](rapfedbeam/docs/REPORT_FOR_SIR.md) for the findings.

**What holds**
- **Best distributed method:** RAP-FedWMMSE is far ahead of Local-WMMSE (+11.3 PF).
- **Learning helps, modestly:** +0.9 PF from the learned parameters at fixed τ. The +4.0 over the untrained
  layers is mostly the slow timescale (+3.1 from τ adaptation and priced association); exchanging ζ gives +7.3
  over Local-WMMSE. See Table III of the paper.
- **Robust to stale CSI:** it beats Centralized at ρ ≤ 0.9.
- **Energy:** it saves 22% power at λ_E = 0.1, for a PF loss of 0.14.
- **Speed:** about an order of magnitude less computation than Centralized (partly solver overhead; timings are
  noisy, and they exclude the per-layer ζ exchange).
- **Weakest users:** relays triple the 5th-percentile rate.

**What does not**
- **Federation:** personalized, FedAvg and no-FL are within noise on PF utility. Personalized uses 45% less
  communication than FedAvg.
- **Small network:** trained is worse than untrained, and it nearly starves a UE on 14 of 100 drops.
- **Cooperation:** L_max = 2 slightly lowers PF utility.
- **Signalling:** ζ per layer exceeds the raw CSI at 2 antennas.

**Timings (G11)** were measured with 11 parallel workers while the CPU power limit was changed. They hold to an
order of magnitude only.

## Running it

From the repository root, with Python 3.12:

```bash
python -m venv .venv && .venv/Scripts/activate
python -m pip install -r requirements.txt
cd rapfedbeam
python -m pytest -q                                      # about 2.5 minutes
python run_experiments.py --drops 100 --deployments 10   # about 3 hours on 12 cores
python run_experiments.py --plot-only                    # redraw from saved results
```

If a worker process is killed (sleep, Task Manager), only the experiment in progress is lost: rerun it with
`--experiments <name>`, then `--plot-only`.

## Conventions

- **Gain is `h^H w`,** so matched beams are `h / ‖h‖`.
- **Channels are noise-normalized:** noise variance is 1 and power is in Watts.
- **SCIP never shares a process with PyTorch.** The exact solver always runs in its own process. Never set
  `KMP_DUPLICATE_LIB_OK`.
- **Association** is stored as a relay index per UE when L_max = 1, or as the binary R×K matrix when L_max > 1.
  Use `sim.rates.as_assoc_matrix` to read either form.

## Where to look

| Need | File |
|---|---|
| Results and points for Sir | `rapfedbeam/docs/REPORT_FOR_SIR.md` |
| The main algorithm and every interpretation choice | `rapfedbeam/docs/RAP_FEDWMMSE.md` |
| All algorithms, shared choices, limits | `rapfedbeam/docs/BASELINES.md` |
| The 14 figures and fairness rules | `rapfedbeam/docs/GRAPH_PLAN.md` |
| WMMSE math | `rapfedbeam/docs/WMMSE_DERIVATION.md` |
| Adding a method | `rapfedbeam/docs/ADDING_AN_ALGORITHM.md` |
