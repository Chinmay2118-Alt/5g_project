# RAP-FedWMMSE handover

Status as of 10 November 2026, for Group 12 and anyone picking up this work.

## Where things stand

- **Done:** Sir's main algorithm, RAP-FedWMMSE, is implemented in full (Steps 1–15 of his document), along with
  every baseline he asked for. All 14 graphs are produced, with 0 failed runs, and 72 tests pass.
- **⚠️ Nothing is committed.** All of this is uncommitted on branch `kushagra`. Commit it first
  (see [Git](#git-read-before-committing)).
- **⚠️ Nothing has been sent to Sir.** A draft is in `rapfedbeam/docs/REPORT_FOR_SIR.md`. It lists four places
  where his algorithm had to be interpreted, which he should confirm.
- **Results are from 20 drops over 4 deployments.** That is enough to see the picture, but not for final claims.
  Run 100+ drops before the report (see [What is left](#what-is-left)).

## Background

- **The paper:** RAP-FedBeam, on personalized federated beam optimization for relay-assisted cell-free massive MIMO.
- **Sir's direction, 15 Oct:** a model-driven personalized federated algorithm based on WMMSE, with no dataset.
  Baselines: Centralized WMMSE-SCA/AO, Local-WMMSE / No-FL, and an exact method (MINLP by spatial branch-and-bound,
  small networks only). Compare 5–6 algorithms across 12–14 graphs.
- **Sir's main algorithm, received 9 Nov:** RAP-FedWMMSE (the document `5g_group_12_solution.pdf`), which
  combines unfolded WMMSE, primal-dual QoS and DF correction, and personalized federated learning.
- **Earlier sessions** fixed the simulator and built the baselines and the experiment system.

## What this session did

### Implemented Sir's algorithm

| Steps | What | Where |
|---|---|---|
| 1 | Relay association by capacitated matching | `rapfedbeam/sim/association.py` |
| 2 | Phase split τ per epoch | `train/rap_fedwmmse.py` (`_best_tau`) |
| 3–4 | Role adapters, personalized backbone, head predicting step sizes, damping, regularization and initial beams | `train/rap_fedwmmse.py` (`NodeModel`) |
| 5–9 | Unfolded WMMSE layers: MMSE combiners, SINRs, DF soft-min, WMMSE variables, learned update, projection, dual correction | `train/unfolded_wmmse.py` |
| 10–11 | Robust CSI loss and local self-supervised training | `train/rap_fedwmmse.py` |
| 12–15 | Event trigger, knapsack client selection, Top-k with error feedback, personalized aggregation | `fl/communication.py`, `_federate` |

It also includes three ablations: FedAvg instead of personalization, no federation, and the same layers untrained.
`rapfedbeam/docs/RAP_FEDWMMSE.md` maps every step to the code and explains every interpretation choice.

### Supporting changes

- **Fixed deployments:** drops share an AP and relay layout, so a personalized model is evaluated on the network it
  was trained for. Training drops never overlap evaluation drops.
- **Mobility:** UEs move between training epochs, so the slow-timescale loop sees changing conditions.
- **One association rule for everyone:** every method now uses Step 1's association, so comparisons stay fair.
- **Centralized optimizes τ,** as P0 does, so it remains a true upper reference.
- **Learning algorithms train in parallel,** once per deployment.
- **The feasibility check rejects NaN.** NaN fails every comparison, so it had been slipping through.
- **Graphs:** G10 became PF utility vs unfolded layers, and G13 and G14 are the federated learning curves.

## Results (default network, 20 drops)

| Method | PF utility | Mean rate | 5th-percentile rate | Time per drop |
|---|---|---|---|---|
| Centralized WMMSE-SCA | 9.33 ± 0.91 | 4.36 | 0.71 | 57 s |
| **RAP-FedWMMSE** | **7.51 ± 0.99** | **3.48** | **0.58** | **0.3 s** |
| Centralized, no relay | 7.71 ± 0.93 | 4.13 | 0.17 | 14 s |
| RAP-FedWMMSE, FedAvg / no FL | 7.45 / 7.37 | 3.43 / 3.40 | 0.59 / 0.59 | 0.2 s |
| Unfolded WMMSE, untrained | 3.61 ± 0.97 | 2.04 | 0.40 | 0.07 s |
| RZF / Local-WMMSE / MRT | −2.39 / −3.28 / −5.47 | 1.08 / 0.87 / 0.58 | 0.22 / 0.24 / 0.20 | ≤ 1.5 s |

### What the results show

- **RAP-FedWMMSE is the best distributed method by a wide margin:** about 80% of the centralized mean rate, about
  180× faster. Its PF utility is close to Centralized without relays: above it at low power, with heavy blockage
  and under aging, and somewhat below it elsewhere (7.51 against 7.71 at default, 4.39 against 5.01 at K = 12).
  It stays below full Centralized in every sweep except aging.
- **Learning helps:** trained beats untrained at every depth on the default network, by 0.9–1.5 at 4–10 layers (G10).
- **Robustness to stale CSI (G09):** RAP-FedWMMSE beats Centralized on average at every ρ up to 0.95. The confidence
  bands overlap.
- **Relays help the worst users (G04, G06):** with relays, the 5th-percentile rate is 3.5× higher than Centralized
  without relays.
- **Federation:** personalized aggregation reached the highest validation utility (7.34 against 7.10 for FedAvg and
  7.15 for no FL) using 11 MB against FedAvg's 20 MB. **With 4 deployments these differences are within noise.**
- **Exact (G11):** Centralized is within 0.05 of the certified bound on the small network. No method exceeded the
  bound (0 violations in 120 records), and SCIP's objective matched the simulator to 10⁻⁶.

### What the results do not show, and weaknesses

- **On the small network, trained RAP-FedWMMSE did worse than untrained** (0.53 against 0.29 below the bound),
  mostly because of one drop at 4.7. The training was tuned on the default network only.
- **Coordination traffic is larger than raw CSI at this size:** 352 complex numbers per layer against 224 for all
  CSI (`docs/RAP_FEDWMMSE.md`, Overhead). It does not grow with antennas, so it would win only with larger arrays.
- **Training is local only at deployment.** In simulation, each node's training gradient is computed through
  every node's layers.
- **Association weights are my calibration,** on held-out deployments. On average they beat the older rule, but
  they lost 3.07 on 1 of 12 calibration drops.
- **τ always goes to its 0.8 limit.** Most UEs have no relay and receive nothing in Phase II, so more Phase I time
  always helps. The τ range may deserve a second look with Sir.
- **No external validation:** nothing has been compared against a published result. The tests check the code
  against the equations and against each other.

## How it was verified

| Check | Result |
|---|---|
| Each node's local gradient (own CSI + ζ) equals the global autograd gradient | Exact, to machine precision |
| An AP's update does not read another AP's CSI | Pass |
| A neutral layer equals textbook WMMSE, written out independently | Pass |
| Every layer meets power, cluster and association constraints | Pass |
| Matching and knapsack against brute force | Optimal on 20 random cases each |
| Top-k error feedback loses nothing | Pass |
| RAP-FedWMMSE gives identical numbers in serial and parallel runs | Pass |
| Full run | 0 failed records; 0 bound violations |

Earlier checks still pass: closed-form rates, a signal-level SINR simulation, the WMMSE bound, and exact
against the simulator.

## What is left

1. **Commit** (see below).
2. **Send `rapfedbeam/docs/REPORT_FOR_SIR.md`** after reading it. Ask him to confirm the four interpretation
   choices it lists.
3. **Final run for the report:** `python run_experiments.py --drops 100 --deployments 10` (several hours on 12
   cores), then `--experiments num_ues --workers 1` for clean timings in G12.
4. **Optional:**
   - investigate the small-network result, with more epochs or a lower learning rate there
   - consider a cluster rule that accounts for the relay hop
   - try a larger τ range
5. **Write the report** from the final numbers. `docs/RAP_FEDWMMSE.md` and `docs/GRAPH_PLAN.md` hold the method
   and evaluation text.

## Running it

From the repository root, with Python 3.12 (the version tested). The virtual environment used in these sessions
was temporary, so each person needs their own.

```bash
python -m pip install -r requirements.txt
cd rapfedbeam
python -m pytest -q                                      # 1–5 minutes
python run_experiments.py --drops 20 --deployments 4     # about an hour on 12 cores
python run_experiments.py --plot-only                    # redraw from saved results
```

Results and graphs go to `rapfedbeam/results/`, which is gitignored. Copy the graphs elsewhere to keep them.

## Conventions

- **Gain is `h^H w`,** so matched beams are `h / ‖h‖`.
- **Channels are noise-normalized:** noise variance is 1 and power is in Watts.
- **Link budget (my choice):**
  - path loss: 3GPP UMi NLOS, −30.5 − 36.7 log₁₀(d) dB
  - noise: 20 MHz bandwidth, 7 dB noise figure
  - power and area: 1 W per AP and relay, 4 dB shadowing, 500 m square
- **SCIP never shares a process with PyTorch.** The two clash, so the exact solver always runs in its own process.
  Never set `KMP_DUPLICATE_LIB_OK`.

## Git: read before committing

- **Branch:** `kushagra`, 2 commits behind `origin/main`.
- **Expect merge conflicts with `main`.** `main` deleted `rapfedbeam/README.md` and changed the 3 PNGs in
  `rapfedbeam/plots/`, and this work changes all four.
  - Keep this work's README, because it is the new guide.
  - Keep this work's PNGs. `main`'s were made with the old buggy simulator.
- **Deliberately deleted:**
  - `baselines/numeric_opt.py`, a fake upper bound
  - `train/local_zeta_train.py`, a stub that Sir's algorithm supersedes
  - `train/rap_fedbeam.py`, the empty slot
  - `docs/MAIN_ALGORITHM_INTEGRATION.md`, replaced by `docs/ADDING_AN_ALGORITHM.md`
  - `docs/QUESTIONS_FOR_SIR.md`, answered and replaced by `docs/REPORT_FOR_SIR.md`

## Where to look

| Need | File |
|---|---|
| The main algorithm and every interpretation choice | `rapfedbeam/docs/RAP_FEDWMMSE.md` |
| All algorithms, shared choices, limits | `rapfedbeam/docs/BASELINES.md` |
| The 14 graphs and fairness rules | `rapfedbeam/docs/GRAPH_PLAN.md` |
| WMMSE math | `rapfedbeam/docs/WMMSE_DERIVATION.md` |
| Adding a method | `rapfedbeam/docs/ADDING_AN_ALGORITHM.md` |
| Message for Sir | `rapfedbeam/docs/REPORT_FOR_SIR.md` |
