# RAP-FedWMMSE — complete handover

Group 12, IIIT Dharwad: Kushagra Kumar Arora (24BCS061), Chinmay Pramod Ardalkar (24BEC007), Rachit Yadav
(24BEC049), Saurav Anand Jha (24BEC054). Supervisor: Prof. Jagadeesha R B.

**Status as of 13 November 2026.** All of Sir's instructions are implemented. All results are final. The IEEE
paper has gone through three Google Deep Review rounds (Revision 3, 14 pages). Everything is committed on branch
`kushagra`, 14 commits ahead of `origin/kushagra` (including this handover) and **not yet pushed**.

**Final submission deadline: 20 November 2026, 5 PM** (Google Form, ZIP of 100 MB at most).

> New to the project? Read [`guide/PROJECT_GUIDE.pdf`](guide/PROJECT_GUIDE.pdf) first. It explains everything
> in simple language, with likely questions from Sir and their answers. This file is the detailed reference.

---

## Contents

1. [What Sir asked for, and where each item is](#1-what-sir-asked-for-and-where-each-item-is)
2. [What is left to do](#2-what-is-left-to-do)
3. [Repository map](#3-repository-map)
4. [How to run everything](#4-how-to-run-everything)
5. [The problem and the system model](#5-the-problem-and-the-system-model)
6. [The algorithm, RAP-FedWMMSE](#6-the-algorithm-rap-fedwmmse)
7. [Interpretation choices (Sir must confirm)](#7-interpretation-choices-sir-must-confirm)
8. [Baselines, ablations and diagnostics](#8-baselines-ablations-and-diagnostics)
9. [Experiments and the 14 figures](#9-experiments-and-the-14-figures)
10. [Final results](#10-final-results)
11. [Where the gain comes from (ablation findings)](#11-where-the-gain-comes-from-ablation-findings)
12. [Results that did not support our hypotheses](#12-results-that-did-not-support-our-hypotheses)
13. [Google Deep Review: three rounds](#13-google-deep-review-three-rounds)
14. [Complete change history](#14-complete-change-history)
15. [Corrections we made to our own claims](#15-corrections-we-made-to-our-own-claims)
16. [Honesty notes and disclosed choices](#16-honesty-notes-and-disclosed-choices)
17. [Conventions and gotchas for developers](#17-conventions-and-gotchas-for-developers)
18. [Reproducibility](#18-reproducibility)

---

## 1. What Sir asked for, and where each item is

| Sir's instruction | Status | Where |
|---|---|---|
| Model-driven personalized federated algorithm based on WMMSE, no dataset (15 Oct email) | Done | `rapfedbeam/train/rap_fedwmmse.py`, `train/unfolded_wmmse.py` |
| Implement his algorithm, Steps 1–15 + pseudocode (`5g_group_12_solution.pdf`) | Done, all steps | Step → code map in `rapfedbeam/docs/RAP_FEDWMMSE.md` |
| Baseline: Centralized WMMSE-SCA/AO | Done | `rapfedbeam/baselines/wmmse.py` (`centralized_wmmse`) |
| Baseline: Local-WMMSE / No-FL | Done | `baselines/wmmse.py` (`local_wmmse`); the no-FL ablation is `rap_fedwmmse_nofl` |
| "etc." baselines | Done | no-relay centralized, RZF, MRT, FedAvg ablation, untrained layers, diagnostic ablations |
| Exact method: global MINLP by spatial branch-and-bound, small inputs | Done (SCIP) | `baselines/exact_minlp.py` |
| Compare 5–6 algorithms across 12–14 significant graphs | Done: 14 figures | `rapfedbeam/figures/` (PNG + CSV of the plotted numbers) |
| IEEE paper in Overleaf (IEEE conference template) | Done (Revision 3) | `paper/main.tex`, `paper/main.pdf`, `paper/Overleaf_Source.zip` |
| ≥ 2 Google Deep Review rounds, a 3rd recommended | Done: 3 rounds | `paper/REVISION_LOG.md`, `paper/deepreview_round{1,2,3}.md` |
| Revision reports (Word/PDF) | Content done; **export pending** | `paper/REVISION_LOG.md` → export |
| Code, install steps, requirements, data/generation instructions, seeds, exact commands | Done | `rapfedbeam/README.md`, `requirements.txt`, `requirements-lock.txt`, `configs/config.yaml` |
| Final ZIP + Google Form | **Pending (team)** | `paper/SUBMISSION_CHECKLIST.md` |

The paper's problem formulation (team paper `5G_2026_Chinmay_team.pdf`) is also covered:
- the L_max > 1 cooperation order;
- the energy term λ_E·P_tot;
- the fronthaul/control-link constraint of Sec. XXXI;
- the QoS outage constraint.

---

## 2. What is left to do

**The submission package is built and verified:** `final_submission/Group12_RAP-FedWMMSE_Final_Submission.zip`
(17.5 MB, not tracked in git). It contains:
- the paper PDF and `Overleaf_Source.zip`;
- the all-comments Deep Review PDF and the revision report PDF;
- the code, with the precomputed results and `REPRODUCIBILITY.md`.

Verification: the Overleaf source compiles standalone; `--plot-only` regenerates all 14 figure CSVs
byte-identically; `diagnostics.analysis` reproduces every quoted statistic; 87/87 tests pass from the packaged
code.

| # | Task | Who | Status |
|---|---|---|---|
| 1 | Verify teammate emails (`<rollno>@iiitdwd.ac.in`) and author order in `paper/main.tex` | Team | Open |
| 2 | Revision reports as Word/PDF | Claude | **Done:** `Revision_Report_Rounds_1-3.pdf` + `DeepReview_All_Comments_Rounds_1-3.pdf` |
| 3 | Create the Overleaf project | Team | Open. New Project → Upload Project → `Overleaf_Source.zip`; compiles as-is with pdfLaTeX |
| 4 | Build the submission ZIP | Claude | **Done** and verified (see above) |
| 5 | Submit the Google Form before 20 Nov, 5 PM | Team | Open. https://forms.gle/9nWd2kZnvqwLCsAKA; upload the ZIP |
| 6 | Send `rapfedbeam/docs/REPORT_FOR_SIR.md` to Sir | Team | Open. It lists the 6 interpretation points he should confirm |
| 7 | Push branch `kushagra` and open a PR into `main` | Needs team approval | Open. Local commits exist only on this laptop |

---

## 3. Repository map

```
5g_project/
├── HANDOVER.md                 ← this file
├── README.md                   → points to rapfedbeam/README.md
├── requirements.txt            minimum versions
├── requirements-lock.txt       exact versions used for the results (Python 3.12.5)
├── guide/
│   ├── PROJECT_GUIDE.tex/.pdf  simple-language study guide + Q&A for Sir
├── paper/
│   ├── main.tex / main.pdf     IEEE paper, Revision 3 (14 pages)
│   ├── figures/                the 14 figures used by the paper
│   ├── Overleaf_Source.zip     upload to Overleaf (not tracked in git; rebuild from main.tex + figures/)
│   ├── REVISION_LOG.md         all Deep Review comments, our checks, changes, comments not acted on
│   ├── deepreview_round{1,2,3}.md   the full review texts
│   ├── SUBMISSION_CHECKLIST.md
│   └── README.md
└── rapfedbeam/                 the code
    ├── configs/config.yaml     every parameter, seed 42
    ├── sim/                    drops, channels, rates, association (Step 1), problem instances
    ├── train/                  rap_fedwmmse.py (algorithm, ablations), unfolded_wmmse.py (layers)
    ├── fl/                     federation: trigger, knapsack, top-k, personalized aggregation
    ├── models/                 role adapters (AP/relay) and the shared backbone
    ├── baselines/              wmmse.py (centralized, local), exact_minlp.py (SCIP), heuristics.py
    ├── experiments/            algorithm registry, runner, sweep definitions
    ├── eval/                   metrics.py (the single scoring path), figures.py (the 14 figures)
    ├── diagnostics/            round2.py, round3.py, runtime.py (review diagnostics)
    ├── figures/                final 14 figures + CSVs
    ├── docs/                   REPORT_FOR_SIR.md, RAP_FEDWMMSE.md, BASELINES.md, GRAPH_PLAN.md, ...
    ├── tests/                  87 tests
    ├── results/                raw JSON results (gitignored)
    └── run_experiments.py
```

Older files from the earlier black-box neural approach still run but are **not** the proposed method:
`train/ctde_train.py`, `train/federated_loop.py`, `run_federated.py`, `run_simulation.py`, `plots/`.

---

## 4. How to run everything

From the repository root, on Windows with Python 3.12:

```bash
python -m venv .venv && .venv/Scripts/activate
python -m pip install -r requirements-lock.txt      # exact versions (or requirements.txt for minimums)
cd rapfedbeam
python -m pytest -q                                  # 87 tests, 2-5 minutes
python run_experiments.py --drops 100 --deployments 10   # all 12 experiments, about 3 hours on 12 cores
python run_experiments.py --experiments energy       # a single experiment
python run_experiments.py --plot-only                # redraw all 14 figures from results/
python -m diagnostics.round2 all                     # Deep Review round 2 diagnostics (about 20 min)
python -m diagnostics.round3 all                     # Deep Review round 3 diagnostics (about 30 min)
python -m diagnostics.runtime                        # clean timing for Table II (run on an idle machine)
```

To compile the paper: `cd paper && pdflatex main.tex && pdflatex main.tex` (MiKTeX installed locally). The same
command in `guide/` builds the study guide.

If a worker process is killed (sleep, Task Manager), only the experiment in progress is lost. Rerun it with
`--experiments <name>`, then `--plot-only`.

---

## 5. The problem and the system model

**Network (defaults).**
- **Nodes:** 8 APs and 2 half-duplex DF relays, each with 2 antennas; 8 single-antenna UEs.
- **Serving:** each UE is served by its 4 strongest APs (cluster); a relay helps at most 4 UEs; L_max = 1 relay
  per UE by default (2 in the cooperation sweep).
- **Channels:** 500 m × 500 m area, 3GPP UMi NLOS path loss −30.5 − 36.7 log₁₀(d/1 m) dB, 4 dB shadowing,
  Rayleigh fading.
- **Link budget:** 20 MHz, −174 dBm/Hz, 7 dB noise figure; 1 W per node.
- **Blockage:** 2 of 8 UEs (round(0.3K)) blocked by an extra 25 dB.
- **Normalization:** channels are noise-normalized, so σ² = 1 and beam power is in W.

**Two-phase frame.**
- **Phase I (fraction τ):** APs transmit; UEs and relays listen.
- **Phase II (fraction 1 − τ):** relays forward.
- **τ range:** [0.2, 0.8] for relay methods; the no-relay baseline uses τ = 1.

**Rates.** C_D = τ·log₂(1+γ¹) + (1−τ)·log₂(1+γ²), C_AR = τ·log₂(1+γ_AR), and r_k = min(C_D, min over assisting
relays of C_AR).

**Assumptions** (stated in the paper):
1. Relays re-encode with independent codebooks.
2. Joint transmitters are phase-synchronized (idealized for relays).
3. Every assigned relay must decode, with no selection-DF fallback.

**Objective (P0).** Maximize Σ ln(r_k + 0.001) − λ_E·P_tot, the PF utility, subject to:
- per-node power;
- association limits (L_max, C_r^max);
- an outage constraint Pr{r_k < R_min = 0.5} ≤ 0.1 over the CSI error;
- τ in its range.

**P0-PF** is P0 without the outage constraint. The baselines and the exact MINLP solve P0-PF.

**Power.**
- **P_tot** (objective): unweighted transmit + circuit power.
- **P̄_tot** (reported alongside): time-averaged; AP transmit power weighted by τ, relay by 1 − τ.

**Fronthaul (Sec. XXXI).** D_FL_m ≤ [F_max·T_e − D_data_m]₊ for APs and D_FL_r ≤ F_ctrl·T_e for relays, with
F_max = 1 Gbit/s, F_ctrl = 10 Mbit/s and T_e = 1 s.
- It gates FL traffic only; user data is not throttled.
- An AP whose user data alone exceeds F_max is counted as "overloaded".
- ζ exchanges are not counted in this budget.

---

## 6. The algorithm, RAP-FedWMMSE

Two timescales. Full step → code map: `rapfedbeam/docs/RAP_FEDWMMSE.md`.

**Slow timescale (once per epoch)**

| Step | What it does |
|---|---|
| 1 Association | Score Γ_rk = α₁β̄RU + α₂β̄AR − α₃Ī + α₄λQ, with α = (1, 1, 0.25, 1). A pair is eligible if the relay hop beats the direct link (relaxed by the QoS price) and Γ > 0.5. Exact capacitated matching (SciPy `milp`) |
| 2 Phase split | 61-point grid over [0.2, 0.8] on the pure PF utility at measured SINRs; usually picks 0.8 |
| 3 Slow vector ζ_s | 4K + RK = 48 real numbers: QoS prices, DF prices, measured interference in both phases, mean rates |

**Fast timescale (every block)**

| Step | What it does |
|---|---|
| 4 Models | Per node: role adapter (local, in→64→32), shared backbone (federated, 32→64→64→32, 8,352 parameters) and head (local, 2L+1+K outputs). The head gives η ∈ (0,2), δ ∈ (0,1) per layer, one λ_reg ≥ 0, and initial power shares φ. Initialized to η = 1, δ = 0.5, λ_reg = 1, equal shares (= the untrained ablation) |
| 5–8 Unfolded layers | L = 6. Each layer: exchange ζ_f (352 values: effective gains, MMSE equalizers, link weights, relay combiners); exact local gradient and curvature (unit-tested against autograd); update W̃ = W + η((Q+λI)⁻¹B − W) with λ = λ_reg·μ (μ = classical WMMSE power price, by bisection); relaxation W ← (1−δ)W + δW̃; mask; radial clipping to the power budget. With η = δ = λ_reg = 1 this is exactly classical per-node WMMSE (unit-tested) |
| 9 Prices | In-layer: λQ += 0.2(R_min − r), ν += 0.2(C_D − C_AR), from detached rates, discarded after the block. Per epoch: step 0.5 |
| 10 Robust loss | Loss = −J (on the estimate) + λ_O·Σ[p̂ − 0.1]₊, where p̂ = mean over 4 CSI-error draws (ρ = 0.98) of sigmoid(10(R_min − r)) |
| 11 Training | Self-supervised, no dataset: 30 epochs × 4 Adam steps × 8 blocks, lr 3e-3. A fresh UE drop every epoch. Gradients go through all nodes (a simulation shortcut) |

**Federation (Steps 12–15, after each epoch)**
- **Trigger (Step 12):** urgency = drift + signature change + age/5 + grad norm; candidate if ≥ 0.3.
- **Selection (Step 13):** exact knapsack within B_FL = 40,000 B and each node's fronthaul headroom. Uploads
  are equal-sized, so this means the top 5 most urgent nodes.
- **Compression (Step 14):** top-10% upload with error feedback (6,736 B); dense download 33,408 B.
- **Aggregation (Step 15):** personalized, α_ij ∝ role-weighted Gaussian kernel on signatures (κ_AR = 0.3); a
  selected node sets θ ← θ_rec + 0.5·Δθ̄.

**Deployment, per drop.**
1. Reset ζ_s.
2. Run 3 warm-up epochs of 4 sampled blocks: association, τ, prices.
3. Run the 6 layers on the block.

---

## 7. Interpretation choices (Sir must confirm)

These are listed in `rapfedbeam/docs/REPORT_FOR_SIR.md`.

1. **Step 9:** as written, the DF price never rises (r − C_AR ≤ 0 always). We price C_D − C_AR, with a
   stop-gradient on C_D.
2. **Step 7:** a regularized WMMSE step instead of a plain gradient step. It keeps Sir's η, δ and λ_reg, and
   reduces to classical WMMSE. A plain gradient step plateaued much lower (PF about 3.6 against 5.6 for
   Centralized on a test drop).
3. **Step 11:** gradients are computed through all nodes in simulation. A real network would need a backward ζ
   exchange.
4. **Step 1:** the α weights and the "relay beats the direct link" rule are ours, calibrated on deployments
   90–95, which are never used for evaluation.
5. **Training data:** a fresh UE drop every epoch; with one moving drop, the model overfit.
6. **Fronthaul:** the limit gates FL only; user data is not throttled.

---

## 8. Baselines, ablations and diagnostics

| Name (code) | What it is |
|---|---|
| `rap_fedwmmse` | The proposed method |
| `centralized_wmmse` | Full CSI, SCA with Clarabel via CVXPY, ≤ 100 iterations (tol 1e-3), τ optimized per block, certainty-equivalent |
| `centralized_no_relay` | The same without relays, τ = 1 |
| `local_wmmse` | 100 iterations of single-node WMMSE on own CSI, no exchange, no statistical regularization, τ = 0.5 |
| `rzf_equal`, `mrt_equal` | RZF (regularization σ²/P) / MRT with equal power, τ = 0.5 |
| `rap_fedwmmse_fedavg`, `rap_fedwmmse_nofl` | Ablations: FedAvg (dense; not limited by B_FL), no federation |
| `unfolded_wmmse` | The same layers untrained (η = 1, δ = 0.5, λ_reg = 1), τ = 0.5, no warm-up, no slow timescale |
| `exact_minlp` | SCIP spatial B&B on a small network (2 APs, 1 relay, 2 UEs, 1 antenna); beams, combiners and binary association; τ = 0.5; 1% gap; 300 s limit |
| `unfolded_wmmse_slow` | (Round 2) untrained layers + training-free slow timescale (Steps 1–2 + warm-up) |
| `classical_wmmse_layers`, `classical_wmmse_slow` | (Round 2) undamped classical per-node WMMSE (η = δ = λ_reg = 1), without and with the slow timescale |
| `unfolded_wmmse_tau_only` | (Round 3) untrained layers + τ search only (zero-price association, zero slow prices) |

**Fairness rules.**
- Same drops for every method, and the same scoring code (`eval/metrics.evaluate_solution`).
- Training uses separate drop indices (from 10⁶).
- The no-relay baseline gets τ = 1.
- G07, G09 and G10 hold τ and the association fixed for every relay method.

---

## 9. Experiments and the 14 figures

All use 100 drops over 10 deployments, seed 42 and 95% CIs. Trained models are trained per deployment and per
sweep point.

| Fig | Content | Experiment |
|---|---|---|
| G01 | Per-user rate CDF | baseline |
| G02 | Outage vs R_min | baseline |
| G03 | (a) mean and (b) 5th-percentile rate vs power, 10–30 dBm | power |
| G04 | PF vs K = 4…12 | num_ues |
| G05 | PF vs blocked fraction 0–0.75 | blocked_fraction |
| G06 | PF vs R = 1…4, with L_max = 2 dashed | num_relays, cooperation |
| G07 | PF vs fixed τ 0.2–0.8 | tau |
| G08 | PF vs aging ρ 0.8–0.99 | aging |
| G09 | PF vs layers 1–10, trained vs untrained, with reference lines | layers |
| G10 | Gap to the certified MINLP bound | optimality_gap |
| G11 | Runtime vs K (under parallel load) | num_ues |
| G12 | FL learning curves and bytes | baseline |
| G13 | PF vs P_tot, swept by λ_E | energy |
| G14 | FL under fronthaul: PF and participation vs F_max | fronthaul |

---

## 10. Final results

**Default network, 100 drops** (Table II of the paper):

| Method | PF utility | Mean rate | 5th pct | Outage @0.5 | Time/drop* |
|---|---|---|---|---|---|
| Centralized WMMSE-SCA/AO | 9.88 ± 0.40 | 4.57 | 0.67 | 3.0% | 5.05 s |
| Centralized, no relay (τ = 1) | 9.80 ± 0.53 | 5.41 | 0.18 | 10.9% | 1.46 s |
| **RAP-FedWMMSE** | **7.88 ± 0.51** | **3.49** | **0.57** | **4.0%** | **0.157 s** (0.124 warm-up + 0.033 block) |
| Unfolded WMMSE, untrained | 3.87 ± 0.46 | 2.06 | 0.36 | 8.1% | 0.03 s |
| Local-WMMSE | −3.43 ± 0.43 | 0.89 | 0.20 | 34.0% | 0.32 s |
| RZF / MRT | −2.43 / −5.60 | 1.13 / 0.58 | 0.20 / 0.20 | 38.5% / 48.5% | ≤ 0.001 s |

\*From one clean run (`diagnostics/runtime.py`): 10 drops, one process, one thread, idle machine. 100 closed-form
layers take 0.47 s. Timings under parallel load (G11) are noisier.

**Key paired results**

| Comparison | Result |
|---|---|
| RAP − Local-WMMSE | +11.31 ± 0.43 (100/100 drops) |
| RAP − RZF | +10.31 ± 0.36 |
| RAP − untrained | +4.0 ± 0.23 (99/100) |
| At their best fixed τ = 0.8 | Local −0.31, RZF 0.34; RAP still leads by +8.2 / +7.5 |
| Mean rate vs centralized | 76% |

**Relays**
- Relays triple the 5th-percentile rate (0.57 vs 0.18 without relays) and cut outage from 10.9% to 3–4%.
- At 75% blocked users, the relay-assisted centralized method wins by +0.97 ± 0.43.
- At 0% blocked, no relay is better by 1.86, because it uses the whole frame.
- More relays help with diminishing returns: centralized 9.32 (R = 1) → 9.97 (R = 4).

**Stale CSI (G08)**
- RAP beats centralized at ρ ≤ 0.9: +1.01 ± 0.40 (0.8), +0.78 (0.85), +0.62 (0.9).
- A model trained once (ρ = 0.98) and evaluated zero-shot still beats it: +0.69, +0.61, +0.41.
- The no-relay benchmark stays above both in PF, but RAP has the better tail (p5 0.15 vs 0.08 at ρ = 0.8).
- **The advantage comes from the damped structure, not from learning** (see §11).

**Energy (G13)**
- At λ_E = 0.1: P_tot 23.9 → 18.7 W (−22%; −19% time-averaged, 17.4 → 14.2 W) for a PF loss of 0.14.
- Time-averaged at λ_E = 0: relay methods 17.4 / 17.7 W vs no relay 20.9 W, the opposite ordering to P_tot.

**Exact (G10, small network)**
- SCIP certified 94/100 drops (6 hit the time limit; the bound is still valid).
- Mean gaps: exact 0.040, centralized 0.348, untrained 0.797, RAP 1.119, Local 1.642, MRT 1.766.
- 0 violations of the bound in 600 records.
- With the association fixed to Step 1, SCIP's bound drops by 0.243 on average: about a fifth of RAP's gap is
  the relay choice.

**Federated learning (G12, G14)**
- Personalized − no FL = −0.06 ± 0.19, i.e. no gain.
- Bytes: personalized 11.0 MB vs FedAvg 20.0 MB (uploads + downloads); no FL needs 0 MB for the same utility.
- At F_max = 100 Mbit/s, 7.1 of 8 APs are overloaded and participation falls to 2.8 of 10 nodes, yet utility
  stays flat.

**Other sweeps**
- **L_max = 2** lowers PF: −0.19 for centralized, −0.81 for RAP at R = 2.
- **Layers:** trained − untrained = +1.37 (L = 1), +0.89 ± 0.16 (L = 6), +0.64 (L = 10). Six trained layers ≈
  ten untrained (4.76 vs 4.74).
- **Load:** RAP drops faster with K (7.88 → 4.80 from K = 8 to 12, against 9.88 → 8.02 for centralized).

---

## 11. Where the gain comes from (ablation findings)

**Decomposition of +11.31 over Local-WMMSE** (paired, Table III):

| Step | Order A | Order B |
|---|---|---|
| Coordination ζ_f (untrained − Local) | +7.30 ± 0.35 | +7.30 ± 0.35 |
| Learned parameters | +0.89 ± 0.16 (at τ = 0.5) | +0.61 ± 0.20 (with Steps 1–2 on) |
| Slow timescale | +3.12 ± 0.27 | +3.40 ± 0.17 |

The PF levels are −3.43 → 3.87 → 4.76 → 7.88 (A) and −3.43 → 3.87 → 7.27 → 7.88 (B).

**Diagnostic findings (Rounds 2–3).** Each was run on the exact final-run drops, after reproducing the
published numbers.

**What the gain is made of**
- **The slow-timescale gain is all τ.** τ alone gives +3.40. The priced association, warmed-up prices and
  warm-up add −0.002 ± 0.047.
- **Learning buys fewer layers, not a better end point.**
  - Undamped classical WMMSE is worse: 1.68 at L = 6 and −2.65 at L = 100, because parallel Jacobi steps
    oscillate.
  - Damped untrained layers reach 5.54 at L = 100, above 6 trained layers (4.76; paired −0.77 ± 0.39) but 1.80
    below centralized.
- **Learned parameters (default network):** AP step 1.18 (1.24–1.34 in layers 1–4, about 1.0 in 5–6),
  relaxation 0.59, effective step 0.70 (vs 0.5 untrained), λ_reg 0.61.
- **η and δ act only through their product** (no nonlinearity between them).
- **The QoS machinery has no effect:** pure-PF RAP (no outage term, no prices) reaches 7.79 vs 7.88 (n.s.),
  with outage 3.5% vs 4.0%.

**Stale CSI**
- **The advantage is not from robust training or learning.**
  - A model trained without the outage term does as well (+0.11, +0.04, +0.09 at ρ = 0.8, 0.85, 0.9).
  - Untrained layers + Steps 1–2 beat trained RAP (+0.39, +0.40, +0.21) and centralized (+1.40, +1.18, +0.83).
- **Zero-shot:** the default model loses only 0.05–0.33 to the per-ρ retrained models.

**Higher load**
- **At K = 12, the fixed depth is the problem.** Untrained + slow reaches 3.73 at L = 6 and 6.85 at L = 100,
  i.e. +2.05 over RAP and −1.17 vs centralized.
- ζ_f grows as O(RK²): 4,320 values per block at K = 12, vs 304 for raw CSI.

**Small-network starvation** (UE rate < 0.02 bit/s/Hz)

| Variant | Starved drops (of 100) | Mean gap |
|---|---|---|
| Trained RAP (published) | 14 | 1.119 |
| Untrained layers | 3 | |
| Untrained + warm-up | 4 | |
| Centralized | 1 | |
| Trained, step capped at η ≤ 1 | 8 | 0.991 |
| Trained, equal shares | 10 | 1.034 |
| Retrained without QoS | 12 | 1.059 |

- λ_reg is lower on the starved drops (0.35 vs 0.58).
- **Conclusion:** the learned parameters together cause it (larger steps, low λ_reg, learned shares). There is
  no single trigger.

---

## 12. Results that did not support our hypotheses

1. **Federation brings no utility gain.** Two reasons: unlimited simulated data per node, and network-wide
   gradients in training. Not re-tested in a data-limited regime, by team decision.
2. **The learned parameters add only +0.6 to +0.9;** 100 untrained layers beat 6 trained ones.
3. **The QoS machinery (Steps 9–10) and the priced association had no measurable effect.**
4. **The stale-CSI advantage is structural, not learned;** learning slightly reduces it.
5. **Small single-antenna network:** trained worse than untrained; 14% of drops starved.
6. **L_max = 2 lowers PF** (full decoding at every relay).
7. **Signaling:** ζ_f (352/layer) exceeds raw CSI (224) at 2 antennas; it reverses only for large arrays.
8. **6 layers are too few at K = 12.**
9. **Models are topology-specific:** a new network size needs a new model trained from scratch.

---

## 13. Google Deep Review: three rounds

Full details are in `paper/REVISION_LOG.md`, where every comment has our check against the code and the change
made or the reason it was not made.

| Round | Paper uploaded | Comments | Key outcome | Revision |
|---|---|---|---|---|
| 1 | Initial draft (9 pp, `a2327e2`) | 20 | Caught our **+4.0 "learning" overclaim** (the untrained ablation is at τ = 0.5) → decomposition added; ζ split into ζ_s/ζ_f; all formulas, parameters and assumptions; runtime claim softened; 7 verified references; citations renumbered | Rev 1 (11 pp, `f4aa710`) |
| 2 | Rev 1 (`3b352b2`) | 16 | No wrong number. New ablations (untrained + slow; undamped classical); zero-shot aging; starvation diagnosis; time-averaged power (energy rerun reproduced all values exactly); clean timing; FL "45%" claim reframed | Rev 2 (13 pp, `af0d741`) |
| 3 | Rev 2 (`af0d741`) | 17 | No wrong number. Slow gain = all τ; pure-PF ablation (QoS no effect); **stale-CSI explanation corrected**; K = 12 depth; default-network parameters; association gap; many definitions | Rev 3 (14 pp, `cd8fb72`) |

Things the reviewers got wrong, which we checked and logged:
- "F_mix" in Fig. 14 (it is F_max);
- the MRT 5th percentile "duplicate" (it really is 0.20);
- a misread of Fig. 8's values.

Each round's references all verified (11/11, then 18/18 twice).

**Not acted on, with reasons logged:**
- a robust/diagonally loaded centralized baseline;
- a Team MMSE baseline;
- a limited-data/detached-gradient FL test;
- a τ-weighted energy objective;
- selection DF;
- resetting the training duals per drop.

---

## 14. Complete change history

| Commit | What |
|---|---|
| `de30fda` … `2f1ec6f` | Early work: reduced RAP-FedBeam simulation and the earlier black-box neural approach (before Sir's 15 Oct direction); PR #1 merged |
| `60a731b` | Implemented RAP-FedWMMSE, all baselines, exact MINLP and the 14-graph experiment system |
| `bad7b62` | **Fixes:** no-relay baseline τ = 1 (was unfairly capped at 0.8); MINLP now optimizes the binary association (a true MINLP); crash from a singular beam update fixed by diagonal loading 1e-10·tr(Q)/N; training on fresh UE drops each epoch (one moving drop overfit) |
| `0d92349` | **Every instruction covered:** full ζ(t) in Step 3; L_max > 1 everywhere (rates, association, both WMMSE baselines, MINLP, RAP); energy term exercised (G13); fronthaul constraint of Sec. XXXI in Step 13 (G14); figures regrouped to 14 |
| `fe27e1d` | Merged `main` into `kushagra` |
| `76f4a2b` | Final run (100 drops × 10 deployments, 12 experiments, 0 failed drops), 14 figures, report for Sir, docs |
| `3c439d2` | IEEE paper (IEEEtran), revision-log template, submission checklist |
| `a2327e2` | Four team authors added |
| `f4aa710` | **Revision 1** (Deep Review round 1); figure labels in math notation |
| `3b352b2` | Chinmay's institute email |
| `af0d741` | **Revision 2:** `diagnostics/round2.py`, `diagnostics/runtime.py`; new ablation algorithms; `average_power` metric in the evaluator; energy experiment rerun (identical PF/P_tot); clean timing; `requirements-lock.txt`; legend fixes ("Local-WMMSE", "(converged)" only for iterative methods); `tests/test_diagnostics.py` |
| `cd8fb72` | **Revision 3:** `diagnostics/round3.py`, `unfolded_wmmse_tau_only`; corrections to the stale-CSI and QoS claims; study guide `guide/PROJECT_GUIDE.pdf` |
| (this commit) | This handover |

**Things tried and reverted:**
- a minimum initial power share;
- a per-stream power-share head.

Both targeted the small-network starvation and did not help. An earlier λ-as-damping layer plateaued at PF about
3.6 and was replaced by the regularized step.

---

## 15. Corrections we made to our own claims

Each was found either by Deep Review or by our own checks, and all are reflected in the paper and in
`REPORT_FOR_SIR.md`:

1. **"Learning gives +4.0"** → the learned parameters give +0.6 to +0.9; the rest is coordination and τ (R1).
2. **"+11.3 over the best non-learned distributed baseline"** → false; the untrained layers are one (R1).
3. **"The loss is averaged over the CSI-error draws"** → the draws enter the outage term only (R1, our check).
4. **"2.8 of 12 nodes"** → 10 nodes (R1, our check).
5. **"50× faster"** → an order of magnitude; partly CVXPY overhead; timings noisy (R1/R2).
6. **"Personalized FL saves 45%"** → only relative to FedAvg; no FL needs 0 MB (R2).
7. **Default-network behavior explained with small-network parameters** → replaced (R3).
8. **"The robust training of Step 10 pays off under stale CSI"** → not supported; the advantage is structural
   (R3).
9. **"Every method peaks at K = 8"** → only the WMMSE-based ones (R3).

No published result value was ever changed by a correction. The corrections concern attribution and wording.

---

## 16. Honesty notes and disclosed choices

All of these are stated in the paper or the docs:

**Tuning and model choices**
- No results were tuned on evaluation drops.
- One mild exception: `redrop_every = 1` was chosen from a 20-drop pilot that overlapped evaluation drops.
- The Step 7 update form was chosen for performance (disclosed as interpretation point 2).
- The Step 1 α weights were calibrated on held-out deployments 90–95.

**Simulation shortcuts**
- Training gradients flow through all nodes (simulation shortcut).
- Training carries ζ_s (including prices) across fresh UE drops; user indices are not matched. Deployment resets
  ζ_s per drop.
- Signature standardization uses global statistics. Adapters and heads are initialized independently.

**Evaluation choices**
- Training always includes the robust outage term with ρ = 0.98, even in perfect-CSI experiments.
- Fronthaul gates FL only; ζ_f/ζ_s are not counted in it.
- P_tot is not duty-cycle weighted. It over-penalizes relay power by τ/(1 − τ) (4× at τ = 0.8); P̄_tot is
  reported alongside.

**Measurement**
- The G11 runtimes were measured under parallel load while the CPU power limit (ThrottleStop) was changed. Table
  II uses the clean run instead.
- One test failed once transiently during Round 3 work. It did not recur, either alone or in the full run that
  followed (87/87 pass).

---

## 17. Conventions and gotchas for developers

**Physics and data conventions**
- **Gain is `h^H w`,** so matched beams are `h / ‖h‖`. Channels are noise-normalized (σ² = 1, power in W).
- **Association** is stored as a relay index per UE (L_max = 1) or the binary R×K matrix (L_max > 1). Always read
  it via `sim.rates.as_assoc_matrix`.
- **The untrained ablation has no slow timescale.** Do not compare it to RAP as "the learning" without the
  diagnostic ablations.

**Process and environment**
- **SCIP never shares a process with PyTorch;** the exact solver runs in its own process. Never set
  `KMP_DUPLICATE_LIB_OK`.
- **Determinism:** seeds are per (seed, deployment, drop) with independent streams. Serial and parallel runs give
  identical numbers, and the energy rerun reproduced every value exactly.
- **Windows:** use Git Bash or PowerShell 5.1. Edit UTF-8 files with Python or the editor, not PowerShell
  `Get-Content | Set-Content` (it corrupted characters once).

**Results and outputs**
- **`results/` is gitignored.** The committed `figures/*.csv` hold the plotted numbers. The diagnostics write to
  `results/diagnostics/` (also gitignored) and are reproducible by command.
- **The training-mode code** `RapFedWMMSETrainer.run()` carries `self.coordination` across drops on purpose (the
  as-run behavior). Changing it would require rerunning everything.

---

## 18. Reproducibility

- **Python and packages:** Python 3.12.5. Exact packages are in `requirements-lock.txt`: torch 2.14.0+cpu,
  numpy 2.5.4, scipy 1.18.1, cvxpy 1.9.3, clarabel 0.11.1, PySCIPOpt 6.3.0, matplotlib 3.11.2.
- **Data:** all data is synthetic, generated by the simulator from `configs/config.yaml` (seed 42). There is no
  external dataset.
- **Commands:** see §4. The final run takes about 3 hours on 12 cores; the diagnostics take about 1 hour in total.
- **The paper** compiles with pdfLaTeX, both locally and in Overleaf.
