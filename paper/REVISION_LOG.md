# Deep Review revision log — Group 12

This log records every Google Deep Review round, as Sir asked: the review comments from each upload, and the
changes made in each revision. Export it to Word or PDF for the final submission.

Rules from Sir's instructions:
- At least TWO complete review–revision rounds are mandatory; a third is strongly recommended.
- Comments are addressed honestly. If a comment requires rerunning an experiment or correcting an equation,
  that is done and stated here. If a comment is not acted on, the reason is given.

---

## Round 1

**Uploaded:** 11–12 November 2026 — initial draft (`main.pdf`, 9 pages, commit `a2327e2`).
**Full review text:** [`deepreview_round1.md`](deepreview_round1.md).

The review rated the evaluation protocol and the reporting of negative results as strengths. Its own check found
every reported number internally consistent with the tables and figures, and all cited references real (11 of 11
verified). Before acting on the comments, we checked each factual claim about our implementation against the
code and the saved results.

### Review comments received

| # | Section | Comment (summary) | Our check |
|---|---|---|---|
| 1.1 | Abstract, VIII-A, VIII-E, IX | The +4.0 "learning" margin over the untrained layers confounds learning with τ adaptation: the untrained ablation runs at fixed τ = 0.5 with no slow timescale. The abstract's "+11.3 over the best non-learned distributed baseline" is wrong, since the untrained layers (3.87) are a non-learned distributed baseline. Decompose the gain. | **Correct.** Confirmed in code (`unfolded_wmmse` uses `instance.tau` = 0.5, zero prices, no warm-up). This was our overclaim. |
| 1.2 | V-C, V-D, V-H, Alg. 1 | ζ is used for two different things: the slow 4K + RK input vector and the fast per-layer 352-value exchange. | Correct. |
| 1.3 | Abstract, VIII-G | The 50× speedup partly reflects CVXPY/Clarabel overhead; per-layer exchange latency is not included; the warm-up count and serial vs. parallel timing are unstated. | Correct. We also found that timings vary up to 4× between runs. |
| 1.4 | V-F, VIII-I | Training back-propagates the global objective through all nodes, so even the no-FL nodes get network-wide gradients. This is a second reason for the federation null result. | Correct. |
| 1.5 | III-C, IV (P0) | The per-epoch fronthaul constraint (8) inside the per-block P0 would cap user rates; the probability measure in the outage constraint is undefined; "r active" is informal; the power terms are not duty-cycle weighted. | Correct. In the code, (8) only gates FL. P_tot is not τ-weighted. |
| 1.6 | III-B | State the codebook, synchronization and full-decoding assumptions behind Eqs. (4)–(6). Note σ² = 1. Make the use of a_rk in Eq. (3) consistent. | Correct. |
| 1.7 | V-D, V-E | Sign convention of B = QW − ∇J (J is a utility); give Q_i and ∇J explicitly; explain the soft-min and DF-price interaction, and bisection plus projection. | Correct. The code minimizes L = −J. |
| 1.8 | V-C, V-E, Alg. 1 | In-layer vs. per-epoch duals are unclear. Duals persist across fresh UE drops in training. | Correct. Confirmed: `self.coordination` is never reset on a new training drop (deployment resets it per drop). |
| 1.9 | V-B | The τ concavity claim holds only for the pure PF utility; state the search and its resolution. | Correct. The code uses a 61-point grid on the pure PF utility. |
| 1.10 | V–VII, Table I | Missing: α, β, κ, η_Q/η_D, λ_O, the sigmoid slope, the CSI-error scaling, network sizes and parameter count, warm-up, the knapsack objective, the signature, the kernel, and baseline iteration limits. | Correct. |
| 1.11 | VI | The exact bound at fixed τ only bounds methods held to the same τ; state the value. | Correct. τ = 0.5, and all methods were already held to it (`SAME_PROBLEM`); the text did not say so. |
| 1.12 | VIII-D | The aging comparison is asymmetric (robustly trained vs. certainty-equivalent); trained once or per ρ?; tail and outage vs. the no-relay benchmark. | Correct. Models are retrained per ρ. |
| 1.13 | VIII-F | Small network: transferred or trained from scratch? Mechanism of the starvation? | Correct. Trained from scratch; the text did not say so. |
| 1.14 | VIII-E | Why validation starts at 6.8 and plateaus below the 7.88 evaluation mean. | Correct. |
| 1.15 | II | Related work misses multi-antenna and distributed WMMSE unfolding, Team MMSE, and relay-assisted cell-free work. | Correct. |
| 1.16 | I | Citations are not numbered in order of first appearance. | Correct. |
| 1.17 | Minor | ε₀ units, undefined N, "4K + RK real numbers", the 224 formula, F_max vs. F_max T_e, Algorithm 1 references, bold duals, d in metres. | Correct. |
| 1.18 | Fig. 14 | The legend reads "F_mix"; stray "(b)" in the legend. | "F_mix" is **not** in our figure (the image says F_max; likely a reading error by the tool). The stray "(b)" is real. |
| 1.19 | Figs. 2, 6, 13, 14 | Plain-text symbols (R_min, L_max, λ_E, F_max) instead of math notation. | Correct (cosmetic). |
| 1.20 | Weakness 2 | Compare against external unfolded or distributed WMMSE baselines (Schynol & Pesavento; Yang et al. 2026; Yi et al. 2025). | Valid suggestion; see "not acted on" below. |

### Modifications made (→ Revision 1)

No experiment was rerun, and no reported result changed. The new numbers come from paired analysis of the saved
final results. The figures were redrawn from the saved results (`--plot-only`); their CSV data are byte-identical.

| # | Addresses | What was changed | Rerun? |
|---|---|---|---|
| R1.1 | 1.1 | New **Table III** decomposes the +11.3 over Local-WMMSE, paired over 100 drops: coordination ζ_f **+7.30 ± 0.35**, learned parameters **+0.89 ± 0.16** (90/100), slow timescale **+3.12 ± 0.27** (99/100). New Section VIII-B. Rewrote the abstract, VIII-A, VIII-E, the limitations and the conclusion: the learned parameters are the smallest contribution. Removed the false "best non-learned baseline" claim; added +10.3 over RZF. The same overclaim was corrected in `REPORT_FOR_SIR.md` and `HANDOVER.md`. | No (saved results) |
| R1.2 | 1.2 | Split the notation into ζ_s(t) ∈ ℝ^{4K+RK} (slow model input) and ζ_f^(ℓ) (352 complex values per layer, listed item by item). Algorithm 1 now shows the per-layer exchange and the deployment phase. | No |
| R1.3 | 1.3 | Removed "50×" everywhere. VIII-H now lists three caveats: solver difference (CVXPY), single process without exchange latency, and timing noise (identical RAP variants took 0.32–1.3 s in one run). Most RAP time is the warm-up (untrained layers: 0.09 s). We claim only an order of magnitude. | No |
| R1.4 | 1.4 | V-F states the cross-node backpropagation as simplification (i) and notes that it also applies to no-FL. VIII-J gives it as the second reason for the null result. | No |
| R1.5 | 1.5 | Section IV split into a per-block P0 (with the unit-norm combiner, and the outage measure defined over p(H\|Ĥ)) and a slow-timescale subsection where Eq. (9) gates FL traffic only, with headroom [F T_e − D_data]_+. Defined b_r = max_k a_rk, and stated that P_tot is the power drawn while transmitting, not duty-cycle weighted. | No |
| R1.6 | 1.6 | III-B states assumptions (i) independent codebooks, (ii) phase-synchronized joint transmitters, (iii) full decoding at every assigned relay. Set σ² = σ_r² = 1, and removed a_rk from Eq. (3). | No |
| R1.7 | 1.7 | V-D defines the loss L = −J and the soft-min. It gives the link weights, ∇L (Eq. 11) and Q_m (Eq. 12) explicitly, the bisection (detached) and why the projection is still needed, and the relaxation convention (neutral = η, δ, λ_reg all 1). V-E explains the soft-min/ν interaction and that the stop-gradient does not stop the APs from shifting power towards the relay. | No |
| R1.8 | 1.8 | V-E: the in-layer prices start from the slow prices, are updated per layer and discarded after the block; the slow prices are updated per epoch. V-F (ii) discloses that ζ_s carries across training drops and is reset per drop at deployment. Also added to `RAP_FEDWMMSE.md`. | No |
| R1.9 | 1.9 | V-B: 61-point grid (step 0.01) on the pure PF utility at measured SINRs, with no outage term. | No |
| R1.10 | 1.10 | All values added (Table I now has algorithm parameters), plus the association formulas, the network architecture (backbone 8,352 parameters, 33.4 kB; adapter and head local), upload/download sizes (6,736 B / 33,408 B, so B_FL admits at most 5 uploads), the knapsack objective, the signature and the kernel. Also: E ~ CN(0, β) per link; J is on the estimate and the draws enter the outage term only (**this corrects a wrong sentence in the draft**, which said J was averaged over the draws). | No |
| R1.11 | 1.11 | VI and the Fig. 10 caption state τ = 0.5 for the exact method and for every method in that experiment. | No |
| R1.12 | 1.12 | VIII-D states the asymmetry and per-ρ retraining. Added the paired PF vs. no relay (−0.29 ± 0.54, −0.48 ± 0.49, −0.49 ± 0.44 at ρ = 0.8, 0.85, 0.9), and RAP's better tail and outage at every ρ. | No (saved results) |
| R1.13 | 1.13 | VIII-F: trained from scratch with the default hyperparameters; the mechanism is given as an unverified hypothesis. III-C: a separate model for every topology, no transfer. | No |
| R1.14 | 1.14 | VIII-E: τ is already adapted to 0.75–0.8 after epoch 1, and validation uses 4 drops per deployment. | No |
| R1.15 | 1.15 | Related work expanded with 7 papers, each verified to exist with exact bibliographic data: Hu et al. (TWC 2021), Pellaco et al. (OJ-COMS 2022), Schynol & Pesavento (JSAC 2023), Miretti et al. (TWC 2022), Nguyen et al. (TWC 2023), Vu et al. (EUSIPCO 2023), Wang et al. (OJ-COMS 2023). Positioning rewritten. | No |
| R1.16 | 1.16 | References renumbered in order of first appearance; the relay references are now cited in the introduction. | No |
| R1.17 | 1.17 | All minor items fixed. Also corrected "2.8 of 12 nodes" to "2.8 of **10** nodes" (8 APs + 2 relays), an error of ours the review did not flag. | No |
| R1.18 | 1.18, 1.19 | Figure labels now use math notation (R_min, L_max, λ_E, F_max, P_tot); the G14 legend now reads "… exceeds F_max (panel b)". Redrawn with `--plot-only`. | No (plot only) |

### Comments not acted on

| # | Comment | Why not |
|---|---|---|
| 1.20 | Add external unfolded or distributed WMMSE baselines (e.g. Schynol & Pesavento's GCN-WMMSE). | Implementing and tuning another learned method fairly is a separate project, beyond the baselines Sir specified (Centralized WMMSE-SCA/AO, Local-WMMSE, exact MINLP). We now cite these works, position against them, and state in VIII-H that the speed comparison is against a solver-based benchmark. We did not cite Yi et al. 2025 or Yang et al. 2026, because we could not verify that they exist. |
| 1.12 (part) | A robust centralized benchmark for the aging comparison. | Not implemented. The text now states that the comparison is against a certainty-equivalent benchmark and lists it as limitation (viii). |
| 1.8 (part) | Reset the duals on every fresh training drop. | Changing it would require retraining and rerunning all experiments. It does not affect evaluation (deployment resets ζ_s per drop), so it is disclosed instead (V-F (ii), limitation (vii)). |

---

## Round 2

**Uploaded:** [date] — Revision 1 (`main.pdf`, 11 pages, commit `[hash]`)

### Review comments received

| # | Section / location | Comment |
|---|---|---|
| 2.1 | | |

### Modifications made (→ Revision 2)

| # | Addresses | What was changed | Required rerun? |
|---|---|---|---|
| R2.1 | 2.1 | | yes / no |

---

## Round 3 (recommended)

**Uploaded:** [date] — Revision 2 (`main.pdf`, commit `[hash]`)

### Review comments received

| # | Section / location | Comment |
|---|---|---|
| 3.1 | | |

### Modifications made (→ final)

| # | Addresses | What was changed | Required rerun? |
|---|---|---|---|
| R3.1 | 3.1 | | yes / no |
