# Deep Review revision log — Group 12

This log records every Google Deep Review round, as Sir asked: the review comments from each upload, and the
changes made in each revision. Three complete rounds were done. The submitted PDF version is
`Revision_Report_Rounds_1-3.pdf`.

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

**Uploaded:** 12 November 2026 — Revision 1 (`main.pdf`, 11 pages, commit `3b352b2`).
**Full review text:** [`deepreview_round2.md`](deepreview_round2.md).

The review again found every number consistent with the tables and figures, and all references real (18 of
18). Its verification summaries confirmed the system model, the formulation, the layer equations, the parameter
counts and the overhead arithmetic. No reported result was found to be wrong. The comments asked for fairer
ablations, verified diagnoses and more detail. Several required new experiments; they were run on the exact
drops of the final run, and each first re-scored a configuration from the final run, reproducing it exactly,
as a determinism check.

### Review comments received

| # | Section | Comment (summary) | Our check |
|---|---|---|---|
| 2.1 | Weakness 1, VIII-B | The learning adds little. Evaluate the untrained layers *with* the training-free slow timescale, and the undamped classical update (δ = 1), to isolate the learned contribution; damping may inflate it. | Valid request. Ran both (R2.1). The damping concern turned out **wrong**: undamped is worse. |
| 2.2 | Weakness 2, VIII-J | No-FL also gets network-wide gradients and unlimited data, so it is not a local baseline; the "45% less than FedAvg" claim is confounded when no-FL needs 0 MB. Suggests limited-data / detached-gradient experiments. | Correct on the claim. Experiment not run (see below). |
| 2.3 | Weakness 3, VIII-E | Aging comparison asymmetric (per-ρ retraining vs. certainty-equivalent); suggests a robust centralized baseline or a single model evaluated zero-shot; Team MMSE not compared. | Valid. Ran the zero-shot test (R2.3). |
| 2.4 | Weakness 4, III-C | P_tot ignores the phase durations τ and 1 − τ; report time-averaged power. | Correct. |
| 2.5 | Weakness 5, VIII-G | The starvation cause is an unverified hypothesis; log the predicted parameters on the starved drops; give the u, equalizer and MSE update equations. | Valid. Diagnosed (R2.5). |
| 2.6 | Abstract | "Overtakes the centralized benchmark under stale CSI" lacks the "certainty-equivalent" qualifier. | Correct. |
| 2.7 | I | The intro bullet omits the per-layer fast exchange and its overhead. | Correct. |
| 2.8 | II | A sentence starts with a bare "[8]"; Team MMSE should be contrasted. | Correct. |
| 2.9 | IV-A | Baselines and the bound solve unconstrained PF, while RAP solves a QoS-penalized relaxation. | Correct. |
| 2.10 | IV-A | Write r_k(Ĥ); define H; mark A and τ as slow. | Correct. |
| 2.11 | IV-B, III-B | User data replicated across the cluster; uncapped rates over the fronthaul are a modeling choice; multi-relay synchronization is idealized. | Correct. |
| 2.12 | V-C, V-D | Missing: initial relayed-beam direction, input-size formulas, head output count, and the fact that η and δ collapse into one effective step. | Correct. Confirmed in code: effective step = ηδ. |
| 2.13 | V-C, V-G | Interference definition in ζ_s, signature entries, gradient-norm timing, age, Δθ, error feedback, aggregation over the selected set. | Correct. |
| 2.14 | VIII-H, Table II | Times mix two runs; warm-up not separated from single-block inference; compare against 100 closed-form iterations. | Correct. |
| 2.15 | Minor | Plain-text "capacity"; units wording; ‖u‖ = 1 only for a_rk = 1; A^AR sign form; ρ overloaded for roles; Algorithm 1 order and the warm-up note; c_mk sum; β not introduced; Table I subscripts; 2- vs 3-decimal gaps; legend labels "(no FL)" and "(converged)". | Correct. |
| 2.16 | VIII-K, Reprod. | Topology-specific models missing from the limitations; no pinned environment. | Correct. |

### Modifications made (→ Revision 2)

| # | Addresses | What was changed | Rerun? |
|---|---|---|---|
| R2.1 | 2.1 | New ablations `unfolded_wmmse_slow` and `classical_wmmse_layers/_slow` (`diagnostics/round2.py`). Table III now gives the decomposition in **both orders**: learning **+0.89 ± 0.16** (A) or **+0.61 ± 0.20** (B, 87/100); slow timescale **+3.12** or **+3.40 ± 0.17**. Undamped classical WMMSE is **worse**, 1.68 at L = 6 and −2.65 at L = 100 (it oscillates), so the damped baseline is the stronger one. 100 damped untrained layers (5.54) **beat** 6 trained ones (paired −0.77 ± 0.39): what learning buys is fewer layers. Abstract, VIII-B, VIII-F, limitations and conclusion updated. | New diagnostics only |
| R2.2 | 2.2 | Rewrote VIII-F and VIII-J, and the abstract: no-FL matches both FL schemes with no FL traffic, so Steps 12–15 have no demonstrated benefit in this regime; the 45% figure is only personalized vs. FedAvg. | No |
| R2.3 | 2.3 | Zero-shot test: the default model (ρ = 0.98) evaluated at every ρ without retraining. It loses only 0.05–0.33 to the retrained models and **still beats the centralized method** at ρ = 0.8, 0.85, 0.9 (+0.69 ± 0.44, +0.61 ± 0.40, +0.41 ± 0.35). Added to VIII-E. | New diagnostic only |
| R2.4 | 2.4 | Added the time-averaged power, Eq. (8), and recorded it by **rerunning the energy experiment**. The rerun reproduced every PF and P_tot value exactly. Finding: time-averaged, the relay methods use **less** energy than the no-relay benchmark (17.4 / 17.7 W vs 20.9 W), the opposite of P_tot. Added to VIII-I, the abstract and limitation (vi). | Yes (energy only; identical results) |
| R2.5 | 2.5 | Starvation diagnosed by re-solving with one change at a time: capping η ≤ 1 cuts starved drops 14 → 8; equal shares → 10; retraining without the QoS machinery → 12. The predicted parameters on starved and other drops are alike. Conclusion: no single trigger; the larger learned steps are the main contributor. Starvation defined as a rate below 0.02 bit/s/Hz. The u_rk, equalizer, MSE and weight equations are written out, with ε′ = 10⁻⁹. | New diagnostic only |
| R2.6–R2.8 | 2.6–2.8 | Abstract qualified ("with robust training … certainty-equivalent"). The intro bullet now states the per-layer exchange and its overhead. "Chowdhury et al. [8]". Team MMSE contrasted and stated as not implemented. | No |
| R2.9–R2.11 | 2.9–2.11 | IV-A states the problem mismatch (QoS-penalized vs. pure PF), r_k(Ĥ), the H definition, and that A and τ are slow. IV-B notes the replication and that uncapped rates are a modeling choice. III-B: multi-relay coherence is an idealized upper bound. | No |
| R2.12–R2.13 | 2.12–2.13 | Initial beams (G^H times the dominant left singular vector; the direct link is not used for a relayed stream; full budget), input formulas, 2L + 1 + K head outputs with one λ_reg, the η·δ product, the ζ_s interference definition, signature entries, age, gradient timing, Δθ, error feedback, aggregation over S; κ indexed by o(i). | No |
| R2.14 | 2.14 | New clean timing run (`diagnostics/runtime.py`): every method on the same 10 drops, one process, one thread, idle machine; warm-up timed separately from single-block inference; 100 closed-form layers timed too. Table II and VIII-H use these numbers. | New timing run |
| R2.15 | 2.15 | All minor items fixed. Figures redrawn: "Local-WMMSE" without "(no FL)", "(converged)" only for iterative methods. | No (plot only) |
| R2.16 | 2.16 | Limitations (xi) topology-specific models and (xii) problem mismatch added. New `requirements-lock.txt` pins the exact versions; the Reproducibility section names it and the diagnostics command. | No |

### Comments not acted on

| # | Comment | Why not |
|---|---|---|
| 2.2 | Limited-data / detached-gradient FL experiment. | By the team's decision we did not run experiments aimed at changing the federation verdict. The paper states the negative result, its two likely causes, and what a fair test would need. |
| 2.3 (part) | Robust (sample-average) centralized baseline; Team MMSE baseline. | Each is a new optimizer beyond the baselines Sir specified. We added the zero-shot test instead, contrasted Team MMSE in Section II, and listed both in limitation (viii). |
| 2.1 (part) | A separate ablation for the in-layer prices alone. | The in-layer price updates run in both the trained and untrained layers (now stated in VI), so the learning comparison is not confounded by them. |

---

## Round 3

**Uploaded:** 13 November 2026 — Revision 2 (`main.pdf`, 13 pages, commit `af0d741`).
**Full review text:** [`deepreview_round3.md`](deepreview_round3.md).

The review again found every number consistent (it re-derived Table III in both orders, the dimension and byte
counts, and the overhead arithmetic), and all 18 references real. No reported result was wrong. It asked for
finer ablations and many precise definitions. Every factual claim it made about our implementation was checked
against the code first; two were wrong (see 3.15, 3.16). New diagnostics (`diagnostics/round3.py`) ran on the
exact drops of the final run, and re-scored the published RAP-FedWMMSE and untrained results exactly as a
determinism check. **Two of them corrected claims in our paper and in the report for Sir (R3.3, R3.4).**

### Review comments received

| # | Section | Comment (summary) | Our check |
|---|---|---|---|
| 3.1 | Weakness 1, VIII-B | The "slow timescale" step bundles τ adaptation, warmed-up prices and priced association; split it. | Valid. Split (R3.1). |
| 3.2 | VIII-A | The uncoordinated baselines are held to τ = 0.5, which costs them about 3. | Correct (Fig. 7). |
| 3.3 | Weakness 3, VIII-B/E | Measure what the QoS and robustness machinery costs at perfect CSI (pure-PF ablation). | Valid. Ran it (R3.3). |
| 3.4 | VIII-E | Is the stale-CSI margin due to the outage surrogate? Is the slow timescale active in Fig. 8? Why is ρ = 0.99 far below perfect CSI? | Valid. Ran it (R3.4). Slow timescale **was** active (τ = 0.8). |
| 3.5 | VIII-C | Is the steeper decline at K = 12 the fixed depth? ζ_f grows as O(RK²). | Valid. Ran it (R3.5). |
| 3.6 | VIII-F | The default network was explained with the *small* network's learned parameters. | **Correct, our slip.** Measured the default network's (R3.6). |
| 3.7 | VIII-G | Starvation: λ_reg not examined; is the warm-up active; single-factor ablation only. | Valid (R3.7). |
| 3.8 | VI, VIII-G | Separate Step 1's association gap from the beam gap in Fig. 10; Section VI wording. | Valid. Ran SCIP with the association fixed (R3.8). |
| 3.9 | III-C, IV | P_tot over-penalizes relay power by τ/(1 − τ); τ search is blind to energy. | Correct; stated (R3.9). |
| 3.10 | III-B | No selection-DF fallback; multi-relay phase alignment comes from the ζ_f exchange. | Correct; stated. |
| 3.11 | IV-A/B | Name the unconstrained problem (P0-PF); r_k(H) with beams fixed; ζ_f and ζ_s not counted in the fronthaul budget. | Correct; stated. |
| 3.12 | V-D | One-layer envelope gradient vs. multi-layer backpropagation; mask before bisection; radial clipping; λ_reg inactive when μ = 0; Jacobi-type parallel step. | Correct. Code confirmed: masked before bisection, radial clipping with gradients. |
| 3.13 | V-E, V-F, Alg. 1 | Price timing and detachment; ν side effects; user indices across drops; the warm-up does not redraw users. | Correct. Code confirmed: price updates use detached rates; J_b uses the layer-L prices. |
| 3.14 | V-G | θ_rec and e_i update rules; the knapsack reduces to top-5; FL bytes up + down; signature statistics are global; adapter alignment; B_FL in bytes; 2-entry one-hot. | Correct. Code confirmed for each. |
| 3.15 | Table II | "MRT 5th-percentile 0.20 may be a duplicate of RZF." | **Not correct**: MRT 0.1965, RZF 0.1959; both round to 0.20. |
| 3.16 | Fig. 8 | Reads the ρ = 0.99 values as about 5.2–5.5 and 6.0. | **Misread**: they are 6.66 and 7.39. The underlying question (why below 7.88/9.88) is answered in R3.4. |
| 3.17 | Minor | "Every method peaks at K = 8"; rounding 0.12 + 0.03 ≠ 0.16; Table III levels; noise PSD; Ī wording; backbone ReLU; φ for shares; link weights real; "nats"; masks in (9d); C = [c_mk]; b_r note; "python" prefix; Fig. 14 FedAvg not bound by B_FL; conclusion vs. the no-relay benchmark; Local-WMMSE called unregularized up front. | Correct (the K-peak sentence was wrong for Local and the heuristics). |

### Modifications made (→ Revision 3)

| # | Addresses | What was changed | Rerun? |
|---|---|---|---|
| R3.1 | 3.1 | New ablation `unfolded_wmmse_tau_only`. The slow-timescale gain is **all τ**: τ alone +3.40 ± 0.17, and priced association + prices + warm-up add **−0.002 ± 0.047** (outage 4.5% vs 4.6%). Abstract, VIII-B, Table III and the conclusion now say so. | New diagnostic |
| R3.2 | 3.2 | VIII-A: at their best fixed τ = 0.8, Local-WMMSE reaches −0.31 and RZF 0.34; RAP still leads by +8.2 ± 0.4 and +7.5 ± 0.3 (from the saved τ sweep). Local-WMMSE is called "unregularized" in the abstract and VIII-A, with the Team MMSE caveat. | No (saved results) |
| R3.3 | 3.3 | Pure-PF RAP-FedWMMSE (no outage term, no prices): 7.79 vs 7.88 (paired −0.09 ± 0.23, n.s.), outage 3.5% vs 4.0%. **The QoS machinery has no measurable effect** at perfect CSI. New paragraph in VIII-B, limitation (xii), report for Sir. | New diagnostic |
| R3.4 | 3.4 | **Corrected our stale-CSI explanation.** (a) A default model trained *without* the outage term does as well zero-shot (+0.11, +0.04, +0.09 at ρ = 0.8, 0.85, 0.9, n.s.). (b) The **untrained** layers with the slow timescale beat the trained model (+0.39 ± 0.22, +0.40 ± 0.17, +0.21 ± 0.13) and the centralized method by more (+1.40, +1.18, +0.83). The advantage is a property of the damped distributed structure, not of learning or robust training. Rewrote VIII-E, the abstract, the Fig. 8 caption, limitation (viii), the conclusion, and finding 3 of the report for Sir (which had credited Step 10). Also stated that Fig. 8 runs the slow timescale, and why ρ = 0.99 is far below perfect CSI. | New diagnostics |
| R3.5 | 3.5 | At K = 12, 100 untrained layers with the slow timescale reach 6.85 ± 0.56, +2.05 ± 0.23 above RAP-FedWMMSE and 1.17 ± 0.14 below centralized: the steeper decline is mainly the fixed depth. Added to VIII-C with the O(RK²) growth (4,320 values per block at K = 12 vs 304 for raw CSI); limitation (xiii). | New diagnostic |
| R3.6 | 3.6 | Replaced the small-network numbers with the default network's learned parameters: AP step 1.18 (1.24–1.34 in layers 1–4, about 1.0 in 5–6), relaxation 0.59 (effective step 0.70), λ_reg 0.61. | New diagnostic |
| R3.7 | 3.7 | λ_reg is lower on the starved drops (0.35 vs 0.58), which matters because Q_i is a scalar when N = 1. The untrained layers *with* the warm-up starve 4 drops (vs 3), so the prices alone do not cause it. Starvation comes from the learned parameters together. | New diagnostic (+ saved probe) |
| R3.8 | 3.8 | SCIP with the association fixed to Step 1 (88 certified, 12 at the time limit): Step 1 costs 0.243 on average (more than 0.01 on 29 drops). Gaps to this bound: centralized 0.105, untrained 0.555, RAP 0.877. About a fifth of RAP's gap is the relay choice. Section VI wording fixed. | New SCIP run |
| R3.9 | 3.9–3.14 | Stated: the τ/(1 − τ) over-penalization; selection-DF not modeled; phase alignment via ζ_f; P0-PF; r_k(H) with fixed beams; ζ_f/ζ_s outside the fronthaul budget (and that ζ_f would exceed 10 Mbit/s over wires); envelope vs. full backpropagation; masking, clipping, λ_reg at μ = 0, Jacobi steps; price timing/detachment; ν side effects; unmatched user indices; warm-up without redraw; θ_rec/e_i rules; top-5 knapsack; up + down bytes; global signature statistics; independent adapter initialization. | No |
| R3.10 | 3.17 | All minor items fixed, including the K-peak sentence, the 0.157 = 0.124 + 0.033 s rounding, Table III intermediate levels, and −174 dBm/Hz in Table I. | No |

### Comments not acted on

| # | Comment | Why not |
|---|---|---|
| 3.3 (part) | Add the soft QoS penalty to the centralized benchmark. | The pure-PF ablation (R3.3) answered the underlying question: the penalty has no measurable effect. |
| 3.4 (part) | A diagonally loaded (statistically robust) centralized benchmark. | A new baseline beyond Sir's list; the finding (R3.4) is now that even untrained layers beat the certainty-equivalent one, which we state with that caveat. |
| 3.9 | Use the τ-weighted energy in the objective. | It would change Sir's objective and require rerunning the energy study; we report the time-averaged power and state the bias. |
| 3.10 | Evaluate selection DF. | A change to the system model; stated as a limitation of assumption (iii). |
| Weakness 2 | Limited-data / detached-gradient FL. | By the team's decision we did not chase the federation verdict; stated as future work. |
