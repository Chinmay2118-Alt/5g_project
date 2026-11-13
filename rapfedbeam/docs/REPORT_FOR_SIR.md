# Message to Sir

Subject: RAP-FedWMMSE: final implementation, baselines and 14 figures (Group 12)

Respected Sir,

We have implemented the RAP-FedWMMSE algorithm you sent, all the baselines you asked for, and the exact method. We
have compared them in 14 figures. All results use 100 network drops over 10 AP/relay deployments, with
95% confidence intervals. Every run completed without a failed drop. The code, figures and documentation are on
the `kushagra` branch of our repository.

## What is implemented

- **RAP-FedWMMSE, Steps 1–15 of your document.** This includes:
  - **Step 1:** relay association by capacitated matching, with cooperation order L_max ≥ 1;
  - **Step 2:** τ adaptation per epoch;
  - **Steps 3–4:** role-specific adapters, a personalized backbone, and the full ζ(t) (QoS prices, DF prices,
    measured interference, rate statistics);
  - **Steps 5–8:** unfolded WMMSE layers with MMSE relay combiners, the DF soft-min, learned step, damping and
    regularization, and exact power projection;
  - **Steps 9–10:** primal–dual QoS/DF correction and the robust CSI-outage loss;
  - **Step 11:** self-supervised training with no dataset;
  - **Steps 12–15:** event-triggered, knapsack-selected, Top-k-compressed, topology- and role-aware personalized
    aggregation.

  The FL exchange also respects the fronthaul and control-link limits of our problem formulation (Sec. XXXI).
- **Baselines:**
  - Centralized WMMSE-SCA/AO;
  - Local-WMMSE / No-FL;
  - Centralized WMMSE without relays;
  - RZF and MRT with equal power;
  - three ablations: RAP-FedWMMSE with FedAvg, with no federation, and the same layers untrained.
- **Exact method:** a global MINLP solved by spatial branch-and-bound (SCIP) on a small network (2 APs, 1 relay,
  2 UEs). Beams, combiners and the binary association are all optimized, and SCIP certifies an upper bound.

## Main results (default network: 8 APs, 2 relays, 8 UEs, 30 dBm)

| Method | PF utility | Mean rate (bit/s/Hz) | 5th-percentile rate | Outage at 0.5 bit/s/Hz | Time per drop |
|---|---|---|---|---|---|
| Centralized WMMSE-SCA/AO | 9.88 ± 0.40 | 4.57 | 0.67 | 3.0% | 5.05 s |
| Centralized WMMSE, no relay | 9.80 ± 0.53 | 5.41 | 0.18 | 10.9% | 1.46 s |
| **RAP-FedWMMSE (proposed)** | **7.88 ± 0.51** | **3.49** | **0.57** | **4.0%** | **0.16 s** (0.03 s per block) |
| Unfolded WMMSE, untrained | 3.87 ± 0.46 | 2.06 | 0.36 | 8.1% | 0.03 s |
| Local-WMMSE / No-FL | −3.43 ± 0.43 | 0.89 | 0.20 | 34.0% | 0.32 s |
| RZF / MRT, equal power | −2.43 / −5.60 | 1.13 / 0.58 | 0.20 / 0.20 | 38.5% / 48.5% | ≤ 0.001 s |

Times are from one clean run: every method on the same 10 drops, one process, one thread, on an idle machine.

## Main findings

1. **Among methods without central CSI, RAP-FedWMMSE is the best:** +11.3 ± 0.4 PF utility over Local-WMMSE,
   +10.3 over RZF, and +4.0 ± 0.2 over the same layers untrained.
   - It reaches 76% of the centralized mean rate, with an outage of 4.0% against 3.0%.
   - It uses only local CSI plus the coordination it exchanges (ζ).
   - Its computation is about an order of magnitude faster than the centralized method, and does not grow with
     the number of UEs (G11). Part of this gap is that the centralized method calls a general conic solver
     (CVXPY), and the timings do not include the latency of exchanging ζ at every layer.
2. **Where the gain comes from** (paired over the same 100 drops; a correction to our earlier draft, which
   credited the whole +4.0 to learning):
   - exchanging ζ between nodes (untrained layers against Local-WMMSE): **+7.3**;
   - the learned layer parameters: **+0.6 to +0.9**, depending on whether Steps 1–2 are already on;
   - the slow timescale of Steps 1–2: **+3.1 to +3.4**, all of it from Step 2's τ adaptation; Step 1's
     QoS-priced association and the warmed-up prices add nothing measurable (−0.002 ± 0.047).

   So the learning helps, but modestly. Most of the gain comes from the coordination and from Steps 1–2.
   100 untrained layers even beat 6 trained ones: what the learning buys is fewer layers. Undamped classical
   WMMSE is much worse than our untrained baseline, because all nodes update in parallel and it oscillates.
3. **Under stale CSI, RAP-FedWMMSE beats the centralized method (G08)** for ρ ≤ 0.9, by +1.01 ± 0.40 at
   ρ = 0.8. A single model trained once, never retrained, also beats it. **Correction to our earlier draft:**
   this is *not* due to the robust training of Step 10, nor to the learning.
   - A model trained without the outage term does as well.
   - The untrained layers do even better (+0.4 over the trained model at ρ = 0.8).

   The advantage comes from the method's structure: a few damped distributed WMMSE steps are less sensitive
   to channel errors than a converged centralized solution, which treats the stale estimate as exact.
4. **The energy term works (G13).** With λ_E = 0.1, RAP-FedWMMSE uses 22% less power (23.9 → 18.7 W) for a PF loss of
   0.14. Weighted by how long each node actually transmits (time-averaged power), relay-assisted operation uses
   *less* energy than the no-relay network (17.4 W against 20.9 W).
5. **Relays mainly protect the weakest users.**
   - At the default 25% blockage, they triple the 5th-percentile rate (0.57 with RAP-FedWMMSE against 0.18 without
     relays) and cut outage from 10.9% to 3–4% (G01–G03).
   - Their PF-utility gain grows with blockage (G05): +0.97 ± 0.43 at 75% blocked UEs.
   - Without blockage, a network without relays is better, because it uses the whole frame (τ = 1).
6. **Exact comparison (G10).**
   - SCIP reached a 1% certified gap on 94 of 100 drops.
   - Centralized WMMSE is 0.35 below the bound, RAP-FedWMMSE 1.12, Local-WMMSE 1.64.
   - No method exceeded the bound in 600 records.

## Results that do not support our hypotheses

- **Federation does not measurably improve PF utility** in our setting. Personalized aggregation, FedAvg and no
  federation are within noise: −0.06 ± 0.19 between personalized and no FL. Two simulation choices favor no FL:
  every node trains on unlimited fresh channels, and training back-propagates the global objective through all
  nodes, so even the no-FL nodes learn from network-wide gradients. Personalized aggregation reaches the
  same utility as FedAvg with 45% less communication (11 MB against 20 MB, G12). With a tight fronthaul, FL
  participation drops sharply, while utility hardly changes (G14).
- **On the small single-antenna network, trained RAP-FedWMMSE is worse than untrained:** 1.12 against 0.80 below
  the bound. On 14 of 100 drops it nearly starves one UE. This does not happen on the default network.
- **The QoS machinery (Steps 9–10) has no measurable effect.** Trained as a pure PF optimizer (no outage
  term, no prices), RAP-FedWMMSE reaches 7.79 against 7.88 (within noise), with an outage of 3.5% against 4.0%.
- **Six layers are too few at higher load.** At 12 UEs, 100 untrained layers reach 6.85 against 4.80 for
  RAP-FedWMMSE.
- **Cooperative relaying (L_max = 2) slightly lowers PF utility (G06),** because every assisting relay must decode
  the stream.
- **Signalling:** ζ per unfolded layer (352 complex values) exceeds the raw CSI (224) at our antenna counts.
  It does not grow with the number of antennas, so this would reverse for larger arrays.

## Points where we had to interpret your algorithm — please confirm

1. **Step 9:** with r_k = min(C_D, C_AR), r_k − C_AR ≤ 0 always holds, so ν_DF could never increase. We price
   C_D − C_AR instead, which is positive exactly when the relay hop is the bottleneck.
2. **Step 7:** instead of W − η∇J, we use the regularized WMMSE step W + η((Q + λI)⁻¹B − W). It keeps your learned
   η, δ and λ_reg, and with η = 1 and λ_reg = 1 it is exactly classical WMMSE. A plain gradient step plateaued
   far lower in our tests.
3. **Step 11:** in simulation, each node's gradient of J is computed by back-propagation through all nodes' layers.
   A real network would need a backward exchange of ζ for this.
4. **Step 1:** the weights α₁–α₄ and a rule that a relay must beat the UE's direct link are our choices. They
   were calibrated on deployments never used for evaluation.
5. **Training data:** each node trains on fresh UE drops on its own deployment every epoch. With a single
   slowly moving drop, the model overfit it.
6. **Fronthaul:** the limit restricts FL traffic only. User data is not throttled, and an AP whose user data
   alone exceeds F_max is reported as overloaded.

We also checked the paper with Google Deep Review in three rounds. None found a wrong number. Round 1 found the
attribution error corrected in finding 2. Rounds 2 and 3 asked for further ablations, which showed what is
corrected in findings 2 and 3 and listed above. The full log is in `paper/REVISION_LOG.md`.

## Assumptions

- **Channels:** 3GPP UMi NLOS path loss, 20 MHz bandwidth, 7 dB noise figure, 1 W per AP and relay, 4 dB shadowing,
  a 500 m square, and 2 of the 8 UEs (25%) blocked by an extra 25 dB.
- **CSI:** perfect by default. Channel aging appears only in G08.
- **Network size:** 2 antennas per node. This is small for massive MIMO, but it keeps the centralized and exact
  benchmarks tractable.
- **Coherence:** each drop is evaluated as one coherence block. τ ∈ [0.2, 0.8] for relay networks.

We would be grateful for your feedback, especially on the six points above.

Sincerely,
Group 12
