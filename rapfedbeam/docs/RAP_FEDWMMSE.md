# RAP-FedWMMSE: how Sir's algorithm is implemented

This document maps each step of Sir's RAP-FedWMMSE document (Steps 1–15 and the pseudocode) to the code. It also
lists every place where the text left a choice open or could not be followed literally, with the reason for
what was done.

The main code is in [`train/rap_fedwmmse.py`](../train/rap_fedwmmse.py) (epoch loop, models, federation)
and [`train/unfolded_wmmse.py`](../train/unfolded_wmmse.py) (unfolded layers). The tests are in
[`tests/test_rap_fedwmmse.py`](../tests/test_rap_fedwmmse.py).

## Step-by-step map

| Sir's step | Where | What it does |
|---|---|---|
| 1. Relay association | `sim/association.py` | Scores Γ_rk = α₁β̄ᴿᵁ + α₂β̄ᴬᴿ − α₃Ī + α₄λᵠ, then solves the capacitated matching exactly as a binary program (SciPy `milp`), with Σ_r a_rk ≤ L_max and Σ_k a_rk ≤ C_r^max. L_max > 1 (cooperative multi-relay forwarding) is supported throughout |
| 2. Phase duration | `RapFedWMMSETrainer._best_tau` | 1-D search over [τ_min, τ_max] on the epoch utility |
| 3. Local state, adapters, backbone | `ap_features`, `relay_features`, `SlowCoordination`, `NodeModel` | x_m = {ĥ_mk, Ĝ_mr, β_m, ζ}, x_r = {Ĝ_mr, ĝ_rk, β_r, ζ}, where ζ holds the QoS prices, DF prices, measured aggregate interference and previous rate statistics → role adapter → personalized backbone θ_i |
| 4. Model-driven initialization | `NodeModel.forward`, `initial_beams` | The head predicts η⁽ˡ⁾, δ⁽ˡ⁾, λ_reg and initial power shares; W⁽⁰⁾, V⁽⁰⁾ are MRT directions with those shares |
| 5. Unfolded layers: combiners, SINRs, DF rate | `relay_combiners`, `link_rates`, `user_rates` | Exactly the paper's formulas; soft-min with κ inside the layers |
| 6. WMMSE variables | `mmse_equalizer`, `coordination` | c_k, ε_k, ρ_k = 1/(ε_k + ϵ) for Phase I, Phase II and relay decoding |
| 7. Learned regularized update | `ap_gradient`, `relay_gradient`, `_regularized_step` | Gradient of J from local CSI + ζ, a curvature-scaled step, damping |
| 8. Feasibility projection | `project` | Frobenius scaling per node after masking c_mk and a_rk |
| 9. Primal-dual correction | `update_duals`, `augmented_objective` | λᵠ and ν updated inside every layer, and between epochs |
| 10. Robust CSI loss | `outage_surrogate` | S draws of H ~ p(H \| Ĥ) from the aging model; sigmoid outage; λ_O Σ[p̂ − ϵ]⁺ |
| 11. Local self-supervised training | `RapFedWMMSETrainer._local_step` | No labels; Adam on each node's own parameters |
| 12. Event trigger | `fl/communication.py: urgency` | Γᶠᴸ = β₁d^θ + β₂d^H + β₃A/A_max + β₄‖∇‖/G_max; eligible when ≥ δ_FL |
| 13. Budgeted selection | `select_clients`, `_fl_headroom` | Binary knapsack, solved exactly, under B_FL and each node's fronthaul or control-link headroom (paper Sec. XXXI) |
| 14. Top-k compression | `top_k` | Top-k of Δθ + e with error feedback |
| 15. Personalized aggregation | `personalized_updates`, `_federate` | α_ij = S_ij p_j / (Σ S_il p_l + ϵ); θ_i ← θ_i + ρ_p Δθ̄_i |
| U. Deployment | `RapFedWMMSETrainer.solve` | Local CSI → adapter → backbone → L layers → projection |

## Interpretation choices

### Step 1: association features

The text names β̄ᴿᵁ, β̄ᴬᴿ and Ī without fixing their scale. For a score Γ > 0 to mean "this relay helps", each
term is measured in decades (10 dB) **relative to the UE's direct link**:
- β̄ᴿᵁ is relay-to-UE gain minus the direct gain.
- β̄ᴬᴿ is the gain from the UE's serving APs to the relay, minus the direct gain.
- Ī is the other APs' gain at the relay relative to the serving APs'.

A pair is chosen only if Γ > `association.min_score` (0). Every algorithm uses this matching with zero QoS prices,
so all methods share one association rule. RAP-FedWMMSE also feeds its QoS prices into the score.

### Step 2: τ search

Sir allows "projected gradient or one-dimensional search". The search uses the SINRs measured in the epoch's
last training batch, with beams held fixed. Under that approximation the utility is concave in τ, so a
61-point grid finds the maximum. τ is a slow-timescale variable: it is set once per epoch, not per block.

### Step 3: what ζ(t) carries

Sir lists ζ(t) as "QoS dual prices, measured aggregate interference, DF constraint prices, and previous rate
statistics". `SlowCoordination` holds exactly these, set once per slow-timescale epoch:
- λᵠ_k and ν_rk, after the epoch's dual ascent;
- log₁₀(1 + I_k/σ²) for Phase I and Phase II, the aggregate interference UE k measured;
- the mean rate of each UE over the epoch.

That is 4K + RK numbers, the same for every node, with no antenna-domain CSI. Every node's network input is its own
CSI magnitudes, large-scale gains and cluster or association row, plus this vector
(`test_zeta_carries_every_quantity_sir_lists`). At deployment, the warm-up epochs on a new drop produce it.

### Step 1: cooperation order L_max

The association can be stored as a relay index per UE (L_max = 1) or as the binary matrix a_rk (L_max > 1), and
every part of the code accepts both (`sim.rates.as_assoc_matrix`). With L_max > 1:
- the DF rate is r_k = min{C_D_k, min over r with a_rk = 1 of C_AR_rk}: every assisting relay must decode;
- Phase II gains of all assisting relays add coherently at the UE;
- an AP steers a cooperatively relayed stream along the sum of the unit directions towards its relays;
- Centralized WMMSE has one decoding constraint per assisting relay, and the exact MINLP optimizes a_rk under
  Σ_r a_rk ≤ L_max.

The default is L_max = 1, the paper's "simplest implementation". G06 adds L_max = 2.

### Step 4: what the head predicts

Ω_i holds one step η and one damping δ per layer, one λ_reg, and K power shares for W⁽⁰⁾ or V⁽⁰⁾. Beam
**directions** are not learned: they are MRT towards each stream's receiver (the relay, for a relayed UE),
taken from the node's own CSI. This follows Sir's principle of not learning what a model gives reliably.

### Step 7: the gradient, ζ, and the "regularized" step

**The gradient is exact.** With the MMSE equalizer c and weight ρ = 1/ε, the envelope theorem gives
∂log₂(1 + SINR)/∂w* = −(ρ/ln 2) ∂ε/∂w*. Chaining through J gives each link's weight A = −(∂J/∂C)·(time share)·ρ/ln 2.
AP m's gradient is then

```
∇_{w_mj*} J = Σ_k A1_k (|c1_k|² S1_kj − δ_jk c1_k) h_mk
            + Σ_{r,k} AAR_rk (|cAR_rk|² SAR_rkj − δ_jk cAR_rk) G_mr^H u_rk  +  (λ_E/η_A) w_mj
```

AP m supplies its own h_mk and G_mr; everything else is the same for every node. That shared part is **ζ**: the
effective gains S1, S2 and SAR, the equalizers, the weights A, and the relay combiners u. A relay needs only g_rk
plus ζ. `test_local_gradients_equal_the_gradient_of_the_global_objective` checks this against autograd on the
global J, to machine precision.

**ζ is larger than Sir's list.** Sir lists per-UE scalars (prices, aggregate interference, desired-signal
statistics). An exact gradient needs the per-stream gains S_kj, which take K² numbers per phase rather than K.
They are still independent of the antenna counts. Over-the-air, a UE measures them from precoded pilots. See
[Overhead](#overhead) for the honest count.

**"Learned regularization".** Adding (λ/2)‖W_m‖² to the node's quadratic WMMSE surrogate and taking a Newton step
of length η gives

```
W_m ← W_m + η ((Q_m + λ I)^-1 B_m − W_m),    B_m = Q_m W_m − ∇J
```

Here Q_m is the node's own curvature, and λ = λ_reg · μ_m, where μ_m is the classical WMMSE power price found by
bisection from the node's own Q_m and B_m. With η = 1 and λ_reg = 1 this is **exactly** the classical per-node
WMMSE update (`test_neutral_layer_is_the_classical_wmmse_update`). Training starts from a sound optimizer and
adjusts it.

**Diagonal loading.** Q_m has rank at most the number of streams with weight in J. With more antennas than such
streams and a slack power limit (λ ≈ 0), Q_m + λI is singular and the solve failed, which crashed training on some
drops. A load of 10⁻¹⁰ · trace(Q_m)/N is added inside the solve. It selects the minimum-norm WMMSE step and is far
below any physical term (`test_beam_update_survives_a_singular_curvature`).

An earlier version used λ as a damping term instead. It plateaued at PF ≈ 3.6, against 5.6 for Centralized on the
same drop, so it was replaced.

### Step 9: the DF price

Step 5.3 defines r_k = min(C_D, C_AR), so r_k − C_AR ≤ 0 always holds and the DF multiplier ν, as written, would
never leave 0. It therefore follows **C_D − C_AR**, which is positive exactly when the relay hop is the bottleneck.
In J, the penalty is ν (sg(C_D) − C_AR), with C_D held fixed (`sg` = stop-gradient). The price then raises the
relay hop without pushing the destination rate down. This matches Sir's description: "the next beam update gives
greater importance to that user/hop".

### Step 10: CSI-error model

p(H | Ĥ) is the paper's aging model, H = ρĤ + √(1−ρ²) E, with E at each link's large-scale power. ρ follows the
`aging` settings when aging is on. Otherwise it is `fedwmmse.robust_rho` = 0.98, so training stays robust even when
evaluation uses perfect CSI.

### Step 11: what is local in training

Sir's J (Step 9) sums over every UE, so it is the network objective, and θ_i ← θ_i − η ∇_{θ_i} L_i asks for the
gradient of that shared objective with respect to node i's own parameters. That is what is computed.

At **deployment**, every node's computation uses only its own CSI plus ζ, as Sir requires. In **training**,
∇_{θ_i} of the shared J is computed by autograd through every node's layers in one simulated batch. A real network
would need the reverse exchange of ζ for this. This is a standard simulation shortcut, stated here so it is not
mistaken for something stronger.

### Steps 12–15: federation details

- **Shared start.** All nodes start from one common backbone, as in standard FL.
- **What is uploaded.** Δθ_i = θ_i^local − θ_i^received, the node's own progress since its last sync.
- **Selected nodes** set θ_i ← θ_i^received + ρ_p Δθ̄_i and reset their age, as in Step 15.
- **Non-selected nodes** also receive Δθ̄_i, which is added to their local model. Their own pending update stays
  pending until they are selected.
- **Upload size** per client is ⌈κP⌉ × (4-byte value + 4-byte index) plus the signature. The **download** is the
  dense Δθ̄_i to every node, since Sir's text compresses uploads only.
- **Signature.** s_i = [β̃_i, Ĩ_i, L̃_i, role one-hot]. β̃_i is the node's gains to the UEs, normalized over all
  nodes and divided by √K so it does not dominate the distance. Ĩ_i is measured interference, and L̃_i the load.
- **FedAvg comparison.** Dense uploads, an equal-weight average, everyone receives it, and ρ_p = 1.
- **No-FL comparison.** Local training only.

### Step 13: the fronthaul constraint

The paper (Sec. XXXI) requires D_data_m + D_FL_m ≤ F_max_m T_e for each AP and D_FL_r ≤ F_ctrl_r T_e for each relay,
all in bits. `_fl_headroom` computes each node's headroom at the end of an epoch:
- **APs.** D_data_m = B T_e Σ_k c_mk r̄_k, from the epoch's rates, which ζ carries.
- **Relays.** They receive user data over the air, so their control link carries FL traffic only.

A node joins Step 13's knapsack only if its headroom covers its compressed upload plus the dense download. Because
q_i is binary, this is exactly the per-node constraint q_i (S_up + S_down) ≤ headroom_i. A node that was not
selected receives Step 15's aggregated update only if its headroom covers the download. FedAvg obeys the same rule
for its dense exchange.

The user data itself is not throttled. An AP whose data alone exceeds F_max is counted as overloaded (G14 (b)) and
sends no FL traffic. Defaults: F_max = 1 Gbit/s per AP, F_ctrl = 10 Mbit/s per relay, T_e = 1 s. G14 sweeps F_max.

### Multi-timescale operation

Each node's local data set D_i is the channel generator of its own deployment: APs and relays stay fixed, and every
epoch brings a fresh UE drop (`fedwmmse.redrop_every` = 1), as new users arrive. Large-scale statistics are constant
within an epoch and change between epochs, which is the multi-timescale structure of Sir's Section B.

An earlier version trained on **one** UE layout that only moved a few metres per epoch (`redrop_every` = 0 keeps that
behaviour). The model then overfit that layout. On the small network it nearly starved a UE on some evaluation drops,
and trained RAP-FedWMMSE scored below the same layers untrained. In a 20-drop pilot, fresh drops changed the mean PF
utility from −1.45 to −0.98 on the small network (untrained: −1.07; worst drop −7.64 → −3.37) and from 7.47 to 7.56
on the default network, at the same training time. That pilot used evaluation drops, which is a mild form of test-set
selection; the final run uses 100 drops, of which those 20 are a fifth. Evaluation drops are different UE drops on the same deployment;
training and validation drop IDs start at 10⁶ and 2·10⁶ so they can never coincide with evaluation drops.

When solving a new drop, `solve` first runs `warmup_epochs` slow-timescale epochs on freshly sampled fading blocks
of that drop's geometry. These set the QoS prices, association and τ. Sampled blocks need only large-scale
statistics, which nodes have. The policy then runs once on the drop's estimated channels.

## Overhead

One ζ exchange carries 2K² + 4K complex numbers for the UEs, plus R(K² + 2K + K·N_R) for the relays. At the default
size (K = 8, R = 2, N_R = 2) that is **352 complex numbers per layer**, or 2,112 for 6 layers. Uploading all
instantaneous CSI to a CPU would be M·K·N_A + M·R·N_R·N_A + R·K·N_R = **224**.

So at 2 antennas per node, the coordination traffic per block is about 9× the raw CSI. The two travel on different
links (broadcast measurements against fronthaul to a CPU), and ζ does not grow with the antenna counts, so the
comparison reverses for large arrays. At the reduced scale of this project, though, **the method does not save
signaling per block**. Its benefits shown here are distributed computation and fast inference, not less traffic.
