# WMMSE derivation for the two-phase DF rate

This is the math behind [`baselines/wmmse.py`](../baselines/wmmse.py). Textbook WMMSE handles a
weighted sum rate. P0 is different in four ways: a log utility, a two-phase rate, the DF `min`, and
a relay receive combiner. Notation follows the paper. Gains are `h^H w`.

## 1. Problem solved

With association `A` and `τ` fixed, for one coherence block:

```
max  Σ_k ω_k log(r_k + ε0) − λ_E P_tx
s.t. r_k ≤ C_k^D = τ log2(1 + γ_k^(1)) + (1 − τ) log2(1 + γ_k^(2))
     r_k ≤ C_rk^AR = τ log2(1 + γ_rk^AR)      for the relay r with a_rk = 1
     Σ_k ||w_mk||² ≤ P_m^max,  Σ_k ||v_rk||² ≤ P_r^max
     w_mk = 0 if c_mk = 0,  v_rk = 0 if a_rk = 0
```

UEs without a relay have `γ^(2) = 0` and no `C^AR` constraint. The probabilistic QoS constraint is not
included (see [BASELINES.md](BASELINES.md)).

## 2. The rate as an MSE problem

Take any link with desired gain `s_k` and interfering gains `s_j`. With a scalar receiver `a`:

```
e(a, W) = |1 − a* s_k|² + Σ_{j≠k} |a|² |s_j|² + |a|² σ²
```

For any weight `ξ > 0`, using `ln(x) ≥ ln ξ + 1 − ξ/x`:

```
log2(1 + SINR) ≥ (ln ξ + 1 − ξ e(a, W)) / ln 2
```

Equality holds at

```
a* = s_k / (Σ_j |s_j|² + σ²)      ξ* = 1 / e*      e* = 1 / (1 + SINR)
```

The right-hand side is **concave in the beams** for fixed `(a, ξ)`, because `e` is a convex quadratic in
them. `tests/test_baselines.py` checks numerically that this bound touches the rate at its own point
and never exceeds it elsewhere.

### Relay decoding

The relay has a vector receiver `c_rk`, so the same bound applies with

```
e = |1 − c^H G w_k|² + Σ_{j≠k} |c^H G w_j|² + ||c||² σ_r²      where G w_j = Σ_m G_mr w_mj
c* = (Σ_j (G w_j)(G w_j)^H + σ_r² I)^{-1} (G w_k)
```

The SINR does not depend on the scale of `c`, so the paper's unit-norm combiner is `u_rk = c* / ||c*||`.

## 3. Centralized WMMSE-SCA/AO

Repeat until the utility gain falls below `wmmse.tol`:

1. **Receivers and weights.** Compute `a*, ξ*` for both UE phases, and `c*, ξ*` for relay decoding, at
   the current beams.
2. **Beams and rates.** Replace each rate by its bound and solve

   ```
   max  Σ_k ω_k log(r_k + ε0) − λ_E P_tx
   s.t. r_k ≤ τ B_k^(1)(W) + (1 − τ) B_k^(2)(V)
        r_k ≤ τ B_rk^AR(W)
        per-node power budgets
   ```

   This is a convex program: a concave objective with convex quadratic constraints. It keeps the DF
   `min` and the log utility exact, with no reweighting heuristic. It is solved with Clarabel via CVXPY.
3. *(Optional, `wmmse.optimize_tau`)* **Phase split.** For fixed beams the utility is concave in `τ`,
   so a grid search over `[τ_min, τ_max]` is enough.

### Why it converges

Let `f` be the utility on the channel estimates. Step 1 makes the bound tight, so the bound's value
equals `f(x_t)`. Step 2 maximizes that bound over a set containing `x_t`, so its value cannot fall.
Since the bound is below the true rate everywhere, `f(x_{t+1}) ≥ f(x_t)`. `f` is bounded above on the
compact power region, so the sequence converges. The code rejects any step that lowers `f`, which
guards against solver tolerance.

**This shows a monotone sequence, and is not a proof of global optimality.** WMMSE can stop at a local
solution. Spatial branch-and-bound measures how far off it is on small networks. On the 20 pilot drops, SCIP's
best solution was at most 0.005 above WMMSE; on slightly larger test networks, the most was 0.010.

### Numerical note

At high SINR, `ξ` is large and the constant and quadratic terms nearly cancel, which crashed the solver
on one drop. The update is therefore written in the step `d = x − x0`:

```
B(x0 + d) = B(x0) + (f − 2 QᵀQ x0)ᵀ d − ||Q d||²
```

## 4. Single-node closed form

For one transmitter with fixed PF weights `π_k`, the beam update has a closed form:

```
A = Σ_i π_i ξ_i |a_i|² b_i b_i^H
w_k = π_k ξ_k a_k (A + μ I)^{-1} b_k
```

`μ ≥ 0` is the smallest value that meets the power budget, found by bisection (the power falls as `μ`
grows), and it starts from the energy price rather than from 0 when `λ_E > 0`.

The PF weight comes from the gradient of the log utility:

```
π_k = ω_k s_k / (s_k R_k + ε0)      where s_k is the phase's time share (τ or 1 − τ)
```

**Refresh `π_k` on every iteration.** With `π_k` held fixed for 10 inner steps, the weighted sum-rate
update switched weak users off, the log utility collapsed, and the node kept its starting beams. A
test caught this. This update is not guaranteed monotone, so `wmmse_node` keeps its best iterate.

`wmmse_node` implements this and is written so a distributed algorithm can reuse it.

## 5. The distributed version: RAP-FedWMMSE

Sir's algorithm unfolds this update across the network. [RAP_FEDWMMSE.md](RAP_FEDWMMSE.md) covers the details.
It differs from the centralized version in three ways:

- **`W` is coupled across APs.** Coherent joint transmission puts every AP's `w_mk` inside one `|Σ_m ...|²`, so
  an AP's exact gradient needs the per-stream aggregate gains `S_kj`, which the coordination vector `ζ` carries.
  Each AP then takes a step using only its own curvature, so the APs update in parallel rather than jointly. This
  is why the untrained version needs hundreds of layers, and why learned step sizes and damping help.
- **The DF `min` needs a price.** The soft-min weights and the DF dual price decide how much each hop counts.
- **The per-node step is the closed form of Section 4,** generalized to the two-phase DF objective. With neutral
  learned values it is exactly classical per-node WMMSE.
