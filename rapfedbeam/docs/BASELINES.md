# Algorithms and baselines

Every algorithm is scored by one function, `eval.metrics.evaluate_solution`, on the same drops and on the true
channels. It rejects any solution that breaks a power, cluster, association, combiner or τ rule, or that contains
NaN.

| Name in code | Method | CSI used | Notes |
|---|---|---|---|
| `rap_fedwmmse` | **RAP-FedWMMSE, Sir's algorithm** | Local + ζ | See [RAP_FEDWMMSE.md](RAP_FEDWMMSE.md) |
| `rap_fedwmmse_fedavg` | Ablation: FedAvg instead of personalized aggregation | Local + ζ | Dense uploads, everyone participates |
| `rap_fedwmmse_nofl` | Ablation: local training only, no federation | Local + ζ | |
| `unfolded_wmmse` | Ablation: the same layers, untrained | Local + ζ | Classical per-node WMMSE with damping 0.5, τ = 0.5, zero prices, no warm-up: no slow timescale at all, so its gap to RAP-FedWMMSE is learning plus Steps 1–2 |
| `unfolded_wmmse_slow` | Diagnostic: the untrained layers plus the training-free slow timescale (Steps 1–2, warm-up) | Local + ζ | Isolates learning from Steps 1–2 (`diagnostics/round2.py`) |
| `classical_wmmse_layers`, `classical_wmmse_slow` | Diagnostic: undamped classical per-node WMMSE (η = δ = λ_reg = 1), without and with Steps 1–2 | Local + ζ | Oscillates when all nodes update in parallel; worse than the damped ablation |
| `centralized_wmmse` | Centralized WMMSE-SCA/AO | Full, at a CPU | Optimizes τ per block, as P0 does |
| `centralized_no_relay` | The same, without relays | Full | τ = 1: with no Phase II the APs transmit for the whole interval |
| `local_wmmse` | Local-WMMSE / No-FL | Local only, no ζ | τ fixed at 0.5 |
| `rzf_equal`, `mrt_equal` | Heuristics | Local only | τ fixed at 0.5 |
| `exact_minlp` | Global MINLP by spatial branch-and-bound (SCIP) | Full | Small networks only. Association a_rk is binary and optimized; τ fixed |
| `mrt_fractional`, `neural_ctde` | Earlier heuristic and earlier neural policy | | Not in the graphs |

## Shared choices

- **Association (Step 1) is used by every method,** with zero QoS prices. RAP-FedWMMSE also adds its own prices.
  - The weights α and the threshold are my calibration, made with Centralized WMMSE on deployments 90–95, which
    are never used for evaluation. There it beat the earlier "relay only blocked UEs" rule by +0.31 PF on average,
    **but lost 3.07 on 1 of 12 drops**.
  - A pair is eligible only if its relay-to-UE hop beats the direct link (relaxed by the QoS price). Without this
    filter, strong AP-to-relay hops made weak relay hops look attractive, and Centralized lost 0.6–1.3 PF on average.
- **τ:** RAP-FedWMMSE adapts it per epoch, and Centralized WMMSE per block. Local-WMMSE and the heuristics keep
  0.5, the fixed split Sir's text names as the conventional choice. G07 shows every method at every fixed τ, so the
  ranking does not hinge on this. G07, G09 and G10 hold τ and association fixed for every relay method.
- **No relay means no Phase II.** `centralized_no_relay` therefore uses τ = 1. An earlier version held it to
  τ ≤ τ_max = 0.8, which wasted a fifth of every block and made the relays look better than they are.
- **QoS:** RAP-FedWMMSE prices R_min through λᵠ, and its training penalizes outage. The baselines only report outage.
- **Cooperation order:** L_max = 1 by default. With L_max > 1 every method handles several relays per UE: the rate
  engine, Centralized WMMSE (one decoding constraint per assisting relay), Local-WMMSE, the heuristics and the
  exact MINLP. G06 compares L_max = 1 and 2.
- **Energy:** λ_E = 0 by default, so the main figures maximize the PF utility alone. G13 sweeps λ_E for every
  method that has a power term.
- **Link budget (my choice):**
  - path loss: 3GPP UMi NLOS, −30.5 − 36.7 log₁₀(d) dB
  - noise: 20 MHz bandwidth, 7 dB noise figure
  - power and area: 1 W per AP and relay, 4 dB shadowing, 500 m square
  - the AP-relay link uses the same path loss as AP-UE links

## Known limits

- **Centralized WMMSE finds a local solution,** since P0 is non-convex. The exact method measures the gap on small
  networks only. It can stop at `wmmse.max_iters` = 100 and records `info.status` per drop.
- **Exact is limited to a small network:** 2 APs, 1 relay, 2 UEs, single antennas, τ fixed, and certified to a 1% gap.
  The association variables a_rk are binary, so it is a true MINLP, and its upper bound covers every association,
  not only the one Step 1 picks. With one more UE or antenna, many runs hit a 90 s limit.
- **Perfect CSI is the default, and pilot overhead is not modelled.** Aging appears only in its own sweep (G08).
- **AP clusters are chosen from the AP-UE gain,** as in the paper, even for relayed UEs. This is also why the
  association features can misjudge a blocked UE's relay.
- **A drop is evaluated as one coherence block.** The multi-timescale loop runs inside RAP-FedWMMSE's training,
  with UE mobility between epochs, and in its warm-up before each evaluated drop.

## Reproducibility

`make_instance(config, seed, drop_idx, deployment)` seeds the drop on `(seed, deployment, drop_idx)` and the
infrastructure on `(seed, deployment)`. Each quantity has its own random stream, so sweep points differ only in what
the sweep changes. Training and validation drops have their own indices, starting at 10⁶ and 2·10⁶. RAP-FedWMMSE
seeds PyTorch per deployment, and serial and parallel runs give identical numbers (a test checks this).

On the small network, `rzf_equal` and `mrt_equal` give identical results. This is correct: with one antenna, the RZF
direction reduces to MRT.
