# Algorithms and baselines

Every algorithm is scored by one function, `eval.metrics.evaluate_solution`, on the same drops and on the true
channels. It rejects any solution that breaks a power, cluster, association, combiner or τ rule, or that contains
NaN.

| Name in code | Method | CSI used | Notes |
|---|---|---|---|
| `rap_fedwmmse` | **RAP-FedWMMSE, Sir's algorithm** | Local + ζ | See [RAP_FEDWMMSE.md](RAP_FEDWMMSE.md) |
| `rap_fedwmmse_fedavg` | Ablation: FedAvg instead of personalized aggregation | Local + ζ | Dense uploads, everyone participates |
| `rap_fedwmmse_nofl` | Ablation: local training only, no federation | Local + ζ | |
| `unfolded_wmmse` | Ablation: the same layers, untrained | Local + ζ | Classical per-node WMMSE with damping 0.5 |
| `centralized_wmmse` | Centralized WMMSE-SCA/AO | Full, at a CPU | Optimizes τ per block, as P0 does |
| `centralized_no_relay` | The same, without relays | Full | |
| `local_wmmse` | Local-WMMSE / No-FL | Local only, no ζ | τ fixed at 0.5 |
| `rzf_equal`, `mrt_equal` | Heuristics | Local only | τ fixed at 0.5 |
| `exact_minlp` | Global MINLP by spatial branch-and-bound (SCIP) | Full | Small networks only, τ fixed |
| `mrt_fractional`, `neural_ctde` | Earlier heuristic and earlier neural policy | | Not in the graphs |

## Shared choices

- **Association (Step 1) is used by every method,** with zero QoS prices. RAP-FedWMMSE also adds its own prices.
  - The weights α and the threshold are my calibration, made with Centralized WMMSE on deployments 90–95, which
    are never used for evaluation. There it beat the earlier "relay only blocked UEs" rule by +0.31 PF on average,
    **but lost 3.07 on 1 of 12 drops**.
  - A pair is eligible only if its relay-to-UE hop beats the direct link (relaxed by the QoS price). Without this
    filter, strong AP-to-relay hops made weak relay hops look attractive, and Centralized lost 0.6–1.3 PF on average.
- **τ:** RAP-FedWMMSE adapts it per epoch, and Centralized WMMSE per block. The others keep 0.5. G08, G10 and G11
  hold τ and association fixed for every method.
- **QoS:** RAP-FedWMMSE prices R_min through λᵠ, and its training penalizes outage. The baselines only report outage.
- **Link budget (my choice):**
  - path loss: 3GPP UMi NLOS, −30.5 − 36.7 log₁₀(d) dB
  - noise: 20 MHz bandwidth, 7 dB noise figure
  - power and area: 1 W per AP and relay, 4 dB shadowing, 500 m square
  - the AP-relay link uses the same path loss as AP-UE links

## Known limits

- **Centralized WMMSE finds a local solution,** since P0 is non-convex. The exact method measures the gap on small
  networks only. It can stop at `wmmse.max_iters` = 100 and records `info.status` per drop.
- **Exact is limited to a small network:** 2 APs, 1 relay, 2 UEs, single antennas, τ fixed, and certified to a 1% gap.
  These sizes were tested before the association change: 20 of 20 drops reached the 1% gap. With one more UE or
  antenna, many runs hit a 90 s limit.
- **Perfect CSI is the default, and pilot overhead is not modelled.** Aging appears only in its own sweep (G09).
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
