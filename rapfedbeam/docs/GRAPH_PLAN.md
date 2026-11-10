# Graph plan

The 14 graphs comparing the algorithms Sir asked for. `python run_experiments.py` produces them in
`results/figures/`, each PNG with a CSV of its plotted numbers. Bands are 95% confidence intervals over drops.

## Algorithms

| Name in graphs | Role |
|---|---|
| RAP-FedWMMSE | The proposed method (Sir's algorithm) |
| Centralized WMMSE-SCA | Benchmark with full CSI at a CPU; optimizes τ too |
| Centralized WMMSE, no relay (τ = 1) | Shows what the relays add; uses the whole interval since it has no Phase II |
| Local-WMMSE (no FL) | Each node alone, without coordination or learning |
| RZF, equal power and MRT, equal power | Simple heuristics |
| Exact MINLP (B&B) | Global optimum with a certified bound, on a small network only |
| RAP-FedWMMSE with FedAvg / no FL; Unfolded WMMSE, untrained | Ablations: what personalization, federation and learning each add |

## Graphs

| # | Graph | Question it answers | Experiment |
|---|---|---|---|
| G01 | Per-user rate CDF | Who gains and who is left behind? | `baseline` |
| G02 | Outage vs R_min | How reliable is each method against a QoS target? | `baseline` |
| G03 | Mean rate vs transmit power | What does extra power buy? | `power` |
| G04 | 5th-percentile rate vs transmit power | What does it buy the worst users? | `power` |
| G05 | PF utility vs K | How does each method handle load? | `num_ues` |
| G06 | PF utility vs blocked fraction | Does the relay gain grow with blockage? | `blocked_fraction` |
| G07 | PF utility vs R | Do more relays help? | `num_relays` |
| G08 | PF utility vs τ, held fixed | What is the best phase split? | `tau` |
| G09 | PF utility vs aging ρ | How sensitive is each method to stale CSI? | `aging` |
| G10 | PF utility vs unfolded layers L | How much do the learned parameters add, and how many layers are needed? | `layers` |
| G11 | Gap to the certified upper bound | How far is each method from the best possible, on a small network? | `optimality_gap` |
| G12 | Runtime vs K | What does each method cost to run? | `num_ues` |
| G13 | Validation PF utility vs federated epoch | How fast do RAP-FedWMMSE and its ablations learn? | `baseline` |
| G14 | Validation PF utility vs cumulative FL bytes | How much communication does that learning take? | `baseline` |

## Fair comparison

- **Same drops for every method.** They share AP and relay layouts (deployments), and training uses separate drops.
- **Every method uses Step 1's association** with zero QoS prices. RAP-FedWMMSE also feeds in its own prices.
- **τ is optimized by RAP-FedWMMSE** (per epoch) and by Centralized WMMSE (per block). Local-WMMSE and the
  heuristics keep τ = 0.5, the fixed split Sir's text calls conventional. G08 shows all methods at every fixed τ.
- **The no-relay baseline uses τ = 1,** since a network without relays has no Phase II.
- **G08, G10 and G11 hold τ and association fixed for every relay method,** so all solve the same problem. G11's
  bound is valid under that condition; since SCIP also optimizes the association, it covers every association.
- **G12 times inference.** For RAP-FedWMMSE that includes the slow-timescale warm-up but not the training, whose
  time is saved separately as `fit_seconds`.

## Commands

```bash
python run_experiments.py --drops 20 --deployments 4      # about an hour on 12 cores
python run_experiments.py --experiments power tau         # some sweeps
python run_experiments.py --plot-only                     # redraw from saved results
```

For the report, use more drops (for example `--drops 100 --deployments 10`), and check that the confidence bands
separate the methods before claiming one beats another. Re-run G12 with `--workers 1`, since timings measured
next to 10 other workers are noisy.
