# Graph plan

The 14 figures comparing the algorithms Sir asked for. `python run_experiments.py` produces them in
`results/figures/`, and the final set is copied to [`figures/`](../figures/). Each PNG has a CSV of its plotted
numbers beside it. Bands and error bars are 95% confidence intervals over drops. Three figures have two panels,
(a) and (b), showing two views of the same experiment.

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

## Figures

| # | Figure | Question it answers | Experiment |
|---|---|---|---|
| G01 | Per-user rate CDF | Who gains and who is left behind? | `baseline` |
| G02 | Outage vs R_min | How reliable is each method against a QoS target? | `baseline` |
| G03 | (a) Mean rate and (b) 5th-percentile rate vs transmit power | What does extra power buy, on average and for the worst users? | `power` |
| G04 | PF utility vs K | How does each method handle load? | `num_ues` |
| G05 | PF utility vs blocked fraction | Does the relay gain grow with blockage? | `blocked_fraction` |
| G06 | PF utility vs R, with L_max = 1 (solid) and L_max = 2 (dashed) | Do more relays, and cooperating relays, help? | `num_relays`, `cooperation` |
| G07 | PF utility vs τ, held fixed | What is the best phase split? | `tau` |
| G08 | PF utility vs aging ρ | How sensitive is each method to stale CSI? | `aging` |
| G09 | PF utility vs unfolded layers L, every other algorithm as a reference line | How much do the learned parameters add, and how many layers are needed? | `layers`, `tau` at τ = 0.5 |
| G10 | Gap to the certified MINLP upper bound | How far is each method from the best possible, on a small network? | `optimality_gap` |
| G11 | Runtime vs K | What does each method cost to run? | `num_ues` |
| G12 | Validation PF utility vs (a) federated epoch and (b) cumulative FL bytes | How fast do RAP-FedWMMSE and its ablations learn, and at what communication cost? | `baseline` |
| G13 | PF utility vs network power consumption, swept by λ_E | What does the energy term λ_E·P_tot of Sir's objective trade? | `energy` |
| G14 | (a) PF utility and (b) FL participation vs fronthaul capacity | How does the fronthaul constraint of the paper (Sec. XXXI) limit federation? | `fronthaul` |

## Fair comparison

- **Same drops for every method.** They share AP and relay layouts (deployments), and training uses separate drops.
- **Every method uses Step 1's association** with zero QoS prices. RAP-FedWMMSE also feeds in its own prices.
- **τ is optimized by RAP-FedWMMSE** (per epoch) and by Centralized WMMSE (per block). Local-WMMSE and the
  heuristics keep τ = 0.5, the fixed split Sir's text calls conventional. G07 shows all methods at every fixed τ.
- **The no-relay baseline uses τ = 1,** since a network without relays has no Phase II.
- **G07, G09 and G10 hold τ and association fixed for every relay method,** so all solve the same problem. G09's
  reference lines come from the τ = 0.5 point of G07: the same drops, τ and association as its curves. G10's bound
  is valid under that condition; since SCIP also optimizes the association, it covers every association.
- **G11 times inference.** For RAP-FedWMMSE that includes the slow-timescale warm-up but not the training, whose
  time is saved separately as `fit_seconds`. Timings were measured with 11 worker processes on 12 cores, one
  thread each, so every method ran under the same load.
- **G13 sweeps λ_E for every method that has a power term** (RAP-FedWMMSE, both centralized methods,
  Local-WMMSE). The heuristics always use full power, so each appears as a single point.
- **G14 applies the fronthaul limit to every federated variant alike.** FedAvg needs room for a dense upload and
  download; RAP-FedWMMSE for its compressed upload and dense download. User data is not capped by the limit: an AP
  whose user data alone exceeds F_max is counted as overloaded (dashed line in (b)) and sends no FL traffic.

## Commands

```bash
python run_experiments.py --drops 100 --deployments 10    # the final run, about 3 hours on 12 cores
python run_experiments.py --experiments power tau         # some sweeps
python run_experiments.py --plot-only                     # redraw from saved results
```
