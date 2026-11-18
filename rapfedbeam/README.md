# RAP-FedWMMSE

Role-aware personalized federated beam optimization for intelligent-relay-assisted cell-free massive MIMO.

This repository implements **RAP-FedWMMSE**: unfolded WMMSE with learned parameters, primal-dual
QoS and DF correction, and event-triggered, compressed, topology-aware personalized federated learning. It also
implements every baseline it is compared against, including an exact MINLP solved by spatial branch-and-bound, and it
produces the 14 comparison figures in [`figures/`](figures/).

## Quick start

Install the dependencies from the repository root. Python 3.12 is the tested version.

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use .venv/bin/activate elsewhere
python -m pip install -r requirements.txt
```

Then, from `rapfedbeam/`:

```bash
python -m pytest -q                                          # 87 tests, about 2.5 minutes
python run_experiments.py --drops 100 --deployments 10       # the final run, about 3 hours on 12 cores
python run_experiments.py --experiments energy               # a single sweep
python run_experiments.py --plot-only                        # redraw the figures from saved results
```

Raw results go to `results/`, which is gitignored. The final figures, each with a CSV of its plotted numbers, are
committed in [`figures/`](figures/).

## Results (default network: 8 APs, 2 relays, 8 UEs; 100 drops over 10 deployments)

| Method | PF utility | Mean rate (bit/s/Hz) | 5th-percentile rate | Time per drop |
|---|---|---|---|---|
| Centralized WMMSE-SCA/AO | 9.88 ± 0.40 | 4.57 | 0.67 | 5.05 s |
| Centralized WMMSE, no relay (τ = 1) | 9.80 ± 0.53 | 5.41 | 0.18 | 1.46 s |
| **RAP-FedWMMSE** | **7.88 ± 0.51** | **3.49** | **0.57** | **0.16 s** (0.03 s per block) |
| Unfolded WMMSE, untrained | 3.87 ± 0.46 | 2.06 | 0.36 | 0.03 s |
| Local-WMMSE / No-FL | −3.43 ± 0.43 | 0.89 | 0.20 | 0.32 s |

Times: one clean run (`python -m diagnostics.runtime`), every method on the same 10 drops, one thread, idle machine.
Extra ablations and diagnostics for the paper's review: `python -m diagnostics.round2 all`. Exact package versions:
`requirements-lock.txt` at the repository root.

The findings, including what did not work, are in the paper ([`../paper/main.pdf`](../paper/main.pdf)).

## Figures

| # | Figure | # | Figure |
|---|---|---|---|
| G01 | Per-user rate CDF | G08 | PF utility vs channel aging |
| G02 | Outage vs QoS target | G09 | PF utility vs unfolded layers |
| G03 | Mean and 5th-percentile rate vs power | G10 | Gap to the certified MINLP bound |
| G04 | PF utility vs number of UEs | G11 | Runtime vs number of UEs |
| G05 | PF utility vs blockage | G12 | Federated learning curves and communication |
| G06 | PF utility vs number of relays, L_max = 1 and 2 | G13 | PF utility vs power, swept by λ_E |
| G07 | PF utility vs phase split τ | G14 | Federated learning under the fronthaul constraint |

## Documents

| Document | Contents |
|---|---|
| [docs/RAP_FEDWMMSE.md](docs/RAP_FEDWMMSE.md) | **The main algorithm**: each step mapped to the code, and every interpretation choice |
| [docs/BASELINES.md](docs/BASELINES.md) | Every algorithm, the shared choices, and known limits |
| [docs/GRAPH_PLAN.md](docs/GRAPH_PLAN.md) | The 14 figures and how the comparison is kept fair |
| [docs/WMMSE_DERIVATION.md](docs/WMMSE_DERIVATION.md) | The math for the two-phase DF rate |
| [docs/ADDING_AN_ALGORITHM.md](docs/ADDING_AN_ALGORITHM.md) | How to add a method or ablation |

## Directory map

```
configs/      network, link budget, association, fronthaul, RAP-FedWMMSE and solver settings (config.yaml)
sim/          drops and deployments, channels, rates, association (Step 1), ProblemInstance and BeamSolution
train/        RAP-FedWMMSE (rap_fedwmmse.py, unfolded_wmmse.py); the earlier neural CTDE policy
fl/           RAP-FedWMMSE's federation (communication.py); kernel and aggregation helpers
baselines/    Centralized and Local WMMSE, exact MINLP, heuristics
experiments/  algorithm registry, sweep runner, sweep definitions
eval/         scoring (metrics.py), the 14 figures (figures.py), earlier plots (plots.py)
models/       role adapters and the shared backbone, used by RAP-FedWMMSE
figures/      the final 14 figures with their CSVs
docs/         algorithm, baselines, derivation, graph plan
tests/        closed forms, solver cross-checks, RAP-FedWMMSE step by step, L_max > 1, the experiment harness
```

`train/ctde_train.py`, `train/federated_loop.py`, `run_federated.py` and `run_simulation.py` (with `plots/`) are
from the earlier black-box neural approach, before the 15 October direction. They still run, but they are not the
proposed method.
