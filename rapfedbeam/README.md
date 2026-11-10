# RAP-FedWMMSE

Role-aware personalized federated beam optimization for intelligent-relay-assisted cell-free massive MIMO.
The repository implements Sir's algorithm, **RAP-FedWMMSE** (unfolded WMMSE with learned parameters,
primal-dual QoS and DF correction, and event-triggered, compressed, topology-aware personalized federated
learning). It also implements every baseline it is compared against, and produces the 14 comparison graphs.

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Run from `rapfedbeam/`:

```bash
python -m pytest -q                                          # about 5 minutes
python run_experiments.py --drops 20 --deployments 4         # all sweeps and graphs, about an hour on 12 cores
python run_experiments.py --plot-only                        # redraw graphs from saved results
python run_simulation.py --drops 20                          # quick heuristic-only check, written to plots/
```

Results and graphs go to `results/` (gitignored).

## Documents

| Document | Contents |
|---|---|
| [docs/RAP_FEDWMMSE.md](docs/RAP_FEDWMMSE.md) | **The main algorithm**: each step mapped to the code, and every interpretation choice |
| [docs/BASELINES.md](docs/BASELINES.md) | Every algorithm, the shared choices, and known limits |
| [docs/WMMSE_DERIVATION.md](docs/WMMSE_DERIVATION.md) | The math for the two-phase DF rate |
| [docs/GRAPH_PLAN.md](docs/GRAPH_PLAN.md) | The 14 graphs and how the comparison is kept fair |
| [docs/ADDING_AN_ALGORITHM.md](docs/ADDING_AN_ALGORITHM.md) | How to add a method or ablation |

## Directory map

```
configs/      network, link budget, association, RAP-FedWMMSE and solver settings (config.yaml)
sim/          drops and deployments, channels, rates, association (Step 1), ProblemInstance and BeamSolution
train/        RAP-FedWMMSE (rap_fedwmmse.py, unfolded_wmmse.py); the earlier neural CTDE policy
fl/           RAP-FedWMMSE's federation (communication.py); kernel and aggregation helpers
baselines/    Centralized and Local WMMSE, exact MINLP, heuristics
experiments/  algorithm registry, sweep runner, sweep definitions
eval/         scoring (metrics.py), the 14 graphs (figures.py), earlier plots (plots.py)
models/       role adapters and the shared backbone, used by RAP-FedWMMSE
docs/         algorithm, baselines, derivation, graph plan, how to add an algorithm
tests/        closed forms, solver cross-checks, RAP-FedWMMSE step by step, the experiment harness
```

`train/ctde_train.py`, `train/federated_loop.py` and `run_federated.py` are the earlier black-box neural approach
from before the 15 October direction. They still run, but they are not the proposed method.
