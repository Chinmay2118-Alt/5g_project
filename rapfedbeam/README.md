# RAP-FedBeam (reduced scope)

This repository implements the reduced-scope course-project architecture. The
simulator and baseline path use NumPy; policy, projection, and federated-model
code use PyTorch. Stage B is centralized-training/decentralized-execution (CTDE),
as required by the implementation architecture, rather than the paper's exact
local-Lagrangian mechanism.

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Run the tests from `rapfedbeam/`:

```bash
python -m pytest -q
```

Run the physical-layer simulation and print relay/no-relay, heuristic, PF, and
phase-split results:

```bash
python run_simulation.py --drops 20
```

The module order is simulator and rates, baselines, role-specific models,
topology-aware FL, then training and evaluation.

## Honesty reminder (see Section 0 of the implementation plan)

`train/ctde_train.py` (Stage B, the v1 main method) is a **centralized-training /
decentralized-execution** simplification. It is NOT the paper's proposed local-Lagrangian
mechanism. Only `train/local_zeta_train.py` (Stage C, stretch goal) actually tests the
paper's claim that local CSI + a broadcast zeta vector is sufficient. State this
distinction explicitly in the report — do not present Stage B results as if they were
the paper's method.

## Directory map

```
configs/        scenario size, blockage, tau, training mode (config.yaml)
sim/            scenario generation, channel sampling, SINR/rate engine
baselines/      heuristics + numeric-optimum benchmark (Stage A)
models/         adapters -> backbone -> heads -> projection
fl/             topology signature, RBF kernel, aggregation modes, comm cost
train/          Stage B (ctde), Stage C (local zeta), federated round loop
eval/           metrics + plots for E1-E6
tests/          closed-form sanity checks
```
