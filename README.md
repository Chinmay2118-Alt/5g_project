# RAP-FedWMMSE (Group 12, IIIT Dharwad)

Role-aware personalized federated beam optimization for intelligent-relay-assisted cell-free massive MIMO.

| Where | What |
|---|---|
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Install, data (synthetic), parameters, seeds, and the exact commands for every result and figure |
| [rapfedbeam/](rapfedbeam/) | The code ([README](rapfedbeam/README.md)); the 14 figures are in `rapfedbeam/figures/` |
| [paper/](paper/) | The IEEE paper (`main.tex`, `main.pdf`) and the Google Deep Review revision log |

Quick start: `python -m venv .venv`, activate it, `python -m pip install -r requirements-lock.txt`, then from
`rapfedbeam/` run `python -m pytest -q` and `python run_experiments.py --drops 100 --deployments 10`.
