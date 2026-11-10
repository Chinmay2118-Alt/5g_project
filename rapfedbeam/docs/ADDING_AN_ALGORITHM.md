# Adding an algorithm

Any new method or ablation needs one class. Drops, scoring, feasibility checks, sweeps and graphs already work for
it. RAP-FedWMMSE in [`train/rap_fedwmmse.py`](../train/rap_fedwmmse.py) is a complete example.

## Steps

1. **Write the class** and register it:

   ```python
   from experiments.algorithms import Algorithm, register

   @register
   class MyMethod(Algorithm):
       name = "my_method"
       label = "My method"
       needs_fit = True          # False if nothing is learned across drops

       def fit(self, config, seed, deployment):
           # Train on one deployment, using training drops such as
           # sim.problem.make_scenario(config, seed, TRAINING_DROPS, deployment).
           # Optionally return per-epoch records for the learning-curve figure G12:
           return [{"round": e, "pf_utility": u, "cumulative_bytes": b}, ...]

       def solve(self, instance):
           return BeamSolution(W, V, U, tau, relay_assoc)
   ```

   If the class lives outside `experiments/algorithms.py`, import its module in `_load_late_registrations`.
2. **Give it a style** in `STYLE` in `eval/figures.py`, so it keeps its color and marker in every graph.
3. **Add it to an experiment** in `experiments/definitions.py`.
4. **Test it** under `tests/`, then run `python -m pytest -q`.

## How fitting works

The runner spreads drops round-robin over `--deployments` AP and relay layouts. A learning algorithm is fitted
once per deployment, in a worker process, and then solves that deployment's drops in the same process. The other
algorithms solve the same drops one by one. Training must use drop indices from `sim.problem.TRAINING_DROPS` or
`VALIDATION_DROPS`, never 0, 1, 2…, which are reserved for evaluation.

## The contract

`solve(instance)` receives a `sim.problem.ProblemInstance` and returns a `BeamSolution`:

| Field | Type and shape | Rule |
|---|---|---|
| `W` | complex `[M, K, N_A]` | Gain is `h^H w`; `w_mk = 0` if `c_mk = 0`; per-AP power budget |
| `V` | complex `[R, K, N_R]` | Gain is `g^H v`; `v_rk = 0` if `a_rk = 0`; per-relay power budget |
| `U` | complex `[R, K, N_R]` | Unit norm for associated pairs; gain is `u^H G w` |
| `tau` | float | In `[τ_min, τ_max]` |
| `relay_assoc` | int `[K]` | Relay of each UE, `-1` for none; at most `C_r^max` per relay |
| `info` | dict | Optional; saved with the results if JSON-friendly |

`evaluate_solution` raises an error on any violation, including NaN, and the runner records that drop as failed
rather than scoring it.

## Conventions that are easy to get wrong

- **Use only the estimates:** `instance.channels.h_hat`, `G_hat` and `g_hat`. The true channels exist for scoring.
  Nothing enforces this in code, so check it in review.
- **The matched beam is `w = h / ‖h‖`,** not `conj(h)` (`mrt_beam_directions`).
- **Channels are noise-normalized:** noise variance is 1 at every receiver, and power is in Watts.
- **Relayed streams target the relay in Phase I** (`ap_target_channels`).
- **SCIP must not share a process with PyTorch:** `exact_minlp` already runs SCIP in its own process. Never set
  `KMP_DUPLICATE_LIB_OK`.
