"""
experiments/runner.py
Responsibility: run algorithms over sweep points and identical drops, and save per-drop results.
Figures are drawn from the saved results only (eval/figures.py).
"""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import time
import traceback

import numpy as np

from eval.metrics import evaluate_solution
from experiments.algorithms import get_algorithm
from sim.problem import make_instance


def apply_overrides(config, overrides):
    """Deep copy of config with dotted keys such as 'network.num_ues' replaced."""
    result = copy.deepcopy(config)
    for dotted, value in overrides.items():
        node = result
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        if leaf not in node:
            raise KeyError(f"unknown config key: {dotted}")
        node[leaf] = value
    return result


def deployment_of(drop_idx, deployments):
    """Drops are dealt round-robin to deployments; without deployments each drop is its own."""
    return None if not deployments else drop_idx % deployments


def _score(algorithm, config, seed, drop_idx, deployment):
    instance = make_instance(config, seed, drop_idx, deployment)
    started = time.perf_counter()
    solution = algorithm.solve(instance)
    seconds = time.perf_counter() - started
    metrics = evaluate_solution(instance, solution)
    return {
        "drop": drop_idx,
        "deployment": drop_idx if deployment is None else deployment,
        "pf_utility": metrics["pf_utility"],
        "sum_rate": metrics["sum_rate"],
        "rates": metrics["rates"].tolist(),
        "transmit_power": metrics["transmit_power"],
        "total_power": metrics["total_power"],
        "seconds": seconds,
        "info": {key: value for key, value in solution.info.items()},
    }


def _run_drop(task):
    # Top level so worker processes can import it. A failed drop is recorded, never hidden.
    name, config, seed, drop_idx, deployment = task
    try:
        return "drop", name, _score(get_algorithm(name), config, seed, drop_idx, deployment)
    except Exception:
        return "drop", name, {"drop": drop_idx, "error": traceback.format_exc()}


def _run_fit(task):
    """Fit on one deployment, then solve that deployment's drops in the same process, since the
    trained model never has to travel between processes this way."""
    name, config, seed, deployment, drop_ids, deployments = task
    started = time.perf_counter()
    try:
        algorithm = get_algorithm(name)
        history = algorithm.fit(config, seed, deployment)
    except Exception:
        error = traceback.format_exc()
        return "fit", name, (deployment, None, 0.0, [{"drop": d, "error": error} for d in drop_ids])
    fit_seconds = time.perf_counter() - started
    records = []
    for drop_idx in drop_ids:
        try:
            records.append(_score(algorithm, config, seed, drop_idx, deployment_of(drop_idx, deployments)))
        except Exception:
            records.append({"drop": drop_idx, "error": traceback.format_exc()})
    return "fit", name, (deployment, history, fit_seconds, records)


def run_point(config, algorithm_names, drops, seed, workers=1, deployments=None):
    """Every algorithm gets the same drops at this sweep point. A learning algorithm is fitted once
    per deployment and then solves that deployment's drops."""
    results = {name: [] for name in algorithm_names}
    fit_history, fit_seconds = {}, {}
    tasks = []
    for name in algorithm_names:
        if get_algorithm(name).needs_fit:
            groups = deployments or drops
            for group in range(groups):
                drop_ids = [d for d in range(drops) if (deployment_of(d, deployments) if deployments else d) == group]
                if drop_ids:
                    tasks.append((_run_fit, (name, config, seed, group, drop_ids, deployments)))
        else:
            tasks.extend((_run_drop, (name, config, seed, d, deployment_of(d, deployments))) for d in range(drops))
    if workers > 1 and tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            finished = [future.result() for future in [pool.submit(function, task) for function, task in tasks]]
    else:
        finished = [function(task) for function, task in tasks]
    for kind, name, payload in finished:
        if kind == "drop":
            results[name].append(payload)
            continue
        deployment, history, seconds, records = payload
        results[name].extend(records)
        if history:
            fit_history.setdefault(name, []).append({"deployment": deployment, "history": history})
        fit_seconds.setdefault(name, []).append(seconds)
    for name in results:
        results[name].sort(key=lambda record: record["drop"])
    return results, fit_history, fit_seconds


def run_experiment(experiment, base_config, algorithm_names, drops, seed, workers=1, out_dir="results", deployments=None):
    names = [name for name in algorithm_names if name not in experiment.get("exclude", ())]
    base = experiment["prepare"](base_config) if experiment.get("prepare") else base_config
    base = apply_overrides(base, experiment.get("base_overrides", {}))
    points = []
    for x_value, overrides in experiment["points"]:
        started = time.perf_counter()
        results, fit_history, fit_seconds = run_point(apply_overrides(base, overrides), names, drops, seed, workers, deployments)
        points.append({"x": x_value, "overrides": overrides, "results": results, "fit_history": fit_history, "fit_seconds": fit_seconds})
        errors = sum("error" in record for records in results.values() for record in records)
        print(f"  {experiment['name']} x={x_value}: {time.perf_counter() - started:.1f}s, failed drops: {errors}", flush=True)
    payload = {
        "experiment": experiment["name"],
        "x_label": experiment["x_label"],
        "seed": seed,
        "drops": drops,
        "deployments": deployments,
        "algorithms": names,
        "points": points,
        "config": base,
    }
    path = Path(out_dir) / f"{experiment['name']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=_json_default))
    return payload


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot store {type(value)} in results")
