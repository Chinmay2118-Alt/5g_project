"""
tests/test_experiments.py
Responsibility: the harness every algorithm runs through. The dummy classes below exercise the
same path RAP-FedWMMSE takes, including fit() per deployment with an epoch history.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from baselines.heuristics import heuristic_solution
from eval.figures import draw_all
from experiments.algorithms import Algorithm, available_algorithms, register
from experiments.definitions import BASELINE_ALGORITHMS, DEFAULT_ALGORITHMS, EXPERIMENTS, GAP_ALGORITHMS
from experiments.runner import apply_overrides, run_experiment, run_point
from sim.problem import make_instance


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).parents[1] / "configs" / "config.yaml").read_text())


@register
class RecordingAlgorithm(Algorithm):
    """Stands in for a learning algorithm: fits per deployment, then solves that deployment's drops."""

    name = "test_recording"
    label = "Test recording"
    needs_fit = True
    seen = []

    def fit(self, config, seed, deployment=0):
        self.deployment = deployment
        return [{"round": e, "pf_utility": float(e), "cumulative_bytes": 100 * e} for e in range(1, 4)]

    def solve(self, instance):
        RecordingAlgorithm.seen.append((self.deployment, instance.channels.h_mk.copy(), instance.scenario.ap_positions.copy()))
        return heuristic_solution(instance)


@register
class InfeasibleAlgorithm(Algorithm):
    name = "test_infeasible"
    label = "Test infeasible"

    def solve(self, instance):
        solution = heuristic_solution(instance)
        solution.W = solution.W * 100.0
        return solution


def test_every_listed_algorithm_is_registered_and_available():
    for name in set(DEFAULT_ALGORITHMS + BASELINE_ALGORITHMS + GAP_ALGORITHMS):
        assert name in available_algorithms(), name


def test_overrides_copy_the_config_and_reject_unknown_keys(config):
    changed = apply_overrides(config, {"network.num_ues": 5})
    assert changed["network"]["num_ues"] == 5 and config["network"]["num_ues"] == 8
    with pytest.raises(KeyError):
        apply_overrides(config, {"network.num_uez": 5})


def test_every_experiment_override_exists_in_the_config(config):
    for experiment in EXPERIMENTS.values():
        base = experiment["prepare"](config) if experiment.get("prepare") else config
        base = apply_overrides(base, experiment.get("base_overrides", {}))
        for _, overrides in experiment["points"]:
            apply_overrides(base, overrides)


def test_drops_of_a_deployment_share_infrastructure_and_differ_in_ues(config):
    first, second, other = make_instance(config, 9, 0, 0), make_instance(config, 9, 2, 0), make_instance(config, 9, 1, 1)
    assert np.array_equal(first.scenario.ap_positions, second.scenario.ap_positions)
    assert np.array_equal(first.scenario.relay_positions, second.scenario.relay_positions)
    assert not np.array_equal(first.scenario.ue_positions, second.scenario.ue_positions)
    assert not np.array_equal(first.scenario.ap_positions, other.scenario.ap_positions)


def test_learning_algorithms_fit_once_per_deployment_and_see_the_same_drops(config):
    RecordingAlgorithm.seen = []
    results, history, _ = run_point(config, ["test_recording"], drops=4, seed=9, deployments=2)
    assert len(history["test_recording"]) == 2
    for deployment, channels, ap_positions in RecordingAlgorithm.seen:
        matches = [d for d in range(4) if d % 2 == deployment and np.array_equal(channels, make_instance(config, 9, d, deployment).channels.h_mk)]
        assert len(matches) == 1
        assert np.array_equal(ap_positions, make_instance(config, 9, matches[0], deployment).scenario.ap_positions)
    assert [record["drop"] for record in results["test_recording"]] == [0, 1, 2, 3]


def test_infeasible_solution_is_recorded_as_a_failure_not_scored(config):
    results, _, _ = run_point(config, ["test_infeasible"], drops=1, seed=1)
    assert "AP power constraint violated" in results["test_infeasible"][0]["error"]


def test_parallel_and_serial_runs_give_the_same_numbers(config):
    # RAP-FedWMMSE trains in a worker process here, so this also checks its training is reproducible.
    config["network"].update(num_aps=4, num_relays=2, num_ues=4, cluster_top_l=2)
    config["fedwmmse"].update(epochs=1, local_steps=1, batch_blocks=2, layers=2, validation_instances=1, warmup_epochs=1, warmup_blocks=1)
    names = ["mrt_equal", "rap_fedwmmse"]
    serial, _, _ = run_point(config, names, drops=2, seed=5, workers=1, deployments=2)
    parallel, _, _ = run_point(config, names, drops=2, seed=5, workers=2, deployments=2)
    for name in names:
        assert "error" not in serial[name][0], serial[name][0].get("error")
        assert [r["pf_utility"] for r in serial[name]] == [r["pf_utility"] for r in parallel[name]]


def test_new_algorithm_flows_through_to_saved_results(config, tmp_path):
    experiment = {"name": "baseline", "x_label": "Default", "points": [("default", {})]}
    run_experiment(experiment, config, ["test_recording", "mrt_equal"], 2, 1, 1, tmp_path, deployments=2)
    saved = json.loads((tmp_path / "baseline.json").read_text())
    runs = saved["points"][0]["fit_history"]["test_recording"]
    assert sorted(run["deployment"] for run in runs) == [0, 1]
    assert runs[0]["history"][2]["cumulative_bytes"] == 300
    assert len(saved["points"][0]["results"]["test_recording"]) == 2


def test_every_figure_draws_and_writes_its_numbers(config, tmp_path):
    from eval import figures
    figures.STYLE["test_recording"] = ("#2a78d6", "o", "-")
    small = {"fedwmmse.epochs": 1, "fedwmmse.local_steps": 1, "fedwmmse.batch_blocks": 2, "fedwmmse.validation_instances": 1,
             "fedwmmse.warmup_epochs": 1, "fedwmmse.warmup_blocks": 1, "fedwmmse.validation_every": 1, "wmmse.max_iters": 3}
    quick = apply_overrides(config, small)
    run_experiment({"name": "baseline", "x_label": "Default", "points": [("default", {})]}, quick,
                   ["rap_fedwmmse", "test_recording", "centralized_wmmse", "mrt_equal"], 2, 1, 1, tmp_path, deployments=1)
    run_experiment({"name": "tau", "x_label": "τ", "points": [(0.4, {"phase.tau_fixed": 0.4}), (0.6, {"phase.tau_fixed": 0.6})]},
                   quick, ["mrt_equal", "local_wmmse"], 2, 1, 1, tmp_path)
    layers = dict(EXPERIMENTS["layers"], points=[(1, {"fedwmmse.layers": 1}), (2, {"fedwmmse.layers": 2})])
    run_experiment(layers, quick, ["unfolded_wmmse", "rap_fedwmmse"], 1, 1, 1, tmp_path, deployments=1)
    run_experiment(dict(EXPERIMENTS["optimality_gap"]), quick, ["exact_minlp", "mrt_equal"], 1, 1, 1, tmp_path)
    drawn, _ = draw_all(tmp_path, tmp_path / "figures")
    for stem in ("G01_rate_cdf", "G02_outage", "G08_pf_vs_tau", "G10_pf_vs_layers", "G11_optimality_gap", "G13_fl_epochs", "G14_fl_bytes"):
        assert (tmp_path / "figures" / f"{stem}.png").exists(), stem
        assert (tmp_path / "figures" / f"{stem}.csv").exists(), stem


def test_neural_baseline_still_trains_and_scores_on_the_new_simulator(config):
    config["network"].update(num_aps=3, num_relays=1, num_ues=3, cluster_top_l=2)
    config["training"].update(epochs=2, batch_drops=1)
    results, _, _ = run_point(config, ["neural_ctde"], drops=1, seed=1)
    record = results["neural_ctde"][0]
    assert "error" not in record and np.isfinite(record["pf_utility"])


def test_federated_modes_start_from_the_same_weights(config):
    from train.federated_loop import run_federated_experiment
    config["network"].update(num_aps=2, num_relays=1, num_ues=3, cluster_top_l=2)
    config["fl"].update(rounds=1, local_steps_per_round=1)
    config["training"]["batch_drops"] = 1
    _, local = run_federated_experiment(config, "local_only")
    _, again = run_federated_experiment(config, "local_only")
    for key, value in local["AP0"].state_dict().items():
        assert np.array_equal(value.numpy(), again["AP0"].state_dict()[key].numpy())
    history, _ = run_federated_experiment(config, "topology_kernel")
    assert np.isfinite(history[0][1]) and history[0][2] > 0
