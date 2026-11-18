"""The diagnostic ablations added for Google Deep Review round 2, and the time-averaged power."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from eval.metrics import evaluate_solution
from experiments.algorithms import get_algorithm
from sim.problem import make_instance
import train.rap_fedwmmse  # noqa: F401  registers the ablations


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).parents[1] / "configs" / "config.yaml").read_text())


def test_diagnostic_ablations_solve_feasibly_and_power_is_time_averaged(config):
    config["network"].update(num_aps=3, num_relays=1, num_ues=3, cluster_top_l=2)
    config["fedwmmse"].update(warmup_epochs=1, warmup_blocks=1)
    instance = make_instance(config, 1, 0, 0)
    for name in ("unfolded_wmmse", "classical_wmmse_layers", "unfolded_wmmse_slow", "classical_wmmse_slow"):
        solution = get_algorithm(name).solve(instance)
        metrics = evaluate_solution(instance, solution)   # raises on any infeasible solution
        assert np.isfinite(metrics["pf_utility"]), name
        # Each transmitter radiates only in its own phase, so the block average cannot exceed P_tot.
        assert metrics["average_power"] <= metrics["total_power"] + 1e-9, name


def test_undamped_layers_differ_from_the_damped_ablation(config):
    config["network"].update(num_aps=3, num_relays=1, num_ues=3, cluster_top_l=2)
    instance = make_instance(config, 2, 0, 0)
    damped = get_algorithm("unfolded_wmmse").solve(instance)
    undamped = get_algorithm("classical_wmmse_layers").solve(instance)
    assert not np.allclose(damped.W, undamped.W)
