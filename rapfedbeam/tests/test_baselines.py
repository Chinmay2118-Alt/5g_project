"""
tests/test_baselines.py
Responsibility: confirm the WMMSE and exact baselines against closed forms, monotonicity,
feasibility, and each other.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from baselines.exact_minlp import small_instance_config, solve_exact
from baselines.heuristics import heuristic_solution
from baselines.wmmse import _Layout, _link_bound, centralized_wmmse, local_wmmse, wmmse_node
from eval.metrics import check_feasibility, evaluate_solution
from sim.problem import make_instance


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).parents[1] / "configs" / "config.yaml").read_text())


@pytest.fixture
def small(config):
    return small_instance_config(config)


def test_wmmse_bound_touches_the_rate_and_never_exceeds_it(small):
    # The whole AO rests on this: the bound equals log2(1 + SINR) at its own point, below it elsewhere.
    instance = make_instance(small, 1, 0)
    layout = _Layout(instance, np.full(2, -1))
    rng = np.random.default_rng(0)
    row = rng.normal(size=layout.size) + 1j * rng.normal(size=layout.size)
    z = rng.normal(size=layout.size) + 1j * rng.normal(size=layout.size)

    def true_rate(point):
        gains = [np.sum((row * point)[layout.stream == j]) for j in range(2)]
        return np.log2(1.0 + abs(gains[0]) ** 2 / (abs(gains[1]) ** 2 + 1.0))

    gains = [np.sum((row * z)[layout.stream == j]) for j in range(2)]
    total = sum(abs(g) ** 2 for g in gains) + 1.0
    a = gains[0] / total
    xi = 1.0 / (1.0 - abs(gains[0]) ** 2 / total)
    const, f, Q = _link_bound(row, 0, np.conj(a), abs(a) ** 2, xi, layout)

    def bound(point):
        x = np.concatenate([point.real, point.imag])
        return const + f @ x - np.sum((Q @ x) ** 2)

    assert np.isclose(bound(z), true_rate(z))
    for _ in range(200):
        other = rng.normal(size=layout.size) + 1j * rng.normal(size=layout.size)
        assert bound(other) <= true_rate(other) + 1e-9


def test_wmmse_node_single_user_matches_full_power_mrt():
    b = np.array([[1.0 + 1.0j, 2.0 - 1.0j]])
    w = wmmse_node(b, 1.0, 4.0, [1.0], [1.0], 1e-3)
    assert np.isclose(np.sum(np.abs(w) ** 2), 4.0)
    assert np.isclose(abs(np.conj(b[0]) @ w[0]), 2.0 * np.linalg.norm(b))


def test_wmmse_node_respects_budget_and_beats_equal_mrt():
    rng = np.random.default_rng(1)
    b = 5.0 * (rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2)))

    def utility(beams):
        gains = b.conj() @ beams.T
        power = np.abs(gains) ** 2
        rates = np.log2(1.0 + np.diag(power) / (power.sum(axis=1) - np.diag(power) + 1.0))
        return np.sum(np.log(rates + 1e-3))

    w = wmmse_node(b, 1.0, 1.0, np.ones(4), np.ones(4), 1e-3)
    assert np.sum(np.abs(w) ** 2) <= 1.0 + 1e-9
    assert utility(w) > utility(b / np.linalg.norm(b, axis=1, keepdims=True) * 0.5)


def test_centralized_utility_never_decreases_and_solution_is_feasible(config):
    instance = make_instance(config, 42, 0)
    solution = centralized_wmmse(instance, max_iters=8)
    history = np.array(solution.info["history"])
    assert np.all(np.diff(history) >= 0.0)
    check_feasibility(instance, solution)
    for k, relay_id in enumerate(solution.relay_assoc):
        if relay_id >= 0:
            assert np.isclose(np.linalg.norm(solution.U[relay_id, k]), 1.0)


def test_centralized_beats_heuristics_and_relays_help_blocked_users(config):
    instance = make_instance(config, 42, 1)
    central = evaluate_solution(instance, centralized_wmmse(instance))
    assert central["pf_utility"] > evaluate_solution(instance, heuristic_solution(instance))["pf_utility"]
    assert central["pf_utility"] > evaluate_solution(instance, heuristic_solution(instance, precoder="rzf"))["pf_utility"]
    no_relay = evaluate_solution(instance, centralized_wmmse(instance, use_relay=False))
    blocked = instance.scenario.blocked_mask
    assert central["rates"][blocked].mean() > no_relay["rates"][blocked].mean()


def test_no_relay_baseline_does_not_depend_on_the_number_of_relays(config):
    utilities = []
    for num_relays in (1, 3):
        config["network"]["num_relays"] = num_relays
        instance = make_instance(config, 42, 0)
        utilities.append(evaluate_solution(instance, heuristic_solution(instance, use_relay=False))["pf_utility"])
    assert np.isclose(utilities[0], utilities[1])


def test_centralized_handles_a_network_without_relayed_users(config):
    # The earlier rule relays blocked UEs only, so with none blocked no UE has a relay.
    config["association"]["policy"] = "blocked_strongest"
    config["blockage"]["frac_blocked_ues"] = 0.0
    instance = make_instance(config, 42, 2)
    solution = centralized_wmmse(instance, max_iters=3)
    check_feasibility(instance, solution)
    assert np.all(solution.V == 0)


def test_no_relay_baseline_uses_the_whole_interval(config):
    # Without relays there is no Phase II, so the APs transmit for all of it: tau = 1.
    instance = make_instance(config, 42, 5)
    solution = centralized_wmmse(instance, max_iters=3, use_relay=False)
    assert solution.tau == 1.0
    check_feasibility(instance, solution)
    # Rates of a single-phase network scale with tau, so the same beams at tau_max lose exactly that share.
    at_one = evaluate_solution(instance, solution)["rates"]
    capped = evaluate_solution(instance, type(solution)(solution.W, solution.V, solution.U, config["phase"]["tau_max"], solution.relay_assoc))["rates"]
    assert np.allclose(capped, at_one * config["phase"]["tau_max"])


def test_tau_of_one_is_rejected_when_a_ue_is_relayed(config):
    instance = make_instance(config, 42, 5)
    solution = centralized_wmmse(instance, max_iters=1)
    assert np.any(solution.relay_assoc >= 0)
    solution.tau = 1.0
    with pytest.raises(ValueError, match="tau"):
        check_feasibility(instance, solution)


def test_centralized_optimized_tau_stays_in_bounds_and_is_no_worse(config):
    instance = make_instance(config, 42, 3)
    fixed = centralized_wmmse(instance, max_iters=5, optimize_tau=False)
    tuned = centralized_wmmse(instance, max_iters=5, optimize_tau=True)
    check_feasibility(instance, tuned)
    assert config["phase"]["tau_min"] <= tuned.tau <= config["phase"]["tau_max"]
    assert tuned.info["history"][-1] >= fixed.info["history"][-1] - 1e-6


def test_energy_price_lowers_transmit_power(config):
    instance = make_instance(config, 42, 4)
    free = evaluate_solution(instance, centralized_wmmse(instance, max_iters=5))
    config["objective"]["lambda_e"] = 5.0
    priced = evaluate_solution(instance, centralized_wmmse(instance, max_iters=5))
    assert priced["transmit_power"] < free["transmit_power"]


def test_local_wmmse_uses_only_local_csi(config):
    # Changing relay r's links to UEs must not change any AP's beams: APs never see g_rk.
    instance = make_instance(config, 42, 5)
    first = local_wmmse(instance)
    instance.channels.g_hat = instance.channels.g_hat * 3.0 * np.exp(1j)
    second = local_wmmse(instance)
    assert np.allclose(first.W, second.W)
    # Changing another AP's channels must not change this AP's beams.
    instance = make_instance(config, 42, 5)
    instance.channels.h_hat[1:] *= 2.0
    instance.channels.G_hat[1:] *= 2.0
    third = local_wmmse(instance)
    assert np.allclose(first.W[0], third.W[0])
    check_feasibility(instance, third)


def test_exact_matches_closed_form_for_one_ap_one_user(small):
    small["association"]["policy"] = "blocked_strongest"   # no relay, so the direct link alone counts
    small["network"].update(num_aps=1, num_ues=1, cluster_top_l=1, num_relays=1)
    small["blockage"]["frac_blocked_ues"] = 0.0
    instance = make_instance(small, 3, 0)
    # The closed form is for the direct link alone; with association free, SCIP may (rightly) add the relay.
    solution = solve_exact(instance, time_limit_s=60, gap_limit=1e-4, optimize_association=False)
    snr =np.abs(instance.channels.h_hat[0, 0, 0]) ** 2 * small["power"]["p_max_ap"]
    expected = np.log(instance.tau * np.log2(1.0 + snr) + instance.epsilon_0)
    assert np.isclose(evaluate_solution(instance, solution)["pf_utility"], expected, atol=1e-2)
    assert solution.info["upper_bound"] >= expected - 1e-3


@pytest.mark.parametrize("drop", [0, 1])
def test_exact_bound_holds_and_agrees_with_the_simulator(small, drop):
    # Two independent implementations of P0 must agree: SCIP's model and the NumPy rate engine.
    # Exact holds tau fixed, so the bound applies only to methods that do the same.
    small["wmmse"]["optimize_tau"] = False
    instance = make_instance(small, 42, drop)
    exact = solve_exact(instance, time_limit_s=60)
    scored = evaluate_solution(instance, exact)["pf_utility"]
    assert np.isclose(scored, exact.info["incumbent"], atol=1e-2)
    central = evaluate_solution(instance, centralized_wmmse(instance))["pf_utility"]
    assert central <= exact.info["upper_bound"] + 1e-3
    assert evaluate_solution(instance, heuristic_solution(instance))["pf_utility"] <= exact.info["upper_bound"] + 1e-3


def test_exact_with_free_association_is_at_least_as_good(small):
    instance = make_instance(small, 42, 0)
    fixed = solve_exact(instance, time_limit_s=60, optimize_association=False)
    free = solve_exact(instance, time_limit_s=120, optimize_association=True)
    # Fixed association is one feasible choice of the free problem, so the free bound must cover it.
    assert free.info["upper_bound"] >= fixed.info["incumbent"] - 1e-3
    check_feasibility(instance, free)


def test_exact_refuses_networks_that_are_too_large(config):
    with pytest.raises(ValueError, match="too large"):
        solve_exact(make_instance(config, 42, 0))
