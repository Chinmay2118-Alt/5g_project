"""
tests/test_simulator.py
Responsibility: the fixes that every result depends on: beam conventions, link budget,
drop-level shadowing, aging, reproducibility, and the full rate pipeline.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from baselines.heuristics import heuristic_solution
from eval.metrics import check_feasibility, evaluate_solution
from sim.channels import apply_aging, noise_power_w, sample_full_channel_set
from sim.problem import BeamSolution, make_instance
from sim.rates import compute_full_rates, mmse_combiners, mrt_beam_directions, phase1_relay_decode_sinr, phase2_sinr_ue, rzf_beam_directions
from sim.scenario import assign_relay_association, generate_drop


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).parents[1] / "configs" / "config.yaml").read_text())


def complex_normal(rng, shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)


def test_mrt_gain_equals_channel_norm():
    h = complex_normal(np.random.default_rng(0), (3, 4, 2))
    gain = np.sum(np.conj(h) * mrt_beam_directions(h), axis=-1)
    assert np.allclose(gain, np.linalg.norm(h, axis=-1))


def test_rzf_nulls_other_users_when_regularization_vanishes():
    h = complex_normal(np.random.default_rng(1), (1, 2, 3))
    w = rzf_beam_directions(h, 1e-10)
    gains = np.conj(h[0]) @ w[0].T
    assert abs(gains[0, 1]) < 1e-6 and abs(gains[1, 0]) < 1e-6
    assert abs(gains[0, 0]) > 1e-3 and abs(gains[1, 1]) > 1e-3


def test_noise_power_matches_link_budget(config):
    # -174 dBm/Hz + 10 log10(20 MHz) + 7 dB = -93.99 dBm
    assert np.isclose(10.0 * np.log10(noise_power_w(config, "noise_figure_ue_db")) + 30.0, -93.99, atol=0.01)


def test_channel_power_matches_the_drops_large_scale_gain(config):
    rng = np.random.default_rng(2)
    scenario = generate_drop(config, rng)
    samples = np.array([np.abs(sample_full_channel_set(scenario, config, rng).h_mk[0, 0, 0]) ** 2 for _ in range(6000)])
    expected = 10.0 ** (scenario.beta_ap_ue_db[0, 0] / 10.0) / noise_power_w(config, "noise_figure_ue_db")
    assert abs(samples.mean() / expected - 1.0) < 0.1


def test_blockage_lowers_the_channel_used_for_blocked_ues(config):
    config["blockage"]["frac_blocked_ues"] = 0.5
    config["simulation"]["shadow_std_db"] = 0.0
    config["simulation"]["area_size"] = 1.0  # all distances equal the minimum distance
    scenario = generate_drop(config, np.random.default_rng(3))
    blocked = scenario.beta_ap_ue_db[:, scenario.blocked_mask]
    clear = scenario.beta_ap_ue_db[:, ~scenario.blocked_mask]
    assert np.isclose(clear.mean() - blocked.mean(), config["blockage"]["extra_loss_db"])


def test_aging_error_follows_the_papers_model():
    rng = np.random.default_rng(4)
    estimate = 50.0 * complex_normal(rng, (2000, 1, 1))
    true = apply_aging(estimate, 0.8, np.full((2000, 1), 50.0), rng)
    assert abs(np.mean(np.abs(true) ** 2) / 2500.0 - 1.0) < 0.1
    error = true - 0.8 * estimate
    assert abs(np.mean(np.abs(error) ** 2) / (0.36 * 2500.0) - 1.0) < 0.1


def test_same_seed_and_drop_give_identical_instances(config):
    first, second = make_instance(config, 7, 3), make_instance(config, 7, 3)
    assert np.array_equal(first.channels.h_mk, second.channels.h_mk)
    assert np.array_equal(first.scenario.relay_assoc, second.scenario.relay_assoc)
    assert not np.array_equal(first.channels.h_mk, make_instance(config, 7, 4).channels.h_mk)


def test_sweep_points_differ_only_in_what_the_sweep_changes(config):
    # A false trend appears in a graph if adding a relay reshuffles the APs, UEs or their channels.
    base = make_instance(config, 7, 3)
    config["network"]["num_relays"] = 4
    more_relays = make_instance(config, 7, 3)
    assert np.array_equal(base.scenario.ue_positions, more_relays.scenario.ue_positions)
    assert np.array_equal(base.scenario.ap_positions, more_relays.scenario.ap_positions)
    assert np.array_equal(base.scenario.beta_ap_ue_db, more_relays.scenario.beta_ap_ue_db)
    assert np.array_equal(base.channels.h_mk, more_relays.channels.h_mk)

    config["network"]["num_relays"] = 2
    config["network"]["num_ues"] = 12
    more_ues = make_instance(config, 7, 3)
    assert np.array_equal(base.scenario.ue_positions, more_ues.scenario.ue_positions[:8])
    assert np.array_equal(base.scenario.ap_positions, more_ues.scenario.ap_positions)


def test_relay_capacity_limit_is_enforced():
    beta = np.array([[0.0, -1.0, -2.0], [-20.0, -21.0, -22.0]])
    assoc = assign_relay_association(None, None, np.array([True, True, True]), beta, max_streams=2)
    assert list(assoc) == [0, 0, 1]


def test_mmse_combiner_beats_random_unit_combiners():
    rng = np.random.default_rng(5)
    G, w = complex_normal(rng, (2, 1, 3, 2)), complex_normal(rng, (2, 3, 2))
    u = mmse_combiners(G, w, 1.0)
    assert np.allclose(np.linalg.norm(u, axis=-1), 1.0)
    assoc = np.zeros(3, dtype=int)
    gamma = phase1_relay_decode_sinr(G, w, u, assoc, 1.0)
    for _ in range(200):
        candidate = complex_normal(rng, (1, 3, 3))
        candidate /= np.linalg.norm(candidate, axis=-1, keepdims=True)
        assert np.all(phase1_relay_decode_sinr(G, w, candidate, assoc, 1.0) <= gamma + 1e-9)


def test_phase2_adds_relays_coherently_as_in_the_paper():
    g = np.array([[[1.0 + 0j]], [[1.0 + 0j]]])
    v = np.array([[[1.0 + 0j]], [[1.0 + 0j]]])
    assert np.isclose(phase2_sinr_ue(g, v, np.array([0]), 1.0)[0], 4.0)


def test_full_pipeline_against_hand_computed_rates(config):
    # 1 AP, 1 relay, 2 UEs with single antennas: every SINR can be written down by hand.
    instance = make_instance(config, 1, 0)
    scenario = instance.scenario
    scenario.cluster_mask = np.array([[True, True]])
    scenario.relay_assoc = np.array([0, -1])
    channels = instance.channels
    channels.h_mk = np.array([[[1.0 + 0j], [2.0 + 0j]]])
    channels.G_mr = np.array([[[[3.0 + 0j]]]])
    channels.g_rk = np.array([[[2.0 + 0j], [1.0 + 0j]]])
    W = np.array([[[1.0 + 0j], [0.5 + 0j]]])
    V = np.array([[[1.0 + 0j], [0j]]])
    U = np.array([[[1.0 + 0j], [1.0 + 0j]]])
    rates, debug = compute_full_rates(scenario, channels, W, V, U, 0.25)

    gamma1 = np.array([1.0 / (0.25 + 1.0), 1.0 / (4.0 + 1.0)])  # |h w_k|^2 / (|h w_j|^2 + 1)
    gamma2_ue0 = 4.0 / 1.0
    gamma_ar = 9.0 / (2.25 + 1.0)
    assert np.allclose(debug["gamma1"], gamma1)
    assert np.isclose(debug["gamma2"][0], gamma2_ue0) and debug["gamma2"][1] == 0.0
    assert np.isclose(debug["gamma_AR"][0, 0], gamma_ar)
    expected_0 = min(0.25 * np.log2(1 + gamma1[0]) + 0.75 * np.log2(1 + gamma2_ue0), 0.25 * np.log2(1 + gamma_ar))
    assert np.isclose(rates[0], expected_0)
    assert np.isclose(rates[1], 0.25 * np.log2(1 + gamma1[1]))


def test_sinr_formulas_match_a_signal_level_simulation(config):
    # Independent of the formulas: transmit random symbols through the paper's received-signal
    # equations (Sec. IX, X, XIII), and measure each SINR from the samples.
    instance = make_instance(config, 3, 0)
    solution = heuristic_solution(instance)
    scenario, channels = instance.scenario, instance.channels
    _, debug = compute_full_rates(scenario, channels, solution.W, solution.V, solution.U, solution.tau)
    rng = np.random.default_rng(6)
    samples, num_ues = 200_000, config["network"]["num_ues"]
    symbols = complex_normal(rng, (samples, num_ues))

    def measured_sinr(desired_gain, total_received):
        # Whatever is uncorrelated with the desired symbol counts as interference plus noise.
        signal = desired_gain * symbols[:, k]
        return np.mean(np.abs(signal) ** 2) / np.mean(np.abs(total_received - signal) ** 2)

    tx_ap = np.einsum("sk,mka->sma", symbols, solution.W)
    tx_relay = np.einsum("sk,rkn->srn", symbols, solution.V)
    for k in range(num_ues):
        noise = complex_normal(rng, (samples,))
        phase1 = np.einsum("ma,sma->s", np.conj(channels.h_mk[:, k]), tx_ap) + noise
        gain1 = np.einsum("ma,ma->", np.conj(channels.h_mk[:, k]), solution.W[:, k])
        assert abs(measured_sinr(gain1, phase1) / debug["gamma1"][k] - 1.0) < 0.05

        relay_id = scenario.relay_assoc[k]
        if relay_id < 0:
            continue
        noise_vec = complex_normal(rng, (samples, config["network"]["n_antennas_relay"]))
        at_relay = np.einsum("mna,sma->sn", channels.G_mr[:, relay_id], tx_ap) + noise_vec
        combined = at_relay @ np.conj(solution.U[relay_id, k])
        gain_ar = np.conj(solution.U[relay_id, k]) @ np.einsum("mna,ma->n", channels.G_mr[:, relay_id], solution.W[:, k])
        assert abs(measured_sinr(gain_ar, combined) / debug["gamma_AR"][relay_id, k] - 1.0) < 0.05

        phase2 = np.einsum("rn,srn->s", np.conj(channels.g_rk[:, k]), tx_relay) + complex_normal(rng, (samples,))
        gain2 = np.einsum("rn,rn->", np.conj(channels.g_rk[:, k]), solution.V[:, k])
        assert abs(measured_sinr(gain2, phase2) / debug["gamma2"][k] - 1.0) < 0.05


def test_rates_are_scored_on_true_channels_not_estimates(config):
    config["aging"]["enabled"] = True
    config["aging"].update(rho_h=0.5, rho_G=0.5, rho_g=0.5)
    instance = make_instance(config, 1, 0)
    assert not np.allclose(instance.channels.h_mk, instance.channels.h_hat)
    solution = heuristic_solution(instance)
    expected, _ = compute_full_rates(instance.scenario, instance.channels, solution.W, solution.V, solution.U, solution.tau)
    assert np.allclose(evaluate_solution(instance, solution)["rates"], expected)


def test_realistic_link_budget_gives_sensible_rates(config):
    rates = np.concatenate([evaluate_solution(inst, heuristic_solution(inst))["rates"] for inst in (make_instance(config, 42, d) for d in range(5))])
    assert 0.1 < rates.mean() < 10.0


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s.W.__imul__(100.0), "AP power"),
    (lambda s: s.V.__imul__(100.0), "relay power"),
])
def test_feasibility_check_rejects_power_violations(config, mutate, message):
    instance = make_instance(config, 1, 0)
    solution = heuristic_solution(instance)
    mutate(solution)
    with pytest.raises(ValueError, match=message):
        check_feasibility(instance, solution)


def test_feasibility_check_rejects_beams_outside_cluster_and_association(config):
    instance = make_instance(config, 1, 0)
    solution = heuristic_solution(instance)
    m, k = np.argwhere(~instance.scenario.cluster_mask)[0]
    solution.W = solution.W * 0.5  # leaves power headroom, so only the cluster rule can trigger
    solution.W[m, k, 0] = 0.01
    with pytest.raises(ValueError, match="outside its cluster"):
        check_feasibility(instance, solution)

    solution = heuristic_solution(instance)
    solution.V = solution.V * 0.5
    solution.V[0, np.flatnonzero(solution.relay_assoc != 0)[0], 0] = 0.01
    with pytest.raises(ValueError, match="not associated"):
        check_feasibility(instance, solution)


def test_beam_solution_shape_is_checked(config):
    instance = make_instance(config, 1, 0)
    good = heuristic_solution(instance)
    bad = BeamSolution(good.W[:, :, :1], good.V, good.U, good.tau, good.relay_assoc)
    with pytest.raises(ValueError, match="shape"):
        check_feasibility(instance, bad)
