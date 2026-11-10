"""
tests/test_rap_fedwmmse.py
Responsibility: RAP-FedWMMSE, step by step: the local gradient equals the global one, every layer is
feasible, the neutral update is classical WMMSE, the slow-timescale and federated building blocks
are optimal or exact where they claim to be, and the trained policy is feasible and reproducible.
"""

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from eval.metrics import check_feasibility, evaluate_solution
from experiments.algorithms import get_algorithm
from fl.communication import personalized_updates, select_clients, top_k
from sim.association import solve_matching
from sim.problem import TRAINING_DROPS, VALIDATION_DROPS, make_instance, make_scenario
from sim.scenario import evolve_drop
from train.rap_fedwmmse import RapFedWMMSETrainer, fixed_params
from train.unfolded_wmmse import (RDTYPE, ap_gradient, augmented_objective, coordination, initial_beams, network_from,
                                  outage_surrogate, power_multiplier, relay_gradient, unfold, user_rates)


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).parents[1] / "configs" / "config.yaml").read_text())


def quick(config):
    config["network"].update(num_aps=4, num_relays=2, num_ues=4, cluster_top_l=2)
    config["fedwmmse"].update(epochs=2, local_steps=1, batch_blocks=2, layers=2, validation_instances=1,
                              warmup_epochs=1, warmup_blocks=1, validation_every=1)
    return config


def random_state(config, seed=0):
    instance = make_instance(config, 42, 0, 0)
    net = network_from(instance.scenario, [instance.channels], config, instance.scenario.relay_assoc, 0.5)
    torch.manual_seed(seed)
    M, R, K = net.h.shape[1], net.g.shape[1], net.h.shape[2]
    W, V = initial_beams(net, torch.randn(1, M, K, dtype=RDTYPE), torch.randn(1, R, K, dtype=RDTYPE))
    duals = (torch.rand(1, K, dtype=RDTYPE), torch.rand(1, R, K, dtype=RDTYPE))
    return instance, net, W, V, duals


def test_local_gradients_equal_the_gradient_of_the_global_objective(config):
    # The central claim: each node's update needs only its own CSI plus zeta, yet follows dJ/dW.
    # PyTorch returns 2 dJ/dW* for a real function of complex inputs, hence the factor 2.
    _, net, W0, V0, duals = random_state(config)
    W, V = W0.clone().requires_grad_(True), V0.clone().requires_grad_(True)
    augmented_objective(net, W, V, duals, 10.0)[0].sum().backward()
    z = coordination(net, W0, V0, duals, 10.0)
    ap_grad, _ = ap_gradient(net.h, net.G, W0, z, net)
    relay_grad, _ = relay_gradient(net.g, V0, z, net)
    served = net.cluster[None, :, :, None].bool().expand_as(ap_grad)
    relayed = net.assoc[None, :, :, None].bool().expand_as(relay_grad)
    assert relayed.any()
    assert torch.allclose(W.grad[served], 2 * ap_grad[served], rtol=1e-6, atol=1e-10)
    assert torch.allclose(V.grad[relayed], 2 * relay_grad[relayed], rtol=1e-6, atol=1e-10)


def test_an_aps_update_does_not_read_another_aps_csi(config):
    _, net, W, V, duals = random_state(config)
    z = coordination(net, W, V, duals, 10.0)
    before, _ = ap_gradient(net.h, net.G, W, z, net)
    net.h[:, 1:] *= 3.0
    net.G[:, 1:] *= 3.0
    after, _ = ap_gradient(net.h, net.G, W, z, net)   # zeta held fixed, as it would be in one layer
    assert torch.allclose(before[:, 0], after[:, 0])
    assert not torch.allclose(before[:, 1], after[:, 1])


def test_every_layer_is_feasible(config):
    _, net, W, V, duals = random_state(config)
    params = fixed_params(net, 1)
    with torch.no_grad():
        for _ in range(5):
            W, V, duals = unfold(net, W, V, params, duals, 1, 10.0, 0.2, 0.2)
            assert torch.all((W.abs() ** 2).sum(dim=(2, 3)) <= net.p_ap * (1 + 1e-9))
            assert torch.all((V.abs() ** 2).sum(dim=(2, 3)) <= net.p_relay * (1 + 1e-9))
            assert torch.all(W[:, net.cluster == 0] == 0) and torch.all(V[:, net.assoc == 0] == 0)
            assert torch.all(duals[0] >= 0) and torch.all(duals[1] >= 0)


def test_neutral_layer_is_the_classical_wmmse_update(config):
    # With step 1, damping 1 and reg 1, a layer must equal the textbook WMMSE beam update, written
    # out here independently: w = (A |c|^2 h h^H + mu I)^-1 A c h, with mu the power price.
    config["association"]["policy"] = "blocked_strongest"
    config["network"].update(num_aps=1, num_relays=1, num_ues=1, cluster_top_l=1)
    config["blockage"]["frac_blocked_ues"] = 0.0
    config["power"]["p_max_ap"] = 1e-6   # low enough that the power constraint binds
    instance = make_instance(config, 3, 0, 0)
    net = network_from(instance.scenario, [instance.channels], config, instance.scenario.relay_assoc, 0.5)
    W = torch.zeros_like(net.h)
    W[0, 0, 0, 0] = 1e-4
    zeros = (torch.zeros(1, 1, dtype=RDTYPE), torch.zeros(1, 1, 1, dtype=RDTYPE))
    with torch.no_grad():
        updated, _, _ = unfold(net, W, torch.zeros_like(net.g), fixed_params(net, 1, damping=1.0), zeros, 1, None, 0.0, 0.0)

    h, w = net.h[0, 0, 0].numpy(), W[0, 0, 0].numpy()
    s = np.vdot(h, w)
    c = s / (abs(s) ** 2 + 1.0)
    rate = 0.5 * np.log2(1 + abs(s) ** 2)
    A = (1.0 / (rate + 1e-3)) * 0.5 * (1.0 / (1.0 - abs(s) ** 2 / (abs(s) ** 2 + 1.0) + 1e-9)) / np.log(2)
    Q, b = A * abs(c) ** 2 * np.outer(h, h.conj()), A * c * h
    low, high = 0.0, 1e12
    for _ in range(200):
        mu = 0.5 * (low + high)
        if np.sum(np.abs(np.linalg.solve(Q + mu * np.eye(2), b)) ** 2) > config["power"]["p_max_ap"]:
            low = mu
        else:
            high = mu
    expected = np.linalg.solve(Q + high * np.eye(2), b)
    assert np.allclose(updated[0, 0, 0].numpy(), expected, rtol=1e-5)
    assert np.isclose(abs(np.vdot(h, expected)), np.linalg.norm(h) * np.linalg.norm(expected))   # MRT direction


def test_power_multiplier_meets_the_budget_exactly_when_it_binds():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    Q = torch.as_tensor(A @ A.conj().T)[None]
    B = torch.as_tensor(rng.normal(size=(3, 2)) * 50 + 0j)[None]
    mu = power_multiplier(Q, B, 1.0)
    X = torch.linalg.solve(Q + mu[:, None, None] * torch.eye(3, dtype=Q.dtype), B)
    assert mu.item() > 0 and np.isclose(float((X.abs() ** 2).sum()), 1.0, rtol=1e-6)
    assert power_multiplier(Q, B * 1e-6, 1.0).item() < 1e-6


def test_matching_is_optimal_and_respects_l_max_and_capacity():
    rng = np.random.default_rng(2)
    for _ in range(20):
        scores = rng.normal(size=(2, 4))
        result = solve_matching(scores, 1, 2)
        assert np.all(result.sum(axis=0) <= 1) and np.all(result.sum(axis=1) <= 2)
        best = max((sum(scores[r, k] for k, r in enumerate(choice) if r >= 0)
                    for choice in product([-1, 0, 1], repeat=4)
                    if all(sum(1 for r in choice if r == rr) <= 2 for rr in (0, 1))), default=0.0)
        assert np.isclose((scores * result).sum(), best)


def test_matching_never_relays_a_weak_relay_hop_unless_qos_demands_it(config):
    from sim.association import association_features, matched_association
    for deployment in range(5):
        instance = make_instance(config, 42, 0, deployment)
        beta_ru, _, _ = association_features(instance.scenario)
        assoc = matched_association(instance.scenario, config)
        for k, r in enumerate(assoc):
            if r >= 0:
                assert beta_ru[r, k] > 0.0
    weak = np.argwhere(beta_ru.max(axis=0, keepdims=True) <= 0.0)
    if len(weak):
        prices = np.zeros(config["network"]["num_ues"])
        prices[weak[0][1]] = 10.0
        assert matched_association(instance.scenario, config, prices)[weak[0][1]] >= 0


def test_qos_price_makes_a_relay_more_attractive(config):
    from sim.association import relay_scores
    instance = make_instance(config, 42, 0, 0)
    prices = np.zeros(config["network"]["num_ues"])
    prices[2] = 3.0
    assert np.all(relay_scores(instance.scenario, config, prices)[:, 2] > relay_scores(instance.scenario, config)[:, 2])


def test_client_selection_solves_the_knapsack():
    rng = np.random.default_rng(3)
    for _ in range(20):
        scores, sizes = rng.uniform(0, 1, 6), rng.integers(1, 5, 6).astype(float)
        eligible = rng.uniform(size=6) > 0.3
        chosen = select_clients(scores, eligible, sizes, 7.0)
        assert np.all(~chosen | eligible) and sizes[chosen].sum() <= 7.0
        best = max((scores[list(s)].sum() for n in range(7) for s in combinations(range(6), n)
                    if all(eligible[i] for i in s) and sizes[list(s)].sum() <= 7.0), default=0.0)
        assert np.isclose(scores[chosen].sum(), best)


def test_error_feedback_loses_nothing_over_rounds():
    rng = np.random.default_rng(4)
    residual, sent_total, update_total = np.zeros(50), np.zeros(50), np.zeros(50)
    for _ in range(10):
        update = rng.normal(size=50)
        sent, residual = top_k(update, residual, 0.1)
        assert np.count_nonzero(sent) == 5
        sent_total += sent
        update_total += update
    assert np.allclose(sent_total + residual, update_total)


def test_personalized_weights_reduce_to_averaging_for_identical_nodes():
    signatures = {i: np.ones(4) for i in "abc"}
    roles = {i: "AP" for i in "abc"}
    updates = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]), "c": np.array([5.0, 5.0])}
    kappa = {"kappa_ap_ap": 1.0, "kappa_relay_relay": 1.0, "kappa_ap_relay": 0.3}
    result = personalized_updates(signatures, roles, updates, {"a": True, "b": True, "c": False}, 1.0, kappa, {i: 1.0 for i in "abc"})
    for node in "abc":
        assert np.allclose(result[node], [0.5, 0.5])   # c was not selected, so its update is not used


def test_outage_surrogate_with_perfect_csi_scores_the_nominal_rates(config):
    _, net, W, V, duals = random_state(config)
    z = coordination(net, W, V, duals, 10.0)
    p_out = outage_surrogate(net, W, V, z.u, 1.0, 2, 10.0, 10.0, torch.Generator().manual_seed(0))
    assert torch.allclose(p_out, torch.sigmoid(10.0 * (net.r_min - z.rate)))


def test_epochs_move_ues_but_not_the_infrastructure(config):
    scenario, _ = make_scenario(config, 42, TRAINING_DROPS, 0)
    later = evolve_drop(scenario, config, np.random.default_rng(0))
    assert np.array_equal(scenario.ap_positions, later.ap_positions)
    assert np.array_equal(scenario.beta_ap_relay_db, later.beta_ap_relay_db)
    assert np.array_equal(scenario.blocked_mask, later.blocked_mask)
    moved = np.linalg.norm(later.ue_positions - scenario.ue_positions, axis=1)
    assert np.all(moved > 0) and np.all(moved < 10 * config["mobility"]["step_m"])
    shadow_change = np.abs(later.beta_ap_ue_db - scenario.beta_ap_ue_db)
    assert np.median(shadow_change) < 3 * config["simulation"]["shadow_std_db"]


def test_training_and_validation_never_reuse_evaluation_drops():
    assert TRAINING_DROPS >= 10 ** 6 and VALIDATION_DROPS >= 10 ** 6 and TRAINING_DROPS != VALIDATION_DROPS


@pytest.mark.parametrize("mode", ["rap", "fedavg", "none"])
def test_federation_modes(config, mode):
    trainer = RapFedWMMSETrainer(quick(config), 42, 0, mode)
    history = trainer.run()
    backbones = [np.concatenate([p.detach().numpy().ravel() for p in m.backbone.parameters()]) for m in trainer.nodes.values()]
    if mode == "fedavg":
        assert all(np.allclose(b, backbones[0]) for b in backbones)
        assert history[-1]["cumulative_bytes"] > 0
    elif mode == "none":
        assert history[-1]["cumulative_bytes"] == 0
        assert not all(np.allclose(b, backbones[0]) for b in backbones)
    else:
        assert not all(np.allclose(b, backbones[0]) for b in backbones)   # personalized
    assert all(np.isfinite(record["pf_utility"]) for record in history)


def test_trained_policy_is_feasible_reproducible_and_beats_mrt(config):
    config = quick(config)
    algorithm = get_algorithm("rap_fedwmmse")
    algorithm.fit(config, 42, 0)
    instance = make_instance(config, 42, 0, 0)
    first, second = algorithm.solve(instance), algorithm.solve(instance)
    check_feasibility(instance, first)
    assert np.array_equal(first.W, second.W) and first.tau == second.tau
    from baselines.heuristics import heuristic_solution
    assert evaluate_solution(instance, first)["pf_utility"] > evaluate_solution(instance, heuristic_solution(instance))["pf_utility"]


def test_hard_min_rates_have_no_nan_when_a_relay_hop_binds(config):
    _, net, W, V, duals = random_state(config)
    with torch.no_grad():
        W, V, _ = unfold(net, W, V, fixed_params(net, 1), duals, 5, None, 0.2, 0.2)
    assert torch.isfinite(W.abs()).all() and torch.isfinite(V.abs()).all()
    z = coordination(net, W, V, duals, None)
    rate, _, _ = user_rates(z.C_D, z.C_AR, net.assoc)
    assert torch.isfinite(rate).all()


def test_beam_update_survives_a_singular_curvature():
    # Two antennas, one weighted stream and no learned regularization: Q has rank 1 and the power
    # limit is slack, which once made the solve fail. The update must stay finite and feasible.
    from train.unfolded_wmmse import CDTYPE, RDTYPE, _regularized_step
    h = torch.tensor([1.0 + 0.5j, 0.0], dtype=CDTYPE)                   # exactly rank 1
    Q = (h[:, None] * h.conj()[None, :])[None, None]                     # [1, 1, 2, 2], rank 1
    W = torch.zeros(1, 1, 2, 2, dtype=CDTYPE)                            # [B, nodes, K, N]
    grad = torch.zeros_like(W)
    grad[0, 0, 0] = -1e-3 * h
    ones = torch.ones(1, 1, dtype=RDTYPE)
    updated = _regularized_step(grad, Q, W, ones, ones, torch.zeros(1, 1, dtype=RDTYPE), 1.0)
    assert torch.all(torch.isfinite(updated.real)) and torch.all(torch.isfinite(updated.imag))
