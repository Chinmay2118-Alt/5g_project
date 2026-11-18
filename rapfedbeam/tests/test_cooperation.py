"""
tests/test_cooperation.py
Responsibility: cooperative multi-relay forwarding (L_max > 1) through every layer, and the full
coordination vector zeta(t) of RAP-FedWMMSE Step 3.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from baselines.exact_minlp import small_instance_config, solve_exact
from baselines.heuristics import heuristic_solution
from baselines.wmmse import centralized_wmmse, local_wmmse
from eval.metrics import check_feasibility, evaluate_solution
from experiments.algorithms import get_algorithm
from sim.association import matched_association
from sim.problem import make_instance
from sim.rates import as_assoc_matrix, compute_full_rates
from train.rap_fedwmmse import SlowCoordination, ap_features, feature_sizes, relay_features
from train.unfolded_wmmse import RDTYPE, network_from


@pytest.fixture
def config():
    config = yaml.safe_load((Path(__file__).parents[1] / "configs" / "config.yaml").read_text())
    config["network"]["l_max"] = 2
    return config


def cooperative_instance(config, drop=0):
    """A drop where UE 0 is assisted by both relays and UE 1 by relay 1."""
    instance = make_instance(config, 42, drop)
    assoc = np.zeros((config["network"]["num_relays"], config["network"]["num_ues"]), dtype=int)
    assoc[:, 0] = 1
    assoc[1, 1] = 1
    instance.scenario.relay_assoc = assoc
    return instance


def test_every_assisting_relay_must_decode(config):
    # Sir's DF rate: r_k = min(C_D_k, min over r with a_rk = 1 of C_AR_rk).
    instance = cooperative_instance(config)
    solution = heuristic_solution(instance)
    rates, debug = compute_full_rates(instance.scenario, instance.channels, solution.W, solution.V, solution.U, solution.tau)
    expected = min(debug["C_D"][0], debug["C_AR"][0, 0], debug["C_AR"][1, 0])
    assert np.isclose(rates[0], expected)
    assert debug["C_AR"][0, 0] > 0 and debug["C_AR"][1, 0] > 0


def test_index_and_matrix_forms_give_the_same_rates(config):
    config["network"]["l_max"] = 1
    instance = make_instance(config, 42, 1)
    vector = np.asarray(instance.scenario.relay_assoc)
    assert np.any(vector >= 0)
    solution = heuristic_solution(instance)
    as_vector = evaluate_solution(instance, solution)["rates"]
    instance.scenario.relay_assoc = as_assoc_matrix(vector, config["network"]["num_relays"])
    solution.relay_assoc = instance.scenario.relay_assoc
    assert np.allclose(evaluate_solution(instance, solution)["rates"], as_vector)


def test_feasibility_enforces_the_cooperation_order(config):
    instance = cooperative_instance(config)
    solution = heuristic_solution(instance)
    check_feasibility(instance, solution)
    instance.config["network"]["l_max"] = 1
    with pytest.raises(ValueError, match="L_max"):
        check_feasibility(instance, solution)


def test_matching_returns_a_matrix_within_l_max(config):
    config["association"]["min_score"] = -10.0           # make many pairs worth choosing
    config["association"]["eligibility"] = "none"
    instance = make_instance(config, 42, 2)
    assoc = matched_association(instance.scenario, config, np.full(config["network"]["num_ues"], 5.0))
    assert assoc.ndim == 2
    assert assoc.sum(axis=0).max() <= 2 and assoc.sum(axis=1).max() <= config["network"]["max_streams_per_relay"]
    assert assoc.sum(axis=0).max() == 2                   # cooperation actually happens


def test_wmmse_baselines_handle_cooperative_relays(config):
    instance = cooperative_instance(config)
    central = centralized_wmmse(instance, max_iters=4)
    check_feasibility(instance, central)
    assert np.all(np.diff(central.info["history"]) >= -1e-9)
    assert np.linalg.norm(central.V[0, 0]) > 0 and np.linalg.norm(central.V[1, 0]) > 0
    local = local_wmmse(instance)
    check_feasibility(instance, local)
    assert evaluate_solution(instance, central)["pf_utility"] > evaluate_solution(instance, local)["pf_utility"]


def test_rap_fedwmmse_trains_and_solves_with_cooperative_relays(config):
    config["network"].update(num_aps=4, num_relays=2, num_ues=4, cluster_top_l=2)
    config["fedwmmse"].update(epochs=2, local_steps=1, batch_blocks=2, layers=2, validation_instances=1,
                              warmup_epochs=1, warmup_blocks=1, adapt_association=False)
    algorithm = get_algorithm("rap_fedwmmse")
    algorithm.fit(config, 42, 0)
    instance = cooperative_instance(config, drop=3)
    solution = algorithm.solve(instance)
    check_feasibility(instance, solution)
    assert as_assoc_matrix(solution.relay_assoc, 2)[:, 0].sum() == 2


def test_exact_minlp_with_two_relays_per_ue_bounds_centralized(config):
    small = small_instance_config(config)
    small["network"].update(num_relays=2, l_max=2)
    small["wmmse"]["optimize_tau"] = False
    instance = make_instance(small, 42, 0)
    exact = solve_exact(instance, time_limit_s=120, optimize_association=True)
    check_feasibility(instance, exact)
    central = evaluate_solution(instance, centralized_wmmse(instance))["pf_utility"]
    assert central <= exact.info["upper_bound"] + 1e-3


def test_zeta_carries_every_quantity_sir_lists(config):
    # Step 3: QoS prices, DF prices, measured aggregate interference and previous rate statistics.
    config["network"]["l_max"] = 1
    instance = make_instance(config, 42, 0)
    net = network_from(instance.scenario, [instance.channels], config, instance.scenario.relay_assoc, 0.5)
    K, R = config["network"]["num_ues"], config["network"]["num_relays"]
    ap_dim, relay_dim = feature_sizes(config)
    base = SlowCoordination.initial(K, R)
    vector = lambda c: torch.as_tensor(c.vector(), dtype=RDTYPE)[None]
    assert ap_features(net, 0, vector(base)).shape[1] == ap_dim
    assert relay_features(net, 0, vector(base)).shape[1] == relay_dim
    for field, value in (("qos", np.ones(K)), ("df", np.ones((R, K))), ("interference1", np.ones(K)),
                         ("interference2", np.ones(K)), ("rates", np.ones(K))):
        changed = SlowCoordination(**{**base.__dict__, field: value})
        assert not torch.equal(ap_features(net, 0, vector(base)), ap_features(net, 0, vector(changed))), field
        assert not torch.equal(relay_features(net, 0, vector(base)), relay_features(net, 0, vector(changed))), field


@pytest.mark.parametrize("mode", ["rap", "fedavg"])
def test_fronthaul_without_headroom_keeps_aps_out_of_federation(config, mode):
    # Paper Sec. XXXI: D_data_m + D_FL_m <= F_max_m T_e. An AP whose fronthaul is full of user data
    # cannot exchange model updates; relays use their own control link and still can.
    from train.rap_fedwmmse import RapFedWMMSETrainer
    config["network"].update(num_aps=4, num_relays=2, num_ues=4, cluster_top_l=2, l_max=1)
    config["fedwmmse"].update(epochs=3, local_steps=1, batch_blocks=2, layers=2, validation_instances=1,
                              warmup_epochs=1, warmup_blocks=1, validation_every=1, trigger_threshold=0.0)
    config["fronthaul"]["ap_capacity_bps"] = 1.0                  # 1 bit/s: no headroom at all
    trainer = RapFedWMMSETrainer(config, 42, 0, mode)
    before = {i: trainer.theta_received[i].copy() for i in trainer.nodes}
    history = trainer.run()
    assert all(record["participants"] <= 2 for record in history)  # relays only
    assert history[-1]["mean_overloaded_aps"] > 0
    for m in range(4):
        assert np.array_equal(trainer.theta_received[f"AP{m}"], before[f"AP{m}"])
    config["fronthaul"]["ap_capacity_bps"] = 1.0e9
    open_history = RapFedWMMSETrainer(config, 42, 0, mode).run()
    assert max(record["participants"] for record in open_history) > 2
