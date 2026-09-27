"""
tests/test_rates.py
Responsibility: closed-form sanity checks (week 2 exit criterion).
"""

import numpy as np
from types import SimpleNamespace
from sim.rates import (
    destination_capacity,
    end_to_end_rate,
    phase1_sinr_ue,
    phase2_sinr_ue,
)


def test_single_ap_single_ue_no_interference():
    # PSEUDOCODE:
    # construct a 1-AP, 1-UE scenario, no relay, single antenna, known channel gain |h|^2
    # analytically: gamma1 = |h|^2 * p_max / noise_var
    # analytically: rate = tau * log2(1 + gamma1)   (only Phase I contributes, no relay)
    # run sim/rates.py pipeline on this same handcrafted input
    # assert computed rate matches analytical rate within small tolerance
    h = np.array([[[2.0 + 0j]]])
    w = np.array([[[0.5 + 0j]]])
    gamma = phase1_sinr_ue(h, w, np.array([[True]]), 0.25)
    assert np.allclose(gamma[0], 4.0)
    C_D = destination_capacity(gamma, np.zeros(1), 0.5)
    rate = end_to_end_rate(C_D, np.zeros((1, 1)), np.array([-1]))
    assert np.allclose(rate[0], 0.5 * np.log2(5.0))


def test_relay_zero_when_not_associated():
    # PSEUDOCODE:
    # construct scenario where UE k has relay_assoc[k] == -1
    # assert gamma2_k == 0 and end-to-end rate == C_D_k (Phase-I-only contribution)
    g = np.ones((1, 1, 1), dtype=complex)
    v = np.ones((1, 1, 1), dtype=complex)
    gamma2 = phase2_sinr_ue(g, v, np.array([-1]), 1.0)
    C_D = destination_capacity(np.array([3.0]), gamma2, 0.5)
    rate = end_to_end_rate(C_D, np.zeros((1, 1)), np.array([-1]))
    assert gamma2[0] == 0.0
    assert np.allclose(rate, C_D)


def test_df_bottleneck_binding():
    # PSEUDOCODE:
    # construct a scenario where the AP-relay link is deliberately very weak
    #  (low C_AR_rk) but the destination link C_D_k is artificially high
    # assert end_to_end_rate(...) == C_AR_rk, NOT C_D_k  (confirms the min() is applied correctly)
    C_D = np.array([10.0])
    C_AR = np.array([[2.0]])
    assert np.allclose(end_to_end_rate(C_D, C_AR, np.array([0])), 2.0)


def test_hard_vs_soft_min_convergence():
    # PSEUDOCODE:
    # for a fixed C_D_k, C_AR_rk pair, compute soft-min at decreasing temperature values
    # assert soft-min result converges toward hard min() as temperature -> 0
    C_D = np.array([4.0])
    C_AR = np.array([[2.0]])
    hard = end_to_end_rate(C_D, C_AR, np.array([0]))[0]
    soft = [end_to_end_rate(C_D, C_AR, np.array([0]), False, temperature)[0] for temperature in (1.0, 0.1, 0.01)]
    assert abs(soft[-1] - hard) < abs(soft[0] - hard)
