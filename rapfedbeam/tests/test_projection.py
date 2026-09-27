"""
tests/test_projection.py
Responsibility: confirm feasibility projection guarantees power constraints hold BY
CONSTRUCTION, regardless of what a (possibly untrained/random) network outputs.
"""

import numpy as np
import torch
from models.projection import project_ap_power, project_relay_power


def test_ap_power_constraint_holds_for_random_logits():
    # PSEUDOCODE:
    # for many random raw_logits (including large/extreme values):
    #     power_coeffs = project_ap_power(raw_logits, p_max_ap, cluster_mask_row)
    #     assert sum(power_coeffs ** 2) <= p_max_ap + small_epsilon
    logits = torch.tensor([100.0, -100.0, 0.0])
    projected = project_ap_power(logits, 2.0, torch.tensor([1, 1, 1]))
    assert float(torch.sum(projected ** 2)) <= 2.0 + 1e-6


def test_ap_projection_zeroes_unserved_ues():
    # PSEUDOCODE:
    # construct cluster_mask_row with some UEs marked unserved (0)
    # assert power_coeffs at those indices == 0, regardless of raw_logits value there
    projected = project_ap_power(torch.tensor([0.0, 4.0]), 1.0, torch.tensor([1, 0]))
    assert projected[1].item() == 0.0


def test_relay_power_constraint_and_association_mask():
    # PSEUDOCODE:
    # construct relay_assoc_row where this relay is NOT assigned to some UEs
    # assert power_coeffs at those UE indices == 0 (a_rk = 0 => v_rk = 0)
    # assert sum(power_coeffs ** 2) <= p_max_relay + small_epsilon for the rest
    projected = project_relay_power(torch.tensor([1.0, 1.0, 1.0]), 1.0, torch.tensor([0, 1, 2]), 1)
    assert projected[0].item() == 0.0 and projected[2].item() == 0.0
    assert float(torch.sum(projected ** 2)) <= 1.0 + 1e-6


def test_projection_is_differentiable():
    # PSEUDOCODE:
    # pass raw_logits as a torch tensor with requires_grad=True through project_ap_power
    # call .backward() on the output sum
    # assert raw_logits.grad is not None and contains no NaN/Inf
    # (this matters because Stage B trains by backpropagating THROUGH this projection)
    logits = torch.randn(5, requires_grad=True)
    project_ap_power(logits, 1.0, torch.ones(5)).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
