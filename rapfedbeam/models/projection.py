"""
models/projection.py
Responsibility: turn raw head outputs into feasible power coefficients that satisfy
power constraints BY CONSTRUCTION (no soft penalty needed for feasibility itself --
though a soft power penalty is still used in the loss to shape the solution, see train/).
Paper mapping: Sec. XXII
"""

import torch
import torch.nn.functional as F


def project_ap_power(raw_logits, p_max_ap, cluster_mask_row):
    # PSEUDOCODE:
    # 1. zero out entries where cluster_mask_row == 0 (AP doesn't serve that UE)
    # 2. squared_logits = softplus(raw_logits) ** 2      # ensure non-negative "power" pre-norm
    # 3. total = sum(squared_logits)
    # 4. scale = sqrt(p_max_ap) / sqrt(max(total, p_max_ap))   # only shrink, never grow
    # 5. power_coeffs = sqrt(squared_logits) * scale
    # return power_coeffs   # guarantees sum(power_coeffs^2) <= p_max_ap
    mask = torch.as_tensor(cluster_mask_row, device=raw_logits.device, dtype=raw_logits.dtype)
    positive_power = F.softplus(raw_logits).square() * mask
    total_power = positive_power.sum(dim=-1, keepdim=True)
    scale = torch.sqrt(torch.as_tensor(p_max_ap, dtype=raw_logits.dtype, device=raw_logits.device) / torch.clamp(total_power, min=p_max_ap))
    return torch.sqrt(positive_power) * scale


def project_relay_power(raw_logits, p_max_relay, relay_assoc_row, this_relay_id):
    # PSEUDOCODE:
    # 1. association mask: zero out UEs not assigned to this_relay_id (a_rk = 0 => v_rk = 0)
    # 2. same softplus + normalize recipe as project_ap_power, using p_max_relay
    # return power_coeffs   # guarantees sum(power_coeffs^2) <= p_max_relay, masked by association
    mask = (torch.as_tensor(relay_assoc_row, device=raw_logits.device) == int(this_relay_id)).to(raw_logits.dtype)
    positive_power = F.softplus(raw_logits).square() * mask
    total_power = positive_power.sum(dim=-1, keepdim=True)
    scale = torch.sqrt(torch.as_tensor(p_max_relay, dtype=raw_logits.dtype, device=raw_logits.device) / torch.clamp(total_power, min=p_max_relay))
    return torch.sqrt(positive_power) * scale
