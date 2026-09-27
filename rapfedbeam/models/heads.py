"""
models/heads.py
Responsibility: role-specific output heads turning backbone representation into
POWER COEFFICIENTS (not raw complex beams -- v1 simplification), one per served UE.
Stays LOCAL to each node, never transferred.
Paper mapping: Sec. XXI (simplified: power coeffs instead of full complex W / V,U)
"""

from torch import nn

class APOutputHead(nn.Module):
	def __init__(self, d_in, max_ues_served):
		super().__init__()
		self.net = nn.Linear(d_in, max_ues_served)

	def forward(self, z):
		return self.net(z)
#     def __init__(self, d_in, max_ues_served):
#         # self.net = Linear(d_in, max_ues_served)
#     def forward(self, z):
#         # raw = self.net(z)                # unconstrained real-valued logits
#         # return raw                        # shape [max_ues_served], passed to projection.py


class RelayOutputHead(nn.Module):
	def __init__(self, d_in, max_ues_served):
		super().__init__()
		self.net = nn.Linear(d_in, max_ues_served)

	def forward(self, z):
		return self.net(z)
#     def __init__(self, d_in, max_ues_served):
#         # self.net = Linear(d_in, max_ues_served)
#         # (receive combiner U is closed-form MMEE in v1, NOT produced by this head --
#         #  see sim/rates.py / a shared mmse_combiner util)
#     def forward(self, z):
#         # raw = self.net(z)
#         # return raw   # shape [max_ues_served], passed to projection.py -> relay tx power coeffs
