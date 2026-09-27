"""
models/adapters.py
Responsibility: role-specific input adapters mapping raw local features into a
fixed-size latent vector z0 in R^d0. Stays LOCAL to each node, never transferred.
Paper mapping: Sec. XIX
"""

import numpy as np
import torch
from torch import nn

class APInputAdapter(nn.Module):
    def __init__(self, input_dim, d0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, d0))

    def forward(self, x_m):
        return self.net(x_m)
#     def __init__(self, input_dim, d0):
#         # input_dim: flattened features from |h_hat_mk| for k in served UEs
#         #            + |G_hat_mr| for relays this AP can reach
#         # self.net = small MLP: Linear(input_dim, 64) -> ReLU -> Linear(64, d0)
#     def forward(self, x_m):
#         # x_m: local feature vector for AP m (already flattened/normalized outside)
#         # return self.net(x_m)   # z0_m, shape [d0]


class RelayInputAdapter(nn.Module):
    def __init__(self, input_dim, d0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, d0))

    def forward(self, x_r):
        return self.net(x_r)
#     def __init__(self, input_dim, d0):
#         # input_dim: flattened features from |G_hat_mr| for reachable APs
#         #            + |g_hat_rk| for served UEs
#         # self.net = small MLP: Linear(input_dim, 64) -> ReLU -> Linear(64, d0)
#     def forward(self, x_r):
#         # return self.net(x_r)   # z0_r, shape [d0]


def build_ap_feature_vector(h_hat_mk_row, G_hat_mr_row):
    # PSEUDOCODE:
    # take channel MAGNITUDES (or log-magnitude) as features -- phase is not needed
    # since only power coefficients are learned in v1, not complex beam weights.
    # concat and flatten into one 1-D vector, normalize (e.g. z-score using running stats)
    values = np.concatenate([np.abs(np.asarray(h_hat_mk_row)).ravel(), np.abs(np.asarray(G_hat_mr_row)).ravel()])
    return np.log1p(values).astype(np.float32)


def build_relay_feature_vector(G_hat_mr_col, g_hat_rk_row):
    # PSEUDOCODE: same recipe, relay-specific slice of channels
    values = np.concatenate([np.abs(np.asarray(G_hat_mr_col)).ravel(), np.abs(np.asarray(g_hat_rk_row)).ravel()])
    return np.log1p(values).astype(np.float32)
