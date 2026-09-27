"""
models/backbone.py
Responsibility: structurally identical network shared across nodes, but each node
keeps its OWN parameter copy theta_i. This is the part that gets personalized
federated aggregation applied to it (see fl/aggregate.py).
Paper mapping: Sec. XX
"""

from torch import nn

class SharedBackbone(nn.Module):
    def __init__(self, d0, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d0, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))

    def forward(self, z0):
        return self.net(z0)
#     def __init__(self, d0, d_hidden, d_out):
#         # self.net = MLP: Linear(d0, d_hidden) -> ReLU -> Linear(d_hidden, d_hidden) -> ReLU
#         #                 -> Linear(d_hidden, d_out)
#     def forward(self, z0):
#         # return self.net(z0)   # internal representation, shape [d_out]


def make_backbone_for_node(config):
    # PSEUDOCODE:
    # instantiate a fresh SharedBackbone with the SAME architecture/hyperparameters
    # for every node (AP or relay) -- but each call returns a separate parameter object.
    # This is what gives every node in the system its OWN theta_i.
    model = config.get("models", {})
    return SharedBackbone(model.get("adapter_dim", 32), model.get("hidden_dim", 64), model.get("backbone_dim", 32))


def get_backbone_state_dict(backbone):
    # PSEUDOCODE: return backbone.state_dict() -- used when computing delta_theta for FL round
    return {key: value.detach().clone() for key, value in backbone.state_dict().items()}


def set_backbone_state_dict(backbone, state_dict):
    # PSEUDOCODE: backbone.load_state_dict(state_dict) -- used after aggregation to update node
    backbone.load_state_dict(state_dict)
