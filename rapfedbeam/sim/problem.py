"""
sim/problem.py
Responsibility: the contract between the simulator and every algorithm. An algorithm receives
a ProblemInstance and returns a BeamSolution; scoring happens in eval/metrics.py only.
Paper mapping: Sec. XVII (variables of P0)
"""

from dataclasses import dataclass, field
import numpy as np

from sim.channels import ChannelSet, sample_full_channel_set
from sim.scenario import Scenario, generate_drop


@dataclass
class ProblemInstance:
    scenario: Scenario
    channels: ChannelSet
    config: dict

    @property
    def tau(self):
        return self.config["phase"]["tau_fixed"]

    @property
    def ue_weights(self):
        # omega_k, uniform unless the config supplies them
        return np.asarray(self.config.get("objective", {}).get("ue_weights", np.ones(self.config["network"]["num_ues"])), dtype=float)

    @property
    def epsilon_0(self):
        return float(self.config.get("objective", {}).get("epsilon_0", 1e-3))


@dataclass
class BeamSolution:
    W: np.ndarray              # complex [M, K, N_A]
    V: np.ndarray              # complex [R, K, N_R]
    U: np.ndarray              # complex [R, K, N_R], unit norm for associated pairs
    tau: float
    relay_assoc: np.ndarray    # int [K], -1 when the UE has no relay
    info: dict = field(default_factory=dict)


# Drop indices used for training and validation start here, so they can never coincide with an
# evaluation drop (evaluation uses 0, 1, 2, ...).
TRAINING_DROPS = 1_000_000
VALIDATION_DROPS = 2_000_000


def make_scenario(config, seed, drop_idx, deployment=None):
    """The drop's geometry. Drops with the same deployment share AP and relay positions; without one,
    every drop is its own deployment."""
    deployment = drop_idx if deployment is None else deployment
    rng = np.random.default_rng([int(seed), int(deployment), int(drop_idx)])
    infra_rng = np.random.default_rng([int(seed), int(deployment)])
    return generate_drop(config, rng, infra_rng), rng


def make_instance(config, seed, drop_idx, deployment=None):
    # Seeding on (seed, deployment, drop_idx) gives every algorithm and every sweep point the same drops.
    scenario, rng = make_scenario(config, seed, drop_idx, deployment)
    channels = sample_full_channel_set(scenario, config, rng)
    return ProblemInstance(scenario, channels, config)
