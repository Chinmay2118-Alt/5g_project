"""
experiments/definitions.py
Responsibility: the sweeps behind the 14 graphs (docs/GRAPH_PLAN.md). Network size and
parameter ranges live here, so adding an algorithm never requires touching them.
"""

from baselines.exact_minlp import small_instance_config

# The algorithms compared in the sweeps: the proposed method, the centralized benchmark and its
# no-relay version, Local-WMMSE / No-FL, and two simple heuristics.
DEFAULT_ALGORITHMS = ["rap_fedwmmse", "centralized_wmmse", "centralized_no_relay", "local_wmmse", "rzf_equal", "mrt_equal"]
# The default network also gets the ablations behind the learning curves.
BASELINE_ALGORITHMS = DEFAULT_ALGORITHMS + ["rap_fedwmmse_fedavg", "rap_fedwmmse_nofl", "unfolded_wmmse"]
GAP_ALGORITHMS = ["exact_minlp", "rap_fedwmmse", "centralized_wmmse", "local_wmmse", "unfolded_wmmse", "mrt_equal"]

# Holds every method to the same tau and association, so all solve the same problem.
SAME_PROBLEM = {"wmmse.optimize_tau": False, "fedwmmse.adapt_tau": False, "fedwmmse.adapt_association": False}


def _power_w(dbm):
    return 10.0 ** ((dbm - 30.0) / 10.0)


EXPERIMENTS = {
    "baseline": {
        "name": "baseline",
        "x_label": "Default network",
        "base_overrides": {"fedwmmse.validation_every": 1},   # every epoch, for the learning curves
        "points": [("default", {})],
        "algorithms": BASELINE_ALGORITHMS,
    },
    "power": {
        "name": "power",
        "x_label": "Transmit power per AP and relay (dBm)",
        "points": [(dbm, {"power.p_max_ap": _power_w(dbm), "power.p_max_relay": _power_w(dbm)}) for dbm in (10, 15, 20, 25, 30)],
    },
    "num_ues": {
        "name": "num_ues",
        "x_label": "Number of UEs K",
        "points": [(k, {"network.num_ues": k}) for k in (4, 6, 8, 10, 12)],
    },
    "blocked_fraction": {
        "name": "blocked_fraction",
        "x_label": "Fraction of blocked UEs",
        "points": [(f, {"blockage.frac_blocked_ues": f}) for f in (0.0, 0.25, 0.5, 0.75)],
    },
    "num_relays": {
        "name": "num_relays",
        "x_label": "Number of relays R",
        "points": [(r, {"network.num_relays": r}) for r in (1, 2, 3, 4)],
    },
    "tau": {
        "name": "tau",
        "x_label": "AP-transmission time fraction τ (fixed for every relay method; no relay always uses τ = 1)",
        "points": [(t, {"phase.tau_fixed": t, **SAME_PROBLEM}) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)],
    },
    "aging": {
        "name": "aging",
        "x_label": "Channel aging coefficient ρ",
        "points": [(rho, {"aging.enabled": True, "aging.rho_h": rho, "aging.rho_G": rho, "aging.rho_g": rho}) for rho in (0.8, 0.85, 0.9, 0.95, 0.99)],
    },
    "layers": {
        "name": "layers",
        "x_label": "Unfolded WMMSE layers L (same τ and association for both)",
        "base_overrides": SAME_PROBLEM,   # isolates what the learning adds
        "points": [(layers, {"fedwmmse.layers": layers}) for layers in (1, 2, 4, 6, 8, 10)],
        "algorithms": ["rap_fedwmmse", "unfolded_wmmse"],
    },
    "optimality_gap": {
        "name": "optimality_gap",
        "x_label": "Small network",
        "prepare": small_instance_config,
        "base_overrides": SAME_PROBLEM,
        "points": [("small", {})],
        "algorithms": GAP_ALGORITHMS,
    },
}
