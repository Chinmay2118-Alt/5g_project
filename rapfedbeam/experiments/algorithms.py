"""
experiments/algorithms.py
Responsibility: the common interface and registry of every algorithm compared in the graphs.
"""

from baselines.heuristics import heuristic_solution
from sim.problem import BeamSolution

REGISTRY = {}


class Algorithm:
    """Implement solve(); implement fit() as well if the algorithm learns across drops.

    fit() runs once per sweep point and deployment, before any drop of that deployment is solved.
    It may return a list of per-epoch records for the learning-curve graphs.
    """

    name = None
    label = None
    available = True
    needs_fit = False

    def fit(self, config, seed, deployment=0):
        return None

    def solve(self, instance):
        raise NotImplementedError


def register(algorithm_class):
    REGISTRY[algorithm_class.name] = algorithm_class
    return algorithm_class


def get_algorithm(name):
    _load_late_registrations()
    return REGISTRY[name]()


def available_algorithms():
    _load_late_registrations()
    return [name for name, algorithm_class in REGISTRY.items() if algorithm_class.available]


def _load_late_registrations():
    import train.rap_fedwmmse  # noqa: F401 (registers itself)


@register
class CentralizedWMMSE(Algorithm):
    name = "centralized_wmmse"
    label = "Centralized WMMSE-SCA"

    def solve(self, instance):
        from baselines.wmmse import centralized_wmmse
        return centralized_wmmse(instance)


@register
class CentralizedWMMSENoRelay(Algorithm):
    name = "centralized_no_relay"
    label = "Centralized WMMSE, no relay"

    def solve(self, instance):
        from baselines.wmmse import centralized_wmmse
        return centralized_wmmse(instance, use_relay=False)


@register
class LocalWMMSE(Algorithm):
    name = "local_wmmse"
    label = "Local-WMMSE (no FL)"

    def solve(self, instance):
        from baselines.wmmse import local_wmmse
        return local_wmmse(instance)


@register
class RZFEqual(Algorithm):
    name = "rzf_equal"
    label = "RZF, equal power"

    def solve(self, instance):
        return heuristic_solution(instance, "equal", precoder="rzf")


@register
class MRTEqual(Algorithm):
    name = "mrt_equal"
    label = "MRT, equal power"

    def solve(self, instance):
        return heuristic_solution(instance, "equal")


@register
class MRTFractional(Algorithm):
    name = "mrt_fractional"
    label = "MRT, fractional power"

    def solve(self, instance):
        return heuristic_solution(instance, "fractional")


@register
class ExactMINLP(Algorithm):
    name = "exact_minlp"
    label = "Exact MINLP (B&B)"

    def solve(self, instance):
        from baselines.exact_minlp import solve_exact
        return solve_exact(instance)


@register
class NeuralCTDE(Algorithm):
    """The earlier black-box policy, kept as an optional extra baseline. It is not the main method."""

    name = "neural_ctde"
    label = "Neural CTDE"
    needs_fit = True

    def fit(self, config, seed, deployment=0):
        from train.ctde_train import train_stage_b
        fit_config = {**config, "seed": seed}
        self.models, _ = train_stage_b(fit_config)
        return None

    def solve(self, instance):
        from train.ctde_train import forward_all_nodes
        W, V, U = forward_all_nodes(self.models, instance.channels, instance.scenario, instance.config)
        return BeamSolution(W, V, U, instance.tau, instance.scenario.relay_assoc)
