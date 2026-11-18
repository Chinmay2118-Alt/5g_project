"""
train/rap_fedwmmse.py
Responsibility: RAP-FedWMMSE, the main algorithm: role-aware personalized federated unfolded WMMSE.
The epoch loop follows Sir's pseudocode: slow-timescale association and tau (Steps 1-2), local
self-supervised training of each node's model through the unfolded WMMSE layers (Steps 3-11), then
event-triggered, budgeted, compressed, personalized federation of the backbones (Steps 12-15).
Derivation and interpretation choices: docs/RAP_FEDWMMSE.md
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
from torch import nn

from experiments.algorithms import Algorithm, register
from fl.communication import (BYTES_PER_FLOAT, compressed_bytes, model_drift, personalized_updates,
                              select_clients, top_k, urgency)
from models.adapters import APInputAdapter, RelayInputAdapter
from models.backbone import SharedBackbone
from sim.association import matched_association
from sim.channels import sample_full_channel_set
from sim.problem import TRAINING_DROPS, VALIDATION_DROPS, BeamSolution, make_instance, make_scenario
from sim.rates import as_assoc_matrix, mmse_combiners
from sim.scenario import evolve_drop
from train.unfolded_wmmse import (CDTYPE, RDTYPE, LayerParams, augmented_objective, coordination, initial_beams,
                                  network_from, outage_surrogate, unfold, user_rates, zeta_complex_numbers)


# --- Steps 3-4: role-specific adapters, personalized backbone, optimizer-parameter heads ---

def _log_magnitude(x):
    return torch.log10(x.abs() + 1e-6) / 3.0


@dataclass
class SlowCoordination:
    """zeta(t) of Step 3: the low-dimensional coordination information every node receives, set
    once per slow-timescale epoch. It holds the QoS dual prices, the DF constraint prices, the
    measured aggregate interference at each UE in both phases, and the previous epoch's rate
    statistics. No antenna-domain CSI is included."""
    qos: np.ndarray            # [K], lambda_Q
    df: np.ndarray             # [R, K], nu_DF
    interference1: np.ndarray  # [K], log10(1 + Phase-I interference / noise), measured at the UE
    interference2: np.ndarray  # [K], the same in Phase II
    rates: np.ndarray          # [K], mean rate over the previous epoch

    @classmethod
    def initial(cls, num_ues, num_relays):
        zeros = np.zeros(num_ues)
        return cls(zeros.copy(), np.zeros((num_relays, num_ues)), zeros.copy(), zeros.copy(), zeros.copy())

    def vector(self):
        return np.concatenate([self.qos, self.df.ravel(), self.interference1 / 3.0, self.interference2 / 3.0,
                               np.log1p(self.rates)])

    @staticmethod
    def size(num_ues, num_relays):
        return 4 * num_ues + num_relays * num_ues


def measured_interference(S, noise):
    """log10(1 + sum_{j != k} |S_kj|^2 / noise), averaged over the blocks: what UE k measures. [K]"""
    power = np.abs(S) ** 2
    interference = power.sum(-1) - np.diagonal(power, axis1=-2, axis2=-1)
    return np.log10(1.0 + interference / noise).mean(axis=0)


def ap_features(net, m, zeta):
    """x_m = {h_mk, G_mr, beta_m, zeta}: AP m's own channel magnitudes, large-scale gains and
    cluster row, plus the coordination vector zeta. [B, F]"""
    batch = net.batch
    return torch.cat([
        _log_magnitude(net.h[:, m]).reshape(batch, -1),
        _log_magnitude(net.G[:, m]).reshape(batch, -1),
        ((net.beta_ap_db[m] + 100.0) / 30.0).expand(batch, -1),
        net.cluster[m].expand(batch, -1),
        zeta,
    ], dim=1).to(torch.float32)


def relay_features(net, r, zeta):
    """x_r = {G_mr, g_rk, beta_r, zeta}: relay r's own channel magnitudes, large-scale gains and
    association row, plus the coordination vector zeta. [B, F]"""
    batch = net.batch
    return torch.cat([
        _log_magnitude(net.G[:, :, r]).reshape(batch, -1),
        _log_magnitude(net.g[:, r]).reshape(batch, -1),
        ((net.beta_relay_db[r] + 100.0) / 30.0).expand(batch, -1),
        net.assoc[r].expand(batch, -1),
        zeta,
    ], dim=1).to(torch.float32)


def feature_sizes(config):
    n = config["network"]
    M, R, K, NA, NR = n["num_aps"], n["num_relays"], n["num_ues"], n["n_antennas_ap"], n["n_antennas_relay"]
    zeta = SlowCoordination.size(K, R)
    return K * NA + R * NR * NA + 2 * K + zeta, M * NR * NA + K * NR + 2 * K + zeta


# Initial outputs of an untrained head: step 1, damping 0.5, the classical power price (reg 1),
# equal power.
INITIAL_STEP, INITIAL_DAMPING, INITIAL_REG = 1.0, 0.5, 1.0


class NodeModel(nn.Module):
    """pi_i = {adapter (local), backbone theta_i (federated), head (local)}. The head predicts
    Omega_i: a step size and a damping per layer, a regularization, and initial power shares."""

    def __init__(self, role, input_dim, config):
        super().__init__()
        models, layers, num_ues = config["models"], config["fedwmmse"]["layers"], config["network"]["num_ues"]
        self.role, self.layers = role, layers
        self.adapter = (APInputAdapter if role == "AP" else RelayInputAdapter)(input_dim, models["adapter_dim"])
        self.backbone = SharedBackbone(models["adapter_dim"], models["hidden_dim"], models["backbone_dim"])
        self.head = nn.Linear(models["backbone_dim"], 2 * layers + 1 + num_ues)
        with torch.no_grad():
            self.head.weight.mul_(0.01)
            self.head.bias.zero_()
            self.head.bias[2 * layers] = math.log(math.expm1(INITIAL_REG))

    def forward(self, features):
        raw = self.head(self.backbone(self.adapter(features))).to(RDTYPE)
        L = self.layers
        step = 2.0 * torch.sigmoid(raw[:, :L])                     # eta in (0, 2)
        damping = torch.sigmoid(raw[:, L:2 * L])                    # delta in (0, 1)
        reg = nn.functional.softplus(raw[:, 2 * L])                 # lambda_reg >= 0
        return step, damping, reg, raw[:, 2 * L + 1:]


def fixed_params(net, layers, step=INITIAL_STEP, damping=INITIAL_DAMPING, reg=INITIAL_REG):
    """Layer parameters for the untrained ablation: the same for every node and layer."""
    B, M, R = net.batch, net.h.shape[1], net.g.shape[1]
    full = lambda nodes, value, *extra: torch.full((B, nodes, *extra), value, dtype=RDTYPE)
    return LayerParams(full(M, step, layers), full(M, damping, layers), full(M, reg),
                       full(R, step, layers), full(R, damping, layers), full(R, reg))


def backbone_vector(model):
    return torch.nn.utils.parameters_to_vector(model.backbone.parameters()).detach().numpy().astype(np.float64)


def set_backbone_vector(model, vector):
    torch.nn.utils.vector_to_parameters(torch.as_tensor(vector, dtype=torch.float32), model.backbone.parameters())


# --- the algorithm ---

class RapFedWMMSETrainer:
    def __init__(self, config, seed, deployment, fl_mode):
        self.config, self.seed, self.deployment, self.fl_mode = config, int(seed), int(deployment), fl_mode
        self.settings = config["fedwmmse"]
        net_cfg = config["network"]
        self.M, self.R, self.K = net_cfg["num_aps"], net_cfg["num_relays"], net_cfg["num_ues"]
        torch.manual_seed(self.seed * 1000 + self.deployment)
        ap_dim, relay_dim = feature_sizes(config)
        self.nodes = {**{f"AP{m}": NodeModel("AP", ap_dim, config) for m in range(self.M)},
                      **{f"R{r}": NodeModel("relay", relay_dim, config) for r in range(self.R)}}
        # Federated learning starts every node from one common backbone.
        common = backbone_vector(self.nodes["AP0"])
        for model in self.nodes.values():
            set_backbone_vector(model, common)
        self.optimizers = {i: torch.optim.Adam(model.parameters(), lr=self.settings["lr"]) for i, model in self.nodes.items()}
        self.theta_received = {i: common.copy() for i in self.nodes}
        self.residual = {i: np.zeros_like(common) for i in self.nodes}
        self.age = {i: 0 for i in self.nodes}
        self.previous_signature = {}
        self.tau = float(config["phase"]["tau_fixed"])
        self.coordination = SlowCoordination.initial(self.K, self.R)
        self.generator = torch.Generator().manual_seed(self.seed * 1000 + self.deployment)

    # Steps 3-4 for every node, then Steps 5-9
    def policy(self, net, coordination, layers=None):
        layers = self.settings["layers"] if layers is None else layers
        prices = torch.as_tensor(coordination.qos, dtype=RDTYPE).expand(net.batch, -1)
        zeta = torch.as_tensor(coordination.vector(), dtype=RDTYPE).expand(net.batch, -1)
        df_prices = coordination.df
        ap_out = [self.nodes[f"AP{m}"](ap_features(net, m, zeta)) for m in range(self.M)]
        relay_out = [self.nodes[f"R{r}"](relay_features(net, r, zeta)) for r in range(self.R)]
        params = LayerParams(*(torch.stack([o[i] for o in ap_out], dim=1) for i in range(3)),
                             *(torch.stack([o[i] for o in relay_out], dim=1) for i in range(3)))
        W0, V0 = initial_beams(net, torch.stack([o[3] for o in ap_out], dim=1), torch.stack([o[3] for o in relay_out], dim=1))
        duals = (prices.clone(), torch.as_tensor(df_prices, dtype=RDTYPE).expand(net.batch, -1, -1).clone())
        return unfold(net, W0, V0, params, duals, layers, self.settings["softmin_kappa"], self.settings["eta_q"], self.settings["eta_d"])

    def _robust_rho(self):
        aging = self.config.get("aging", {})
        return aging.get("rho_h", 0.95) if aging.get("enabled", False) else self.settings["robust_rho"]

    # Steps 10-11
    def _local_step(self, net):
        W, V, duals = self.policy(net, self.coordination)
        J, z = augmented_objective(net, W, V, duals, self.settings["softmin_kappa"])
        p_out = outage_surrogate(net, W, V, z.u, self._robust_rho(), self.settings["csi_samples"],
                                 self.settings["outage_sharpness"], self.settings["softmin_kappa"], self.generator)
        loss = (J + self.settings["outage_weight"] * torch.relu(p_out - net.eps_out).sum(-1)).mean()
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        loss.backward()
        grad_norms = {}
        for i, model in self.nodes.items():
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.settings["grad_clip"])
            grads = [p.grad.reshape(-1) for p in model.backbone.parameters() if p.grad is not None]
            grad_norms[i] = float(torch.linalg.vector_norm(torch.cat(grads))) if grads else 0.0
            self.optimizers[i].step()
        hard, _, _ = user_rates(z.C_D.detach(), z.C_AR.detach(), net.assoc)
        return hard, z, grad_norms

    # Step 1
    def _association(self, scenario, qos_prices, adapt):
        return matched_association(scenario, self.config, qos_prices) if adapt else np.asarray(scenario.relay_assoc)

    # Step 2: the epoch utility is concave in tau for fixed SINRs, so a fine grid finds its maximum.
    def _best_tau(self, gammas, assoc, weights, epsilon_0, current):
        gamma1, gamma2, gamma_ar = (x.detach().numpy() for x in gammas[:3])
        phase = self.config["phase"]
        best, best_value = current, -np.inf
        for tau in np.linspace(phase["tau_min"], phase["tau_max"], 61):
            rate = tau * np.log2(1 + gamma1) + (1 - tau) * np.log2(1 + gamma2)
            for r, k in zip(*np.nonzero(as_assoc_matrix(assoc, self.R))):
                rate[:, k] = np.minimum(rate[:, k], tau * np.log2(1 + gamma_ar[:, r, k]))
            value = float(np.mean(np.sum(weights * np.log(rate + epsilon_0), axis=1)))
            if value > best_value:
                best, best_value = float(tau), value
        return best

    def _slow_coordination(self, coordination, rates, z, assoc_mat, net):
        """End of an epoch: dual ascent on the QoS and DF prices (Step 9, slow timescale), and the
        measured interference and rate statistics that zeta carries into the next epoch."""
        s = self.settings
        qos = np.maximum(0.0, coordination.qos + s["eta_q_epoch"] * (self.config["qos"]["r_min"] - rates.mean(axis=0)))
        gap = (z.C_D.detach().numpy()[:, None, :] - z.C_AR.detach().numpy()).mean(axis=0)
        df = np.maximum(0.0, coordination.df + s["eta_d_epoch"] * assoc_mat * gap)
        return SlowCoordination(qos, df, measured_interference(z.S1.detach().numpy(), net.noise_ue),
                                measured_interference(z.S2.detach().numpy(), net.noise_ue), rates.mean(axis=0))

    # Steps 12-15
    def _signatures(self, scenario, z, assoc):
        """s_i = [beta_tilde_i, I_tilde_i, L_tilde_i, e_role(i)], the same length for APs and relays."""
        betas = np.vstack([scenario.beta_ap_ue_db, scenario.beta_relay_ue_db])
        center, spread = betas.mean(), betas.std() + 1e-12
        S1 = z.S1.detach().numpy()
        power = np.abs(S1) ** 2
        ue_interference = np.log10(1.0 + (power.sum(-1) - np.diagonal(power, axis1=1, axis2=2)) / scenario.noise_var_ue).mean(axis=0)
        noise_ar = z.gammas[3].detach().numpy()
        sar_power = np.abs(z.SAR.detach().numpy()) ** 2
        relay_interference = np.log10(1.0 + (sar_power.sum(-1) - np.diagonal(sar_power, axis1=2, axis2=3)) / np.maximum(noise_ar, 1e-12)).mean(axis=0)
        capacity = self.config["network"].get("max_streams_per_relay", self.K)
        signatures, roles = {}, {}
        for m in range(self.M):
            served = np.asarray(scenario.cluster_mask[m], dtype=bool)
            interference = ue_interference[served].mean() / 3.0 if served.any() else 0.0
            signatures[f"AP{m}"] = np.concatenate([(scenario.beta_ap_ue_db[m] - center) / (spread * math.sqrt(self.K)),
                                                    [interference, served.mean(), 1.0, 0.0]])
            roles[f"AP{m}"] = "AP"
        for r in range(self.R):
            streams = as_assoc_matrix(assoc, self.R)[r] > 0
            interference = relay_interference[r][streams].mean() / 3.0 if streams.any() else 0.0
            signatures[f"R{r}"] = np.concatenate([(scenario.beta_relay_ue_db[r] - center) / (spread * math.sqrt(self.K)),
                                                   [interference, streams.sum() / capacity, 0.0, 1.0]])
            roles[f"R{r}"] = "relay"
        return signatures, roles

    def _fl_headroom(self, scenario, rates):
        """Bytes each node may still send over its link this epoch (paper Sec. XXXI).

        AP m:     F_max_m T_e - D_data_m, with D_data_m = B T_e sum_k c_mk mean_t r_k(t), the user data it
                  must carry on its fronthaul. Every term is in bits, as the paper requires.
        Relay r:  F_ctrl_r T_e, since relays receive user data over the air, not over the control link.
        A negative AP headroom means the fronthaul cannot even carry the user data; the AP then sends no
        FL traffic, and the shortfall is reported."""
        link = self.config["fronthaul"]
        duration = link["epoch_duration_s"]
        bandwidth = self.config["link_budget"]["bandwidth_hz"]
        data_bits = bandwidth * duration * (np.asarray(scenario.cluster_mask, dtype=float) @ rates.mean(axis=0))   # [M]
        headroom = {f"AP{m}": (link["ap_capacity_bps"] * duration - data_bits[m]) / 8.0 for m in range(self.M)}
        headroom.update({f"R{r}": link["relay_control_bps"] * duration / 8.0 for r in range(self.R)})
        overloaded = int(np.sum(data_bits > link["ap_capacity_bps"] * duration))
        return headroom, overloaded

    def _federate(self, signatures, roles, grad_norms, headroom):
        """Steps 12-15 under the FL budget B_FL and each node's fronthaul or control-link headroom.
        Returns (bytes exchanged, participants)."""
        if self.fl_mode == "none":
            return 0, 0
        s, fl = self.settings, self.config["fl"]
        local = {i: backbone_vector(model) for i, model in self.nodes.items()}
        num_params = len(next(iter(local.values())))
        updates = {i: local[i] - self.theta_received[i] for i in self.nodes}
        signature_bytes = len(next(iter(signatures.values()))) * BYTES_PER_FLOAT
        dense = num_params * BYTES_PER_FLOAT

        if self.fl_mode == "fedavg":
            # FedAvg: dense upload and download for every node whose link can carry both.
            joined = [i for i in self.nodes if headroom[i] >= 2 * dense]
            if not joined:
                return 0, 0
            mean = sum(updates[i] for i in joined) / len(joined)
            for i in joined:
                self.theta_received[i] = self.theta_received[i] + mean
                set_backbone_vector(self.nodes[i], self.theta_received[i])
            return 2 * len(joined) * dense, len(joined)

        for i in self.nodes:
            self.age[i] += 1
        scores = {}
        for i in self.nodes:
            topology = float(np.linalg.norm(signatures[i] - self.previous_signature[i])) if i in self.previous_signature else 0.0
            scores[i] = urgency(model_drift(local[i], self.theta_received[i]), topology, self.age[i], grad_norms[i],
                                s["trigger_weights"], s["age_max"], s["grad_max"])
        self.previous_signature = {i: v.copy() for i, v in signatures.items()}
        ids = list(self.nodes)
        size = compressed_bytes(num_params, s["topk_fraction"]) + signature_bytes
        # Step 13 with the per-node link constraint q_i (S_up + S_down) <= headroom_i: as q_i is binary,
        # a node whose link cannot carry its exchange is simply not eligible.
        eligible = [scores[i] >= s["trigger_threshold"] and headroom[i] >= size + dense for i in ids]
        chosen = select_clients([scores[i] for i in ids], eligible, [size] * len(ids), s["budget_bytes"])
        selected = dict(zip(ids, chosen))
        sent = {}
        for i in ids:
            if selected[i]:
                sent[i], self.residual[i] = top_k(updates[i], self.residual[i], s["topk_fraction"])
        if not any(chosen):
            return 0, 0
        kappa = {"kappa_ap_ap": fl["kappa_ap_ap"], "kappa_relay_relay": fl["kappa_relay_relay"], "kappa_ap_relay": fl["kappa_ap_relay"]}
        aggregated = personalized_updates(signatures, roles, sent, selected, fl["kernel_sigma"], kappa, {i: 1.0 for i in ids})
        receivers = 0
        for i, model in self.nodes.items():
            if not selected[i] and headroom[i] < dense:
                continue   # its link cannot carry the download this epoch
            receivers += 1
            step = fl["rho_p"] * aggregated[i]
            if selected[i]:
                self.theta_received[i] = self.theta_received[i] + step
                set_backbone_vector(model, self.theta_received[i])
                self.age[i] = 0
            else:
                # Its own update was not uploaded, so it keeps it and still counts as pending.
                set_backbone_vector(model, local[i] + step)
                self.theta_received[i] = self.theta_received[i] + step
        return int(sum(chosen)) * size + receivers * dense, int(sum(chosen))

    def run(self):
        s, config = self.settings, self.config
        scenario, _ = make_scenario(config, self.seed, TRAINING_DROPS, self.deployment)
        rng = np.random.default_rng([self.seed, self.deployment, TRAINING_DROPS, 1])
        validation = [make_instance(config, self.seed, VALIDATION_DROPS + v, self.deployment) for v in range(s["validation_instances"])]
        history, total_bytes, total_participants, total_overloaded = [], 0, 0, 0
        for epoch in range(s["epochs"]):
            redrop = s.get("redrop_every", 0)
            if epoch and redrop and epoch % redrop == 0:
                # New UEs arrive on the same deployment, so the node's local data covers many UE layouts.
                scenario, _ = make_scenario(config, self.seed, TRAINING_DROPS + epoch, self.deployment)
            elif epoch:
                scenario = evolve_drop(scenario, config, rng)
            assoc = self._association(scenario, self.coordination.qos, s["adapt_association"])
            assoc_mat = as_assoc_matrix(assoc, self.R).astype(float)
            epoch_rates, grad_norms = [], {}
            for _ in range(s["local_steps"]):
                blocks = [sample_full_channel_set(scenario, config, rng) for _ in range(s["batch_blocks"])]
                net = network_from(scenario, blocks, config, assoc, self.tau)
                rates, z, grad_norms = self._local_step(net)
                epoch_rates.append(rates.numpy())
            rates = np.concatenate(epoch_rates)
            self.coordination = self._slow_coordination(self.coordination, rates, z, assoc_mat, net)
            if s["adapt_tau"]:
                self.tau = self._best_tau(z.gammas, assoc, net.weights.numpy(), net.epsilon_0, self.tau)
            signatures, roles = self._signatures(scenario, z, assoc)
            headroom, overloaded = self._fl_headroom(scenario, rates)
            sent, participants = self._federate(signatures, roles, grad_norms, headroom)
            total_bytes += sent
            total_participants += participants
            total_overloaded += overloaded
            every = s["validation_every"]
            if (every and (epoch + 1) % every == 0) or epoch + 1 == s["epochs"]:
                utility = float(np.mean([evaluate(self, instance) for instance in validation]))
                history.append({"round": epoch + 1, "pf_utility": utility, "cumulative_bytes": total_bytes,
                                "participants": participants, "mean_participants": total_participants / (epoch + 1),
                                "mean_overloaded_aps": total_overloaded / (epoch + 1), "tau": self.tau})
        return history

    @torch.no_grad()
    def solve(self, instance):
        """Deployment: a short slow-timescale warm-up on the drop's statistics sets the QoS prices,
        association and tau, then the policy runs on the drop's estimated channels."""
        s, config = self.settings, instance.config
        scenario = instance.scenario
        key = hashlib.sha256(np.ascontiguousarray(scenario.ue_positions).tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        slow = SlowCoordination.initial(self.K, self.R)
        tau = self.tau if s["adapt_tau"] else instance.tau
        for _ in range(s["warmup_epochs"]):
            assoc = self._association(scenario, slow.qos, s["adapt_association"])
            assoc_mat = as_assoc_matrix(assoc, self.R).astype(float)
            blocks = [sample_full_channel_set(scenario, config, rng) for _ in range(s["warmup_blocks"])]
            net = network_from(scenario, blocks, config, assoc, tau)
            W, V, duals = self.policy(net, slow)
            z = coordination(net, W, V, duals, None)
            hard, _, _ = user_rates(z.C_D, z.C_AR, net.assoc)
            slow = self._slow_coordination(slow, hard.numpy(), z, assoc_mat, net)
            if s["adapt_tau"]:
                tau = self._best_tau(z.gammas, assoc, net.weights.numpy(), net.epsilon_0, tau)
        assoc = self._association(scenario, slow.qos, s["adapt_association"])
        net = network_from(scenario, [instance.channels], config, assoc, tau)
        W, V, _ = self.policy(net, slow)
        W, V = W[0].numpy(), V[0].numpy()
        U = mmse_combiners(instance.channels.G_hat, W, scenario.noise_var_relay)
        info = {"tau": tau, "layers": s["layers"], "relayed_ues": int(as_assoc_matrix(assoc, self.R).any(axis=0).sum()),
                "zeta_complex_per_layer": zeta_complex_numbers(self.K, self.R, config["network"]["n_antennas_relay"])}
        return BeamSolution(W, V, U, tau, assoc, info=info)


def evaluate(trainer, instance):
    from eval.metrics import evaluate_solution
    return evaluate_solution(instance, trainer.solve(instance))["pf_utility"]


@register
class RapFedWMMSE(Algorithm):
    name = "rap_fedwmmse"
    label = "RAP-FedWMMSE"
    needs_fit = True
    fl_mode = "rap"

    def fit(self, config, seed, deployment=0):
        self.trainer = RapFedWMMSETrainer(config, seed, deployment, self.fl_mode)
        return self.trainer.run()

    def solve(self, instance):
        return self.trainer.solve(instance)


@register
class RapFedWMMSEFedAvg(RapFedWMMSE):
    name = "rap_fedwmmse_fedavg"
    label = "RAP-FedWMMSE with FedAvg"
    fl_mode = "fedavg"


@register
class RapFedWMMSELocal(RapFedWMMSE):
    name = "rap_fedwmmse_nofl"
    label = "RAP-FedWMMSE, no FL"
    fl_mode = "none"


@register
class UnfoldedWMMSE(Algorithm):
    """Ablation: the same unfolded layers with fixed, untrained parameters and no slow-timescale
    adaptation. It shows what the learning adds."""

    name = "unfolded_wmmse"
    label = "Unfolded WMMSE, untrained"
    relaxation = INITIAL_DAMPING

    @torch.no_grad()
    def solve(self, instance):
        config, scenario = instance.config, instance.scenario
        layers = config["fedwmmse"]["layers"]
        net = network_from(scenario, [instance.channels], config, scenario.relay_assoc, instance.tau)
        W0, V0 = initial_beams(net, torch.zeros(1, net.h.shape[1], net.h.shape[2], dtype=RDTYPE),
                               torch.zeros(1, net.g.shape[1], net.g.shape[2], dtype=RDTYPE))
        duals = (torch.zeros(1, net.h.shape[2], dtype=RDTYPE), torch.zeros(1, net.g.shape[1], net.h.shape[2], dtype=RDTYPE))
        s = config["fedwmmse"]
        params = fixed_params(net, layers, damping=self.relaxation)
        W, V, _ = unfold(net, W0, V0, params, duals, layers, s["softmin_kappa"], s["eta_q"], s["eta_d"])
        W, V = W[0].numpy(), V[0].numpy()
        U = mmse_combiners(instance.channels.G_hat, W, scenario.noise_var_relay)
        return BeamSolution(W, V, U, instance.tau, np.asarray(scenario.relay_assoc), info={"layers": layers})


@register
class ClassicalWMMSELayers(UnfoldedWMMSE):
    """Diagnostic: the same layers with eta = delta = lambda_reg = 1, which is undamped classical
    per-node WMMSE with every node updating in parallel (tests/test_rap_fedwmmse.py checks the
    equivalence). No slow timescale, tau fixed."""

    name = "classical_wmmse_layers"
    label = "Per-node WMMSE, undamped"
    relaxation = 1.0


class _UntrainedTrainer(RapFedWMMSETrainer):
    """RAP-FedWMMSE's deployment path (warm-up epochs, tau search, QoS-priced association, in-layer
    prices started from the slow prices) driven by the fixed, untrained layer parameters instead of
    the node models. It isolates the training-free slow timescale (Steps 1-2) from the learning."""

    def __init__(self, config, seed, deployment, relaxation):
        super().__init__(config, seed, deployment, "none")
        self.relaxation = relaxation

    def policy(self, net, coordination, layers=None):
        layers = self.settings["layers"] if layers is None else layers
        B, M, K, R = net.batch, net.h.shape[1], net.h.shape[2], net.g.shape[1]
        W0, V0 = initial_beams(net, torch.zeros(B, M, K, dtype=RDTYPE), torch.zeros(B, R, K, dtype=RDTYPE))
        duals = (torch.as_tensor(coordination.qos, dtype=RDTYPE).expand(B, -1).clone(),
                 torch.as_tensor(coordination.df, dtype=RDTYPE).expand(B, -1, -1).clone())
        params = fixed_params(net, layers, damping=self.relaxation)
        return unfold(net, W0, V0, params, duals, layers, self.settings["softmin_kappa"], self.settings["eta_q"], self.settings["eta_d"])


@register
class UnfoldedWMMSESlow(Algorithm):
    """Ablation: the untrained layers plus the training-free slow timescale (Steps 1-2 and the
    warm-up), exactly as RAP-FedWMMSE deploys, but with no learned parameters."""

    name = "unfolded_wmmse_slow"
    label = "Unfolded WMMSE, untrained, with Steps 1-2"
    relaxation = INITIAL_DAMPING

    def solve(self, instance):
        return _UntrainedTrainer(instance.config, 0, 0, self.relaxation).solve(instance)


@register
class UnfoldedWMMSETauOnly(UnfoldedWMMSESlow):
    """Diagnostic: the untrained layers with Step 2's tau search alone. The association keeps zero
    prices and the slow prices stay at zero, so it separates tau adaptation from the priced
    association and the warmed-up prices."""

    name = "unfolded_wmmse_tau_only"
    label = "Unfolded WMMSE, untrained, with Step 2 only"
    overrides = {"adapt_association": False, "eta_q_epoch": 0.0, "eta_d_epoch": 0.0}

    def solve(self, instance):
        trainer = _UntrainedTrainer(instance.config, 0, 0, self.relaxation)
        trainer.settings = {**trainer.settings, **self.overrides}
        return trainer.solve(instance)


@register
class ClassicalWMMSESlow(UnfoldedWMMSESlow):
    name = "classical_wmmse_slow"
    label = "Per-node WMMSE, undamped, with Steps 1-2"
    relaxation = 1.0
