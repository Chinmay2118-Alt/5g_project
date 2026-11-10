"""
train/rap_fedwmmse.py
Responsibility: RAP-FedWMMSE, the main algorithm: role-aware personalized federated unfolded WMMSE.
The epoch loop follows Sir's pseudocode: slow-timescale association and tau (Steps 1-2), local
self-supervised training of each node's model through the unfolded WMMSE layers (Steps 3-11), then
event-triggered, budgeted, compressed, personalized federation of the backbones (Steps 12-15).
Derivation and interpretation choices: docs/RAP_FEDWMMSE.md
"""

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
from sim.rates import mmse_combiners
from sim.scenario import evolve_drop
from train.unfolded_wmmse import (CDTYPE, RDTYPE, LayerParams, augmented_objective, coordination, initial_beams,
                                  network_from, outage_surrogate, unfold, user_rates, zeta_complex_numbers)


# --- Steps 3-4: role-specific adapters, personalized backbone, optimizer-parameter heads ---

def _log_magnitude(x):
    return torch.log10(x.abs() + 1e-6) / 3.0


def ap_features(net, m, qos_prices):
    """x_m: AP m's own |h_mk|, |G_mr|, beta_m, cluster row, and the QoS prices in zeta. [B, F]"""
    batch = net.batch
    return torch.cat([
        _log_magnitude(net.h[:, m]).reshape(batch, -1),
        _log_magnitude(net.G[:, m]).reshape(batch, -1),
        ((net.beta_ap_db[m] + 100.0) / 30.0).expand(batch, -1),
        net.cluster[m].expand(batch, -1),
        qos_prices,
    ], dim=1).to(torch.float32)


def relay_features(net, r, qos_prices):
    """x_r: relay r's own |G_mr|, |g_rk|, beta_r, association row, and the QoS prices. [B, F]"""
    batch = net.batch
    return torch.cat([
        _log_magnitude(net.G[:, :, r]).reshape(batch, -1),
        _log_magnitude(net.g[:, r]).reshape(batch, -1),
        ((net.beta_relay_db[r] + 100.0) / 30.0).expand(batch, -1),
        net.assoc[r].expand(batch, -1),
        qos_prices,
    ], dim=1).to(torch.float32)


def feature_sizes(config):
    n = config["network"]
    M, R, K, NA, NR = n["num_aps"], n["num_relays"], n["num_ues"], n["n_antennas_ap"], n["n_antennas_relay"]
    return K * NA + R * NR * NA + 3 * K, M * NR * NA + K * NR + 3 * K


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
        self.qos_prices = np.zeros(self.K)
        self.df_prices = np.zeros((self.R, self.K))
        self.generator = torch.Generator().manual_seed(self.seed * 1000 + self.deployment)

    # Steps 3-4 for every node, then Steps 5-9
    def policy(self, net, qos_prices, df_prices, layers=None):
        layers = self.settings["layers"] if layers is None else layers
        prices = torch.as_tensor(qos_prices, dtype=RDTYPE).expand(net.batch, -1)
        ap_out = [self.nodes[f"AP{m}"](ap_features(net, m, prices)) for m in range(self.M)]
        relay_out = [self.nodes[f"R{r}"](relay_features(net, r, prices)) for r in range(self.R)]
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
        W, V, duals = self.policy(net, self.qos_prices, self.df_prices)
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
            for k, r in enumerate(assoc):
                if r >= 0:
                    rate[:, k] = np.minimum(rate[:, k], tau * np.log2(1 + gamma_ar[:, r, k]))
            value = float(np.mean(np.sum(weights * np.log(rate + epsilon_0), axis=1)))
            if value > best_value:
                best, best_value = float(tau), value
        return best

    def _slow_duals(self, prices, df, rates, z, assoc_mat):
        s = self.settings
        prices = np.maximum(0.0, prices + s["eta_q_epoch"] * (self.config["qos"]["r_min"] - rates.mean(axis=0)))
        gap = (z.C_D.detach().numpy()[:, None, :] - z.C_AR.detach().numpy()).mean(axis=0)
        df = np.maximum(0.0, df + s["eta_d_epoch"] * assoc_mat * gap)
        return prices, df

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
            streams = np.asarray(assoc) == r
            interference = relay_interference[r][streams].mean() / 3.0 if streams.any() else 0.0
            signatures[f"R{r}"] = np.concatenate([(scenario.beta_relay_ue_db[r] - center) / (spread * math.sqrt(self.K)),
                                                   [interference, streams.sum() / capacity, 0.0, 1.0]])
            roles[f"R{r}"] = "relay"
        return signatures, roles

    def _federate(self, signatures, roles, grad_norms):
        """Returns the bytes exchanged this epoch."""
        if self.fl_mode == "none":
            return 0, 0
        s, fl = self.settings, self.config["fl"]
        local = {i: backbone_vector(model) for i, model in self.nodes.items()}
        num_params = len(next(iter(local.values())))
        updates = {i: local[i] - self.theta_received[i] for i in self.nodes}
        signature_bytes = len(next(iter(signatures.values()))) * BYTES_PER_FLOAT

        if self.fl_mode == "fedavg":
            # FedAvg: everyone uploads a dense update, the server averages, everyone gets the result.
            mean = sum(updates.values()) / len(updates)
            for i, model in self.nodes.items():
                self.theta_received[i] = self.theta_received[i] + mean
                set_backbone_vector(model, self.theta_received[i])
            return 2 * len(self.nodes) * num_params * BYTES_PER_FLOAT, len(self.nodes)

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
        chosen = select_clients([scores[i] for i in ids], [scores[i] >= s["trigger_threshold"] for i in ids],
                                [size] * len(ids), s["budget_bytes"])
        selected = dict(zip(ids, chosen))
        sent = {}
        for i in ids:
            if selected[i]:
                sent[i], self.residual[i] = top_k(updates[i], self.residual[i], s["topk_fraction"])
        if not any(chosen):
            return 0, 0
        kappa = {"kappa_ap_ap": fl["kappa_ap_ap"], "kappa_relay_relay": fl["kappa_relay_relay"], "kappa_ap_relay": fl["kappa_ap_relay"]}
        aggregated = personalized_updates(signatures, roles, sent, selected, fl["kernel_sigma"], kappa, {i: 1.0 for i in ids})
        for i, model in self.nodes.items():
            step = fl["rho_p"] * aggregated[i]
            if selected[i]:
                self.theta_received[i] = self.theta_received[i] + step
                set_backbone_vector(model, self.theta_received[i])
                self.age[i] = 0
            else:
                # Its own update was not uploaded, so it keeps it and still counts as pending.
                set_backbone_vector(model, local[i] + step)
                self.theta_received[i] = self.theta_received[i] + step
        uploaded = int(sum(chosen)) * size
        downloaded = len(ids) * num_params * BYTES_PER_FLOAT
        return uploaded + downloaded, int(sum(chosen))

    def run(self):
        s, config = self.settings, self.config
        scenario, _ = make_scenario(config, self.seed, TRAINING_DROPS, self.deployment)
        rng = np.random.default_rng([self.seed, self.deployment, TRAINING_DROPS, 1])
        validation = [make_instance(config, self.seed, VALIDATION_DROPS + v, self.deployment) for v in range(s["validation_instances"])]
        history, total_bytes = [], 0
        for epoch in range(s["epochs"]):
            if epoch:
                scenario = evolve_drop(scenario, config, rng)
            assoc = self._association(scenario, self.qos_prices, s["adapt_association"])
            assoc_mat = np.zeros((self.R, self.K))
            for k, r in enumerate(assoc):
                if r >= 0:
                    assoc_mat[r, k] = 1.0
            epoch_rates, grad_norms = [], {}
            for _ in range(s["local_steps"]):
                blocks = [sample_full_channel_set(scenario, config, rng) for _ in range(s["batch_blocks"])]
                net = network_from(scenario, blocks, config, assoc, self.tau)
                rates, z, grad_norms = self._local_step(net)
                epoch_rates.append(rates.numpy())
            rates = np.concatenate(epoch_rates)
            self.qos_prices, self.df_prices = self._slow_duals(self.qos_prices, self.df_prices, rates, z, assoc_mat)
            if s["adapt_tau"]:
                self.tau = self._best_tau(z.gammas, assoc, net.weights.numpy(), net.epsilon_0, self.tau)
            signatures, roles = self._signatures(scenario, z, assoc)
            sent, participants = self._federate(signatures, roles, grad_norms)
            total_bytes += sent
            every = s["validation_every"]
            if (every and (epoch + 1) % every == 0) or epoch + 1 == s["epochs"]:
                utility = float(np.mean([evaluate(self, instance) for instance in validation]))
                history.append({"round": epoch + 1, "pf_utility": utility, "cumulative_bytes": total_bytes,
                                "participants": participants, "tau": self.tau})
        return history

    @torch.no_grad()
    def solve(self, instance):
        """Deployment: a short slow-timescale warm-up on the drop's statistics sets the QoS prices,
        association and tau, then the policy runs on the drop's estimated channels."""
        s, config = self.settings, instance.config
        scenario = instance.scenario
        key = hashlib.sha256(np.ascontiguousarray(scenario.ue_positions).tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        prices, df = np.zeros(self.K), np.zeros((self.R, self.K))
        tau = self.tau if s["adapt_tau"] else instance.tau
        for _ in range(s["warmup_epochs"]):
            assoc = self._association(scenario, prices, s["adapt_association"])
            assoc_mat = np.zeros((self.R, self.K))
            for k, r in enumerate(assoc):
                if r >= 0:
                    assoc_mat[r, k] = 1.0
            blocks = [sample_full_channel_set(scenario, config, rng) for _ in range(s["warmup_blocks"])]
            net = network_from(scenario, blocks, config, assoc, tau)
            W, V, duals = self.policy(net, prices, df)
            z = coordination(net, W, V, duals, None)
            hard, _, _ = user_rates(z.C_D, z.C_AR, net.assoc)
            prices, df = self._slow_duals(prices, df, hard.numpy(), z, assoc_mat)
            if s["adapt_tau"]:
                tau = self._best_tau(z.gammas, assoc, net.weights.numpy(), net.epsilon_0, tau)
        assoc = self._association(scenario, prices, s["adapt_association"])
        net = network_from(scenario, [instance.channels], config, assoc, tau)
        W, V, _ = self.policy(net, prices, df)
        W, V = W[0].numpy(), V[0].numpy()
        U = mmse_combiners(instance.channels.G_hat, W, scenario.noise_var_relay)
        info = {"tau": tau, "layers": s["layers"], "relayed_ues": int(np.sum(assoc >= 0)),
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

    @torch.no_grad()
    def solve(self, instance):
        config, scenario = instance.config, instance.scenario
        layers = config["fedwmmse"]["layers"]
        net = network_from(scenario, [instance.channels], config, scenario.relay_assoc, instance.tau)
        W0, V0 = initial_beams(net, torch.zeros(1, net.h.shape[1], net.h.shape[2], dtype=RDTYPE),
                               torch.zeros(1, net.g.shape[1], net.g.shape[2], dtype=RDTYPE))
        duals = (torch.zeros(1, net.h.shape[2], dtype=RDTYPE), torch.zeros(1, net.g.shape[1], net.h.shape[2], dtype=RDTYPE))
        s = config["fedwmmse"]
        W, V, _ = unfold(net, W0, V0, fixed_params(net, layers), duals, layers, s["softmin_kappa"], s["eta_q"], s["eta_d"])
        W, V = W[0].numpy(), V[0].numpy()
        U = mmse_combiners(instance.channels.G_hat, W, scenario.noise_var_relay)
        return BeamSolution(W, V, U, instance.tau, np.asarray(scenario.relay_assoc), info={"layers": layers})
