"""
train/unfolded_wmmse.py
Responsibility: the differentiable physical layer and the unfolded WMMSE layers of RAP-FedWMMSE
(Steps 5-9 of the algorithm), for a batch of coherence blocks in PyTorch.

AP m's update reads only its own h_mk and G_mr plus the coordination vector zeta, and relay r's only
its own g_rk plus zeta (ap_step, relay_step). tests/test_rap_fedwmmse.py checks that the gradient
built this way equals the gradient of the global objective J.
Derivation and interpretation choices: docs/RAP_FEDWMMSE.md
"""

from dataclasses import dataclass
import math

import numpy as np
import torch

from sim.channels import channel_amplitude, noise_power_w
from sim.rates import as_assoc_matrix, ap_target_channels, mrt_beam_directions

LN2 = math.log(2.0)
CDTYPE = torch.complex128
RDTYPE = torch.float64
EPS = 1e-9            # Sir's epsilon in rho = 1/(e + eps) and in the combiner normalization
NOT_ASSOCIATED = 1e6  # stands in for C_AR of a pair with a_rk = 0, so it never wins the min
TINY = 1e-30          # keeps square roots of zero differentiable
LOADING = 1e-10       # relative diagonal load that keeps a rank-deficient beam update solvable


def sq(z):
    """|z|^2, written so its derivative exists at z = 0."""
    return z.real ** 2 + z.imag ** 2


@dataclass
class Network:
    """A batch of coherence blocks on one geometry, with the slow-timescale decisions fixed."""
    h: torch.Tensor          # [B, M, K, N_A], gain h^H w
    G: torch.Tensor          # [B, M, R, N_R, N_A]
    g: torch.Tensor          # [B, R, K, N_R]
    cluster: torch.Tensor    # [M, K], c_mk as 0/1
    assoc: torch.Tensor      # [R, K], a_rk as 0/1
    ap_dirs: torch.Tensor    # [B, M, K, N_A], unit MRT directions towards each stream's receiver
    relay_dirs: torch.Tensor # [B, R, K, N_R]
    beta_ap_db: torch.Tensor     # [M, K]
    beta_relay_db: torch.Tensor  # [R, K]
    amp_h: torch.Tensor      # [M, K], large-scale amplitudes for CSI-error samples
    amp_G: torch.Tensor      # [M, R]
    amp_g: torch.Tensor      # [R, K]
    weights: torch.Tensor    # [K], omega_k
    tau: float
    noise_ue: float
    noise_relay: float
    p_ap: float
    p_relay: float
    eta_ap: float            # power-amplifier efficiencies
    eta_relay: float
    lambda_e: float
    epsilon_0: float
    r_min: float
    eps_out: float           # epsilon_k, permitted outage

    @property
    def batch(self):
        return self.h.shape[0]


def assoc_matrix(assoc, num_relays):
    """a_rk as floats [R, K], from a relay index per UE or the binary matrix (L_max > 1)."""
    return as_assoc_matrix(assoc, num_relays).astype(float)


def network_from(scenario, channel_sets, config, assoc, tau, true_channels=False):
    """Stack coherence blocks of one scenario. Uses the estimates unless true_channels is set."""
    if true_channels:
        h, G, g = (np.stack(x) for x in zip(*[(c.h_mk, c.G_mr, c.g_rk) for c in channel_sets]))
    else:
        h, G, g = (np.stack(x) for x in zip(*[(c.h_hat, c.G_hat, c.g_hat) for c in channel_sets]))
    network, power = config["network"], config["power"]
    targets = np.stack([ap_target_channels(c.h_hat, c.G_hat, assoc) for c in channel_sets])
    ap_dirs = targets / np.maximum(np.linalg.norm(targets, axis=-1, keepdims=True), 1e-300)
    relay_dirs = np.stack([mrt_beam_directions(c.g_hat) for c in channel_sets])
    noise_ue, noise_relay = noise_power_w(config, "noise_figure_ue_db"), noise_power_w(config, "noise_figure_relay_db")
    as_c = lambda x: torch.as_tensor(np.asarray(x), dtype=CDTYPE)
    as_r = lambda x: torch.as_tensor(np.asarray(x, dtype=float), dtype=RDTYPE)
    weights = config.get("objective", {}).get("ue_weights", np.ones(network["num_ues"]))
    return Network(
        h=as_c(h), G=as_c(G), g=as_c(g),
        cluster=as_r(scenario.cluster_mask), assoc=as_r(assoc_matrix(assoc, network["num_relays"])),
        ap_dirs=as_c(ap_dirs), relay_dirs=as_c(relay_dirs),
        beta_ap_db=as_r(scenario.beta_ap_ue_db), beta_relay_db=as_r(scenario.beta_relay_ue_db),
        amp_h=as_r(channel_amplitude(scenario.beta_ap_ue_db, noise_ue)),
        amp_G=as_r(channel_amplitude(scenario.beta_ap_relay_db, noise_relay)),
        amp_g=as_r(channel_amplitude(scenario.beta_relay_ue_db, noise_ue)),
        weights=as_r(weights), tau=float(tau),
        noise_ue=float(scenario.noise_var_ue), noise_relay=float(scenario.noise_var_relay),
        p_ap=float(power["p_max_ap"]), p_relay=float(power["p_max_relay"]),
        eta_ap=float(power["pa_efficiency_ap"]), eta_relay=float(power["pa_efficiency_relay"]),
        lambda_e=float(config.get("objective", {}).get("lambda_e", 0.0)),
        epsilon_0=float(config.get("objective", {}).get("epsilon_0", 1e-3)),
        r_min=float(config["qos"]["r_min"]), eps_out=float(config["qos"]["epsilon_k"]),
    )


# --- physical layer (Step 5) ---

def gains(h, G, g, W, V):
    """S1[b,k,j] = sum_m h_mk^H w_mj,  q[b,r,:,k] = sum_m G_mr w_mk,  S2[b,k,j] = sum_r g_rk^H v_rj."""
    S1 = torch.einsum("bmka,bmja->bkj", h.conj(), W)
    q = torch.einsum("bmrna,bmka->brnk", G, W)
    S2 = torch.einsum("brkn,brjn->bkj", g.conj(), V)
    return S1, q, S2


def relay_combiners(q, noise_relay):
    """u_rk = (q_rk q_rk^H + R^I_rk + eps I)^-1 q_rk, normalized; the matrix is the same for every k."""
    eye = torch.eye(q.shape[2], dtype=CDTYPE)
    covariance = q @ q.conj().transpose(-1, -2) + (noise_relay + EPS) * eye
    u_tilde = torch.linalg.solve(covariance, q)
    return u_tilde / (torch.sqrt(sq(u_tilde).sum(dim=2, keepdim=True) + TINY) + EPS)


def relay_gains(u, q):
    """SAR[b,r,k,j] = u_rk^H q_rj, the gain of stream j after relay r's combiner for stream k."""
    return torch.einsum("brnk,brnj->brkj", u.conj(), q)


def sinr(S, noise):
    power = sq(S)
    desired = torch.diagonal(power, dim1=-2, dim2=-1)
    return desired / (power.sum(-1) - desired + noise)


def mmse_equalizer(S, noise):
    """Step 6: scalar MMSE equalizer c_k and its MSE, for gains S[..., k, j]."""
    total = sq(S).sum(-1) + noise
    desired = torch.diagonal(S, dim1=-2, dim2=-1)
    return desired / total, 1.0 - sq(desired) / total


def link_rates(net, S1, SAR, S2, u):
    noise_ar = net.noise_relay * sq(u).sum(dim=2)       # sigma_r^2 ||u_rk||^2, [B, R, K]
    gamma1, gamma2, gamma_ar = sinr(S1, net.noise_ue), sinr(S2, net.noise_ue), sinr(SAR, noise_ar)
    C_D = net.tau * torch.log2(1.0 + gamma1) + (1.0 - net.tau) * torch.log2(1.0 + gamma2)
    C_AR = net.tau * torch.log2(1.0 + gamma_ar)
    return C_D, C_AR, (gamma1, gamma2, gamma_ar, noise_ar)


def user_rates(C_D, C_AR, assoc, kappa=None):
    """r_k = min(C_D_k, C_AR_rk for a_rk = 1), as the soft-min with kappa or the hard min without.
    Also returns d r_k / d C_D_k and d r_k / d C_AR_rk."""
    masked = torch.where(assoc.bool()[None], C_AR, torch.full_like(C_AR, NOT_ASSOCIATED))
    candidates = torch.cat([C_D[:, None], masked], dim=1)            # [B, 1 + R, K]
    if kappa is None:
        rate, index = candidates.min(dim=1)
        share = torch.nn.functional.one_hot(index, candidates.shape[1]).permute(0, 2, 1).to(RDTYPE)
    else:
        rate = -torch.logsumexp(-kappa * candidates, dim=1) / kappa
        share = torch.softmax(-kappa * candidates, dim=1)
    return rate, share[:, 0], share[:, 1:]


# --- coordination vector (Steps 5, 6, 9) ---

@dataclass
class Zeta:
    """What the air interface delivers to every node in one layer: per-UE and per-relay-stream
    scalars, and the relay combiners. No antenna-domain CSI of another node is included."""
    S1: torch.Tensor     # [B, K, K]
    SAR: torch.Tensor    # [B, R, K, K]
    S2: torch.Tensor     # [B, K, K]
    u: torch.Tensor      # [B, R, N_R, K]
    c1: torch.Tensor
    c2: torch.Tensor
    car: torch.Tensor
    A1: torch.Tensor     # [B, K], weight of UE k's Phase-I MSE in the gradient of J
    A2: torch.Tensor     # [B, K]
    AAR: torch.Tensor    # [B, R, K]
    rate: torch.Tensor   # [B, K]
    C_D: torch.Tensor
    C_AR: torch.Tensor
    gammas: tuple


def coordination(net, W, V, duals, kappa):
    """Measure zeta at the current beams.

    The link weights come from the envelope theorem: log(1 + SINR) = max over c, rho of
    (ln rho - rho e + 1), so d log2(1 + SINR) / d w* = -(rho / ln 2) d e / d w* at the MMSE c and
    rho = 1/e. Chaining through J gives the weight A of each link's MSE.
    """
    lam_q, nu = duals
    S1, q, S2 = gains(net.h, net.G, net.g, W, V)
    u = relay_combiners(q, net.noise_relay)
    SAR = relay_gains(u, q)
    C_D, C_AR, gammas = link_rates(net, S1, SAR, S2, u)
    rate, share_d, share_ar = user_rates(C_D, C_AR, net.assoc, kappa)
    c1, e1 = mmse_equalizer(S1, net.noise_ue)
    c2, e2 = mmse_equalizer(S2, net.noise_ue)
    car, ear = mmse_equalizer(SAR, gammas[3])
    rho1, rho2, rho_ar = 1.0 / (e1 + EPS), 1.0 / (e2 + EPS), 1.0 / (ear + EPS)
    dJ_dr = -net.weights / (rate + net.epsilon_0) - lam_q                    # [B, K]
    dJ_dCD = dJ_dr * share_d
    dJ_dCAR = dJ_dr[:, None, :] * share_ar - net.assoc * nu                   # DF price on the relay hop
    A1 = -dJ_dCD * net.tau * rho1 / LN2
    A2 = -dJ_dCD * (1.0 - net.tau) * rho2 / LN2
    AAR = -dJ_dCAR * net.tau * rho_ar / LN2 * net.assoc
    return Zeta(S1, SAR, S2, u, c1, c2, car, A1, A2, AAR, rate, C_D, C_AR, gammas)


def zeta_complex_numbers(num_ues, num_relays, n_antennas_relay):
    """Complex numbers in one exchange of zeta, for the overhead count."""
    per_ue = 2 * num_ues * num_ues + 2 * num_ues + 2 * num_ues               # S1, S2, c1, c2, A1, A2
    per_relay = num_relays * (num_ues * num_ues + 2 * num_ues + num_ues * n_antennas_relay)   # SAR, cAR, AAR, u
    return per_ue + per_relay


# --- local beam updates (Steps 7, 8) ---

def project(X, p_max):
    """Step 8: scale a node's beams onto sum ||x||^2 <= P_max, [B, K, N]."""
    norm = torch.sqrt(sq(X).sum(dim=(-2, -1)) + TINY)
    scale = math.sqrt(p_max) / torch.clamp(norm, min=math.sqrt(p_max))
    return X * scale[..., None, None]


@torch.no_grad()
def power_multiplier(Q, B, p_max, iterations=50):
    """The smallest mu >= 0 with ||(Q + mu I)^-1 B||_F^2 <= P_max: classical WMMSE's power price.
    Found by bisection on the eigenvalues, from the node's own Q and B. Q [..., N, N], B [..., N, K]."""
    eigenvalues, eigenvectors = torch.linalg.eigh(Q)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    projected = sq(eigenvectors.conj().transpose(-1, -2) @ B)                     # [..., N, K]
    weight = projected.sum(-1)                                                    # [B, N]
    floor = 1e-12 * torch.clamp(eigenvalues.max(-1).values, min=1e-300)

    def power(mu):
        return (weight / (eigenvalues + floor[..., None] + mu[..., None]) ** 2).sum(-1)

    low = torch.zeros_like(floor)
    high = torch.sqrt(weight.sum(-1) / p_max) + floor
    floor = torch.clamp(floor, min=1e-150)  # keeps (eigenvalue + floor)^2 from underflowing to 0
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        too_much = power(middle) > p_max
        low = torch.where(too_much, middle, low)
        high = torch.where(too_much, high, middle)
    return torch.where(power(torch.zeros_like(floor)) <= p_max, torch.zeros_like(high), high) + floor


def _regularized_step(grad, Q, W, step, damping, reg, p_max):
    """Step 7 with the learned regularization lambda = reg * mu, where mu is the node's power price.

    Adding (lambda / 2) ||W||^2 to the node's quadratic WMMSE surrogate and taking a Newton step
    of length eta gives W + eta ((Q + lambda I)^-1 B - W), with B = Q W - grad the surrogate's linear
    term. With eta = 1 and reg = 1 this is exactly the classical per-node WMMSE update, so the
    learning starts from a sound optimizer and adjusts it. Damping then mixes with the old beams.
    Shapes: grad, W [B, nodes, K, N]; Q [B, nodes, N, N]; step, damping, reg [B, nodes].
    """
    eye = torch.eye(Q.shape[-1], dtype=CDTYPE)
    B = Q @ W.transpose(-1, -2) - grad.transpose(-1, -2)                       # [B, nodes, N, K]
    # A node whose streams carry no weight in J (e.g. a relay whose hop is not the binding one
    # under the hard min) has Q = 0 and B = 0: it keeps its beams rather than solving 0 x = 0.
    flat = torch.diagonal(Q, dim1=-2, dim2=-1).real.sum(-1) <= 1e-30
    safe_Q = torch.where(flat[..., None, None], eye.expand_as(Q), Q)
    safe_B = torch.where(flat[..., None, None], W.transpose(-1, -2), B)
    lam = (reg * power_multiplier(safe_Q, safe_B, p_max)).to(CDTYPE)
    lam = torch.where(flat, torch.zeros_like(lam), lam)
    # Q has rank at most the number of weighted streams, so with N > streams and an inactive power
    # limit (lam ~ 0) it is singular. A load of 1e-10 of Q's scale keeps the solve defined and is
    # far below any physical term.
    load = (LOADING * torch.diagonal(safe_Q, dim1=-2, dim2=-1).real.mean(-1)).to(CDTYPE)
    target = torch.linalg.solve(safe_Q + (lam + load)[..., None, None] * eye, safe_B).transpose(-1, -2)
    raw = W + step[..., None, None].to(CDTYPE) * (target - W)
    damping = damping[..., None, None].to(CDTYPE)
    return (1.0 - damping) * W + damping * raw


def ap_gradient(h, G, W, z, net):
    """dJ/dW_m* and its curvature Q_m for every AP m, each from AP m's own CSI and zeta only.

    h [B, M, K, N_A], G [B, M, R, N_R, N_A], W [B, M, K, N_A]; index m of the outputs uses only
    index m of h, G and W. The rest is zeta, which is the same for every node.
    """
    weight1 = (z.A1 * sq(z.c1)).to(CDTYPE)                                     # [B, K]
    grad = torch.einsum("bk,bkj,bmka->bmja", weight1, z.S1, h) - (z.A1 * z.c1)[:, None, :, None] * h
    b = torch.einsum("bmrna,brnk->bmrka", G.conj(), z.u)                       # G_mr^H u_rk
    weight_ar = (z.AAR * sq(z.car)).to(CDTYPE)                                 # [B, R, K]
    grad = grad + torch.einsum("brk,brkj,bmrka->bmja", weight_ar, z.SAR, b) - torch.einsum("brk,bmrka->bmka", z.AAR * z.car, b)
    price = net.lambda_e / net.eta_ap
    grad = grad + price * W
    Q = torch.einsum("bk,bmka,bmkc->bmac", weight1, h, h.conj()) + torch.einsum("brk,bmrka,bmrkc->bmac", weight_ar, b, b.conj())
    Q = Q + price * torch.eye(h.shape[-1], dtype=CDTYPE)
    return grad, Q


def relay_gradient(g, V, z, net):
    """dJ/dV_r* and its curvature for every relay r, from relay r's own CSI and zeta only.
    g [B, R, K, N_R], V [B, R, K, N_R]."""
    weight2 = (z.A2 * sq(z.c2)).to(CDTYPE)
    grad = torch.einsum("bk,bkj,brka->brja", weight2, z.S2, g) - (z.A2 * z.c2)[:, None, :, None] * g
    price = net.lambda_e / net.eta_relay
    grad = grad + price * V
    Q = torch.einsum("bk,brka,brkc->brac", weight2, g, g.conj()) + price * torch.eye(g.shape[-1], dtype=CDTYPE)
    return grad, Q


def ap_step(W, z, net, step, damping, reg):
    """Every AP's update, in parallel. step, damping, reg [B, M]."""
    grad, Q = ap_gradient(net.h, net.G, W, z, net)
    mask = net.cluster[None, :, :, None]
    updated = _regularized_step(grad * mask, Q, W, step, damping, reg, net.p_ap) * mask
    return project(updated, net.p_ap)


def relay_step(V, z, net, step, damping, reg):
    """Every relay's update, in parallel. step, damping, reg [B, R]."""
    grad, Q = relay_gradient(net.g, V, z, net)
    mask = net.assoc[None, :, :, None]
    updated = _regularized_step(grad * mask, Q, V, step, damping, reg, net.p_relay) * mask
    return project(updated, net.p_relay)


def update_duals(duals, z, net, eta_q, eta_d):
    """Step 9, inside each layer. With r_k = min(C_D, C_AR), r_k - C_AR <= 0 always, so the DF
    price instead follows C_D - C_AR: it rises exactly when the relay hop is the bottleneck."""
    lam_q, nu = duals
    lam_q = torch.relu(lam_q + eta_q * (net.r_min - z.rate.detach()))
    nu = torch.relu(nu + eta_d * net.assoc * (z.C_D.detach()[:, None, :] - z.C_AR.detach()))
    return lam_q, nu


@dataclass
class LayerParams:
    """What the node models predict (Step 4), per block: [B, nodes, L] for step and damping,
    [B, nodes] for the regularization."""
    ap_step: torch.Tensor
    ap_damping: torch.Tensor
    ap_reg: torch.Tensor
    relay_step: torch.Tensor
    relay_damping: torch.Tensor
    relay_reg: torch.Tensor


def unfold(net, W, V, params, duals, layers, kappa, eta_q, eta_d):
    """Step 5 to Step 9 for `layers` layers. Every node updates in parallel from the same zeta."""
    trained_layers = params.ap_step.shape[-1]
    for layer in range(layers):
        index = min(layer, trained_layers - 1)
        z = coordination(net, W, V, duals, kappa)
        W = ap_step(W, z, net, params.ap_step[..., index], params.ap_damping[..., index], params.ap_reg)
        V = relay_step(V, z, net, params.relay_step[..., index], params.relay_damping[..., index], params.relay_reg)
        duals = update_duals(duals, z, net, eta_q, eta_d)
    return W, V, duals


def masked_softmax(logits, mask):
    """Softmax over the last axis restricted to mask; all-zero rows where mask is empty."""
    mask = mask.bool().expand_as(logits)
    filled = torch.where(mask, logits, torch.full_like(logits, -1e30))
    return torch.softmax(filled, dim=-1) * mask * mask.any(dim=-1, keepdim=True)


def initial_beams(net, ap_logits, relay_logits):
    """Step 4's model-informed start: MRT directions from local CSI, with learned power shares."""
    ap_share = masked_softmax(ap_logits, net.cluster[None])                   # [B, M, K]
    relay_share = masked_softmax(relay_logits, net.assoc[None])               # [B, R, K]
    W0 = torch.sqrt(net.p_ap * ap_share + TINY).to(CDTYPE)[..., None] * net.ap_dirs * net.cluster[None, :, :, None]
    V0 = torch.sqrt(net.p_relay * relay_share + TINY).to(CDTYPE)[..., None] * net.relay_dirs * net.assoc[None, :, :, None]
    return W0, V0


# --- objective and robust loss (Steps 9, 10) ---

def augmented_objective(net, W, V, duals, kappa):
    """J per block: -sum omega log(r + eps) + lambda_E P_tx + QoS penalty + DF penalty.
    P_circ is a constant and is left out. C_D is held fixed in the DF penalty, so the price only
    raises the relay hop rather than lowering the destination rate."""
    lam_q, nu = duals
    z = coordination(net, W, V, (lam_q, nu), kappa)
    p_tx = sq(W).sum(dim=(1, 2, 3)) / net.eta_ap + sq(V).sum(dim=(1, 2, 3)) / net.eta_relay
    J = (-(net.weights * torch.log(z.rate + net.epsilon_0)).sum(-1) + net.lambda_e * p_tx
         + (lam_q * (net.r_min - z.rate)).sum(-1)
         + (net.assoc * nu * (z.C_D.detach()[:, None, :] - z.C_AR)).sum(dim=(1, 2)))
    return J, z


def _complex_normal(shape, generator):
    real = torch.randn(shape, generator=generator, dtype=RDTYPE)
    imag = torch.randn(shape, generator=generator, dtype=RDTYPE)
    return torch.complex(real, imag) / math.sqrt(2.0)


def outage_surrogate(net, W, V, u, rho, samples, sharpness, kappa, generator):
    """Step 10: draw H ~ p(H | H_hat) from the aging model, score the fixed beams and combiners on
    each draw, and average sigmoid(sharpness (R_min - r)). Returns p_out per block and UE."""
    spread = math.sqrt(max(1.0 - rho ** 2, 0.0))
    outages = []
    for _ in range(samples):
        h = rho * net.h + spread * net.amp_h[None, :, :, None] * _complex_normal(net.h.shape, generator)
        G = rho * net.G + spread * net.amp_G[None, :, :, None, None] * _complex_normal(net.G.shape, generator)
        g = rho * net.g + spread * net.amp_g[None, :, :, None] * _complex_normal(net.g.shape, generator)
        S1, q, S2 = gains(h, G, g, W, V)
        C_D, C_AR, _ = link_rates(net, S1, relay_gains(u, q), S2, u)
        rate, _, _ = user_rates(C_D, C_AR, net.assoc, kappa)
        outages.append(torch.sigmoid(sharpness * (net.r_min - rate)))
    return torch.stack(outages).mean(dim=0)
