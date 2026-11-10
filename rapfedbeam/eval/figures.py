"""
eval/figures.py
Responsibility: draw the 14 planned figures (docs/GRAPH_PLAN.md) from saved results only, with a CSV of
the plotted numbers beside each PNG. An algorithm keeps its color and marker in every figure.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.algorithms import REGISTRY, get_algorithm

SURFACE, INK, INK_SECONDARY, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BOOTSTRAP_SAMPLES = 1000

# Fixed slots: the color, marker and line style follow the algorithm, whichever others are present.
# The two RAP-FedWMMSE ablations share its hue and differ by line style.
STYLE = {
    "rap_fedwmmse": ("#2a78d6", "o", "-"),
    "rap_fedwmmse_fedavg": ("#2a78d6", "o", "--"),
    "rap_fedwmmse_nofl": ("#2a78d6", "o", ":"),
    "centralized_wmmse": ("#eb6834", "s", "-"),
    "local_wmmse": ("#1baf7a", "^", "-"),
    "rzf_equal": ("#eda100", "D", "-"),
    "mrt_equal": ("#e87ba4", "v", "-"),
    "centralized_no_relay": ("#008300", "P", "-"),
    "exact_minlp": ("#6250d6", "X", "-"),
    "unfolded_wmmse": ("#e34948", "*", "-"),
    "mrt_fractional": ("#52514e", "h", "-"),
    "neural_ctde": ("#52514e", "h", "--"),
}

# Graphs that compare every method show the main comparison set only, so they stay readable.
MAIN_COMPARISON = ["rap_fedwmmse", "centralized_wmmse", "centralized_no_relay", "local_wmmse", "rzf_equal", "mrt_equal"]


def _label(name):
    get_algorithm("mrt_equal")  # makes sure late registrations are loaded
    return REGISTRY[name].label


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def _axes(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150, facecolor=SURFACE)
    _style(ax, title, xlabel, ylabel)
    return fig, ax


def _axes_pair(suptitle, panels):
    """Two panels side by side; panels = [(title, xlabel, ylabel), (title, xlabel, ylabel)]."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=150, facecolor=SURFACE)
    fig.suptitle(suptitle, color=INK, fontsize=11, x=0.01, ha="left")
    for ax, (title, xlabel, ylabel) in zip(axes, panels):
        _style(ax, title, xlabel, ylabel)
    return fig, axes


def _save(fig, ax, out_dir, stem, header, rows):
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        # Below the plot, so it can never cover a line or a band.
        fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="lower center", ncol=3)
        fig.tight_layout(rect=(0, 0.06 + 0.035 * ((len(handles) - 1) // 3), 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{stem}.png", facecolor=SURFACE)
    plt.close(fig)
    with open(Path(out_dir) / f"{stem}.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _main(payload):
    return [name for name in MAIN_COMPARISON if name in payload["algorithms"]]


def _load(results_dir, name):
    path = Path(results_dir) / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _good(records):
    return [record for record in records if "error" not in record]


def _ci(values):
    values = np.asarray(values, dtype=float)
    return 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0


def _sweep_stat(point, name, statistic):
    records = _good(point["results"].get(name, []))
    if not records:
        return float("nan"), 0.0
    if statistic == "pf_utility":
        values = np.array([record["pf_utility"] for record in records])
    elif statistic == "mean_rate":
        values = np.array([np.mean(record["rates"]) for record in records])
    elif statistic == "total_power":
        values = np.array([record["total_power"] for record in records])
    elif statistic == "p5_rate":
        # A single percentile over all UEs and drops. Its 95% interval comes from a bootstrap that
        # resamples whole drops, since the UEs of one drop are not independent.
        per_drop = [np.asarray(record["rates"]) for record in records]
        estimate = float(np.percentile(np.concatenate(per_drop), 5))
        if len(per_drop) < 2:
            return estimate, 0.0
        rng = np.random.default_rng(0)
        resampled = [np.percentile(np.concatenate([per_drop[i] for i in rng.integers(0, len(per_drop), len(per_drop))]), 5)
                     for _ in range(BOOTSTRAP_SAMPLES)]
        low, high = np.percentile(resampled, [2.5, 97.5])
        return estimate, float((high - low) / 2.0)
    elif statistic == "seconds":
        values = np.array([record["seconds"] for record in records])
    else:
        raise ValueError(statistic)
    return float(values.mean()), float(_ci(values))


def _draw_sweep(ax, payload, statistic, rows, names=None, line_style=None, suffix=""):
    x_values = [point["x"] for point in payload["points"]]
    for name in names or payload["algorithms"]:
        stats = [_sweep_stat(point, name, statistic) for point in payload["points"]]
        means = np.array([value for value, _ in stats])
        if np.all(np.isnan(means)):
            continue
        half = np.array([width for _, width in stats])
        color, marker, line = STYLE[name]
        ax.fill_between(x_values, means - half, means + half, color=color, alpha=0.12, linewidth=0)
        ax.plot(x_values, means, color=color, marker=marker, markersize=6, linewidth=1.5, linestyle=line_style or line,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=_label(name) + suffix)
        rows.extend([x, name + suffix, mean, width] for x, mean, width in zip(x_values, means, half))
    return x_values


def _plot_sweep(payload, statistic, ylabel, title, out_dir, stem, log_y=False, overlay=None):
    fig, ax = _axes(f"{title} ({payload['drops']} drops)", payload["x_label"], ylabel)
    rows = []
    x_values = _draw_sweep(ax, payload, statistic, rows)
    if overlay:
        extra = _draw_sweep(ax, overlay, statistic, rows, line_style="--", suffix=", L_max = 2")
        x_values = sorted(set(x_values) | set(extra))
    ax.set_xticks(x_values)  # the tested values only, which rules out ticks like "1.5 relays"
    if log_y:
        ax.set_yscale("log")
    _save(fig, ax, out_dir, stem, ["x", "algorithm", statistic, "ci95_half_width"], rows)


def figure_rate_cdf(results_dir, out_dir):
    payload = _load(results_dir, "baseline")
    if not payload:
        return False
    fig, ax = _axes(f"G01  Per-user rate distribution ({payload['drops']} drops)", "Per-user rate (bit/s/Hz)", "CDF")
    rows = []
    for name in _main(payload):
        records = _good(payload["points"][0]["results"].get(name, []))
        if not records:
            continue
        values = np.sort(np.concatenate([record["rates"] for record in records]))
        color, _, line = STYLE[name]
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, linewidth=1.5, linestyle=line, label=_label(name))
        rows.extend([name, *quantile] for quantile in zip(np.percentile(values, [5, 25, 50, 75, 95]), [5, 25, 50, 75, 95]))
    _save(fig, ax, out_dir, "G01_rate_cdf", ["algorithm", "rate", "percentile"], rows)
    return True


def figure_outage(results_dir, out_dir):
    payload = _load(results_dir, "baseline")
    if not payload:
        return False
    thresholds = np.linspace(0.1, 3.0, 30)
    fig, ax = _axes(f"G02  Outage against the QoS target ({payload['drops']} drops)", "Target rate R_min (bit/s/Hz)", "Outage probability")
    rows = []
    for name in _main(payload):
        records = _good(payload["points"][0]["results"].get(name, []))
        if not records:
            continue
        values = np.concatenate([record["rates"] for record in records])
        outage = [float(np.mean(values < threshold)) for threshold in thresholds]
        color, _, line = STYLE[name]
        ax.plot(thresholds, outage, color=color, linewidth=1.5, linestyle=line, label=_label(name))
        rows.extend([threshold, name, value] for threshold, value in zip(thresholds, outage))
    target = payload["config"]["qos"]["epsilon_k"]
    ax.axhline(target, color=AXIS, linewidth=1.0, linestyle="--")
    ax.text(thresholds[-1], target, f"Permitted outage ε = {target}", color=INK_SECONDARY, fontsize=8, va="bottom", ha="right")
    _save(fig, ax, out_dir, "G02_outage", ["r_min", "algorithm", "outage"], rows)
    return True


def figure_power(results_dir, out_dir):
    """G03: (a) mean rate and (b) 5th-percentile rate against transmit power."""
    payload = _load(results_dir, "power")
    if not payload:
        return False
    fig, axes = _axes_pair(f"G03  Rate vs transmit power ({payload['drops']} drops)",
                           [("(a) Mean rate", payload["x_label"], "Mean rate (bit/s/Hz)"),
                            ("(b) Cell-edge rate", payload["x_label"], "5th-percentile rate (bit/s/Hz)")])
    rows = []
    for ax, statistic in zip(axes, ("mean_rate", "p5_rate")):
        panel_rows = []
        ax.set_xticks(_draw_sweep(ax, payload, statistic, panel_rows))
        rows.extend([statistic, *row] for row in panel_rows)
    _save(fig, axes[0], out_dir, "G03_rate_vs_power", ["statistic", "x", "algorithm", "value", "ci95_half_width"], rows)
    return True


def figure_layers(results_dir, out_dir):
    """G09: PF utility against unfolded layers. The other algorithms solve the same problem (same drops,
    tau = 0.5, same association: the tau sweep at 0.5) and are drawn as reference lines."""
    payload = _load(results_dir, "layers")
    if not payload:
        return False
    fig, ax = _axes(f"G09  PF utility vs unfolded layers ({payload['drops']} drops)", payload["x_label"], "PF utility")
    rows = []
    x_values = _draw_sweep(ax, payload, "pf_utility", rows)
    tau = _load(results_dir, "tau")
    same = [point for point in tau["points"] if np.isclose(float(point["x"]), 0.5)] if tau else []
    if same:
        for name in [name for name in MAIN_COMPARISON if name != "rap_fedwmmse"]:
            value, width = _sweep_stat(same[0], name, "pf_utility")
            if np.isnan(value):
                continue
            color, _, _ = STYLE[name]
            ax.axhline(value, color=color, linewidth=1.2, linestyle=(0, (4, 3)), label=f"{_label(name)} (converged)")
            rows.append(["reference", name, value, width])
    ax.set_xticks(x_values)
    _save(fig, ax, out_dir, "G09_pf_vs_layers", ["x", "algorithm", "pf_utility", "ci95_half_width"], rows)
    return True


def figure_optimality_gap(results_dir, out_dir):
    payload = _load(results_dir, "optimality_gap")
    if not payload:
        return False
    results = payload["points"][0]["results"]
    exact = {record["drop"]: record for record in _good(results.get("exact_minlp", []))}
    if not exact:
        return False
    names, means, half_widths = [], [], []
    for name in payload["algorithms"]:
        gaps = [exact[record["drop"]]["info"]["upper_bound"] - record["pf_utility"] for record in _good(results.get(name, [])) if record["drop"] in exact]
        if gaps:
            names.append(name)
            means.append(float(np.mean(gaps)))
            half_widths.append(float(_ci(gaps)))
    fig, ax = _axes(f"G10  Distance from the certified MINLP upper bound, small network ({payload['drops']} drops)",
                    "PF utility below SCIP's upper bound", "")
    positions = np.arange(len(names))[::-1]
    ax.barh(positions, means, height=0.55, color=[STYLE[name][0] for name in names], xerr=half_widths, error_kw={"ecolor": INK_SECONDARY, "linewidth": 1.0})
    ax.set_yticks(positions, [_label(name) for name in names])
    ax.grid(False, axis="y")
    for position, value in zip(positions, means):
        ax.annotate(f"{value:.3f}", (value, position), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    _save(fig, ax, out_dir, "G10_optimality_gap", ["algorithm", "mean_gap", "ci95_half_width"], list(zip(names, means, half_widths)))
    return True


def figure_fl(results_dir, out_dir):
    """G12: validation PF utility (a) per federated epoch and (b) against cumulative FL bytes."""
    payload = _load(results_dir, "baseline")
    history = payload["points"][0].get("fit_history") if payload else None
    if not history:
        return False
    deployments = len(next(iter(history.values())))
    fig, axes = _axes_pair(f"G12  Federated learning of RAP-FedWMMSE ({deployments} deployments)",
                           [("(a) Learning curve", "Federated epoch", "Validation PF utility"),
                            ("(b) Utility vs communication", "Cumulative FL bytes exchanged (log scale)", "Validation PF utility")])
    rows = []
    for ax, key in zip(axes, ("round", "cumulative_bytes")):
        for name, runs in history.items():
            length = min(len(run["history"]) for run in runs)
            xs = np.mean([[record[key] for record in run["history"][:length]] for run in runs], axis=0)
            ys = np.mean([[record["pf_utility"] for record in run["history"][:length]] for run in runs], axis=0)
            color, _, line = STYLE[name]
            rows.extend([key, x, name, y] for x, y in zip(xs, ys))
            if key == "cumulative_bytes" and xs[-1] == 0:
                # A method that never communicates has no place on a bytes axis: show its final level.
                ax.axhline(ys[-1], color=color, linewidth=1.5, linestyle=line)
                continue
            ax.plot(xs, ys, color=color, linewidth=1.5, linestyle=line, label=_label(name))
        if key == "cumulative_bytes":
            ax.set_xscale("log")
    _save(fig, axes[0], out_dir, "G12_fl_learning", ["axis", "x", "algorithm", "pf_utility"], rows)
    return True


def figure_energy(results_dir, out_dir):
    """G13: the rate-energy trade-off that lambda_E * P_tot in Sir's objective controls. Each curve
    runs over lambda_E; methods without a power term stay at full power."""
    payload = _load(results_dir, "energy")
    if not payload:
        return False
    fig, ax = _axes(f"G13  PF utility vs network power consumption, swept by λ_E ({payload['drops']} drops)",
                    "Total network power P_tot (W)", "PF utility")
    rows = []
    for name in _main(payload):
        stats = [(point["x"], _sweep_stat(point, name, "total_power"), _sweep_stat(point, name, "pf_utility")) for point in payload["points"]]
        powers = np.array([power for _, (power, _), _ in stats])
        utilities = np.array([utility for _, _, (utility, _) in stats])
        if np.all(np.isnan(utilities)):
            continue
        color, marker, line = STYLE[name]
        ax.plot(powers, utilities, color=color, marker=marker, markersize=6, linewidth=1.5, linestyle=line,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=_label(name))
        rows.extend([lam, name, power, power_ci, utility, utility_ci] for lam, (power, power_ci), (utility, utility_ci) in stats)
        if name == "rap_fedwmmse":
            for lam, power, utility in zip([x for x, _, _ in stats], powers, utilities):
                ax.annotate(f"λ_E={lam:g}", (power, utility), xytext=(4, -10), textcoords="offset points", fontsize=7, color=color)
    _save(fig, ax, out_dir, "G13_energy_tradeoff", ["lambda_e", "algorithm", "total_power_w", "power_ci95", "pf_utility", "pf_ci95"], rows)
    return True


def figure_fronthaul(results_dir, out_dir):
    """G14: (a) PF utility and (b) FL participation against the per-AP fronthaul capacity."""
    payload = _load(results_dir, "fronthaul")
    if not payload:
        return False
    fig, axes = _axes_pair(f"G14  Federated learning under the fronthaul constraint ({payload['drops']} drops)",
                           [("(a) PF utility on the evaluation drops", payload["x_label"], "PF utility"),
                            ("(b) FL participation per epoch", payload["x_label"], "Nodes per epoch")])
    rows = []
    x_values = _draw_sweep(axes[0], payload, "pf_utility", rows)
    axes[0].set_xscale("log")
    axes[0].set_xticks(x_values, [str(x) for x in x_values])
    for name in payload["algorithms"]:
        participants, overloaded = [], []
        for point in payload["points"]:
            runs = point.get("fit_history", {}).get(name, [])
            participants.append(np.mean([run["history"][-1]["mean_participants"] for run in runs]) if runs else np.nan)
            overloaded.append(np.mean([run["history"][-1]["mean_overloaded_aps"] for run in runs]) if runs else np.nan)
        color, marker, line = STYLE[name]
        axes[1].plot(x_values, participants, color=color, marker=marker, markersize=6, linewidth=1.5, linestyle=line,
                     markeredgecolor=SURFACE, markeredgewidth=1.0)
        rows.extend([x, name + " (participants)", value, ""] for x, value in zip(x_values, participants))
        if name == "rap_fedwmmse":
            axes[1].plot(x_values, overloaded, color=AXIS, linewidth=1.2, linestyle="--")
            rows.extend([x, "overloaded_aps", value, ""] for x, value in zip(x_values, overloaded))
    axes[1].set_xscale("log")
    axes[1].set_xticks(x_values, [str(x) for x in x_values])
    # The shared legend is built from panel (a), so panel (b)'s extra line gets a proxy entry there.
    axes[0].plot([], [], color=AXIS, linewidth=1.2, linestyle="--", label="(b) APs whose user data alone exceeds F_max")
    _save(fig, axes[0], out_dir, "G14_fronthaul", ["x", "series", "value", "ci95_half_width"], rows)
    return True


SWEEP_FIGURES = [
    ("G04_pf_vs_ues", "num_ues", "pf_utility", "PF utility", "G04  PF utility vs number of UEs", False, None),
    ("G05_pf_vs_blocked", "blocked_fraction", "pf_utility", "PF utility", "G05  PF utility vs blocked UEs", False, None),
    ("G06_pf_vs_relays", "num_relays", "pf_utility", "PF utility", "G06  PF utility vs number of relays", False, "cooperation"),
    ("G07_pf_vs_tau", "tau", "pf_utility", "PF utility", "G07  PF utility vs phase split", False, None),
    ("G08_pf_vs_aging", "aging", "pf_utility", "PF utility", "G08  PF utility vs channel aging", False, None),
    ("G11_runtime_vs_ues", "num_ues", "seconds", "Seconds per drop (log scale)", "G11  Runtime vs number of UEs", True, None),
]


def draw_all(results_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    drawn, skipped = [], []
    for stem, function in (("G01_rate_cdf", figure_rate_cdf), ("G02_outage", figure_outage), ("G03_rate_vs_power", figure_power),
                           ("G09_pf_vs_layers", figure_layers), ("G10_optimality_gap", figure_optimality_gap),
                           ("G12_fl_learning", figure_fl), ("G13_energy_tradeoff", figure_energy), ("G14_fronthaul", figure_fronthaul)):
        (drawn if function(results_dir, out_dir) else skipped).append(stem)
    for stem, experiment, statistic, ylabel, title, log_y, overlay in SWEEP_FIGURES:
        payload = _load(results_dir, experiment)
        if payload:
            _plot_sweep(payload, statistic, ylabel, title, out_dir, stem, log_y, _load(results_dir, overlay) if overlay else None)
            drawn.append(stem)
        else:
            skipped.append(stem)
    return drawn, sorted(skipped)
