"""
eval/figures.py
Responsibility: draw the planned graphs (docs/GRAPH_PLAN.md) from saved results only, with a CSV of
the plotted numbers beside each PNG. An algorithm keeps its color and marker in every graph.
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


def _axes(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    return fig, ax


def _save(fig, ax, out_dir, stem, header, rows):
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        # Below the plot, so it can never cover a line or a band.
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{stem}.png", facecolor=SURFACE)
    plt.close(fig)
    with open(Path(out_dir) / f"{stem}.csv", "w", newline="") as handle:
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


def _sweep_stat(point, name, statistic):
    records = _good(point["results"].get(name, []))
    if not records:
        return float("nan"), 0.0
    if statistic == "pf_utility":
        values = np.array([record["pf_utility"] for record in records])
    elif statistic == "mean_rate":
        values = np.array([np.mean(record["rates"]) for record in records])
    elif statistic == "p5_rate":
        # A single percentile over all UEs and drops, so it carries no per-drop interval.
        return float(np.percentile(np.concatenate([record["rates"] for record in records]), 5)), 0.0
    elif statistic == "seconds":
        values = np.array([record["seconds"] for record in records])
    else:
        raise ValueError(statistic)
    half_width = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(half_width)


def _plot_sweep(payload, statistic, ylabel, title, out_dir, stem, log_y=False):
    fig, ax = _axes(f"{title} ({payload['drops']} drops)", payload["x_label"], ylabel)
    x_values = [point["x"] for point in payload["points"]]
    rows = []
    for name in payload["algorithms"]:
        stats = [_sweep_stat(point, name, statistic) for point in payload["points"]]
        means = np.array([value for value, _ in stats])
        if np.all(np.isnan(means)):
            continue
        half = np.array([width for _, width in stats])
        color, marker, line = STYLE[name]
        ax.fill_between(x_values, means - half, means + half, color=color, alpha=0.12, linewidth=0)
        ax.plot(x_values, means, color=color, marker=marker, markersize=6, linewidth=1.5, linestyle=line,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=_label(name))
        rows.extend([x, name, mean, width] for x, mean, width in zip(x_values, means, half))
    ax.set_xticks(x_values)  # the tested values only, which rules out ticks like "1.5 relays"
    if log_y:
        ax.set_yscale("log")
    _save(fig, ax, out_dir, stem, ["x", "algorithm", statistic, "ci95_half_width"], rows)


def figure_rate_cdf(results_dir, out_dir):
    payload = _load(results_dir, "baseline")
    if not payload:
        return False
    fig, ax = _axes(f"Per-user rate distribution ({payload['drops']} drops)", "Per-user rate (bit/s/Hz)", "CDF")
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
    fig, ax = _axes(f"Outage against the QoS target ({payload['drops']} drops)", "Target rate R_min (bit/s/Hz)", "Outage probability")
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


def figure_layers(results_dir, out_dir):
    """G10: PF utility against unfolded layers, with Centralized WMMSE as a reference line."""
    payload = _load(results_dir, "layers")
    if not payload:
        return False
    fig, ax = _axes(f"PF utility vs unfolded layers ({payload['drops']} drops)", payload["x_label"], "PF utility")
    x_values = [point["x"] for point in payload["points"]]
    rows = []
    for name in payload["algorithms"]:
        stats = [_sweep_stat(point, name, "pf_utility") for point in payload["points"]]
        means = np.array([value for value, _ in stats])
        half = np.array([width for _, width in stats])
        color, marker, line = STYLE[name]
        ax.fill_between(x_values, means - half, means + half, color=color, alpha=0.12, linewidth=0)
        ax.plot(x_values, means, color=color, marker=marker, markersize=6, linewidth=1.5, linestyle=line,
                markeredgecolor=SURFACE, markeredgewidth=1.0, label=_label(name))
        rows.extend([x, name, mean, width] for x, mean, width in zip(x_values, means, half))
    baseline = _load(results_dir, "baseline")
    reference = _good(baseline["points"][0]["results"].get("centralized_wmmse", [])) if baseline else []
    if reference:
        value = float(np.mean([record["pf_utility"] for record in reference]))
        iterations = float(np.mean([len(record["info"]["history"]) - 1 for record in reference]))
        ax.axhline(value, color=STYLE["centralized_wmmse"][0], linewidth=1.0, linestyle="--")
        ax.text(x_values[-1], value, f"Centralized WMMSE-SCA, {iterations:.0f} iterations on average", color=INK_SECONDARY,
                fontsize=8, va="bottom", ha="right")
        rows.append(["reference", "centralized_wmmse", value, iterations])
    ax.set_xticks(x_values)
    _save(fig, ax, out_dir, "G10_pf_vs_layers", ["x", "algorithm", "pf_utility", "ci95_half_width_or_iterations"], rows)
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
            half_widths.append(1.96 * float(np.std(gaps, ddof=1)) / np.sqrt(len(gaps)) if len(gaps) > 1 else 0.0)
    fig, ax = _axes(f"Distance from the certified upper bound, small network ({payload['drops']} drops)", "PF utility below SCIP's upper bound", "")
    positions = np.arange(len(names))[::-1]
    ax.barh(positions, means, height=0.55, color=[STYLE[name][0] for name in names], xerr=half_widths, error_kw={"ecolor": INK_SECONDARY, "linewidth": 1.0})
    ax.set_yticks(positions, [_label(name) for name in names])
    ax.grid(False, axis="y")
    for position, value in zip(positions, means):
        ax.annotate(f"{value:.3f}", (value, position), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    _save(fig, ax, out_dir, "G11_optimality_gap", ["algorithm", "mean_gap", "ci95_half_width"], list(zip(names, means, half_widths)))
    return True


def figure_fl(results_dir, out_dir):
    """G13/G14: validation PF utility per federated epoch, averaged over deployments."""
    payload = _load(results_dir, "baseline")
    history = payload["points"][0].get("fit_history") if payload else None
    if not history:
        return False
    for stem, key, xlabel, title in (("G13_fl_epochs", "round", "Federated epoch", "Learning curve"),
                                     ("G14_fl_bytes", "cumulative_bytes", "Cumulative FL bytes exchanged (log scale)", "Utility vs communication")):
        fig, ax = _axes(f"{title} ({len(next(iter(history.values())))} deployments)", xlabel, "Validation PF utility")
        rows = []
        for name, runs in history.items():
            length = min(len(run["history"]) for run in runs)
            xs = np.mean([[record[key] for record in run["history"][:length]] for run in runs], axis=0)
            ys = np.mean([[record["pf_utility"] for record in run["history"][:length]] for run in runs], axis=0)
            color, marker, line = STYLE[name]
            rows.extend([x, name, y] for x, y in zip(xs, ys))
            if key == "cumulative_bytes" and xs[-1] == 0:
                # A method that never communicates has no place on a bytes axis: show its final level.
                ax.axhline(ys[-1], color=color, linewidth=1.5, linestyle=line, label=f"{_label(name)} (0 bytes, final value)")
                continue
            ax.plot(xs, ys, color=color, linewidth=1.5, linestyle=line, label=_label(name))
        if key == "cumulative_bytes":
            ax.set_xscale("log")
        _save(fig, ax, out_dir, stem, [key, "algorithm", "pf_utility"], rows)
    return True


SWEEP_FIGURES = [
    ("G03_mean_rate_vs_power", "power", "mean_rate", "Mean rate (bit/s/Hz)", "Mean rate vs transmit power", False),
    ("G04_p5_rate_vs_power", "power", "p5_rate", "5th-percentile rate (bit/s/Hz)", "Cell-edge rate vs transmit power", False),
    ("G05_pf_vs_ues", "num_ues", "pf_utility", "PF utility", "PF utility vs number of UEs", False),
    ("G06_pf_vs_blocked", "blocked_fraction", "pf_utility", "PF utility", "PF utility vs blocked UEs", False),
    ("G07_pf_vs_relays", "num_relays", "pf_utility", "PF utility", "PF utility vs number of relays", False),
    ("G08_pf_vs_tau", "tau", "pf_utility", "PF utility", "PF utility vs phase split", False),
    ("G09_pf_vs_aging", "aging", "pf_utility", "PF utility", "PF utility vs channel aging", False),
    ("G12_runtime_vs_ues", "num_ues", "seconds", "Seconds per drop (log scale)", "Runtime vs number of UEs", True),
]


def draw_all(results_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    drawn, skipped = [], []
    for stem, function in (("G01_rate_cdf", figure_rate_cdf), ("G02_outage", figure_outage), ("G10_pf_vs_layers", figure_layers),
                           ("G11_optimality_gap", figure_optimality_gap), ("G13/G14_fl", figure_fl)):
        (drawn if function(results_dir, out_dir) else skipped).append(stem)
    for stem, experiment, statistic, ylabel, title, log_y in SWEEP_FIGURES:
        payload = _load(results_dir, experiment)
        if payload:
            _plot_sweep(payload, statistic, ylabel, title, out_dir, stem, log_y)
            drawn.append(stem)
        else:
            skipped.append(stem)
    return drawn, skipped
