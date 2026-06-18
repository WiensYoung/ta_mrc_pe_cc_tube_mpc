"""Publication-quality visualization suite.

Generates:
- Time-series comparison plots (CPA evolution, control inputs)
- Sensitivity tornado plots
- Radar/spider charts for multi-metric comparison
- Trajectory snapshot panels
- Failure heatmaps by scenario × method
- Violin plots with statistical significance annotations
- CDF comparison plots
- Forest plots for effect sizes
- Safety distance breakdown stacked area charts
- COLREGS compliance by encounter type
- Parameter sensitivity heatmaps
- Kinematic phase portraits
- Encounter geometry diagrams
- Multi-panel composite figures
- LaTeX-formatted output with proper font rendering
"""

import os
import warnings
from contextlib import contextmanager
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Plot style configuration
# ---------------------------------------------------------------------------

# Try to use LaTeX for publication-quality rendering (Q1 fix — 2026-06-10)
USING_LATEX = False
try:
    import shutil
    if shutil.which("latex") and shutil.which("dvipng"):
        matplotlib.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
        })
        USING_LATEX = True
    else:
        # Fallback: use matplotlib's built-in math renderer with STIX fonts
        matplotlib.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
        })
except Exception:
    pass
finally:
    # Always apply these (non-LaTeX-dependent)
    matplotlib.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.format": "pdf",
    })

# ── Journal figure size templates (mm converted to inches) ─────────────────
# Standard journal column widths for LaTeX \includegraphics[width=\columnwidth]
JOURNAL_SIZES = {
    "single_col": (3.35, 2.5),    # ~85 mm  × ~63 mm
    "double_col": (7.0, 4.5),     # ~178 mm × ~114 mm
    "full_page": (7.0, 9.0),      # ~178 mm × ~229 mm
    "half_page": (7.0, 4.0),      # ~178 mm × ~102 mm
}

# ── Color palette for methods (Paul Tol high-contrast, colorblind-friendly) ─
METHOD_PALETTE = {
    "Proposed": "#0072B2",  # blue
    "B7":       "#009E73",  # bluish green
    "B6":       "#F0E442",  # yellow
    "B5":       "#E69F00",  # orange
    "B4":       "#D55E00",  # vermillion
    "B3":       "#CC79A7",  # reddish purple
    "B2":       "#56B4E9",  # sky blue
    "B1":       "#999999",  # grey
}

METHOD_MARKERS = {
    "Proposed": "o",
    "B7": "s",
    "B6": "^",
    "B5": "D",
    "B4": "v",
    "B3": "p",
    "B2": "*",
    "B1": "x",
}

METHOD_LINESTYLES = {
    "Proposed": "-",
    "B7": "--",
    "B6": "-.",
    "B5": ":",
    "B4": "-",
    "B3": "--",
    "B2": "-.",
    "B1": ":",
}


def _get_style_for_method(method: str) -> dict:
    """Return unified {color, marker, linestyle} dict for a method."""
    color = "#333333"
    marker = "."
    linestyle = "-"
    for key in METHOD_PALETTE:
        if key in method:
            color = METHOD_PALETTE[key]
            marker = METHOD_MARKERS.get(key, ".")
            linestyle = METHOD_LINESTYLES.get(key, "-")
            break
    return {"color": color, "marker": marker, "linestyle": linestyle}


def _get_color(method: str) -> str:
    return _get_style_for_method(method)["color"]


def _get_marker(method: str) -> str:
    return _get_style_for_method(method)["marker"]


# ---------------------------------------------------------------------------
# Context manager for journal-specific styling
# ---------------------------------------------------------------------------

@contextmanager
def journal_style(journal: str = "ocean_engineering", usetex: bool = False):
    """Context manager for journal-specific matplotlib styling.

    Args:
        journal: One of 'ocean_engineering', 'ieee', 'springer', 'generic'.
        usetex: Enable LaTeX rendering (requires LaTeX installation).

    Yields:
        None — modifies rcParams within the context block.
    """
    old_params = {k: matplotlib.rcParams.get(k) for k in [
        "font.family", "font.size", "axes.titlesize", "axes.labelsize",
        "legend.fontsize", "xtick.labelsize", "ytick.labelsize",
        "text.usetex", "figure.dpi", "savefig.dpi",
    ]}

    journals = {
        "ocean_engineering": {"font.family": "serif", "font.size": 9,
                              "axes.titlesize": 10, "axes.labelsize": 9,
                              "legend.fontsize": 8},
        "ieee": {"font.family": "serif", "font.size": 8,
                 "axes.titlesize": 9, "axes.labelsize": 8,
                 "legend.fontsize": 7},
        "springer": {"font.family": "sans-serif", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8},
        "generic": {"font.family": "serif", "font.size": 10,
                    "axes.titlesize": 11, "axes.labelsize": 10,
                    "legend.fontsize": 8},
    }
    style = journals.get(journal, journals["generic"])
    style["text.usetex"] = usetex
    matplotlib.rcParams.update(style)
    try:
        yield
    finally:
        for k, v in old_params.items():
            if v is not None:
                matplotlib.rcParams[k] = v


def _journal_figsize(template: str) -> tuple[float, float]:
    """Return (width, height) in inches for a journal figure size template."""
    return JOURNAL_SIZES.get(template, JOURNAL_SIZES["double_col"])


# ---------------------------------------------------------------------------
# Statistical annotation utilities
# ---------------------------------------------------------------------------

def _significance_label(p_value: float) -> str:
    """Return significance star label from p-value.

    ***  p < 0.001
    **   p < 0.01
    *    p < 0.05
    ns   p >= 0.05
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return "ns"


def add_significance_brackets(
    ax,
    pairs: list[tuple[int, int]],
    pvalues: list[float],
    y_data: list[list[float]],
    x_positions: list[float],
    height_offset: float = 0.03,
    bracket_linewidth: float = 0.8,
    fontsize: int = 8,
):
    """Add significance brackets and star labels between groups.

    Args:
        ax: Matplotlib Axes to annotate.
        pairs: List of (idx_a, idx_b) group index pairs to compare.
        pvalues: List of p-values corresponding to each pair.
        y_data: List of data arrays (one per group) for computing bracket height.
        x_positions: X-axis positions of each group.
        height_offset: Fraction of data range to add as vertical spacing.
        bracket_linewidth: Line width for bracket bars.
        fontsize: Font size for significance labels.
    """
    y_maxs = []
    for data in y_data:
        if data is not None and len(data) > 0:
            y_maxs.append(np.max(data))
        else:
            y_maxs.append(0)
    if not y_maxs:
        return
    y_range = max(y_maxs) - min(y_maxs)
    base_y = max(y_maxs) + y_range * height_offset

    for pair_idx, ((i, j), pval) in enumerate(zip(pairs, pvalues)):
        y = base_y + pair_idx * y_range * height_offset * 3
        label = _significance_label(pval)

        # Bracket bar
        ax.plot([x_positions[i], x_positions[i], x_positions[j], x_positions[j]],
                [y, y + y_range * height_offset, y + y_range * height_offset, y],
                lw=bracket_linewidth, color="black", clip_on=False)
        # Label
        ax.text(
            (x_positions[i] + x_positions[j]) / 2,
            y + y_range * height_offset * 1.5,
            label, ha="center", va="bottom", fontsize=fontsize,
        )

    # Extend y-axis if needed
    ax.set_ylim(top=base_y + len(pairs) * y_range * height_offset * 3 + y_range * 0.1)


# ---------------------------------------------------------------------------
# Time-series CPA evolution
# ---------------------------------------------------------------------------

def plot_cpa_evolution(
    results_by_method: dict[str, list],  # method -> list[EpisodeResult]
    save_path: Optional[str] = None,
    title: str = "CPA Evolution Comparison",
    max_time: float = 600.0,
    n_bins: int = 60,
    figsize: tuple = None,
):
    """Plot CPA evolution over time with confidence bands.

    Args:
        results_by_method: Dict mapping method name to list of EpisodeResult.
        save_path: Output file path.
        title: Plot title.
        max_time: Maximum simulation time [s].
        n_bins: Number of time bins for aggregation.
        figsize: Optional (w, h) override in inches.
    """
    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("double_col"))

    time_bins = np.linspace(0, max_time, n_bins + 1)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

    for method, results in results_by_method.items():
        all_dcpa = []
        for r in results:
            if not r.target_histories:
                continue
            timesteps = min(len(r.state_history), len(r.target_histories[0]) if r.target_histories else 0)
            dcpa_series = []
            for k in range(timesteps):
                if k < len(r.state_history) and r.target_histories:
                    own = r.state_history[k]
                    tgt_hist = r.target_histories[0]
                    if k < len(tgt_hist):
                        tgt_s = tgt_hist[k]
                        d = np.sqrt((own.x - tgt_s.x)**2 + (own.y - tgt_s.y)**2)
                        dcpa_series.append((k * 0.5, d))  # dt=0.5
            if dcpa_series:
                all_dcpa.append(dcpa_series)

        if not all_dcpa:
            continue

        # Bin by time
        means = np.zeros(n_bins)
        lowers = np.zeros(n_bins)
        uppers = np.zeros(n_bins)

        for i, (t_start, t_end) in enumerate(zip(time_bins[:-1], time_bins[1:])):
            bin_vals = []
            for series in all_dcpa:
                for t, d in series:
                    if t_start <= t < t_end:
                        bin_vals.append(d)
            if bin_vals:
                arr = np.array(bin_vals)
                means[i] = np.mean(arr)
                lowers[i] = np.percentile(arr, 5)
                uppers[i] = np.percentile(arr, 95)

        style = _get_style_for_method(method)
        ax.plot(bin_centers, means, color=style["color"], linewidth=1.5, label=method)
        ax.fill_between(bin_centers, lowers, uppers, color=style["color"], alpha=0.12)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance to Nearest Target [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_time)

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Control input time-series
# ---------------------------------------------------------------------------

def plot_control_inputs(
    results_by_method: dict[str, list],
    save_path: Optional[str] = None,
    max_time: float = 600.0,
    figsize: tuple = None,
):
    """Plot control inputs (rudder, propeller) over time.

    Args:
        results_by_method: Dict mapping method name to list of EpisodeResult.
        save_path: Output file path.
        max_time: Maximum time [s].
        figsize: Optional (w, h) override in inches.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize or _journal_figsize("double_col"), sharex=True)

    for method, results in results_by_method.items():
        for r_idx, r in enumerate(results):
            if r_idx > 2:  # plot up to 3 episodes per method
                break
            times = [i * 0.5 for i in range(len(r.command_history))]
            rudders = [c.rudder for c in r.command_history]
            props = [c.propeller for c in r.command_history]

            style = _get_style_for_method(method)
            alpha = 0.7 if r_idx == 0 else 0.3

            ax1.plot(times[:len(rudders)], rudders, color=style["color"], alpha=alpha,
                    linewidth=0.8, label=method if r_idx == 0 else "")
            ax2.plot(times[:len(props)], props, color=style["color"], alpha=alpha,
                    linewidth=0.8, label=method if r_idx == 0 else "")

    ax1.axhline(y=35, color="red", linestyle="--", alpha=0.3, linewidth=0.5)
    ax1.axhline(y=-35, color="red", linestyle="--", alpha=0.3, linewidth=0.5)
    ax1.set_ylabel("Rudder [deg]")
    ax1.legend(loc="upper right", ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-40, 40)

    ax2.set_ylabel("Propeller [-]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)

    fig.suptitle("Control Input Evolution")
    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sensitivity tornado plot
# ---------------------------------------------------------------------------

def plot_sensitivity_tornado(
    sensitivity_data: dict[str, tuple[float, float, float]],  # label -> (low, base, high)
    metric_name: str = "min_dcpa",
    save_path: Optional[str] = None,
    figsize: tuple = None,
):
    """Tornado plot showing sensitivity of a metric to parameter variation.

    Args:
        sensitivity_data: Dict mapping parameter label to (low_value, base_value, high_value).
        metric_name: Name of the metric being analyzed.
        save_path: Output file path.
        figsize: Optional (w, h) override in inches.
    """
    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("single_col"))

    # Sort by impact (range)
    items = sorted(sensitivity_data.items(),
                   key=lambda x: abs(x[1][2] - x[1][0]),
                   reverse=True)

    labels = [item[0] for item in items]
    y_pos = range(len(labels))

    base_val = items[0][1][1] if items else 0

    for i, (label, (low, base, high)) in enumerate(items):
        ax.barh(i, high - base, left=base, height=0.5,
                color="#d7191c", alpha=0.7, label="High" if i == 0 else "")
        ax.barh(i, low - base, left=base, height=0.5,
                color="#2c7bb6", alpha=0.7, label="Low" if i == 0 else "")

    ax.axvline(x=base_val, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric_name)
    ax.set_title(f"Sensitivity Tornado: {metric_name}")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Radar / spider chart
# ---------------------------------------------------------------------------

def plot_radar_chart(
    metrics_by_method: dict[str, dict[str, float]],  # method -> {metric: value}
    save_path: Optional[str] = None,
    title: str = "Multi-Metric Comparison",
    figsize: tuple = None,
):
    """Radar chart comparing multiple methods across normalized metrics.

    Args:
        metrics_by_method: Dict mapping method to dict of metric->value.
        save_path: Output file path.
        title: Plot title.
        figsize: Optional (w, h) override in inches.
    """
    methods = list(metrics_by_method.keys())
    if not methods:
        return  # no data to plot
    metric_names = list(next(iter(metrics_by_method.values())).keys())
    n_metrics = len(metric_names)

    # Normalize to [0, 1] where 1 = best
    normalized = {}
    for metric in metric_names:
        vals = [metrics_by_method[m].get(metric, 0) for m in methods]
        min_v, max_v = min(vals), max(vals)
        range_v = max_v - min_v if max_v != min_v else 1.0
        _LOWER_BETTER_SUFFIXES = (
            "_rate", "_violation", "collision", "runtime", "_count",
            "_miss", "failure", "infeasible",
        )
        lower_better = any(metric.lower().endswith(kw)
                          for kw in _LOWER_BETTER_SUFFIXES)
        for m in methods:
            if m not in normalized:
                normalized[m] = []
            v = metrics_by_method[m].get(metric, 0)
            if lower_better:
                normalized[m].append(1.0 - (v - min_v) / range_v)
            else:
                normalized[m].append((v - min_v) / range_v)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("single_col"),
                           subplot_kw=dict(polar=True))

    for method in methods:
        values = normalized[method] + normalized[method][:1]
        style = _get_style_for_method(method)
        ax.fill(angles, values, color=style["color"], alpha=0.1)
        ax.plot(angles, values, color=style["color"], linewidth=1.5, label=method,
                marker=style["marker"], markersize=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=7)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Failure heatmap
# ---------------------------------------------------------------------------

def plot_failure_heatmap_from_dict(
    failure_rates: dict[str, dict[str, float]],  # scenario -> {method: failure_rate}
    save_path: Optional[str] = None,
    title: str = "Failure Rate by Scenario and Method",
    figsize: tuple = None,
):
    """Heatmap of failure rates from a nested dict (legacy API).

    Args:
        failure_rates: Nested dict scenario_id -> method -> failure_rate.
        save_path: Output file path.
        title: Plot title.
        figsize: Optional (w, h) override in inches.
    """
    scenarios = sorted(failure_rates.keys())
    if not scenarios:
        return  # no data to plot
    methods = sorted(set(m for s in scenarios for m in failure_rates[s]))

    data = np.zeros((len(scenarios), len(methods)))
    for i, sc in enumerate(scenarios):
        for j, m in enumerate(methods):
            data[i, j] = failure_rates[sc].get(m, 0.0)

    fig, ax = plt.subplots(figsize=figsize or (max(6, len(methods) * 1.2),
                                                max(4, len(scenarios) * 0.6)))

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(0.5, data.max()))

    for i in range(len(scenarios)):
        for j in range(len(methods)):
            text_color = "white" if data[i, j] > 0.25 else "black"
            ax.text(j, i, f"{data[i, j]:.1%}", ha="center", va="center",
                   color=text_color, fontsize=8)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Failure Rate")

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trajectory snapshot panel
# ---------------------------------------------------------------------------

def plot_trajectory_snapshots(
    results_by_method: dict[str, list],
    timestamps: list[float] = None,
    save_path: Optional[str] = None,
    scenario_name: str = "",
    figsize: tuple = None,
):
    """Multi-panel trajectory snapshot at key time points.

    Args:
        results_by_method: Dict method -> list of EpisodeResult.
        timestamps: List of times at which to show snapshots.
        save_path: Output file path.
        scenario_name: Scenario name for the title.
        figsize: Optional (w, h) override in inches.
    """
    if timestamps is None:
        timestamps = [60, 180, 360, 540]

    n_cols = len(timestamps)
    n_rows = len(results_by_method)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (3 * n_cols, 2.5 * n_rows),
                             squeeze=False)

    methods = list(results_by_method.keys())

    for row, method in enumerate(methods):
        result = results_by_method[method][0] if results_by_method[method] else None
        if result is None:
            continue

        for col, t_snap in enumerate(timestamps):
            ax = axes[row, col]
            step = int(t_snap / 0.5)

            # Plot ownship trajectory up to this point
            if step < len(result.state_history):
                own_x = [s.x for s in result.state_history[:step + 1]]
                own_y = [s.y for s in result.state_history[:step + 1]]
                style = _get_style_for_method(method)
                ax.plot(own_x, own_y, color=style["color"], linewidth=1.2, alpha=0.6)
                ax.plot(own_x[-1], own_y[-1], marker="o", color=style["color"], markersize=6)

            # Plot target trajectories
            for i, tgt_hist in enumerate(result.target_histories):
                if step < len(tgt_hist):
                    tgt_x = [s.x for s in tgt_hist[:step + 1]]
                    tgt_y = [s.y for s in tgt_hist[:step + 1]]
                    ax.plot(tgt_x, tgt_y, "gray", linewidth=0.8, linestyle="--", alpha=0.5)
                    ax.plot(tgt_x[-1], tgt_y[-1], marker="s", color="gray", markersize=4)

            if row == 0:
                ax.set_title(f"t = {t_snap:.0f}s", fontsize=9)
            if col == 0:
                ax.set_ylabel(method, fontsize=9)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=6)

    fig.suptitle(f"Trajectory Snapshots: {scenario_name}", fontsize=11)
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Box plot with swarm overlay
# ---------------------------------------------------------------------------

def plot_metric_boxplot_swarm(
    metrics_dict: dict[str, list[float]],  # method -> list of values
    metric_name: str = "",
    save_path: Optional[str] = None,
    figsize: tuple = None,
):
    """Box plot with individual data points overlaid.

    Args:
        metrics_dict: Dict method -> list of metric values.
        metric_name: Name of the metric.
        save_path: Output file path.
        figsize: Optional (w, h) override in inches.
    """
    methods = [m for m in metrics_dict if metrics_dict[m]]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=figsize or (max(5, len(methods) * 1.5), 5))

    positions = range(1, len(methods) + 1)

    for i, method in enumerate(methods):
        values = metrics_dict[method]
        style = _get_style_for_method(method)

        bp = ax.boxplot([values], positions=[positions[i]], widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "linewidth": 1})

        if bp.get("boxes") and len(bp["boxes"]) > 0:
            bp["boxes"][0].set_facecolor(style["color"])
            bp["boxes"][0].set_alpha(0.3)

        # Jittered swarm overlay
        if len(values) <= 200:
            jitter = np.random.normal(0, 0.06, len(values))
            ax.scatter(positions[i] + jitter, values, color=style["color"], alpha=0.4,
                      s=8, edgecolors="none")

    ax.set_xticks(positions)
    ax.set_xticklabels(methods)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Distribution")
    ax.grid(True, axis="y", alpha=0.3)

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# NEW — SCI Q1 enhanced plot types
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Violin plot with statistical significance annotations
# ---------------------------------------------------------------------------

def plot_violin_with_significance(
    metrics_dict: dict[str, list[float]],
    metric_name: str = "",
    pvalues: dict[tuple[str, str], float] = None,
    reference_method: str = "Proposed",
    save_path: Optional[str] = None,
    title: str = "",
    figsize: tuple = None,
):
    """Violin plot with swarm overlay and significance brackets.

    Args:
        metrics_dict: Dict method -> list of values.
        metric_name: Name of the metric for y-axis label.
        pvalues: Dict mapping (method_a, method_b) -> p-value.
        reference_method: Method to use as reference for comparisons.
        save_path: Output file path.
        title: Plot title (defaults to metric_name).
        figsize: Optional (w, h) override in inches.
    """
    methods = [m for m in metrics_dict if metrics_dict[m]]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("double_col"))

    positions = range(len(methods))
    data_list = []
    for method in methods:
        data_list.append(np.array(metrics_dict[method]))

    # Violin plots
    violin_parts = ax.violinplot(data_list, positions=positions,
                                  showmeans=False, showmedians=True,
                                  widths=0.7)

    for i, method in enumerate(methods):
        style = _get_style_for_method(method)
        for pc in violin_parts["bodies"]:
            if i < len(violin_parts["bodies"]):
                pass  # color all bodies
        if "bodies" in violin_parts:
            for idx, body in enumerate(violin_parts["bodies"]):
                body.set_facecolor(_get_style_for_method(methods[idx])["color"])
                body.set_alpha(0.4)
                body.set_edgecolor("black")
                body.set_linewidth(0.5)

    # Swarm overlay (capped at 200 points)
    for i, method in enumerate(methods):
        values = metrics_dict[method]
        if len(values) <= 200:
            jitter = np.random.normal(positions[i], 0.06, len(values))
            style = _get_style_for_method(method)
            ax.scatter(jitter, values, color=style["color"], alpha=0.5,
                      s=6, edgecolors="none", zorder=3)

    # Significance brackets
    if pvalues and reference_method in methods:
        ref_idx = methods.index(reference_method)
        pairs = []
        pvals = []
        for i, method in enumerate(methods):
            if method == reference_method:
                continue
            key1 = (reference_method, method)
            key2 = (method, reference_method)
            pv = pvalues.get(key1, pvalues.get(key2))
            if pv is not None:
                pairs.append((ref_idx, i))
                pvals.append(pv)
        if pairs:
            add_significance_brackets(ax, pairs, pvals, data_list,
                                      list(positions))

    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"{metric_name} Distribution with Significance")
    ax.grid(True, axis="y", alpha=0.3)

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CDF comparison plot
# ---------------------------------------------------------------------------

def plot_cdf_comparison(
    metrics_dict: dict[str, list[float]],
    metric_name: str = "",
    save_path: Optional[str] = None,
    title: str = "",
    log_x: bool = False,
    figsize: tuple = None,
):
    """Cumulative Distribution Function comparison across methods.

    Args:
        metrics_dict: Dict method -> list of values.
        metric_name: Name of the metric.
        save_path: Output file path.
        title: Plot title.
        log_x: Use logarithmic x-axis (useful for runtime metrics).
        figsize: Optional (w, h) override in inches.
    """
    methods = [m for m in metrics_dict if metrics_dict[m]]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("single_col"))

    for method in methods:
        values = sorted(metrics_dict[method])
        if not values:
            continue
        n = len(values)
        cdf = np.arange(1, n + 1) / n
        style = _get_style_for_method(method)

        # Plot every Nth point for marker clarity
        marker_every = max(1, n // 20)
        ax.plot(values, cdf, color=style["color"], linewidth=1.5, label=method,
               marker=style["marker"], markevery=marker_every, markersize=4)

    ax.set_xlabel(metric_name)
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(title or f"CDF: {metric_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(0, 1.05)

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Forest plot for effect sizes
# ---------------------------------------------------------------------------

def plot_forest_effect_sizes(
    effect_sizes: dict[str, dict],  # method -> {cohens_d, ci_lower, ci_upper, p_value}
    reference_method: str = "Proposed",
    save_path: Optional[str] = None,
    title: str = "Effect Sizes vs. Proposed",
    figsize: tuple = None,
):
    """Forest plot of Cohen's d with 95% confidence intervals.

    Args:
        effect_sizes: Dict mapping method name to dict with keys:
            cohens_d, ci_lower, ci_upper, p_value (optional).
        reference_method: Method used as reference (for label).
        save_path: Output file path.
        title: Plot title.
        figsize: Optional (w, h) override in inches.
    """
    methods = list(effect_sizes.keys())
    if not methods:
        return
    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("single_col"))

    y_positions = range(len(methods))
    ds = []
    cis_lower = []
    cis_upper = []
    pvals = []

    for method in methods:
        d = effect_sizes[method].get("cohens_d", 0.0)
        ci_l = effect_sizes[method].get("ci_lower", d - 0.2)
        ci_u = effect_sizes[method].get("ci_upper", d + 0.2)
        pv = effect_sizes[method].get("p_value", 1.0)
        ds.append(d)
        cis_lower.append(ci_l)
        cis_upper.append(ci_u)
        pvals.append(pv)

    y_pos = list(y_positions)

    for i, (method, d, ci_l, ci_u, pv) in enumerate(zip(methods, ds, cis_lower, cis_upper, pvals)):
        style = _get_style_for_method(method)
        is_sig = pv < 0.05
        marker = "s" if is_sig else "o"
        size = 8 if is_sig else 6
        ax.errorbar(d, y_pos[i], xerr=[[d - ci_l], [ci_u - d]],
                   fmt=marker, color=style["color"], capsize=3,
                   markersize=size, linewidth=1.2,
                   label=f"{method} ({_significance_label(pv)})")

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel(f"Cohen's d (vs. {reference_method})")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, axis="x", alpha=0.3)

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MPC convergence histogram
# ---------------------------------------------------------------------------

def plot_mpc_convergence(
    results_by_method: dict[str, list],  # method -> list[EpisodeResult]
    save_path: Optional[str] = None,
    title: str = "MPC Solver Convergence",
    n_bins: int = 20,
    figsize: tuple = None,
):
    """Histogram of IPOPT iterations or solve success rate per method.

    Args:
        results_by_method: Dict mapping method to list of EpisodeResult.
        save_path: Output file path.
        title: Plot title.
        n_bins: Number of histogram bins.
        figsize: Optional (w, h) override in inches.
    """
    methods = [m for m in results_by_method if results_by_method[m]]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("double_col"))

    all_iters = []
    labels = []

    for method in methods:
        iters = []
        for r in results_by_method[method]:
            mpc_info = r.metadata.get("mpc_info", {})
            n_iter = mpc_info.get("n_iterations", 0)
            if n_iter > 0:
                iters.append(n_iter)
        if iters:
            all_iters.append(iters)
            labels.append(method)

    if not all_iters:
        plt.close(fig)
        return

    colors = [_get_color(m) for m in labels]
    ax.hist(all_iters, bins=n_bins, label=labels, color=colors,
            alpha=0.6, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("IPOPT Iterations")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Safety distance breakdown stacked area chart
# ---------------------------------------------------------------------------

def plot_safety_breakdown_stacked(
    safety_breakdowns: list[dict],
    timestamps: list[float],
    save_path: Optional[str] = None,
    title: str = "Safety Distance Breakdown Over Time",
    figsize: tuple = None,
):
    """Stacked area chart of safety distance components over time.

    Each dict in *safety_breakdowns* is expected to have keys:
        base_domain, dynamic_margin, chance_margin, tube_margin, rule_margin, extra_margin

    Args:
        safety_breakdowns: List of per-timestep dicts with safety distance components.
        timestamps: List of timestamps [s] matching breakdowns.
        save_path: Output file path.
        title: Plot title.
        figsize: Optional (w, h) override in inches.
    """
    if not safety_breakdowns or not timestamps:
        return
    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("double_col"))

    n = min(len(safety_breakdowns), len(timestamps))
    t_arr = np.array(timestamps[:n])

    keys = ["base_domain", "dynamic_margin", "chance_margin",
            "tube_margin", "rule_margin", "extra_margin"]
    colors_stacked = ["#1b9e77", "#d95f02", "#7570b3",
                       "#e7298a", "#66a61e", "#e6ab02"]

    y_offsets = np.zeros(n)
    for i, key in enumerate(keys):
        vals = np.array([bd.get(key, 0.0) for bd in safety_breakdowns[:n]])
        ax.fill_between(t_arr, y_offsets, y_offsets + vals,
                        label=key.replace("_", " ").title(),
                        color=colors_stacked[i % len(colors_stacked)],
                        alpha=0.7, linewidth=0.3)
        y_offsets = y_offsets + vals

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Safety Distance [m]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# COLREGS compliance by encounter type
# ---------------------------------------------------------------------------

def plot_colregs_compliance_by_encounter(
    compliance_data: dict[str, dict[str, float]],  # method -> {encounter_type: compliance_rate}
    save_path: Optional[str] = None,
    title: str = "COLREGS Compliance by Encounter Type",
    figsize: tuple = None,
):
    """Grouped bar chart of COLREGS compliance rate per method and encounter type.

    Args:
        compliance_data: Nested dict method -> encounter_type -> compliance_rate.
        save_path: Output file path.
        title: Plot title.
        figsize: Optional (w, h) override in inches.
    """
    methods = sorted(compliance_data.keys())
    if not methods:
        return

    # Collect all encounter types
    encounter_types = sorted(set(
        et for m in methods for et in compliance_data[m]
    ))
    if not encounter_types:
        return

    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("double_col"))

    n_groups = len(encounter_types)
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    x = np.arange(n_groups)

    for i, method in enumerate(methods):
        values = [compliance_data[method].get(et, 0.0) for et in encounter_types]
        style = _get_style_for_method(method)
        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=method,
               color=style["color"], alpha=0.8, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([et.replace("_", " ").title() for et in encounter_types],
                       rotation=30, ha="right")
    ax.set_ylabel("Compliance Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parameter sensitivity heatmap
# ---------------------------------------------------------------------------

def plot_parameter_sensitivity_heatmap(
    sensitivity_matrix: np.ndarray,
    param_names: list[str],
    metric_names: list[str],
    save_path: Optional[str] = None,
    title: str = "Parameter Sensitivity Matrix",
    figsize: tuple = None,
    cmap: str = "RdBu_r",
):
    """Heatmap of parameter × metric sensitivity scores.

    Args:
        sensitivity_matrix: 2D array of shape (n_params, n_metrics).
        param_names: List of parameter names (y-axis labels).
        metric_names: List of metric names (x-axis labels).
        save_path: Output file path.
        title: Plot title.
        figsize: Optional (w, h) override in inches.
        cmap: Colormap name (diverging: RdBu_r, coolwarm; sequential: YlOrRd).
    """
    if sensitivity_matrix.size == 0:
        return
    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("double_col"))

    im = ax.imshow(sensitivity_matrix, cmap=cmap, aspect="auto",
                   vmin=-1.0, vmax=1.0)

    for i in range(len(param_names)):
        for j in range(len(metric_names)):
            val = sensitivity_matrix[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   color=text_color, fontsize=7)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names, fontsize=8)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Sensitivity Score")

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Kinematic phase portrait (u-v phase plane)
# ---------------------------------------------------------------------------

def plot_kinematic_phase_portrait(
    state_histories: dict[str, list],  # method -> list[VesselState]
    save_path: Optional[str] = None,
    title: str = "Kinematic Phase Portrait (u-v)",
    figsize: tuple = None,
):
    """Phase portrait of surge (u) vs sway (v) velocity for each method.

    Args:
        state_histories: Dict mapping method name to list of VesselState.
        save_path: Output file path.
        title: Plot title.
        figsize: Optional (w, h) override in inches.
    """
    methods = [m for m in state_histories if state_histories[m]]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("single_col"))

    for method in methods:
        states = state_histories[method]
        u_vals = [s.u for s in states]
        v_vals = [s.v for s in states]
        style = _get_style_for_method(method)

        ax.plot(u_vals, v_vals, color=style["color"], linewidth=1.0, alpha=0.7, label=method)
        # Mark start and end
        if u_vals:
            ax.plot(u_vals[0], v_vals[0], marker="o", color=style["color"], markersize=6)
            ax.plot(u_vals[-1], v_vals[-1], marker="s", color=style["color"], markersize=6)

    ax.set_xlabel("Surge Velocity u [m/s]")
    ax.set_ylabel("Sway Velocity v [m/s]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Encounter geometry diagram
# ---------------------------------------------------------------------------

def plot_encounter_geometry(
    own_state,    # VesselState
    targets: list,   # list[TargetShip]
    domain_results: list[dict] = None,
    save_path: Optional[str] = None,
    title: str = "Encounter Geometry",
    view_radius: float = 1000.0,
    figsize: tuple = None,
):
    """Top-down encounter geometry: ship hulls, safety domains, COLREGS sectors.

    Args:
        own_state: Ownship VesselState at the snapshot moment.
        targets: List of TargetShip objects.
        domain_results: Optional list of per-target domain dicts (with 'total' key).
        save_path: Output file path.
        title: Plot title.
        view_radius: View radius around ownship [m].
        figsize: Optional (w, h) override in inches.
    """
    fig, ax = plt.subplots(figsize=figsize or _journal_figsize("single_col"))

    own_x, own_y, own_psi = own_state.x, own_state.y, own_state.psi
    # Vessel dimensions — try to read from episode context, fall back to defaults
    own_L = getattr(own_state, "length_oa", None) or 180.0
    own_B = getattr(own_state, "beam", None) or 30.0

    # Draw ownship as oriented rectangle
    _draw_oriented_rect(ax, own_x, own_y, own_psi, own_L, own_B,
                        facecolor="#0072B2", edgecolor="black", alpha=0.6, label="Ownship")

    # Draw heading vector
    head_len = own_L * 0.8
    ax.arrow(own_x, own_y,
             head_len * np.cos(own_psi), head_len * np.sin(own_psi),
             head_width=15, head_length=30, fc="#0072B2", ec="#0072B2", alpha=0.5)

    # Draw COLREGS sectors
    _draw_colregs_sectors(ax, own_x, own_y, own_psi, view_radius)

    # Draw target ships
    for i, tgt in enumerate(targets):
        tgt_s = tgt.state
        tgt_L = getattr(tgt, "length", 100.0)
        tgt_B = getattr(tgt, "beam", 15.0)
        _draw_oriented_rect(ax, tgt_s.x, tgt_s.y, tgt_s.psi, tgt_L, tgt_B,
                           facecolor="#E41A1C", edgecolor="black", alpha=0.4,
                           label=f"Target {i+1}" if i == 0 else "")

        # Draw safety domain circle
        if domain_results and i < len(domain_results):
            d_safe = domain_results[i].get("total", 500.0) if isinstance(domain_results[i], dict) else 500.0
        else:
            d_safe = 500.0
        circle = plt.Circle((tgt_s.x, tgt_s.y), d_safe, fill=False,
                           linestyle="--", color="red", alpha=0.3, linewidth=0.8)
        ax.add_patch(circle)

        # Distance label
        d = np.sqrt((own_x - tgt_s.x)**2 + (own_y - tgt_s.y)**2)
        mid_x = (own_x + tgt_s.x) / 2
        mid_y = (own_y + tgt_s.y) / 2
        ax.annotate(f"{d:.0f}m", (mid_x, mid_y), fontsize=7, ha="center",
                   color="darkred", alpha=0.7)

    ax.set_xlim(own_x - view_radius, own_x + view_radius)
    ax.set_ylim(own_y - view_radius, own_y + view_radius)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


def _draw_oriented_rect(ax, cx, cy, psi, length, beam, facecolor, edgecolor, alpha, label=""):
    """Draw an oriented rectangle (ship hull) on the given axes."""
    from matplotlib.patches import Polygon
    c = np.cos(psi)
    s = np.sin(psi)
    half_l = length / 2
    half_b = beam / 2
    corners = np.array([
        [ half_l, -half_b],
        [ half_l,  half_b],
        [-half_l,  half_b],
        [-half_l, -half_b],
    ])
    R = np.array([[c, -s], [s, c]])
    world_corners = corners @ R.T + np.array([cx, cy])
    poly = Polygon(world_corners, facecolor=facecolor, edgecolor=edgecolor,
                   alpha=alpha, linewidth=0.8, label=label)
    ax.add_patch(poly)


def _draw_colregs_sectors(ax, cx, cy, psi, radius):
    """Draw COLREGS encounter sectors as annotated wedges."""
    from matplotlib.patches import Wedge

    sectors = [
        ("Head-on\n(Rule 14)", 355, 5, "#FF6B6B", 0.08),
        ("Crossing\nGive-way\n(Rule 15)", 5, 112.5, "#FFD93D", 0.06),
        ("Overtaking\n(Rule 13)", 112.5, 247.5, "#6BCB77", 0.06),
        ("Crossing\nStand-on\n(Rule 15)", 247.5, 355, "#4D96FF", 0.06),
    ]

    for label, start_deg, end_deg, color, alpha in sectors:
        # Convert from North-up bearing to math angle (CCW from East)
        theta1 = 90 - end_deg
        theta2 = 90 - start_deg
        # Handle wrap-around
        if start_deg > end_deg:
            wedge = Wedge((cx, cy), radius, theta1, theta2,
                         facecolor=color, alpha=alpha, edgecolor="none")
        else:
            wedge = Wedge((cx, cy), radius, theta1, theta2,
                         facecolor=color, alpha=alpha, edgecolor="none")
        ax.add_patch(wedge)

        # Label at mid-angle
        mid_deg = (start_deg + end_deg) / 2
        if start_deg > end_deg:
            mid_deg = (start_deg + end_deg + 360) / 2 % 360
        label_r = radius * 0.6
        label_theta = np.radians(90 - mid_deg)
        ax.text(cx + label_r * np.cos(label_theta),
                cy + label_r * np.sin(label_theta),
                label, fontsize=5, ha="center", va="center", alpha=0.7)


# ---------------------------------------------------------------------------
# Multi-panel composite — Figure 1 (overview)
# ---------------------------------------------------------------------------

def plot_figure1_overview(
    results_by_method: dict[str, list],
    scenario_name: str = "",
    save_path: Optional[str] = None,
    timestamps: list[float] = None,
    max_time: float = 600.0,
    figsize: tuple = None,
):
    """4-panel composite figure: Trajectory + CPA + Control + Safety Margin.

    Designed as the first figure in a journal paper, providing a qualitative
    overview of a representative scenario across all methods.

    Layout (2×2):
        [Trajectory Snapshots]  [CPA Evolution]
        [Control Inputs]        [Safety Margin Over Time]

    Args:
        results_by_method: Dict method -> list of EpisodeResult.
        scenario_name: Name of the scenario for the title.
        save_path: Output file path.
        timestamps: List of snapshot timestamps for trajectory panel.
        max_time: Maximum time [s] for time-series panels.
        figsize: Optional (w, h) override in inches.
    """
    if timestamps is None:
        timestamps = [60, 300, 540]
    methods = list(results_by_method.keys())

    fig = plt.figure(figsize=figsize or _journal_figsize("full_page"))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax_traj = fig.add_subplot(gs[0, 0])
    ax_cpa = fig.add_subplot(gs[0, 1])
    ax_ctrl = fig.add_subplot(gs[1, 0])
    ax_margin = fig.add_subplot(gs[1, 1])

    # ── Panel 1: Trajectory snapshots (representative method) ─────────
    if methods:
        representative = methods[0]
        result = results_by_method[representative][0] if results_by_method[representative] else None
        if result:
            # Ownship full trajectory
            own_x = [s.x for s in result.state_history]
            own_y = [s.y for s in result.state_history]
            ax_traj.plot(own_x, own_y, "b-", linewidth=1.2, alpha=0.7, label="Ownship")
            ax_traj.plot(own_x[0], own_y[0], "bo", markersize=6, label="Start")
            ax_traj.plot(own_x[-1], own_y[-1], "b*", markersize=8, label="End")

            # Target trajectories
            for i, tgt_hist in enumerate(result.target_histories):
                if tgt_hist:
                    tx = [s.x for s in tgt_hist]
                    ty = [s.y for s in tgt_hist]
                    ax_traj.plot(tx, ty, "gray", linewidth=0.8, linestyle="--", alpha=0.5)

            # Mark snapshot points
            for ts in timestamps:
                step = int(ts / 0.5)
                if step < len(result.state_history):
                    s = result.state_history[step]
                    ax_traj.plot(s.x, s.y, "o", color="red", markersize=5, alpha=0.5)
                    ax_traj.annotate(f"{ts}s", (s.x, s.y), fontsize=6, alpha=0.7)

        ax_traj.set_xlabel("X [m]")
        ax_traj.set_ylabel("Y [m]")
        ax_traj.set_title(f"Trajectory — {scenario_name}")
        ax_traj.set_aspect("equal")
        ax_traj.legend(fontsize=6, loc="upper right")
        ax_traj.grid(True, alpha=0.3)

    # ── Panel 2: CPA evolution ─────────────────────────────────────────
    time_bins = np.linspace(0, max_time, 60 + 1)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

    for method in methods:
        all_d = []
        for r in results_by_method[method]:
            if not r.target_histories or not r.target_histories[0]:
                continue
            for k in range(min(len(r.state_history), len(r.target_histories[0]))):
                own = r.state_history[k]
                tgt = r.target_histories[0][k]
                d = np.sqrt((own.x - tgt.x)**2 + (own.y - tgt.y)**2)
                all_d.append((k * 0.5, d))
        if not all_d:
            continue

        means = np.zeros(60)
        lowers = np.zeros(60)
        uppers = np.zeros(60)
        for i, (t_start, t_end) in enumerate(zip(time_bins[:-1], time_bins[1:])):
            bin_vals = [d for t, d in all_d if t_start <= t < t_end]
            if bin_vals:
                arr = np.array(bin_vals)
                means[i] = np.mean(arr)
                lowers[i] = np.percentile(arr, 5)
                uppers[i] = np.percentile(arr, 95)
        style = _get_style_for_method(method)
        ax_cpa.plot(bin_centers, means, color=style["color"], linewidth=1.2, label=method)
        ax_cpa.fill_between(bin_centers, lowers, uppers, color=style["color"], alpha=0.10)

    ax_cpa.set_xlabel("Time [s]")
    ax_cpa.set_ylabel("Distance to Target [m]")
    ax_cpa.set_title("CPA Evolution")
    ax_cpa.legend(fontsize=6, ncol=2, loc="upper right")
    ax_cpa.grid(True, alpha=0.3)
    ax_cpa.set_xlim(0, max_time)

    # ── Panel 3: Control inputs (rudder only, representative episode) ──
    if methods:
        result = results_by_method[methods[0]][0] if results_by_method[methods[0]] else None
        if result:
            for method in methods[:3]:  # top 3 methods
                r = results_by_method[method][0] if results_by_method[method] else None
                if r is None:
                    continue
                times_c = [i * 0.5 for i in range(len(r.command_history))]
                rudders = [c.rudder for c in r.command_history]
                style = _get_style_for_method(method)
                ax_ctrl.plot(times_c[:len(rudders)], rudders, color=style["color"],
                           linewidth=1.0, alpha=0.7, label=method)

    ax_ctrl.axhline(y=35, color="red", linestyle="--", alpha=0.3, linewidth=0.5)
    ax_ctrl.axhline(y=-35, color="red", linestyle="--", alpha=0.3, linewidth=0.5)
    ax_ctrl.set_xlabel("Time [s]")
    ax_ctrl.set_ylabel("Rudder [deg]")
    ax_ctrl.set_title("Control Input (Rudder)")
    ax_ctrl.legend(fontsize=6, loc="upper right")
    ax_ctrl.grid(True, alpha=0.3)
    ax_ctrl.set_ylim(-40, 40)

    # ── Panel 4: Safety margin over time (with bootstrap CI bands) ──────
    for method in methods:
        all_margins = []
        for r in results_by_method[method]:
            sbd = r.metadata.get("safety_distance_breakdown", [])
            if sbd and r.target_histories and r.target_histories[0]:
                for k in range(min(len(r.state_history), len(r.target_histories[0]),
                                  len(sbd) if len(sbd) == 1 else len(r.state_history))):
                    own = r.state_history[k]
                    tgt = r.target_histories[0][k]
                    d = np.sqrt((own.x - tgt.x)**2 + (own.y - tgt.y)**2)
                    total_safe = sbd[0].get("total", 500.0) if sbd else 500.0
                    all_margins.append((k * 0.5, d - total_safe))

        if not all_margins:
            continue
        means_m = np.zeros(60)
        lowers_m = np.zeros(60)
        uppers_m = np.zeros(60)
        for i, (t_start, t_end) in enumerate(zip(time_bins[:-1], time_bins[1:])):
            bin_vals = [m for t, m in all_margins if t_start <= t < t_end]
            if bin_vals:
                arr = np.array(bin_vals)
                means_m[i] = np.mean(arr)
                lowers_m[i] = np.percentile(arr, 5)
                uppers_m[i] = np.percentile(arr, 95)
        style = _get_style_for_method(method)
        ax_margin.plot(bin_centers, means_m, color=style["color"], linewidth=1.2, label=method)
        ax_margin.fill_between(bin_centers, lowers_m, uppers_m, color=style["color"], alpha=0.10)

    ax_margin.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="Safety boundary")
    ax_margin.set_xlabel("Time [s]")
    ax_margin.set_ylabel("Safety Margin [m]")
    ax_margin.set_title("Safety Margin (d - d_safe)")
    ax_margin.legend(fontsize=6, ncol=2, loc="lower left")
    ax_margin.grid(True, alpha=0.3)
    ax_margin.set_xlim(0, max_time)

    fig.suptitle(f"Figure 1 — Overview: {scenario_name}", fontsize=12, y=1.01)
    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-panel composite — Figure 2 (statistical comparison)
# ---------------------------------------------------------------------------

def plot_figure2_statistical_comparison(
    results_by_method: dict[str, list],
    stats_report: dict = None,
    metric_name: str = "min_dcpa",
    reference_method: str = "Proposed",
    save_path: Optional[str] = None,
    figsize: tuple = None,
):
    """3-panel composite: Violin+significance + CDF + Forest plot.

    Designed as the second figure in a journal paper, providing a
    quantitative statistical comparison of all methods.

    Layout (1×3):
        [Violin + Significance]  [CDF Comparison]  [Forest Plot]

    Args:
        results_by_method: Dict method -> list of EpisodeResult.
        stats_report: Optional dict from run_full_statistical_analysis.
        metric_name: Key metric to visualize.
        reference_method: Method used as reference for forest plot.
        save_path: Output file path.
        figsize: Optional (w, h) override in inches.
    """
    methods = sorted(results_by_method.keys())

    fig = plt.figure(figsize=figsize or (14, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    ax_violin = fig.add_subplot(gs[0, 0])
    ax_cdf = fig.add_subplot(gs[0, 1])
    ax_forest = fig.add_subplot(gs[0, 2])

    # ── Panel 1: Violin with significance ──────────────────────────────
    metrics_dict = {}
    for method in methods:
        vals = [r.metrics.get(metric_name, np.nan) for r in results_by_method[method]]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            metrics_dict[method] = vals

    if metrics_dict:
        methods_plot = list(metrics_dict.keys())
        positions = range(len(methods_plot))
        data_list = [np.array(metrics_dict[m]) for m in methods_plot]

        violin_parts = ax_violin.violinplot(data_list, positions=positions,
                                             showmeans=False, showmedians=True, widths=0.7)
        if "bodies" in violin_parts:
            for idx, body in enumerate(violin_parts["bodies"]):
                if idx < len(methods_plot):
                    body.set_facecolor(_get_style_for_method(methods_plot[idx])["color"])
                    body.set_alpha(0.4)
                    body.set_edgecolor("black")
                    body.set_linewidth(0.5)

        for i, method in enumerate(methods_plot):
            vals = metrics_dict[method]
            if len(vals) <= 200:
                jitter = np.random.normal(positions[i], 0.06, len(vals))
                ax_violin.scatter(jitter, vals, color=_get_color(method),
                                alpha=0.5, s=4, edgecolors="none", zorder=3)

        # Significance brackets from stats report
        if stats_report:
            pairwise = stats_report.get("pairwise_comparisons", {}).get(metric_name, {})
            pairs = []
            pvals = []
            for i, method in enumerate(methods_plot):
                if method == reference_method:
                    continue
                tests = pairwise.get(method, {})
                ttest = tests.get("ttest", {})
                pv = ttest.get("p_value")
                if pv is not None and np.isfinite(pv):
                    ref_idx = methods_plot.index(reference_method) if reference_method in methods_plot else 0
                    pairs.append((ref_idx, i))
                    pvals.append(pv)
            if pairs:
                add_significance_brackets(ax_violin, pairs, pvals, data_list, list(positions))

        ax_violin.set_xticks(positions)
        ax_violin.set_xticklabels(methods_plot, rotation=30, ha="right", fontsize=8)
        ax_violin.set_ylabel(metric_name.replace("_", " ").title())
        ax_violin.set_title(f"(a) {metric_name}")
        ax_violin.grid(True, axis="y", alpha=0.3)

    # ── Panel 2: CDF ────────────────────────────────────────────────────
    for method in methods_plot if metrics_dict else methods:
        vals = sorted(metrics_dict.get(method, []))
        if not vals:
            continue
        n = len(vals)
        cdf = np.arange(1, n + 1) / n
        style = _get_style_for_method(method)
        marker_every = max(1, n // 15)
        ax_cdf.plot(vals, cdf, color=style["color"], linewidth=1.2, label=method,
                   marker=style["marker"], markevery=marker_every, markersize=3)

    ax_cdf.set_xlabel(metric_name.replace("_", " ").title())
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.set_title(f"(b) CDF: {metric_name}")
    ax_cdf.legend(fontsize=6, loc="lower right")
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.set_ylim(0, 1.05)

    # ── Panel 3: Forest plot ───────────────────────────────────────────
    if stats_report:
        pairwise = stats_report.get("pairwise_comparisons", {}).get(metric_name, {})
        forest_methods = []
        ds = []
        ci_ls = []
        ci_us = []
        pvals = []
        for method in methods:
            if method == reference_method:
                continue
            tests = pairwise.get(method, {})
            cohens_d = tests.get("cohens_d", 0.0)
            if not np.isfinite(cohens_d):
                cohens_d = 0.0
            ttest = tests.get("ttest", {})
            ci_l = ttest.get("ci_95_lower", cohens_d - 0.5)
            ci_u = ttest.get("ci_95_upper", cohens_d + 0.5)
            pv = ttest.get("p_value", 1.0)
            if not np.isfinite(ci_l):
                ci_l = cohens_d - 0.5
            if not np.isfinite(ci_u):
                ci_u = cohens_d + 0.5
            forest_methods.append(method)
            ds.append(cohens_d)
            ci_ls.append(ci_l)
            ci_us.append(ci_u)
            pvals.append(pv)

        if forest_methods:
            y_pos = range(len(forest_methods))
            for i, (method, d, ci_l, ci_u, pv) in enumerate(
                zip(forest_methods, ds, ci_ls, ci_us, pvals)
            ):
                style = _get_style_for_method(method)
                is_sig = pv < 0.05
                ax_forest.errorbar(d, i, xerr=[[d - ci_l], [ci_u - d]],
                                 fmt="s" if is_sig else "o", color=style["color"],
                                 capsize=3, markersize=7 if is_sig else 5,
                                 linewidth=1.2,
                                 label=f"{method} ({_significance_label(pv)})")

            ax_forest.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax_forest.set_yticks(list(y_pos))
            ax_forest.set_yticklabels(forest_methods, fontsize=8)
            ax_forest.set_xlabel(f"Cohen's d (vs. {reference_method})")
            ax_forest.set_title(f"(c) Effect Sizes")
            ax_forest.legend(fontsize=6, loc="best")
            ax_forest.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Figure 2 — Statistical Comparison: {metric_name}", fontsize=12)
    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Failure heatmap (Q5 fix — 2026-06-10)
# ---------------------------------------------------------------------------

def plot_failure_heatmap(
    results_df,
    save_path: str = None,
    title: str = "Failure Mode Distribution by Scenario × Method",
):
    """Plot a failure-mode heatmap matrix (scenario × method).

    Each cell shows the primary failure rate (color) and the most common
    failure type (annotation).  This is a standard figure in Q1 collision
    avoidance papers (Ocean Engineering, IEEE T-ITS).

    Args:
        results_df: pandas DataFrame with columns:
            scenario_id, method, failure_type, failure_flags.
        save_path: Optional output path.
        title: Figure title.
    """
    import pandas as pd

    scenarios = sorted(results_df["scenario_id"].unique())
    methods = sorted(results_df["method"].unique())
    failure_types = [f"F{i}" for i in range(1, 11)]

    # Build matrix: (scenario, method) → failure_rate
    matrix = np.zeros((len(scenarios), len(methods)))
    annotations = np.empty((len(scenarios), len(methods)), dtype=object)

    for si, sc in enumerate(scenarios):
        sc_data = results_df[results_df["scenario_id"] == sc]
        for mi, meth in enumerate(methods):
            meth_data = sc_data[sc_data["method"] == meth]
            n = len(meth_data)
            if n == 0:
                matrix[si, mi] = 0.0
                annotations[si, mi] = "N/A"
                continue
            # Failure rate = fraction of episodes with any failure
            fail_rate = meth_data["failure_type"].notna().mean()
            matrix[si, mi] = fail_rate
            # Most common failure type
            ft_counts = meth_data["failure_type"].value_counts()
            if len(ft_counts) > 0:
                annotations[si, mi] = f"{ft_counts.index[0]}\n({fail_rate:.0%})"
            else:
                annotations[si, mi] = "safe"

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), max(4, len(scenarios) * 0.6)))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for si in range(len(scenarios)):
        for mi in range(len(methods)):
            color = "white" if matrix[si, mi] > 0.5 else "black"
            ax.text(mi, si, annotations[si, mi], ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, label="Failure Rate", shrink=0.8)
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Safety distance stacked area chart (Q7 fix — 2026-06-10)
# ---------------------------------------------------------------------------

def plot_safety_distance_stacked(
    safety_breakdowns: list,
    timestamps: list = None,
    save_path: str = None,
    title: str = "Safety Distance Decomposition over Time",
):
    """Plot a stacked area chart showing safety distance components over time.

    This figure demonstrates HOW the proposed method achieves safety by
    showing the additive decomposition of the total safety distance into
    base_domain, dynamic_margin, chance_margin, tube_margin, and rule_margin.

    Required by top-venue reviewers to understand method internals.

    Args:
        safety_breakdowns: List of per-step lists of dicts with keys:
            base_domain, dynamic_margin, chance_margin, tube_margin,
            rule_margin, extra_margin, total.
        timestamps: Optional list of timestamp labels.
        save_path: Optional output path.
        title: Figure title.
    """
    if not safety_breakdowns:
        return

    n_steps = len(safety_breakdowns)
    if n_steps < 2:
        return

    components = ["base_domain", "dynamic_margin", "chance_margin",
                  "tube_margin", "rule_margin"]
    labels = ["Base Domain", "Dynamic Margin", "Chance Margin",
              "Tube Margin", "Rule Margin"]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

    # Aggregate across targets per step (mean)
    data = {c: np.zeros(n_steps) for c in components}
    for k in range(n_steps):
        step_breakdowns = safety_breakdowns[k]
        if not step_breakdowns:
            continue
        for c in components:
            vals = [bd.get(c, 0.0) for bd in step_breakdowns if isinstance(bd, dict)]
            data[c][k] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(8, 4))
    x = timestamps if timestamps is not None else np.arange(n_steps)

    # Build stacked arrays
    y_stack = np.zeros((len(components), n_steps))
    for i, c in enumerate(components):
        y_stack[i, :] = data[c]

    ax.stackplot(x, y_stack, labels=labels, colors=colors, alpha=0.85)

    # Plot total as a dashed line
    total = np.sum(y_stack, axis=0)
    ax.plot(x, total, "k--", linewidth=1.5, label="Total", alpha=0.7)

    ax.set_xlabel("Time Step" if timestamps is None else "Time [s]")
    ax.set_ylabel("Safety Distance [m]")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trajectory comparison plot (G3 fix — 2026-06-10)
# ---------------------------------------------------------------------------

def plot_trajectory_comparison(
    trajectory_data: dict,
    scenario_id: str = "",
    save_path: str = None,
    title: str = None,
):
    """Plot bird's-eye trajectory comparison across methods.

    Shows ownship trajectory + target ship trajectories for multiple methods
    overlaid on the same axes.  Critical for qualitative evaluation of
    collision avoidance behaviour.

    Args:
        trajectory_data: dict mapping method_name → dict with keys:
            own_x, own_y (arrays), target_x, target_y (list of arrays),
            own_params (VesselParams for hull outline).
        scenario_id: Scenario identifier for title.
        save_path: Optional output path.
        title: Override figure title.
    """
    if not trajectory_data:
        return

    methods = list(trajectory_data.keys())
    n_methods = len(methods)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        data = trajectory_data[method]

        # Plot ownship trajectory
        own_x = np.array(data.get("own_x", []))
        own_y = np.array(data.get("own_y", []))
        if len(own_x) > 1:
            ax.plot(own_x, own_y, "b-", linewidth=1.5, label="Ownship", alpha=0.9)
            # Mark start and end
            ax.scatter(own_x[0], own_y[0], c="blue", marker="o", s=60, zorder=5, label="Start")
            ax.scatter(own_x[-1], own_y[-1], c="blue", marker="s", s=60, zorder=5, label="End")

        # Plot target ship trajectories
        tgt_x_list = data.get("target_x", [])
        tgt_y_list = data.get("target_y", [])
        for t_idx, (tx, ty) in enumerate(zip(tgt_x_list, tgt_y_list)):
            if len(tx) > 1:
                color = colors[(t_idx + 1) % len(colors)]
                ax.plot(tx, ty, "--", color=color, linewidth=1.2,
                        label=f"Target {t_idx + 1}", alpha=0.8)
                ax.scatter(tx[0], ty[0], c=color, marker="o", s=40, zorder=4)
                ax.scatter(tx[-1], ty[-1], c=color, marker="x", s=40, zorder=4)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{method}", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="best")

    # Hide unused subplots
    for ax_idx in range(n_methods, len(axes)):
        axes[ax_idx].set_visible(False)

    fig_title = title or f"Trajectory Comparison — {scenario_id}"
    fig.suptitle(fig_title, fontsize=12)
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Runtime vs Safety Pareto frontier (G4 fix — 2026-06-10)
# ---------------------------------------------------------------------------

def plot_runtime_safety_pareto(
    results_df,
    safety_metric: str = "min_safety_margin",
    runtime_metric: str = "mean_runtime",
    save_path: str = None,
    title: str = None,
):
    """Plot runtime vs safety trade-off (Pareto frontier).

    Each point = one method.  X-axis = mean runtime [s], Y-axis = safety
    metric.  A Pareto frontier highlights methods that are NOT dominated.
    Standard figure in IEEE T-ITS / Ocean Engineering for demonstrating
    the efficiency-safety trade-off.

    Args:
        results_df: DataFrame with columns: method, {safety_metric}, {runtime_metric}.
        safety_metric: Column name for safety (higher = safer).
        runtime_metric: Column name for runtime (lower = faster).
        save_path: Optional output path.
        title: Override title.
    """
    import pandas as pd

    methods = results_df["method"].unique()
    if len(methods) < 2:
        return

    # Aggregate per method
    agg = results_df.groupby("method").agg(
        mean_safety=(safety_metric, "mean"),
        mean_runtime=(runtime_metric, "mean"),
        std_safety=(safety_metric, "std"),
        std_runtime=(runtime_metric, "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))

    # Color by method category
    mpc_methods = {"B3", "B4", "B5", "B6", "B7", "B8", "Proposed"}
    colors = []
    for m in agg["method"]:
        if m == "Proposed":
            colors.append("#d62728")  # red
        elif m in mpc_methods:
            colors.append("#1f77b4")  # blue
        else:
            colors.append("#7f7f7f")  # gray

    # Scatter with error bars
    for i, row in agg.iterrows():
        ax.errorbar(
            row["mean_runtime"], row["mean_safety"],
            xerr=row["std_runtime"], yerr=row["std_safety"],
            fmt="o", capsize=4, color=colors[i], markersize=10,
            label=row["method"], markeredgewidth=1.5, markeredgecolor="black",
        )

    # Draw Pareto frontier
    pareto_x = agg["mean_runtime"].values
    pareto_y = agg["mean_safety"].values
    # Sort by x (runtime)
    idx = np.argsort(pareto_x)
    px, py = pareto_x[idx], pareto_y[idx]
    # Pareto: points where no other point has both lower x and higher y
    pareto_mask = np.ones(len(px), dtype=bool)
    for i in range(len(px)):
        for j in range(len(px)):
            if i != j and px[j] <= px[i] and py[j] >= py[i]:
                if px[j] < px[i] or py[j] > py[i]:
                    pareto_mask[i] = False
                    break
    ax.plot(px[pareto_mask], py[pareto_mask], "k--", linewidth=1.0, alpha=0.5,
            label="Pareto frontier")

    ax.set_xlabel(f"Mean Runtime [s] ({runtime_metric})")
    ax.set_ylabel(f"Safety ({safety_metric})")
    ax.set_title(title or "Runtime–Safety Trade-off")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Robustness degradation curve (G5 fix — 2026-06-10)
# ---------------------------------------------------------------------------

def plot_robustness_degradation(
    uncertainty_levels: list[str],
    safety_by_level: dict,  # method → list of safety values per level
    save_path: str = None,
    title: str = None,
):
    """Plot safety metric degradation across uncertainty levels.

    Demonstrates how each method's safety performance degrades as
    AIS uncertainty increases (low → medium → high).  Key figure for
    showing the chance-constrained tube-MPC's robustness advantage.

    Args:
        uncertainty_levels: List of level labels (e.g. ["Low", "Medium", "High"]).
        safety_by_level: dict mapping method → list of safety values,
            one per uncertainty level.
        save_path: Optional output path.
        title: Override title.
    """
    if not safety_by_level or len(uncertainty_levels) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(uncertainty_levels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(safety_by_level)))
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "h"]

    for i, (method, values) in enumerate(safety_by_level.items()):
        if len(values) != len(uncertainty_levels):
            continue
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linewidth = 2.5 if method == "Proposed" else 1.5
        linestyle = "-" if method == "Proposed" else "--"
        ax.plot(x, values, linestyle, color=color, marker=marker,
                linewidth=linewidth, markersize=8, label=method,
                markeredgewidth=1, markeredgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(uncertainty_levels, fontsize=10)
    ax.set_xlabel("AIS Uncertainty Level")
    ax.set_ylabel("Safety Metric (min_safety_margin [m])")
    ax.set_title(title or "Robustness Degradation Under Increasing AIS Uncertainty")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_figure(fig, path: str):
    """Save figure to disk, creating directories as needed."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
