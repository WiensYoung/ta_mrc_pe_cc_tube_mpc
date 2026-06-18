"""Visualization: trajectory plots, metric comparisons, failure distributions."""

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(
    own_states: list,
    target_states_list: list[list],
    scenario_name: str = "",
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """Plot ownship and target trajectories in 2D.

    Args:
        own_states: List of ownship VesselState.
        target_states_list: List of lists of target VesselState.
        scenario_name: Name for the plot title.
        save_path: Path to save the figure.
        dpi: Figure resolution.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Ownship
    own_x = [s.x for s in own_states]
    own_y = [s.y for s in own_states]
    if own_x:  # guard against empty state history
        ax.plot(own_x, own_y, "b-", linewidth=1.5, label="Ownship")
        ax.plot(own_x[0], own_y[0], "bo", markersize=8, label="Start")
        ax.plot(own_x[-1], own_y[-1], "b*", markersize=10, label="End")

    # Targets
    # Color palette: supports up to 10 targets via distinct, colorblind-friendly colors
    colors = [
        "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
        "#FF7F00", "#A65628", "#F781BF", "#999999",
        "#66C2A5", "#FC8D62",
    ]
    for i, tgt_states in enumerate(target_states_list):
        if not tgt_states:  # skip empty target histories
            continue
        color = colors[i % len(colors)]
        tx = [s.x for s in tgt_states]
        ty = [s.y for s in tgt_states]
        if tx:
            ax.plot(tx, ty, "--", color=color, linewidth=1.0, label=f"Target {i+1}")
            ax.plot(tx[0], ty[0], "o", color=color, markersize=6)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Trajectory: {scenario_name}")
    ax.legend(loc="best")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(
    methods: list[str],
    metric_values: dict,  # method -> list of values
    metric_name: str = "",
    save_path: Optional[str] = None,
):
    """Box plot comparing metrics across methods.

    Args:
        methods: Method names.
        metric_values: Dict mapping method name to list of values.
        metric_name: Name of the metric for y-axis label.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [metric_values.get(m, []) for m in methods]
    bp = ax.boxplot(data, labels=methods, patch_artist=True)

    for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(methods)))):
        patch.set_facecolor(color)

    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Comparison Across Methods")
    ax.grid(True, axis="y", alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_failure_distribution(
    failure_counts: dict,  # failure code -> count
    save_path: Optional[str] = None,
):
    """Bar chart of failure distribution across types.

    Args:
        failure_counts: Dict mapping failure code to count.
        save_path: Path to save the figure.
    """
    from ..constants import FAILURE_CODES

    codes = [f"F{i}" for i in range(1, 11)]
    names = [FAILURE_CODES.get(c, c) for c in codes]
    counts = [failure_counts.get(c, 0) for c in codes]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, counts, color=plt.cm.Reds(np.linspace(0.3, 0.9, len(codes))))

    ax.set_xlabel("Failure Type")
    ax.set_ylabel("Count")
    ax.set_title("Failure Distribution")
    ax.tick_params(axis="x", rotation=45)

    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(count), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
