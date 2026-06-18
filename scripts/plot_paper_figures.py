#!/usr/bin/env python
"""Generate paper-quality figures from experiment results.

Input:  experiment output directory with metrics_by_episode.csv
Output: PDF figures in <output>/figures/

Usage:
    python scripts/plot_paper_figures.py --input results/core --output results/core/figures
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ta_mrc_pe_cc_tube_mpc.utils.io_utils import read_csv_safe
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Publication style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.format": "pdf",
})


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--input", required=True, help="Directory with metrics_by_episode.csv")
    parser.add_argument("--output", default=None, help="Output directory for figures.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = _resolve_csv(args.input)
    if args.dry_run:
        print(f"DRY-RUN: Would read {csv_path}")
        print("  Outputs: safety_efficiency_pareto.pdf, min_cpa_distribution.pdf, "
              "runtime_distribution.pdf, ablation_summary.pdf, failure_modes.pdf")
        return

    output_dir = args.output or os.path.join(os.path.dirname(csv_path), "figures")
    os.makedirs(output_dir, exist_ok=True)

    df = read_csv_safe(csv_path)
    methods = sorted(df["method"].unique())
    print(f"Loaded {len(df)} rows, methods={methods}")

    # ── Enhanced Figures (delegating to pub_plots where possible) ────────
    from ta_mrc_pe_cc_tube_mpc.evaluation.pub_plots import (
        plot_cdf_comparison,
        plot_forest_effect_sizes,
        plot_failure_heatmap,
        plot_radar_chart,
        plot_sensitivity_tornado,
        plot_metric_boxplot_swarm,
    )

    # ── Figure 1: Safety-efficiency Pareto ───────────────────────────────
    _fig_safety_efficiency_pareto(df, methods, output_dir)

    # ── Figure 2: Min CPA distribution (violin + boxplot) ────────────────
    _fig_min_cpa_distribution(df, methods, output_dir)

    # ── Figure 3: CDF comparison ─────────────────────────────────────────
    _fig_cdf_comparison(df, methods, output_dir)

    # ── Figure 4: Runtime distribution ───────────────────────────────────
    _fig_runtime_distribution(df, methods, output_dir)

    # ── Figure 5: Ablation summary ───────────────────────────────────────
    _fig_ablation_summary(df, methods, output_dir)

    # ── Figure 6: Failure modes ──────────────────────────────────────────
    _fig_failure_modes(df, methods, output_dir)

    # ── Figure 7: Failure heatmap ────────────────────────────────────────
    _fig_failure_heatmap(df, methods, output_dir)

    # ── Figure 8: Effect sizes forest plot ───────────────────────────────
    _fig_forest_plot(df, methods, output_dir)

    # Metadata
    meta = {"timestamp": datetime.now().isoformat(), "input": csv_path}
    with open(os.path.join(output_dir, "figures_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nFigures saved → {output_dir}/")


def _resolve_csv(path):
    if os.path.isfile(path): return path
    for sub in ["metrics_by_episode.csv", "raw/core_results.csv"]:
        p = os.path.join(path, sub)
        if os.path.exists(p): return p
    print(f"ERROR: No CSV in {path}")
    sys.exit(1)


# ── Figure generators ────────────────────────────────────────────────────────

def _fig_safety_efficiency_pareto(df, methods, out):
    """Collision rate vs route efficiency scatter per method."""
    agg = df.groupby("method").agg(
        collision_rate=("collision_rate", "mean"),
    ).reset_index()
    # Use min_dcpa as proxy for safety
    safety = df.groupby("method")["min_dcpa"].mean().reset_index()
    agg = agg.merge(safety, on="method")

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    for i, m in enumerate(methods):
        row = agg[agg["method"] == m]
        if len(row):
            ax.scatter(row["min_dcpa"], row["collision_rate"],
                       label=m, color=colors[i], s=80, edgecolors="black")
    ax.set_xlabel("Mean Min CPA [m]")
    ax.set_ylabel("Collision Rate")
    ax.set_title("Safety-Efficiency Tradeoff")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "safety_efficiency_pareto.pdf"))
    plt.close(fig)


def _fig_min_cpa_distribution(df, methods, out):
    """Box plot of min CPA per method."""
    data = [df[df["method"] == m]["min_dcpa"].dropna().values for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=methods, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.tab10(np.linspace(0, 1, len(methods)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Min CPA [m]")
    ax.set_title("Min CPA Distribution by Method")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # method-labels, not dates
    fig.savefig(os.path.join(out, "min_cpa_distribution.pdf"))
    plt.close(fig)


def _fig_runtime_distribution(df, methods, out):
    """Bar chart of mean/P95 runtime per method."""
    rt_cols = [c for c in df.columns if "runtime" in c.lower() and "mean" in c.lower()]
    if not rt_cols:
        rt_cols = [c for c in df.columns if "mean_runtime" in c.lower()]
    if not rt_cols:
        print("  [WARN] No runtime columns found in CSV — skipping runtime distribution")
        return
    rt_col = rt_cols[0]
    means = [df[df["method"] == m][rt_col].dropna().mean() for m in methods]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(methods, means, color=plt.cm.tab10(np.linspace(0, 1, len(methods))))
    ax.set_ylabel("Mean Runtime [s]")
    ax.set_title("Mean Runtime per Method")
    ax.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="0.1s deadline")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # method-labels, not dates
    fig.savefig(os.path.join(out, "runtime_distribution.pdf"))
    plt.close(fig)


def _fig_ablation_summary(df, methods, out):
    """Bar chart of key metrics across methods (ablation proxy)."""
    metrics = ["collision_rate", "ship_domain_violation_rate",
               "colregs_violation"]
    available = [m for m in metrics if m in df.columns]
    if not available:
        print("  [WARN] No ablation metrics found in CSV — skipping ablation summary")
        return
    agg = df.groupby("method")[available].mean()
    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for ax, metric in zip(axes, available):
        vals = [agg.loc[m, metric] if m in agg.index else 0 for m in methods]
        ax.bar(methods, vals, color=plt.cm.tab10(np.linspace(0, 1, len(methods))))
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Ablation Summary — Key Metrics")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ablation_summary.pdf"))
    plt.close(fig)


def _fig_failure_modes(df, methods, out):
    """Stacked bar of failure counts per method."""
    fail_cols = [c for c in df.columns if c.startswith("fail_")]
    if not fail_cols:
        print("  [WARN] No fail_* columns found in CSV — skipping failure modes figure")
        return
    counts = {}
    for m in methods:
        mdf = df[df["method"] == m]
        counts[m] = [int(mdf[fc].sum()) if fc in mdf.columns else 0 for fc in fail_cols]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.7
    bottom = np.zeros(len(methods))
    colors = plt.cm.tab10(np.linspace(0, 1, len(fail_cols)))
    for i, fc in enumerate(fail_cols):
        vals = [counts[m][i] for m in methods]
        ax.bar(x, vals, width, bottom=bottom, label=fc.replace("fail_", ""),
               color=colors[i])
        bottom += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Failure Count")
    ax.set_title("Failure Modes by Method")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # method-labels, not dates
    fig.savefig(os.path.join(out, "failure_modes.pdf"))
    plt.close(fig)


def _fig_failure_heatmap(df, methods, out):
    """Failure rate heatmap by scenario and method — uses UNION of all F1-F10 flags."""
    fail_cols = [c for c in df.columns if c.startswith("fail_")]
    if not fail_cols:
        print("  [WARN] No fail_* columns found in CSV — skipping failure heatmap")
        return

    scenarios = sorted(df["scenario_id"].unique())
    failure_rates = {}
    for sc in scenarios:
        failure_rates[sc] = {}
        for m in methods:
            mdf = df[(df["scenario_id"] == sc) & (df["method"] == m)]
            if len(mdf) > 0:
                # Compute UNION failure rate: any episode with ANY failure flag set
                any_failure = mdf[fail_cols].max(axis=1)
                failure_rates[sc][m] = float(any_failure.mean())
            else:
                failure_rates[sc][m] = 0.0

    from ta_mrc_pe_cc_tube_mpc.evaluation.pub_plots import plot_failure_heatmap
    plot_failure_heatmap(
        failure_rates,
        save_path=os.path.join(out, "failure_heatmap.pdf"),
        title="Composite Failure Rate (F1–F10) by Scenario and Method",
    )
    print(f"  → failure_heatmap.pdf")


def _fig_cdf_comparison(df, methods, out):
    """CDF comparison of min_dcpa across methods."""
    metrics_dict = {}
    for m in methods:
        vals = df[df["method"] == m]["min_dcpa"].dropna().values
        if len(vals) > 0:
            metrics_dict[m] = vals.tolist()

    if metrics_dict:
        from ta_mrc_pe_cc_tube_mpc.evaluation.pub_plots import plot_cdf_comparison
        plot_cdf_comparison(
            metrics_dict,
            metric_name="Min CPA [m]",
            save_path=os.path.join(out, "cdf_min_dcpa.pdf"),
            title="CDF: Minimum CPA Across Methods",
        )
        print(f"  → cdf_min_dcpa.pdf")


def _fig_forest_plot(df, methods, out):
    """Forest plot of effect sizes vs Proposed."""
    # Compute effect sizes from the CSV data
    from scipy import stats as sp_stats

    if "Proposed" not in methods:
        print("  [SKIP] No 'Proposed' method for forest plot reference")
        return

    ref_vals_all = df[df["method"] == "Proposed"]["min_dcpa"].dropna().values
    effect_sizes = {}

    for m in methods:
        if m == "Proposed":
            continue
        m_vals_all = df[df["method"] == m]["min_dcpa"].dropna().values

        # Align by scenario_id + seed for paired comparison
        pairs_ref = []
        pairs_m = []
        for _, row_ref in df[df["method"] == "Proposed"].iterrows():
            match = df[(df["method"] == m) &
                      (df["scenario_id"] == row_ref["scenario_id"]) &
                      (df["seed"] == row_ref["seed"])]
            if len(match) > 0:
                pairs_ref.append(row_ref["min_dcpa"])
                pairs_m.append(match.iloc[0]["min_dcpa"])

        if len(pairs_ref) < 3:
            continue

        diff = np.array(pairs_m) - np.array(pairs_ref)
        d = float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-12))
        se = np.std(diff, ddof=1) / np.sqrt(len(diff))
        ci_lo = d - 1.96 * se
        ci_hi = d + 1.96 * se

        try:
            _, pv = sp_stats.ttest_rel(pairs_m, pairs_ref)
        except Exception:
            pv = 1.0

        effect_sizes[m] = {
            "cohens_d": d,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": pv,
        }

    if effect_sizes:
        from ta_mrc_pe_cc_tube_mpc.evaluation.pub_plots import plot_forest_effect_sizes
        plot_forest_effect_sizes(
            effect_sizes,
            reference_method="Proposed",
            save_path=os.path.join(out, "forest_effect_sizes.pdf"),
            title="Effect Sizes (Cohen's d) vs. Proposed",
        )
        print(f"  → forest_effect_sizes.pdf")


if __name__ == "__main__":
    main()
