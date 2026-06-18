#!/usr/bin/env python
"""Run full statistical analysis on experiment results.

Implements the statistical framework from configs/statistics.yaml:
  - Paired t-test and Wilcoxon signed-rank for pairwise method comparison
  - Holm-Bonferroni and Benjamini-Hochberg multiple comparison correction
  - Cohen's d and Cliff's delta effect sizes
  - Cluster bootstrap for perturbation-augmented episodes
  - Mixed-effects model (Metric ~ Method + Scenario + Waterway + ShipType + Disturbance + (1|Seed))
  - Failure taxonomy analysis
  - Report tables

Usage:
    python scripts/run_statistics.py --input results/raw/core_results.csv --output_dir results/processed
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.evaluation.statistics import (
    benjamini_hochberg,
    cliffs_delta,
    cluster_bootstrap,
    cohens_d,
    compute_summary_statistics,
    holm_bonferroni,
    mixed_effects_model_interface,
    paired_ttest,
    wilcoxon_test,
)
from ta_mrc_pe_cc_tube_mpc.utils.io_utils import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Run statistical analysis on experiment results.")
    parser.add_argument("--input", required=True, help="Path to results CSV file.")
    parser.add_argument("--output_dir", default="results/processed", help="Output directory.")
    parser.add_argument("--statistics_config", default="configs/statistics.yaml",
                        help="Statistics config file.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    input_path = os.path.join(project_root, args.input) if not os.path.isabs(args.input) else args.input
    output_dir = os.path.join(project_root, args.output_dir)
    tables_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Load config
    stats_config = {}
    config_path = os.path.join(project_root, args.statistics_config)
    if os.path.exists(config_path):
        stats_config = load_yaml(config_path)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Scenarios: {df['scenario_id'].unique().tolist()}")

    # Identify metric columns (exclude metadata, failures, and runtime columns)
    exclude_cols = {"episode_id", "scenario_id", "method", "seed", "failure_type"}
    runtime_patterns = ("mean_runtime", "p95_runtime", "p99_runtime", "max_runtime",
                        "deadline_miss", "runtime_", "requested_backend", "actual_backend",
                        "actual_solver", "backend_degraded", "fallback_reason",
                        "timing_breakdown")
    metric_cols = [c for c in df.columns
                   if c not in exclude_cols
                   and not c.startswith("fail_")
                   and not c.startswith("nonfinite_")
                   and not any(p in c for p in runtime_patterns)]

    # Failures
    failure_cols = [c for c in df.columns if c.startswith("fail_")]

    methods = sorted(df["method"].unique())
    results_all = {}

    # -- 1. Descriptive statistics by method --
    print("\n=== Descriptive Statistics ===")
    descriptive = {}
    for method in methods:
        method_df = df[df["method"] == method]
        method_stats = {}
        for col in metric_cols:
            values = method_df[col].dropna().values
            if len(values) >= 3:
                method_stats[col] = compute_summary_statistics(values)
        descriptive[method] = method_stats

    with open(os.path.join(output_dir, "descriptive_statistics.json"), "w") as f:
        json.dump(descriptive, f, indent=2, default=str)
    print(f"Saved descriptive statistics for {len(methods)} methods")

    # -- 2. Pairwise method comparisons --
    print("\n=== Pairwise Comparisons ===")
    pairwise_results = []
    # Select key metrics for pairwise comparison (safety-critical and non-redundant)
    PAIRWISE_PRIORITY_METRICS = [
        "collision_rate", "min_dcpa", "worst_5_percent_dcpa",
        "ship_domain_violation_rate", "colregs_violation",
        "rudder_saturation_rate", "cbf_intervention_rate",
        "fallback_trigger_rate", "mpc_infeasibility",
        "mean_runtime",
    ]
    pairwise_metrics = [m for m in PAIRWISE_PRIORITY_METRICS if m in df.columns]
    for metric in pairwise_metrics:
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                # Align by episode_id for valid paired comparison
                x_df = df[df["method"] == m1][["episode_id", metric]].dropna()
                y_df = df[df["method"] == m2][["episode_id", metric]].dropna()
                merged = x_df.merge(y_df, on="episode_id", suffixes=("_x", "_y"))
                x = merged[f"{metric}_x"].values
                y = merged[f"{metric}_y"].values
                n = len(x)
                if n < 5:
                    continue

                t_result = paired_ttest(x, y)
                w_result = wilcoxon_test(x, y)
                d_val = cohens_d(x, y)
                c_val = cliffs_delta(x, y)

                pairwise_results.append({
                    "metric": metric,
                    "method_1": m1,
                    "method_2": m2,
                    "n": n,
                    "mean_1": float(np.mean(x)),
                    "mean_2": float(np.mean(y)),
                    "mean_diff": t_result["mean_diff"],
                    "t_statistic": t_result["statistic"],
                    "p_value_t": t_result["p_value"],
                    "p_value_wilcoxon": w_result["p_value"],
                    "cohens_d": d_val,
                    "cliffs_delta": c_val,
                })

    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_path = os.path.join(output_dir, "pairwise_comparisons.csv")
    pairwise_df.to_csv(pairwise_path, index=False)

    # -- 3. Multiple comparison correction --
    if len(pairwise_results) > 0 and stats_config.get("tests", {}).get("holm_bonferroni", True):
        print("\n=== Multiple Comparison Correction ===")
        p_values_t = [r["p_value_t"] for r in pairwise_results]
        p_values_w = [r["p_value_wilcoxon"] for r in pairwise_results]

        hb_t = holm_bonferroni(p_values_t, alpha=args.alpha)
        hb_w = holm_bonferroni(p_values_w, alpha=args.alpha)
        bh_t = benjamini_hochberg(p_values_t, alpha=args.alpha)
        bh_w = benjamini_hochberg(p_values_w, alpha=args.alpha)

        for idx, r in enumerate(pairwise_results):
            r["significant_t_holm"] = hb_t[idx]
            r["significant_w_holm"] = hb_w[idx]
            r["significant_t_bh"] = bh_t[idx]
            r["significant_w_bh"] = bh_w[idx]

        sig_holm = sum(1 for r in pairwise_results if r["significant_t_holm"])
        sig_bh = sum(1 for r in pairwise_results if r["significant_t_bh"])
        print(f"Holm-Bonferroni significant (t-test): {sig_holm}/{len(pairwise_results)}")
        print(f"Benjamini-Hochberg significant (t-test): {sig_bh}/{len(pairwise_results)}")

        pairwise_df = pd.DataFrame(pairwise_results)
        pairwise_df.to_csv(pairwise_path, index=False)

    # -- 4. Effect size summary --
    print("\n=== Effect Sizes ===")
    effect_sizes = []
    for metric in pairwise_metrics:
        d_vals = []
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                x = df[df["method"] == m1][metric].dropna().values
                y = df[df["method"] == m2][metric].dropna().values
                min_n = min(len(x), len(y))
                if min_n >= 5:
                    d_vals.append(abs(cohens_d(x[:min_n], y[:min_n])))
        if d_vals:
            effect_sizes.append({
                "metric": metric,
                "mean_abs_cohens_d": float(np.mean(d_vals)),
                "max_abs_cohens_d": float(np.max(d_vals)),
            })

    effect_df = pd.DataFrame(effect_sizes)
    effect_df.to_csv(os.path.join(output_dir, "effect_sizes.csv"), index=False)
    items = [
        f"{r['metric']}={r['mean_abs_cohens_d']:.3f}"
        for r in effect_sizes
    ]
    print(f"Mean Cohen's d per metric: {', '.join(items)}")

    # -- 5. Cluster bootstrap (by scenario) --
    if stats_config.get("tests", {}).get("cluster_bootstrap", True) and "scenario_id" in df.columns:
        print("\n=== Cluster Bootstrap ===")
        cluster_ids = df["scenario_id"].astype("category").cat.codes.values
        bootstrap_results = {}
        for method in methods:
            method_mask = df["method"] == method
            for metric in ["min_dcpa", "mean_runtime"]:
                if metric not in df.columns:
                    continue
                # Align metric data with cluster IDs via shared index after dropna
                method_df = df.loc[method_mask, [metric, "scenario_id"]].dropna()
                if len(method_df) < 10:
                    continue
                metric_data = method_df[metric].values
                aligned_clusters = method_df["scenario_id"].astype("category").cat.codes.values

                bs = cluster_bootstrap(
                    metric_data,
                    aligned_clusters,
                    statistic_fn=np.mean,
                    n_bootstrap=stats_config.get("significance", {}).get("cluster_bootstrap_n", 500),
                )
                bootstrap_results[f"{method}_{metric}"] = bs

        with open(os.path.join(output_dir, "cluster_bootstrap.json"), "w") as f:
            json.dump(bootstrap_results, f, indent=2, default=str)
        print(f"Cluster bootstrap complete for {len(bootstrap_results)} method-metric pairs")

    # -- 6. Mixed-effects model --
    if stats_config.get("tests", {}).get("mixed_effects_model", True):
        print("\n=== Mixed-Effects Model ===")
        me_config = stats_config.get("mixed_effects", {})
        me_results = {}
        for metric in ["min_dcpa", "collision_rate", "mean_runtime"]:
            if metric not in df.columns:
                continue
            model_df = df[[metric, "method", "scenario_id", "seed"]].dropna().copy()
            model_df = model_df.rename(columns={"scenario_id": "Scenario", "method": "Method"})
            # Add placeholder waterway/ship type columns if missing
            if "waterway" not in model_df.columns:
                model_df["Waterway"] = model_df["Scenario"]
            if "ship_type" not in model_df.columns:
                model_df["ShipType"] = "cargo"
            if "disturbance" not in model_df.columns:
                model_df["Disturbance"] = "none"

            formula = me_config.get("formula", f"{metric} ~ Method + Scenario")
            # Use the actual column names present in the DataFrame
            predictor_cols = [c for c in ["Method", "Scenario", "Waterway", "ShipType", "Disturbance"]
                             if c in model_df.columns]
            predictor_str = " + ".join(predictor_cols) if predictor_cols else "Method"
            adapted_formula = f"{metric} ~ {predictor_str}"

            me_result = mixed_effects_model_interface(
                model_df,
                formula=adapted_formula,
                random_effects=me_config.get("random_effects", "1|Seed"),
            )
            me_results[metric] = me_result

        with open(os.path.join(output_dir, "mixed_effects_model.json"), "w") as f:
            json.dump(me_results, f, indent=2, default=str)
        for metric, mr in me_results.items():
            status = "converged" if mr.get("converged") else f"fallback={mr.get('fallback', 'none')}"
            print(f"  {metric}: {status}")

    # -- 7. Failure taxonomy summary --
    print("\n=== Failure Taxonomy ===")
    failure_summary = {}
    for method in methods:
        method_failures = df[df["method"] == method]
        failure_summary[method] = {
            "n_total": len(method_failures),
            "any_failure": int(method_failures["failure_type"].notna().sum()),
            "any_failure_rate": float(method_failures["failure_type"].notna().mean()),
        }
        for fc in failure_cols:
            if fc in df.columns:
                failure_summary[method][fc] = {
                    "count": int(method_failures[fc].sum()),
                    "rate": float(method_failures[fc].mean()),
                }

    with open(os.path.join(output_dir, "failure_taxonomy.json"), "w") as f:
        json.dump(failure_summary, f, indent=2, default=str)

    # -- 8. Generate LaTeX tables --
    print("\n=== Generating Report Tables ===")
    _generate_tables(df, methods, metric_cols, failure_cols, tables_dir, descriptive)

    # -- Final summary --
    print(f"\nAnalysis complete. Outputs saved to {output_dir}/")
    print(f"  - descriptive_statistics.json")
    print(f"  - pairwise_comparisons.csv")
    print(f"  - effect_sizes.csv")
    print(f"  - cluster_bootstrap.json")
    print(f"  - mixed_effects_model.json")
    print(f"  - failure_taxonomy.json")
    print(f"Tables saved to {tables_dir}/")


def _generate_tables(df, methods, metric_cols, failure_cols, tables_dir, descriptive):
    """Generate LaTeX report tables."""

    # Table: Core metrics summary (mean +/-std per method)
    core_metrics = ["collision_rate", "min_dcpa", "ship_domain_violation_rate",
                    "colregs_violation", "mean_runtime"]
    core_metrics = [m for m in core_metrics if m in metric_cols]

    rows = []
    for method in methods:
        row = [method]
        for metric in core_metrics:
            stats = descriptive.get(method, {}).get(metric, {})
            mean_val = stats.get("mean", float("nan"))
            std_val = stats.get("std", float("nan"))
            row.append(f"{mean_val:.3f} +/-{std_val:.3f}")
        rows.append(row)

    if rows:
        header = ["Method"] + [m.replace("_", " ").title() for m in core_metrics]
        _write_latex_table(
            os.path.join(tables_dir, "core_metrics_summary.tex"),
            header, rows,
            caption="Core metrics summary (mean +/-std) across all scenarios.",
            label="tab:core_metrics",
        )

    # Table: Failure taxonomy
    fail_rows = []
    for method in methods:
        method_df = df[df["method"] == method]
        row = [method, str(len(method_df))]
        for fc in failure_cols:
            if fc in method_df.columns:
                count = int(method_df[fc].sum())
                rate = float(method_df[fc].mean())
                row.append(f"{count} ({rate:.1%})")
        fail_rows.append(row)

    fail_header = ["Method", "N"] + [fc.replace("fail_", "") for fc in failure_cols if fc in df.columns]
    _write_latex_table(
        os.path.join(tables_dir, "failure_taxonomy_summary.tex"),
        fail_header, fail_rows,
        caption="Failure taxonomy summary: count (rate) per failure type.",
        label="tab:failure_taxonomy",
    )


def _write_latex_table(path, header, rows, caption="", label=""):
    """Write a simple LaTeX table."""
    n_cols = len(header)
    col_spec = "l" + "c" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        lines.append("    " + " & ".join(str(c) for c in row) + r" \\")
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
