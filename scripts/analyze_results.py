#!/usr/bin/env python
"""Unified analysis entry point — descriptive stats, significance, effect sizes.

Usage:
    # After running experiments:
    python scripts/analyze_results.py --input results/quick_test --output results/quick_test/analysis

    # The input directory should contain a raw/ subdirectory with metrics.csv
    # (or a direct path to a CSV file).

Outputs (inside --output):
    summary.csv               per-method descriptive statistics
    pairwise_comparisons.csv  paired t-test + Wilcoxon + effect sizes
    effect_sizes.csv          Cohen's d summary per metric
    significance_tests.csv    Holm-Bonferroni / Benjamini-Hochberg corrected
    failure_taxonomy.json     failure rates by method
    metadata.json             run parameters and config snapshot
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ta_mrc_pe_cc_tube_mpc.utils.io_utils import read_csv_safe

from ta_mrc_pe_cc_tube_mpc.evaluation.statistics import (
    benjamini_hochberg,
    cliffs_delta,
    cohens_d,
    compute_summary_statistics,
    holm_bonferroni,
    paired_ttest,
    wilcoxon_test,
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results with full statistical pipeline."
    )
    parser.add_argument("--input", required=True,
                        help="Path to results CSV or directory containing raw/metrics.csv.")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <input>/analysis).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level.")
    args = parser.parse_args()

    # Resolve input CSV
    input_path = args.input
    if os.path.isdir(input_path):
        candidates = [
            os.path.join(input_path, "metrics.csv"),
            os.path.join(input_path, "raw", "core_results.csv"),
            os.path.join(input_path, "raw", "metrics.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                input_path = c
                break
        else:
            print(f"ERROR: No metrics CSV found in {input_path}")
            print(f"  Looked for: {candidates}")
            sys.exit(1)

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    output_dir = args.output or os.path.join(
        os.path.dirname(input_path) or ".", "analysis"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")

    # ── Load data ───────────────────────────────────────────────────────
    df = read_csv_safe(input_path)
    print(f"Loaded {len(df)} rows")
    print(f"Methods:  {sorted(df['method'].unique())}")
    print(f"Scenarios: {sorted(df['scenario_id'].unique())}")

    methods = sorted(df["method"].unique())

    # ISSUE #18 (2026-06-10): Detect and warn about mixed backends
    if "backend" in df.columns:
        backends = df["backend"].unique()
        print(f"Backends: {list(backends)}")
        if len(backends) > 1:
            print("  WARNING: Multiple backends detected — results may be confounded.")
            print(f"  Backends: {list(backends)}")
            for b in backends:
                n_rows = int((df["backend"] == b).sum())
                print(f"    {b}: {n_rows} rows")
            print("  Consider filtering by backend before analysis.")
            print()

    # Identify metric columns
    exclude = {"episode_id", "scenario_id", "method", "seed", "failure_type",
               "trajectory_file", "trajectory_format", "backend",
               "gpu_devices", "gpu_device_count"}
    metric_cols = [c for c in df.columns
                   if c not in exclude
                   and not c.startswith("fail_")
                   and not c.startswith("runtime_")
                   and not c.startswith("Unnamed")]
    failure_cols = [c for c in df.columns if c.startswith("fail_")]
    runtime_cols = [c for c in df.columns if "runtime" in c.lower()]

    # ── 1. Summary CSV ──────────────────────────────────────────────────
    print("\n=== Summary Statistics ===")
    summary_rows = []
    for method in methods:
        mdf = df[df["method"] == method]
        row = {"method": method, "n_episodes": len(mdf)}
        for col in metric_cols + runtime_cols:
            vals = mdf[col].dropna().values
            if len(vals) >= 3:
                s = compute_summary_statistics(vals)
                row[f"{col}_mean"] = s.get("mean", np.nan)
                row[f"{col}_median"] = s.get("median", np.nan)
                row[f"{col}_std"] = s.get("std", np.nan)
                row[f"{col}_valid_n"] = s.get("valid_count", 0)
                row[f"{col}_nan_n"] = s.get("nan_count", 0)
        for fc in failure_cols:
            if fc in mdf.columns:
                row[fc] = float(mdf[fc].mean())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  → {summary_path}")

    # ── 2. Pairwise comparisons + effect sizes ──────────────────────────
    print("\n=== Pairwise Comparisons ===")
    pairwise = []
    for metric in metric_cols[:10]:
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                xdf = df[df["method"] == m1][["episode_id", metric]].dropna()
                ydf = df[df["method"] == m2][["episode_id", metric]].dropna()
                merged = xdf.merge(ydf, on="episode_id", suffixes=("_x", "_y"))
                x = merged[f"{metric}_x"].values
                y = merged[f"{metric}_y"].values
                n = len(x)
                if n < 5:
                    continue
                tres = paired_ttest(x, y)
                wres = wilcoxon_test(x, y)
                pairwise.append({
                    "metric": metric, "method_1": m1, "method_2": m2,
                    "n": n, "n_excluded": tres.get("n_excluded", 0),
                    "mean_1": float(np.mean(x)), "mean_2": float(np.mean(y)),
                    "mean_diff": tres["mean_diff"],
                    "t_statistic": tres["statistic"],
                    "p_value_t": tres["p_value"],
                    "p_value_wilcoxon": wres["p_value"],
                    "cohens_d": cohens_d(x, y),
                    "cliffs_delta": cliffs_delta(x, y),
                })

    if pairwise:
        pw = pd.DataFrame(pairwise)
        # Multiple comparison correction
        pvals_t = [r["p_value_t"] for r in pairwise]
        hb = holm_bonferroni(pvals_t, args.alpha)
        bh = benjamini_hochberg(pvals_t, args.alpha)
        for i, r in enumerate(pairwise):
            r["significant_holm"] = hb[i]
            r["significant_bh"] = bh[i]

        pw = pd.DataFrame(pairwise)
        pw.to_csv(os.path.join(output_dir, "pairwise_comparisons.csv"), index=False)
        print(f"  → pairwise_comparisons.csv ({len(pw)} rows)")

        # Significance tests table
        sig = pw[pw["significant_holm"] | pw["significant_bh"]]
        sig.to_csv(os.path.join(output_dir, "significance_tests.csv"), index=False)
        print(f"  → significance_tests.csv ({len(sig)} significant)")

    # ── 3. Effect sizes ─────────────────────────────────────────────────
    print("\n=== Effect Sizes ===")
    es_rows = []
    for metric in metric_cols[:10]:
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                x = df[df["method"] == m1][metric].dropna().values
                y = df[df["method"] == m2][metric].dropna().values
                mn = min(len(x), len(y))
                if mn >= 5:
                    es_rows.append({
                        "metric": metric, "method_1": m1, "method_2": m2,
                        "cohens_d": cohens_d(x[:mn], y[:mn]),
                        "cliffs_delta": cliffs_delta(x[:mn], y[:mn]),
                    })
    if es_rows:
        es_df = pd.DataFrame(es_rows)
        es_df.to_csv(os.path.join(output_dir, "effect_sizes.csv"), index=False)
        print(f"  → effect_sizes.csv ({len(es_df)} rows)")

    # ── 4. Failure taxonomy ─────────────────────────────────────────────
    print("\n=== Failure Taxonomy ===")
    ft = {}
    for method in methods:
        mdf = df[df["method"] == method]
        entry = {"n_total": len(mdf),
                 "any_failure": int(mdf["failure_type"].notna().sum()),
                 "any_failure_rate": float(mdf["failure_type"].notna().mean())}
        for fc in failure_cols:
            if fc in mdf.columns:
                entry[fc] = {"count": int(mdf[fc].sum()),
                             "rate": float(mdf[fc].mean())}
        ft[method] = entry
    with open(os.path.join(output_dir, "failure_taxonomy.json"), "w") as f:
        json.dump(ft, f, indent=2, default=str)
    print(f"  → failure_taxonomy.json")

    # ── 5. Metadata ─────────────────────────────────────────────────────
    meta = {
        "analysis_timestamp": datetime.now().isoformat(),
        "input_file": os.path.abspath(input_path),
        "n_rows": len(df),
        "methods": methods,
        "scenarios": sorted(df["scenario_id"].unique()),
        "metric_columns": metric_cols,
        "failure_columns": failure_cols,
        "alpha": args.alpha,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  → metadata.json")

    print(f"\nAnalysis complete. Output: {output_dir}/")
    for fname in sorted(os.listdir(output_dir)):
        print(f"  {fname}")


if __name__ == "__main__":
    main()
