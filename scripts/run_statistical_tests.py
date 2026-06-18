#!/usr/bin/env python
"""Run full statistical analysis: pairwise tests, effect sizes, corrections.

Input:  metrics_by_episode.csv (from experiment output)
Output: statistical_tests.csv  — Proposed vs each baseline, all metrics

Usage:
    python scripts/run_statistical_tests.py --input results/core --output results/core/analysis
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ta_mrc_pe_cc_tube_mpc.utils.io_utils import read_csv_safe

from ta_mrc_pe_cc_tube_mpc.evaluation.statistics import (
    benjamini_hochberg,
    cliffs_delta,
    cohens_d,
    holm_bonferroni,
    paired_ttest,
    wilcoxon_test,
)


def main():
    parser = argparse.ArgumentParser(description="Statistical tests on experiment results.")
    parser.add_argument("--input", required=True, help="Directory or CSV file.")
    parser.add_argument("--output", default=None, help="Output directory.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = _resolve_csv(args.input)
    if args.dry_run:
        print(f"DRY-RUN: Would read {csv_path}")
        return

    output_dir = args.output or os.path.join(os.path.dirname(csv_path), "analysis")
    os.makedirs(output_dir, exist_ok=True)

    df = read_csv_safe(csv_path)
    print(f"Loaded {len(df)} rows")

    methods = sorted(df["method"].unique())
    # Compare Proposed vs each baseline
    baselines = [m for m in methods if m != "Proposed"]
    if "Proposed" not in methods:
        print("WARNING: 'Proposed' not found in methods — comparing first vs others")
        baselines = methods[1:]
        proposed_name = methods[0]
    else:
        proposed_name = "Proposed"

    # Identify metric columns
    exclude = {"episode_id", "scenario_id", "method", "seed", "failure_type",
               "trajectory_file", "trajectory_format"}
    metric_cols = [c for c in df.columns if c not in exclude
                   and not c.startswith("fail_") and not c.startswith("runtime_")]

    results = []
    for metric in metric_cols[:15]:  # top 15 metrics
        for bl in baselines:
            # Align by episode for paired comparison
            xdf = df[df["method"] == proposed_name][["scenario_id", "seed", metric]].dropna()
            ydf = df[df["method"] == bl][["scenario_id", "seed", metric]].dropna()
            merged = xdf.merge(ydf, on=["scenario_id", "seed"], suffixes=("_x", "_y"))
            x = merged[f"{metric}_x"].values
            y = merged[f"{metric}_y"].values
            n = len(x)
            if n < 5:
                continue

            tres = paired_ttest(x, y)
            wres = wilcoxon_test(x, y)
            d_val = cohens_d(x, y)
            c_val = cliffs_delta(x, y)

            results.append({
                "metric": metric,
                "method_a": proposed_name, "method_b": bl,
                "n_pairs": n, "n_excluded": tres.get("n_excluded", 0),
                f"mean_{proposed_name}": float(np.mean(x)),
                f"mean_{bl}": float(np.mean(y)),
                "mean_diff": tres["mean_diff"],
                "t_statistic": tres["statistic"],
                "p_value_t": tres["p_value"],
                "p_value_wilcoxon": wres["p_value"],
                "cohens_d": d_val,
                "cliffs_delta": c_val,
                "ci_95_lower": tres.get("ci_95_lower", np.nan),
                "ci_95_upper": tres.get("ci_95_upper", np.nan),
            })

    if results:
        dfr = pd.DataFrame(results)
        # Multiple comparison correction
        pvals_t = [r["p_value_t"] for r in results]
        hb = holm_bonferroni(pvals_t, args.alpha)
        bh = benjamini_hochberg(pvals_t, args.alpha)
        dfr["significant_holm"] = hb
        dfr["significant_bh"] = bh

        out_path = os.path.join(output_dir, "statistical_tests.csv")
        dfr.to_csv(out_path, index=False)
        sig_count = sum(hb)
        print(f"  → statistical_tests.csv ({len(dfr)} rows, {sig_count} significant after Holm)")

        # Effect size summary
        es = dfr.groupby("metric").agg(
            mean_cohens_d=("cohens_d", lambda x: np.mean(np.abs(x))),
            max_cohens_d=("cohens_d", lambda x: np.max(np.abs(x))),
        ).reset_index()
        es.to_csv(os.path.join(output_dir, "effect_size_summary.csv"), index=False)
        print(f"  → effect_size_summary.csv")

    # Metadata
    meta = {"timestamp": datetime.now().isoformat(), "alpha": args.alpha,
            "proposed": proposed_name, "baselines": baselines, "n_rows": len(df)}
    with open(os.path.join(output_dir, "stats_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nAnalysis complete → {output_dir}/")


def _resolve_csv(path: str) -> str:
    if os.path.isfile(path):
        return path
    for sub in ["metrics_by_episode.csv", "raw/core_results.csv", "raw/metrics.csv"]:
        p = os.path.join(path, sub)
        if os.path.exists(p):
            return p
    print(f"ERROR: No CSV found in {path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
