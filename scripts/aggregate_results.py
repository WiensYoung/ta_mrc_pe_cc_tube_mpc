#!/usr/bin/env python
"""Aggregate raw episode metrics into summary tables.

Input:  experiment output directory containing metrics_by_episode.csv
Output:
    metrics_by_scenario_method.csv  — mean per (scenario, method)
    overall_summary.csv              — mean ± std per method (all scenarios)
    runtime_summary.csv              — mean/P95/P99 runtime per method
    failure_summary.csv              — failure counts and rates per method

Usage:
    python scripts/aggregate_results.py --input results/core --output results/core/aggregated
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


def main():
    parser = argparse.ArgumentParser(description="Aggregate episode metrics.")
    parser.add_argument("--input", required=True, help="Directory containing metrics_by_episode.csv")
    parser.add_argument("--output", default=None, help="Output directory (default: <input>/aggregated)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, do not write files.")
    args = parser.parse_args()

    input_dir = args.input
    csv_path = os.path.join(input_dir, "metrics_by_episode.csv")
    if not os.path.exists(csv_path):
        # Try raw/ subdirectory
        alt = os.path.join(input_dir, "raw", "core_results.csv")
        if os.path.exists(alt):
            csv_path = alt
        else:
            print(f"ERROR: No metrics CSV found at {csv_path} or {alt}")
            sys.exit(1)

    output_dir = args.output or os.path.join(input_dir, "aggregated")
    if args.dry_run:
        print(f"DRY-RUN: Would read {csv_path} → {output_dir}/")
        print("  Outputs: metrics_by_scenario_method.csv, overall_summary.csv, "
              "runtime_summary.csv, failure_summary.csv")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = read_csv_safe(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    methods = sorted(df["method"].unique())
    scenarios = sorted(df["scenario_id"].unique())
    print(f"Methods: {methods}")
    print(f"Scenarios: {scenarios}")

    # Identify metric columns (exclude metadata)
    exclude = {"episode_id", "scenario_id", "method", "seed", "failure_type",
               "trajectory_file", "trajectory_format", "Unnamed: 0"}
    metric_cols = [c for c in df.columns if c not in exclude
                   and not c.startswith("fail_") and not c.startswith("runtime_")]
    failure_cols = [c for c in df.columns if c.startswith("fail_")]
    runtime_cols = [c for c in df.columns if "runtime" in c.lower()]

    # ── 1. metrics_by_scenario_method ────────────────────────────────────
    agg_sm = df.groupby(["scenario_id", "method"])[metric_cols].mean().reset_index()
    agg_sm.to_csv(os.path.join(output_dir, "metrics_by_scenario_method.csv"), index=False)
    print(f"  → metrics_by_scenario_method.csv ({len(agg_sm)} rows)")

    # ── 2. overall_summary ───────────────────────────────────────────────
    rows = []
    for method in methods:
        mdf = df[df["method"] == method]
        row = {"method": method, "n_episodes": len(mdf)}
        for col in metric_cols + runtime_cols:
            vals = mdf[col].dropna().values
            if len(vals) >= 3:
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_std"] = float(np.std(vals, ddof=1))
        for fc in failure_cols:
            if fc in mdf.columns:
                row[fc.replace("fail_", "") + "_rate"] = float(mdf[fc].mean())
        rows.append(row)
    overall = pd.DataFrame(rows)
    overall.to_csv(os.path.join(output_dir, "overall_summary.csv"), index=False)
    print(f"  → overall_summary.csv ({len(overall)} rows)")

    # ── 3. runtime_summary ───────────────────────────────────────────────
    rt_rows = []
    for method in methods:
        mdf = df[df["method"] == method]
        row = {"method": method}
        for rc in runtime_cols:
            vals = mdf[rc].dropna().values
            if len(vals) >= 3:
                row[f"{rc}_mean"] = float(np.mean(vals))
                row[f"{rc}_p95"] = float(np.percentile(vals, 95))
                row[f"{rc}_p99"] = float(np.percentile(vals, 99))
        rt_rows.append(row)
    if rt_rows:
        rt = pd.DataFrame(rt_rows)
        rt.to_csv(os.path.join(output_dir, "runtime_summary.csv"), index=False)
        print(f"  → runtime_summary.csv ({len(rt)} rows)")

    # ── 4. failure_summary ───────────────────────────────────────────────
    fail_rows = []
    for method in methods:
        mdf = df[df["method"] == method]
        row = {"method": method, "n_total": len(mdf),
               "n_any_failure": int(mdf["failure_type"].notna().sum() if "failure_type" in mdf.columns else 0)}
        if "failure_type" in mdf.columns:
            row["any_failure_rate"] = float(mdf["failure_type"].notna().mean())
        for fc in failure_cols:
            if fc in mdf.columns:
                row[fc] = int(mdf[fc].sum())
                row[fc + "_rate"] = float(mdf[fc].mean())
        fail_rows.append(row)
    if fail_rows:
        fail = pd.DataFrame(fail_rows)
        fail.to_csv(os.path.join(output_dir, "failure_summary.csv"), index=False)
        print(f"  → failure_summary.csv ({len(fail)} rows)")

    # ── Metadata ─────────────────────────────────────────────────────────
    meta = {"aggregation_timestamp": datetime.now().isoformat(),
            "input_file": csv_path, "n_rows": len(df),
            "methods": methods, "scenarios": scenarios,
            "metric_columns": metric_cols}
    with open(os.path.join(output_dir, "aggregation_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nAggregation complete → {output_dir}/")


if __name__ == "__main__":
    main()
