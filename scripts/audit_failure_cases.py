#!/usr/bin/env python
"""Audit failure cases from experiment results.

Extracts and categorises failures, producing:
    failure_cases.csv         — all failure episodes with replay commands
    failure_summary.json      — counts and rates by type × method

Usage:
    python scripts/audit_failure_cases.py --input results/core --output results/core/audit
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FAILURE_LABELS = {
    "F1": "Collision",
    "F2": "Near-miss",
    "F3": "Rule violation",
    "F4": "Channel boundary violation",
    "F5": "Bank clearance violation",
    "F6": "Inter-ship clearance violation",
    "F7": "MPC infeasibility / solver failure / solver timeout",
    "F8": "CBF infeasible / over-intervention",
    "F9": "Fallback activation / fallback failed",
    "F10": "Runtime deadline miss",
    # F11-F15 are detected indirectly via metrics (not failure flags):
    #   grounding_violation, nonfinite_*_count, chance_constraint_violations,
    #   min_safety_margin < 0, backend_degraded
}


def main():
    parser = argparse.ArgumentParser(description="Audit failure cases in experiment results.")
    parser.add_argument("--input", required=True, help="Directory with metrics_by_episode.csv")
    parser.add_argument("--output", default=None, help="Output directory for audit.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    csv_path = _resolve_csv(args.input)
    if args.dry_run:
        print(f"DRY-RUN: Would read {csv_path}")
        return

    output_dir = args.output or os.path.join(os.path.dirname(csv_path), "audit")
    os.makedirs(output_dir, exist_ok=True)

    df = read_csv_safe(csv_path)
    print(f"Loaded {len(df)} rows")

    methods = sorted(df["method"].unique())
    fail_cols = [c for c in df.columns if c.startswith("fail_")]
    has_failure_type = "failure_type" in df.columns

    # ── Extract all failure cases ────────────────────────────────────────
    failures = []
    for _, row in df.iterrows():
        ft = row.get("failure_type", "") if has_failure_type else ""
        # NaN is truthy in Python — guard explicitly (BUG #3 fix)
        if isinstance(ft, float) and np.isnan(ft):
            ft = ""
        flags = {fc.replace("fail_", ""): bool(row.get(fc, False)) for fc in fail_cols}
        active_flags = [k for k, v in flags.items() if v]
        if ft or active_flags:
            # seed: guard against NaN which raises ValueError on int() (BUG #4 fix)
            raw_seed = row.get("seed", 0)
            if isinstance(raw_seed, float) and np.isnan(raw_seed):
                seed_val = 0
            else:
                seed_val = int(raw_seed)
            rec = {
                "episode_id": row.get("episode_id", ""),
                "scenario_id": row.get("scenario_id", ""),
                "method": row.get("method", ""),
                "seed": seed_val,
                "failure_type": ft or "+".join(active_flags),
                "failure_labels": [FAILURE_LABELS.get(f, f) for f in (active_flags or [ft])],
            }
            # Add key metrics for context
            for mc in ["min_dcpa", "collision_rate", "colregs_violation",
                        "mean_runtime", "ship_domain_violation_rate"]:
                if mc in row:
                    rec[mc] = row[mc]
            # Build replay command
            rec["replay_command"] = (
                f"python scripts/run_single_scenario.py "
                f"--scenario {rec['scenario_id']} --method {rec['method']} "
                f"--seed {rec['seed']}"
            )
            failures.append(rec)

    if failures:
        fail_df = pd.DataFrame(failures)
        fail_df.to_csv(os.path.join(output_dir, "failure_cases.csv"), index=False)
        print(f"  → failure_cases.csv ({len(fail_df)} cases)")

    # ── Summary by category ──────────────────────────────────────────────
    summary = {}
    for method in methods:
        mdf = df[df["method"] == method]
        entry = {"n_total": len(mdf),
                 "n_any_failure": int(mdf["failure_type"].notna().sum()) if has_failure_type else 0}
        for fc in fail_cols:
            if fc in mdf.columns:
                entry[fc.replace("fail_", "")] = {
                    "count": int(mdf[fc].sum()),
                    "rate": float(round(mdf[fc].mean(), 4)),
                }
        summary[method] = entry

    # Add categorized breakdown
    for method in methods:
        mf = [f for f in failures if f["method"] == method]
        summary[method]["failures_by_type"] = {}
        for f in mf:
            for lbl in f.get("failure_labels", []):
                summary[method]["failures_by_type"][lbl] = \
                    summary[method]["failures_by_type"].get(lbl, 0) + 1

    with open(os.path.join(output_dir, "failure_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  → failure_summary.json")

    # ── Special categories ───────────────────────────────────────────────
    _extract_special_categories(failures, output_dir)

    # Metadata
    meta = {"timestamp": datetime.now().isoformat(), "input": csv_path,
            "n_total_episodes": len(df), "n_failure_episodes": len(failures),
            "methods": methods}
    with open(os.path.join(output_dir, "audit_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nAudit complete → {output_dir}/")


def _extract_special_categories(failures, output_dir):
    """Extract and save special failure categories."""
    categories = {
        "collision_cases": [f for f in failures if "F1" in str(f.get("failure_type", ""))],
        "near_miss_cases": [f for f in failures if "F2" in str(f.get("failure_type", ""))],
        "rule_violation_cases": [f for f in failures if "F3" in str(f.get("failure_type", ""))],
        "grounding_cases": [f for f in failures if "F11" in str(f.get("failure_type", ""))],
        "solver_failure_cases": [f for f in failures if "F7" in str(f.get("failure_type", ""))],
        "cbf_infeasible_cases": [f for f in failures if "F8" in str(f.get("failure_type", ""))],
        "fallback_failed_cases": [f for f in failures if "F9" in str(f.get("failure_type", ""))],
        "nonfinite_dynamics_cases": [f for f in failures if "F12" in str(f.get("failure_type", ""))],
        "chance_constraint_violation_cases": [f for f in failures if "F13" in str(f.get("failure_type", ""))],
        "safety_margin_violation_cases": [f for f in failures if "F14" in str(f.get("failure_type", ""))],
        "backend_degraded_cases": [f for f in failures if "F15" in str(f.get("failure_type", ""))],
    }
    for cat_name, items in categories.items():
        if items:
            pd.DataFrame(items).to_csv(
                os.path.join(output_dir, f"{cat_name}.csv"), index=False,
            )
            print(f"  → {cat_name}.csv ({len(items)} cases)")


def _resolve_csv(path):
    if os.path.isfile(path): return path
    for sub in ["metrics_by_episode.csv", "raw/core_results.csv"]:
        p = os.path.join(path, sub)
        if os.path.exists(p): return p
    print(f"ERROR: No CSV in {path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
