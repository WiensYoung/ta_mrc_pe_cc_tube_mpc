"""Analyze experiment results: statistics, plots, and report generation."""

import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from ..evaluation.plots import (
    plot_failure_distribution,
    plot_metric_comparison,
    plot_trajectory,
)
from ..evaluation.report_tables import save_all_tables
from ..evaluation.statistics import compute_summary_statistics
from ..utils.io_utils import load_json, save_json


def _build_reproducibility_metadata(results_csv: str) -> dict:
    """Build reproducibility metadata block for Q8 compliance.

    Records the exact environment, versions, and input hash so that
    any researcher can reproduce the analysis exactly.
    """
    import hashlib

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
    }
    # Dependency versions
    for pkg in ["numpy", "scipy", "pandas", "matplotlib", "cvxpy"]:
        try:
            mod = __import__(pkg)
            meta[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            meta[f"{pkg}_version"] = "not_installed"

    # Input file hash (first 1MB for speed)
    try:
        h = hashlib.md5()
        with open(results_csv, "rb") as f:
            h.update(f.read(1_048_576))
        meta["input_md5_first_1mb"] = h.hexdigest()
    except Exception:
        meta["input_md5_first_1mb"] = "unavailable"

    return meta


def analyze_results(
    results_csv: str,
    output_dir: str = "results/processed",
    generate_plots: bool = True,
) -> dict:
    """Analyze experiment results and generate reports.

    Args:
        results_csv: Path to results CSV file.
        output_dir: Output directory for processed results.
        generate_plots: Whether to generate figure files.

    Returns:
        dict with summary statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "..", "figures")
    tables_dir = os.path.join(output_dir, "..", "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Q8 fix (2026-06-10): Save reproducibility metadata alongside results
    repro_meta = _build_reproducibility_metadata(results_csv)
    with open(os.path.join(output_dir, "reproducibility_metadata.json"), "w") as f:
        json.dump(repro_meta, f, indent=2, default=str)

    df = pd.read_csv(results_csv)

    # Summary statistics by method
    methods = df["method"].unique()
    summary = {}

    # Only compute summary statistics for numeric columns.
    # String columns (encounter_type, etc.) must be excluded.
    METADATA_COLS = {"episode_id", "scenario_id", "method", "seed", "failure_type"}
    numeric_cols = df.select_dtypes(include="number").columns
    metric_cols = [c for c in numeric_cols
                   if c not in METADATA_COLS and not c.startswith("fail_")]

    for method in methods:
        method_df = df[df["method"] == method]
        method_summary = {}
        for col in metric_cols:
            if col in method_df.columns:
                values = method_df[col].dropna().values
                if len(values) > 0:
                    method_summary[col] = compute_summary_statistics(values)
        summary[method] = method_summary

    save_json(summary, os.path.join(output_dir, "summary_statistics.json"))

    # Failure distribution
    failure_cols = [f"fail_F{i}" for i in range(1, 11)]
    failure_counts = {}
    for col in failure_cols:
        if col in df.columns:
            failure_counts[col.replace("fail_", "")] = int(df[col].sum())

    if generate_plots:
        plot_failure_distribution(
            failure_counts,
            save_path=os.path.join(figures_dir, "failure_distribution.png"),
        )

        # Metric comparison box plots
        for metric in ["min_dcpa", "mean_runtime"]:
            if metric in df.columns:
                metric_dict = {}
                for method in methods:
                    vals = df[df["method"] == method][metric].dropna().values
                    metric_dict[method] = vals
                plot_metric_comparison(
                    list(methods),
                    metric_dict,
                    metric_name=metric,
                    save_path=os.path.join(figures_dir, f"{metric}_comparison.png"),
                )

    # Build report tables
    # Reconstruct results from CSV (approximate)
    # For full analysis, EpisodeResult objects would be needed
    # This provides CSV-level summary

    return summary


def build_episode_level_csv(
    results: list,  # list of EpisodeResult
    output_path: str,
):
    """Build detailed episode-level CSV from EpisodeResult objects.

    Args:
        results: List of EpisodeResult objects.
        output_path: Output CSV path.
    """
    rows = []
    for r in results:
        row = {
            "episode_id": r.episode_id,
            "scenario_id": r.scenario_id,
            "method": r.method,
            "seed": r.seed,
            "real_episode_id": r.real_episode_id,
            "perturbation_id": r.perturbation_id,
            "failure_type": r.failure_type or "",
        }
        # Failure flags
        for code, flag in r.failure_flags.items():
            row[f"fail_{code}"] = flag
        # Metrics
        for key, value in r.metrics.items():
            if isinstance(value, (int, float, bool, str)):
                row[f"metric_{key}"] = value
        # Runtime
        for key, value in r.runtime_stats.items():
            row[f"runtime_{key}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
