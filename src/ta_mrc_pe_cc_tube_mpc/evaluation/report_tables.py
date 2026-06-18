"""Report table generation.

Generates:
    1. Core metrics summary table
    2. Failure taxonomy summary
    3. Ablation comparison table
    4. Sensitivity summary table
    5. Runtime summary table
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from ..evaluation.statistics import compute_summary_statistics


def build_core_metrics_table(
    results: list,
    metrics_list: list[str] = None,
) -> pd.DataFrame:
    """Build a core metrics summary table by method.

    Each row is a method, each column a metric with mean ± std.

    Args:
        results: List of EpisodeResult objects.
        metrics_list: List of metric names to include.

    Returns:
        DataFrame with methods as rows and metrics as columns.
    """
    if metrics_list is None:
        metrics_list = [
            "collision_rate", "min_dcpa", "worst_5_percent_dcpa",
            "ship_domain_violation_rate", "mean_runtime", "p95_runtime",
        ]

    methods = sorted(set(r.method for r in results))
    rows = []

    for method in methods:
        method_results = [r for r in results if r.method == method]
        row = {"Method": method, "N": len(method_results)}

        for metric in metrics_list:
            values = [r.metrics.get(metric, np.nan) for r in method_results]
            # Filter None and NaN
            values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if values:
                stats = compute_summary_statistics(np.array(values))
                # Q2+Q6 fix (2026-06-10): Include 95% CI and effect size.
                # Format: "mean [CI_low, CI_high]"
                ci_low = stats.get("ci_95_lower", 0)
                ci_high = stats.get("ci_95_upper", 0)
                row[metric] = (
                    f"{stats['mean']:.3f} [{ci_low:.3f}, {ci_high:.3f}]"
                )
            else:
                row[metric] = "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


def build_failure_taxonomy_table(
    results: list,
) -> pd.DataFrame:
    """Build failure taxonomy summary table.

    Args:
        results: List of EpisodeResult objects.

    Returns:
        DataFrame with failure counts by type and method.
    """
    from ..constants import FAILURE_CODES
    from .failure_taxonomy import compute_failure_rates_by_method

    rates = compute_failure_rates_by_method(results)
    methods = sorted(rates.keys())
    rows = []

    for method in methods:
        row = {"Method": method}
        for code, name in FAILURE_CODES.items():
            rate = rates[method].get(code, 0.0)
            row[name] = f"{rate:.1%}"
        rows.append(row)

    return pd.DataFrame(rows)


def build_ablation_comparison_table(
    results: list,
) -> pd.DataFrame:
    """Build ablation comparison table.

    Each row is a method (Proposed full, Proposed w/o X).
    Columns show key metrics with changes from full model.

    Args:
        results: List of EpisodeResult objects.

    Returns:
        DataFrame with ablation comparison.
    """
    methods = sorted(set(r.method for r in results))
    metrics_to_show = ["collision_rate", "min_dcpa", "mean_runtime"]

    rows = []
    for method in methods:
        method_results = [r for r in results if r.method == method]
        row = {"Method": method}

        for metric in metrics_to_show:
            values = [r.metrics.get(metric, np.nan) for r in method_results]
            values = [v for v in values if v is not None and isinstance(v, (int, float)) and not np.isnan(v)]
            if values:
                row[metric] = f"{np.mean(values):.3f}"
            else:
                row[metric] = "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


def build_sensitivity_summary_table(
    sensitivity_results: dict,
) -> pd.DataFrame:
    """Build sensitivity analysis summary table.

    Args:
        sensitivity_results: Dict with nested results organized by parameter.

    Returns:
        DataFrame with sensitivity analysis summary.
    """
    rows = []
    for param_name, param_values in sensitivity_results.items():
        for value, metrics in param_values.items():
            row = {"Parameter": param_name, "Value": str(value)}
            for metric_name, metric_val in metrics.items():
                row[metric_name] = metric_val
            rows.append(row)
    return pd.DataFrame(rows)


def build_runtime_summary_table(
    results: list,
) -> pd.DataFrame:
    """Build runtime performance summary table.

    Args:
        results: List of EpisodeResult objects.

    Returns:
        DataFrame with runtime metrics by method.
    """
    methods = sorted(set(r.method for r in results))
    rows = []

    for method in methods:
        method_results = [r for r in results if r.method == method]
        runtimes = [r.runtime_stats.get("mean_runtime", 0.0) for r in method_results]
        p95s = [r.runtime_stats.get("p95_runtime", 0.0) for r in method_results]
        deadline_misses = [r.runtime_stats.get("deadline_misses", 0) for r in method_results]

        rows.append({
            "Method": method,
            "Mean Runtime [s]": f"{np.mean(runtimes):.4f}",
            "P95 Runtime [s]": f"{np.mean(p95s):.4f}",
            "Max Runtime [s]": f"{np.max(runtimes):.4f}" if runtimes else "N/A",
            "Deadline Misses": sum(deadline_misses),
        })

    return pd.DataFrame(rows)


def save_all_tables(
    results: list,
    output_dir: str,
    sensitivity_results: dict = None,
):
    """Generate and save all report tables to CSV files.

    Args:
        results: List of EpisodeResult objects.
        output_dir: Directory to save output.
        sensitivity_results: Optional sensitivity analysis results.
    """
    os.makedirs(output_dir, exist_ok=True)

    tables = {
        "core_metrics_summary": build_core_metrics_table(results),
        "failure_taxonomy_summary": build_failure_taxonomy_table(results),
        "ablation_comparison": build_ablation_comparison_table(results),
        "runtime_summary": build_runtime_summary_table(results),
    }

    for name, df in tables.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)

    if sensitivity_results:
        sens_df = build_sensitivity_summary_table(sensitivity_results)
        sens_df.to_csv(os.path.join(output_dir, "sensitivity_summary.csv"), index=False)

    return tables


# ---------------------------------------------------------------------------
# LaTeX table export
# ---------------------------------------------------------------------------

def to_latex_table(
    df,
    caption: str = "",
    label: str = "",
    column_format: str = None,
    fontsize: str = "\\small",
) -> str:
    """Convert a pandas DataFrame to a LaTeX ``tabular`` environment.

    Args:
        df: pandas DataFrame.
        caption: Table caption (wrapped in \\caption{}).
        label: LaTeX label (wrapped in \\label{}).
        column_format: LaTeX column format string (e.g. 'lcccc'). Auto-generated if None.
        fontsize: Font size command (e.g. '\\small', '\\footnotesize').

    Returns:
        LaTeX table source as a string.
    """
    n_cols = len(df.columns)
    if column_format is None:
        column_format = "l" + "c" * (n_cols - 1)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append(f"  {fontsize}")
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{column_format}}}")
    lines.append("    \\toprule")

    # Header
    header_cells = [f"\\textbf{{{col}}}" for col in df.columns]
    lines.append("    " + " & ".join(header_cells) + " \\\\")
    lines.append("    \\midrule")

    # Data rows
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            elif isinstance(val, (int, bool)):
                cells.append(str(val))
            else:
                # Escape LaTeX special characters
                escaped = str(val).replace("&", "\\&").replace("%", "\\%") \
                                 .replace("$", "\\$").replace("#", "\\#") \
                                 .replace("_", "\\_").replace("{", "\\{") \
                                 .replace("}", "\\}")
                cells.append(escaped)
        lines.append("    " + " & ".join(cells) + " \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def save_all_tables_latex(
    results: list,
    output_dir: str,
    sensitivity_results: dict = None,
):
    """Generate and save all report tables as .tex files for LaTeX inclusion.

    Args:
        results: List of EpisodeResult objects.
        output_dir: Directory to save .tex files.
        sensitivity_results: Optional sensitivity analysis results.

    Returns:
        dict mapping table name to file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    tables = {
        "core_metrics_summary": (
            build_core_metrics_table(results),
            "Core Metrics Summary (Mean $\\pm$ Std)",
            "tab:core_metrics",
        ),
        "failure_taxonomy_summary": (
            build_failure_taxonomy_table(results),
            "Failure Taxonomy Summary",
            "tab:failure_taxonomy",
        ),
        "ablation_comparison": (
            build_ablation_comparison_table(results),
            "Ablation Comparison",
            "tab:ablation",
        ),
        "runtime_summary": (
            build_runtime_summary_table(results),
            "Runtime Performance Summary",
            "tab:runtime",
        ),
    }

    outputs = {}
    for name, (df, caption, label) in tables.items():
        tex = to_latex_table(df, caption=caption, label=label)
        path = os.path.join(output_dir, f"{name}.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        outputs[name] = path

    if sensitivity_results:
        sens_df = build_sensitivity_summary_table(sensitivity_results)
        tex = to_latex_table(sens_df, caption="Sensitivity Analysis Summary",
                            label="tab:sensitivity")
        path = os.path.join(output_dir, "sensitivity_summary.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        outputs["sensitivity_summary"] = path

    return outputs
