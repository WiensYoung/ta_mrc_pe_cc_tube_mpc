"""Automated statistical analysis pipeline.

Runs the complete statistical comparison workflow:
1. Pairwise method comparisons (t-test + Wilcoxon)
2. Multiple comparison corrections (Holm-Bonferroni + Benjamini-Hochberg)
3. Effect sizes (Cohen's d + Cliff's delta)
4. Cluster bootstrap confidence intervals
5. Mixed-effects model
6. Summary report generation

Produces a comprehensive statistics report suitable for journal publication.
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .statistics import (
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


def run_full_statistical_analysis(
    results: list,  # list[EpisodeResult]
    metrics: list[str] = None,
    alpha: float = 0.05,
    reference_method: str = "Proposed",
    output_path: str = None,
) -> dict:
    """Run the complete statistical analysis pipeline.

    Args:
        results: List of EpisodeResult objects.
        metrics: Metrics to analyze (default: key safety/performance metrics).
        alpha: Significance level.
        reference_method: Method to use as reference for comparisons.
        output_path: Optional path to save the report CSV.

    Returns:
        Nested dict with all statistical results.
    """
    if metrics is None:
        metrics = [
            "min_dcpa", "mean_dcpa", "worst_5_percent_dcpa",
            "ship_domain_violation_rate", "collision_rate",
            "mean_runtime", "p95_runtime",
            "rudder_saturation_rate", "cbf_intervention_rate",
            "fallback_trigger_rate", "ais_uncertainty_robustness",
        ]

    # Group results by method and scenario
    methods = sorted(set(r.method for r in results))
    scenarios = sorted(set(r.scenario_id for r in results))

    report = {
        "methods": methods,
        "scenarios": scenarios,
        "metrics": metrics,
        "reference_method": reference_method,
        "alpha": alpha,
        "descriptive": {},
        "pairwise_comparisons": {},
        "effect_sizes": {},
        "multiple_correction": {},
        "cluster_bootstrap": {},
        "mixed_effects": {},
    }

    # 1. Descriptive statistics per method per metric
    for metric in metrics:
        report["descriptive"][metric] = {}
        for method in methods:
            values = _extract_metric(results, method, metric)
            if len(values) > 0:
                report["descriptive"][metric][method] = compute_summary_statistics(values)

    # 2. Pairwise comparisons: reference vs all others
    for metric in metrics:
        ref_values = _extract_metric(results, reference_method, metric)
        report["pairwise_comparisons"][metric] = {}

        for method in methods:
            if method == reference_method:
                continue
            method_values = _extract_metric(results, method, metric)
            paired = _align_paired(ref_values, results, reference_method, method, metric)

            if len(paired["ref"]) >= 5:
                report["pairwise_comparisons"][metric][method] = {
                    "ttest": paired_ttest(paired["ref"], paired["method"]),
                    "wilcoxon": wilcoxon_test(paired["ref"], paired["method"]),
                    "cohens_d": cohens_d(paired["ref"], paired["method"]),
                    "cliffs_delta": cliffs_delta(paired["ref"], paired["method"]),
                }

    # 3. Multiple comparison correction (across all metrics)
    all_pvalues = []
    pvalue_labels = []
    for metric in metrics:
        for method in methods:
            if method == reference_method:
                continue
            comp = report["pairwise_comparisons"].get(metric, {}).get(method, {})
            ttest = comp.get("ttest", {})
            if "p_value" in ttest:
                all_pvalues.append(ttest["p_value"])
                pvalue_labels.append(f"{metric}:{method}")

    if all_pvalues:
        holm_result = holm_bonferroni(all_pvalues, alpha)
        bh_result = benjamini_hochberg(all_pvalues, alpha)

        report["multiple_correction"] = {
            "holm_bonferroni": {label: sig for label, sig in zip(pvalue_labels, holm_result)},
            "benjamini_hochberg": {label: sig for label, sig in zip(pvalue_labels, bh_result)},
        }

    # 4. Cluster bootstrap (by scenario)
    for metric in metrics:
        report["cluster_bootstrap"][metric] = {}
        for method in methods:
            values = _extract_metric(results, method, metric)
            clusters = _extract_cluster(results, method)
            if len(values) >= 10 and len(np.unique(clusters)) >= 3:
                bs = cluster_bootstrap(
                    values, clusters,
                    statistic_fn=lambda x: np.mean(x),
                    n_bootstrap=200,
                )
                report["cluster_bootstrap"][metric][method] = bs

    # 5. Mixed-effects model
    df = _build_model_df(results, metrics, scenarios)
    if df is not None and len(df) > 10:
        for metric in metrics[:3]:
            try:
                me = mixed_effects_model_interface(
                    df, f"{metric} ~ Method + scenario_id", "1|Seed",
                )
                report["mixed_effects"][metric] = me
            except Exception:
                report["mixed_effects"][metric] = {"converged": False, "error": "model_failed"}

    # Save report
    if output_path:
        _save_report_csv(report, output_path)

    return report


def _extract_metric(results, method, metric):
    """Extract metric values for a specific method.

    BUG #9 fix (2026-06-10): Handle None values (from _safe_float conversion
    of NaN/Inf) gracefully instead of crashing on ``np.isnan(None)``.
    """
    vals = []
    for r in results:
        if r.method == method:
            v = r.metrics.get(metric, np.nan)
            # Skip None, NaN, and Inf values
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(fv) or np.isinf(fv):
                continue
            vals.append(fv)
    return np.array(vals)


def _extract_cluster(results, method):
    """Extract scenario IDs as cluster labels using deterministic hash.

    Uses MD5-based hashing to ensure reproducibility across Python runs,
    independent of PYTHONHASHSEED.
    """
    import hashlib
    clusters = []
    for i, r in enumerate(results):
        if r.method == method:
            clusters.append(int(hashlib.md5(str(r.scenario_id).encode("utf-8")).hexdigest()[:8], 16) % 10000)
    return np.array(clusters)


def _align_paired(ref_values, results, ref_method, method, metric):
    """Build paired arrays by matching episodes across methods for a specific metric."""
    ref_by_ep = {}
    met_by_ep = {}

    for r in results:
        key = (r.scenario_id, r.seed)
        v = r.metrics.get(metric, np.nan)
        if v is None or (isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v))):
            continue
        if r.method == ref_method:
            ref_by_ep[key] = float(v)
        elif r.method == method:
            met_by_ep[key] = float(v)

    common_keys = set(ref_by_ep.keys()) & set(met_by_ep.keys())
    ref_arr = np.array([ref_by_ep[k] for k in sorted(common_keys)])
    met_arr = np.array([met_by_ep[k] for k in sorted(common_keys)])

    return {"ref": ref_arr, "method": met_arr}


def _build_model_df(results, metrics, scenarios):
    """Build a pandas DataFrame for mixed-effects modeling.

    Rows with NaN/Inf metrics are skipped (rather than zero-filled)
    to avoid contaminating model estimates with fabricated safe values.
    Callers should check the returned DataFrame length against the
    expected number of rows.
    """
    import warnings
    rows = []
    skipped = 0
    for r in results:
        row = {"Method": r.method, "scenario_id": r.scenario_id,
               "Seed": r.seed, "Episode": r.episode_id}
        has_nan_inf = False
        for m in metrics:
            v = r.metrics.get(m, np.nan)
            if v is None or (isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v))):
                has_nan_inf = True
                break
            row[m] = float(v)
        if has_nan_inf:
            skipped += 1
            continue
        rows.append(row)
    if skipped > 0:
        warnings.warn(
            f"Mixed-effects model: skipped {skipped} rows with NaN/Inf metrics "
            f"out of {len(results)} total results. These may indicate "
            f"method failures or incomplete episodes."
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def _save_report_csv(report: dict, path: str):
    """Save key statistical results to CSV for inclusion in publications."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    rows = []
    for metric, methods_dict in report.get("pairwise_comparisons", {}).items():
        for method, tests in methods_dict.items():
            row = {
                "Metric": metric,
                "Comparison": f"{report['reference_method']} vs {method}",
            }
            ttest = tests.get("ttest", {})
            wilcox = tests.get("wilcoxon", {})
            row["t_statistic"] = ttest.get("statistic", "")
            row["t_pvalue"] = ttest.get("p_value", "")
            row["mean_diff"] = ttest.get("mean_diff", "")
            row["wilcoxon_pvalue"] = wilcox.get("p_value", "")
            row["cohens_d"] = tests.get("cohens_d", "")
            row["cliffs_delta"] = tests.get("cliffs_delta", "")
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def generate_stats_summary_markdown(report: dict) -> str:
    """Generate a markdown-formatted statistical summary for the paper.

    Args:
        report: Output from run_full_statistical_analysis.

    Returns:
        Markdown string.
    """
    lines = ["# Statistical Analysis Summary\n"]
    ref = report["reference_method"]

    # Descriptive table
    lines.append("## Descriptive Statistics\n")
    for metric, methods_data in report.get("descriptive", {}).items():
        lines.append(f"### {metric}\n")
        lines.append("| Method | Mean | Std | Median | CI 95% | Worst 5% | N |")
        lines.append("|--------|------|-----|--------|--------|----------|---|")
        for method, stats in methods_data.items():
            lines.append(
                f"| {method} | {stats.get('mean', 0):.3f} | {stats.get('std', 0):.3f} "
                f"| {stats.get('median', 0):.3f} "
                f"| [{stats.get('ci_95_lower', 0):.3f}, {stats.get('ci_95_upper', 0):.3f}] "
                f"| {stats.get('worst_5_percent', 0):.3f} | {stats.get('n', 0)} |"
            )
        lines.append("")

    # Pairwise significance
    lines.append("## Pairwise Comparisons vs Proposed\n")
    for metric, comps in report.get("pairwise_comparisons", {}).items():
        lines.append(f"### {metric}\n")
        lines.append("| Method | Mean Diff | t-test p | Wilcoxon p | Cohen's d | Cliff's δ |")
        lines.append("|--------|-----------|----------|------------|-----------|-----------|")
        for method, tests in comps.items():
            ttest = tests.get("ttest", {})
            wilcox = tests.get("wilcoxon", {})
            lines.append(
                f"| {method} | {ttest.get('mean_diff', 0):.3f} "
                f"| {ttest.get('p_value', 1):.4f} | {wilcox.get('p_value', 1):.4f} "
                f"| {tests.get('cohens_d', 0):.3f} | {tests.get('cliffs_delta', 0):.3f} |"
            )
        lines.append("")

    # Multiple correction summary
    lines.append("## Multiple Comparison Correction\n")
    mc = report.get("multiple_correction", {})
    hb = mc.get("holm_bonferroni", {})
    bh = mc.get("benjamini_hochberg", {})
    if hb:
        n_sig_hb = sum(1 for v in hb.values() if v)
        n_sig_bh = sum(1 for v in bh.values() if v)
        lines.append(f"- Holm-Bonferroni: {n_sig_hb}/{len(hb)} comparisons significant")
        lines.append(f"- Benjamini-Hochberg: {n_sig_bh}/{len(bh)} comparisons significant\n")

    return "\n".join(lines)
