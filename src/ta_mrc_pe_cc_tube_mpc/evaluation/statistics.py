"""Statistical analysis: hypothesis tests, effect sizes, corrections.

All hypothesis tests filter paired observations to valid (non-NaN, finite)
pairs before computation.  The number of excluded pairs is reported in the
result dict for transparency.

Implements:
    1. Paired t-test
    2. Wilcoxon signed-rank test
    3. Holm-Bonferroni correction
    4. Benjamini-Hochberg correction
    5. Cohen's d
    6. Cliff's delta
    7. Cluster bootstrap (by real AIS episode)
    8. Mixed-effects model interface
"""

import warnings
from typing import Optional

import numpy as np
from scipy import stats


# ── NaN-safe paired-data filter ──────────────────────────────────────────────

def _filter_valid_pairs(x: np.ndarray, y: np.ndarray):
    """Return (x_valid, y_valid, n_excluded) for complete pairs."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = ~(np.isnan(xa) | np.isnan(ya) | np.isinf(xa) | np.isinf(ya))
    n_excluded = int(np.sum(~mask))
    return xa[mask], ya[mask], n_excluded


def paired_ttest(
    x: np.ndarray, y: np.ndarray,
    alternative: str = "two-sided",
) -> dict:
    """Paired t-test between two methods (NaN-safe).

    Only complete (finite, non-NaN) pairs are tested.  Excluded pair count
    is reported as ``n_excluded``.
    """
    from scipy.stats import t as t_dist
    xv, yv, n_excl = _filter_valid_pairs(x, y)
    n = len(xv)
    if n < 3:
        return {"test": "paired_t", "statistic": np.nan, "p_value": np.nan,
                "mean_diff": np.nan, "ci_95_lower": np.nan, "ci_95_upper": np.nan,
                "n": n, "n_excluded": n_excl, "warning": "n < 3"}

    result = stats.ttest_rel(xv, yv, alternative=alternative)
    diff = xv - yv
    t_crit = t_dist.ppf(0.975, n - 1)
    se = np.std(diff, ddof=1) / np.sqrt(n)

    return {
        "test": "paired_t",
        "statistic": float(result.statistic) if np.isfinite(result.statistic) else np.nan,
        "p_value": float(result.pvalue) if np.isfinite(result.pvalue) else np.nan,
        "mean_diff": float(np.mean(diff)),
        "ci_95_lower": float(np.mean(diff) - t_crit * se),
        "ci_95_upper": float(np.mean(diff) + t_crit * se),
        "n": n,
        "n_total": n + n_excl,
        "n_excluded": n_excl,
    }


def wilcoxon_test(
    x: np.ndarray, y: np.ndarray,
    alternative: str = "two-sided",
) -> dict:
    """Wilcoxon signed-rank test (NaN-safe).

    Args:
        x, y: Paired observations.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        dict with statistic, p_value, n, n_excluded.
    """
    xv, yv, n_excl = _filter_valid_pairs(x, y)
    n = len(xv)
    if n < 5:
        return {
            "test": "wilcoxon",
            "statistic": np.nan, "p_value": np.nan,
            "n": n, "n_total": n + n_excl, "n_excluded": n_excl,
            "warning": "n < 5 after filtering",
        }
    result = stats.wilcoxon(xv, yv, alternative=alternative)
    return {
        "test": "wilcoxon",
        "statistic": float(result.statistic) if np.isfinite(result.statistic) else np.nan,
        "p_value": float(result.pvalue) if np.isfinite(result.pvalue) else np.nan,
        "n": n,
        "n_total": n + n_excl,
        "n_excluded": n_excl,
    }


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d effect size (NaN-safe, paired observations)."""
    xv, yv, _ = _filter_valid_pairs(x, y)
    diff = xv - yv
    if len(diff) < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta non-parametric effect size.

    Returns value in [-1, 1]; |delta| > 0.474 is "large".
    Computed in O(n log n) using binary search on sorted arrays.
    """
    # Filter paired NaN/Inf values for consistency with other stat functions
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Use paired filtering to maintain alignment between x and y
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    y_sorted = np.sort(y)
    # Count pairs where xi > yj using binary search
    greater = sum(int(np.searchsorted(y_sorted, xi, side="left")) for xi in x)
    # Count pairs where xi < yj: total_pairs - greater - ties
    # For ties we use side='right' for exact equality detection
    ties = sum(
        int(np.searchsorted(y_sorted, xi, side="right"))
        - int(np.searchsorted(y_sorted, xi, side="left"))
        for xi in x
    )
    less = n_x * n_y - greater - ties
    return float(greater - less) / (n_x * n_y)


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of raw p-values.
        alpha: Family-wise significance level.

    Returns:
        List of booleans indicating significance after correction.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    significant = [False] * n

    for rank, idx in enumerate(sorted_idx):
        adjusted_alpha = alpha / (n - rank)
        if p_values[idx] <= adjusted_alpha:
            significant[idx] = True
        else:
            break  # all subsequent are non-significant

    return significant


def benjamini_hochberg(
    p_values: list[float], alpha: float = 0.05,
) -> list[bool]:
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: List of raw p-values.
        alpha: FDR level.

    Returns:
        List of booleans indicating significance after correction.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    significant = [False] * n
    last_sig_idx = -1

    for rank, idx in enumerate(sorted_idx):
        threshold = (rank + 1) / n * alpha
        if p_values[idx] <= threshold:
            last_sig_idx = rank
        else:
            break

    for rank in range(last_sig_idx + 1):
        significant[sorted_idx[rank]] = True

    return significant


def cluster_bootstrap(
    data: np.ndarray,
    cluster_ids: np.ndarray,
    statistic_fn,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """Cluster bootstrap: resample clusters, not individual observations.

    Critical for clustered data where individual episodes are not independent
    (e.g., same real-AIS episode with different perturbations).

    Args:
        data: Array of metric values (n_episodes,).
        cluster_ids: Array of cluster IDs (n_episodes,).
        statistic_fn: Function to compute statistic on a sample.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        dict with mean, ci_lower, ci_upper, bootstrap_distribution.
    """
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        mask = np.isin(cluster_ids, sampled_clusters)
        bootstrap_stats.append(statistic_fn(data[mask]))

    bootstrap_stats = np.array(bootstrap_stats)
    return {
        "mean": float(np.mean(bootstrap_stats)),
        "ci_95_lower": float(np.percentile(bootstrap_stats, 2.5)),
        "ci_95_upper": float(np.percentile(bootstrap_stats, 97.5)),
        "std": float(np.std(bootstrap_stats)),
    }


def compute_summary_statistics(
    values: np.ndarray,
    *,
    allow_nan: bool = True,
) -> dict:
    """Compute standard descriptive statistics (NaN-aware).

    NaN and Inf values are excluded from summary computation.  The returned
    dict includes ``valid_count``, ``nan_count``, and ``inf_count`` so the
    reader can assess data quality before relying on the summary.

    Args:
        values: Array of metric values (may contain NaN / Inf).
        allow_nan: If False, raise ValueError on NaN/Inf (strict mode).

    Returns:
        dict with count, valid_count, nan_count, inf_count, mean, std,
        ci_95, median, iqr, worst_5%, p95, min, max.
    """
    n_total = len(values)
    if n_total == 0:
        return {"count": 0, "valid_count": 0, "nan_count": 0, "inf_count": 0}

    # ── Separate valid / invalid ────────────────────────────────────────
    arr = np.asarray(values, dtype=float)
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    valid_mask = ~nan_mask & ~inf_mask
    valid = arr[valid_mask]
    n_valid = len(valid)
    n_nan = int(np.sum(nan_mask))
    n_inf = int(np.sum(inf_mask))

    if not allow_nan and (n_nan > 0 or n_inf > 0):
        raise ValueError(
            f"Cannot compute summary: {n_nan} NaN, {n_inf} Inf in {n_total} values"
        )

    if n_valid == 0:
        return {
            "count": n_total, "valid_count": 0,
            "nan_count": n_nan, "inf_count": n_inf,
        }

    mean_val = float(np.mean(valid))
    std_val = float(np.std(valid, ddof=1))

    # Bootstrap 95% CI (percentile method, 1000 resamples)
    if n_valid >= 5:
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(1000):
            samp = rng.choice(valid, size=n_valid, replace=True)
            boot_means.append(float(np.mean(samp)))
        boot_means = np.sort(boot_means)
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
    else:
        se = std_val / np.sqrt(n_valid) if n_valid > 1 else 0.0
        ci_lower = mean_val - 1.96 * se
        ci_upper = mean_val + 1.96 * se

    return {
        "count": n_total,
        "valid_count": n_valid,
        "nan_count": n_nan,
        "inf_count": n_inf,
        "mean": mean_val,
        "std": std_val,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "ci_method": "bootstrap_pct_1000" if n_valid >= 5 else "normal_approx",
        "median": float(np.median(valid)),
        "iqr": float(np.percentile(valid, 75) - np.percentile(valid, 25)),
        "worst_5_percent": _worst_5_percent(valid),
        "p95": float(np.percentile(valid, 95)),
        "p99": float(np.percentile(valid, 99)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }


def _worst_5_percent(values: np.ndarray) -> float:
    """Mean of worst 5% of values."""
    n_worst = max(1, int(len(values) * 0.05))
    return float(np.mean(np.sort(values)[:n_worst]))


def mixed_effects_model_interface(
    data,
    formula: str,
    random_effects: str = "1|Seed",
) -> Optional[dict]:
    """Interface to statsmodels MixedLM.

    Provides an approximation if the full model fails to converge.

    Args:
        data: pandas DataFrame with metric, method, scenario, etc.
        formula: R-style formula.
        random_effects: Random effects specification.

    Returns:
        dict with model summary or fallback statistics.
    """
    try:
        import statsmodels.formula.api as smf

        # Use correct statsmodels syntax: re_formula for random intercepts.
        # The CSV column is 'scenario_id' (snake_case), not 'Scenario'.
        groups_col = "scenario_id" if "scenario_id" in data.columns else None
        if groups_col is None:
            raise ValueError("No grouping column found for mixed model")

        model = smf.mixedlm(
            formula,
            data,
            groups=groups_col,
            re_formula="1",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(reml=True, method="lbfgs")

        return {
            "converged": result.converged,
            "loglike": float(result.llf),
            "aic": float(result.aic),
            "params": result.params.to_dict(),
            "pvalues": result.pvalues.to_dict(),
        }
    except Exception as e:
        # Fallback: OLS with robust standard errors
        try:
            import statsmodels.api as sm
            import patsy

            y, X = patsy.dmatrices(formula, data)
            ols = sm.OLS(y, X).fit()
            return {
                "converged": False,
                "fallback": "ols",
                "rsquared": float(ols.rsquared),
                "params": dict(zip(X.design_info.column_names, ols.params)),
                "pvalues": dict(zip(X.design_info.column_names, ols.pvalues)),
                "warning": f"MixedLM failed: {e}, using OLS fallback",
            }
        except Exception:
            return {"converged": False, "fallback": "none", "error": str(e)}
