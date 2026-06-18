"""Conservatism analysis for the additive tube radius model.

The tube radius in this framework is the sum of eight independent
components:
    rho_tube = rho_0 + rho_AIS + rho_current + rho_wind
             + rho_shallow + rho_bank + rho_ship + rho_maneuver

This additive formulation is *conservative by design*: it assumes all
disturbance components simultaneously attain their worst-case values
and align constructively. This module provides tools to:

1. Quantify the conservatism (tube utilization ratio)
2. Analyze component correlations (to justify RSS-based alternatives)
3. Estimate joint bounds via Monte Carlo
4. Calibrate reduction factors from empirical data

Key references:
    - Mayne, D. Q., Seron, M. M., & Raković, S. V. (2005).
      "Robust model predictive control of constrained linear systems with
      bounded disturbances." Automatica, 41(2), 219-224.
    - Zhang, C. et al. (2025).
      "Nonlinear Model Predictive Control for Path Following of Autonomous
      Inland Vessels in Confined Waterways." Ocean Engineering, 334, 121592.
    - Lee, H., Tran, H., & Kim, J. (2024).
      "Safety-Guaranteed Ship Berthing Using Cascade Tube-Based Model
      Predictive Control." IEEE TCST, 32(4), 1504-1511.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ConservatismAnalysis:
    """Results of conservatism analysis for the additive tube radius.

    Attributes:
        component_means: Mean value of each tube component over trajectory.
        component_stds: Standard deviation of each component.
        component_maxs: Maximum observed value of each component.
        sum_of_bounds: Σ ρ_i (current conservative method).
        rss_bound: sqrt(Σ ρ_i²) (independent-components alternative).
        joint_bound_99: 99th percentile of Monte Carlo joint distribution.
        joint_bound_95: 95th percentile of Monte Carlo joint distribution.
        reduction_factor: joint_bound_99 / sum_of_bounds.
        tube_utilization_mean: Mean(actual_error / total_tube).
        tube_utilization_p95: 95th percentile of tube utilization.
        correlation_matrix: Pearson correlation between components.
        is_correlated: Whether any |r| >= correlation_threshold.
    """

    component_means: dict[str, float] = field(default_factory=dict)
    component_stds: dict[str, float] = field(default_factory=dict)
    component_maxs: dict[str, float] = field(default_factory=dict)
    sum_of_bounds: float = 0.0
    rss_bound: float = 0.0
    joint_bound_99: float = 0.0
    joint_bound_95: float = 0.0
    reduction_factor: float = 1.0
    tube_utilization_mean: float = 0.0
    tube_utilization_p95: float = 0.0
    tube_utilization_p99: float = 0.0
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.eye(8))
    component_names: list[str] = field(default_factory=list)
    is_correlated: bool = False
    n_samples: int = 0
    rss_safety_factor: float = 1.0


# ---------------------------------------------------------------------------
# Component names
# ---------------------------------------------------------------------------

# ── Tube component keys in the new structured format (v2) ──────────────
# compute_tube_radius returns:
#   {"radius": total, "components": {key: float, ...}, "metadata": {...}}
# These are the keys inside "components" (legacy flat keys shown in comments).
TUBE_COMPONENT_NAMES = [
    "base_uncertainty",   # was rho_0
    "model_mismatch",     # was rho_maneuver
    "target_prediction",  # was rho_AIS
    "current",            # was rho_current
    "wind",               # was rho_wind
    "shallow_water",      # was rho_shallow
    "bank_effect",        # was rho_bank
    "ship_interaction",   # was rho_ship
]


# ---------------------------------------------------------------------------
# Tube utilization analysis
# ---------------------------------------------------------------------------


def tube_utilization_analysis(
    tube_component_history: list[dict],
    actual_error_history: list[float],
) -> ConservatismAnalysis:
    """Analyze tube utilization from simulation logs.

    For each timestep:
        actual_error_k = ||x_true_k - x_nominal_k||
        tube_bound_k = Σ rho_i,k
        utilization_k = actual_error_k / tube_bound_k

    A low utilization ratio indicates the additive bound is conservative.
    A utilization near 1.0 indicates the bound is tight.

    Args:
        tube_component_history: List of dicts from compute_tube_radius()
                                at each timestep, each with component keys
                                and a 'total' key.
        actual_error_history: List of actual ||x_true - x_nominal|| values
                              at each timestep.

    Returns:
        ConservatismAnalysis dataclass.
    """
    if not tube_component_history or not actual_error_history:
        return ConservatismAnalysis()

    n_steps = len(tube_component_history)
    actual = np.array(actual_error_history[:n_steps])

    # Identify component names from first entry.
    # Supports both new-style {"radius", "components": {...}, "metadata": {...}}
    # and legacy flat dict format.
    first = tube_component_history[0]
    if "components" in first:
        # New structured format
        component_names = list(first["components"].keys())
        n_comp = len(component_names)

        comp_series = np.zeros((n_steps, n_comp))
        totals = np.zeros(n_steps)
        for t, entry in enumerate(tube_component_history):
            comps = entry["components"]
            for j, name in enumerate(component_names):
                comp_series[t, j] = abs(comps.get(name, 0.0))
            totals[t] = entry.get("radius", np.sum(comp_series[t, :]))
    else:
        # Legacy flat format — maintain backward compatibility
        component_names = [k for k in first if k != "total"]
        n_comp = len(component_names)

        comp_series = np.zeros((n_steps, n_comp))
        totals = np.zeros(n_steps)
        for t, entry in enumerate(tube_component_history):
            for j, name in enumerate(component_names):
                comp_series[t, j] = abs(entry.get(name, 0.0))
            totals[t] = entry.get("total", np.sum(comp_series[t, :]))

    # Component statistics
    means = {name: float(np.mean(comp_series[:, j])) for j, name in enumerate(component_names)}
    stds = {name: float(np.std(comp_series[:, j])) for j, name in enumerate(component_names)}
    maxs = {name: float(np.max(comp_series[:, j])) for j, name in enumerate(component_names)}

    # Sum-of-bounds (current method) vs RSS
    sum_of_bounds_val = float(np.mean(totals))
    rss_val = float(np.sqrt(np.sum(np.mean(comp_series, axis=0) ** 2)))

    # Tube utilization
    # Guard against zero total
    safe_totals = np.where(totals < 1e-12, 1.0, totals)
    utilization = actual / safe_totals
    utilization = np.clip(utilization, 0.0, 10.0)  # clip outliers

    # Correlation matrix
    corr = np.eye(n_comp)
    if n_steps >= 10 and n_comp >= 2:
        try:
            corr = np.corrcoef(comp_series.T)
            corr[np.isnan(corr)] = 0.0
        except Exception:
            pass

    # Check for significant correlations (|r| >= 0.3)
    is_correlated = False
    off_diag_max = 0.0
    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            if abs(corr[i, j]) >= 0.3:
                is_correlated = True
                off_diag_max = max(off_diag_max, abs(corr[i, j]))

    # RSS safety factor: if components are independent, sqrt(1/n_comp)
    # is the theoretical reduction. We use a conservative multiplier.
    if not is_correlated:
        rss_safety_factor = 1.0 / np.sqrt(max(n_comp, 1))
    else:
        # With correlations, the RSS under-estimates. Use correlation-aware factor.
        # Based on average off-diagonal correlation inflating effective dimension
        rss_safety_factor = 1.0 / np.sqrt(max(n_comp * (1.0 - off_diag_max), 1))

    # Monte Carlo joint bound estimation
    jb_95, jb_99 = _monte_carlo_joint_bound_internal(comp_series, totals)

    return ConservatismAnalysis(
        component_means=means,
        component_stds=stds,
        component_maxs=maxs,
        sum_of_bounds=sum_of_bounds_val,
        rss_bound=rss_val,
        joint_bound_99=jb_99,
        joint_bound_95=jb_95,
        reduction_factor=jb_99 / max(sum_of_bounds_val, 1e-12),
        tube_utilization_mean=float(np.mean(utilization)),
        tube_utilization_p95=float(np.percentile(utilization, 95)),
        tube_utilization_p99=float(np.percentile(utilization, 99)),
        correlation_matrix=corr,
        component_names=component_names,
        is_correlated=is_correlated,
        n_samples=n_steps,
        rss_safety_factor=min(rss_safety_factor, 1.0),
    )


def _monte_carlo_joint_bound_internal(
    comp_series: np.ndarray,
    totals: np.ndarray,
) -> tuple[float, float]:
    """Internal: Estimate joint bound percentiles from empirical data.

    Draws from the empirical joint distribution via bootstrapping and
    computes the distribution of total disturbance magnitudes.

    Args:
        comp_series: (n_steps, n_comp) component time series.
        totals: (n_steps,) total tube radius time series.

    Returns:
        (p95_bound, p99_bound): 95th and 99th percentile joint bounds.
    """
    n_steps, n_comp = comp_series.shape

    if n_steps < 10:
        # Not enough data: return sum-of-bounds as conservative estimate
        s = float(np.mean(totals))
        return s, s

    # Bootstrap: resample with replacement
    n_bootstrap = min(5000, n_steps * 10)
    rng = np.random.RandomState(42)

    # Independent sampling per component (assumes statistical independence
    # between tube components). When components are significantly correlated
    # (|r| >= 0.3), this systematically underestimates the joint bound.
    # The caller (tube_utilization_analysis) reports the correlation matrix
    # so users can assess whether the independence assumption holds.
    boot_independent = np.zeros(n_bootstrap)
    for j in range(n_comp):
        boot_independent += rng.choice(comp_series[:, j], size=n_bootstrap)

    p95 = float(np.percentile(boot_independent, 95))
    p99 = float(np.percentile(boot_independent, 99))

    return p95, p99


def monte_carlo_joint_bound(
    component_distributions: dict[str, Callable[[int], np.ndarray]],
    n_samples: int = 100_000,
    confidence: float = 0.99,
) -> float:
    """Estimate joint disturbance bound via Monte Carlo.

    Instead of assuming all disturbances simultaneously achieve their
    worst-case values (sum of bounds), this method samples from the joint
    distribution of disturbance components and computes the empirical
    confidence interval.

    This is the preferred method when detailed component distribution
    data is available (e.g., from AIS-derived per-component statistics).

    Args:
        component_distributions: Dict mapping component name to a
            callable(n) that returns n i.i.d. samples.
        n_samples: Number of Monte Carlo samples.
        confidence: Confidence level (e.g., 0.99 for 99% bound).

    Returns:
        float: The joint bound at the specified confidence level.
    """
    if not component_distributions:
        return 0.0

    samples = np.zeros(n_samples)
    for dist_fn in component_distributions.values():
        samples += dist_fn(n_samples)

    return float(np.percentile(samples, confidence * 100))


def component_correlation_analysis(
    tube_component_history: list[dict],
    component_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute Pearson correlation matrix between tube components.

    If components are uncorrelated (|r| < 0.3), the RSS bound
        sqrt(Σ ρ_i²)
    provides a tighter, still statistically justified alternative to the
    sum-of-bounds. This is because:
        Var(Σ X_i) = Σ Var(X_i) + 2 Σ_{i<j} Cov(X_i, X_j)
    and for independent components, Cov = 0, giving:
        SD(Σ X_i) = sqrt(Σ SD(X_i)²)

    Args:
        tube_component_history: List of per-timestep component dicts.
        component_names: Optional list of component keys to analyze
                         (defaults to all non-'total' keys).

    Returns:
        (corr_matrix, summary): Correlation matrix and summary dict with
            'off_diagonal_max', 'off_diagonal_mean_abs',
            'is_uncorrelated', 'num_significant_pairs'.
    """
    if not tube_component_history:
        return np.eye(1), {"off_diagonal_max": 0.0, "is_uncorrelated": True}

    first = tube_component_history[0]
    if component_names is None:
        if "components" in first:
            component_names = list(first["components"].keys())
        else:
            component_names = [k for k in first if k != "total"]

    n_comp = len(component_names)
    if n_comp < 2:
        return np.eye(n_comp), {"off_diagonal_max": 0.0, "is_uncorrelated": True}

    # Build component matrix
    n_steps = len(tube_component_history)
    comp_matrix = np.zeros((n_steps, n_comp))
    for t, entry in enumerate(tube_component_history):
        comps = entry.get("components", entry)  # new nested or legacy flat
        for j, name in enumerate(component_names):
            comp_matrix[t, j] = abs(comps.get(name, 0.0))

    # Compute correlation
    try:
        corr = np.corrcoef(comp_matrix.T)
        corr[np.isnan(corr)] = 0.0
    except Exception:
        corr = np.eye(n_comp)

    # Summary statistics
    off_diag_abs = []
    sig_pairs = 0
    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            r = abs(corr[i, j])
            off_diag_abs.append(r)
            if r >= 0.3:
                sig_pairs += 1

    off_diag_max = max(off_diag_abs) if off_diag_abs else 0.0
    off_diag_mean = np.mean(off_diag_abs) if off_diag_abs else 0.0

    summary = {
        "off_diagonal_max": off_diag_max,
        "off_diagonal_mean_abs": off_diag_mean,
        "is_uncorrelated": off_diag_max < 0.3,
        "num_significant_pairs": sig_pairs,
        "num_pairs": len(off_diag_abs),
    }

    return corr, summary


def calibrate_reduction_factor(
    tube_component_history: list[dict],
    actual_error_history: list[float],
    confidence_level: float = 0.99,
) -> ConservatismAnalysis:
    """Calibrate the conservatism reduction factor from empirical data.

    Full pipeline combining utilization analysis with correlation-based
    RSS assessment. This is the recommended entry point for quantitative
    conservatism reporting.

    Args:
        tube_component_history: Per-timestep tube component dicts.
        actual_error_history: Per-timestep actual deviation magnitudes.
        confidence_level: Confidence level for joint bound.

    Returns:
        ConservatismAnalysis with full diagnostics.
    """
    result = tube_utilization_analysis(
        tube_component_history,
        actual_error_history,
    )

    # Augment with correlation analysis
    corr, summary = component_correlation_analysis(
        tube_component_history,
        result.component_names,
    )
    result.correlation_matrix = corr
    result.is_correlated = not summary.get("is_uncorrelated", False)

    return result


def summarize_conservatism(result: ConservatismAnalysis) -> str:
    """Generate a human-readable conservatism summary string.

    Args:
        result: ConservatismAnalysis from calibrate_reduction_factor().

    Returns:
        Multi-line summary string suitable for paper discussion section.
    """
    lines = []
    lines.append("─" * 60)
    lines.append("TUBE CONSERVATISM ANALYSIS")
    lines.append("─" * 60)
    lines.append(f"  Samples analyzed:           {result.n_samples}")
    lines.append(f"  Sum-of-bounds (additive):   {result.sum_of_bounds:.2f} m")
    lines.append(f"  RSS bound (independent):    {result.rss_bound:.2f} m")
    lines.append(f"  Joint bound (95% MC):       {result.joint_bound_95:.2f} m")
    lines.append(f"  Joint bound (99% MC):       {result.joint_bound_99:.2f} m")
    lines.append(f"  Reduction factor:           {result.reduction_factor:.3f}")
    lines.append(f"  Tube utilization (mean):    {result.tube_utilization_mean:.3f}")
    lines.append(f"  Tube utilization (P95):     {result.tube_utilization_p95:.3f}")
    lines.append(f"  Tube utilization (P99):     {result.tube_utilization_p99:.3f}")
    lines.append(f"  Components correlated:      {result.is_correlated}")
    lines.append(f"  RSS safety factor:          {result.rss_safety_factor:.3f}")
    lines.append("-" * 60)
    lines.append("Component breakdown (mean ± std / max):")
    for name in result.component_names:
        mean_v = result.component_means.get(name, 0)
        std_v = result.component_stds.get(name, 0)
        max_v = result.component_maxs.get(name, 0)
        lines.append(f"  {name:20s} {mean_v:7.2f} ± {std_v:6.2f} / {max_v:7.2f} m")
    lines.append("─" * 60)

    # Interpretation
    lines.append("Interpretation:")
    if result.tube_utilization_mean < 0.3:
        lines.append(
            "  ✓ Low mean utilization — additive bound is conservative,"
            " which is intentional for safety."
        )
    elif result.tube_utilization_mean < 0.5:
        lines.append(
            "  ~ Moderate utilization — tube bound reasonably balanced"
            " between conservatism and performance."
        )
    else:
        lines.append(
            "  ⚠ High utilization — tube bound may be insufficient for"
            " worst-case disturbance scenarios."
        )

    if not result.is_correlated:
        lines.append(
            "  ✓ Components are largely uncorrelated — an RSS-based"
            f" fusion (≈{result.reduction_factor:.1%} reduction) is"
            " statistically justified under independence assumptions."
        )
    else:
        lines.append(
            "  ~ Components show significant correlation — the additive"
            " formulation is the safe default. Correlation-aware"
            " covariance inflation could reduce conservatism."
        )

    return "\n".join(lines)
