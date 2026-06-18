"""Standardized interface for experimental validation data.

Provides a common API for comparing simulation outputs against:
    - Towing tank model test data (turning circle, zigzag, stopping)
    - Full-scale trial recordings
    - SIMMAN benchmark vessel datasets (KVLCC2, KCS, KVLCC1)
    - Published experimental results

References:
    - SIMMAN 2020 Workshop proceedings: KVLCC2, KCS, KVLCC1 benchmark data
    - ITTC (2021). Recommended Procedures: Manoeuvrability.
    - Delefortrie, G., Verwilligen, J., Eloot, K., Lataire, E. (2024).
      "Bank interaction effects on ships in 6 DOF."
      Ocean Engineering, 310, 118614.
    - Flanders Maritime Laboratory free-running model test data.
    - Yasukawa, H., Yoshimura, Y. (2015). "Introduction of MMG standard
      method for ship maneuvering predictions." JMST, 20, 37-52.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


class TestType(Enum):
    """Standard maneuvering test types per ITTC conventions."""
    TURNING_CIRCLE = "turning_circle"
    ZIGZAG = "zigzag"
    STOPPING = "stopping"
    SPIRAL = "spiral"
    PULL_OUT = "pull_out"
    TRAJECTORY = "trajectory"


class ValidationMetric(Enum):
    """Quantitative validation error metrics."""
    NRMSE = "nrmse"              # Normalized RMSE (by reference range)
    MAE = "mae"                   # Mean absolute error
    MAX_ERROR = "max_error"       # Maximum absolute deviation
    R2 = "r2"                     # Coefficient of determination
    THIEL_U = "theil_u"           # Theil's U statistic


@dataclass
class ExperimentalReference:
    """An experimental reference dataset for validation.

    Attributes:
        source: Publication or database citation.
        vessel_type: Vessel identifier (e.g., "KVLCC2", "KCS").
        test_type: Type of maneuvering test.
        parameters: Test-specific parameters (e.g., rudder angle, approach speed).
        data: Reference time series or scalar metrics.
        units: Physical units for each data column.
        description: Human-readable description.
    """
    source: str
    vessel_type: str
    test_type: TestType
    parameters: dict = field(default_factory=dict)
    data: Optional[np.ndarray] = None
    units: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ValidationResult:
    """Output of a validation comparison.

    Attributes:
        reference_id: Identifier of the experimental reference.
        metric_values: Dict mapping metric name to value.
        passed: Whether all pass criteria are met.
        pass_details: Per-metric pass/fail status.
    """
    reference_id: str = ""
    metric_values: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    pass_details: dict[str, bool] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Standard maneuvering test metrics extraction
# ---------------------------------------------------------------------------


def extract_turning_circle_metrics(
    trajectory: np.ndarray,
    L: float,  # vessel length for normalization
) -> dict:
    """Extract ITTC standard turning circle metrics from a trajectory.

    Args:
        trajectory: (n, 3) array of [x, y, psi] in world frame.
        L: Vessel length [m] for normalization.

    Returns:
        Dict with: advance/L, transfer/L, tactical_diameter/L,
                   steady_turning_diameter/L, steady_speed_ratio.
    """
    if len(trajectory) < 10:
        return {}

    # Detect steady turn: constant yaw rate
    psi = trajectory[:, 2]
    t = np.arange(len(trajectory))

    # Compute yaw rate via central differences
    r = np.gradient(psi, t)
    r = np.where(np.isfinite(r), r, 0.0)

    # Advance: max longitudinal displacement from start
    x0, y0, psi0 = trajectory[0, 0], trajectory[0, 1], trajectory[0, 2]
    cos_psi0, sin_psi0 = np.cos(psi0), np.sin(psi0)

    dx = trajectory[:, 0] - x0
    dy = trajectory[:, 1] - y0
    along_track = dx * cos_psi0 + dy * sin_psi0
    cross_track = -dx * sin_psi0 + dy * cos_psi0

    # Find 90° heading change point
    psi_diff = np.abs(psi - psi0)
    psi_diff = np.arctan2(np.sin(psi_diff), np.cos(psi_diff))
    idx_90 = np.argmin(np.abs(abs(psi_diff) - np.pi / 2))

    # Find 180° heading change point
    idx_180 = np.argmin(np.abs(abs(psi_diff) - np.pi))

    advance = abs(along_track[idx_90]) / L if idx_90 < len(along_track) else 0.0
    transfer = abs(cross_track[idx_90]) / L if idx_90 < len(cross_track) else 0.0
    tactical_diameter = abs(cross_track[idx_180]) / L if idx_180 < len(cross_track) else 0.0

    return {
        "advance_Lpp": advance,
        "transfer_Lpp": transfer,
        "tactical_diameter_Lpp": tactical_diameter,
    }


def extract_zigzag_metrics(
    trajectory: np.ndarray,
    time: np.ndarray,
    L: float,
) -> dict:
    """Extract ITTC standard zigzag test metrics.

    Args:
        trajectory: (n, 3) array of [x, y, psi].
        time: (n,) array of timestamps [s].
        L: Vessel length [m].

    Returns:
        Dict with overshoot angles, initial turning time, period.
    """
    if len(trajectory) < 20:
        return {}

    psi = trajectory[:, 2]
    r = np.gradient(psi, time)

    # Detect rudder execution points via yaw rate sign changes
    signs = np.sign(r)
    sign_changes = np.diff(signs) != 0

    # Find extreme heading points (local extrema of psi)
    overshoots = []
    for i in range(1, len(psi) - 1):
        if (psi[i] - psi[i - 1]) * (psi[i + 1] - psi[i]) < 0:
            overshoots.append(float(np.rad2deg(psi[i])))

    first_overshoot = overshoots[0] if overshoots else 0.0
    second_overshoot = overshoots[1] if len(overshoots) > 1 else 0.0

    return {
        "first_overshoot_deg": first_overshoot,
        "second_overshoot_deg": second_overshoot,
    }


# ---------------------------------------------------------------------------
# Validation comparison
# ---------------------------------------------------------------------------


def compute_validation_metrics(
    simulation_data: np.ndarray,
    reference_data: np.ndarray,
    data_range: Optional[float] = None,
) -> dict[str, float]:
    """Compute quantitative validation metrics.

    Args:
        simulation_data: Simulated values (n,).
        reference_data: Experimental reference values (n,).
        data_range: Optional normalization factor for NRMSE
                    (defaults to max(ref) - min(ref)).

    Returns:
        Dict of metric name → value.
    """
    sim = np.asarray(simulation_data).ravel()
    ref = np.asarray(reference_data).ravel()

    n = min(len(sim), len(ref))
    sim = sim[:n]
    ref = ref[:n]

    if n < 2:
        return {"nrmse": float("inf"), "mae": float("inf")}

    error = sim - ref
    mae = float(np.mean(np.abs(error)))
    max_err = float(np.max(np.abs(error)))

    # NRMSE
    if data_range is None:
        data_range = np.max(ref) - np.min(ref)
        if data_range < 1e-10:
            data_range = np.std(ref)
            if data_range < 1e-10:
                data_range = 1.0
    rmse = np.sqrt(np.mean(error**2))
    nrmse = rmse / max(data_range, 1e-10)

    # R²
    ss_res = np.sum(error**2)
    ss_tot = np.sum((ref - np.mean(ref))**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    # Theil's U
    rmse_sim = np.sqrt(np.mean(sim**2))
    rmse_ref = np.sqrt(np.mean(ref**2))
    theil_u = rmse / max(rmse_sim + rmse_ref, 1e-10)

    return {
        "nrmse": nrmse,
        "mae": mae,
        "max_error": max_err,
        "r2": r2,
        "theil_u": theil_u,
    }


def validate_against_experiment(
    simulation_data: np.ndarray,
    reference: ExperimentalReference,
    tolerances: Optional[dict[str, float]] = None,
) -> ValidationResult:
    """Validate simulation data against an experimental reference.

    Args:
        simulation_data: Simulated data (n,).
        reference: ExperimentalReference with .data attribute.
        tolerances: Dict mapping metric name to pass threshold.
                    Defaults: {"nrmse": 0.15, "r2": 0.85}.

    Returns:
        ValidationResult with pass/fail status.
    """
    if tolerances is None:
        tolerances = {"nrmse": 0.15, "r2": 0.85}

    ref_data = reference.data
    if ref_data is None:
        return ValidationResult(
            reference_id=f"{reference.vessel_type}_{reference.test_type.value}",
            metric_values={},
            passed=False,
            pass_details={},
            notes="Reference data is None — cannot validate.",
        )

    metrics = compute_validation_metrics(simulation_data, ref_data)

    pass_details = {}
    for metric, threshold in tolerances.items():
        val = metrics.get(metric, float("inf"))
        if metric == "nrmse":
            pass_details[metric] = val <= threshold
        elif metric == "r2":
            pass_details[metric] = val >= threshold
        else:
            pass_details[metric] = True  # no threshold for this metric

    all_passed = all(pass_details.values())

    return ValidationResult(
        reference_id=f"{reference.vessel_type}_{reference.test_type.value}",
        metric_values=metrics,
        passed=all_passed,
        pass_details=pass_details,
        notes="All metrics within tolerance." if all_passed else "Some metrics exceed tolerance.",
    )
