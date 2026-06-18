"""Shallow water effect model.

Classifies water depth relative to vessel draught:
    lambda_h = h / T_i
    UKC = h - T_i

Depth/draught classification per PIANC and ITTC recommendations:
    lambda_h > 4.0 : deep water
    3.0 < lambda_h <= 4.0 : mild shallow
    2.0 < lambda_h <= 3.0 : moderate shallow
    1.5 < lambda_h <= 2.0 : strong shallow
    lambda_h <= 1.5 : high grounding risk / forbidden

References:
    - PIANC (2014). Report 121 — Harbour Approach Channels: Design
      Guidelines. The World Association for Waterborne Transport
      Infrastructure.
      → Section 4.2: Depth/draught ratio classification thresholds.

    - ITTC (2021). "Recommended Procedures and Guidelines: Manoeuvrability
      in Shallow Water." 29th ITTC, Manoeuvring Committee.
      → Shallow water classification criteria and test procedures.

    - Yoshimura, Y. (1986). "Mathematical model for manoeuvring ship
      motion in shallow water." J. Soc. Nav. Arch. Japan, 160, 49-58.
      → MMG-model shallow-water derivative corrections.

    - Zhang, C., Dhyani, A., Ringsberg, J.W., Thies, F., Reppa, V.,
      Negenborn, R.R. (2025). "Nonlinear Model Predictive Control for
      Path Following of Autonomous Inland Vessels in Confined
      Waterways." Ocean Engineering, 334, 121592.
      DOI: 10.1016/j.oceaneng.2025.121592
      → Application of shallow water + bank effects in inland NMPC.

The disturbance bound rho_shallow is a conservative estimate and does
not implement full MMG shallow-water corrections for hydrodynamic
derivatives. See ASSUMPTIONS.md Section 5 for limitations.
"""

from dataclasses import dataclass

import numpy as np

from ..constants import SHALLOW_DEEP, SHALLOW_MILD, SHALLOW_MODERATE, SHALLOW_STRONG


@dataclass
class ShallowWaterResult:
    """Output of shallow water assessment.

    Pathway tags (which part of the control pipeline each field feeds):
        enters_dynamics       — ``disturbance_bound`` (resistance force)
        enters_safety_margin  — NOT directly; UKC violation → P0 constraint
        enters_tube_radius    — ``rho_shallow``
        enters_cost           — ``I_shallow`` (penalty weight in MPC cost)
        metadata_only         — ``lambda_h``, ``ukc``, ``risk_level``
    """
    lambda_h: float               # depth/draught ratio  [metadata]
    ukc: float                     # under-keel clearance [m]  [metadata]
    ukc_violation: bool            # UKC < UKC_min  [→ P0 hard constraint]
    shallow_risk_level: str        # deep, mild, moderate, strong, forbidden  [metadata]
    I_shallow: float               # shallow indicator [0, 1]  [→ cost penalty]
    rho_shallow: float             # tube radius contribution [m]  [→ tube_radius]
    disturbance_bound: float       # max disturbance force magnitude bound [N]  [→ dynamics]


def assess_shallow_water(
    water_depth: float,
    vessel_draught: float,
    vessel_speed: float,
    vessel_length: float,
    U_ref: float = 7.0,
    ukc_min: float = 1.0,
    config: dict = None,
) -> ShallowWaterResult:
    """Assess shallow water risk and compute disturbance bound.

    Args:
        water_depth: Local water depth h [m].
        vessel_draught: Vessel draught T [m].
        vessel_speed: Current vessel speed [m/s].
        vessel_length: Vessel length L [m].
        U_ref: Reference speed [m/s].
        ukc_min: Minimum under-keel clearance [m].
        config: Optional parameter overrides.

    Returns:
        ShallowWaterResult with classification and disturbance bounds.
    """
    cfg = config or {}
    lambda_h = water_depth / max(vessel_draught, 0.1)
    ukc = water_depth - vessel_draught

    # Classification
    if lambda_h > SHALLOW_DEEP:
        risk_level = "deep"
        I_shallow = 0.0
    elif lambda_h > SHALLOW_MILD:
        risk_level = "mild_shallow"
        I_shallow = 0.25
    elif lambda_h > SHALLOW_MODERATE:
        risk_level = "moderate_shallow"
        I_shallow = 0.5
    elif lambda_h > SHALLOW_STRONG:
        risk_level = "strong_shallow"
        I_shallow = 0.75
    else:
        risk_level = "forbidden"
        I_shallow = 1.0

    # UKC violation
    ukc_violation = ukc < ukc_min

    # Tube radius contribution: rho_shallow increases as water gets shallower.
    # Capped at 50m to prevent the tube from exceeding typical channel widths.
    # The old formula (factor=2.0 × L × I) gave 360m for a 180m vessel in
    # moderate shallow water, making the MPC infeasible in most channels.
    rho_factor = cfg.get("rho_shallow_factor", 0.15)
    rho_shallow = min(
        rho_factor * I_shallow * vessel_length * (vessel_speed / U_ref),
        50.0,  # hard cap [m]
    )

    # Disturbance bound (conservative).
    # BUG #6 fix (2026-06-10): gamma_shallow_factor converts the dimensionless
    # coefficient into physical Newtons. Default 1.0 preserves backward compat.
    gamma_shallow = cfg.get("gamma_shallow", 0.1)
    # Default to physical force scale: 0.5 * rho_water * L * T * U_ref^2.
    # For L=180m, T=10.5m, U_ref=7 m/s: ≈ 4.73e7 N.
    _F_ref_default = 0.5 * 1025.0 * vessel_length * max(vessel_draught, 1.0) * U_ref**2
    gamma_shallow_factor = cfg.get("gamma_shallow_factor", _F_ref_default)
    disturbance_bound = (
        gamma_shallow * I_shallow
        * vessel_speed**2 / U_ref
        * vessel_length
        * gamma_shallow_factor
    )

    return ShallowWaterResult(
        lambda_h=lambda_h,
        ukc=ukc,
        ukc_violation=ukc_violation,
        shallow_risk_level=risk_level,
        I_shallow=I_shallow,
        rho_shallow=rho_shallow,
        disturbance_bound=disturbance_bound,
    )


def compute_shallow_water_disturbance(
    shallow_result: ShallowWaterResult,
    vessel_speed: float,
    U_ref: float = 7.0,
) -> np.ndarray:
    """Generate a conservative shallow water disturbance force vector.

    Returns [X, Y, N] in body frame (surge, sway, yaw moment).
    The disturbance is bounded by the shallow result's disturbance_bound.
    """
    if shallow_result.I_shallow < 1e-6:
        return np.zeros(3)

    mag = shallow_result.disturbance_bound
    # Simplified: resistance increase in surge, random sway
    return np.array([-mag, 0.0, 0.0])  # dominant effect is added resistance
