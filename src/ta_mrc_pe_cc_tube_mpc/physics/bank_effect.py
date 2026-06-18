"""Bank effect model.

Conservative bounded disturbance based on published ship-bank interaction data:
    ||w_bank|| <= gamma_b * (U_i/U_ref)^2 * (d_ref/max(d_bank, d_min))^2

References:
    - Delefortrie, G., Verwilligen, J., Eloot, K., Lataire, E. (2024).
      "Bank interaction effects on ships in 6 DOF."
      Ocean Engineering, 310, 118614.
      DOI: 10.1016/j.oceaneng.2024.118614
      → Mathematical formulations for all 6 DOF based on 10,000+ towing tank tests.

    - Vantorre, M., Delefortrie, G., Eloot, K., Lataire, E. (2017).
      "Experimental study of ship-bank interaction."
      → Empirically fitted coefficients for suction-moment reversal.

    - Lataire, E., Vantorre, M., Delefortrie, G. (2012).
      "A prediction method for squat in restricted and unrestricted
      rectangular fairways." Ocean Engineering, 55, 71-80.
      → Semi-empirical squat and bank effect prediction.

Our model uses a conservative envelope of the Vantorre/Lataire bounds,
with gamma_b0 = 0.05 selected as the 95th percentile of published
coefficients (see ASSUMPTIONS.md Section 2 for full details).

NOT a precision CFD model — this is a conservative bounded-disturbance
approach suitable for robust control design and comparative evaluation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BankEffectResult:
    """Output of bank effect assessment.

    Pathway tags:
        enters_dynamics       — ``bank_disturbance_bound`` (lateral suction force [N])
        enters_safety_margin  — ``bank_safety_margin`` (geometric constraint)
        enters_tube_radius    — ``rho_bank``
        enters_cost           — violation → MPC bank penalty
        metadata_only         — ``gamma_b``, ``k_b``, ``d_bank_effective``
    """
    gamma_b: float                  # dimensionless coefficient  [metadata]
    k_b: float                     # tube radius coefficient  [metadata]
    bank_disturbance_bound: float   # ||w_bank|| bound [N]  [→ dynamics]
    rho_bank: float                 # tube radius contribution [m]  [→ tube_radius]
    bank_safety_margin: float       # required bank clearance [m]  [→ safety_margin]
    bank_clearance_violation: bool  # [→ P0 constraint]
    d_bank_effective: float         # effective distance to nearest bank [m]  [metadata]
    _dimensional_scale: float = 1.0  # internal: converts dimensionless ratio → [N]


def assess_bank_effect(
    vessel_speed: float,
    vessel_length: float,
    vessel_beam: float,
    bank_distance_left: Optional[float],
    bank_distance_right: Optional[float],
    U_ref: float = 7.0,
    d_ref: float = 1000.0,
    d_min: float = 10.0,
    config: dict = None,
) -> BankEffectResult:
    """Assess bank effect and compute disturbance/tube bounds.

    Args:
        vessel_speed: Current vessel speed U [m/s].
        vessel_length: Vessel length L [m].
        vessel_beam: Vessel beam B [m].
        bank_distance_left: Distance to left bank [m].
        bank_distance_right: Distance to right bank [m].
        U_ref: Reference speed [m/s].
        d_ref: Reference distance [m].
        d_min: Minimum distance for numerical stability [m].
        config: Optional parameter overrides.

    Returns:
        BankEffectResult.
    """
    cfg = config or {}
    gamma_b0 = cfg.get("gamma_b", 0.05)
    k_b0 = cfg.get("k_b", 0.02)

    # Effective bank distance: nearest bank
    d_left = bank_distance_left if bank_distance_left is not None else float("inf")
    d_right = bank_distance_right if bank_distance_right is not None else float("inf")
    d_bank = min(d_left, d_right)

    if d_bank >= float("inf") - 1:
        return BankEffectResult(
            gamma_b=gamma_b0, k_b=k_b0,
            bank_disturbance_bound=0.0, rho_bank=0.0,
            bank_safety_margin=0.0, bank_clearance_violation=False,
            d_bank_effective=float("inf"),
        )

    # Speed ratio
    U_ratio = max(vessel_speed / U_ref, 0.01)

    # Distance ratio (lower = closer to bank = stronger effect)
    d_ratio = d_ref / max(d_bank, d_min)

    # ── Disturbance bound (BUG #6 fix — 2026-06-10) ────────────────────
    # The product gamma_b × U_ratio² × d_ratio² is a dimensionless lateral
    # force coefficient (per Vantorre et al. 2017 / Delefortrie et al. 2024).
    # To convert to physical Newtons, multiply by the reference hydrodynamic
    # force:  F_ref = ½ ρ_water · L · T · U_ref².
    # For L=180 m, T=10.5 m, U_ref=7 m/s → F_ref ≈ 4.7×10⁷ N.
    # The *gamma_b_factor* config key (default 1.0, dimensionless) allows
    # scaling into physical units without changing gamma_b itself.
    # Set gamma_b_factor = 0.5 * 1025.0 * vessel_length * vessel_beam * U_ref**2
    # in config for physically dimensioned forces [N].
    # Default to physical force scale: 0.5 * rho_water * L * B * U_ref^2.
    # For L=180m, B=30m, U_ref=7 m/s: ≈ 1.35e8 N.
    _F_ref_default = 0.5 * 1025.0 * vessel_length * vessel_beam * U_ref**2
    gamma_b_factor = cfg.get("gamma_b_factor", _F_ref_default)
    bank_disturbance_bound = gamma_b0 * U_ratio**2 * d_ratio**2 * gamma_b_factor

    # Tube radius contribution — capped at 30m to prevent infeasibility.
    # The old formula (k_b0 × d_ratio² × L) gave 36,000m for d_bank=50m
    # because d_ratio = 1000/50 = 20, d_ratio² = 400.  Using d_ratio (not
    # squared) and a hard cap keeps the tube physically reasonable.
    rho_bank = min(
        k_b0 * U_ratio**2 * d_ratio * vessel_length * 0.01,
        30.0,  # hard cap [m]
    )

    # Bank safety margin (minimum clearance)
    bank_safety_margin = 2.0 * vessel_beam  # conservative default

    # Check violation
    bank_clearance_violation = d_bank < bank_safety_margin

    return BankEffectResult(
        gamma_b=gamma_b0,
        k_b=k_b0,
        bank_disturbance_bound=bank_disturbance_bound,
        rho_bank=rho_bank,
        bank_safety_margin=bank_safety_margin,
        bank_clearance_violation=bank_clearance_violation,
        d_bank_effective=d_bank,
    )


def compute_bank_disturbance_force(
    result: BankEffectResult,
    d_left: float,
    d_right: float,
    vessel_speed: float,
) -> np.ndarray:
    """Generate a bank disturbance force vector.

    Returns [X, Y, N] in body frame.
    The sign is determined by which bank is closer (suction toward nearest bank).
    """
    if result.bank_disturbance_bound < 1e-12:
        return np.zeros(3)

    mag = result.bank_disturbance_bound

    # Direction: toward nearest bank
    # Body-frame convention (MMG): positive sway (v, Y) = port direction.
    # When left bank is closer, suction pulls toward port (positive sway).
    if d_left < d_right:
        sway_force = mag    # pull toward left bank (positive = port)
    else:
        sway_force = -mag   # pull toward right bank (negative = starboard)

    # Yaw moment: bow tends to be pushed away from bank, stern toward bank
    # Negative sign: when sway_force > 0 (port pull), yaw is negative (starboard turn),
    # pushing bow away from the port bank — physically correct.
    yaw_moment = -sway_force * 0.1  # simplified coupling

    return np.array([0.0, sway_force, yaw_moment])
