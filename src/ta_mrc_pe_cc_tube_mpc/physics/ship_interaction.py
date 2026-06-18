"""Ship-ship hydrodynamic interaction model.

Activation conditions:
    I_ship_ij = 1 if:
        d_ij < k_L * (L_i + L_j)
        or d_lat_ij < k_B * (B_i + B_j)
        or TCPA_ij < T_crit

Disturbance bound:
    ||w_ship_ij|| <= gamma_s * (U_rel_ij/U_ref)^2 * (d_ref/max(d_ij, d_min))^2

References:
    - Vantorre, M., Laforce, E., Delefortrie, G. (2013).
      "Ship-ship interaction: model test-based empirical formulations."
      → Overtaking and passing hydrodynamic interaction coefficients.

    - Lataire, E., Vantorre, M. (2011).
      "Navigation in confined waters: influence of bank characteristics
      on ship-bank interaction." 2nd Int. Conf. on Ship Manoeuvring in
      Shallow and Confined Water, Trondheim.
      → Coupled bank and ship-ship interaction effects.

    - Lee, H., Tran, H., Kim, J. (2024).
      "Safety-Guaranteed Ship Berthing Using Cascade Tube-Based Model
      Predictive Control." IEEE Trans. Control Systems Technology,
      32(4), 1504-1511. DOI: 10.1109/TCST.2024.3381234
      → Tube-MPC applied to underactuated ships with bounded disturbances.

The model uses conservative upper-bound estimation. Does NOT implement
full potential-flow or RANS-based interaction modeling. See
ASSUMPTIONS.md Section 3 for details and limitations.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..constants import (
    D_MIN_DEFAULT,
    D_REF_DEFAULT,
    K_B_INTERACTION,
    K_L_INTERACTION,
    T_CRIT_INTERACTION,
    U_REF_DEFAULT,
)
from ..risk.cpa import compute_tcpa_dcpa
from ..types import TargetShip, VesselState
from ..utils.coordinates import body_to_world, distance


@dataclass
class ShipInteractionResult:
    """Output of ship-ship interaction assessment."""
    active: bool
    risk_score: float              # [0, 1] interaction risk
    rho_ship: float                # tube radius contribution [m]
    disturbance_bound: float       # ||w_ship|| bound [N]
    clearance_violation: bool
    min_inter_ship_clearance: float
    details: list                  # per-target details


def assess_ship_interaction(
    own_state: VesselState,
    own_length: float,
    own_beam: float,
    targets: list[TargetShip],
    U_ref: float = U_REF_DEFAULT,
    d_ref: float = D_REF_DEFAULT,
    d_min: float = D_MIN_DEFAULT,
    config: dict = None,
) -> ShipInteractionResult:
    """Assess ship-ship interaction for all nearby targets.

    Args:
        own_state: Ownship state.
        own_length: Ownship length L_i [m].
        own_beam: Ownship beam B_i [m].
        targets: List of target ships.
        U_ref: Reference speed [m/s].
        d_ref: Reference distance [m].
        d_min: Minimum distance [m].
        config: Optional parameter overrides.

    Returns:
        ShipInteractionResult.
    """
    cfg = config or {}
    gamma_s0 = cfg.get("gamma_s", 0.03)
    k_s0 = cfg.get("k_s", 0.015)
    k_L = cfg.get("k_L", K_L_INTERACTION)
    k_B = cfg.get("k_B", K_B_INTERACTION)
    T_crit = cfg.get("T_crit", T_CRIT_INTERACTION)

    own_speed = own_state.speed()
    own_pos = own_state.position()

    rho_total = 0.0
    disturbance_total = 0.0
    risk_total = 0.0
    any_active = False
    min_clearance = float("inf")
    details = []

    for target in targets:
        tgt_pos = target.state.position()
        d_ij = distance(own_pos, tgt_pos)

        # Lateral distance: project relative position onto ownship body-frame y-axis
        dx = target.state.x - own_state.x
        dy = target.state.y - own_state.y
        lat_vec = np.array([-np.sin(own_state.psi), np.cos(own_state.psi)])
        d_lat_ij = abs(float(np.dot(lat_vec, np.array([dx, dy]))))

        # Relative speed — vector magnitude, not scalar difference.
        # For crossing encounters, abs(own_speed - tgt_speed) ≈ 0 when
        # speeds are similar, severely underestimating the disturbance.
        own_vel_w = body_to_world(own_state.u, own_state.v, own_state.psi)
        tgt_vel_w = body_to_world(target.state.u, target.state.v, target.state.psi)
        U_rel = float(np.linalg.norm(np.array(own_vel_w) - np.array(tgt_vel_w)))

        # Activation check
        active_longitudinal = d_ij < k_L * (own_length + target.length)
        active_lateral = d_lat_ij < k_B * (own_beam + target.beam)
        # TCPA check — reuse velocity vectors computed above
        cpa = compute_tcpa_dcpa(own_pos, own_vel_w, tgt_pos, tgt_vel_w, horizon=T_crit)
        active_tcpa = cpa["tcpa"] < T_crit and cpa["dcpa"] < 500.0

        is_active = active_longitudinal or active_lateral or active_tcpa

        detail = {
            "target_mmsi": target.mmsi,
            "d_ij": d_ij,
            "d_lat_ij": d_lat_ij,
            "active_longitudinal": active_longitudinal,
            "active_lateral": active_lateral,
            "active_tcpa": active_tcpa,
            "is_active": is_active,
            "tcpa": cpa["tcpa"],
        }

        if is_active:
            any_active = True
            # Disturbance bound for this target.
            # BUG #6 fix (2026-06-10): gamma_s_factor converts the dimensionless
            # coefficient product into physical Newtons (same convention as
            # bank_effect.py). Default 1.0 preserves backward compatibility.
            # Default to physical force scale: 0.5 * rho_water * L * T * U_ref^2.
            # Use own_length as proxy for L, own_beam for T (approximate).
            _F_ref_default = 0.5 * 1025.0 * own_length * own_beam * 0.35 * U_ref**2
            gamma_s_factor = cfg.get("gamma_s_factor", _F_ref_default)
            U_ratio = max(U_rel / U_ref, 0.01)
            d_ratio = d_ref / max(d_ij, d_min)
            d_bound = gamma_s0 * U_ratio**2 * d_ratio**2 * gamma_s_factor

            # Tube radius contribution
            # Tube radius per target pair — capped at 20m to prevent infeasibility.
            # The old formula (k_s0 × d_ratio² × (L_i+L_j)) gave thousands of
            # meters for moderate inter-ship distances because d_ratio² explodes.
            rho_ij = min(
                k_s0 * U_ratio**2 * d_ratio * (own_length + target.length) * 0.01,
                20.0,  # hard cap per target [m]
            )

            # Risk score — guard against zero coefficients to prevent ZeroDivisionError
            risk_a = 0.0
            if abs(gamma_s0) > 1e-12:
                risk_a = min(1.0, d_bound / gamma_s0)
            risk_b = 0.0
            denom_b = k_s0 * (own_length + target.length)
            if abs(denom_b) > 1e-12:
                risk_b = min(1.0, rho_ij / denom_b)
            risk_ij = risk_a * risk_b

            detail["disturbance_bound"] = d_bound
            detail["rho_ij"] = rho_ij
            detail["risk_ij"] = risk_ij

            rho_total += rho_ij
            disturbance_total += d_bound
            risk_total = max(risk_total, risk_ij)

        min_clearance = min(min_clearance, d_ij)
        details.append(detail)

    clearance_violation = min_clearance < 0.5 * (own_length + own_beam)

    return ShipInteractionResult(
        active=any_active,
        risk_score=risk_total,
        rho_ship=rho_total,
        disturbance_bound=disturbance_total,
        clearance_violation=clearance_violation,
        min_inter_ship_clearance=min_clearance,
        details=details,
    )


def compute_ship_interaction_force(
    result: ShipInteractionResult,
    own_state: VesselState,
    targets: list[TargetShip],
) -> np.ndarray:
    """Generate ship interaction disturbance force.

    Returns [X, Y, N] in body frame. Simplified repulsion model.
    """
    if not result.active or result.disturbance_bound < 1e-12:
        return np.zeros(3)

    # Repulsion from each active target
    force = np.zeros(3)
    own_pos = own_state.position()

    for i, detail in enumerate(result.details):
        if not detail.get("is_active", False):
            continue
        if i >= len(targets):
            continue
        target = targets[i]
        tgt_pos = target.state.position()
        rel_vec = own_pos - tgt_pos
        d = float(np.linalg.norm(rel_vec))
        if d < 1e-3:
            continue
        direction = rel_vec / d

        # Rotate world-frame direction into body frame
        cos_psi = np.cos(own_state.psi)
        sin_psi = np.sin(own_state.psi)
        body_dir_x = direction[0] * cos_psi + direction[1] * sin_psi
        body_dir_y = -direction[0] * sin_psi + direction[1] * cos_psi

        # Repulsive sway force (body-frame Y from lateral component)
        bound = detail.get("disturbance_bound", 0.0)
        sway_force = bound * body_dir_y
        force[1] += sway_force
        force[2] += sway_force * 0.05  # small yaw coupling

    return force
