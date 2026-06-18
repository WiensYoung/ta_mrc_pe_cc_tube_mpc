"""TCPA/DCPA computation with strict boundary handling.

Implements the formulas:
    p_rel = p_target - p_own
    v_rel = v_target - v_own
    TCPA_raw = -p_rel^T v_rel / (||v_rel||^2 + eps_v)
    TCPA = clip(TCPA_raw, 0, T_H)
    DCPA = ||p_rel + TCPA * v_rel||
"""

import numpy as np


def compute_tcpa_dcpa(
    own_pos: np.ndarray,
    own_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    horizon: float = 600.0,
    eps_v: float = 1e-6,
) -> dict:
    """Compute TCPA and DCPA with boundary handling.

    Args:
        own_pos: Ownship position [x, y] in world frame.
        own_vel: Ownship velocity [vx, vy] in world frame.
        target_pos: Target position [x, y] in world frame.
        target_vel: Target velocity [vx, vy] in world frame.
        horizon: Prediction horizon T_H [s].
        eps_v: Small velocity for numerical stability when ||v_rel|| ≈ 0.

    Returns:
        dict with keys:
            tcpa: Clamped time to CPA [s] (0 <= tcpa <= horizon).
            dcpa: Distance at CPA [m].
            tcpa_raw: Unclamped TCPA [s].
            v_rel_norm: Relative speed [m/s].
            is_converging: True if vessels are closing range.
            risk_assessment: 'critical', 'warning', or 'safe'.
    """
    p_rel = target_pos[:2] - own_pos[:2]
    v_rel = target_vel[:2] - own_vel[:2]

    # Clip to prevent overflow in dot products
    MAX_COMP = 1e4
    p_rel = np.clip(p_rel, -MAX_COMP, MAX_COMP)
    v_rel = np.clip(v_rel, -MAX_COMP, MAX_COMP)

    v_rel_sq = float(np.dot(v_rel, v_rel))
    v_rel_norm = float(np.sqrt(max(v_rel_sq, 0.0)))

    # TCPA computation with epsilon guard
    if v_rel_sq < eps_v:
        # Nearly zero relative velocity: vessels moving together
        # Risk determined by current distance
        tcpa_raw = float("inf")
        tcpa = horizon  # conservative: treat as constant close proximity
        dcpa = float(np.linalg.norm(p_rel))
        is_converging = False
    else:
        tcpa_raw = float(-np.dot(p_rel, v_rel) / (v_rel_sq + eps_v))
        # Clip TCPA to [0, horizon]
        tcpa = float(np.clip(tcpa_raw, 0.0, horizon))
        # DCPA at the clamped TCPA
        p_at_cpa = p_rel + tcpa * v_rel
        dcpa = float(np.linalg.norm(p_at_cpa))
        is_converging = tcpa_raw > 0

    # Risk assessment
    current_distance = float(np.linalg.norm(p_rel))
    risk = _assess_risk(tcpa, dcpa, current_distance, is_converging, v_rel_norm)

    # NaN guard on return values
    def _safe(v, default):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return default
        return v

    return {
        "tcpa": _safe(tcpa, horizon),
        "dcpa": _safe(dcpa, 1e4),
        "tcpa_raw": _safe(tcpa_raw, horizon),
        "v_rel_norm": _safe(v_rel_norm, 0.0),
        "is_converging": is_converging,
        "current_distance": _safe(current_distance, 1e4),
        "risk_assessment": risk,
    }


def _assess_risk(
    tcpa: float,
    dcpa: float,
    current_distance: float,
    is_converging: bool,
    v_rel_norm: float,
    d_safe: float = 500.0,
    t_warn: float = 300.0,
) -> str:
    """Classify risk level based on TCPA/DCPA.

    Returns one of: 'critical', 'warning', 'safe'.
    """
    if dcpa < d_safe and tcpa < t_warn and is_converging:
        if tcpa < 60.0 or dcpa < 0.5 * d_safe:
            return "critical"
        return "warning"
    if dcpa < d_safe and not is_converging:
        # Already passed CPA but still close
        if current_distance < 0.5 * d_safe:
            return "warning"
    return "safe"


def compute_cpa_for_anchored(
    own_pos: np.ndarray,
    own_vel: np.ndarray,
    target_pos: np.ndarray,
    target_length: float = 100.0,
    target_beam: float = 15.0,
    horizon: float = 600.0,
) -> dict:
    """Compute CPA for an anchored or stationary target.

    An anchored vessel is treated as a static obstacle with dimensions.
    Risk is based on closest approach of ownship to the target position,
    with the target's physical extent included in the safe distance.

    Args:
        own_pos: Ownship position [x, y].
        own_vel: Ownship velocity [vx, vy].
        target_pos: Target (anchored) position [x, y].
        target_length: Target vessel length [m].
        target_beam: Target vessel beam [m].
        horizon: Prediction horizon [s].

    Returns:
        TCPA/DCPA dict with static target treatment.
    """
    target_vel = np.zeros(2)
    result = compute_tcpa_dcpa(own_pos, own_vel, target_pos, target_vel, horizon)

    # Inflate safe distance by target dimensions
    target_extent = np.sqrt(target_length**2 + target_beam**2)
    result["target_extent"] = target_extent
    result["dcpa_effective"] = max(0.0, result["dcpa"] - 0.5 * target_extent)

    return result


def compute_cpa_for_vessel_pair(
    vessel_a_state,
    vessel_b_state,
    horizon: float = 600.0,
) -> dict:
    """Compute CPA between two vessels using their state objects.

    Args:
        vessel_a_state: VesselState for ship A.
        vessel_b_state: VesselState for ship B.
        horizon: Prediction horizon [s].

    Returns:
        TCPA/DCPA dict.
    """
    from ..utils.coordinates import body_to_world

    pos_a = np.array([vessel_a_state.x, vessel_a_state.y])
    vel_a = body_to_world(vessel_a_state.u, vessel_a_state.v, vessel_a_state.psi)
    pos_b = np.array([vessel_b_state.x, vessel_b_state.y])
    vel_b = body_to_world(vessel_b_state.u, vessel_b_state.v, vessel_b_state.psi)

    result = compute_tcpa_dcpa(pos_a, vel_a, pos_b, vel_b, horizon)
    result["vessel_pair"] = (id(vessel_a_state), id(vessel_b_state))
    return result
