"""Enhanced COLREGs implementation — Rules 6, 8, 18 and multi-vessel consistency.

Adds:
  - Rule 18: Complete vessel responsibility hierarchy (NUC > RAM > CBD > fishing > sailing > power)
  - Rule 8:  Verification that avoiding action is "substantial and readily apparent"
  - Rule 6:  Structured safe-speed assessment framework
  - Multi-vessel pairwise consistency check
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 18 — Vessel Responsibility Hierarchy
# ═══════════════════════════════════════════════════════════════════════════════
#
# COLREGs Rule 18 (a)-(f) establishes a strict hierarchy of responsibilities
# between vessels in sight of one another. A vessel higher in the hierarchy
# must be given way by vessels lower in the hierarchy.
#
# Hierarchy (highest priority first):
#   1. Not Under Command (NUC)
#   2. Restricted in Ability to Manoeuvre (RAM)
#   3. Constrained by Draught (CBD)
#   4. Engaged in Fishing
#   5. Sailing vessel
#   6. Power-driven vessel
#   7. Seaplane / WIG craft

# Rank values: LOWER number = HIGHER priority (must be given way)
VESSEL_RESPONSIBILITY_RANK = {
    "not_under_command":             1,
    "restricted_maneuverability":    2,
    "constrained_by_draught":        3,
    "fishing":                       4,
    "sailing":                       5,
    "power_driven":                  6,
    "towing":                        6,   # treated as power-driven (but see Rule 18(e))
    "tug_barge":                     6,
    "cargo":                         6,
    "tanker":                        6,
    "ferry":                         6,
    "small_craft":                   6,
    "seaplane":                      7,
    "wig":                           7,
    "at_anchor":                     0,   # anchored = static obstacle (not in hierarchy)
    "moored":                        0,
    "aground":                       0,
    "underway":                      6,   # default = power-driven
}

def resolve_responsibility(
    own_nav_status: str,
    own_vessel_type: str,
    target_nav_status: str,
    target_vessel_type: str,
) -> dict:
    """Determine give-way / stand-on based on Rule 18 hierarchy.

    Args:
        own_nav_status: Ownship navigation status (e.g., "underway", "not_under_command").
        own_vessel_type: Ownship vessel type (e.g., "cargo", "fishing").
        target_nav_status: Target navigation status.
        target_vessel_type: Target vessel type.

    Returns:
        dict with:
            own_rank: Ownship responsibility rank (lower = higher priority).
            target_rank: Target responsibility rank.
            own_is_privileged: True if ownship has higher priority.
            own_must_give_way: True if Rule 18 requires ownship to give way.
            rule18_basis: Explanation string.
    """
    # Map nav_status to effective vessel class.
    # Priority: special nav_status (NUC, RAM, anchored, etc.) overrides vessel_type.
    # But "underway" is generic — vessel_type (fishing, sailing, etc.) takes precedence.
    def _effective_class(nav_status: str, vessel_type: str) -> str:
        if nav_status in ("at_anchor", "moored", "aground"):
            return nav_status
        if nav_status in ("not_under_command", "restricted_maneuverability",
                          "constrained_by_draught"):
            return nav_status
        # For generic "underway", use vessel_type to determine hierarchy
        if vessel_type in ("fishing", "sailing", "seaplane", "wig"):
            return vessel_type
        return "power_driven"  # default for underway vessels

    own_class = _effective_class(own_nav_status, own_vessel_type)
    tgt_class = _effective_class(target_nav_status, target_vessel_type)

    own_rank = VESSEL_RESPONSIBILITY_RANK.get(own_class, 6)
    tgt_rank = VESSEL_RESPONSIBILITY_RANK.get(tgt_class, 6)

    # Anchored / moored / aground are static obstacles, not in hierarchy
    if own_rank == 0 or tgt_rank == 0:
        return {
            "own_rank": own_rank,
            "target_rank": tgt_rank,
            "own_is_privileged": own_rank == 0,
            "own_must_give_way": tgt_rank == 0,
            "rule18_basis": "Static obstacle (anchored/moored/aground) — Rule 18 does not apply; obstacle avoidance required",
        }

    # Lower rank number = higher priority
    own_privileged = own_rank < tgt_rank
    must_give_way = own_rank > tgt_rank

    if own_privileged:
        basis = f"Own ({own_class}, rank {own_rank}) has priority over target ({tgt_class}, rank {tgt_rank}) — Rule 18"
    elif must_give_way:
        basis = f"Target ({tgt_class}, rank {tgt_rank}) has priority over own ({own_class}, rank {own_rank}) — Rule 18"
    else:
        basis = f"Equal rank ({own_rank}) — standard COLREGs encounter rules apply (Rules 13-15)"

    return {
        "own_rank": own_rank,
        "target_rank": tgt_rank,
        "own_is_privileged": own_privileged,
        "own_must_give_way": must_give_way,
        "rule18_basis": basis,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 8 — Action to Avoid Collision (substantial, readily apparent)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Rule 8(a): Any action to avoid collision shall be taken in accordance with
# the Rules and, if the circumstances permit, be positive, made in ample time,
# and with due regard to the observance of good seamanship.
#
# Rule 8(b): Any alteration of course and/or speed to avoid collision shall
# be large enough to be readily apparent to another vessel observing visually
# or by radar. A succession of small course and/or speed alterations should
# be avoided.
#
# Rule 8(d): Action taken to avoid collision shall result in passing at a
# safe distance. The effectiveness of the action shall be carefully checked
# until the other vessel is finally past and clear.

# Recommended minimum course changes per COLREGs guidance and maritime practice
# Sources: Cockcroft & Lameijer "Guide to the Collision Avoidance Rules"
MINIMUM_SUBSTANTIAL_COURSE_CHANGE = {
    "head_on":              30.0,   # [deg]
    "crossing_giveway":     20.0,   # [deg]
    "crossing_standon":      0.0,   # stand-on maintains course
    "overtaking_giveway":   10.0,   # [deg] smaller for overtaking
    "overtaking_standon":    0.0,
    "none":                  0.0,
}

MINIMUM_SUBSTANTIAL_SPEED_CHANGE = 0.20  # 20% change is "readily apparent"


def verify_action_substantial(
    encounter_type: str,
    course_change_deg: float,
    speed_change_fraction: float,
    own_speed: float,
) -> dict:
    """Verify that a COLREGs avoiding action is substantial and readily apparent.

    Per Rule 8(b): alterations must be large enough to be readily apparent.
    A succession of small alterations should be avoided (Rule 8(b) and 8(c)).

    Args:
        encounter_type: COLREGs encounter type.
        course_change_deg: Proposed course change [deg], positive = starboard.
        speed_change_fraction: Fractional speed change (e.g., -0.3 = 30% reduction).
        own_speed: Current speed [m/s].

    Returns:
        dict with verification results.
    """
    min_course = MINIMUM_SUBSTANTIAL_COURSE_CHANGE.get(encounter_type, 0.0)
    min_speed_frac = MINIMUM_SUBSTANTIAL_SPEED_CHANGE

    course_substantial = abs(course_change_deg) >= min_course if min_course > 0 else True

    # Speed-change-only is substantial if speed change meets threshold.
    # If course change IS the primary action (recommended by COLREGs),
    # speed change is optional and does NOT need to meet the threshold.
    is_stand_on = encounter_type in ("crossing_standon", "overtaking_standon", "none")
    if abs(course_change_deg) < min_course and abs(speed_change_fraction) > 0:
        # Speed change is being used as primary action → must meet threshold
        speed_substantial = abs(speed_change_fraction) >= min_speed_frac
    else:
        # Course change is the primary action → speed change not required
        speed_substantial = True

    # Stand-on vessels maintain course AND speed → no action needed
    if is_stand_on:
        course_substantial = True
        speed_substantial = True

    # Rule 8(c): avoid crossing ahead if give-way → prefer course change over speed change
    # (speed change alone is less apparent on radar)
    radar_apparent = True
    if abs(course_change_deg) < 5.0 and abs(speed_change_fraction) > 0 and not is_stand_on:
        radar_apparent = False  # speed-only change not readily apparent on radar

    # Rule 8(d): verify passing distance after action
    # (Cannot verify without trajectory — flag for post-hoc check)

    all_substantial = course_substantial and speed_substantial and radar_apparent

    issues = []
    if not course_substantial:
        issues.append(
            f"Course change {course_change_deg:.1f}° < minimum {min_course:.0f}° "
            f"for {encounter_type} (Rule 8(b))"
        )
    if not speed_substantial and encounter_type != "none":
        issues.append(
            f"Speed change {speed_change_fraction*100:.0f}% < minimum {min_speed_frac*100:.0f}% "
            f"(Rule 8(b))"
        )
    if not radar_apparent:
        issues.append(
            "Speed-only alteration not readily apparent on radar (Rule 8(b)/(c))"
        )

    return {
        "course_change_deg": course_change_deg,
        "speed_change_fraction": speed_change_fraction,
        "min_course_required": min_course,
        "course_substantial": course_substantial,
        "speed_substantial": speed_substantial,
        "radar_apparent": radar_apparent,
        "all_substantial": all_substantial,
        "issues": issues,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Rule 6 — Safe Speed Assessment
# ═══════════════════════════════════════════════════════════════════════════════
#
# Rule 6: Every vessel shall at all times proceed at a safe speed so that
# she can take proper and effective action to avoid collision and be stopped
# within a distance appropriate to the prevailing circumstances and conditions.
#
# Factors to consider (Rule 6(a)):
#   (i)   State of visibility
#   (ii)  Traffic density, including concentrations of fishing vessels
#   (iii) Manoeuvrability (stopping distance, turning ability)
#   (iv)  Background lighting (shore lights, back-scatter)
#   (v)   Wind, sea, current, proximity of navigational hazards
#   (vi)  Draught in relation to available water depth (UKC)

@dataclass
class SafeSpeedAssessment:
    """Structured safe speed assessment per Rule 6."""

    # Input factors
    visibility_nm: float = 10.0          # visibility [nautical miles]
    traffic_density: float = 0.0          # 0.0 (none) to 1.0 (congested)
    own_length_m: float = 180.0
    own_draught_m: float = 10.5
    water_depth_m: float = 50.0
    current_speed_kn: float = 0.0        # current speed [knots]
    wind_speed_kn: float = 0.0           # wind speed [knots]
    bank_proximity: bool = False         # near bank/shore
    restricted_visibility: bool = False  # fog, heavy rain, etc.

    # Output
    safe_speed_factor: float = 1.0       # multiplier on max_speed (0.0-1.0)
    speed_reduction_reasons: list[str] = field(default_factory=list)
    stopping_distance_m: float = 0.0


def assess_safe_speed(
    own_params,              # VesselParams
    visibility_nm: float = 10.0,
    traffic_density: float = 0.0,
    water_depth_m: float = 50.0,
    current_speed_kn: float = 0.0,
    wind_speed_kn: float = 0.0,
    bank_proximity: bool = False,
    num_nearby_targets: int = 0,
    ukc_m: Optional[float] = None,
) -> SafeSpeedAssessment:
    """Structured safe speed assessment per Rule 6.

    Returns a SafeSpeedAssessment with a recommended speed factor (0.0–1.0)
    to be applied to own_params.max_speed.

    Args:
        own_params: VesselParams with max_speed, length, draught, etc.
        visibility_nm: Visibility range [nm].
        traffic_density: Traffic density score 0.0–1.0.
        water_depth_m: Water depth [m].
        current_speed_kn: Current speed [kn].
        wind_speed_kn: Wind speed [kn].
        bank_proximity: Whether vessel is near bank.
        num_nearby_targets: Number of targets within 2nm.
        ukc_m: Under-keel clearance [m] (computed if None).
    """
    assessment = SafeSpeedAssessment(
        visibility_nm=visibility_nm,
        traffic_density=traffic_density,
        own_length_m=own_params.length,
        own_draught_m=own_params.draught,
        water_depth_m=water_depth_m,
        current_speed_kn=current_speed_kn,
        wind_speed_kn=wind_speed_kn,
        bank_proximity=bank_proximity,
    )

    factor = 1.0  # start at full speed, reduce for each risk factor

    # Rule 6(a)(i): Visibility
    if visibility_nm < 0.5:
        factor = min(factor, 0.4)
        assessment.speed_reduction_reasons.append("Visibility < 0.5nm (Rule 6(a)(i))")
    elif visibility_nm < 1.0:
        factor = min(factor, 0.55)
        assessment.speed_reduction_reasons.append("Visibility < 1.0nm (Rule 6(a)(i))")
    elif visibility_nm < 2.0:
        factor = min(factor, 0.7)
        assessment.speed_reduction_reasons.append("Visibility < 2.0nm — restricted (Rule 6(a)(i))")
    elif visibility_nm < 5.0:
        factor = min(factor, 0.85)
        assessment.speed_reduction_reasons.append("Visibility < 5.0nm (Rule 6(a)(i))")

    assessment.restricted_visibility = visibility_nm < 2.0

    # Rule 6(a)(ii): Traffic density
    if traffic_density > 0.8:
        factor = min(factor, 0.5)
        assessment.speed_reduction_reasons.append(f"High traffic density ({traffic_density:.1f}) (Rule 6(a)(ii))")
    elif traffic_density > 0.5:
        factor = min(factor, 0.7)
        assessment.speed_reduction_reasons.append(f"Moderate traffic density ({traffic_density:.1f}) (Rule 6(a)(ii))")
    elif num_nearby_targets >= 3:
        factor = min(factor, 0.8)
        assessment.speed_reduction_reasons.append(f"{num_nearby_targets} nearby targets (Rule 6(a)(ii))")

    # Rule 6(a)(iii): Manoeuvrability (simplified: larger vessel = longer stopping distance)
    # Stopping distance ~ V² / (2 * a), where a ≈ 0.1–0.3 m/s² for large vessels
    max_speed_ms = own_params.max_speed * 0.51444  # kn to m/s
    stopping_dist = max_speed_ms ** 2 / (2 * 0.15)
    assessment.stopping_distance_m = stopping_dist
    if stopping_dist > 2000:  # very long stopping distance
        factor = min(factor, 0.85)
        assessment.speed_reduction_reasons.append(f"Long stopping distance ({stopping_dist:.0f}m) (Rule 6(a)(iii))")

    # Rule 6(a)(v): Environmental conditions
    if wind_speed_kn > 30:
        factor = min(factor, 0.7)
        assessment.speed_reduction_reasons.append(f"Strong wind ({wind_speed_kn:.0f}kn) (Rule 6(a)(v))")
    if current_speed_kn > 3:
        factor = min(factor, 0.8)
        assessment.speed_reduction_reasons.append(f"Strong current ({current_speed_kn:.0f}kn) (Rule 6(a)(v))")
    if bank_proximity:
        factor = min(factor, 0.75)
        assessment.speed_reduction_reasons.append("Bank proximity — squat/interaction risk (Rule 6(a)(v))")

    # Rule 6(a)(vi): UKC
    if ukc_m is None:
        ukc_m = water_depth_m - own_params.draught
    if ukc_m < 2.0:
        factor = min(factor, 0.5)
        assessment.speed_reduction_reasons.append(f"Minimal UKC ({ukc_m:.1f}m) (Rule 6(a)(vi))")
    elif ukc_m < 5.0:
        factor = min(factor, 0.7)
        assessment.speed_reduction_reasons.append(f"Reduced UKC ({ukc_m:.1f}m) (Rule 6(a)(vi))")

    assessment.safe_speed_factor = max(factor, 0.2)  # minimum 20% — never stop completely
    return assessment


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-vessel Pairwise Consistency Check
# ═══════════════════════════════════════════════════════════════════════════════
#
# When multiple COLREGs encounters exist simultaneously, the controller must
# avoid contradictory actions (e.g., turning starboard for target A while
# turning port toward target B). This is a practical consistency check, not
# a formal multi-vessel COLREGs resolution (which is an open research problem).

def check_multi_vessel_consistency(
    own_state,
    target_ships: list,
    colregs_actions: list[dict],
) -> dict:
    """Check pairwise COLREGs action consistency across multiple targets.

    Detects conflicts where the recommended action for one target contradicts
    the recommended action for another target (e.g., starboard turn for A
    but port turn for B, both recommended simultaneously).

    Args:
        own_state: Ownship state.
        target_ships: List of TargetShip objects.
        colregs_actions: List of per-target COLREGs action dicts.

    Returns:
        dict with:
            consistent: True if no pairwise conflicts found.
            conflicts: List of conflict descriptions.
            recommended_resolution: Suggested priority-based resolution.
    """
    conflicts = []

    if len(colregs_actions) < 2:
        return {"consistent": True, "conflicts": [], "recommended_resolution": None}

    for i in range(len(colregs_actions)):
        for j in range(i + 1, len(colregs_actions)):
            act_i = colregs_actions[i]
            act_j = colregs_actions[j]

            # Conflict type 1: opposite course changes
            course_i = act_i.get("required_course_change", 0.0)
            course_j = act_j.get("required_course_change", 0.0)

            if course_i * course_j < 0 and abs(course_i) > 5 and abs(course_j) > 5:
                # Opposite signs and both substantial
                conflicts.append(
                    f"Target {i} ({act_i.get('encounter_type','?')}) requires "
                    f"{course_i:+.0f}° vs target {j} ({act_j.get('encounter_type','?')}) "
                    f"requires {course_j:+.0f}° — contradictory course changes"
                )

            # Conflict type 2: one requires speed-up, other requires slow-down
            speed_i = act_i.get("required_speed_change", 0.0)
            speed_j = act_j.get("required_speed_change", 0.0)
            if speed_i * speed_j < 0 and abs(speed_i) > 2 and abs(speed_j) > 2:
                conflicts.append(
                    f"Target {i} requires speed change {speed_i:+.1f}kn vs "
                    f"target {j} requires {speed_j:+.1f}kn — contradictory speed changes"
                )

    if conflicts:
        # Resolution: prioritize by Rule 18 hierarchy + encounter severity
        recommendation = (
            "Prioritize target with higher Rule 18 rank (NUC/RAM/CBD first). "
            "If equal, prioritize head-on over crossing over overtaking. "
            "Flag for CBF-QP to resolve remaining conflict at runtime."
        )
    else:
        recommendation = None

    return {
        "consistent": len(conflicts) == 0,
        "conflicts": conflicts,
        "recommended_resolution": recommendation,
    }
