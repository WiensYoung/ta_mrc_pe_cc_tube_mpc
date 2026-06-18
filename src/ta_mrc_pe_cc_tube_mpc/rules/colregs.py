"""COLREGs rule implementation for pairwise encounters.

Implements Rules 13-18 (overtaking, head-on, crossing, responsibilities).
"""

import numpy as np

from ..risk.encounter_classifier import classify_encounter


def colregs_action(
    own_state,
    target_state,
    own_params,
    waterway_context: str = "international",
) -> dict:
    """Determine COLREGs-required action for a pairwise encounter.

    Args:
        own_state: Ownship VesselState.
        target_state: Target VesselState.
        own_params: Ownship VesselParams.
        waterway_context: 'international' or 'inland' (U.S. Inland Rules).

    Returns:
        dict with recommended action, constraints, and metadata.
    """
    encounter = classify_encounter(own_state, target_state)

    action = {
        "encounter_type": encounter["type"],
        "ownship_role": encounter["ownship_role"],
        "required_course_change": 0.0,     # [deg] positive = starboard
        "required_speed_change": 0.0,       # [kn] negative = slow down
        "action_description": "",
        "is_stand_on": encounter["ownship_role"] == "stand_on",
    }

    etype = encounter["type"]
    role = encounter["ownship_role"]

    if etype == "head_on":
        # Rule 14 + Rule 8: both alter course to starboard
        # alteration must be substantial and readily apparent (≥30°)
        action["required_course_change"] = 30.0  # deg to starboard
        action["action_description"] = "Alter course to starboard (Rule 14)"
    elif etype == "crossing_giveway":
        # Rule 15 + Rule 8: give-way vessel avoids crossing ahead
        # alteration must be substantial and readily apparent
        action["required_course_change"] = 20.0  # deg to starboard
        action["action_description"] = "Give way: alter course to starboard, avoid crossing ahead (Rule 15)"
    elif etype == "crossing_standon":
        # Rule 17: stand-on vessel maintains course and speed
        action["required_course_change"] = 0.0
        action["action_description"] = "Stand on: maintain course and speed (Rule 17)"
    elif etype == "overtaking_giveway":
        # Rule 13: overtaking vessel keeps clear
        action["required_course_change"] = 10.0  # Must meet Rule 8 minimum (10 deg)
        action["action_description"] = "Overtaking: keep clear of overtaken vessel (Rule 13)"
    elif etype == "overtaking_standon":
        action["required_course_change"] = 0.0
        action["action_description"] = "Being overtaken: maintain course and speed (Rule 13)"

    # Restricted visibility (Rule 19)
    # When visibility is restricted, all vessels must proceed at safe speed
    # with engines ready. Radar-only detection requires early and substantial
    # avoiding action per Rule 19(d)-(e).
    if waterway_context == "restricted_visibility":
        action = _apply_rule19(action, own_params)

    return action


def inland_rules_action(
    own_state,
    target_state,
    own_params,
    env=None,
    channel_width: float = None,
    downbound: bool = False,
) -> dict:
    """U.S. Inland Rules variant with downbound privilege and narrow channel rules.

    Differences from international COLREGs:
    - Downbound vessel has right-of-way on Western Rivers
    - Narrow channel passing signals and constraints
    - Bend signal requirements
    """
    from .inland_rules import (
        inland_crossing_rule,
        inland_meeting_rule,
        inland_overtaking_rule,
    )

    base_action = colregs_action(own_state, target_state, own_params, waterway_context="inland")
    etype = base_action["encounter_type"]

    # Apply inland-specific rule modifications
    own_len = own_params.length
    # NOTE: target_state is VesselState which lacks 'length'.
    # target_length should be passed by the caller from TargetShip.length.
    other_len = getattr(target_state, 'length', 100.0) if hasattr(target_state, 'length') else 100.0
    ch_width = channel_width if channel_width is not None else 500.0

    if etype == "head_on":
        meeting = inland_meeting_rule(
            ch_width, own_len, other_len, downbound=downbound,
        )
        base_action["inland_meeting"] = meeting
        base_action["minimum_separation"] = meeting["minimum_separation"]

    elif etype in ("crossing_giveway", "crossing_standon"):
        own_y = own_state.y
        tgt_y = target_state.y if hasattr(target_state, 'y') else 0.0
        crossing = inland_crossing_rule(own_y, tgt_y, channel_center_y=0.0)
        base_action["inland_crossing"] = crossing
        if crossing.get("required_course_change", 0.0) != 0:
            base_action["required_course_change"] = crossing["required_course_change"]

    elif "overtaking" in etype:
        is_narrow = ch_width < 10 * own_len
        overtaking = inland_overtaking_rule(own_len, other_len, ch_width, is_narrow)
        base_action["inland_overtaking"] = overtaking
        if not overtaking.get("overtaking_allowed", True):
            base_action["required_course_change"] = 0.0
            base_action["action_description"] = "Overtaking restricted — insufficient channel width"

    base_action["inland_specific"] = {
        "downbound_privilege_applicable": downbound,
        "narrow_channel_rules_applicable": ch_width < 10 * own_len,
        "bend_signal_required": False,
    }
    return base_action


def _apply_rule19(action: dict, own_params) -> dict:
    """Apply Rule 19 (restricted visibility) constraints to a COLREGs action.

    Rule 19 requires:
      (b) Safe speed adapted to visibility conditions.
      (d) Detect risk of collision by radar alone; avoid scanty information.
      (e) Avoiding action must be taken in ample time, be substantial, and
          avoid altering course to port for a vessel forward of the beam.

    Modifies the action dict in place and returns it.
    """
    action["rule_19_applied"] = True
    action["rule_19_safe_speed"] = min(
        action.get("required_speed_ahead", own_params.max_speed),
        0.5 * own_params.max_speed,
    )
    required = action.get("required_course_change", 0.0)

    # Rule 19(e): avoiding action must be substantial and readily apparent
    if abs(required) > 0:
        action["required_course_change"] = required * 1.5

    # Rule 19(e)(ii): avoid altering course to port for vessel forward of beam.
    # Re-read the (now amplified) value to preserve the 1.5x factor.
    # Encounter types use the naming convention "crossing_giveway", "overtaking_giveway"
    # (no underscore between "give" and "way") — match accordingly.
    encounter_type = action.get("encounter_type", "")
    if "head_on" in encounter_type or "giveway" in encounter_type:
        updated = action.get("required_course_change", 0.0)
        if updated < 0:
            action["required_course_change"] = abs(updated)
            action["rule_19_port_override"] = True

    # Rule 19(b): engine ready for immediate manoeuvre
    action["rule_19_engine_ready"] = True

    return action
