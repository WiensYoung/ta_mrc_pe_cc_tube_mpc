"""U.S. Inland Rules implementation.

Key differences from International COLREGs:
- Western Rivers: downbound vessel has right-of-way
- Great Lakes: specific modifications
- Narrow channel rules with sound signals
"""


def is_inland_waterway(waterway_id: str) -> bool:
    """Check if a waterway operates under Inland Rules.

    Args:
        waterway_id: Waterway identifier.

    Returns:
        True if Inland Rules apply.
    """
    inland_waterways = [
        "western_rivers", "mississippi", "ohio_river",
        "great_lakes", "kill_van_kull", "east_river",
        "hudson_river", "chicago_sanitary_canal",
    ]
    return any(w in waterway_id.lower() for w in inland_waterways)


def inland_narrow_channel_rules(
    ownship_length: float,
    channel_width: float,
    meeting_situation: bool = False,
) -> dict:
    """Apply Inland narrow channel rules.

    Args:
        ownship_length: Ownship length [m].
        channel_width: Channel width [m].
        meeting_situation: Whether a meeting situation exists.

    Returns:
        dict with rules and constraints.
    """
    rules = {
        "narrow_channel": False,
        "keep_starboard": False,
        "passing_signal": None,
        "bend_signal": False,
    }

    # Narrow channel: width < 10 * vessel length
    if channel_width < 10 * ownship_length:
        rules["narrow_channel"] = True
        rules["keep_starboard"] = True

    return rules


def inland_overtaking_rule(
    overtaking_vessel_length: float,
    overtaken_vessel_length: float,
    channel_width: float,
    is_narrow_channel: bool = False,
) -> dict:
    """Inland Rule 13 variant: overtaking in narrow channels.

    Inland Rules require:
      - Sound signals: one short blast (port side) or two short blasts (starboard).
      - Overtaken vessel must maintain course and, if in agreement, facilitate
        safe passing.
      - In narrow channels, overtaking restricted unless overtaken vessel
        signals agreement.

    Returns:
        dict with overtaking action constraints.
    """
    action = {
        "overtaking_allowed": True,
        "preferred_side": "port",  # default: overtake on port side
        "sound_signal": "one_short",
        "requires_agreement": is_narrow_channel,
    }

    d_min = 3.0 * (overtaking_vessel_length + overtaken_vessel_length)
    if channel_width < d_min and is_narrow_channel:
        action["overtaking_allowed"] = False
        action["reason"] = "insufficient channel width for safe overtaking"

    return action


def inland_meeting_rule(
    channel_width: float,
    vessel_length_own: float,
    vessel_length_other: float,
    downbound: bool = False,
) -> dict:
    """Inland Rule 14 variant: meeting (head-on) in rivers/channels.

    Inland Rules differ from COLREGs in:
      - Western Rivers: downbound vessel has privilege (right-of-way).
      - Great Lakes: specific passing arrangements.
      - Both vessels alter course to starboard, but the downbound vessel
        may hold course if the upbound vessel has more maneuverability.

    Returns:
        dict with meeting action constraints.
    """
    action = {
        "alter_course_to_starboard": True,
        "passing_side": "port_to_port",
        "minimum_separation": 2.0 * (vessel_length_own + vessel_length_other),
    }

    if downbound:
        action["downbound_privilege"] = True
        action["preferred_action"] = "maintain_course"

    return action


def inland_crossing_rule(
    own_position_y: float,
    other_position_y: float,
    channel_center_y: float = 0.0,
    river_flow_direction: float = 0.0,  # heading angle [rad] of downstream
) -> dict:
    """Inland Rule 15 variant: crossing in rivers.

    In narrow channels/rivers:
      - Crossing vessel must avoid impeding the passage of a vessel
        proceeding along the channel.
      - Vessel proceeding with the current may have limited maneuverability.

    Returns:
        dict with crossing action constraints.
    """
    own_near_center = abs(own_position_y - channel_center_y)
    other_near_center = abs(other_position_y - channel_center_y)

    action = {
        "crossing_constraint": "standard_colregs",
    }

    # Vessel in channel center has priority
    if own_near_center < other_near_center:
        action["crossing_constraint"] = "stand_on"
        action["reason"] = "vessel proceeding along channel has priority"
        action["required_course_change"] = 0.0
    else:
        action["crossing_constraint"] = "give_way"
        action["reason"] = "crossing vessel must not impede channel traffic"
        action["required_course_change"] = 10.0  # substantial alteration

    return action
