"""COLREGs encounter classifier.

Classifies pairwise encounters into:
- head_on
- crossing (give_way or stand_on)
- overtaking (give_way or stand_on)
"""

import numpy as np

from ..constants import (
    CROSSING_GIVEWAY_RANGE,
    CROSSING_STANDON_RANGE,
    HEAD_ON_BEARING_RANGE,
    OVERTAKING_RANGE,
)
from ..utils.coordinates import relative_bearing


def classify_encounter(
    own_state,
    target_state,
    encounter_distance: float = 6000.0,  # [m] max distance for encounter
) -> dict:
    """Classify the COLREGs encounter type between ownship and a target.

    Args:
        own_state: Ownship VesselState.
        target_state: Target VesselState.
        encounter_distance: Maximum distance to consider as encounter [m].

    Returns:
        dict with:
            type: 'head_on', 'crossing_giveway', 'crossing_standon',
                  'overtaking_giveway', 'overtaking_standon', or 'none'.
            ownship_role: 'give_way', 'stand_on', or 'none'.
            target_role:  'give_way', 'stand_on', or 'none'.
                          (complement of ownship_role except head-on where
                           both are 'give_way', and 'none' where no encounter)
            rel_bearing: Relative bearing from ownship to target [deg].
            rel_bearing_target: Relative bearing from target to ownship [deg].
            distance: Current distance [m].
            aspect: Target aspect angle [deg].
    """
    from ..utils.coordinates import distance

    pos_own = own_state.position()
    pos_tgt = target_state.position()
    d = distance(pos_own, pos_tgt)

    if d > encounter_distance:
        return {"type": "none", "ownship_role": "none", "target_role": "none",
                "rel_bearing": 0.0, "rel_bearing_target": 0.0,
                "distance": d, "aspect": 0.0}

    # Relative bearing from ownship to target
    rel_brg = relative_bearing(pos_own, own_state.psi, pos_tgt)
    # Relative bearing from target to ownship
    rel_brg_target = relative_bearing(pos_tgt, target_state.psi, pos_own)

    encounter_type, own_role = _determine_encounter(rel_brg, rel_brg_target)

    # Target's role is the complement of ownship's:
    #   own=give_way → target=stand_on (except head-on: both give_way)
    #   own=stand_on → target=give_way
    #   own=none     → target=none
    if own_role == "give_way" and encounter_type == "head_on":
        target_role = "give_way"   # both give way in head-on
    elif own_role == "give_way":
        target_role = "stand_on"
    elif own_role == "stand_on":
        target_role = "give_way"
    else:
        target_role = "none"

    aspect = float(rel_brg_target)

    return {
        "type": encounter_type,
        "ownship_role": own_role,
        "target_role": target_role,
        "rel_bearing": rel_brg,
        "rel_bearing_target": rel_brg_target,
        "distance": d,
        "aspect": aspect,
    }


def _determine_encounter(
    rel_brg: float,
    rel_brg_target: float,
) -> tuple[str, str]:
    """Determine COLREGs encounter type from relative bearings and aspect.

    Returns (encounter_type, ownship_role).
    """
    # Head-on: both vessels see each other ahead (~0 deg relative)
    h_min, h_max = HEAD_ON_BEARING_RANGE
    is_head_on_own = _in_bearing_range(rel_brg, h_min, h_max)
    is_head_on_tgt = _in_bearing_range(rel_brg_target, h_min, h_max)

    if is_head_on_own and is_head_on_tgt:
        return "head_on", "give_way"  # both give way in head-on

    # Overtaking: check BOTH directions.
    # - If target is in ownship's overtaking sector (astern of ownship),
    #   ownship is BEING OVERTAKEN → stand-on.
    # - If ownship is in target's overtaking sector (astern of target),
    #   ownship is OVERTAKING → give-way (Rule 13).
    o_min, o_max = OVERTAKING_RANGE
    is_being_overtaken = _in_bearing_range(rel_brg, o_min, o_max)
    is_overtaking = _in_bearing_range(rel_brg_target, o_min, o_max)

    if is_overtaking:
        # Ownship is overtaking vessel → give-way (Rule 13)
        return "overtaking_giveway", "give_way"
    if is_being_overtaken:
        # Ownship is being overtaken → stand-on (Rule 13)
        return "overtaking_standon", "stand_on"

    # Crossing
    c_give_min, c_give_max = CROSSING_GIVEWAY_RANGE
    if _in_bearing_range(rel_brg, c_give_min, c_give_max):
        return "crossing_giveway", "give_way"

    c_stand_min, c_stand_max = CROSSING_STANDON_RANGE
    if _in_bearing_range(rel_brg, c_stand_min, c_stand_max):
        return "crossing_standon", "stand_on"

    return "none", "none"


def _in_bearing_range(bearing: float, min_b: float, max_b: float) -> bool:
    """Check if bearing [deg] falls within [min_b, max_b] considering wrap-around."""
    if min_b > max_b:
        # Range wraps through 0/360
        return bearing >= min_b or bearing <= max_b
    return min_b <= bearing <= max_b

