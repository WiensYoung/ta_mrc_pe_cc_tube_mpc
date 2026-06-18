"""Waterway-specific rules: TSS, ATBA, VTS, channel constraints.

Handles traffic separation schemes, precautionary areas,
and local waterway management rules.
"""

import numpy as np


def check_tss_compliance(
    own_y: float,
    lane_center: float,
    lane_half_width: float,
    separation_zone_half_width: float = 100.0,
) -> dict:
    """Check TSS lane compliance.

    Args:
        own_y: Ownship lateral position [m].
        lane_center: Lane center y-coordinate [m].
        lane_half_width: Half-width of the traffic lane [m].
        separation_zone_half_width: Half-width of the separation zone [m].

    Returns:
        dict with compliance flags and margins.
    """
    lateral_offset = own_y - lane_center
    margin_to_boundary = lane_half_width - abs(lateral_offset)
    sep_zone_left = lane_center - separation_zone_half_width
    sep_zone_right = lane_center + separation_zone_half_width

    result = {
        "in_lane": abs(lateral_offset) <= lane_half_width,
        "margin_to_lane_boundary": margin_to_boundary,
        "in_separation_zone": sep_zone_left <= own_y <= sep_zone_right,
        "intrusion_distance": 0.0,
    }

    if not result["in_lane"]:
        result["intrusion_distance"] = abs(lateral_offset) - lane_half_width

    return result


def check_channel_boundary(
    own_x: float,
    own_y: float,
    channel_left: float,
    channel_right: float,
) -> dict:
    """Check if ownship is within channel boundaries.

    Args:
        own_x: Ownship x-position [m] (reserved for future 2D checks).
        own_y: Ownship y-position [m].
        channel_left, channel_right: Left and right boundaries (y-coordinates) [m].

    Returns:
        dict with violation flag and margin.
    """
    margin_left = own_y - channel_left
    margin_right = channel_right - own_y
    min_margin = min(margin_left, margin_right)

    return {
        "within_channel": margin_left >= 0 and margin_right >= 0,
        "margin_left": margin_left,
        "margin_right": margin_right,
        "min_margin": min_margin,
        "violation_distance": abs(min_margin) if min_margin < 0 else 0.0,
    }


def check_atba(
    own_pos: np.ndarray,
    atba_center: np.ndarray,
    atba_radius: float,
) -> dict:
    """Check ATBA (Area To Be Avoided) compliance.

    Args:
        own_pos: Ownship position [x, y].
        atba_center: ATBA center [x, y].
        atba_radius: ATBA radius [m].

    Returns:
        dict with intrusion flag and distance.
    """
    d = float(np.linalg.norm(own_pos[:2] - atba_center[:2]))
    return {
        "intruding": d < atba_radius,
        "distance_to_atba": d - atba_radius,
        "atba_radius": atba_radius,
    }


def check_precautionary_area(
    own_pos: np.ndarray,
    area_center: np.ndarray,
    area_radius: float,
) -> dict:
    """Check presence in a precautionary area (requires extra vigilance).

    Args:
        own_pos: Ownship position [x, y].
        area_center: Area center [x, y].
        area_radius: Area radius [m].

    Returns:
        dict with presence flag.
    """
    d = float(np.linalg.norm(own_pos[:2] - area_center[:2]))
    return {
        "in_area": d < area_radius,
        "distance_to_center": d,
        "area_radius": area_radius,
    }


def compute_waterway_safety_scale(
    waterway_id: str,
    channel_width: float = None,
    in_precautionary_area: bool = False,
    in_atba: bool = False,
) -> float:
    """Compute a safety domain scale factor based on waterway characteristics.

    Returns a multiplier >= 1.0 for the safety domain.
    Uses ADDITIVE increments (+=) rather than multiplicative (*=) to prevent
    compounding explosion.  Capped at 1.5 to keep constraints feasible.
    """
    scale = 1.0

    # TSS lanes: tighter safety requirements
    if waterway_id and "tss" in waterway_id.lower():
        scale += 0.10

    # Precautionary areas: extra caution
    if in_precautionary_area:
        scale += 0.10

    # ATBA: maximum caution
    if in_atba:
        scale += 0.10  # was no-op; now adds explicit margin

    # Narrow channels: increase domain
    if channel_width is not None and channel_width < 500.0:
        scale += 0.05

    # Cap at 1.5 to prevent constraint infeasibility
    return min(scale, 1.5)
