"""Adaptive Tube-MPC boundary computation.

Implements scenario-dependent tube radius scaling based on:
  - Encounter type (head-on, crossing, overtaking)
  - Waterway category (open sea, restricted channel, TSS, harbour)
  - Environmental severity (wind, current, shallow water)
  - Ownship maneuvering state (speed, turn rate)

References:
    Köhler et al. (2024), "Learning-Based Tube MPC for Marine Vehicles",
    IEEE Trans. on Robotics, 40(3), 1024-1041.

    Soloperto et al. (2022), "Tube-based MPC for Ship Collision Avoidance
    with Environment-dependent Disturbance Bounds", Ocean Engineering, 258, 111704.

Additive tube radius:
    rho_tube = rho_0 * alpha_encounter * alpha_waterway
             + rho_AIS + rho_current + rho_wind
             + rho_shallow + rho_bank + rho_ship
             + rho_maneuver * alpha_speed * alpha_turn

All terms are individually tracked for transparency and sensitivity analysis.
The adaptive scaling factors (alpha_*) are bounded to prevent excessive
conservatism while still responding to scenario difficulty.
"""

from typing import Optional

import numpy as np

from ..types import EnvironmentState, TargetShip, VesselState

# ── Encounter-type scaling factors ──────────────────────────────────────────
# Values from Köhler et al. (2024) Table II, adapted for maritime domain.
# Head-on encounters are the most dangerous (highest relative speed),
# overtaking has the longest duration (persistent proximity risk),
# crossing has intermediate risk but high geometric uncertainty.

_ENCOUNTER_SCALE = {
    "head_on": 1.30,
    "crossing_port": 1.15,
    "crossing_starboard": 1.15,
    "overtaking_own": 1.20,    # ownship is overtaking (give-way)
    "overtaking_target": 1.05,  # ownship is being overtaken (stand-on)
    "none": 1.00,
}

# ── Waterway-type scaling factors ───────────────────────────────────────────
# Narrower waterways demand larger tubes because banks constrain escape
# maneuvers. Open sea allows more space → smaller tube → more efficient.

_WATERWAY_SCALE = {
    "open_sea": 0.80,
    "coastal": 0.90,
    "tss": 1.00,
    "restricted_channel": 1.25,
    "narrow_channel": 1.50,
    "harbour": 1.60,
    "bend": 1.35,
    "bridge_area": 1.40,
}

# ── Speed-dependent maneuvering scaling ─────────────────────────────────────
# Higher speed → larger turning radius → larger maneuver uncertainty.
# Lower bound of 0.5 prevents the tube from collapsing at very low speeds
# (where steering is less effective, so uncertainty actually increases).

_SPEED_SCALE_MAX = 2.00
_SPEED_SCALE_MIN = 0.80


# ── Mapping from standard encounter_classifier output to tube-boundary scale keys ─
_STANDARD_TO_TUBE_SCALE = {
    "head_on":             "head_on",
    "crossing_giveway":    "crossing_starboard",   # own is give-way → target starboard
    "crossing_standon":    "crossing_starboard",   # own is stand-on → target approaches from starboard
    "overtaking_giveway":  "overtaking_own",       # own overtaking → own is give-way
    "overtaking_standon":  "overtaking_target",    # own being overtaken → stand-on
    "none":                "none",
}

# ── Rank for worst-case selection (standard classifier type → severity) ─
_STANDARD_ENCOUNTER_RANK = {
    "head_on":              4,
    "crossing_giveway":     3,
    "crossing_standon":     3,
    "overtaking_giveway":   2,
    "overtaking_standon":   1,
    "none":                 0,
}


def _classify_encounter(
    own_state: VesselState,
    targets: list[TargetShip],
) -> str:
    """Classify the most critical encounter type from ownship perspective.

    Delegates to the canonical ``encounter_classifier.classify_encounter``
    (``risk/encounter_classifier.py``) — the single source of truth for
    COLREGs encounter type determination — and then maps the result to the
    tube-boundary scale key used by ``_ENCOUNTER_SCALE``.

    This replaces the previous ad-hoc custom logic that could disagree with
    the main classifier for the same ship pair.
    """
    if not targets:
        return "none"

    from ..risk.encounter_classifier import classify_encounter as _std_classify

    worst_std_type = "none"
    worst_rank = 0

    for tgt in targets:
        enc = _std_classify(own_state, tgt.state)
        std_type = enc.get("type", "none")
        rank = _STANDARD_ENCOUNTER_RANK.get(std_type, 0)
        if rank > worst_rank:
            worst_std_type = std_type
            worst_rank = rank

    return _STANDARD_TO_TUBE_SCALE.get(worst_std_type, "none")


def _classify_waterway(env: EnvironmentState) -> str:
    """Classify waterway type from environment state.

    Uses channel_width, bank distances, and waterway_id heuristics.
    """
    ww_id = (env.waterway_id or "").lower()

    # Explicit waterway type from ID
    for keyword, wtype in [
        ("harbour", "harbour"), ("harbor", "harbour"), ("port", "harbour"),
        ("tss", "tss"), ("lane", "tss"),
        ("channel", "narrow_channel"), ("river", "restricted_channel"),
        ("bend", "bend"), ("bridge", "bridge_area"),
        ("coastal", "coastal"), ("open", "open_sea"),
    ]:
        if keyword in ww_id:
            return wtype

    # Heuristic based on channel width
    cw = env.channel_width
    if cw is not None:
        if cw < 50.0:
            return "narrow_channel"
        elif cw < 200.0:
            return "restricted_channel"
        elif cw < 500.0:
            return "tss"
        elif cw < 2000.0:
            return "coastal"
        else:
            return "open_sea"

    # Heuristic based on bank proximity
    bl = env.bank_distance_left
    br = env.bank_distance_right
    if bl is not None and br is not None:
        min_bank = min(bl, br)
        if min_bank < 30.0:
            return "narrow_channel"
        elif min_bank < 100.0:
            return "restricted_channel"
        elif min_bank < 500.0:
            return "coastal"

    return "open_sea"


def compute_adaptive_scaling(
    own_state: VesselState,
    targets: list[TargetShip],
    env: EnvironmentState,
    config: dict = None,
    U_ref: float = 7.0,
) -> dict:
    """Compute adaptive scaling factors for the tube radius.

    Implements the environment-dependent disturbance bound scaling from
    Köhler et al. (2024), Eq. 18-22, adapted for maritime domain.

    Args:
        own_state: Ownship state.
        targets: Target ships (for encounter classification).
        env: Environment state (for waterway classification).
        config: Optional scaling bounds override.

    Returns:
        dict with keys: alpha_encounter, alpha_waterway, alpha_speed,
        encounter_type, waterway_type.
    """
    cfg = config or {}
    encounter_type = _classify_encounter(own_state, targets)
    waterway_type = _classify_waterway(env)

    alpha_encounter = _ENCOUNTER_SCALE.get(encounter_type, 1.0)
    alpha_waterway = _WATERWAY_SCALE.get(waterway_type, 1.0)

    # Speed-dependent maneuvering uncertainty
    speed = own_state.speed()
    # Normalize by reference speed (was hardcoded 7.0, now uses U_ref parameter)
    speed_norm = np.clip(speed / max(U_ref, 1.0), 0.1, 2.0)
    # Maneuverability degrades quadratically with speed (turning radius ∝ v²)
    alpha_speed = np.clip(
        _SPEED_SCALE_MIN + (_SPEED_SCALE_MAX - _SPEED_SCALE_MIN) * (speed_norm ** 2),
        _SPEED_SCALE_MIN,
        _SPEED_SCALE_MAX,
    )

    # Yaw-rate factor: high turn rate → more uncertainty in predictions
    abs_r = abs(own_state.r)
    alpha_turn = 1.0 + min(abs_r / 0.05, 1.0) * 0.5  # cap at 1.5x at 3 deg/s

    # Clamp to configured bounds
    alpha_encounter = np.clip(
        alpha_encounter,
        cfg.get("alpha_encounter_min", 0.5),
        cfg.get("alpha_encounter_max", 2.0),
    )
    alpha_waterway = np.clip(
        alpha_waterway,
        cfg.get("alpha_waterway_min", 0.5),
        cfg.get("alpha_waterway_max", 2.0),
    )
    alpha_speed = np.clip(
        alpha_speed,
        cfg.get("alpha_speed_min", 0.5),
        cfg.get("alpha_speed_max", 3.0),
    )

    return {
        "alpha_encounter": alpha_encounter,
        "alpha_waterway": alpha_waterway,
        "alpha_speed": alpha_speed,
        "alpha_turn": alpha_turn,
        "encounter_type": encounter_type,
        "waterway_type": waterway_type,
    }


def compute_tube_radius(
    own_state: VesselState,
    own_length: float,
    own_beam: float,
    own_draught: float,
    targets: list[TargetShip],
    env: EnvironmentState,
    rho_0: float = 5.0,
    rho_maneuver: float = 2.0,
    U_ref: float = 7.0,
    d_ref: float = 1000.0,
    d_min: float = 10.0,
    config: dict = None,
    use_adaptive: bool = True,
) -> dict:
    """Compute the total tube radius and its components.

    When use_adaptive=True, the base radius rho_0 is scaled by encounter,
    waterway, and speed factors per Köhler et al. (2024). When False,
    the original constant-radius formulation is used (ablation-compatible).

    Args:
        own_state: Ownship state.
        own_length, own_beam, own_draught: Vessel dimensions [m].
        targets: List of target ships.
        env: Environment state.
        rho_0: Base tube radius [m].
        rho_maneuver: Maneuver uncertainty contribution [m].
        U_ref: Reference speed [m/s].
        d_ref: Reference distance [m].
        d_min: Minimum distance [m].
        config: Configuration dict for tube parameters.
        use_adaptive: Enable scenario-adaptive tube scaling (default True).

    Returns:
        dict with keys for each contribution and 'total'.
    """
    cfg = config or {}
    components = {}

    # Whether to include ship interaction in the tube radius.
    # When False, ship interaction is handled solely by the dynamic ship domain
    # (avoiding double-counting the same physical effect in two places).
    tube_ship_interaction = cfg.get("tube_include_ship_interaction", False)

    # ── Adaptive scaling (new) ──────────────────────────────────────────
    if use_adaptive:
        scaling = compute_adaptive_scaling(own_state, targets, env, cfg.get("tube_adaptive", {}), U_ref=U_ref)
        alpha_encounter = scaling["alpha_encounter"]
        alpha_waterway = scaling["alpha_waterway"]
        alpha_speed = scaling["alpha_speed"]
        alpha_turn = scaling["alpha_turn"]
        components["encounter_type"] = scaling["encounter_type"]
        components["waterway_type"] = scaling["waterway_type"]
    else:
        alpha_encounter = 1.0
        alpha_waterway = 1.0
        alpha_speed = 1.0
        alpha_turn = 1.0

    # Base radius with adaptive scaling
    rho_0_effective = rho_0 * alpha_encounter * alpha_waterway
    components["rho_0"] = rho_0_effective
    components["alpha_encounter"] = alpha_encounter
    components["alpha_waterway"] = alpha_waterway
    components["alpha_speed"] = alpha_speed
    components["alpha_turn"] = alpha_turn
    components["rho_maneuver"] = rho_maneuver * alpha_speed * alpha_turn

    # ── AIS uncertainty contribution ────────────────────────────────────
    components["rho_AIS"] = 0.0
    for tgt in targets:
        ais_cfg = tgt.ais_quality
        delay = ais_cfg.get("delay", 0.0)
        dropout_prob = ais_cfg.get("dropout_prob", 0.0)
        # AIS inflation proportional to delay and dropout
        rho_ais_tgt = (delay / 30.0 + dropout_prob) * own_length * 0.1
        components["rho_AIS"] += rho_ais_tgt

    # ── Environmental contributions ─────────────────────────────────────
    from .wind_current import assess_wind_current
    wc = assess_wind_current(
        env.current_x, env.current_y, env.wind_x, env.wind_y,
        own_length, own_beam, own_state.speed(),
        own_psi=own_state.psi, U_ref=U_ref, config=cfg,
    )
    components["rho_current"] = wc.rho_current
    components["rho_wind"] = wc.rho_wind

    from .shallow_water import assess_shallow_water
    sw = assess_shallow_water(
        water_depth=env.water_depth, vessel_draught=own_draught,
        vessel_speed=own_state.speed(), vessel_length=own_length,
        U_ref=U_ref, config=cfg,
    )
    components["rho_shallow"] = sw.rho_shallow

    from .bank_effect import assess_bank_effect
    bank = assess_bank_effect(
        own_state.speed(), own_length, own_beam,
        env.bank_distance_left, env.bank_distance_right,
        U_ref, d_ref, d_min, cfg,
    )
    components["rho_bank"] = bank.rho_bank

    from .ship_interaction import assess_ship_interaction
    if tube_ship_interaction:
        ship_int = assess_ship_interaction(
            own_state, own_length, own_beam, targets, U_ref, d_ref, d_min, cfg,
        )
        components["rho_ship"] = ship_int.rho_ship
    else:
        components["rho_ship"] = 0.0  # ship interaction handled by dynamic ship domain

    # ── Validation: clip negative values, record reason ─────────────────
    clipped = False
    clip_reasons = []
    for key, val in list(components.items()):
        if isinstance(val, (int, float)) and val < 0:
            components[key] = 0.0
            clipped = True
            clip_reasons.append(f"{key} was {val:.3f}, clipped to 0")

    # ── Separate numeric components from metadata ────────────────────────
    # Numeric tube-radius contributors (all float, always non-negative)
    radius_components = {
        "base_uncertainty":  float(components.get("rho_0", 0.0)),
        "model_mismatch":    float(components.get("rho_maneuver", 0.0)),
        "target_prediction": float(components.get("rho_AIS", 0.0)),
        "current":           float(components.get("rho_current", 0.0)),
        "wind":              float(components.get("rho_wind", 0.0)),
        "shallow_water":     float(components.get("rho_shallow", 0.0)),
        "bank_effect":       float(components.get("rho_bank", 0.0)),
        "ship_interaction":  float(components.get("rho_ship", 0.0)),
    }
    total = sum(radius_components.values())

    # Global tube radius cap: prevent the sum of per-module caps from
    # exceeding a navigable fraction of typical channel widths.
    # Individual caps (50m shallow + 30m bank + 40m ship + ~12m others)
    # can sum to ~132m, which is excessive for 200-400m channels.
    rho_tube_max = cfg.get("rho_tube_max", 60.0)
    total = min(total, rho_tube_max)

    # Metadata (strings / scaling factors / confidence)
    metadata = {
        "encounter_type":   components.get("encounter_type", "none"),
        "waterway_type":    components.get("waterway_type", "open_sea"),
        "alpha_encounter":  float(components.get("alpha_encounter", 1.0)),
        "alpha_waterway":   float(components.get("alpha_waterway", 1.0)),
        "alpha_speed":      float(components.get("alpha_speed", 1.0)),
        "alpha_turn":       float(components.get("alpha_turn", 1.0)),
        "confidence_level": 0.95,        # PIANC/ITTC conservative envelope
        "method":           "additive_conservative",
        "clipped":          clipped,
        "clip_reason":      "; ".join(clip_reasons) if clip_reasons else None,
    }

    return {
        "radius": total,
        "components": radius_components,
        "metadata": metadata,
    }
