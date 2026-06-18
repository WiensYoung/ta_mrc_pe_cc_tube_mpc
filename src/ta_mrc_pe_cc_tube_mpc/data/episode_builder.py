"""Episode builder: constructs EpisodeConfig from YAML scenario definitions.

Supports:
  - Synthetic scenarios from YAML configs
  - Real AIS-derived episodes
  - ENC layer loading from processed JSON
  - Time-varying environment sequences via environment_loader
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np

from ..constants import KNOTS_TO_MPS
from ..types import EpisodeConfig, EnvironmentState, TargetShip, VesselParams, VesselState
from ..utils.io_utils import load_yaml


def _waterway_type_from_id(waterway_id: str) -> str:
    """Map waterway_id to a canonical waterway type for AIS quality lookup."""
    w = (waterway_id or "").lower()
    if any(k in w for k in ("harbour", "harbor", "port", "kill_van_kull", "east_river")):
        return "harbour"
    if any(k in w for k in ("channel", "river", "narrow", "restricted", "tss")):
        return "restricted_channel"
    if any(k in w for k in ("coastal", "bay", "sound", "puget", "san_francisco")):
        return "coastal"
    return "coastal"


def _get_ais_quality_for_target(vessel_type: str, waterway_id: str) -> dict:
    """Get stratified AIS quality defaults for a target vessel (GAP #7)."""
    try:
        from .ais_schema import get_ais_quality_for_vessel as _gaq
        return _gaq(vessel_type, _waterway_type_from_id(waterway_id))
    except Exception:
        return {}


# Waterway ENC layer paths (relative to project root)
_WATERWAY_ENC_PATHS = {
    "puget_sound": "data/processed/enc_layer_puget_sound.json",
    "puget_sound_tss": "data/processed/enc_layer_puget_sound.json",
    "puget_sound_crossing": "data/processed/enc_layer_puget_sound.json",
    "new_york_harbor": "data/processed/enc_layer_new_york_harbor.json",
    "new_york_harbor_nj": "data/processed/enc_layer_new_york_harbor_nj.json",
    "kill_van_kull": "data/processed/enc_layer_new_york_harbor.json",
    "san_francisco_bay": "data/processed/enc_layer_san_francisco_bay.json",
}


def _get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))


def _load_enc_layer(waterway_id: str):
    """Load ENC layer data for a waterway if available.

    Returns EncLayer object or None if not available.
    """
    if waterway_id not in _WATERWAY_ENC_PATHS:
        return None

    enc_path = os.path.join(_get_project_root(), _WATERWAY_ENC_PATHS[waterway_id])
    if not os.path.exists(enc_path):
        return None

    try:
        from .enc_layers import EncLayer
        return EncLayer.from_json(enc_path)
    except Exception:
        return None


def _build_environment_sequence(
    env_cfg: dict,
    duration: float,
    dt: float,
    waterway_id: str,
    ownship_track: list[tuple[float, float, float]] = None,
) -> list[EnvironmentState]:
    """Build time-varying environment sequence.

    Uses environment_loader if available, otherwise creates constant sequence.
    """
    n_steps = int(duration / dt) + 1

    try:
        from .environment_loader import build_environment_sequence as bes
        # Check if we have gridded fields available
        current_field = env_cfg.get("current_field")
        wind_field = env_cfg.get("wind_field")
        if current_field is not None or wind_field is not None:
            return bes(
                waterway_id=waterway_id,
                duration=duration,
                dt=dt,
                water_depth=env_cfg.get("water_depth", 50.0),
                channel_width=env_cfg.get("channel_width"),
                bank_distance_left=env_cfg.get("bank_distance_left"),
                bank_distance_right=env_cfg.get("bank_distance_right"),
                ownship_track=ownship_track,
            )
    except ImportError:
        logger.debug("VTS rules module not available for environment loading")

    # Constant environment fallback.
    # BUG #11 fix (2026-06-10): `[base_env] * n_steps` creates N references
    # to the SAME object. Use a list comprehension to create independent copies.
    base_env = EnvironmentState(
        water_depth=env_cfg.get("water_depth", 50.0),
        current_x=env_cfg.get("current_x", 0.0),
        current_y=env_cfg.get("current_y", 0.0),
        wind_x=env_cfg.get("wind_x", 0.0),
        wind_y=env_cfg.get("wind_y", 0.0),
        bank_distance_left=env_cfg.get("bank_distance_left"),
        bank_distance_right=env_cfg.get("bank_distance_right"),
        channel_width=env_cfg.get("channel_width"),
        waterway_id=waterway_id,
    )
    return [EnvironmentState(
        water_depth=base_env.water_depth,
        current_x=base_env.current_x,
        current_y=base_env.current_y,
        wind_x=base_env.wind_x,
        wind_y=base_env.wind_y,
        bank_distance_left=base_env.bank_distance_left,
        bank_distance_right=base_env.bank_distance_right,
        channel_width=base_env.channel_width,
        waterway_id=base_env.waterway_id,
    ) for _ in range(n_steps)]


def build_episode_from_scenario(
    scenario_config: dict,
    scenario_id: str,
    seed: int = 42,
    load_enc: bool = True,
) -> EpisodeConfig:
    """Build an EpisodeConfig from a scenario dictionary (from YAML).

    Args:
        scenario_config: Parsed scenario configuration.
        scenario_id: Scenario identifier (e.g. 'S1').
        seed: Random seed.
        load_enc: Whether to load ENC layer data for the waterway.

    Returns:
        EpisodeConfig ready for simulation.
    """
    sc = scenario_config["scenarios"][scenario_id]

    waterway_id = sc.get("waterway", "")

    # Ownship initial state
    own_init = sc.get("ownship_initial", {})
    own_state = VesselState(
        x=own_init.get("x", 0.0),
        y=own_init.get("y", 0.0),
        psi=own_init.get("psi", 0.0),
        u=own_init.get("u", 0.0),
        v=own_init.get("v", 0.0),
        r=own_init.get("r", 0.0),
    )

    # Ownship params (defaults, can be overridden)
    own_params = VesselParams()

    # Target ships
    targets = []
    for tgt_cfg in sc.get("targets", []):
        t_init = tgt_cfg["initial"]
        tgt_state = VesselState(
            x=t_init.get("x", 0.0),
            y=t_init.get("y", 0.0),
            psi=t_init.get("psi", 0.0),
            u=t_init.get("u", 0.0),
            v=t_init.get("v", 0.0),
            r=t_init.get("r", 0.0),
        )
        tgt = TargetShip(
            mmsi=tgt_cfg.get("mmsi", ""),
            state=tgt_state,
            length=tgt_cfg.get("length", 100.0),
            beam=tgt_cfg.get("beam", 15.0),
            draught=tgt_cfg.get("draught", 5.0),
            vessel_type=tgt_cfg.get("vessel_type", "cargo"),
            nav_status=tgt_cfg.get("nav_status", "underway"),
            cog=float(np.rad2deg(tgt_state.psi) % 360.0),
            sog=float(tgt_state.speed() / KNOTS_TO_MPS),
            heading=float(np.rad2deg(tgt_state.psi) % 360.0),
            covariance=np.eye(2) * 100.0,
            # GAP #7 (2026-06-10): Use stratified AIS quality defaults
            # from ais_schema when no explicit ais_config is provided.
            ais_quality=tgt_cfg.get("ais_config") or _get_ais_quality_for_target(
                tgt_cfg.get("vessel_type", "cargo"), waterway_id,
            ),
        )
        targets.append(tgt)

    # Environment — build time-varying sequence
    env_cfg = sc.get("environment", {})
    duration = sc.get("duration", 600.0)
    dt = sc.get("dt", 0.5)

    # Build ownship track for environment field sampling
    n_steps = int(duration / dt) + 1
    ownship_track = []
    for k in range(n_steps):
        t = k * dt
        ownship_track.append((own_state.x, own_state.y, t))

    env_sequence = _build_environment_sequence(
        env_cfg, duration, dt, waterway_id, ownship_track,
    )

    # Load ENC layer
    enc_layer = None
    if load_enc:
        enc_layer = _load_enc_layer(waterway_id)

    # Build perturbation config with ENC metadata
    pert_config = {}
    if enc_layer is not None:
        pert_config["enc_layer_loaded"] = True
        pert_config["enc_buoys"] = len(enc_layer.buoy_positions) if hasattr(enc_layer, "buoy_positions") else 0

    # Load VTS zone for the waterway
    vts_zone = None
    coast_pilot_rules = None
    try:
        from .vts_rules import (
            build_coast_pilot_rules,
            build_new_york_vts,
            build_puget_sound_vts,
            build_san_francisco_vts,
        )
        if waterway_id in ("puget_sound", "puget_sound_tss", "puget_sound_crossing"):
            vts_zone = build_puget_sound_vts()
        elif waterway_id in ("san_francisco_bay",):
            vts_zone = build_san_francisco_vts()
        elif waterway_id in ("new_york_harbor", "new_york_harbor_nj", "kill_van_kull"):
            vts_zone = build_new_york_vts()

        # Map waterway_id to canonical for Coast Pilot rules
        cp_waterway = waterway_id
        if waterway_id in ("puget_sound_tss", "puget_sound_crossing"):
            cp_waterway = "puget_sound"
        elif waterway_id == "kill_van_kull":
            cp_waterway = "new_york_harbor"
        coast_pilot_rules = build_coast_pilot_rules(cp_waterway)
    except Exception:
        logger.debug("Failed to load VTS/coast pilot rules for %s", waterway_id, exc_info=True)

    episode = EpisodeConfig(
        scenario_id=scenario_id,
        waterway=waterway_id,
        duration=duration,
        dt=dt,
        ownship_initial_state=own_state,
        ownship_params=own_params,
        targets=targets,
        environment_sequence=env_sequence,
        random_seed=seed,
        perturbation_config=pert_config,
        enc_layer=enc_layer,
        vts_zone=vts_zone,
        coast_pilot_rules=coast_pilot_rules,
    )

    return episode


def build_episode_from_yaml(
    scenario_path: str,
    scenario_id: str,
    seed: int = 42,
    load_enc: bool = True,
) -> EpisodeConfig:
    """Load a scenario from a YAML file and build an EpisodeConfig.

    Args:
        scenario_path: Path to the scenario YAML file.
        scenario_id: Scenario identifier.
        seed: Random seed.
        load_enc: Whether to load ENC layer data.

    Returns:
        EpisodeConfig.
    """
    config = load_yaml(scenario_path)
    return build_episode_from_scenario(config, scenario_id, seed, load_enc)
