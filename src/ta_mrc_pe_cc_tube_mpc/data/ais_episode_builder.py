"""AIS episode builder: extract real encounter fragments and build EpisodeConfig.

Pipeline:
  1. Scan preprocessed AIS data for CPA encounters between vessel pairs.
  2. Select encounter window (t_cpa +- window_half).
  3. Build TargetShip objects with inferred covariance and ais_quality metadata.
  4. Construct EpisodeConfig with real initial states, ENC environment, etc.

Supports both in-memory (DataFrame) and chunked (CSV path) processing for
large datasets that exceed available RAM.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from ..types import (
    EnvironmentState,
    EpisodeConfig,
    TargetShip,
    VesselParams,
    VesselState,
)
from ..utils.coordinates import normalize_angle_deg
from ..constants import KNOTS_TO_MPS


def extract_encounters(
    df: pd.DataFrame,
    ownship_mmsi: str,
    cpa_threshold_m: float = 2000.0,
    min_duration_s: float = 120.0,
    min_target_points: int = 5,
) -> list[dict]:
    """Scan AIS data for CPA encounters between ownship and target vessels.

    Uses the preprocessed DataFrame with local (x, y) coordinates.

    Args:
        df: Preprocessed AIS DataFrame with columns: mmsi, timestamp, x, y, sog, cog,
            heading, vessel_type, nav_status, length, beam, draught.
        ownship_mmsi: MMSI of the ownship vessel.
        cpa_threshold_m: Maximum CPA distance [m] to qualify as an encounter.
        min_duration_s: Minimum encounter duration [s] (ownship must be near target
            for at least this long).
        min_target_points: Minimum AIS records for a target in the encounter window.

    Returns:
        List of encounter dicts, each with:
            - ownship_mmsi, target_mmsi, cpa_m, t_cpa (timestamp)
            - encounter_start, encounter_end (timestamp)
            - ownship_track, target_track (DataFrame slices)
    """
    # Coerce MMSI column to same type as ownship_mmsi to handle
    # mixed-type CSV data (string vs int MMSI values).
    # Work on a copy to avoid mutating the caller's DataFrame.
    df = df.copy()
    df["mmsi"] = df["mmsi"].astype(type(ownship_mmsi))
    own = df[df["mmsi"] == ownship_mmsi].copy()
    if len(own) < 2:
        return []

    own = own.sort_values("timestamp")
    other_mmsis = df[df["mmsi"] != ownship_mmsi]["mmsi"].unique()

    encounters = []
    for tgt_mmsi in other_mmsis:
        tgt = df[df["mmsi"] == tgt_mmsi].sort_values("timestamp")
        if len(tgt) < min_target_points:
            continue

        # Find overlapping time window
        t_min = max(own["timestamp"].min(), tgt["timestamp"].min())
        t_max = min(own["timestamp"].max(), tgt["timestamp"].max())
        if t_max - t_min < min_duration_s:
            continue

        # Compute pairwise distances at common timestamps
        # Use merge_asof for approximate time matching (within 10s)
        ow = own[(own["timestamp"] >= t_min) & (own["timestamp"] <= t_max)]
        tg = tgt[(tgt["timestamp"] >= t_min) & (tgt["timestamp"] <= t_max)]

        if len(ow) < 2 or len(tg) < 2:
            continue

        # Merge on nearest timestamp
        merged = pd.merge_asof(
            ow[["timestamp", "x", "y"]].sort_values("timestamp"),
            tg[["timestamp", "x", "y"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=30,  # 30s tolerance
            suffixes=("_own", "_tgt"),
        ).dropna()

        if len(merged) < min_target_points:
            continue

        # Compute distances
        dx = merged["x_own"].values - merged["x_tgt"].values
        dy = merged["y_own"].values - merged["y_tgt"].values
        distances = np.sqrt(dx**2 + dy**2)

        min_idx = int(np.argmin(distances))
        cpa_m = float(distances[min_idx])
        t_cpa = float(merged["timestamp"].iloc[min_idx])

        if cpa_m > cpa_threshold_m:
            continue

        # Encounters are contiguous windows where distance < cpa_threshold_m * 2
        near_mask = distances < cpa_threshold_m * 2
        if not near_mask.any():
            continue

        # Find contiguous near segments
        segments = _find_contiguous_segments(near_mask)
        for seg_start, seg_end in segments:
            seg_duration = float(
                merged["timestamp"].iloc[seg_end]
                - merged["timestamp"].iloc[seg_start]
            )
            if seg_duration < min_duration_s:
                continue

            encounter_start = float(merged["timestamp"].iloc[seg_start])
            encounter_end = float(merged["timestamp"].iloc[seg_end])

            # Extract track slices
            own_track = own[
                (own["timestamp"] >= encounter_start)
                & (own["timestamp"] <= encounter_end)
            ]
            tgt_track = tgt[
                (tgt["timestamp"] >= encounter_start)
                & (tgt["timestamp"] <= encounter_end)
            ]

            if len(own_track) < 2 or len(tgt_track) < min_target_points:
                continue

            encounters.append({
                "ownship_mmsi": str(ownship_mmsi),
                "target_mmsi": str(tgt_mmsi),
                "cpa_m": cpa_m,
                "t_cpa": t_cpa,
                "encounter_start": encounter_start,
                "encounter_end": encounter_end,
                "ownship_track": own_track,
                "target_track": tgt_track,
            })

    # Sort by CPA distance (closest encounters first)
    encounters.sort(key=lambda e: e["cpa_m"])
    return encounters


def _find_contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True segments in a boolean mask."""
    segments = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def build_target_from_track(
    track: pd.DataFrame,
    mmsi: str,
    at_time: Optional[float] = None,
) -> TargetShip:
    """Build a TargetShip from an AIS track slice.

    Infers position covariance from measurement spacing and AIS accuracy metadata.
    Computes ais_quality metadata dict from track statistics.

    Args:
        track: AIS track DataFrame for the target vessel (must have x, y, sog, cog,
            heading, vessel_type, nav_status, length, beam, draught columns).
        mmsi: Target MMSI as string.
        at_time: Specific timestamp to extract state for. If None, uses first record.

    Returns:
        TargetShip with state, parameters, covariance, and ais_quality.
    """
    track = track.sort_values("timestamp")

    if at_time is not None:
        idx = (track["timestamp"] - at_time).abs().idxmin()
        row = track.loc[idx]
    else:
        row = track.iloc[0]

    # State at the selected time
    x = float(row["x"]) if not pd.isna(row.get("x", np.nan)) else 0.0
    y = float(row["y"]) if not pd.isna(row.get("y", np.nan)) else 0.0
    sog_val = row.get("sog", 0)
    cog_val = row.get("cog", 0)
    sog_kn = float(sog_val) if not pd.isna(sog_val) else 0.0
    cog_deg = float(cog_val) if not pd.isna(cog_val) else 0.0
    hdg_val = row.get("heading", np.nan)
    heading = float(hdg_val) if not pd.isna(hdg_val) else None

    psi = np.deg2rad(cog_deg) if cog_deg != 0 else 0.0
    u = sog_kn * KNOTS_TO_MPS
    v = 0.0

    state = VesselState(x=x, y=y, psi=psi, u=u, v=v, r=0.0)

    len_val = row.get("length", 100)
    beam_raw = row.get("beam", 15)
    draught_raw = row.get("draught", 5)
    length = float(len_val) if not pd.isna(len_val) else 100.0
    beam_val = float(beam_raw) if not pd.isna(beam_raw) else 15.0
    draught = float(draught_raw) if not pd.isna(draught_raw) else 5.0

    vt = row.get("vessel_type", 0)
    vessel_type = _map_vessel_type(vt)

    ns = row.get("nav_status", 0)
    nav_status = _map_nav_status(ns)

    # Infer covariance from track statistics
    covariance = _infer_covariance(track)

    # Build ais_quality dict
    ais_quality = _build_ais_quality(track)

    return TargetShip(
        mmsi=mmsi,
        state=state,
        length=length,
        beam=beam_val,
        draught=draught,
        vessel_type=vessel_type,
        nav_status=nav_status,
        cog=cog_deg,
        sog=sog_kn,
        heading=heading if heading is not None else None,
        covariance=covariance,
        ais_quality=ais_quality,
    )


def _infer_covariance(track: pd.DataFrame) -> np.ndarray:
    """Infer 2x2 position covariance [m^2] from AIS track statistics.

    Uses:
      - Time gaps between messages (larger gaps → higher uncertainty)
      - Position changes between consecutive reports
      - AIS position_accuracy field if available

    Returns:
        2x2 covariance matrix in ENU coordinates.
    """
    base_var = 100.0  # 10m std default

    # Use position_accuracy if available
    # AIS spec: accuracy=1 → DGPS/high accuracy (<10m)
    #            accuracy=0 → standard GPS/low accuracy (>10m)
    if "position_accuracy" in track.columns:
        pa = track["position_accuracy"].dropna()
        if len(pa) > 0 and pa.iloc[0] == 1:
            base_var = 25.0  # 5m std for DGPS (high accuracy)
        elif len(pa) > 0 and pa.iloc[0] == 0:
            base_var = 100.0  # 10m std for standard GPS

    # Estimate from time gaps
    if len(track) >= 2:
        ts = track["timestamp"].values
        gaps = np.diff(ts)
        gaps_pos = gaps[gaps > 0.1]
        median_gap = float(np.median(gaps_pos)) if len(gaps_pos) > 0 else 10.0

        # Position jitter
        if "x" in track.columns and "y" in track.columns:
            dx = np.diff(track["x"].dropna().values)
            dy = np.diff(track["y"].dropna().values)
            if len(dx) > 1:
                pos_var = float(np.var(dx) + np.var(dy))
                base_var = max(base_var, pos_var)

        # Growth from median gap (√time scaling)
        gap_factor = median_gap / 10.0  # normalize to 10s reporting
        base_var *= gap_factor

    base_var = np.clip(base_var, 25.0, 2500.0)  # 5m to 50m std
    return np.eye(2) * float(base_var)


def _build_ais_quality(track: pd.DataFrame) -> dict:
    """Build ais_quality metadata dict from track statistics.

    Returns dict with:
        - position_accuracy: 0 (DGPS <10m) or 1 (GPS >10m)
        - reporting_interval_s: median time between AIS messages
        - message_count: number of AIS messages in track
        - track_duration_s: duration of track slice
        - message_completeness: fraction of expected messages actually received
        - sog_variance: variance of SOG within track
        - cog_stability: 1 - circular std of COG (0-1, higher = more stable)
    """
    quality = {
        "position_accuracy": 1,
        "reporting_interval_s": 10.0,
        "message_count": len(track),
        "track_duration_s": 0.0,
        "message_completeness": 1.0,
        "sog_variance": 0.0,
        "cog_stability": 1.0,
    }

    if "position_accuracy" in track.columns:
        pa = track["position_accuracy"].dropna()
        if len(pa) > 0:
            quality["position_accuracy"] = int(pa.mode().iloc[0]) if len(pa.mode()) > 0 else 1

    if len(track) >= 2:
        ts = track["timestamp"].values
        gaps = np.diff(ts)
        gaps_pos = gaps[gaps > 0.1]  # filter sub-second duplicates
        quality["reporting_interval_s"] = float(np.median(gaps_pos)) if len(gaps_pos) > 0 else 10.0
        quality["track_duration_s"] = float(ts[-1] - ts[0])

        # Completeness: actual vs expected messages
        if quality["reporting_interval_s"] > 0.1:
            expected = quality["track_duration_s"] / quality["reporting_interval_s"]
            if expected > 0:
                quality["message_completeness"] = min(1.0, len(track) / expected)

        # SOG variance
        if "sog" in track.columns:
            sog_vals = track["sog"].dropna().values
            if len(sog_vals) > 1:
                quality["sog_variance"] = float(np.var(sog_vals))

        # COG stability (circular)
        if "cog" in track.columns:
            cog_vals = track["cog"].dropna().values
            if len(cog_vals) > 1:
                rad = np.deg2rad(cog_vals)
                sin_mean = np.mean(np.sin(rad))
                cos_mean = np.mean(np.cos(rad))
                r = np.sqrt(sin_mean**2 + cos_mean**2)
                quality["cog_stability"] = float(np.clip(r, 0.0, 1.0))

    return quality


def _map_vessel_type(code) -> str:
    """Map AIS vessel type code to string category."""
    try:
        code = int(code)  # handles int, float, and string representations
    except (ValueError, TypeError):
        code = 0
    if code in (30,):  # Fishing
        return "engaged_in_fishing"  # aligned with ais_schema.py NAV_STATUS_MAP[7]
    elif code in (31, 32):  # Tug
        return "tug_barge"
    elif code in (36, 37):  # Pleasure craft / sailing
        return "small_craft"
    elif code in (40, 41, 42, 43, 44, 45, 46, 47, 48, 49):  # High speed
        return "ferry"
    elif code in (50, 51, 52, 53, 54, 55):  # Special
        return "cargo"
    elif code in (60, 61, 62, 63, 64, 65, 66, 67, 68, 69):  # Passenger
        return "ferry"
    elif code in (70, 71, 72, 73, 74, 75, 76, 77, 78, 79):  # Cargo
        return "cargo"
    elif code in (80, 81, 82, 83, 84, 85, 86, 87, 88, 89):  # Tanker
        return "tanker"
    elif code in (90, 91, 92, 93, 94, 95, 96, 97, 98, 99):  # Other
        return "cargo"
    return "cargo"


def _map_nav_status(code) -> str:
    """Map AIS navigation status code to string."""
    try:
        code = int(code)  # handles int, float, and string representations
    except (ValueError, TypeError):
        code = 0
    status_map = {
        0: "underway_engine",
        1: "at_anchor",
        2: "not_under_command",
        3: "restricted_maneuverability",
        4: "constrained_by_draught",
        5: "moored",
        6: "aground",
        7: "engaged_in_fishing",
        8: "underway_sailing",
    }
    return status_map.get(code, "underway")


def _load_navigation_context(waterway_id: str) -> tuple:
    """Load ENC layer, VTS zone, and coast pilot rules for a waterway.

    Args:
        waterway_id: Waterway identifier.

    Returns:
        (enc_layer, vts_zone, coast_pilot_rules) tuple, each may be None.
    """
    enc_layer = None
    vts_zone = None
    coast_pilot_rules = None

    # Load ENC layer
    try:
        from .episode_builder import _load_enc_layer
        enc_layer = _load_enc_layer(waterway_id)
    except Exception:
        logger.debug("Failed to load ENC layer for %s", waterway_id, exc_info=True)

    # Load VTS zone and coast pilot rules
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

        cp_waterway = waterway_id
        if waterway_id in ("puget_sound_tss", "puget_sound_crossing"):
            cp_waterway = "puget_sound"
        elif waterway_id == "kill_van_kull":
            cp_waterway = "new_york_harbor"
        coast_pilot_rules = build_coast_pilot_rules(cp_waterway)
    except Exception:
        logger.debug("Failed to load VTS/coast pilot rules for %s", waterway_id, exc_info=True)

    return enc_layer, vts_zone, coast_pilot_rules


def build_episode_from_ais_dataset(
    df: pd.DataFrame,
    ownship_mmsi: str,
    waterway_id: str = "",
    water_depth: float = 50.0,
    max_encounters: int = 20,
    encounter_window_buffer_s: float = 60.0,
    cpa_threshold_m: float = 2000.0,
    min_duration_s: float = 120.0,
) -> list[EpisodeConfig]:
    """Build EpisodeConfig objects from real AIS encounter data.

    Args:
        df: Preprocessed AIS DataFrame with local (x, y) coordinates.
        ownship_mmsi: MMSI of ownship vessel.
        waterway_id: Waterway identifier for environment setup.
        water_depth: Default water depth [m] for environment.
        max_encounters: Maximum number of encounter episodes to return.
        encounter_window_buffer_s: Extra time [s] to include before/after encounter.
        cpa_threshold_m: Maximum CPA distance [m] to consider an encounter.
        min_duration_s: Minimum encounter duration [s].

    Returns:
        List of EpisodeConfig objects, one per encounter.
    """
    encounters = extract_encounters(
        df,
        ownship_mmsi,
        cpa_threshold_m=cpa_threshold_m,
        min_duration_s=min_duration_s,
    )

    # Deduplicate: keep only the longest segment per target MMSI
    best_per_target = {}
    for enc in encounters:
        tgt = enc["target_mmsi"]
        dur = enc["encounter_end"] - enc["encounter_start"]
        if tgt not in best_per_target or dur > best_per_target[tgt][1]:
            best_per_target[tgt] = (enc, dur)
    unique_encounters = [e for e, _ in sorted(
        best_per_target.values(), key=lambda x: x[0]["cpa_m"]
    )]

    # Load navigation context for this waterway
    enc_layer, vts_zone, coast_pilot_rules = _load_navigation_context(waterway_id)

    episodes = []
    for i, enc in enumerate(unique_encounters[:max_encounters]):
        own_track = enc["ownship_track"]
        tgt_track = enc["target_track"]

        # Build ownship state from track
        own_target = build_target_from_track(own_track, ownship_mmsi)

        # Build target ship
        target = build_target_from_track(tgt_track, enc["target_mmsi"])

        # Compute duration from the encounter window
        t_start = enc["encounter_start"] - encounter_window_buffer_s
        t_end = enc["encounter_end"] + encounter_window_buffer_s
        duration = max(t_end - t_start, min_duration_s)

        # Build environment
        env = EnvironmentState(
            water_depth=water_depth,
            current_x=0.0,
            current_y=0.0,
            wind_x=0.0,
            wind_y=0.0,
            waterway_id=waterway_id,
        )
        n_steps = int(duration / 0.5) + 1
        env_seq = [env] * n_steps

        episode = EpisodeConfig(
            scenario_id=f"AIS_{ownship_mmsi}_{enc['target_mmsi']}_{i:03d}",
            waterway=waterway_id,
            duration=duration,
            dt=0.5,
            ownship_initial_state=own_target.state,
            ownship_params=VesselParams(
                length=own_target.length,
                beam=own_target.beam,
                draught=own_target.draught,
            ),
            targets=[target],
            environment_sequence=env_seq,
            random_seed=i,
            perturbation_config={
                "source": "ais",
                "cpa_m": enc["cpa_m"],
                "t_cpa": enc["t_cpa"],
                "ownship_mmsi": enc["ownship_mmsi"],
                "target_mmsi": enc["target_mmsi"],
            },
            enc_layer=enc_layer,
            vts_zone=vts_zone,
            coast_pilot_rules=coast_pilot_rules,
        )
        episodes.append(episode)

    return episodes


# ---------------------------------------------------------------------------
# Multi-target encounter extraction (GAP #10 — 2026-06-10)
# ---------------------------------------------------------------------------

def extract_multi_target_encounters(
    df: pd.DataFrame,
    ownship_mmsi: str,
    cpa_threshold_m: float = 2000.0,
    min_duration_s: float = 120.0,
    min_target_points: int = 5,
    min_overlap_s: float = 60.0,
    max_targets: int = 5,
) -> list[dict]:
    """Extract episodes where multiple target ships simultaneously encounter ownship.

    Pipeline:
      1. Find all pairwise CPA encounters via ``extract_encounters``.
      2. Cluster pairwise encounters by overlapping time windows.
      3. Build multi-target encounter dicts for clusters with ≥2 targets.

    Args:
        df: Preprocessed AIS DataFrame.
        ownship_mmsi: MMSI of ownship vessel.
        cpa_threshold_m: Maximum CPA distance [m] to qualify as an encounter.
        min_duration_s: Minimum encounter duration [s].
        min_target_points: Minimum AIS records for a target in the encounter.
        min_overlap_s: Minimum overlapping time [s] for encounters to be
            considered simultaneous.
        max_targets: Maximum number of targets per multi-target episode.

    Returns:
        List of multi-target encounter dicts, each with:
            - ownship_mmsi, target_mmsis (list), cpa_m (minimum across targets)
            - encounter_start, encounter_end (union of overlapping windows)
            - ownship_track, target_tracks (list of DataFrames)
            - num_targets
    """
    # 1. Get all pairwise encounters
    pairwise = extract_encounters(
        df, ownship_mmsi,
        cpa_threshold_m=cpa_threshold_m,
        min_duration_s=min_duration_s,
        min_target_points=min_target_points,
    )
    if not pairwise:
        return []

    # 2. Build time intervals for each encounter
    intervals = []
    for enc in pairwise:
        intervals.append({
            "target_mmsi": enc["target_mmsi"],
            "start": enc["encounter_start"],
            "end": enc["encounter_end"],
            "cpa_m": enc["cpa_m"],
            "encounter": enc,
        })

    # 3. Cluster by time overlap using union-find / greedy grouping
    # Sort by start time
    intervals.sort(key=lambda x: x["start"])

    # Greedy overlap clustering: two intervals overlap if their intersection
    # duration ≥ min_overlap_s
    def _overlap_duration(a, b) -> float:
        """Compute overlap duration between two intervals [s]."""
        overlap_start = max(a["start"], b["start"])
        overlap_end = min(a["end"], b["end"])
        return max(0.0, overlap_end - overlap_start)

    clusters = []  # list of lists of interval dicts
    for item in intervals:
        placed = False
        for cluster in clusters:
            # Check if this item overlaps with ALL members of the cluster
            # (conservative: all members must pairwise overlap ≥ min_overlap_s)
            all_overlap = all(
                _overlap_duration(item, member) >= min_overlap_s
                for member in cluster
            )
            if all_overlap and len(cluster) < max_targets:
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])

    # 4. Build multi-target encounter dicts
    multi_encounters = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue  # skip single-target clusters (use pairwise extraction for those)

        # Union time window
        enc_start = min(c["start"] for c in cluster)
        enc_end = max(c["end"] for c in cluster)
        min_cpa = min(c["cpa_m"] for c in cluster)

        # Build target tracks
        target_mmsis = []
        target_tracks = []
        for c in cluster:
            target_mmsis.append(c["target_mmsi"])
            target_tracks.append(c["encounter"]["target_track"])

        # Extract ownship track for the union window
        own_track = df[
            (df["mmsi"] == ownship_mmsi)
            & (df["timestamp"] >= enc_start)
            & (df["timestamp"] <= enc_end)
        ].sort_values("timestamp")

        multi_encounters.append({
            "ownship_mmsi": str(ownship_mmsi),
            "target_mmsis": [str(m) for m in target_mmsis],
            "cpa_m": min_cpa,
            "encounter_start": enc_start,
            "encounter_end": enc_end,
            "ownship_track": own_track,
            "target_tracks": target_tracks,
            "num_targets": len(cluster),
        })

    # Sort by number of targets descending, then by CPA
    multi_encounters.sort(key=lambda e: (-e["num_targets"], e["cpa_m"]))
    return multi_encounters


def build_multi_target_episode_from_encounter(
    encounter: dict,
    waterway_id: str = "",
    water_depth: float = 50.0,
    encounter_window_buffer_s: float = 60.0,
    min_duration_s: float = 120.0,
    episode_index: int = 0,
) -> EpisodeConfig:
    """Build an EpisodeConfig from a multi-target encounter dict.

    Args:
        encounter: Multi-target encounter dict from
            ``extract_multi_target_encounters``.
        waterway_id: Waterway identifier.
        water_depth: Default water depth [m].
        encounter_window_buffer_s: Extra time [s] before/after encounter.
        min_duration_s: Minimum episode duration [s].
        episode_index: Index for scenario ID generation.

    Returns:
        EpisodeConfig with multiple TargetShip objects.
    """
    own_track = encounter["ownship_track"]
    ownship_mmsi = encounter["ownship_mmsi"]

    # Build ownship state from track
    own_target = build_target_from_track(own_track, ownship_mmsi)

    # Build target ships
    targets = []
    for tgt_track in encounter["target_tracks"]:
        if len(tgt_track) < 2:
            continue
        tgt_mmsi = str(tgt_track["mmsi"].iloc[0])
        tgt = build_target_from_track(tgt_track, tgt_mmsi)
        targets.append(tgt)

    if not targets:
        raise ValueError("No valid target ships in multi-target encounter")

    # Compute duration
    t_start = encounter["encounter_start"] - encounter_window_buffer_s
    t_end = encounter["encounter_end"] + encounter_window_buffer_s
    duration = max(t_end - t_start, min_duration_s)

    # Build environment
    env = EnvironmentState(
        water_depth=water_depth,
        current_x=0.0,
        current_y=0.0,
        wind_x=0.0,
        wind_y=0.0,
        waterway_id=waterway_id,
    )
    n_steps = int(duration / 0.5) + 1

    # Load navigation context
    enc_layer, vts_zone, coast_pilot_rules = _load_navigation_context(waterway_id)

    target_mmsis_str = "_".join(encounter["target_mmsis"][:3])
    episode = EpisodeConfig(
        scenario_id=f"AIS_MULTI_{ownship_mmsi}_{target_mmsis_str}_{episode_index:03d}",
        waterway=waterway_id,
        duration=duration,
        dt=0.5,
        ownship_initial_state=own_target.state,
        ownship_params=VesselParams(
            length=own_target.length,
            beam=own_target.beam,
            draught=own_target.draught,
        ),
        targets=targets,
        environment_sequence=[env] * n_steps,
        random_seed=episode_index,
        perturbation_config={
            "source": "ais_multi_target",
            "cpa_m": encounter["cpa_m"],
            "ownship_mmsi": encounter["ownship_mmsi"],
            "target_mmsis": encounter["target_mmsis"],
            "num_targets": encounter["num_targets"],
        },
        enc_layer=enc_layer,
        vts_zone=vts_zone,
        coast_pilot_rules=coast_pilot_rules,
    )
    return episode


def build_multi_target_episodes_from_ais(
    df: pd.DataFrame,
    ownship_mmsi: str,
    waterway_id: str = "",
    water_depth: float = 50.0,
    max_episodes: int = 30,
    cpa_threshold_m: float = 2000.0,
    min_duration_s: float = 120.0,
    min_overlap_s: float = 60.0,
    max_targets: int = 5,
    encounter_window_buffer_s: float = 60.0,
) -> list[EpisodeConfig]:
    """Build multi-target EpisodeConfig objects from real AIS data.

    Extracts simultaneous multi-ship encounters and builds EpisodeConfig
    objects with multiple TargetShips.

    Args:
        df: Preprocessed AIS DataFrame.
        ownship_mmsi: MMSI of ownship vessel.
        waterway_id: Waterway identifier.
        water_depth: Default water depth [m].
        max_episodes: Maximum number of multi-target episodes to return.
        cpa_threshold_m: Maximum CPA distance [m].
        min_duration_s: Minimum encounter duration [s].
        min_overlap_s: Minimum overlap [s] for simultaneous encounters.
        max_targets: Maximum targets per episode.
        encounter_window_buffer_s: Extra time [s] before/after window.

    Returns:
        List of EpisodeConfig objects with multiple TargetShips.
    """
    multi_encounters = extract_multi_target_encounters(
        df, ownship_mmsi,
        cpa_threshold_m=cpa_threshold_m,
        min_duration_s=min_duration_s,
        min_overlap_s=min_overlap_s,
        max_targets=max_targets,
    )

    episodes = []
    for i, enc in enumerate(multi_encounters[:max_episodes]):
        try:
            ep = build_multi_target_episode_from_encounter(
                enc,
                waterway_id=waterway_id,
                water_depth=water_depth,
                encounter_window_buffer_s=encounter_window_buffer_s,
                min_duration_s=min_duration_s,
                episode_index=i,
            )
            episodes.append(ep)
        except Exception:
            logger.debug(
                "Failed to build multi-target episode %d: %s",
                i, enc.get("target_mmsis", []), exc_info=True,
            )

    return episodes


# ---------------------------------------------------------------------------
# Memory-efficient chunked pipeline for large CSV files
# ---------------------------------------------------------------------------

def _most_frequent_mmsi(csv_path: str) -> str:
    """Find the most frequent MMSI by scanning only the mmsi column."""
    mmsi_counts = {}
    for chunk in pd.read_csv(csv_path, usecols=["mmsi"], chunksize=2_000_000):
        for mmsi, cnt in chunk["mmsi"].value_counts().items():
            mmsi_counts[mmsi] = mmsi_counts.get(mmsi, 0) + cnt
    if not mmsi_counts:
        raise ValueError("No MMSI data found")
    return str(max(mmsi_counts, key=mmsi_counts.get))


def _load_vessel_data(csv_path: str, mmsi: str) -> pd.DataFrame:
    """Load all records for a single MMSI using chunked filtering."""
    # Unify MMSI type with whatever the CSV column uses (int or str) so
    # the comparison works regardless of CSV dtype (BUG #8 fix).
    frames = []
    for chunk in pd.read_csv(csv_path, chunksize=2_000_000):
        # Align column type to match the lookup key
        col_dtype = chunk["mmsi"].dtype
        if col_dtype == object or "str" in str(col_dtype):
            lookup = str(mmsi)
        else:
            lookup = int(mmsi)
        subset = chunk[chunk["mmsi"] == lookup]
        if len(subset) > 0:
            frames.append(subset)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _scan_candidate_targets(
    csv_path: str,
    ownship_df: pd.DataFrame,
    ownship_mmsi: str,
    proximity_m: float = 4000.0,
) -> set:
    """Single-pass scan to find all MMSIs that come within *proximity_m* of ownship.

    Uses a KD-tree built from downsampled ownship positions for efficient
    spatial queries, avoiding the O(M) merge_asof calls in the original.
    """
    from scipy.spatial import cKDTree

    ownship_int = int(ownship_mmsi)
    own_min_t = ownship_df["timestamp"].min()
    own_max_t = ownship_df["timestamp"].max()

    # Downsample ownship to ~1 point per 30 s for KD-tree (balances speed vs accuracy)
    own = ownship_df[["timestamp", "x", "y"]].sort_values("timestamp")
    step = max(1, len(own) // 1200)
    own_sample = own.iloc[::step]
    own_positions = own_sample[["x", "y"]].values.astype(np.float64)
    tree = cKDTree(own_positions)

    candidates: set = set()
    chunksize = 1_000_000
    proximity_sq = proximity_m**2

    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        chunk = chunk[
            (chunk["timestamp"] >= own_min_t)
            & (chunk["timestamp"] <= own_max_t)
            & (chunk["mmsi"] != ownship_int)
        ]
        if len(chunk) == 0:
            continue

        # Vectorized KD-tree query for all points in chunk
        tgt_pos = chunk[["x", "y"]].values.astype(np.float64)
        dists, _ = tree.query(tgt_pos, k=1, distance_upper_bound=proximity_m)
        close_mask = dists < proximity_m
        if not np.any(close_mask):
            continue

        close_mmsis = chunk["mmsi"].values[close_mask]
        candidates.update(int(m) for m in np.unique(close_mmsis))

        if chunk_idx % 20 == 0:
            print(f"    Scanned chunk {chunk_idx}, {len(candidates)} candidates found ...")

    return candidates


def extract_encounters_chunked(
    csv_path: str,
    ownship_mmsi: str = "",
    cpa_threshold_m: float = 2000.0,
    min_duration_s: float = 120.0,
    min_target_points: int = 5,
    n_jobs: int = 1,
) -> list[dict]:
    """Memory-efficient encounter extraction for CSV files that exceed RAM.

    Pipeline:
      1. Scan mmsi column to find the most frequent vessel (if ownship not given).
      2. Extract ownship track via chunked filtering (fits in memory).
      3. Single-pass scan to identify candidate target MMSIs near ownship.
      4. Extract candidate target tracks via chunked filtering.
      5. Run the standard ``extract_encounters`` on the in-memory subset.

    Args:
        csv_path: Path to preprocessed AIS CSV.
        ownship_mmsi: Ownship MMSI (empty = auto-detect most frequent).
        cpa_threshold_m: Maximum CPA distance [m] to qualify as an encounter.
        min_duration_s: Minimum encounter duration [s].
        min_target_points: Minimum AIS records for a target in the encounter
            window.
        n_jobs: Unused; reserved for future parallelisation.

    Returns:
        List of encounter dicts (same format as ``extract_encounters``).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"AIS CSV not found: {csv_path}")

    # 1. Identify ownship
    if not ownship_mmsi:
        print("  Finding most frequent MMSI ...")
        ownship_mmsi = _most_frequent_mmsi(csv_path)
        print(f"  Ownship MMSI = {ownship_mmsi}")

    # 2. Load ownship data
    print(f"  Loading ownship track for MMSI {ownship_mmsi} ...")
    own_df = _load_vessel_data(csv_path, ownship_mmsi)
    if len(own_df) < 2:
        print(f"  Ownship has too few records ({len(own_df)}); no encounters possible.")
        return []
    own_df = own_df.sort_values("timestamp")
    print(f"  Ownship: {len(own_df):,} records, "
          f"{own_df['timestamp'].min():.0f} -- {own_df['timestamp'].max():.0f} s")

    # 3. Scan for candidate targets
    scan_radius = max(cpa_threshold_m * 2, 4000.0)
    print(f"  Scanning for candidate targets within {scan_radius:.0f} m ...")
    candidates = _scan_candidate_targets(
        csv_path, own_df, ownship_mmsi, proximity_m=scan_radius,
    )
    print(f"  Found {len(candidates)} candidate targets")

    if not candidates:
        return []

    # 4. Load candidate target tracks
    print("  Loading candidate target tracks ...")
    target_frames: dict[int, list[pd.DataFrame]] = {}
    chunksize = 2_000_000
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        mask = chunk["mmsi"].isin(candidates)
        subset = chunk[mask]
        if len(subset) == 0:
            continue
        for mmsi, grp in subset.groupby("mmsi"):
            target_frames.setdefault(mmsi, []).append(grp)

    target_dfs: dict[int, pd.DataFrame] = {}
    for mmsi, frames in target_frames.items():
        df = pd.concat(frames, ignore_index=True)
        if len(df) >= min_target_points:
            target_dfs[mmsi] = df.sort_values("timestamp")

    print(f"  Kept {len(target_dfs)} targets with sufficient data")

    # Combine ownship + targets into one DataFrame and run standard extraction
    combined = pd.concat([own_df] + list(target_dfs.values()), ignore_index=True)
    return extract_encounters(
        combined,
        ownship_mmsi=ownship_mmsi,
        cpa_threshold_m=cpa_threshold_m,
        min_duration_s=min_duration_s,
        min_target_points=min_target_points,
    )


def build_episodes_from_csv(
    csv_path: str,
    ownship_mmsi: str = "",
    waterway_id: str = "",
    water_depth: float = 50.0,
    max_encounters: int = 20,
    cpa_threshold_m: float = 2000.0,
    min_duration_s: float = 120.0,
    encounter_window_buffer_s: float = 60.0,
    candidate_batch_size: int = 200,
) -> list[EpisodeConfig]:
    """Build EpisodeConfig objects directly from a preprocessed AIS CSV file.

    End-to-end chunked pipeline that never loads the full file into memory.
    Candidates are processed in batches so peak memory stays bounded by
    ownship + (batch_size * avg_target_records).

    Args:
        csv_path: Path to preprocessed AIS CSV (must have x, y columns).
        ownship_mmsi: Ownship MMSI (empty = auto-detect most frequent).
        waterway_id: Waterway identifier for environment metadata.
        water_depth: Default water depth [m].
        max_encounters: Maximum number of encounter episodes to return.
        cpa_threshold_m: Maximum CPA distance [m] to consider an encounter.
        min_duration_s: Minimum encounter duration [s].
        encounter_window_buffer_s: Extra time [s] before/after encounter window.
        candidate_batch_size: Number of candidate MMSIs to process per file pass.

    Returns:
        List of EpisodeConfig objects, one per encounter.
    """
    # 1. Identify ownship
    if not ownship_mmsi:
        print("  Finding most frequent MMSI ...")
        ownship_mmsi = _most_frequent_mmsi(csv_path)
        print(f"  Ownship MMSI = {ownship_mmsi}")

    # 2. Load ownship data
    print(f"  Loading ownship track for MMSI {ownship_mmsi} ...")
    own_df = _load_vessel_data(csv_path, ownship_mmsi)
    if len(own_df) < 2:
        print(f"  Ownship has too few records ({len(own_df)}); no episodes possible.")
        return []
    own_df = own_df.sort_values("timestamp")
    print(f"  Ownship: {len(own_df):,} records, "
          f"{own_df['timestamp'].min():.0f} -- {own_df['timestamp'].max():.0f} s")

    # 3. Scan for candidate targets
    scan_radius = max(cpa_threshold_m * 2, 4000.0)
    print(f"  Scanning for candidate targets within {scan_radius:.0f} m ...")
    candidate_set = _scan_candidate_targets(
        csv_path, own_df, ownship_mmsi, proximity_m=scan_radius,
    )
    print(f"  Found {len(candidate_set)} candidate targets")

    if not candidate_set:
        return []

    # 4. Process candidates in batches to keep memory bounded.
    #    Each batch: single file pass to collect data → per-target encounter
    #    detection (one target at a time) → free memory.
    candidate_list = sorted(candidate_set)
    batch_size = 500
    num_batches = (len(candidate_list) + batch_size - 1) // batch_size
    chunksize = 500_000
    all_encounters: list[dict] = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_mmsis = set(candidate_list[batch_start:batch_start + batch_size])

        # 4a. Single pass: collect data for this batch of MMSIs
        target_frames: dict[int, list[pd.DataFrame]] = {}
        chunk_idx = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            chunk_idx += 1
            mask = chunk["mmsi"].isin(batch_mmsis)
            subset = chunk[mask]
            if len(subset) == 0:
                continue
            for mmsi, grp in subset.groupby("mmsi"):
                target_frames.setdefault(mmsi, []).append(grp)

        # 4b. Combine, sort, and process each target one-at-a-time
        batch_encounters = 0
        for mmsi, frames in target_frames.items():
            tgt_df = pd.concat(frames, ignore_index=True)
            if len(tgt_df) < 5:
                continue
            tgt_df = tgt_df.sort_values("timestamp")
            combined = pd.concat([own_df, tgt_df], ignore_index=True)
            encs = extract_encounters(
                combined,
                ownship_mmsi=ownship_mmsi,
                cpa_threshold_m=cpa_threshold_m,
                min_duration_s=min_duration_s,
            )
            all_encounters.extend(encs)
            batch_encounters += len(encs)

        print(f"  Batch {batch_idx + 1}/{num_batches}: "
              f"{len(target_frames)} targets, {batch_encounters} encounters")

        if len(all_encounters) >= max_encounters * 10:
            break

    print(f"  Total encounters found: {len(all_encounters)}")

    if not all_encounters:
        return []

    # 5. Deduplicate: keep only the longest segment per target MMSI
    best_per_target = {}
    for enc in all_encounters:
        tgt = enc["target_mmsi"]
        dur = enc["encounter_end"] - enc["encounter_start"]
        if tgt not in best_per_target or dur > best_per_target[tgt][1]:
            best_per_target[tgt] = (enc, dur)
    unique_encounters = [e for e, _ in sorted(
        best_per_target.values(), key=lambda x: x[0]["cpa_m"]
    )]

    # 6. Load navigation context for this waterway
    enc_layer, vts_zone, coast_pilot_rules = _load_navigation_context(waterway_id)

    # 7. Build EpisodeConfig objects
    episodes = []
    for i, enc in enumerate(unique_encounters[:max_encounters]):
        own_track = enc["ownship_track"]
        tgt_track = enc["target_track"]

        own_target = build_target_from_track(own_track, ownship_mmsi)
        target = build_target_from_track(tgt_track, enc["target_mmsi"])

        t_start = enc["encounter_start"] - encounter_window_buffer_s
        t_end = enc["encounter_end"] + encounter_window_buffer_s
        duration = max(t_end - t_start, min_duration_s)

        env = EnvironmentState(
            water_depth=water_depth,
            current_x=0.0, current_y=0.0,
            wind_x=0.0, wind_y=0.0,
            waterway_id=waterway_id,
        )
        n_steps = int(duration / 0.5) + 1

        episode = EpisodeConfig(
            scenario_id=f"AIS_{ownship_mmsi}_{enc['target_mmsi']}_{i:03d}",
            waterway=waterway_id,
            duration=duration,
            dt=0.5,
            ownship_initial_state=own_target.state,
            ownship_params=VesselParams(
                length=own_target.length,
                beam=own_target.beam,
                draught=own_target.draught,
            ),
            targets=[target],
            environment_sequence=[env] * n_steps,
            random_seed=i,
            perturbation_config={
                "source": "ais",
                "cpa_m": enc["cpa_m"],
                "t_cpa": enc["t_cpa"],
                "ownship_mmsi": enc["ownship_mmsi"],
                "target_mmsi": enc["target_mmsi"],
            },
            enc_layer=enc_layer,
            vts_zone=vts_zone,
            coast_pilot_rules=coast_pilot_rules,
        )
        episodes.append(episode)

    return episodes
