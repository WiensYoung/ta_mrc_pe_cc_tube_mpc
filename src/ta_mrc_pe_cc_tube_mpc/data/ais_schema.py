"""AIS data schema definitions and standard column names."""

import numpy as np

# Standard AIS message fields for preprocessing
AIS_COLUMNS = [
    "mmsi",
    "timestamp",
    "lon",
    "lat",
    "sog",          # speed over ground [kn]
    "cog",          # course over ground [deg]
    "heading",      # true heading [deg], may be None
    "nav_status",   # 0=underway, 1=anchored, etc.
    "vessel_type",  # AIS ship type code
    "length",       # [m]
    "beam",         # [m]
    "draught",      # [m]
    "rot",          # rate of turn [deg/min]
    "position_accuracy",  # 0=low (>10m), 1=high (<10m)
]

# AIS navigation status codes
NAV_STATUS_MAP = {
    0: "underway_using_engine",
    1: "at_anchor",
    2: "not_under_command",
    3: "restricted_maneuverability",
    4: "constrained_by_draught",
    5: "moored",
    6: "aground",
    7: "engaged_in_fishing",
    8: "underway_sailing",
}

# Columns required for episode building
REQUIRED_COLUMNS = ["mmsi", "timestamp", "lon", "lat", "sog", "cog"]

# Value range checks for AIS fields
VALUE_RANGES = {
    "lon": (-180.0, 180.0),
    "lat": (-90.0, 90.0),
    "sog": (0.0, 60.0),       # 60 kn max plausible
    "cog": (0.0, 360.0),
    "heading": (0.0, 360.0),
    "length": (1.0, 500.0),   # 1-500 m plausible
    "beam": (1.0, 100.0),     # 1-100 m plausible
    "draught": (0.1, 30.0),   # 0.1-30 m plausible
    "position_accuracy": (0, 1),
}


# ── GAP #7 (2026-06-10): AIS quality defaults stratified by vessel type ──
# Real AIS data quality varies significantly by vessel class:
#   Class A (>300 GT, SOLAS): 2-10 s reporting interval, high accuracy
#   Class B (small craft): 30 s - 3 min interval, lower accuracy
#   Ferry: often has high update rate but high dynamics (fast turns)
#   Tanker: Class A, but often in low-visibility conditions
# Reference: ITU-R M.1371-5, IALA Guideline 1082.

AIS_QUALITY_DEFAULTS = {
    "cargo": {
        "position_noise_std": 5.0,    # [m] — Class A, good accuracy
        "delay": 2.0,                  # [s] — typical 2-10 s reporting
        "dropout_prob": 0.02,          # — low dropout in coastal areas
        "ais_class": "A",
    },
    "tanker": {
        "position_noise_std": 5.0,    # [m] — Class A
        "delay": 3.0,                  # [s] — slightly slower reporting
        "dropout_prob": 0.03,          # — moderate dropout
        "ais_class": "A",
    },
    "ferry": {
        "position_noise_std": 3.0,    # [m] — often high-precision GPS
        "delay": 1.0,                  # [s] — fast update for safety
        "dropout_prob": 0.01,          # — low dropout
        "ais_class": "A",
    },
    "tug_barge": {
        "position_noise_std": 8.0,    # [m] — lower accuracy
        "delay": 10.0,                 # [s] — often Class B or tow
        "dropout_prob": 0.08,          # — higher dropout in confined waters
        "ais_class": "B",
    },
    "small_craft": {
        "position_noise_std": 15.0,   # [m] — Class B, consumer GPS
        "delay": 30.0,                 # [s] — 30 s typical, up to 3 min
        "dropout_prob": 0.15,          # — high dropout near structures
        "ais_class": "B",
    },
    "fishing": {
        "position_noise_std": 10.0,
        "delay": 15.0,
        "dropout_prob": 0.10,
        "ais_class": "B",
    },
    "sailing": {
        "position_noise_std": 12.0,
        "delay": 20.0,
        "dropout_prob": 0.12,
        "ais_class": "B",
    },
}


def get_ais_quality_for_vessel(
    vessel_type: str,
    waterway_type: str = "coastal",
    override: dict = None,
) -> dict:
    """Return AIS quality parameters for a vessel type.

    Stratified by vessel class (A vs B transponder) and waterway context.
    Coastal/open waters have better AIS coverage than confined harbours.

    Args:
        vessel_type: One of the keys in AIS_QUALITY_DEFAULTS.
        waterway_type: "coastal", "harbour", "restricted_channel", "open_sea".
        override: Optional per-vessel override dict (merged on top).

    Returns:
        dict with keys: position_noise_std, delay, dropout_prob, ais_class.
    """
    defaults = AIS_QUALITY_DEFAULTS.get(
        vessel_type.lower(),
        AIS_QUALITY_DEFAULTS["cargo"],  # fallback to cargo defaults
    )
    result = dict(defaults)

    # Waterway adjustment: confined waters → worse AIS coverage
    if waterway_type in ("harbour", "restricted_channel", "narrow_channel"):
        result["delay"] = result["delay"] * 1.5
        result["dropout_prob"] = min(result["dropout_prob"] * 1.5, 0.3)
    elif waterway_type == "open_sea":
        result["delay"] = result["delay"] * 0.8
        result["dropout_prob"] = result["dropout_prob"] * 0.7

    # Per-vessel override takes final precedence
    if override:
        result.update(override)

    return result


def validate_ais_record(record: dict) -> list[str]:
    """Check required fields exist and have valid values in an AIS record.

    Returns a list of validation error descriptions.
    """
    errors = []
    for col in REQUIRED_COLUMNS:
        val = record.get(col)
        if val is None:
            errors.append(f"{col}: missing")
        elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            errors.append(f"{col}: NaN/Inf")
    for col, (vmin, vmax) in VALUE_RANGES.items():
        val = record.get(col)
        if val is not None and isinstance(val, (int, float)) and not (np.isnan(val) or np.isinf(val)):
            if val < vmin or val > vmax:
                errors.append(f"{col}: {val} out of range [{vmin}, {vmax}]")
    return errors
