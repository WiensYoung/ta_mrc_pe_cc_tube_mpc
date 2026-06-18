"""AIS data preprocessing: interpolation, outlier removal, coordinate conversion."""

from typing import Optional

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Column mapping from NOAA AIS bulk CSV to standard AIS schema
NOAA_COLUMN_MAP = {
    "mmsi": "mmsi",
    "timestamp_unix_s": "timestamp",
    "timestamp_utc": "timestamp_utc",
    "lon": "lon",
    "lat": "lat",
    "sog": "sog",
    "cog": "cog",
    "heading": "heading",
    "vessel_type": "vessel_type",
    "nav_status": "nav_status",
    "length_m": "length",
    "width_m": "beam",
    "draught_m": "draught",
}

# Columns to keep after mapping (standard schema + extras)
KEEP_COLUMNS = [
    "mmsi", "timestamp", "timestamp_utc", "lon", "lat", "sog", "cog",
    "heading", "vessel_type", "nav_status", "length", "beam", "draught",
    "rot", "position_accuracy",
]


# Required fields for real AIS data validation.
# Missing any of these triggers a clear error (not silent synthetic fallback).
REQUIRED_AIS_FIELDS = frozenset({
    "mmsi", "timestamp", "lon", "lat", "sog", "cog",
})

REQUIRED_ENC_FIELDS = frozenset({
    "depth", "shore_boundary", "channel_boundary",
})


def validate_ais_dataframe(df: pd.DataFrame, source: str = "unknown") -> list[str]:
    """Validate that a DataFrame has all required AIS fields.

    Args:
        df: Input DataFrame (can have any column names at this point).
        source: Descriptive label for error messages.

    Returns:
        List of missing field names (empty = valid).
    """
    missing = []
    for field in REQUIRED_AIS_FIELDS:
        if field not in df.columns:
            # Check common aliases
            aliases = {
                "mmsi": ["MMSI", "vessel_id"],
                "timestamp": ["time", "ts", "epoch", "BaseDateTime"],
                "lon": ["LON", "longitude", "long"],
                "lat": ["LAT", "latitude"],
                "sog": ["SOG", "speed"],
                "cog": ["COG", "course"],
            }
            found = any(alias in df.columns for alias in aliases.get(field, []))
            if not found:
                missing.append(field)
    if missing:
        logger.error(
            "AIS data from '%s' is missing required fields: %s. "
            "Available columns: %s. "
            "Real AIS/ENC data must be obtained from MarineCadastre.gov "
            "or NOAA OCS and placed in the configured data directory. "
            "See docs/reproducibility.md for instructions.",
            source, missing, list(df.columns)[:20],
        )
    return missing


def normalize_noaa_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename NOAA AIS columns to standard schema and add missing defaults.

    Args:
        df: DataFrame with NOAA bulk AIS column names.

    Returns:
        DataFrame with standard AIS column names.
    """
    df = df.rename(columns=NOAA_COLUMN_MAP)
    # Add missing columns with default values
    for col in ["rot", "position_accuracy"]:
        if col not in df.columns:
            df[col] = None
    # Keep only standard columns that exist
    keep = [c for c in KEEP_COLUMNS if c in df.columns]
    return df[keep].copy()


def preprocess_ais_chunked(
    input_path: str,
    output_path: str,
    lon_ref: float = 0.0,
    lat_ref: float = 0.0,
    chunksize: int = 1_000_000,
    max_sog: float = 50.0,
    max_accel: float = 0.5,  # kn/s — typical merchant vessel accel is 0.1–0.3 kn/s
) -> str:
    """Preprocess a large AIS CSV in chunks.

    Reads and processes the file in chunks to handle 100GB+ files.
    Results are accumulated and written to a single output CSV.

    Args:
        input_path: Path to raw AIS CSV.
        output_path: Path for processed output CSV.
        lon_ref: Reference longitude for local coordinate conversion.
        lat_ref: Reference latitude for local coordinate conversion.
        chunksize: Rows per chunk.
        max_sog: Maximum plausible speed [kn].
        max_accel: Maximum plausible acceleration [kn/s].

    Returns:
        Path to output CSV.
    """
    import os

    first_chunk = True
    total_in = 0
    total_out = 0
    validated = False

    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        total_in += len(chunk)
        chunk = normalize_noaa_columns(chunk)

        # Validate required AIS fields on the first chunk only
        if not validated:
            missing = validate_ais_dataframe(chunk, source=input_path)
            if missing:
                raise ValueError(
                    f"AIS data validation failed for '{input_path}'. "
                    f"Missing fields: {missing}. "
                    f"Real AIS data must contain at least: {sorted(REQUIRED_AIS_FIELDS)}. "
                    f"Obtain data from MarineCadastre.gov (AIS) or NOAA OCS (ENC)."
                )
            validated = True

        # Convert timestamp: prefer timestamp_utc (ISO format) over raw unix
        if "timestamp_utc" in chunk.columns:
            chunk["timestamp"] = pd.to_datetime(
                chunk["timestamp_utc"], errors="coerce"
            ).astype("int64") // 10**9
        elif "timestamp" not in chunk.columns:
            chunk["timestamp"] = range(len(chunk))

        # Ensure numeric types
        for c in ["sog", "cog", "heading", "lon", "lat", "length", "beam", "draught"]:
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        # Remove outlier records
        try:
            chunk = remove_outliers_speed(chunk, max_sog=max_sog, max_accel=max_accel)
        except Exception:
            logger.warning("Outlier removal failed for chunk — using unfiltered data", exc_info=True)

        # Interpolate tracks per MMSI
        n_interp_failures = 0
        try:
            frames = []
            for mmsi, group in chunk.groupby("mmsi"):
                if len(group) < 2:
                    frames.append(group)
                    continue
                try:
                    group = interpolate_track(group)
                except Exception:
                    logger.debug("Track interpolation failed for MMSI %s", mmsi, exc_info=True)
                    n_interp_failures += 1
                frames.append(group)
            if frames:
                chunk = pd.concat(frames, ignore_index=True)
        except Exception:
            logger.warning("Interpolation failed for chunk — using raw data", exc_info=True)
        if n_interp_failures > 0:
            logger.info("Interpolation failed for %d MMSI(s) in chunk", n_interp_failures)

        # Convert to local coordinates
        try:
            chunk = convert_ais_to_local(chunk, lon_ref, lat_ref)
        except Exception:
            logger.warning("Coordinate conversion failed for chunk — records may be in WGS84", exc_info=True)

        total_out += len(chunk)

        # Write header only for first chunk
        mode = "w" if first_chunk else "a"
        header = first_chunk
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        chunk.to_csv(output_path, mode=mode, header=header, index=False)
        first_chunk = False

        print(f"\r  Processed {total_in:,} → {total_out:,} records...", end="", flush=True)

    print()  # newline
    print(f"Done. {total_out:,} records written to {output_path}")
    return output_path


def remove_outliers_speed(
    df: pd.DataFrame, max_sog: float = 50.0, max_accel: float = 0.5
) -> pd.DataFrame:
    """Remove AIS records with physically implausible speed or acceleration.

    Args:
        df: DataFrame with 'sog' column [kn].
        max_sog: Maximum plausible speed [kn].
        max_accel: Maximum plausible acceleration between records [kn/s].

    Returns:
        Filtered DataFrame.
    """
    mask = df["sog"] <= max_sog
    if "timestamp" in df.columns and len(df) > 1:
        dt = df["timestamp"].diff().fillna(1.0)
        dv = df["sog"].diff().abs().fillna(0.0)
        accel = dv / dt.clip(lower=0.5)
        mask = mask & (accel <= max_accel)
    n_removed = (~mask).sum()
    if n_removed > 0:
        logger.info(f"Removed {n_removed} outlier AIS records (speed/acceleration).")
    return df[mask].copy()


def interpolate_track(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    freq: str = "10s",
    max_gap: float = 300.0,
) -> pd.DataFrame:
    """Interpolate AIS track to uniform time grid.

    Gaps larger than max_gap seconds are left as NaN
    (not interpolated across missing data segments).

    Args:
        df: DataFrame with time column and position columns.
        time_col: Name of time column.
        freq: Resample frequency.
        max_gap: Maximum interpolation gap [s].

    Returns:
        Resampled and interpolated DataFrame.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
    df = df.set_index(time_col).sort_index()
    df = df[~df.index.duplicated()]

    resampled = df.resample(freq).mean()

    # Linear interpolation for numeric columns
    numeric_cols = resampled.select_dtypes(include=[np.number]).columns
    resampled[numeric_cols] = resampled[numeric_cols].interpolate(
        method="linear", limit=int(max_gap / pd.Timedelta(freq).total_seconds())
    )

    # Heading needs circular interpolation
    if "heading" in resampled.columns:
        resampled["heading"] = _circular_interpolate(
            resampled["heading"].values, max_gap_seconds=max_gap, freq_seconds=_freq_to_seconds(freq)
        )

    return resampled.reset_index()


def _circular_interpolate(
    values: np.ndarray, max_gap_seconds: float, freq_seconds: float
) -> np.ndarray:
    """Linear interpolation on the circle for heading data."""
    result = values.copy()
    nan_mask = np.isnan(values)
    if not nan_mask.any():
        return result

    sin_vals = np.sin(np.deg2rad(values))
    cos_vals = np.cos(np.deg2rad(values))

    max_steps = int(max_gap_seconds / freq_seconds)
    for _ in range(3):  # iterative fill
        sin_vals = pd.Series(sin_vals).interpolate(limit=max_steps).values
        cos_vals = pd.Series(cos_vals).interpolate(limit=max_steps).values

    result = np.rad2deg(np.arctan2(sin_vals, cos_vals)) % 360.0
    return result


def _freq_to_seconds(freq: str) -> float:
    """Convert pandas frequency string to seconds."""
    if freq.endswith("s"):
        return float(freq[:-1])
    elif freq.endswith("min") or freq.endswith("T"):
        return float(freq[:-3]) * 60.0
    return 10.0


def convert_ais_to_local(
    df: pd.DataFrame,
    lon_ref: float = 0.0,
    lat_ref: float = 0.0,
) -> pd.DataFrame:
    """Convert AIS lon/lat to local ENU coordinates (approximate flat-Earth).

    Uses simple equirectangular approximation valid for limited-area navigation.

    Args:
        df: DataFrame with 'lon', 'lat' columns.
        lon_ref, lat_ref: Reference point for origin.

    Returns:
        DataFrame with added 'x', 'y' columns in meters.
    """
    from ..constants import DEG_TO_RAD

    lat_mid = np.deg2rad(lat_ref)
    cos_lat = np.cos(lat_mid)
    df = df.copy()
    df["x"] = (df["lon"] - lon_ref) * DEG_TO_RAD * 6378137.0 * cos_lat
    df["y"] = (df["lat"] - lat_ref) * DEG_TO_RAD * 6378137.0
    return df
