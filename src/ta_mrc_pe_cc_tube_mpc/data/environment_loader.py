"""Environment data loader: gridded current/wind fields with spatial interpolation.

Supports:
  - Loading from gridded NetCDF/CSV files
  - Bilinear spatial interpolation of vector fields
  - Temporal interpolation between time slices
  - Synthetic field generation for testing
  - Building time-varying EnvironmentState sequences
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..types import EnvironmentState


class VectorField2D:
    """2D vector field (e.g. current or wind) defined on a regular grid.

    Attributes:
        lon_grid: 1D array of longitude values.
        lat_grid: 1D array of latitude values.
        u: 2D array of eastward velocity component [m/s].
        v: 2D array of northward velocity component [m/s].
        time: Optional timestamp [unix seconds] for time-varying fields.
    """

    def __init__(
        self,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        time: Optional[float] = None,
    ):
        nlon, nlat = len(lon_grid), len(lat_grid)
        if u.shape != (nlat, nlon):
            raise ValueError(
                f"u shape {u.shape} != (nlat={nlat}, nlon={nlon})"
            )
        if v.shape != (nlat, nlon):
            raise ValueError(
                f"v shape {v.shape} != (nlat={nlat}, nlon={nlon})"
            )
        self.lon_grid = np.asarray(lon_grid, dtype=np.float64)
        self.lat_grid = np.asarray(lat_grid, dtype=np.float64)
        self.u = np.asarray(u, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.time = time

    def sample(self, lon: float, lat: float) -> tuple[float, float]:
        """Bilinearly interpolate the vector field at (lon, lat).

        Returns (u_east, v_north) in [m/s]. Clamps to nearest grid edge
        if the query point is outside the grid extent.
        """
        lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))

        i_lon = np.searchsorted(self.lon_grid, lon_arr, side="right") - 1
        i_lat = np.searchsorted(self.lat_grid, lat_arr, side="right") - 1

        i_lon = np.clip(i_lon, 0, len(self.lon_grid) - 2)
        i_lat = np.clip(i_lat, 0, len(self.lat_grid) - 2)

        # Fractional positions
        fx = (lon_arr - self.lon_grid[i_lon]) / max(
            self.lon_grid[i_lon + 1] - self.lon_grid[i_lon], 1e-12,
        )
        fy = (lat_arr - self.lat_grid[i_lat]) / max(
            self.lat_grid[i_lat + 1] - self.lat_grid[i_lat], 1e-12,
        )
        fx = np.clip(fx, 0, 1)
        fy = np.clip(fy, 0, 1)

        # Bilinear interpolation on each velocity component
        u00 = self.u[i_lat, i_lon]
        u10 = self.u[i_lat, i_lon + 1]
        u01 = self.u[i_lat + 1, i_lon]
        u11 = self.u[i_lat + 1, i_lon + 1]

        v00 = self.v[i_lat, i_lon]
        v10 = self.v[i_lat, i_lon + 1]
        v01 = self.v[i_lat + 1, i_lon]
        v11 = self.v[i_lat + 1, i_lon + 1]

        u_out = (1 - fy) * ((1 - fx) * u00 + fx * u10) + fy * ((1 - fx) * u01 + fx * u11)
        v_out = (1 - fy) * ((1 - fx) * v00 + fx * v10) + fy * ((1 - fx) * v01 + fx * v11)

        return float(u_out[0]), float(v_out[0])


class TimeVaryingField:
    """Time-varying vector field composed of multiple time slices.

    Linearly interpolates between slices in time.
    """

    def __init__(self, fields: list[VectorField2D]):
        if not fields:
            raise ValueError("At least one field is required")
        self.fields = sorted(fields, key=lambda f: f.time or 0.0)

    def sample(self, lon: float, lat: float, t: float) -> tuple[float, float]:
        """Sample the field at (lon, lat, t) with temporal interpolation.

        Returns (u_east, v_north) in [m/s].
        """
        if len(self.fields) == 1 or self.fields[0].time is None:
            return self.fields[0].sample(lon, lat)

        times = np.array([f.time for f in self.fields])
        idx = np.searchsorted(times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.fields) - 2)

        f0 = self.fields[idx]
        f1 = self.fields[idx + 1]
        dt = float(times[idx + 1] - times[idx])
        alpha = (t - float(times[idx])) / dt if dt > 0 else 0.0
        alpha = np.clip(alpha, 0, 1)

        u0, v0 = f0.sample(lon, lat)
        u1, v1 = f1.sample(lon, lat)
        return u0 + alpha * (u1 - u0), v0 + alpha * (v1 - v0)


def load_current_field_from_csv(
    path: str,
    time: Optional[float] = None,
) -> VectorField2D:
    """Load a gridded current field from CSV.

    Expected CSV columns: lon, lat, u_east [m/s], v_north [m/s].
    Rows must form a regular lon-lat grid.
    """
    df = pd.read_csv(path)
    required = {"lon", "lat", "u_east", "v_north"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    lon_vals = np.sort(df["lon"].unique())
    lat_vals = np.sort(df["lat"].unique())

    u = np.zeros((len(lat_vals), len(lon_vals)))
    v = np.zeros((len(lat_vals), len(lon_vals)))

    lon_to_i = {v: i for i, v in enumerate(lon_vals)}
    lat_to_j = {v: j for j, v in enumerate(lat_vals)}

    for _, row in df.iterrows():
        i = lon_to_i[row["lon"]]
        j = lat_to_j[row["lat"]]
        u[j, i] = row["u_east"]
        v[j, i] = row["v_north"]

    return VectorField2D(lon_vals, lat_vals, u, v, time=time)


def generate_synthetic_current_field(
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    resolution: int = 50,
    base_u: float = 0.5,
    base_v: float = 0.0,
    noise_scale: float = 0.2,
    seed: int = 42,
) -> VectorField2D:
    """Generate a synthetic current field on a regular grid.

    The field consists of a uniform background flow plus smooth random variation.
    Useful for testing when real gridded data is unavailable.

    Args:
        lon_min, lon_max, lat_min, lat_max: Grid extent.
        resolution: Number of grid points in each dimension.
        base_u, base_v: Background flow velocity [m/s].
        noise_scale: Amplitude of spatial variation [m/s].
        seed: Random seed for reproducibility.

    Returns:
        VectorField2D with synthetic current data.
    """
    rng = np.random.default_rng(seed)
    lon_grid = np.linspace(lon_min, lon_max, resolution)
    lat_grid = np.linspace(lat_min, lat_max, resolution)

    # Smooth random field using low-frequency Fourier components
    xx, yy = np.meshgrid(lon_grid, lat_grid)
    u = np.full_like(xx, base_u, dtype=np.float64)
    v = np.full_like(yy, base_v, dtype=np.float64)

    for k in range(1, 5):
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        amp = noise_scale / k
        u += amp * np.sin(k * np.pi * (xx - lon_min) / (lon_max - lon_min) + phase_x)
        v += amp * np.cos(k * np.pi * (yy - lat_min) / (lat_max - lat_min) + phase_y)

    return VectorField2D(lon_grid, lat_grid, u, v)


def generate_synthetic_wind_field(
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
    resolution: int = 50,
    base_speed: float = 5.0,
    base_direction_deg: float = 270.0,  # westerly
    noise_scale: float = 2.0,
    seed: int = 43,
) -> VectorField2D:
    """Generate a synthetic wind field on a regular grid.

    Args:
        base_speed: Mean wind speed [m/s].
        base_direction_deg: Mean wind direction [deg] (meteorological convention:
            0=N, 90=E, 180=S, 270=W).
        noise_scale: Amplitude of spatial variation [m/s].
        seed: Random seed.

    Returns:
        VectorField2D with synthetic wind data.
    """
    rng = np.random.default_rng(seed)
    lon_grid = np.linspace(lon_min, lon_max, resolution)
    lat_grid = np.linspace(lat_min, lat_max, resolution)

    base_dir_rad = np.deg2rad(base_direction_deg)
    u_base = -base_speed * np.sin(base_dir_rad)  # meteorological → oceanographic
    v_base = -base_speed * np.cos(base_dir_rad)

    xx, yy = np.meshgrid(lon_grid, lat_grid)
    u = np.full_like(xx, u_base, dtype=np.float64)
    v = np.full_like(yy, v_base, dtype=np.float64)

    for k in range(1, 4):
        phase = rng.uniform(0, 2 * np.pi)
        amp = noise_scale / k
        u += amp * np.cos(k * np.pi * (xx - lon_min) / (lon_max - lon_min) + phase)
        v += amp * np.sin(k * np.pi * (yy - lat_min) / (lat_max - lat_min) + phase)

    return VectorField2D(lon_grid, lat_grid, u, v)


def build_environment_sequence(
    waterway_id: str,
    duration: float,
    dt: float,
    water_depth: float = 50.0,
    current_field: Optional[VectorField2D] = None,
    wind_field: Optional[VectorField2D] = None,
    channel_width: Optional[float] = None,
    bank_distance_left: Optional[float] = None,
    bank_distance_right: Optional[float] = None,
    ownship_track: Optional[list[tuple[float, float, float]]] = None,
) -> list[EnvironmentState]:
    """Build a time-varying EnvironmentState sequence.

    If current/wind fields are provided, samples them along the ownship track.
    Otherwise uses constant scalar values from defaults.

    Args:
        waterway_id: Waterway identifier.
        duration: Total duration [s].
        dt: Time step [s].
        water_depth: Default water depth [m].
        current_field: Optional gridded current field.
        wind_field: Optional gridded wind field.
        channel_width, bank_distance_left, bank_distance_right: Channel geometry.
        ownship_track: Optional list of (x_local, y_local, timestamp) for
            ownship's planned track. If provided, converts x,y→lon,lat for
            field sampling. If None, uses constant zero current/wind.

    Returns:
        List of EnvironmentState, length = int(duration / dt) + 1.
    """
    n_steps = int(duration / dt) + 1
    times = np.linspace(0, duration, n_steps)

    # Approximate lon/lat → local conversion factors
    # Use cos_lat for mid-latitude waterways (~45 deg N for US waterways)
    cos_lat = np.cos(np.deg2rad(45.0))
    meters_per_deg_lon = 111320.0 * cos_lat
    meters_per_deg_lat = 111320.0

    sequence = []
    for i, t in enumerate(times):
        cx, cy = 0.0, 0.0
        wx, wy = 0.0, 0.0

        if current_field is not None or wind_field is not None:
            # Get ownship position at this time step
            if ownship_track is not None and i < len(ownship_track):
                x_loc, y_loc, _ = ownship_track[i]
                lon = x_loc / meters_per_deg_lon
                lat = y_loc / meters_per_deg_lat
            else:
                lon, lat = 0.0, 0.0

            if current_field is not None:
                cx, cy = current_field.sample(lon, lat)
            if wind_field is not None:
                wx, wy = wind_field.sample(lon, lat)

        env = EnvironmentState(
            water_depth=water_depth,
            current_x=cx,
            current_y=cy,
            wind_x=wx,
            wind_y=wy,
            bank_distance_left=bank_distance_left,
            bank_distance_right=bank_distance_right,
            channel_width=channel_width,
            waterway_id=waterway_id,
        )
        sequence.append(env)

    return sequence


# ═══════════════════════════════════════════════════════════════════════════════
# GAP #5 — Tidal depth model (2026-06-10)
# ═══════════════════════════════════════════════════════════════════════════════

def tidal_depth_model(
    base_depth: float,
    t: float,
    amplitude: float = 1.5,
    period_hours: float = 12.42,
    phase_rad: float = 0.0,
) -> float:
    """Compute tidally-varying water depth.

    Semidiurnal tide model (M2 constituent dominant in most US ports):
        depth(t) = base_depth + amplitude * sin(2π·t / period_hours·3600 + phase)

    Args:
        base_depth: Chart datum / mean water depth [m].
        t: Simulation time [s].
        amplitude: Tidal amplitude [m] (default 1.5 m — moderate tide).
        period_hours: Tidal period [hours] (default 12.42 = M2 semidiurnal).
        phase_rad: Phase offset [rad].

    Returns:
        Time-varying water depth [m], never below 0.5 m.

    References:
        NOAA Tides & Currents — M2 constituent dominant period 12.4206 hours.
        Puget Sound: amplitude 1.5–3.0 m; New York Harbor: 1.0–2.0 m;
        San Francisco Bay: 1.0–2.5 m.
    """
    period_s = period_hours * 3600.0
    depth = base_depth + amplitude * np.sin(2.0 * np.pi * t / period_s + phase_rad)
    return float(max(depth, 0.5))  # minimum 0.5 m — never dry


# ═══════════════════════════════════════════════════════════════════════════════
# GAP #6 — Gust wind model (2026-06-10)
# ═══════════════════════════════════════════════════════════════════════════════

def gust_wind_model(
    base_wind_x: float,
    base_wind_y: float,
    t: float,
    gust_amplitude: float = 3.0,
    gust_period_s: float = 60.0,
    turbulence_intensity: float = 0.15,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute gusty wind components with stochastic turbulence.

    Models wind as:
        wind(t) = base_wind + gust_amplitude * sin(2π·t / gust_period)
                + turbulence * N(0, 1)

    Args:
        base_wind_x, base_wind_y: Mean wind components [m/s].
        t: Simulation time [s].
        gust_amplitude: Peak gust deviation from mean [m/s].
        gust_period_s: Characteristic gust oscillation period [s].
        turbulence_intensity: Fraction of gust_amplitude for random turbulence.
        seed: Deterministic seed for reproducibility.

    Returns:
        (wind_x, wind_y) at time t [m/s].

    References:
        DNV GL ST-0437: Loads and site conditions for wind turbines.
        IEC 61400-1 Ed.4: Wind turbulence models.
        Typical maritime gust: 3–8 m/s deviation, 30–120 s period.
    """
    rng = np.random.default_rng(seed + int(t * 1000) % (2**31))
    # Deterministic gust oscillation
    gust = gust_amplitude * np.sin(2.0 * np.pi * t / gust_period_s)
    # Random turbulence (scaled by intensity)
    turb = rng.normal(0.0, gust_amplitude * turbulence_intensity)

    base_mag = np.sqrt(base_wind_x**2 + base_wind_y**2)
    if base_mag > 0.01:
        # Apply gust + turbulence along the wind direction
        dir_x = base_wind_x / base_mag
        dir_y = base_wind_y / base_mag
        wx = base_wind_x + dir_x * gust + dir_x * turb
        wy = base_wind_y + dir_y * gust + dir_y * turb
    else:
        # No base wind → gust is isotropic
        angle = rng.uniform(0, 2 * np.pi)
        wx = (gust + turb) * np.cos(angle)
        wy = (gust + turb) * np.sin(angle)

    return float(wx), float(wy)
