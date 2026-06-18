"""Wind and current disturbance models.

Current: enters the kinematics additively (relative velocity).
Wind: treated as a bounded disturbance force.

Configurable current levels: 0, 1, 2, 3 kn cross-current.
"""

from dataclasses import dataclass

import numpy as np

from ..constants import KNOTS_TO_MPS


@dataclass
class WindCurrentResult:
    """Output of wind/current assessment."""
    current_magnitude: float       # [m/s]
    current_direction: float       # [rad]
    wind_magnitude: float          # [m/s]
    wind_direction: float          # [rad]
    rho_current: float             # tube radius contribution [m]
    rho_wind: float                # tube radius contribution [m]
    wind_force: np.ndarray         # [X, Y, N] in body frame
    current_stress_test: bool      # True if current >= 2.5 kn


def assess_wind_current(
    current_x: float,
    current_y: float,
    wind_x: float,
    wind_y: float,
    vessel_length: float,
    vessel_beam: float,
    vessel_speed: float,
    own_psi: float = 0.0,
    U_ref: float = 7.0,
    config: dict = None,
) -> WindCurrentResult:
    """Assess wind and current effects.

    Args:
        current_x, current_y: Current velocity components in WORLD frame [m/s].
        wind_x, wind_y: Wind velocity components in WORLD frame [m/s].
        vessel_length: Vessel length L [m].
        vessel_beam: Vessel beam B [m].
        vessel_speed: Current vessel speed [m/s].
        own_psi: Ownship heading angle [rad] — required to rotate wind
                 into the vessel body frame for correct lateral force sign.
        U_ref: Reference speed [m/s].
        config: Optional overrides.

    Returns:
        WindCurrentResult.
    """
    cfg = config or {}
    k_current = cfg.get("k_current", 0.01)
    k_wind = cfg.get("k_wind", 0.005)

    current_mag = float(np.sqrt(current_x**2 + current_y**2))
    current_dir = float(np.arctan2(current_y, current_x))
    wind_mag = float(np.sqrt(wind_x**2 + wind_y**2))
    wind_dir = float(np.arctan2(wind_y, wind_x))

    # Tube radius contributions
    U_ratio = max(vessel_speed / U_ref, 0.01)
    rho_current = k_current * (current_mag / U_ref) * vessel_length * U_ratio
    rho_wind = k_wind * (wind_mag / U_ref) * vessel_length * U_ratio

    # Wind force (simplified: proportional to projected area and wind speed squared)
    rho_air = 1.225
    C_wind = cfg.get("C_wind", 1.0)
    A_transverse = vessel_length * 15.0  # approximate above-water lateral area [m^2]
    # Typical cargo ships have 15-20m of freeboard + superstructure.
    # Previous value of 5.0 underestimated windage by 3-4x.
    wind_force_mag = 0.5 * rho_air * C_wind * A_transverse * wind_mag**2

    # Wind force in body frame — rotate world-frame wind into body frame
    # using the vessel heading (own_psi).
    cos_psi, sin_psi = np.cos(own_psi), np.sin(own_psi)
    # Wind body-frame components: x' = wx*c + wy*s, y' = -wx*s + wy*c
    wind_body_x = wind_x * cos_psi + wind_y * sin_psi
    wind_body_y = -wind_x * sin_psi + wind_y * cos_psi
    # Lateral force sign: wind from port (positive body-y) → positive sway
    wind_lateral_sign = 1.0 if wind_body_y >= 0 else -1.0
    wind_force = np.array([
        -0.1 * wind_force_mag,                                  # surge: small drag increase
        wind_lateral_sign * wind_force_mag * 0.3,               # sway: lateral force
        wind_lateral_sign * wind_force_mag * 0.1 * vessel_length,  # yaw moment
    ])

    current_kn = current_mag / KNOTS_TO_MPS
    stress_test = current_kn >= 2.5

    return WindCurrentResult(
        current_magnitude=current_mag,
        current_direction=current_dir,
        wind_magnitude=wind_mag,
        wind_direction=wind_dir,
        rho_current=rho_current,
        rho_wind=rho_wind,
        wind_force=wind_force,
        current_stress_test=stress_test,
    )
