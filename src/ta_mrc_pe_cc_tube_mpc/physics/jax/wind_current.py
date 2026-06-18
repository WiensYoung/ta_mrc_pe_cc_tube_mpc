"""JAX port of wind and current disturbance assessment.

Mirrors ``physics/wind_current.py``.
"""

import jax
import jax.numpy as jnp


def assess_wind_current_jax(
    current_x: float,
    current_y: float,
    wind_x: float,
    wind_y: float,
    own_length: float,
    own_beam: float,
    own_speed: float,
    own_psi: float = 0.0,
) -> dict:
    """Assess wind and current conditions using pure JAX.

    Returns dict with: current_mag, wind_mag, both magnitudes.
    """
    current_mag = jnp.sqrt(current_x**2 + current_y**2)
    wind_mag = jnp.sqrt(wind_x**2 + wind_y**2)

    return {
        "current_magnitude": current_mag,
        "wind_magnitude": wind_mag,
        "total_magnitude": jnp.sqrt(current_mag**2 + wind_mag**2),
    }


def compute_wind_force_jax(
    wind_x: float,
    wind_y: float,
    own_psi: float,
    own_length: float,
    own_beam: float,
    Cd_wind: float = 1.2,
) -> jnp.ndarray:
    """Compute wind force [X, Y, N] in body frame.

    Lateral windage area proportional to L * freeboard.
    Frontal windage area proportional to beam * freeboard.

    Returns (X_dist, Y_dist, N_dist) array.
    """
    # Wind magnitude and direction in world frame
    wind_mag = jnp.sqrt(wind_x**2 + wind_y**2)
    wind_dir = jnp.arctan2(wind_y, wind_x)

    # Wind direction relative to ship heading
    wind_rel = wind_dir - own_psi
    wind_u_body = wind_mag * jnp.cos(wind_rel)  # surge component
    wind_v_body = wind_mag * jnp.sin(wind_rel)  # sway component

    # Windage areas (rough estimates)
    freeboard = 0.06 * own_length
    lateral_area = own_length * 0.15 * freeboard  # side profile
    frontal_area = own_beam * freeboard           # front profile

    rho_air = 1.225
    q_wind = 0.5 * rho_air * wind_mag**2

    # Forces proportional to dynamic pressure × area × Cd
    # Use directional projection (cos/sin of wind angle) not just sign
    wind_rel = jnp.arctan2(wind_v_body, wind_u_body + 1e-12)
    X_dist = -Cd_wind * q_wind * frontal_area * jnp.cos(wind_rel) * 0.5
    Y_dist = Cd_wind * q_wind * lateral_area * jnp.sin(wind_rel)
    N_dist = Y_dist * own_length * 0.1  # yaw moment from lateral wind

    return jnp.array([X_dist, Y_dist, N_dist], dtype=jnp.float64)


def compute_current_effect_jax(
    current_x: float,
    current_y: float,
    own_psi: float,
    own_speed: float,
    own_length: float,
) -> jnp.ndarray:
    """Compute current drift force [X, Y, N] in body frame.

    Current enters kinematics additively; this estimates the additional
    drift force when the ship is not perfectly compensating.

    Returns (X_dist, Y_dist, N_dist) array.
    """
    # Current in body frame
    cos_psi = jnp.cos(own_psi)
    sin_psi = jnp.sin(own_psi)
    u_current_body = current_x * cos_psi + current_y * sin_psi
    v_current_body = -current_x * sin_psi + current_y * cos_psi

    current_mag = jnp.sqrt(u_current_body**2 + v_current_body**2)
    rho_water = 1025.0
    Cd_current = 0.5

    # Wetted area approximation
    wetted_area = own_length * own_length * 0.04
    q_current = 0.5 * rho_water * current_mag**2

    X_dist = Cd_current * q_current * wetted_area * jnp.sign(u_current_body) * 0.1
    Y_dist = Cd_current * q_current * wetted_area * jnp.sign(v_current_body)
    N_dist = Y_dist * own_length * 0.02

    return jnp.array([X_dist, Y_dist, N_dist], dtype=jnp.float64)


@jax.jit
def compute_wind_force_batched(
    wind_xs: jnp.ndarray,
    wind_ys: jnp.ndarray,
    own_psis: jnp.ndarray,
    own_lengths: jnp.ndarray,
    own_beams: jnp.ndarray,
) -> jnp.ndarray:
    """Batch wind forces."""
    return jax.vmap(compute_wind_force_jax)(wind_xs, wind_ys, own_psis, own_lengths, own_beams)
