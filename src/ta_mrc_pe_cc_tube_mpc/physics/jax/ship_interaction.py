"""JAX port of ship-ship hydrodynamic interaction.

Mirrors ``physics/ship_interaction.py``.
"""

import jax
import jax.numpy as jnp


def assess_ship_interaction_jax(
    own_x: float, own_y: float, own_psi: float,
    own_length: float, own_beam: float,
    tgt_x: float, tgt_y: float, tgt_psi: float,
    tgt_u: float, tgt_v: float,
    tgt_length: float, tgt_beam: float,
) -> dict:
    """Assess ship-ship interaction risk using pure JAX.

    Activation criteria:
      - d_ij < k_L * (L_i + L_j)  OR
      - lateral distance < k_B * (B_i + B_j)

    Returns dict with: active (bool), risk_score (0-1).
    """
    k_L = 2.0
    k_B = 5.0

    # Distance between ships
    dx = own_x - tgt_x
    dy = own_y - tgt_y
    d_ij = jnp.sqrt(dx**2 + dy**2)

    # Longitudinal and lateral separation in ownship frame
    cos_psi = jnp.cos(own_psi)
    sin_psi = jnp.sin(own_psi)
    dx_body = dx * cos_psi + dy * sin_psi
    dy_body = -dx * sin_psi + dy * cos_psi

    d_long = jnp.abs(dx_body)
    d_lat = jnp.abs(dy_body)

    # Activation thresholds
    active_long = d_long < k_L * (own_length + tgt_length)
    active_lat = d_lat < k_B * (own_beam + tgt_beam)
    active = active_long | active_lat

    # Risk score: 1.0 at zero distance, decays to 0 at threshold
    long_risk = jnp.where(
        active_long,
        jnp.clip(1.0 - d_long / (k_L * (own_length + tgt_length)), 0.0, 1.0),
        0.0,
    )
    lat_risk = jnp.where(
        active_lat,
        jnp.clip(1.0 - d_lat / (k_B * (own_beam + tgt_beam)), 0.0, 1.0),
        0.0,
    )
    risk_score = jnp.maximum(long_risk, lat_risk)

    return {
        "active": active,
        "risk_score": risk_score,
        "distance": d_ij,
        "d_long": d_long,
        "d_lat": d_lat,
        "dx_body": dx_body,  # signed, for force direction
        "dy_body": dy_body,  # signed, for force direction
    }


def compute_ship_interaction_jax(
    risk_score: float,
    d_ij: float,
    d_long: float,
    d_lat: float,
    own_speed: float,
    gamma_s0: float = 0.04,
    dx_body: float = 0.0,
    dy_body: float = 0.0,
) -> jnp.ndarray:
    """Compute ship interaction disturbance force [X, Y, N].

    Args:
        risk_score: Interaction risk score [0,1].
        d_ij: Distance between ships [m].
        d_long: Longitudinal separation (absolute) [m].
        d_lat: Lateral separation (absolute) [m].
        own_speed: Ownship speed [m/s].
        gamma_s0: Base interaction coefficient.
        dx_body: Signed longitudinal offset in body frame [m].
        dy_body: Signed lateral offset in body frame [m].

    Returns (X_dist, Y_dist, N_dist) array.
    """
    U_ref = 7.0
    speed_factor = (own_speed / U_ref)**2

    # Force magnitude proportional to risk and inverse distance
    mag = gamma_s0 * risk_score * speed_factor * 1e5

    # Direction: repulsive force away from closest approach
    # Use signed body-frame offsets for correct direction
    lat_sign = jnp.sign(dy_body)
    long_sign = jnp.sign(dx_body)

    X_dist = -long_sign * mag * 0.3  # smaller longitudinal component
    Y_dist = -lat_sign * mag         # dominant lateral component
    N_dist = -lat_sign * mag * 0.05  # small yaw moment

    return jnp.array([X_dist, Y_dist, N_dist], dtype=jnp.float64)


@jax.jit
def compute_ship_interaction_batched(
    risk_scores: jnp.ndarray,
    distances: jnp.ndarray,
    d_longs: jnp.ndarray,
    d_lats: jnp.ndarray,
    own_speeds: jnp.ndarray,
    gamma_s0: float = 0.04,
    dx_bodies: jnp.ndarray = None,
    dy_bodies: jnp.ndarray = None,
) -> jnp.ndarray:
    """Batch ship interaction forces."""
    if dx_bodies is None:
        dx_bodies = jnp.zeros_like(risk_scores)
    if dy_bodies is None:
        dy_bodies = jnp.zeros_like(risk_scores)
    return jax.vmap(compute_ship_interaction_jax, in_axes=(0, 0, 0, 0, 0, None, 0, 0))(
        risk_scores, distances, d_longs, d_lats, own_speeds, gamma_s0, dx_bodies, dy_bodies,
    )
