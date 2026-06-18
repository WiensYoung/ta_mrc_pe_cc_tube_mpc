"""JAX port of bank effect assessment and disturbance force.

Mirrors ``physics/bank_effect.py``.
"""

import jax
import jax.numpy as jnp


def assess_bank_effect_jax(
    own_speed: float,
    own_length: float,
    own_beam: float,
    bank_distance_left: float,
    bank_distance_right: float,
) -> dict:
    """Assess bank effect risk using pure JAX.

    Returns dict with: active (bool), risk_level (float 0-1), side (str).
    """
    d_left = jnp.where(jnp.isfinite(bank_distance_left), bank_distance_left, jnp.inf)
    d_right = jnp.where(jnp.isfinite(bank_distance_right), bank_distance_right, jnp.inf)
    d_min = jnp.minimum(d_left, d_right)

    # Activation: within 5 ship lengths
    threshold = 5.0 * own_length
    active = d_min < threshold

    # Risk level: 1.0 at 0 distance, 0.0 at threshold
    risk_level = jnp.where(
        active,
        jnp.clip(1.0 - d_min / threshold, 0.0, 1.0),
        0.0,
    )

    # Which side is closer: -1 = left, +1 = right (numeric for JAX compatibility)
    side_flag = jnp.where(d_left < d_right, jnp.float64(-1.0), jnp.float64(1.0))

    return {
        "active": active,
        "risk_level": risk_level,
        "side_flag": side_flag,  # -1=left, +1=right
        "distance_to_bank": d_min,
    }


def compute_bank_disturbance_jax(
    risk_level: float,
    d_left: float,
    d_right: float,
    own_speed: float,
    own_length: float,
    gamma_b0: float = 0.05,
) -> jnp.ndarray:
    """Compute bank disturbance force [X, Y, N] in body frame.

    Conservative bounded disturbance model:
        ||w_bank|| <= gamma_b * (U_i/U_ref)^2 * (d_ref/max(d_bank, d_min))^2

    Returns (X_dist, Y_dist, N_dist) array.
    """
    U_ref = 7.0
    d_ref = own_length
    d_min = 1.0

    d_bank = jnp.minimum(
        jnp.where(jnp.isfinite(d_left), d_left, jnp.inf),
        jnp.where(jnp.isfinite(d_right), d_right, jnp.inf),
    )

    # Bank force magnitude
    speed_factor = (own_speed / U_ref)**2
    dist_factor = (d_ref / jnp.maximum(d_bank, d_min))**2
    mag = gamma_b0 * speed_factor * dist_factor * risk_level

    # Direction: pull toward nearest bank
    # Left bank → negative Y, Right bank → positive Y
    side_sign = jnp.where(
        jnp.where(jnp.isfinite(d_left), d_left, jnp.inf) < jnp.where(jnp.isfinite(d_right), d_right, jnp.inf),
        -1.0,  # pull toward left bank (negative Y)
        1.0,   # pull toward right bank (positive Y)
    )
    # If no bank nearby, sign is 0
    side_sign = jnp.where(jnp.isfinite(d_bank) & (d_bank < 5.0 * own_length), side_sign, 0.0)

    # Body-frame forces: bank mainly affects lateral (Y) force
    X_dist = 0.0
    Y_dist = side_sign * mag * 1e5  # scale to meaningful force
    N_dist = side_sign * mag * 1e4  # smaller yaw moment

    return jnp.array([X_dist, Y_dist, N_dist], dtype=jnp.float64)


@jax.jit
def compute_bank_disturbance_batched(
    risk_levels: jnp.ndarray,
    d_lefts: jnp.ndarray,
    d_rights: jnp.ndarray,
    own_speeds: jnp.ndarray,
    own_lengths: jnp.ndarray,
    gamma_b0: float = 0.05,
) -> jnp.ndarray:
    """Batch bank disturbance forces."""
    return jax.vmap(compute_bank_disturbance_jax, in_axes=(0, 0, 0, 0, 0, None))(
        risk_levels, d_lefts, d_rights, own_speeds, own_lengths, gamma_b0,
    )
