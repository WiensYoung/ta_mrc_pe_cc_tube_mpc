"""JAX port of shallow water assessment and disturbance.

Mirrors ``physics/shallow_water.py``.
"""

import jax
import jax.numpy as jnp


def assess_shallow_water_jax(
    water_depth: float,
    vessel_draught: float,
    own_speed: float,
    own_length: float,
) -> dict:
    """Assess shallow water effects using pure JAX.

    Classification based on depth/draft ratio:
        > 4.0 → deep water
        3.0-4.0 → mild shallow
        2.0-3.0 → moderate shallow
        1.5-2.0 → strong shallow
        < 1.5 → severe / forbidden

    Returns dict with: shallow_level (0-4), lambda_h (ratio), active (bool).
    """
    lambda_h = water_depth / jnp.maximum(vessel_draught, 0.5)

    # Level: 0=deep, 1=mild, 2=moderate, 3=strong, 4=severe
    shallow_level = jnp.where(
        lambda_h > 4.0, 0,
        jnp.where(lambda_h > 3.0, 1,
                  jnp.where(lambda_h > 2.0, 2,
                            jnp.where(lambda_h > 1.5, 3, 4))),
    )

    return {
        "shallow_level": shallow_level,
        "lambda_h": lambda_h,
        "active": shallow_level > 0,
    }


def compute_shallow_water_disturbance_jax(
    shallow_level: int,
    own_speed: float,
    gamma_s0: float = 0.03,
) -> jnp.ndarray:
    """Compute shallow water disturbance force [X, Y, N].

    Returns (X_dist, Y_dist, N_dist) array.
    """
    # Resistance multiplier per level
    resistance_factor = jnp.array([1.0, 1.2, 1.5, 2.0, 3.0])[shallow_level]

    # Added resistance in surge
    X_dist = -gamma_s0 * own_speed**2 * (resistance_factor - 1.0) * 1e5

    # Reduced lateral controllability
    Y_dist = jnp.array(0.0)
    N_dist = jnp.array(0.0)

    return jnp.array([X_dist, Y_dist, N_dist], dtype=jnp.float64)


@jax.jit
def compute_shallow_water_disturbance_batched(
    shallow_levels: jnp.ndarray,
    own_speeds: jnp.ndarray,
    gamma_s0: float = 0.03,
) -> jnp.ndarray:
    """Batch shallow water disturbance forces."""
    return jax.vmap(compute_shallow_water_disturbance_jax, in_axes=(0, 0, None))(
        shallow_levels, own_speeds, gamma_s0,
    )
