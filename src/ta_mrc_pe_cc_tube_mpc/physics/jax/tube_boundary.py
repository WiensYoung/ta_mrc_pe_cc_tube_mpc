"""JAX port of tube boundary computation.

Computes the total tube radius as the sum of independent disturbance
contributions.  Mirrors ``physics/tube_boundary.py``.
"""

import jax
import jax.numpy as jnp


def compute_tube_radius_jax(
    own_speed: float,
    own_length: float,
    own_beam: float,
    own_draught: float,
    n_targets: int,
    env_current_mag: float,
    env_wind_mag: float,
    water_depth: float,
    bank_distance_min: float,
    ship_interaction_risk: float,
    rho_0: float = 5.0,
    rho_maneuver: float = 2.0,
    use_adaptive: bool = True,
    config: dict = None,
) -> dict:
    """Compute tube radius using pure JAX operations.

    Args:
        own_speed: Ownship speed [m/s].
        own_length: Ownship length [m].
        own_beam: Ownship beam [m].
        own_draught: Ownship draught [m].
        n_targets: Number of target ships.
        env_current_mag: Current magnitude [m/s].
        env_wind_mag: Wind magnitude [m/s].
        water_depth: Water depth [m].
        bank_distance_min: Minimum distance to either bank [m] (inf if none).
        ship_interaction_risk: Ship interaction risk score [0, 1].
        rho_0: Base tube radius [m].
        rho_maneuver: Maneuver uncertainty contribution [m].
        use_adaptive: Whether to use adaptive scaling.
        config: Optional configuration dict.

    Returns:
        Dict with keys: "radius", "components" (dict of per-source contributions).
    """
    cfg = config or {}

    # ── Individual components ──────────────────────────────────────
    # AIS uncertainty contribution
    rho_ais = rho_0 * 0.2 * jnp.minimum(jnp.array(n_targets, dtype=jnp.float64), 4.0)

    # Current disturbance
    rho_current = rho_0 * 0.15 * jnp.clip(env_current_mag / 1.0, 0.0, 2.0)

    # Wind disturbance
    rho_wind = rho_0 * 0.1 * jnp.clip(env_wind_mag / 5.0, 0.0, 2.0)

    # Shallow water contribution
    depth_ratio = water_depth / jnp.maximum(own_draught, 0.5)
    rho_shallow = jnp.where(
        depth_ratio < 4.0,
        rho_0 * 0.25 * (1.0 - jnp.clip(depth_ratio / 4.0, 0.0, 1.0)),
        0.0,
    )

    # Bank effect contribution
    rho_bank = jnp.where(
        jnp.isfinite(bank_distance_min) & (bank_distance_min < 5.0 * own_length),
        rho_0 * 0.2 * (1.0 - jnp.clip(bank_distance_min / (5.0 * own_length), 0.0, 1.0)),
        0.0,
    )

    # Ship interaction contribution
    rho_ship = rho_0 * 0.15 * jnp.clip(ship_interaction_risk, 0.0, 1.0)

    # Maneuver uncertainty (speed-dependent)
    speed_ratio = own_speed / 7.0
    rho_man = rho_maneuver * jnp.clip(speed_ratio, 0.3, 3.0)

    # ── Total ──────────────────────────────────────────────────────
    rho_total = rho_0 + rho_ais + rho_current + rho_wind + rho_shallow + rho_bank + rho_ship + rho_man

    # Adaptive scaling — use jnp.where instead of Python if for JIT compatibility
    alpha_min = float(cfg.get("alpha_encounter_min", 0.5))
    alpha_max = float(cfg.get("alpha_encounter_max", 2.0))
    scaled = rho_total * jnp.clip(alpha_min + 0.5 * (n_targets / 2.0), alpha_min, alpha_max)
    rho_total = jnp.where(use_adaptive, scaled, rho_total)

    return {
        "radius": rho_total,
        "components": {
            "rho_0": rho_0,
            "rho_ais": rho_ais,
            "rho_current": rho_current,
            "rho_wind": rho_wind,
            "rho_shallow": rho_shallow,
            "rho_bank": rho_bank,
            "rho_ship": rho_ship,
            "rho_maneuver": rho_man,
        },
    }


# vmap-batched version for parallel evaluation
@jax.jit
def compute_tube_radius_batched(
    own_speeds: jnp.ndarray,
    own_lengths: jnp.ndarray,
    own_beams: jnp.ndarray,
    own_draughts: jnp.ndarray,
    n_targets_arr: jnp.ndarray,
    current_mags: jnp.ndarray,
    wind_mags: jnp.ndarray,
    water_depths: jnp.ndarray,
    bank_distances: jnp.ndarray,
    ship_risks: jnp.ndarray,
    rho_0: float = 5.0,
    rho_maneuver: float = 2.0,
) -> jnp.ndarray:
    """Batch compute tube radii for multiple states."""
    def _single(*args):
        return compute_tube_radius_jax(*args, rho_0=rho_0, rho_maneuver=rho_maneuver)["radius"]
    return jax.vmap(_single)(
        own_speeds, own_lengths, own_beams, own_draughts,
        n_targets_arr, current_mags, wind_mags,
        water_depths, bank_distances, ship_risks,
    )
