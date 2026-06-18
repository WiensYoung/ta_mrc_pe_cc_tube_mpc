"""JAX-accelerated physics modules for GPU-batched disturbance computation.

Each module mirrors its NumPy counterpart in ``physics/`` but uses pure
JAX operations, enabling JIT compilation, vmap batching, and GPU execution.

Modules:
    - tube_boundary    — compute_tube_radius_jax, compute_tube_radius_batched
    - bank_effect      — assess_bank_effect_jax, compute_bank_disturbance_jax
    - ship_interaction — assess_ship_interaction_jax, compute_ship_interaction_jax
    - shallow_water    — assess_shallow_water_jax, compute_shallow_water_disturbance_jax
    - wind_current     — assess_wind_current_jax, compute_wind_force_jax

Usage::

    from ta_mrc_pe_cc_tube_mpc.physics.jax import (
        compute_tube_radius_jax,
        compute_bank_disturbance_jax,
        compute_shallow_water_disturbance_jax,
        compute_wind_force_jax,
        compute_ship_interaction_jax,
    )
"""

from .bank_effect import assess_bank_effect_jax, compute_bank_disturbance_jax
from .shallow_water import assess_shallow_water_jax, compute_shallow_water_disturbance_jax
from .ship_interaction import assess_ship_interaction_jax, compute_ship_interaction_jax
from .tube_boundary import compute_tube_radius_jax
from .wind_current import assess_wind_current_jax, compute_wind_force_jax
