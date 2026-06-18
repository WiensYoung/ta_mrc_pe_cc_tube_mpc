"""Target ship model: trajectory propagation and uncertainty injection."""

import numpy as np

from ..constants import KNOTS_TO_MPS
from ..types import TargetShip, VesselState


def propagate_target(
    target: TargetShip,
    dt: float,
    duration: float,
    trajectory: list[VesselState] = None,
) -> list[VesselState]:
    """Generate or extend a target ship trajectory.

    If a predefined trajectory is provided, it is used directly.
    Otherwise, the target maintains constant course and speed.

    Args:
        target: Target ship configuration.
        dt: Time step [s].
        duration: Total duration [s].
        trajectory: Optional predefined trajectory.

    Returns:
        List of VesselState at each time step.
    """
    if trajectory is not None and len(trajectory) > 0:
        # Use provided trajectory, pad if needed
        n_needed = int(duration / dt) + 1
        if len(trajectory) >= n_needed:
            return trajectory[:n_needed]
        # Extend with constant velocity from last state
        last = trajectory[-1]
        extra = []
        c = np.cos(last.psi)
        s = np.sin(last.psi)
        for k in range(len(trajectory), n_needed):
            t = k * dt
            dt_val = t - last.timestamp
            # Full body-to-world transform: include sway velocity v
            extra.append(VesselState(
                x=last.x + (last.u * c - last.v * s) * dt_val,
                y=last.y + (last.u * s + last.v * c) * dt_val,
                psi=last.psi, u=last.u, v=last.v, r=last.r, timestamp=t,
            ))
        return trajectory + extra

    # Constant course and speed
    n_steps = int(duration / dt) + 1
    states = []
    state0 = target.state
    c0 = np.cos(state0.psi)
    s0 = np.sin(state0.psi)
    for k in range(n_steps):
        t = k * dt
        # Full body-to-world transform: include sway velocity v
        states.append(VesselState(
            x=state0.x + (state0.u * c0 - state0.v * s0) * t,
            y=state0.y + (state0.u * s0 + state0.v * c0) * t,
            psi=state0.psi, u=state0.u, v=state0.v, r=state0.r, timestamp=t,
        ))
    return states


def get_noisy_target_state(
    true_state: VesselState,
    target: TargetShip,
    current_time: float,
    rng: np.random.Generator,
) -> VesselState:
    """Apply AIS delay and noise to true target state.

    Args:
        true_state: True target state at current time.
        target: Target ship with AIS quality config.
        current_time: Current simulation time [s].
        rng: Seeded random state.

    Returns:
        Noisy (observed) target state, or None if dropout.
    """
    ais_cfg = target.ais_quality
    delay = ais_cfg.get("delay", 0.0)
    dropout_prob = ais_cfg.get("dropout_prob", 0.0)
    noise_std = ais_cfg.get("position_noise_std", 10.0)

    # Dropout
    if rng.random() < dropout_prob:
        return None

    # Delay: back-project position to (current_time - delay).
    # Full body-to-world transform including sway velocity v.
    c = np.cos(true_state.psi)
    s = np.sin(true_state.psi)
    delayed = VesselState(
        x=true_state.x - (true_state.u * c - true_state.v * s) * delay,
        y=true_state.y - (true_state.u * s + true_state.v * c) * delay,
        psi=true_state.psi,
        u=true_state.u,
        v=true_state.v,
        r=true_state.r,
        timestamp=true_state.timestamp - delay,
    )

    # Add Gaussian noise
    noisy = VesselState(
        x=delayed.x + rng.normal(0, noise_std),
        y=delayed.y + rng.normal(0, noise_std),
        psi=delayed.psi + rng.normal(0, 0.5) * np.pi / 180.0,
        u=delayed.u + rng.normal(0, 0.1),
        v=delayed.v + rng.normal(0, 0.1),
        r=delayed.r,
        timestamp=current_time,
    )
    return noisy


def update_target_covariance(
    target: TargetShip,
    delay: float,
    dropout_occurred: bool,
) -> np.ndarray:
    """Update target position covariance based on AIS quality.

    Args:
        target: Target ship.
        delay: Current AIS delay [s].
        dropout_occurred: Whether dropout occurred at this step.

    Returns:
        Updated 2x2 covariance matrix.
    """
    base_std = target.ais_quality.get("position_noise_std", 10.0)
    # covariance can be None when target is constructed from data without it
    Sigma = target.covariance.copy() if target.covariance is not None else np.eye(2) * (base_std ** 2)

    # Inflate covariance with delay
    if delay > 0:
        v_mag = target.state.speed()
        delay_inflation = (v_mag * delay) ** 2
        Sigma += np.eye(2) * delay_inflation

    # Inflate covariance if dropout occurred
    if dropout_occurred:
        Sigma *= 2.0

    return Sigma
