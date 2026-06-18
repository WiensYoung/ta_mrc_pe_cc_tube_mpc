"""COLREGs-role-aware target behavior hypothesis generation.

Generates multiple complete trajectory hypotheses per target vessel:
  1. Constant velocity (baseline)
  2. Turn-rate estimated from AIS history
  3. COLREGs-role-aware maneuver (uses target_role, NOT ownship_role)
  4. Channel-constrained
  5. Non-compliant (wrong-direction turn toward ownship)
  6. Emergency evasion (hard starboard + deceleration)
  7. Erratic (random walk in heading)

**Trajectory-level selection** (v2): the SINGLE complete hypothesis
with the smallest minimum distance to ownship across all time steps is
selected.  This avoids the physically impossible per-step hypothesis
switching present in v1.

All hypotheses are **conservative behavior envelopes** for safety
planning — they do NOT claim to predict the target's true intent.
The module is designed for single-agent control: the ownship plans
for the worst plausible target behavior without assuming cooperation,
consistent with the IMO MASS Code transparency principle.

Key fix (v2): COLREGs maneuver uses ``target_role`` from the encounter
classifier, not ``ownship_role``.  Previously ownship's role was
mistakenly used to predict target behavior.
"""

import hashlib
from enum import Enum
from typing import Optional

import numpy as np

from ..types import TargetShip, VesselState


class TargetBehaviorMode(str, Enum):
    """Behavioral modes for target ship trajectory prediction.

    Used to generate richer prediction hypotheses beyond constant velocity.
    The MPC selects the most conservative (minimum-distance) hypothesis
    for safety — this is safety-preserving under non-cooperative targets.

    Relevant to the discussion on single-agent vs multi-agent collision
    avoidance: our framework controls only the ownship and does not assume
    cooperation from target vessels, consistent with the IMO MASS Code's
    staged deployment framework where early autonomous ships operate
    alongside conventional manned vessels.
    """
    CONSTANT_VELOCITY = "constant_velocity"
    """Maintain current heading and speed (baseline hypothesis)."""

    COLREGS_COMPLIANT = "colregs_compliant"
    """Expected COLREGs-compliant maneuver:
       - Give-way vessel: starboard turn after ~30s reaction delay
       - Stand-on vessel: maintain course and speed
       - Head-on: both alter to starboard
    """

    NON_COMPLIANT = "non_compliant"
    """Non-COLREGs-compliant behavior:
       - Wrong-direction turn
       - No action when give-way
       - Speed change that increases collision risk
       (Conservative hypothesis for worst-case planning.)
    """

    CHANNEL_FOLLOWING = "channel_following"
    """Stay within waterway channel boundaries, adjust heading to
    follow channel centerline when approaching edges.
    """

    EMERGENCY_EVASION = "emergency_evasion"
    """Aggressive evasion: hard starboard turn, full astern propulsion.
    (Low-probability but worst-case for safety analysis.)
    """

    ERRATIC = "erratic"
    """Unpredictable behavior: random walk in heading with ±3° std.
    (Captures human error, mechanical failure, miscommunication.)
    """


def predict_target_trajectories(
    targets: list[TargetShip],
    own_state: VesselState,
    horizon_steps: int,
    dt: float,
    history: dict[str, list[VesselState]] = None,
    channel_constraints: dict = None,
) -> list[np.ndarray]:
    """Predict target positions over the MPC horizon.

    **Trajectory-level worst-branch selection** (v2):
    For each target, generates multiple COMPLETE trajectory hypotheses
    (constant-velocity, COLREGs-compliant, non-compliant, channel-following,
    emergency-evasion, erratic).  Selects the SINGLE hypothesis whose
    minimum distance to ownship across all time steps is smallest — i.e.,
    the most conservative COMPLETE trajectory.  This avoids the physically
    impossible per-step hypothesis switching (bug in v1).

    Returns:
        List of (horizon_steps) arrays of shape (n_targets, 2).
    """
    n_targets = len(targets)
    if n_targets == 0:
        return []

    # Compute ownship trajectory for distance reference (constant-velocity)
    own_c = np.cos(own_state.psi)
    own_s = np.sin(own_state.psi)
    own_traj = np.zeros((horizon_steps, 2))
    for k in range(horizon_steps):
        t = k * dt
        own_traj[k, 0] = own_state.x + (own_state.u * own_c - own_state.v * own_s) * t
        own_traj[k, 1] = own_state.y + (own_state.u * own_s + own_state.v * own_c) * t

    predictions = []
    for k in range(horizon_steps):
        predictions.append(np.zeros((n_targets, 2)))

    for j, target in enumerate(targets):
        hypotheses = _generate_hypotheses(
            target, own_state, horizon_steps, dt,
            history.get(target.mmsi, []) if history else [],
            channel_constraints,
        )
        if not hypotheses:
            continue

        # ── Trajectory-level selection: pick the SINGLE complete hypothesis
        #     whose minimum distance to ownship across all steps is smallest.
        best_hyp = None
        best_min_dist = float("inf")
        for hyp in hypotheses:
            min_d_sq = float("inf")
            for k in range(horizon_steps):
                pos = hyp[min(k, len(hyp) - 1)]
                dx = pos[0] - own_traj[k, 0]
                dy = pos[1] - own_traj[k, 1]
                d_sq = dx**2 + dy**2
                if d_sq < min_d_sq:
                    min_d_sq = d_sq
            if min_d_sq < best_min_dist:
                best_min_dist = min_d_sq
                best_hyp = hyp

        # Use the selected hypothesis for ALL steps (consistent trajectory)
        if best_hyp is not None:
            for k in range(horizon_steps):
                pos = best_hyp[min(k, len(best_hyp) - 1)]
                predictions[k][j, 0] = pos[0]
                predictions[k][j, 1] = pos[1]

    return predictions


def _generate_hypotheses(
    target: TargetShip,
    own_state: VesselState,
    horizon_steps: int,
    dt: float,
    history: list[VesselState],
    channel_constraints: dict = None,
) -> list[list[np.ndarray]]:
    """Generate multiple trajectory hypotheses for a single target.

    Returns list of trajectories, each a list of (horizon_steps) positions.
    """
    hypotheses = []
    s = target.state

    # Hypothesis 1: Constant velocity (baseline)
    # Full body-to-world transform including sway velocity v
    cv_traj = []
    c = np.cos(s.psi)
    ss = np.sin(s.psi)
    for k in range(horizon_steps):
        t = k * dt
        x = s.x + (s.u * c - s.v * ss) * t
        y = s.y + (s.u * ss + s.v * c) * t
        cv_traj.append(np.array([x, y]))
    hypotheses.append(cv_traj)

    # Hypothesis 2: Estimated turn rate from history
    if len(history) >= 3:
        r_est = _estimate_turn_rate(history)
        if abs(r_est) > 0.001:
            turn_traj = []
            psi = s.psi
            x, y = s.x, s.y
            speed = s.speed()
            for k in range(horizon_steps):
                t = k * dt
                if k > 0:
                    psi_prev = psi
                    psi = psi_prev + r_est * dt
                    x = x + speed * np.cos((psi + psi_prev) / 2) * dt
                    y = y + speed * np.sin((psi + psi_prev) / 2) * dt
                turn_traj.append(np.array([x, y]))
            hypotheses.append(turn_traj)

    # Hypothesis 3: COLREGs-expected maneuver
    colregs_traj = _predict_colregs_maneuver(
        target, own_state, horizon_steps, dt,
    )
    if colregs_traj is not None:
        hypotheses.append(colregs_traj)

    # Hypothesis 4: Channel-constrained
    if channel_constraints:
        channel_traj = _predict_channel_constrained(
            target, horizon_steps, dt, channel_constraints,
        )
        if channel_traj is not None:
            hypotheses.append(channel_traj)

    # Hypothesis 5: Non-compliant (worst-case) behavior
    noncompliant_traj = _predict_non_compliant(
        target, own_state, horizon_steps, dt,
    )
    if noncompliant_traj is not None:
        hypotheses.append(noncompliant_traj)

    # Hypothesis 6: Emergency evasion (low-probability worst-case)
    emergency_traj = _predict_emergency_evasion(
        target, horizon_steps, dt,
    )
    if emergency_traj is not None:
        hypotheses.append(emergency_traj)

    # Hypothesis 7: Erratic behavior (random walk in heading)
    erratic_traj = _predict_erratic(
        target, horizon_steps, dt,
    )
    if erratic_traj is not None:
        hypotheses.append(erratic_traj)

    return hypotheses


def _estimate_turn_rate(history: list[VesselState]) -> float:
    """Estimate yaw rate from recent state history.

    History list is ordered [oldest, ..., newest]. We assign exponentially
    increasing weights so that recent pairs contribute most.
    """
    if len(history) < 2:
        return 0.0
    n_hist = len(history)
    total_r = 0.0
    total_w = 0.0
    for i in range(1, min(n_hist, 5)):
        dt_i = history[i].timestamp - history[i - 1].timestamp
        if dt_i > 0.001:
            dpsi = history[i].psi - history[i - 1].psi
            dpsi = np.arctan2(np.sin(dpsi), np.cos(dpsi))
            # Weight increases with recency: newest pair weight=1.0,
            # oldest pair weight=2^(-(n_hist-2)) ≈ 0.125 for n_hist=5
            w = 2.0 ** (i - n_hist + 1)
            total_r += w * dpsi / dt_i
            total_w += w
    return total_r / max(total_w, 1e-6)


def _predict_colregs_maneuver(
    target: TargetShip,
    own_state: VesselState,
    horizon_steps: int,
    dt: float,
) -> Optional[list[np.ndarray]]:
    """Predict **target vessel** trajectory under typical COLREGs behavior.

    Uses ``target_role`` (NOT ownship_role) to decide what the target
    is expected to do:
      - target is give_way → target turns starboard after delay
      - target is stand_on → target maintains course/speed (no maneuver)
      - head_on → both give way → target turns starboard

    This is a **hypothesis** for safety planning, not a claim that the
    target *will* behave this way.
    """
    from ..risk.encounter_classifier import classify_encounter

    enc = classify_encounter(own_state, target.state)
    etype = enc.get("type", "none")
    # ── Use TARGET role for target behavior prediction ──────────────────
    target_role = enc.get("target_role", "none")

    s = target.state
    traj = []

    # Only adjust if the TARGET has a clear give-way responsibility
    turn_delay_steps = max(1, int(30.0 / dt))
    turn_rate = 0.0

    if etype == "head_on":
        # Both vessels give way → target expected to turn starboard
        turn_rate = np.deg2rad(1.0)
    elif target_role == "give_way":
        if "crossing" in etype:
            turn_rate = np.deg2rad(0.8)
        elif "overtaking" in etype:
            turn_rate = np.deg2rad(0.5)

    if abs(turn_rate) < 1e-6:
        return None  # Target is stand-on or no encounter → no maneuver expected

    x, y = s.x, s.y
    psi = s.psi
    speed = s.speed()

    for k in range(horizon_steps):
        if k >= turn_delay_steps:
            psi_prev = psi
            psi = psi_prev + turn_rate * dt
            x = x + speed * np.cos((psi + psi_prev) / 2) * dt
            y = y + speed * np.sin((psi + psi_prev) / 2) * dt
        else:
            x = x + speed * np.cos(psi) * dt
            y = y + speed * np.sin(psi) * dt
        traj.append(np.array([x, y]))

    return traj


def _predict_channel_constrained(
    target: TargetShip,
    horizon_steps: int,
    dt: float,
    channel_constraints: dict,
) -> Optional[list[np.ndarray]]:
    """Predict target trajectory constrained to stay within channel."""
    s = target.state
    half_w = channel_constraints.get("half_width")
    if half_w is None:
        return None

    x, y = s.x, s.y
    psi = s.psi
    speed = s.speed()
    traj = []

    for k in range(horizon_steps):
        # Apply course correction BEFORE computing position for this step
        y_tentative = y + speed * np.sin(psi) * dt
        if abs(y_tentative) > half_w * 0.9:
            correction = -np.sign(y_tentative) * np.deg2rad(2.0) * dt
            psi = psi + correction

        # Compute position with the (possibly corrected) heading
        x = x + speed * np.cos(psi) * dt
        y = y + speed * np.sin(psi) * dt
        traj.append(np.array([x, y]))

    return traj


# ---------------------------------------------------------------------------
# Additional behavior hypothesis generators
# ---------------------------------------------------------------------------

# NOTE: _select_most_conservative (per-step hypothesis selection) has been
# removed in favour of trajectory-level selection in predict_target_trajectories
# (v2, line ~130).  Trajectory-level selection avoids the physically impossible
# per-step hypothesis switching present in the original per-step implementation.


def _predict_non_compliant(
    target: TargetShip,
    own_state: VesselState,
    horizon_steps: int,
    dt: float,
    turn_rate_deg: float = 1.5,
) -> list[np.ndarray]:
    """Predict target trajectory under non-COLREGs-compliant behavior.

    The target makes a WRONG-direction turn that increases collision risk.
    This is a conservative hypothesis for the ownship MPC: we plan for
    the worst plausible target behavior, consistent with the safety-first
    principle that an autonomous vessel must not assume cooperation.

    Args:
        target: Target ship.
        own_state: Ownship state (for determining which direction is 'wrong').
        horizon_steps: Prediction horizon steps.
        dt: Time step [s].
        turn_rate_deg: Aggressiveness of wrong turn [deg/s].

    Returns:
        List of (horizon_steps) positions, or None if no maneuver predicted.
    """
    s = target.state
    turn_delay_steps = max(1, int(15.0 / dt))  # shorter delay for unexpected behavior

    # Determine which direction INCREASES collision risk
    # (turning TOWARD ownship rather than away)
    dx = own_state.x - s.x
    dy = own_state.y - s.y
    # Compute body-frame relative position of ownship from target:
    #   x_body = dx*cos(psi) + dy*sin(psi)   (along target's heading)
    #   y_body = -dx*sin(psi) + dy*cos(psi)  (starboard of target's heading)
    cos_psi, sin_psi = np.cos(s.psi), np.sin(s.psi)
    y_body = -dx * sin_psi + dy * cos_psi
    # Turn toward ownship: sign(y_body) gives the direction of the target turn
    # (positive = ownship to starboard, turn starboard = positive rudder).
    # Default to starboard turn when y_body=0 (head-on geometry) to maintain
    # worst-case coverage — the old np.sign(0)=0 degenerated to constant velocity.
    direction = np.sign(y_body) if abs(y_body) > 1e-6 else 1.0
    turn_rate = direction * np.deg2rad(turn_rate_deg)

    x, y = s.x, s.y
    psi = s.psi
    speed = s.speed()
    traj = []

    for k in range(horizon_steps):
        if k >= turn_delay_steps:
            psi_prev = psi
            psi = psi_prev + turn_rate * dt
            x = x + speed * np.cos((psi + psi_prev) / 2) * dt
            y = y + speed * np.sin((psi + psi_prev) / 2) * dt
        else:
            x = x + speed * np.cos(psi) * dt
            y = y + speed * np.sin(psi) * dt
        traj.append(np.array([x, y]))

    return traj


def _predict_emergency_evasion(
    target: TargetShip,
    horizon_steps: int,
    dt: float,
) -> list[np.ndarray]:
    """Predict target trajectory under emergency evasion behavior.

    Hard starboard turn with speed reduction, representing a target
    vessel taking aggressive last-second evasion. Low-probability but
    creates the closest possible approach — testing the ownship
    framework's safety margin against worst-case scenarios.

    Args:
        target: Target ship.
        horizon_steps: Prediction horizon steps.
        dt: Time step [s].

    Returns:
        List of (horizon_steps) positions.
    """
    s = target.state
    turn_delay_steps = max(1, int(5.0 / dt))   # 5 s reaction delay (emergency)
    turn_rate = np.deg2rad(3.0)                 # aggressive 3°/s turn
    speed_decay = 0.95                          # 5% speed reduction per step

    x, y = s.x, s.y
    psi = s.psi
    speed = s.speed()
    traj = []

    for k in range(horizon_steps):
        if k >= turn_delay_steps:
            psi_prev = psi
            psi = psi_prev + turn_rate * dt
            speed = speed * speed_decay
            x = x + speed * np.cos((psi + psi_prev) / 2) * dt
            y = y + speed * np.sin((psi + psi_prev) / 2) * dt
        else:
            x = x + speed * np.cos(psi) * dt
            y = y + speed * np.sin(psi) * dt
        traj.append(np.array([x, y]))

    return traj


def _predict_erratic(
    target: TargetShip,
    horizon_steps: int,
    dt: float,
    heading_std_deg: float = 3.0,
) -> list[np.ndarray]:
    """Predict target trajectory under erratic (unpredictable) behavior.

    Random walk in heading with Gaussian randomness. Models:
    - Human error (watchkeeper distraction, fatigue)
    - Mechanical failure (rudder jam, autopilot malfunction)
    - Miscommunication (incorrect VHF instruction)

    Uses a fixed random seed per target MMSI for deterministic
    reproducibility while maintaining realistic variability.

    Args:
        target: Target ship.
        horizon_steps: Prediction horizon steps.
        dt: Time step [s].
        heading_std_deg: Standard deviation of heading random walk [deg/step].

    Returns:
        List of (horizon_steps) positions.
    """
    s = target.state
    # Deterministic randomness: use MD5 for cross-session reproducibility.
    # Python's built-in hash() is salted by PYTHONHASHSEED (non-deterministic).
    rng_hash = int(hashlib.md5(str(target.mmsi).encode()).hexdigest()[:8], 16) % (2**31)
    rng = np.random.RandomState(rng_hash)

    heading_std = np.deg2rad(heading_std_deg) * np.sqrt(dt)

    x, y = s.x, s.y
    psi = s.psi
    speed = s.speed()
    traj = []

    for k in range(horizon_steps):
        # Random walk in heading
        dpsi = rng.normal(0.0, heading_std)
        psi = psi + dpsi
        x = x + speed * np.cos(psi) * dt
        y = y + speed * np.sin(psi) * dt
        traj.append(np.array([x, y]))

    return traj
