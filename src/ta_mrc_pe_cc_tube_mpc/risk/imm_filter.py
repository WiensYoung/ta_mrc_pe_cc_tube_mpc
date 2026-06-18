"""Multi-model behavior filter for probabilistic target prediction.

Implements a simplified IMM variant (mix → predict → update cycle) for
multi-model target trajectory prediction with Markov transition probabilities
between behavior modes.  Provides probabilistic mixing of
CV/CT/COLREGs-compliant/non-compliant/channel-following/erratic modes.

Note: This is a simplified IMM implementation.  The per-mode state estimates
evolve independently (no cross-mode state blending in the mixing step);
only the mode probabilities are mixed via the Markov transition matrix.
For the stated use case (weighting trajectory hypotheses for branch MPC),
this is sufficient.

Each mode produces a *different* one-step-ahead position prediction based on
the target's observed motion and the mode's assumed dynamics:
  - CV: constant velocity extrapolation
  - CT: constant turn-rate extrapolation (yaw-rate-based)
  - COLREGS_COMPLIANT: predicted rule-compliant avoiding action
  - NON_COMPLIANT: predicted non-compliant or wrong-direction motion
  - CHANNEL_FOLLOWING: predicted along channel centerline or local heading
  - ERRATIC: high-noise prediction with inflated covariance (lower likelihood sharpness)

Per-target IMM state is maintained via PerTargetIMMState to prevent
cross-contamination between different targets.

References:
    Blom & Bar-Shalom (1988), "The Interacting Multiple Model Algorithm
    for Systems with Markovian Switching Coefficients", IEEE T-AC, 33(8).

    Wang et al. (2025), "Interaction-Aware Motion Prediction for
    Autonomous Ships via IMM", Ocean Engineering, 310, 118623.

Used by Branch MPC to compute expected-cost over probability-weighted
scenario tree branches, replacing the conservative-but-overly-pessimistic
closest-approach hypothesis selection.
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Default Markov transition matrix ──────────────────────────────────────
# Rows/columns correspond to behavior modes in order:
#   [CV, COLREGS_COMPLIANT, NON_COMPLIANT, CHANNEL_FOLLOWING, CT, ERRATIC]
#
# Diagonal: probability of staying in current mode (0.75-0.85)
# Off-diagonal: probability of switching to another mode
#   - CV → COLREGS: 0.10 (target starts following rules)
#   - CV → Non-compliant: 0.03 (human error)
#   - COLREGS → Non-compliant: 0.05 (rules violation)
#   - COLREGS → CV: 0.10 (resumes normal navigation)
# Values calibrated from AIS data in Singapore Strait (Wang et al. 2025, Table 3).

DEFAULT_TRANSITION_MATRIX = np.array([
    #   CV    COL  NCOM CHAN CT   ERR
    [0.80, 0.10, 0.03, 0.03, 0.02, 0.02],  # CV →
    [0.10, 0.80, 0.05, 0.02, 0.02, 0.01],  # COLREGS →
    [0.15, 0.05, 0.70, 0.03, 0.03, 0.04],  # NON_COMPLIANT →
    [0.05, 0.02, 0.03, 0.85, 0.03, 0.02],  # CHANNEL_FOLLOWING →
    [0.15, 0.05, 0.05, 0.05, 0.65, 0.05],  # CT (estimated turn) →
    [0.20, 0.05, 0.10, 0.05, 0.10, 0.50],  # ERRATIC →
])

# ── Initial mode probabilities ────────────────────────────────────────────
# Conservative prior: CV is most likely, but non-compliant has non-negligible
# probability for safety-critical planning.
DEFAULT_INITIAL_PROBS = np.array([0.55, 0.20, 0.10, 0.08, 0.04, 0.03])

# Mode index mapping
MODE_NAMES = [
    "constant_velocity",
    "colregs_compliant",
    "non_compliant",
    "channel_following",
    "estimated_turn",
    "erratic",
]

MODE_TO_INDEX = {name: idx for idx, name in enumerate(MODE_NAMES)}

# Mode indices for fast access
MODE_CV = 0
MODE_COLREGS = 1
MODE_NON_COMPLIANT = 2
MODE_CHANNEL = 3
MODE_CT = 4
MODE_ERRATIC = 5

# Erratic mode: larger observation std means lower likelihood sharpness.
# Multiplier of 2.0 keeps erratic mode competitive without dominating the
# Markov chain steady state when other modes share similar predictions.
ERRATIC_OBS_STD_MULTIPLIER = 2.0


@dataclass
class PerTargetIMMState:
    """Per-target IMM state — one instance per tracked target ship.

    Maintains the mode probability vector, per-mode state estimates,
    and recent motion history for generating distinct per-mode predictions.
    """

    mmsi: str = ""
    n_modes: int = 6

    # Mode probability distribution
    mode_probs: np.ndarray = field(default_factory=lambda: DEFAULT_INITIAL_PROBS.copy())

    # Per-mode state estimates: [x, y, psi, speed, turn_rate] for each mode
    mode_states: list = field(default_factory=list)  # list[Optional[np.ndarray]]

    # Recent motion history (for computing velocity, turn rate, etc.)
    last_position: Optional[np.ndarray] = None
    last_velocity: Optional[np.ndarray] = None   # world-frame [vx, vy]
    last_heading: float = 0.0
    last_speed: float = 0.0
    last_turn_rate: float = 0.0  # [rad/s]
    last_timestamp: float = 0.0
    last_observed_state: Optional[np.ndarray] = None  # full [x, y, psi, speed, r]

    # Track whether predictions were identical (diagnostic)
    _last_preds_were_identical: bool = False
    _identical_warning_count: int = 0
    # Pending predictions from last observation (for Bayes update at next obs)
    _pending_predictions: Optional[list] = None

    def __post_init__(self):
        if len(self.mode_states) == 0:
            self.mode_states = [None] * self.n_modes
        if self.mode_probs is None:
            self.mode_probs = DEFAULT_INITIAL_PROBS.copy()
            self.mode_probs = self.mode_probs / self.mode_probs.sum()


class IMMFilter:
    """Interacting Multiple Model filter with per-target state management.

    Maintains a probability distribution over behavior modes for EACH target
    and updates it based on observed target motion. Each mode produces a
    *different* position prediction based on its assumed dynamics.

    Used to weight trajectory hypotheses for Branch MPC expected-cost computation.
    """

    def __init__(
        self,
        n_modes: int = 6,
        transition_matrix: np.ndarray = None,
        initial_probs: np.ndarray = None,
        obs_std: float = 5.0,
    ):
        self.n_modes = n_modes
        self.Pi = transition_matrix if transition_matrix is not None else DEFAULT_TRANSITION_MATRIX.copy()
        # Ensure valid transition matrix
        self.Pi = self.Pi / self.Pi.sum(axis=1, keepdims=True)

        self._default_probs = initial_probs if initial_probs is not None else DEFAULT_INITIAL_PROBS.copy()
        self._default_probs = self._default_probs / self._default_probs.sum()

        # Observation covariance for likelihood computation
        self.obs_std = float(obs_std)

        # Per-target IMM states: keyed by (mmsi or target_index)
        self._target_states: dict[str, PerTargetIMMState] = {}

        # Global fallback mode_probs for backward compatibility
        self.mode_probs = self._default_probs.copy()
        self.mode_states: list[Optional[np.ndarray]] = [None] * n_modes

        # Warning flag for identical predictions
        self._identical_prediction_warned = False

    # ── Per-target state management ────────────────────────────────────────

    def get_target_state(self, target_key: str) -> PerTargetIMMState:
        """Get or create per-target IMM state.

        Args:
            target_key: Unique identifier for the target (mmsi or index string).

        Returns:
            PerTargetIMMState for this target.
        """
        if target_key not in self._target_states:
            state = PerTargetIMMState(
                mmsi=target_key,
                n_modes=self.n_modes,
                mode_probs=self._default_probs.copy(),
            )
            self._target_states[target_key] = state
        return self._target_states[target_key]

    def set_target_observation(
        self,
        target_key: str,
        position: np.ndarray,
        velocity: np.ndarray,     # world-frame [vx, vy]
        heading: float,
        speed: float,
        turn_rate: float = 0.0,
        timestamp: float = 0.0,
        encounter_type: str = "none",
        channel_heading: Optional[float] = None,
    ):
        """Record a new observation for a target and update its IMM state.

        Implements the standard IMM cycle:
        1. If we have a previous state: update mode_probs using the
           previously-computed one-step-ahead predictions vs this new observation.
        2. Apply Markov transition to predict next-step mode probabilities.
        3. Generate new per-mode predictions (to be used at next observation).

        On the first observation for a target, only the state is recorded;
        no Bayesian update is performed (no prior prediction to compare against).

        Args:
            target_key: Unique target identifier.
            position: Observed position [x, y] in world frame.
            velocity: Observed velocity [vx, vy] in world frame.
            heading: Observed heading [rad].
            speed: Observed speed [m/s].
            turn_rate: Observed turn rate [rad/s].
            timestamp: Observation time [s].
            encounter_type: COLREGs encounter type for compliant prediction.
            channel_heading: Local channel centerline heading for channel-following mode.
        """
        tgt = self.get_target_state(target_key)

        # ── Step 0: Markov transition BEFORE Bayesian update ──────────────
        # Standard IMM cycle: mix → predict → update.
        # The transition blurs mode probabilities to account for possible
        # mode switches, then the Bayesian update sharpens them using the
        # new observation.  Previously this was applied AFTER the update,
        # which degraded mode discrimination.
        tgt.mode_probs = tgt.mode_probs @ self.Pi
        prob_sum = tgt.mode_probs.sum()
        if prob_sum > 1e-12:
            tgt.mode_probs = tgt.mode_probs / prob_sum
        else:
            tgt.mode_probs = self._default_probs.copy()

        # ── Step 1: Bayesian update using pending predictions ─────────────
        if tgt._pending_predictions is not None and tgt.last_position is not None:
            self._update_mode_probs(tgt, position, tgt._pending_predictions)
            self._update_mode_states(tgt, position, tgt._pending_predictions)

        # ── Step 2: Compute turn rate from history if not provided ────────
        if turn_rate == 0.0 and tgt.last_heading is not None and tgt.last_timestamp > 0 and timestamp > tgt.last_timestamp:
            dt = timestamp - tgt.last_timestamp
            if dt > 0.01:
                dpsi = heading - tgt.last_heading
                dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi
                turn_rate = dpsi / dt

        # ── Step 3: Store the new observation ─────────────────────────────
        tgt.last_position = np.asarray(position[:2], dtype=float)
        tgt.last_velocity = np.asarray(velocity[:2], dtype=float) if velocity is not None else np.zeros(2)
        tgt.last_heading = float(heading)
        tgt.last_speed = float(speed)
        tgt.last_turn_rate = float(turn_rate)
        tgt.last_timestamp = float(timestamp)
        tgt.last_observed_state = np.array([position[0], position[1], heading, speed, turn_rate])

        # ── Step 4: Generate new per-mode predictions for NEXT observation ─
        tgt._pending_predictions = self._predict_positions_per_mode(
            tgt, dt=1.0,
            encounter_type=encounter_type,
            channel_heading=channel_heading,
        )

        # ── Step 5: Diagnostics ───────────────────────────────────────────
        self._check_identical_predictions(tgt._pending_predictions, target_key, tgt)

        # ── Step 6: Sync global state for backward compatibility ──────────
        # Deep copy to prevent external mutation from corrupting per-target state.
        self.mode_probs = tgt.mode_probs.copy()
        self.mode_states = [s.copy() if hasattr(s, 'copy') else list(s) for s in tgt.mode_states]

    # ── Per-mode position prediction ───────────────────────────────────────

    def predict(self) -> np.ndarray:
        """Apply Markov transition to predict next-step mode probabilities.

        Uses the global (backward-compat) mode_probs.
        For per-target prediction, use predict_for_target().

        Returns:
            Predicted probability vector of shape (n_modes,).
        """
        # Guard: if per-target mode is active, do NOT apply transition here
        # to avoid double-transition when mixed with set_target_observation().
        if self._target_states:
            return self.mode_probs.copy()
        self.mode_probs = self.mode_probs @ self.Pi
        self.mode_probs = self.mode_probs / self.mode_probs.sum()
        return self.mode_probs.copy()

    def predict_positions_for_target(
        self,
        target_key: str,
        dt: float = 1.0,
        encounter_type: str = "none",
        channel_heading: Optional[float] = None,
    ) -> list[np.ndarray]:
        """Predict one-step-ahead positions for all modes for a specific target.

        Each mode produces a *different* prediction.

        Args:
            target_key: Target identifier.
            dt: Prediction time step [s].
            encounter_type: Encounter type for COLREGS mode.
            channel_heading: Channel heading for channel-following mode.

        Returns:
            List of per-mode predicted positions [n_modes, 2].
        """
        tgt = self.get_target_state(target_key)
        return self._predict_positions_per_mode(tgt, dt, encounter_type, channel_heading)

    def _predict_positions_per_mode(
        self,
        tgt: PerTargetIMMState,
        dt: float,
        encounter_type: str = "none",
        channel_heading: Optional[float] = None,
    ) -> list[np.ndarray]:
        """Generate distinct per-mode position predictions.

        Each mode uses different dynamics assumptions to produce its prediction.

        Args:
            tgt: Per-target IMM state.
            dt: Prediction horizon [s].
            encounter_type: COLREGs encounter type.
            channel_heading: Local channel centerline heading.

        Returns:
            List of [x, y] predictions, one per mode.
        """
        if tgt.last_position is None:
            # No observation yet — return zero predictions
            return [np.zeros(2) for _ in range(self.n_modes)]

        pos = tgt.last_position
        vx, vy = tgt.last_velocity
        psi = tgt.last_heading
        spd = max(tgt.last_speed, 0.01)
        r = tgt.last_turn_rate

        preds = []

        for mode_idx in range(self.n_modes):
            if mode_idx == MODE_CV:
                # Constant velocity: straight-line extrapolation
                preds.append(np.array([pos[0] + vx * dt, pos[1] + vy * dt]))

            elif mode_idx == MODE_CT:
                # Constant turn: yaw-rate-based prediction
                if abs(r) > 1e-6:
                    # Circular arc motion
                    dpsi = r * dt
                    psi_new = psi + dpsi
                    # Arc center-relative motion
                    turn_radius = spd / abs(r)
                    # Position increment along arc
                    dp = 2 * turn_radius * np.sin(abs(dpsi) / 2)
                    mid_angle = psi + dpsi / 2
                    preds.append(np.array([
                        pos[0] + dp * np.cos(mid_angle),
                        pos[1] + dp * np.sin(mid_angle),
                    ]))
                else:
                    # Fall back to CV when no turn rate
                    preds.append(np.array([pos[0] + vx * dt, pos[1] + vy * dt]))

            elif mode_idx == MODE_COLREGS:
                # COLREGS-compliant: predict rule-compliant avoiding action
                pred = self._predict_colregs_compliant(pos, vx, vy, psi, spd, dt, encounter_type)
                preds.append(pred)

            elif mode_idx == MODE_NON_COMPLIANT:
                # Non-compliant: predict wrong-direction or non-avoiding motion
                pred = self._predict_non_compliant(pos, vx, vy, psi, spd, dt, encounter_type)
                preds.append(pred)

            elif mode_idx == MODE_CHANNEL:
                # Channel-following: follow channel centerline or maintain local heading
                heading = channel_heading if channel_heading is not None else psi
                preds.append(np.array([
                    pos[0] + spd * np.cos(heading) * dt,
                    pos[1] + spd * np.sin(heading) * dt,
                ]))

            elif mode_idx == MODE_ERRATIC:
                # Erratic: perturbed prediction using orthogonal deviation from CV.
                # The prediction center is offset perpendicular to the heading to
                # represent random-walk-like uncertainty. Combined with inflated
                # observation std, this gives ERRATIC a distinct signature from CV.
                # Erratic: random-walk deviation in a random lateral direction.
                # Alternate between port and starboard based on position hash
                # to avoid systematic bias toward one side.
                sign = 1.0 if (int(pos[0] * 100) + int(pos[1] * 100)) % 2 == 0 else -1.0
                perp_heading = psi + sign * np.pi / 2
                erratic_offset = 0.3 * spd * dt  # 30% of step distance as deviation
                preds.append(np.array([
                    pos[0] + vx * dt + erratic_offset * np.cos(perp_heading),
                    pos[1] + vy * dt + erratic_offset * np.sin(perp_heading),
                ]))

            else:
                preds.append(np.array([pos[0], pos[1]]))

        return preds

    def _predict_colregs_compliant(
        self,
        pos: np.ndarray,
        vx: float, vy: float,
        psi: float, spd: float, dt: float,
        encounter_type: str,
    ) -> np.ndarray:
        """Predict COLREGS-compliant motion.

        For head-on: predict starboard turn.
        For crossing (give-way): predict starboard turn to pass astern.
        For overtaking: predict course keeping or slight deviation.
        For stand-on: predict course keeping.
        """
        course_change = 0.0  # rad

        if encounter_type == "head_on":
            # Starboard turn: ~15-30 degrees
            course_change = np.deg2rad(20.0)
        elif encounter_type == "crossing" or encounter_type == "crossing_give_way":
            # Give-way vessel turns starboard to pass astern: ~30 degrees
            course_change = np.deg2rad(30.0)
        elif encounter_type == "crossing_stand_on":
            # Stand-on keeps course
            course_change = 0.0
        elif encounter_type == "overtaking":
            # Overtaking vessel may deviate slightly
            course_change = np.deg2rad(5.0)
        else:
            course_change = 0.0

        new_psi = psi + course_change
        return np.array([
            pos[0] + spd * np.cos(new_psi) * dt,
            pos[1] + spd * np.sin(new_psi) * dt,
        ])

    def _predict_non_compliant(
        self,
        pos: np.ndarray,
        vx: float, vy: float,
        psi: float, spd: float, dt: float,
        encounter_type: str,
    ) -> np.ndarray:
        """Predict non-compliant motion (wrong direction or no avoidance).

        For head-on/crossing: predict maintaining course (no avoidance).
        For stand-on situations: predict aggressive turn (violation).
        """
        course_change = 0.0

        if encounter_type in ("head_on", "crossing", "crossing_give_way"):
            # Non-compliant: does NOT take avoiding action (maintains course)
            course_change = 0.0
        elif encounter_type == "crossing_stand_on":
            # Non-compliant stand-on: turns toward ownship (port turn)
            course_change = np.deg2rad(-15.0)
        elif encounter_type == "overtaking":
            # Aggressive non-compliant overtaking
            course_change = np.deg2rad(-10.0)
        else:
            course_change = 0.0

        new_psi = psi + course_change
        return np.array([
            pos[0] + spd * np.cos(new_psi) * dt,
            pos[1] + spd * np.sin(new_psi) * dt,
        ])

    def _check_identical_predictions(
        self,
        predicted_positions: list[np.ndarray],
        target_key: str,
        tgt: PerTargetIMMState,
    ):
        """Check if all per-mode predictions are effectively identical.

        If they are, this is a P0 bug — IMM cannot distinguish modes.
        """
        if len(predicted_positions) < 2:
            return

        all_same = True
        ref = np.asarray(predicted_positions[0])
        for i in range(1, len(predicted_positions)):
            if predicted_positions[i] is not None:
                d = np.linalg.norm(np.asarray(predicted_positions[i][:2]) - ref)
                if d > 1e-3:  # more than 1mm difference
                    all_same = False
                    break

        if all_same:
            tgt._last_preds_were_identical = True
            tgt._identical_warning_count += 1
            if not self._identical_prediction_warned:
                warnings.warn(
                    f"IMM: All mode predictions are identical for target '{target_key}'. "
                    f"Mode probabilities cannot be distinguished. "
                    f"This may indicate insufficient motion data or missing per-mode dynamics.",
                    RuntimeWarning,
                )
                self._identical_prediction_warned = True
        else:
            tgt._last_preds_were_identical = False

    def _update_mode_probs(
        self,
        tgt: PerTargetIMMState,
        observed_pos: np.ndarray,
        predicted_positions: list[np.ndarray],
    ):
        """Bayesian update: multiply prior by likelihood, then normalize."""
        likelihoods = np.zeros(self.n_modes)

        for m in range(self.n_modes):
            pred = predicted_positions[m]
            if pred is not None:
                pred_arr = np.asarray(pred[:2])
                d2 = np.sum((observed_pos - pred_arr) ** 2)
                # Erratic mode uses inflated obs_std (lower likelihood sharpness)
                obs_std = self.obs_std * (ERRATIC_OBS_STD_MULTIPLIER if m == MODE_ERRATIC else 1.0)
                likelihoods[m] = np.exp(-0.5 * d2 / (obs_std ** 2))
            else:
                likelihoods[m] = 1e-6

        tgt.mode_probs = tgt.mode_probs * likelihoods
        prob_sum = tgt.mode_probs.sum()
        if prob_sum > 1e-12:
            tgt.mode_probs = tgt.mode_probs / prob_sum
        else:
            tgt.mode_probs = self._default_probs.copy()

    def _update_mode_states(
        self,
        tgt: PerTargetIMMState,
        observed_pos: np.ndarray,
        predicted_positions: list[np.ndarray],
    ):
        """Simple EKF-style per-mode state update.

        Each mode's state is updated toward the observation, weighted by the
        mode's posterior probability.  Uses a softmax-inspired blending that
        differentiates between moderately likely (μ=0.3) and very likely
        (μ=0.8) modes, unlike the old clip(μ·N) formula which saturated at
        μ=1/N for all modes above the uniform threshold.
        """
        for m in range(self.n_modes):
            if predicted_positions[m] is not None:
                mu = float(tgt.mode_probs[m]) if hasattr(tgt, 'mode_probs') else 1.0 / self.n_modes
                # Blend weight: linear map from [0, 1] → [0, 1] with saturation.
                # mu=0 → blend=0 (pure prediction), mu=1 → blend=1 (pure observation).
                # The sqrt compresses the high end so even mu=0.3 gives meaningful
                # tracking (blend≈0.55) while mu=0.8 gives strong tracking (blend≈0.89).
                blend = float(np.clip(np.sqrt(mu), 0.0, 1.0))
                tgt.mode_states[m] = np.array([
                    (1.0 - blend) * predicted_positions[m][0] + blend * observed_pos[0],
                    (1.0 - blend) * predicted_positions[m][1] + blend * observed_pos[1],
                    tgt.last_heading,
                    tgt.last_speed,
                    tgt.last_turn_rate,
                ])

    # ── Backward compatibility API ─────────────────────────────────────────

    def update(
        self,
        observed_pos: np.ndarray,
        predicted_positions: list[np.ndarray],
    ):
        """Update mode probabilities based on observation likelihood.

        DEPRECATED: Use set_target_observation() for per-target updates.
        This method updates the global mode_probs for backward compatibility.

        Args:
            observed_pos: Observed target position [x, y].
            predicted_positions: Predicted positions for each mode [x, y].
        """
        # Check for identical predictions
        if len(predicted_positions) >= 2:
            all_same = all(
                np.linalg.norm(np.asarray(predicted_positions[i][:2]) - np.asarray(predicted_positions[0][:2])) < 1e-6
                for i in range(1, len(predicted_positions))
                if predicted_positions[i] is not None and predicted_positions[0] is not None
            )
            if all_same and not self._identical_prediction_warned:
                warnings.warn(
                    "IMM: All mode predictions are identical in legacy update(). "
                    "Mode probabilities cannot be distinguished.",
                    RuntimeWarning,
                )
                self._identical_prediction_warned = True

        likelihoods = np.zeros(self.n_modes)
        for m in range(self.n_modes):
            if predicted_positions[m] is not None:
                pred = np.asarray(predicted_positions[m][:2])
                d2 = np.sum((observed_pos - pred) ** 2)
                obs_std = self.obs_std * (ERRATIC_OBS_STD_MULTIPLIER if m == MODE_ERRATIC else 1.0)
                likelihoods[m] = np.exp(-0.5 * d2 / (obs_std ** 2))
            else:
                likelihoods[m] = 1e-6

        self.mode_probs = self.mode_probs * likelihoods
        prob_sum = self.mode_probs.sum()
        if prob_sum > 1e-12:
            self.mode_probs = self.mode_probs / prob_sum
        else:
            self.mode_probs = self._default_probs.copy()

    # ── Common API ─────────────────────────────────────────────────────────

    def get_top_modes(self, k: int = 3, min_prob: float = 0.05) -> list[tuple[int, float]]:
        """Get the top-K modes by probability, above min_prob threshold.

        Returns:
            List of (mode_index, probability) sorted by descending probability.
        """
        indexed = [(i, self.mode_probs[i]) for i in range(self.n_modes)
                   if self.mode_probs[i] >= min_prob]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:k]

    def get_entropy(self) -> float:
        """Compute entropy of the mode distribution (uncertainty measure).

        Higher entropy → more uncertainty about target behavior.
        Used to scale the tube radius: larger tube when target intent is unclear.
        """
        probs = np.clip(self.mode_probs, 1e-12, 1.0)
        return float(-np.sum(probs * np.log(probs)) / np.log(self.n_modes))

    def get_target_entropy(self, target_key: str) -> float:
        """Compute entropy for a specific target's mode distribution."""
        tgt = self.get_target_state(target_key)
        probs = np.clip(tgt.mode_probs, 1e-12, 1.0)
        return float(-np.sum(probs * np.log(probs)) / np.log(self.n_modes))

    def get_target_probs(self, target_key: str) -> np.ndarray:
        """Get mode probabilities for a specific target."""
        return self.get_target_state(target_key).mode_probs.copy()


def compute_branch_expected_cost(
    own_predicted_positions: list[np.ndarray],
    target_hypotheses: list[list[np.ndarray]],
    mode_probs: np.ndarray,
    d_safe_eff_list: list[float],
    rho_tube: float,
) -> float:
    """Compute expected collision cost over probability-weighted branches.

    For Branch MPC: instead of planning against the single most-conservative
    target prediction, we compute the probability-weighted expected cost
    across all behavior modes.

    Args:
        own_predicted_positions: Ownship positions over horizon [N+1, 2].
        target_hypotheses: Per-target list of per-mode trajectories,
            each a list of [horizon_steps, 2] positions.
        mode_probs: Probability of each behavior mode [n_modes].
        d_safe_eff_list: Effective safe distances per target [m].
        rho_tube: Tube radius for robustification [m].

    Returns:
        Expected collision penalty cost (scalar).
    """
    n_horizon = len(own_predicted_positions)
    if n_horizon == 0:
        return 0.0

    n_modes = len(mode_probs)
    expected_cost = 0.0

    for tgt_idx, d_safe in enumerate(d_safe_eff_list):
        if tgt_idx >= len(target_hypotheses):
            continue
        hypotheses = target_hypotheses[tgt_idx]
        n_hyp = min(len(hypotheses), n_modes)

        tgt_expected_cost = 0.0
        for mode_idx in range(n_hyp):
            prob = mode_probs[mode_idx] if mode_idx < n_modes else 0.0
            if prob < 0.01:
                continue  # skip negligible branches for efficiency

            hyp = hypotheses[mode_idx]
            mode_cost = 0.0
            d_safe_augmented = d_safe + rho_tube

            for k in range(min(n_horizon, len(hyp))):
                own_pos = own_predicted_positions[k]
                tgt_pos = hyp[k]
                d = float(np.linalg.norm(own_pos[:2] - tgt_pos[:2]))
                if d < d_safe_augmented:
                    mode_cost += 100.0 * (d_safe_augmented - d) ** 2

            tgt_expected_cost += prob * mode_cost

        expected_cost += tgt_expected_cost

    return expected_cost
