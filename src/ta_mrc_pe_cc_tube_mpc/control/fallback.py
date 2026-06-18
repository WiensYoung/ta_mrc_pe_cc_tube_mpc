"""Fallback strategy for controller failures.

Trigger conditions:
    1. MPC infeasible for M consecutive steps
    2. CBF correction exceeds threshold
    3. d_ij approaches emergency threshold
    4. AIS severe delay or dropout
    5. Near bank / non-navigable depth
    6. Runtime deadline miss

Fallback actions:
    1. Reduce speed
    2. Maintain safe heading
    3. Emergency starboard turn (if rules allow)
    4. Return to channel centerline
    5. Stop / low-speed wait
    6. Expand safety domain and uncertainty bounds
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from ..types import ControlCommand, EnvironmentState, TargetShip, VesselState
from ..utils.coordinates import distance


class FallbackLevel(Enum):
    """Fallback severity levels."""
    NONE = 0
    CAUTION = 1     # increased margins
    REDUCE_SPEED = 2
    STOP = 3        # minimum steerage
    EMERGENCY = 4   # full emergency maneuver


@dataclass
class FallbackState:
    """Tracks fallback state across control steps."""
    level: FallbackLevel = FallbackLevel.NONE
    consecutive_infeasible: int = 0
    consecutive_cbf_over_intervention: int = 0
    emergency_triggered: bool = False
    recovery_attempts: int = 0
    target_heading: Optional[float] = None
    target_speed_factor: float = 1.0
    safety_domain_expansion: float = 1.0


class FallbackController:
    """Fallback strategy manager.

    Maintains state across control steps and escalates/de-escalates
    fallback level based on current conditions.
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: Fallback configuration dict.
        """
        cfg = config or {}
        self.draught = cfg.get("draught", 2.0)
        self.vessel_beam = cfg.get("beam", 32.0)  # vessel beam [m] for bank threshold scaling
        self.vessel_length = cfg.get("length", 180.0)  # vessel length [m] for physical collision distance
        self.max_consecutive_infeasible = cfg.get("max_consecutive_infeasible", 5)
        self.max_cbf_correction_threshold = cfg.get("max_cbf_correction_threshold", 0.5)
        self.emergency_distance_factor = cfg.get("emergency_distance_factor", 0.5)
        self.slow_speed_factor = cfg.get("slow_speed_factor", 0.3)
        self.safety_course_change = cfg.get("safety_course_change", 10.0)  # [deg]
        self.ais_loss_timeout = cfg.get("ais_loss_timeout", 30.0)  # [s]
        self.runtime_deadline = cfg.get("runtime_deadline", 0.5)  # [s] — relaxed from 0.1
        self.extend_safety_domain_factor = cfg.get("extend_safety_domain_factor", 1.2)
        # L6 fix (2026-06-10): config-driven recovery parameters to prevent
        # hardcoded oscillation between fallback levels.
        self.recovery_steps_required = cfg.get("recovery_steps_required", 10)
        self.recovery_domain_decay = cfg.get("recovery_domain_decay", 0.9)

        self.state = FallbackState()

    def update(
        self,
        mpc_feasible: bool,
        cbf_info: dict,
        own_state: VesselState,
        targets: list[TargetShip],
        env: EnvironmentState,
        d_safe_eff_list: list[float],
        runtime_s: float = 0.0,
    ) -> ControlCommand:
        """Update fallback state and compute fallback control if needed.

        Args:
            mpc_feasible: Whether MPC returned a feasible solution.
            cbf_info: CBF diagnostic info from the safety filter.
            own_state: Current ownship state.
            targets: Target ships.
            env: Environment state.
            d_safe_eff_list: Effective safe distances [m].
            runtime_s: Computation time for this step [s].

        Returns:
            ControlCommand (may be fallback or nominal recovery).
        """
        # --- Update counters FIRST so _check_triggers sees the latest state ---
        if not mpc_feasible:
            self.state.consecutive_infeasible += 1
        else:
            self.state.consecutive_infeasible = 0

        if cbf_info.get("over_intervention", False):
            self.state.consecutive_cbf_over_intervention += 1
        else:
            self.state.consecutive_cbf_over_intervention = 0

        # --- Assess triggers (now sees updated counters) ---
        triggers = self._check_triggers(
            mpc_feasible, cbf_info, own_state, targets, env,
            d_safe_eff_list, runtime_s,
        )

        # --- Update state ---
        if triggers["emergency"]:
            self.state.level = FallbackLevel.EMERGENCY
            self.state.emergency_triggered = True
        elif triggers["stop"]:
            self.state.level = FallbackLevel.STOP
        elif triggers["reduce_speed"]:
            if self.state.level.value < FallbackLevel.REDUCE_SPEED.value:
                self.state.level = FallbackLevel.REDUCE_SPEED
        elif triggers["caution"]:
            if self.state.level == FallbackLevel.NONE:
                self.state.level = FallbackLevel.CAUTION
        elif self.state.level != FallbackLevel.NONE:
            # Recovery: gradually de-escalate through levels
            self.state.recovery_attempts += 1
            if self.state.recovery_attempts > self.recovery_steps_required:  # L6: config-driven
                levels = list(FallbackLevel)
                current_idx = levels.index(self.state.level) if self.state.level in levels else -1
                if current_idx > 0:
                    self.state.level = levels[current_idx - 1]
                    self.state.recovery_attempts = 0
                    self.state.safety_domain_expansion = max(
                        1.0, self.state.safety_domain_expansion * self.recovery_domain_decay,
                    )
                else:
                    self.state.level = FallbackLevel.NONE
                    self.state.recovery_attempts = 0
                    self.state.safety_domain_expansion = 1.0

        # --- Compute fallback action ---
        return self._compute_fallback_action(own_state, env, triggers)

    def _check_triggers(
        self,
        mpc_feasible: bool,
        cbf_info: dict,
        own_state: VesselState,
        targets: list[TargetShip],
        env: EnvironmentState,
        d_safe_eff_list: list[float],
        runtime_s: float,
    ) -> dict:
        """Evaluate all fallback triggers."""
        triggers = {
            "emergency": False,
            "stop": False,
            "reduce_speed": False,
            "caution": False,
            "reason": "",
        }

        # MPC infeasible too many times
        if self.state.consecutive_infeasible >= self.max_consecutive_infeasible:
            triggers["reduce_speed"] = True
            triggers["reason"] += "mpc_infeasible; "

        # CBF over-intervention
        if self.state.consecutive_cbf_over_intervention >= 3:
            triggers["reduce_speed"] = True
            triggers["reason"] += "cbf_over_intervention; "

        # Critical distance to any target.
        # IMPORTANT: Use PHYSICAL collision distance (not inflated d_safe) for
        # emergency triggers.  The inflated d_safe (which includes domain margins,
        # rule scaling, tube radius etc.) can reach 1440m for 180m vessels, causing
        # emergency mode to trigger at 720m — far beyond any real danger.  Physical
        # collision distance = 0.5 × (L_i + L_j) ≈ 180m for typical encounters.
        own_pos = own_state.position()
        for i, target in enumerate(targets):
            d_ij = distance(own_pos, target.state.position())
            L_j = getattr(target, 'length', 180.0) or 180.0
            physical_collision_d = 0.5 * (self.vessel_length + L_j)
            # Emergency thresholds based on physical distance, not inflated domain
            emergency_d = max(physical_collision_d * 2.0, 100.0)  # at least 100m
            critical_d = physical_collision_d * 1.0  # hull-to-hull
            if d_ij < critical_d:
                triggers["emergency"] = True
                triggers["reason"] += f"critical_distance_{target.mmsi}; "
            elif d_ij < emergency_d:
                triggers["stop"] = True
                triggers["reason"] += f"emergency_distance_{target.mmsi}; "

        # Bank clearance critical — scale threshold with vessel beam
        bank_threshold = max(20.0, self.vessel_beam * 0.5)
        if env.bank_distance_left is not None and env.bank_distance_left < bank_threshold:
            triggers["emergency"] = True
            triggers["reason"] += "critical_bank_left; "
        if env.bank_distance_right is not None and env.bank_distance_right < bank_threshold:
            triggers["emergency"] = True
            triggers["reason"] += "critical_bank_right; "

        # UKC violation
        if env.water_depth is None:
            ukc = float("inf")  # no depth info → skip UKC check
        else:
            ukc = env.water_depth - self.draught
        if ukc < 0.5:
            triggers["emergency"] = True
            triggers["reason"] += "critical_ukc; "
        elif ukc < 1.5:
            triggers["caution"] = True
            triggers["reason"] += "low_ukc; "

        # AIS severe delay
        for target in targets:
            delay = target.ais_quality.get("delay", 0.0)
            if delay > self.ais_loss_timeout:
                triggers["caution"] = True
                triggers["reason"] += f"ais_delay_{target.mmsi}; "

        # Runtime deadline miss
        if runtime_s > self.runtime_deadline:
            triggers["caution"] = True
            triggers["reason"] += "runtime_deadline; "

        return triggers

    def _compute_fallback_action(
        self,
        own_state: VesselState,
        env: EnvironmentState,
        triggers: dict,
    ) -> ControlCommand:
        """Compute the fallback control action based on current level."""
        level = self.state.level

        if level == FallbackLevel.NONE:
            # Return to normal: no fallback needed
            return ControlCommand(
                rudder=0.0, propeller=0.5, source="fallback",
                fallback_active=False,
                metadata={"fallback_level": "none"},
            )

        rudder = 0.0
        propeller = 0.5

        if level == FallbackLevel.CAUTION:
            # Maintain at least minimum steerage speed (0.15 may be too low for large vessels)
            propeller = max(0.5 * self.slow_speed_factor, 0.25)
            # Use SET (not multiply) to prevent compounding across steps
            self.state.safety_domain_expansion = min(
                self.extend_safety_domain_factor,
                1.2,
            )

        elif level == FallbackLevel.REDUCE_SPEED:
            propeller = max(0.3 * self.slow_speed_factor, 0.15)  # minimum steerage
            # Maintain heading toward channel center
            if env.channel_width is not None and abs(own_state.y) > env.channel_width * 0.3:
                rudder = -np.sign(own_state.y) * 5.0  # steer toward center
            self.state.safety_domain_expansion = min(
                self.extend_safety_domain_factor,
                1.3,
            )

        elif level == FallbackLevel.STOP:
            propeller = 0.05  # minimum steerage
            # Emergency starboard turn if possible
            rudder = self.safety_course_change  # starboard
            self.state.safety_domain_expansion = min(
                self.extend_safety_domain_factor * 1.2,
                1.5,
            )

        elif level == FallbackLevel.EMERGENCY:
            propeller = 0.0  # stop
            rudder = self.safety_course_change  # starboard turn
            self.state.safety_domain_expansion = min(
                self.extend_safety_domain_factor * 1.5,
                2.0,
            )

        return ControlCommand(
            rudder=rudder,
            propeller=propeller,
            source="fallback",
            fallback_active=True,
            metadata={
                "fallback_level": level.name,
                "triggers": triggers,
                "safety_domain_expansion": self.state.safety_domain_expansion,
            },
        )

    def reset(self):
        """Reset fallback state."""
        self.state = FallbackState()

    def is_active(self) -> bool:
        return self.state.level != FallbackLevel.NONE
