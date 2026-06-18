"""B2 baseline: Rule-based Dynamic Window Approach (DWA).

Local search-based method: evaluates a discrete set of (rudder, propeller)
commands in a dynamic window, scoring each by a cost function that combines
COLREGs compliance, collision risk, and trajectory tracking.
"""

from typing import Optional

import numpy as np

from ..types import ControlCommand, EnvironmentState, TargetShip, VesselParams, VesselState
from ..rules.colregs import colregs_action
from ..risk.cpa import compute_tcpa_dcpa
from ..utils.coordinates import body_to_world


class DynamicWindowController:
    """Rule-based Dynamic Window Approach.

    B2 baseline from experiment design Section 13.1.
    Evaluates candidate control actions in a dynamic window against
    a multi-objective cost function with COLREGs rule terms.
    """

    def __init__(self, vessel_params: VesselParams, config: dict = None):
        self.vessel_params = vessel_params
        cfg = config or {}
        controller_cfg = cfg.get("controller", cfg)
        self.predict_time = controller_cfg.get("dwa_predict_time", 60.0)
        self.dt = controller_cfg.get("dt", 0.5)
        self.n_samples = controller_cfg.get("dwa_samples", 200)

        # Search bounds
        self.rudder_range = (-vessel_params.max_rudder, vessel_params.max_rudder)
        self.prop_range = (0.0, 1.0)
        self.n_rudder = controller_cfg.get("dwa_n_rudder", 21)  # ~3.5 deg resolution
        self.n_prop = controller_cfg.get("dwa_n_prop", 5)

        # Cost weights
        self.w_colregs = controller_cfg.get("dwa_w_colregs", 3.0)
        self.w_collision = controller_cfg.get("dwa_w_collision", 5.0)
        self.w_track = controller_cfg.get("dwa_w_track", 1.0)
        self.w_smooth = controller_cfg.get("dwa_w_smooth", 0.5)
        self.w_speed = controller_cfg.get("dwa_w_speed", 0.2)

        self.safety_radius = controller_cfg.get("dwa_safety_radius", 500.0)
        self.yaw_gain = controller_cfg.get("dwa_yaw_gain", 0.02)  # rad/s per deg rudder
        self.previous_rudder = 0.0
        self.previous_propeller = 0.5
        self._step_counter = 0  # for indexing into reference_path during trajectory tracking

        # Derive unsupported features from the controller's capability set.
        # DWA supports: dynamic feasibility search + simple obstacle avoidance.
        _all_features = {
            "multi_rule", "target_size", "speed_sensitive_domain",
            "vessel_type_weight", "behavior_uncertainty", "shallow_current",
            "bank_effect", "ship_interaction", "chance_constraint",
            "tube_mpc", "cbf_filter", "fallback",
            "enc_navigability", "hard_constraints", "uncertainty_ais",
        }
        _supported = {"dynamic_feasibility", "obstacle_avoidance"}
        self.unsupported_features = sorted(_all_features - _supported)

    def compute_control(
        self,
        ownstate: VesselState,
        ownship_params: VesselParams,
        target_ships: list[TargetShip],
        environment: EnvironmentState,
        rule_context: dict = None,
        reference_path: list[VesselState] = None,
    ) -> ControlCommand:
        """Compute control using Dynamic Window Approach."""
        # Generate candidate actions
        rudder_candidates = np.linspace(
            self.rudder_range[0], self.rudder_range[1], self.n_rudder
        )
        prop_candidates = np.linspace(
            self.prop_range[0], self.prop_range[1], self.n_prop
        )

        best_cost = float("inf")
        best_rudder = 0.0
        best_prop = 0.5

        for rudder in rudder_candidates:
            for prop in prop_candidates:
                cost = self._evaluate_action(
                    rudder, prop, ownstate, ownship_params,
                    target_ships, environment, reference_path,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_rudder = rudder
                    best_prop = prop

        # Rudder rate limit (max_rudder_rate is deg/s, scale by dt)
        max_step = self.vessel_params.max_rudder_rate * self.dt
        rudder_rate = abs(best_rudder - self.previous_rudder)
        if rudder_rate > max_step:
            best_rudder = self.previous_rudder + np.sign(
                best_rudder - self.previous_rudder
            ) * max_step

        self.previous_rudder = best_rudder
        self.previous_propeller = best_prop
        self._step_counter += 1  # advance to next reference waypoint

        # Determine encounter type for metadata
        encounter_type = "none"
        if target_ships:
            action = colregs_action(ownstate, target_ships[0].state, ownship_params)
            encounter_type = action.get("encounter_type", "none")

        return ControlCommand(
            rudder=float(best_rudder),
            propeller=float(best_prop),
            source="DWA",
            metadata={
                "unsupported_features": self.unsupported_features,
                "best_cost": best_cost,
                "encounter_type": encounter_type,
            },
        )

    def _evaluate_action(
        self,
        rudder: float,
        prop: float,
        ownstate: VesselState,
        ownship_params: VesselParams,
        targets: list[TargetShip],
        env: EnvironmentState,
        reference_path: list[VesselState],
    ) -> float:
        """Evaluate cost of a candidate (rudder, propeller) pair."""
        cost = 0.0

        # Predict forward
        pred_state = self._predict(rudder, prop, ownstate, env)

        own_pos = pred_state.position()
        own_vel = body_to_world(pred_state.u, pred_state.v, pred_state.psi)

        # Collision cost
        for tgt in targets:
            tgt_pos = tgt.state.position()
            tgt_vel = body_to_world(tgt.state.u, tgt.state.v, tgt.state.psi)

            # Predict target for the same duration as ownship prediction
            pred_tgt_pos = tgt_pos + tgt_vel * self.predict_time

            d = float(np.linalg.norm(own_pos - pred_tgt_pos))
            if d < self.safety_radius:
                cost += self.w_collision * (self.safety_radius - d)**2 / self.safety_radius**2

            # CPA cost — compute between BOTH predicted ownship AND predicted target.
            # BUG #10 fix (2026-06-10): previously used current target position
            # with predicted ownship position, which was temporally inconsistent.
            # Now both are projected to the same future time.
            cpa = compute_tcpa_dcpa(own_pos, own_vel, pred_tgt_pos, tgt_vel)
            if cpa["dcpa"] < self.safety_radius:
                cost += self.w_collision * (self.safety_radius - cpa["dcpa"])**2 / self.safety_radius**2

            # COLREGs compliance: penalize wrong-direction turns
            # Project target forward to same future time for temporal consistency
            pred_tgt_state = VesselState(
                x=pred_tgt_pos[0], y=pred_tgt_pos[1],
                psi=tgt.state.psi, u=tgt.state.u, v=tgt.state.v, r=tgt.state.r,
                timestamp=tgt.state.timestamp + self.predict_time,
            )
            action = colregs_action(pred_state, pred_tgt_state, ownship_params)
            if action.get("required_course_change", 0) != 0:
                # Check if our rudder direction matches the required turn
                required_dir = np.sign(action["required_course_change"])
                actual_dir = np.sign(rudder)
                if required_dir != 0 and actual_dir != required_dir:
                    cost += self.w_colregs * abs(action["required_course_change"]) / 30.0

        # Track following cost: use a step-indexed waypoint instead of
        # always tracking reference_path[0], which would pull the ship
        # back toward the start point indefinitely.
        if reference_path and len(reference_path) > 0:
            idx = min(self._step_counter, len(reference_path) - 1)
            ref = reference_path[idx]
            track_err = (pred_state.x - ref.x)**2 + (pred_state.y - ref.y)**2
            cost += self.w_track * track_err / 10000.0

        # Smoothness (control change)
        cost += self.w_smooth * (rudder - self.previous_rudder)**2 / 100.0

        # Speed preference (prefer higher speed when safe)
        cost -= self.w_speed * prop

        return cost

    def _predict(
        self,
        rudder: float,
        prop: float,
        state: VesselState,
        env: EnvironmentState,
    ) -> VesselState:
        """Simple kinematic prediction over predict_time horizon.

        Uses a simplified 3-DOF prediction: constant velocity with
        rudder-induced yaw rate.
        """
        # Simplified yaw rate from rudder
        k_rudder = self.yaw_gain  # configurable via dwa_yaw_gain
        r = k_rudder * rudder
        r = np.clip(r, -0.05, 0.05)  # ~3 deg/s max

        psi_final = state.psi + r * self.predict_time
        # Average heading over prediction
        psi_avg = state.psi + r * self.predict_time / 2

        # Kinematic position update
        speed = state.speed()
        u_world = speed * np.cos(psi_avg)
        v_world = speed * np.sin(psi_avg)

        drag_factor = 1.0 / (1.0 + 0.05 * self.predict_time)  # simple drag
        speed_final = speed * drag_factor

        return VesselState(
            x=state.x + u_world * self.predict_time,
            y=state.y + v_world * self.predict_time,
            psi=psi_final,
            u=speed_final,  # body-frame forward speed (correctly v=0)
            v=0.0,
            r=r,
            timestamp=state.timestamp + self.predict_time,
        )

    def reset(self):
        self.previous_rudder = 0.0
        self.previous_propeller = 0.5
        self._step_counter = 0
