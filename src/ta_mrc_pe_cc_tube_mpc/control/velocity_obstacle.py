"""B1 baseline: COLREGs + Velocity Obstacle / OZT (Obstacle Zone by Target).

Traditional reactive collision avoidance: computes velocity obstacle cone,
selects collision-free velocity using COLREGs rule-based preference ordering.

Rudder sign convention (consistent across all controllers):
    rudder > 0  →  starboard turn  (right turn, positive δ)
    rudder < 0  →  port turn       (left turn,  negative δ)
    rudder = 0  →  straight ahead

COLREGs encounter actions:
    head_on          → rudder +15° (starboard, Rule 14)
    crossing_giveway → rudder +20° (starboard, avoid crossing ahead, Rule 15)
    overtaking       → rudder ±10° (away from target, Rule 13)

The convention follows COLREGs Rule 8: any alteration of course shall be
"large enough to be readily apparent" (≥ 10°), and for head-on encounters
"each shall alter her course to starboard" (Rule 14a).
"""

from typing import Optional

import numpy as np

from ..types import ControlCommand, EnvironmentState, TargetShip, VesselParams, VesselState
from ..rules.colregs import colregs_action
from ..risk.cpa import compute_tcpa_dcpa
from ..utils.coordinates import body_to_world, world_to_body


class VelocityObstacleController:
    """COLREGs + Velocity Obstacle reactive collision avoidance.

    B1 baseline from experiment design Section 13.1.
    Uses geometric velocity obstacle cones with COLREGs rule-based
    action selection. No MPC, no chance constraints, no tube.

    Sign convention: positive rudder = starboard turn.
    See module docstring for full COLREGs action table.
    """

    def __init__(self, vessel_params: VesselParams, config: dict = None):
        self.vessel_params = vessel_params
        cfg = config or {}
        controller_cfg = cfg.get("controller", cfg)
        self.time_horizon = controller_cfg.get("vo_time_horizon", 600.0)
        self.safety_radius = controller_cfg.get("vo_safety_radius", 500.0)
        self.max_rudder = vessel_params.max_rudder
        self.max_rudder_rate = vessel_params.max_rudder_rate
        self.dt = controller_cfg.get("dt", 0.5)
        self.previous_rudder = 0.0
        self.previous_propeller = 0.5

        # Derive unsupported features from the controller's capability set.
        # VO supports: COLREGs-based rule selection + basic obstacle zones.
        _all_features = {
            "multi_rule", "target_size", "speed_sensitive_domain",
            "vessel_type_weight", "behavior_uncertainty", "shallow_current",
            "bank_effect", "ship_interaction", "chance_constraint",
            "tube_mpc", "cbf_filter", "fallback",
            "enc_navigability", "hard_constraints", "uncertainty_ais",
        }
        _supported = {"colregs_rule_selection", "velocity_obstacle"}
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
        """Compute control using Velocity Obstacle approach."""
        own_pos = ownstate.position()
        own_vel = body_to_world(ownstate.u, ownstate.v, ownstate.psi)
        own_speed = ownstate.speed()

        if not target_ships:
            return ControlCommand(
                rudder=0.0, propeller=0.5, source="VO",
                metadata={"vo_state": "no_targets"},
            )

        # Find most threatening target (smallest CPA)
        best_cpa = float("inf")
        best_target = target_ships[0]
        for tgt in target_ships:
            tgt_pos = tgt.state.position()
            tgt_vel = body_to_world(tgt.state.u, tgt.state.v, tgt.state.psi)
            cpa = compute_tcpa_dcpa(own_pos, own_vel, tgt_pos, tgt_vel)
            if cpa["dcpa"] < best_cpa:
                best_cpa = cpa["dcpa"]
                best_target = tgt

        tgt_pos = best_target.state.position()
        tgt_vel = body_to_world(best_target.state.u, best_target.state.v, best_target.state.psi)
        rel_pos = tgt_pos - own_pos
        rel_vel = own_vel - tgt_vel
        # Clip to prevent overflow in dot products
        MAX_COMP = 1e4
        rel_pos = np.clip(rel_pos, -MAX_COMP, MAX_COMP)
        rel_vel = np.clip(rel_vel, -MAX_COMP, MAX_COMP)
        dist = float(np.linalg.norm(rel_pos))

        # COLREGs action
        action = colregs_action(ownstate, best_target.state, ownship_params)
        encounter_type = action.get("encounter_type", "none")
        role = action.get("ownship_role", "none")
        course_change = action.get("required_course_change", 0.0)

        # Velocity obstacle check
        vo_radius = self.safety_radius
        if dist < 1e-6:
            dist = 1e-6
        rel_vel_sq = np.dot(rel_vel, rel_vel)
        if rel_vel_sq > 1e12:
            rel_vel_sq = 1e12
        # TCPA sign: with VO convention rel_vel = own_vel - tgt_vel,
        # rel_pos(t) = rel_pos - rel_vel * t  (since d(rel_pos)/dt = tgt_vel - own_vel = -rel_vel)
        # TCPA = dot(rel_pos, rel_vel) / ||rel_vel||^2 (positive = converging).
        # DCPA = ||rel_pos - t_cpa * rel_vel|| (minimum distance in relative frame).
        time_to_closest = np.dot(rel_pos, rel_vel) / max(rel_vel_sq, 1e-12)
        time_to_closest = max(0.0, min(time_to_closest, self.time_horizon))
        closest_dist = np.linalg.norm(rel_pos - time_to_closest * rel_vel)
        # BUG #13 fix (2026-06-10): NaN/Inf fallback must be CONSERVATIVE.
        # Previously set to 1e4 (10 km, treated as "safe"), which would
        # silently skip collision avoidance on numerical failure.
        # Now sets to 0 (treats unknown as imminent collision risk).
        if np.isnan(closest_dist) or np.isinf(closest_dist):
            closest_dist = 0.0  # conservative: treat as imminent collision

        rudder = 0.0
        propeller = 0.5

        # Convert relative position to body frame for correct port/starboard
        # decision relative to ownship's heading (not world north).
        body_y = world_to_body(rel_pos[0], rel_pos[1], ownstate.psi)[1]

        if closest_dist < vo_radius:
            # Collision risk: apply COLREGs-preferred evasive action
            if encounter_type in ("head_on", "crossing_giveway"):
                rudder = 30.0  # starboard turn — meets Rule 8 minimum (30 deg)
            elif encounter_type == "overtaking_giveway":
                # body_y > 0 = target to starboard → turn to starboard (positive rudder)
                # to keep clear per Rule 13
                rudder = 10.0 if body_y > 0 else -10.0
            elif course_change != 0:
                rudder = np.clip(course_change, -self.max_rudder, self.max_rudder)
            else:
                # Fallback: turn away from target (body-frame)
                rudder = -15.0 if body_y > 0 else 15.0

            # Reduce speed in close encounters
            if dist < vo_radius * 0.5:
                propeller = 0.2

        # Rudder rate limiting (max_rudder_rate is deg/s, scale by dt)
        rudder = np.clip(rudder, -self.max_rudder, self.max_rudder)
        max_step = self.max_rudder_rate * self.dt
        rudder_rate = abs(rudder - self.previous_rudder)
        if rudder_rate > max_step:
            rudder = self.previous_rudder + np.sign(rudder - self.previous_rudder) * max_step

        self.previous_rudder = rudder
        self.previous_propeller = propeller

        return ControlCommand(
            rudder=float(rudder),
            propeller=float(propeller),
            source="VO",
            metadata={
                "unsupported_features": self.unsupported_features,
                "encounter_type": encounter_type,
                "role": role,
                "vo_radius": vo_radius,
                "closest_dist": closest_dist,
                "dist": dist,
            },
        )

    def reset(self):
        self.previous_rudder = 0.0
        self.previous_propeller = 0.5
