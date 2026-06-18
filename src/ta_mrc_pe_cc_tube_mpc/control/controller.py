"""TA-MRC-PE-CC-Tube-MPC main controller.

Orchestrates:
    1. Rule assessment (multi-source rule engine)
    2. Dynamic ship domain computation
    3. Tube-inspired robust safety buffer (physics-enhanced, adaptive scaling)
    4. Chance-constrained MPC
    5. CBF-QP safety filter
    6. Fallback strategy

Note on naming: The "tube" component is a tube-*inspired* adaptive safety
buffer, NOT a formally certified tube-MPC with RPI guarantee.  See
docs/theory_claim_boundaries.md for the formal vs. empirical claim boundary.

Backend dispatch:
    - ``mpc.backend = "casadi"`` → CasADi + IPOPT (default, CPU, surrogate dynamics)
    - ``mpc.backend = "jax"``    → JAX + L-BFGS-B (GPU-accelerated)
    - ``mpc.backend = "scipy"``  → scipy SLSQP (fallback, full MMG model)
"""

import time
from typing import Optional

import numpy as np

from ..constants import DEG_TO_RAD
from ..models.mmg_3dof import MMG3DOFModel
from ..physics.bank_effect import assess_bank_effect, compute_bank_disturbance_force
from ..physics.shallow_water import assess_shallow_water, compute_shallow_water_disturbance
from ..physics.ship_interaction import assess_ship_interaction, compute_ship_interaction_force
from ..physics.tube_boundary import compute_tube_radius
from ..physics.wind_current import assess_wind_current
from ..risk.dynamic_ship_domain import DynamicShipDomain
from ..risk.encounter_classifier import classify_encounter
from ..risk.imm_filter import IMMFilter
from ..rules.rule_engine import RuleEngine
from ..types import ControlCommand, EnvironmentState, RuleAssessment, TargetShip, VesselParams, VesselState
from ..utils.coordinates import body_to_world
from .adaptive_config import apply_adaptive_config
from .cbf_qp import CBFQPSafetyFilter, CBF_OUTPUT_KEYS
from .chance_constraints import ChanceConstraintEvaluator
from .fallback import FallbackController
from .path_planner import generate_reference_trajectory
from .tube_mpc import TubeMPC

# ── Optional JAX imports ──────────────────────────────────────────────────
try:
    import jax as _jax
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    _jax = None


class TAMRCPECCTubeMPCController:
    """Target-Aware Multi-Rule Constrained Physics-Enhanced
    Chance-Constrained Tube MPC Controller.

    This is the main controller class that orchestrates all components
    of the proposed method.
    """

    def __init__(
        self,
        vessel_params: VesselParams,
        config: dict = None,
    ):
        """
        Args:
            vessel_params: Ownship physical parameters.
            config: Full controller configuration dict.
        """
        self.vessel_params = vessel_params
        self.config = config or {}
        cfg = self.config

        # Controller feature flags
        self.flags = cfg.get("controller", {})
        self.enable_multi_rule = self.flags.get("enable_multi_rule", True)
        self.enable_target_size = self.flags.get("enable_target_size", True)
        self.enable_speed_sensitive_domain = self.flags.get("enable_speed_sensitive_domain", True)
        self.enable_vessel_type_weight = self.flags.get("enable_vessel_type_weight", True)
        self.enable_behavior_uncertainty = self.flags.get("enable_behavior_uncertainty", True)
        self.enable_shallow_water = self.flags.get("enable_shallow_water", True)
        self.enable_wind_current = self.flags.get("enable_wind_current", True)
        self.enable_bank_effect = self.flags.get("enable_bank_effect", True)
        self.enable_ship_interaction = self.flags.get("enable_ship_interaction", True)
        self.enable_chance_constraint = self.flags.get("enable_chance_constraint", True)
        self.enable_tube_mpc = self.flags.get("enable_tube_mpc", True)
        self.enable_cbf_filter = self.flags.get("enable_cbf_filter", True)
        self.enable_fallback = self.flags.get("enable_fallback", True)

        # Diagnostic: log active feature flags for debugging ablation studies
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        _active = [k for k, v in sorted(self.flags.items()) if v and k.startswith("enable_")]
        _disabled = [k for k, v in sorted(self.flags.items()) if not v and k.startswith("enable_")]
        _logger.info(
            "Controller flags — enabled: %s | disabled: %s",
            ", ".join(_active) or "(none)",
            ", ".join(_disabled) or "(none)",
        )

        # MMG dynamics model
        self.mmg = MMG3DOFModel({
            "length": vessel_params.length,
            "beam": vessel_params.beam,
            "draught": vessel_params.draught,
            "mass": vessel_params.mass,
            "Iz": vessel_params.Iz,
            "max_rudder": vessel_params.max_rudder,
            "max_rudder_rate": vessel_params.max_rudder_rate,
            "max_yaw_rate": vessel_params.max_yaw_rate,
            "min_turning_radius": vessel_params.min_turning_radius_multiplier,
        })

        # Sub-components
        self.rule_engine = RuleEngine(cfg.get("rules", {}))
        self.ship_domain = DynamicShipDomain(cfg.get("domain", {}))
        self.chance_evaluator = ChanceConstraintEvaluator(cfg.get("uncertainty", {}))
        self.tube_mpc = TubeMPC(
            dynamics=self._dynamics_wrapper,
            vessel_params=vessel_params,
            horizon=cfg.get("mpc", {}).get("horizon", 20),
            dt=cfg.get("mpc", {}).get("dt", 0.5),
            config=cfg,
        )
        # Detect JAX mode from MPC backend config
        mpc_backend = cfg.get("mpc", {}).get("backend", cfg.get("mpc", {}).get("solver", "casadi"))
        self._jax_mode = str(mpc_backend).lower().strip() == "jax" and _HAS_JAX

        # Configure JAX platform and precision when JAX mode is active
        if self._jax_mode:
            import jax
            jax_cfg = cfg.get("jax", {})
            platform = jax_cfg.get("platform", "gpu")
            enable_x64 = jax_cfg.get("enable_x64", True)
            if platform == "gpu" and any(d.platform == "gpu" for d in jax.devices()):
                jax.config.update("jax_platform_name", "gpu")
            if enable_x64:
                jax.config.update("jax_enable_x64", True)

        # CBF-QP filter: use JAX version when in JAX mode for consistent GPU execution
        if self._jax_mode:
            from .cbf_jax import JAXCBFQPSafetyFilter
            self.cbf_filter = JAXCBFQPSafetyFilter(cfg.get("cbf", {}))
        else:
            self.cbf_filter = CBFQPSafetyFilter(cfg.get("cbf", {}))
            self.cbf_filter.set_mmg_model(self.mmg)

        fallback_cfg = {**(cfg.get("fallback", {})), "draught": vessel_params.draught, "beam": vessel_params.beam, "length": vessel_params.length}
        self.fallback = FallbackController(fallback_cfg)

        self.previous_command: Optional[ControlCommand] = None
        self.reference_trajectory: list[VesselState] = []

        # Per-episode accumulators for CBF, fallback, and timing metrics.
        # These are reset at the start of each episode via reset_episode_counters().
        self._total_cbf_interventions = 0
        self._total_cbf_infeasible = 0
        self._total_fallback_steps = 0
        self._episode_has_fallback = False
        # Cache domain and disturbance results between control steps.
        # Reset at episode boundaries via reset_episode_counters().
        self._cached_domain_results = None
        self._cached_domain_state_hash = None
        self._cached_disturbance = None
        self._cached_disturbance_state_hash = None
        self._timing_breakdown = {
            "rule_assessment": [],
            "ship_domain": [],
            "tube_radius": [],
            "mpc_solve": [],
            "cbf_filter": [],
            "fallback_check": [],
        }

        # IMM filter for multi-modal target intent tracking (Branch MPC).
        # Initialised once at controller construction; per-target observation
        # updates happen each control step via set_target_observation().
        self.imm_filter = IMMFilter()

    def reset_episode_counters(self):
        """Reset per-episode accumulators for CBF, fallback, and timing.

        Called at the start of each episode by the simulator / runner.
        """
        self._total_cbf_interventions = 0
        self._total_cbf_infeasible = 0
        self._total_fallback_steps = 0
        self._episode_has_fallback = False
        self._cached_domain_results = None  # P3: invalidate domain cache on reset
        self._cached_domain_state_hash = None
        self._cached_disturbance = None  # Also invalidate disturbance cache
        self._cached_disturbance_state_hash = None
        self._timing_breakdown = {
            "rule_assessment": [],
            "ship_domain": [],
            "tube_radius": [],
            "mpc_solve": [],
            "cbf_filter": [],
            "fallback_check": [],
        }

    def _dynamics_wrapper(
        self, state: VesselState, command: ControlCommand,
        env: EnvironmentState, dt: float,
    ) -> VesselState:
        """Wrapper around MMG step for MPC internal use."""
        targets = getattr(self, "_mpc_targets", [])
        disturbance = self._compute_total_disturbance(state, targets, env)
        return self.mmg.step(state, command, env, dt, disturbance)

    def _compute_total_disturbance(
        self,
        own_state: VesselState,
        targets: list[TargetShip],
        env: EnvironmentState,
    ) -> np.ndarray:
        """Compute total external disturbance force [X, Y, N] in body frame.

        BUG #6 fix (2026-06-10): Caches the disturbance force to avoid
        redundant computation (was called twice per control step).
        """
        # Cache check: exact state match → reuse.
        # Uses tight precision (0.01m, 0.001rad, 0.01m/s) so the cache only
        # prevents the redundant second call for the SAME state, not for
        # different candidate states during MPC optimization.
        state_hash = (round(own_state.x, 2), round(own_state.y, 2),
                       round(own_state.psi, 3), round(own_state.speed(), 2),
                       round(getattr(env, 'water_depth', 0.0), 1),
                       round(getattr(env, 'bank_distance_left', 0.0) or 0.0, 1),
                       round(getattr(env, 'bank_distance_right', 0.0) or 0.0, 1),
                       len(targets))
        if (self._cached_disturbance is not None
                and self._cached_disturbance_state_hash == state_hash):
            return self._cached_disturbance.copy()

        force = np.zeros(3)

        # Shallow water
        if self.enable_shallow_water:
            sw = assess_shallow_water(
                env.water_depth, self.vessel_params.draught,
                own_state.speed(), self.vessel_params.length,
            )
            force += compute_shallow_water_disturbance(sw, own_state.speed())

        # Bank effect
        if self.enable_bank_effect:
            bank = assess_bank_effect(
                own_state.speed(), self.vessel_params.length,
                self.vessel_params.beam,
                env.bank_distance_left, env.bank_distance_right,
            )
            force += compute_bank_disturbance_force(
                bank,
                env.bank_distance_left if env.bank_distance_left is not None else float("inf"),
                env.bank_distance_right if env.bank_distance_right is not None else float("inf"),
                own_state.speed(),
            )

        # Ship interaction
        if self.enable_ship_interaction and targets:
            ship_int = assess_ship_interaction(
                own_state, self.vessel_params.length,
                self.vessel_params.beam, targets,
            )
            force += compute_ship_interaction_force(ship_int, own_state, targets)

        # Wind / current
        if self.enable_wind_current:
            wc = assess_wind_current(
                env.current_x, env.current_y, env.wind_x, env.wind_y,
                self.vessel_params.length, self.vessel_params.beam,
                own_state.speed(),
                own_psi=own_state.psi,
            )
            force += wc.wind_force

        # BUG #6: Cache for reuse
        self._cached_disturbance = force.copy()
        self._cached_disturbance_state_hash = state_hash
        return force

    def compute_control(
        self,
        ownstate: VesselState,
        ownship_params: VesselParams,
        target_ships: list[TargetShip],
        environment: EnvironmentState,
        rule_context: dict = None,
        reference_path: list[VesselState] = None,
    ) -> ControlCommand:
        """Main control computation pipeline.

        Args:
            ownstate: Current ownship state.
            ownship_params: Ownship parameters.
            target_ships: List of observed target ships.
            environment: Current environment state.
            rule_context: Optional pre-computed rule assessment.
            reference_path: Reference trajectory for path following.

        Returns:
            ControlCommand with rudder and propeller commands.
        """
        t_start = time.perf_counter()
        metadata = {}

        # Reset accumulators at start of first step (episode boundary)
        if self.previous_command is None:
            self.reset_episode_counters()

        t_rule_start = time.perf_counter()
        # Step 0: Apply scenario-adaptive parameter configuration
        adaptive_params = apply_adaptive_config(
            self, environment, target_ships,
            ownship_params.draught, ownship_params.length, self.config,
        )
        metadata["adaptive_params"] = adaptive_params

        # Store context for path planner
        self._last_env = environment

        # Store reference trajectory
        if reference_path is not None:
            self.reference_trajectory = reference_path

        # Step 1: Rule assessment
        enc_layer = None
        vts_zone = None
        coast_pilot_rules = None
        if self.enable_multi_rule:
            if rule_context:
                enc_layer = rule_context.get("enc_layer")
                vts_zone = rule_context.get("vts_zone")
                coast_pilot_rules = rule_context.get("coast_pilot_rules")

            # Build tss_config from ENC layer
            tss_config = None
            if enc_layer is not None:
                tss_config = {
                    "lanes": getattr(enc_layer, "tss_lanes", []),
                    "atba_zones": getattr(enc_layer, "atba_zones", []),
                    "precautionary_areas": getattr(enc_layer, "precautionary_areas", []),
                    "separation_zones": getattr(enc_layer, "separation_zones", []),
                    "channel_boundaries": getattr(enc_layer, "channel_boundaries", []),
                }

            rule_assessment = self.rule_engine.assess(
                ownstate, ownship_params, target_ships, environment,
                enc_layer=enc_layer,
                tss_config=tss_config,
                vts_zone=vts_zone,
                coast_pilot_rules=coast_pilot_rules,
            )
        else:
            # COLREGs-only mode
            rule_assessment = RuleAssessment()
            if target_ships:
                encounter = classify_encounter(ownstate, target_ships[0].state)
                rule_assessment.encounter_type = encounter["type"]
                rule_assessment.ownship_role = encounter["ownship_role"]
        metadata["rule_assessment"] = rule_assessment
        self._last_enc_layer = enc_layer
        t_rule_end = time.perf_counter()

        # Step 2: Compute dynamic ship domain for each target
        encounter_roles = []
        for tgt in target_ships:
            enc = classify_encounter(ownstate, tgt.state)
            encounter_roles.append(enc["ownship_role"])

        # P3 fix (2026-06-10): Use cached domain results when ownship state
        # hasn't changed significantly since the last control step.  Domain
        # computation costs O(targets × 9 factors) and most factors change
        # slowly (vessel type, bank proximity, etc.).
        own_speed = ownstate.speed()
        # Domain cache: 2.0m position grid (was 10.0m — too coarse for
        # close-quarters encounters where 10m position change is significant).
        state_hash = (round(ownstate.x / 2.0), round(ownstate.y / 2.0),
                      round(ownstate.psi, 1), round(own_speed * 2) / 2,
                      round(getattr(environment, 'bank_distance_left', 0.0) or 0.0, 0),
                      round(getattr(environment, 'bank_distance_right', 0.0) or 0.0, 0),
                      len(target_ships))
        use_cached = (
            self._cached_domain_results is not None
            and self._cached_domain_state_hash == state_hash
        )

        domain_results = []
        d_safe_eff_list = []
        if use_cached:
            domain_results = self._cached_domain_results
            d_safe_eff_list = [dr["total"] for dr in domain_results]
        else:
            for i, target in enumerate(target_ships):
                # Check AIS quality
                ais_cfg = target.ais_quality
                delay = ais_cfg.get("delay", 0.0)

                # Compute bank/ship effect flags
                bank_active = False
                if self.enable_bank_effect:
                    bank_active = (
                        (environment.bank_distance_left is not None
                         and environment.bank_distance_left < 5 * self.vessel_params.length)
                        or (environment.bank_distance_right is not None
                            and environment.bank_distance_right < 5 * self.vessel_params.length)
                    )

                ship_risk = 0.0
                if self.enable_ship_interaction:
                    ship_int = assess_ship_interaction(
                        ownstate, self.vessel_params.length,
                        self.vessel_params.beam, [target],
                    )
                    ship_risk = ship_int.risk_score

                domain_result = self.ship_domain.compute(
                    own_state=ownstate,
                    own_params=ownship_params,
                    target=target,
                    env=environment,
                    encounter_role=encounter_roles[i] if i < len(encounter_roles) else "none",
                    ais_delay=delay,
                    ais_dropout=False,
                    ship_interaction_risk=ship_risk,
                    bank_effect_active=bank_active,
                    enable_target_size=self.enable_target_size,
                    enable_speed_sensitive=self.enable_speed_sensitive_domain,
                    enable_vessel_type_weight=self.enable_vessel_type_weight,
                    enable_behavior_uncertainty=self.enable_behavior_uncertainty,
                    enable_environment=self.enable_shallow_water or self.enable_wind_current,
                    enable_bank_effect=self.enable_bank_effect,
                    enable_ship_interaction=self.enable_ship_interaction,
                )
                domain_results.append(domain_result)
                d_safe = domain_result["total"]
                d_safe_eff_list.append(d_safe)

            # P3: Store domain results in cache AFTER the for loop.
            # We are inside the else branch, so domain results were freshly computed.
            self._cached_domain_results = domain_results
            self._cached_domain_state_hash = state_hash

        metadata["domain_results"] = domain_results

        # Apply rule engine safety domain scale
        scale = rule_assessment.safety_domain_scale
        d_safe_eff_list = [d * scale for d in d_safe_eff_list]

        # Cap d_safe_eff at 2× base domain per target to prevent constraint
        # infeasibility.  Each target has its own d_base = 2×(L_i+L_j), so
        # the cap is per-target: larger targets get a proportionally larger cap.
        _default_base = 4.0 * ownship_params.length
        d_safe_eff_list = [
            min(d, 2.0 * (domain_results[i].get("d_base", _default_base) if i < len(domain_results) else _default_base))
            for i, d in enumerate(d_safe_eff_list)
        ]

        metadata["d_safe_eff_list"] = d_safe_eff_list
        metadata["safety_domain_scale"] = scale
        t_domain_end = time.perf_counter()

        # Step 3: Compute tube radius (physics-enhanced)
        if self.enable_tube_mpc:
            # Use the adaptive rho_0 from tube_mpc if available; fall back to
            # the default (5.0) in compute_tube_radius. This ensures that
            # adaptive_config's rho_0 modification is actually propagated.
            tube_components = compute_tube_radius(
                own_state=ownstate,
                own_length=ownship_params.length,
                own_beam=ownship_params.beam,
                own_draught=ownship_params.draught,
                targets=target_ships,
                env=environment,
                rho_0=self.tube_mpc.rho_0,
                rho_maneuver=self.tube_mpc.rho_maneuver,
                config=self.tube_mpc.tube_cfg,
                use_adaptive=self.tube_mpc.use_adaptive_tube,
            )
            rho_tube = tube_components["radius"]
            metadata["tube_components"] = tube_components
        else:
            rho_tube = 0.0
        t_tube_end = time.perf_counter()

        # Step 4: Generate reference trajectory if not provided
        if not self.reference_trajectory:
            self.reference_trajectory = self._generate_default_reference(
                ownstate, self.tube_mpc.N + 1,
            )

        # Step 4.5: Update IMM filter with observed target positions.
        # Each target gets PER-MODE predictions that differ based on dynamics:
        #   CV: constant velocity extrapolation
        #   CT: yaw-rate-based turning prediction
        #   COLREGS: rule-compliant avoidance heading
        #   NON_COMPLIANT: wrong-direction or non-avoiding motion
        #   CHANNEL: along channel centerline heading
        #   ERRATIC: CV center but with inflated observation noise (lower likelihood sharpness)
        # This provides probabilistic mode weights for Branch MPC expected cost.
        imm_mode_probs = None
        imm_entropy = 0.0
        if self.enable_behavior_uncertainty and target_ships:
            encounter_type = rule_assessment.encounter_type if hasattr(rule_assessment, "encounter_type") else "none"
            # Channel heading for channel-following mode.
            # environment.channel_heading does not exist on EnvironmentState;
            # the correct source is enc_layer (always checked first).
            channel_heading = None
            if enc_layer is not None and hasattr(enc_layer, "channel_centerline_heading"):
                channel_heading = enc_layer.channel_centerline_heading

            for i, target in enumerate(target_ships):
                target_key = target.mmsi if target.mmsi else f"target_{i}"
                tgt_state = target.state
                # Compute world-frame velocity
                tgt_vel = body_to_world(tgt_state.u, tgt_state.v, tgt_state.psi)
                tgt_speed = float(np.linalg.norm(tgt_vel))
                tgt_heading = tgt_state.psi
                tgt_turn_rate = tgt_state.r

                # Provide per-target observation to IMM filter
                # This internally: applies Markov transition, generates distinct
                # per-mode predictions, and performs Bayesian update
                self.imm_filter.set_target_observation(
                    target_key=target_key,
                    position=tgt_state.position(),
                    velocity=tgt_vel,
                    heading=tgt_heading,
                    speed=tgt_speed,
                    turn_rate=tgt_turn_rate,
                    timestamp=tgt_state.timestamp,
                    encounter_type=encounter_type,
                    channel_heading=channel_heading,
                )

            # Aggregate mode probabilities across all targets (mean pooling)
            all_probs = []
            for i, target in enumerate(target_ships):
                target_key = target.mmsi if target.mmsi else f"target_{i}"
                all_probs.append(self.imm_filter.get_target_probs(target_key))
            if all_probs:
                imm_mode_probs = np.mean(all_probs, axis=0)
                imm_entropy = float(np.mean([
                    self.imm_filter.get_target_entropy(target.mmsi if target.mmsi else f"target_{i}")
                    for i, target in enumerate(target_ships)
                ]))
            else:
                imm_mode_probs = self.imm_filter.mode_probs.copy()
                imm_entropy = self.imm_filter.get_entropy()

            # Scale tube radius by IMM entropy: more uncertainty → larger tube.
            # Threshold and scale are configurable (were hardcoded 0.5/0.3).
            _entropy_threshold = self.config.get("imm_entropy_threshold", 0.5)
            _entropy_scale = self.config.get("imm_entropy_scale", 0.3)
            if imm_entropy > _entropy_threshold:
                rho_tube *= (1.0 + _entropy_scale * min(imm_entropy, 1.0))

            metadata["imm_mode_probs"] = imm_mode_probs.tolist() if hasattr(imm_mode_probs, "tolist") else list(imm_mode_probs)
            metadata["imm_entropy"] = float(imm_entropy)
            metadata["imm_per_target_count"] = len(target_ships)

        # Step 5: Pass rule context to MPC and solve Tube-MPC
        self.tube_mpc.mpc.set_rule_context(
            soft_constraints=rule_assessment.soft_constraints,
            p5_terms=rule_assessment.p5_terms,
            waterway_safety_scale=scale,
        )
        # Reset MMG rudder state before MPC solve to ensure deterministic
        # dynamics within the optimizer (prevents _prev_rudder from leaking
        # across calls and causing non-reproducible rollouts).
        if self.previous_command is not None:
            self.mmg.reset_rudder_state(
                self.previous_command.rudder * DEG_TO_RAD
            )
        self._mpc_targets = target_ships  # make targets available to _dynamics_wrapper
        own_Sigma = np.eye(2) * (self.flags.get("own_position_std", 1.0) ** 2)
        mpc_cmd, mpc_info = self.tube_mpc.compute_control(
            own_state=ownstate,
            own_Sigma=own_Sigma,
            targets=target_ships,
            env=environment,
            reference_trajectory=self.reference_trajectory,
            d_safe_eff_list=d_safe_eff_list,
            previous_command=self.previous_command,
            forbidden_zones=rule_assessment.forbidden_zones,
            hard_constraints=rule_assessment.hard_constraints,
            enc_layer=enc_layer,
            rho_tube_override=rho_tube,  # 0.0 when disabled, computed value when enabled
            enable_chance_constraint=self.enable_chance_constraint,
        )
        mpc_info["rule_priority"] = rule_assessment.rule_priority
        metadata["mpc_info"] = mpc_info
        mpc_feasible = mpc_cmd is not None
        t_mpc_end = time.perf_counter()

        if not mpc_feasible:
            mpc_cmd = ControlCommand(
                rudder=0.0, propeller=0.3,
                source="mpc_infeasible",
                metadata={"reason": "mpc_failed"},
            )
            metadata["mpc_feasible"] = False
        else:
            metadata["mpc_feasible"] = True
            mpc_cmd.source = "MPC"

        # Step 6: CBF-QP safety filter
        if self.enable_cbf_filter:
            filtered_cmd, cbf_info = self.cbf_filter.filter(
                u_mpc=mpc_cmd,
                own_state=ownstate,
                own_params=ownship_params,
                targets=target_ships,
                env=environment,
                d_safe_eff_list=d_safe_eff_list,
            )
            metadata["cbf_info"] = cbf_info
            # Accumulate per-step CBF counts (B10: use key constants)
            if cbf_info.get(CBF_OUTPUT_KEYS["ACTIVE"], False):
                self._total_cbf_interventions += 1
            if cbf_info.get(CBF_OUTPUT_KEYS["INFEASIBLE"], False):
                self._total_cbf_infeasible += 1
        else:
            filtered_cmd = mpc_cmd
            cbf_info = {"cbf_active": False, "active_constraints": [], "over_intervention": False}
        t_cbf_end = time.perf_counter()

        # Step 7: Fallback check
        if self.enable_fallback:
            runtime = time.perf_counter() - t_start
            fallback_cmd = self.fallback.update(
                mpc_feasible=mpc_feasible,
                cbf_info=cbf_info,
                own_state=ownstate,
                targets=target_ships,
                env=environment,
                d_safe_eff_list=d_safe_eff_list,
                runtime_s=runtime,
            )

            if self.fallback.is_active():
                final_cmd = fallback_cmd
                metadata["fallback_active"] = True
                self._total_fallback_steps += 1
                self._episode_has_fallback = True
                metadata["fallback_level"] = fallback_cmd.metadata.get("fallback_level", "unknown")
                # Apply the fallback's safety domain expansion to the domain scale.
                # This ensures the expanded safety domain calculated by the fallback
                # is actually used in the next control cycle's domain computation.
                expansion = self.fallback.state.safety_domain_expansion
                if expansion > 1.0:
                    scale = scale * expansion
                    d_safe_eff_list = [d * expansion for d in d_safe_eff_list]
                    # Re-apply per-target cap after fallback expansion
                    _default_base = 4.0 * ownship_params.length
                    d_safe_eff_list = [
                        min(d, 2.0 * (domain_results[i].get("d_base", _default_base) if i < len(domain_results) else _default_base))
                        for i, d in enumerate(d_safe_eff_list)
                    ]
                    metadata["d_safe_eff_list"] = d_safe_eff_list
                    metadata["safety_domain_scale"] = scale
            else:
                final_cmd = filtered_cmd
                metadata["fallback_active"] = False
        else:
            final_cmd = filtered_cmd
            metadata["fallback_active"] = False
        t_fallback_end = time.perf_counter()

        # ── Accumulate per-step counters in episode-level metadata ─────────
        metadata["cbf_intervention_total"] = self._total_cbf_interventions
        metadata["cbf_infeasible_total"] = self._total_cbf_infeasible
        metadata["fallback_step_total"] = self._total_fallback_steps
        metadata["episode_has_fallback"] = self._episode_has_fallback

        # Store timing breakdown for profiling
        metadata["timing_breakdown"] = {
            "rule_assessment": t_rule_end - t_rule_start,
            "ship_domain": t_domain_end - t_rule_end,
            "tube_radius": t_tube_end - t_domain_end,
            "mpc_solve": t_mpc_end - t_tube_end,
            "cbf_filter": t_cbf_end - t_mpc_end,
            "fallback": t_fallback_end - t_cbf_end,
        }

        # Store previous command
        self.previous_command = final_cmd

        # Step 8: Evaluate chance constraints for reporting
        cc_results = []  # always defined so downstream code never hits NameError
        if self.enable_chance_constraint and target_ships:
            # Derive waterway_type from environment for correct epsilon selection
            wid = getattr(environment, 'waterway_id', '').lower()
            if 'tss' in wid or 'ferry' in wid:
                wtype = 'tss_ferry'
            elif 'tanker' in wid:
                wtype = 'tanker_lowvis'
            elif 'restricted' in wid:
                wtype = 'restricted'
            else:
                wtype = 'normal'
            cc_results = self.chance_evaluator.evaluate(
                own_state=ownstate,
                own_Sigma=own_Sigma,
                targets=target_ships,
                d_safe_eff_list=d_safe_eff_list,
                rho_tube=rho_tube,
                env=environment,
                waterway_type=wtype,
            )
            metadata["chance_constraint_results"] = cc_results

        # Step 8.5: Build centralized safety distance breakdown.
        # This single source of truth replaces ad-hoc per-module arithmetic
        # and ensures MPC, CBF, and metrics report consistent safety margins.
        from ..types import SafetyDistanceBreakdown

        safety_breakdowns = []
        # Use dynamic domain base (d_base from domain computation) rather than
        # hardcoded 4×L, so the cap aligns with the actual domain model.
        # Per-target: each target has its own d_base = 2×(L_i+L_j).
        _default_base = 4.0 * ownship_params.length
        for i, target in enumerate(target_ships):
            dr = domain_results[i] if i < len(domain_results) else {}
            base_domain = dr.get("d_base", _default_base)
            d_safe = d_safe_eff_list[i] if i < len(d_safe_eff_list) else 500.0
            # Extract dynamic margin as the sum of all dynamic expansions
            dynamic_total = dr.get("total", base_domain) - base_domain
            # L2 fix (2026-06-10): negative dynamic_total indicates a domain
            # computation error (total < base). Log warning instead of silently
            # clamping to zero, which would hide the bug.
            if dynamic_total < -1.0:
                import logging
                _log = logging.getLogger(__name__)
                _log.warning(
                    "SafetyDistanceBreakdown: negative dynamic_margin=%.1f for "
                    "target %d. Domain total=%.1f, base_domain=%.1f. "
                    "This may indicate a dynamic ship domain computation error.",
                    dynamic_total, i, dr.get("total", 0.0), base_domain,
                )
            # Extract chance constraint margin
            cc_margin = 0.0
            if self.enable_chance_constraint and cc_results and i < len(cc_results):
                cc_margin = cc_results[i].get("uncertainty_margin", 0.0)
            # Extract tube margin
            tube_margin_val = rho_tube if self.enable_tube_mpc else 0.0
            # Extract rule margin (the scaling beyond 1.0)
            rule_margin_val = (scale - 1.0) * base_domain if scale > 1.0 else 0.0
            breakdown = SafetyDistanceBreakdown(
                base_domain=base_domain,
                dynamic_margin=max(0.0, dynamic_total),
                chance_margin=cc_margin,
                tube_margin=tube_margin_val,
                rule_margin=rule_margin_val,
            )
            safety_breakdowns.append(breakdown.to_dict())
            # NOTE: breakdown.total is CAPPED at 2× base_domain.
            # Use total_raw for evaluation metrics (stored in breakdown dict).
        metadata["safety_distance_breakdown"] = safety_breakdowns

        # Store evaluation-safe d_safe_eff_list (uncapped) for metrics.
        # The capped version is already in d_safe_eff_list (used for MPC).
        metadata["d_safe_evaluation_list"] = [
            bd["total_raw"] for bd in safety_breakdowns
        ]

        # Step 9: Compute total disturbance force for simulator dynamics
        # This ensures the actual ship motion reflects the same physical
        # disturbances (bank, shallow, ship interaction, wind/current) that
        # the MPC predicts internally via _dynamics_wrapper.
        total_disturbance = self._compute_total_disturbance(
            ownstate, target_ships, environment,
        )
        metadata["total_disturbance"] = total_disturbance.tolist()

        # Aggregate metadata
        runtime = time.perf_counter() - t_start
        final_cmd.metadata = {**final_cmd.metadata, **metadata, "runtime": runtime,
                              "backend": self.tube_mpc.backend,
                              "jax_mode": self._jax_mode}

        # Track GPU device info in JAX mode (ISSUE #16 — 2026-06-10)
        if self._jax_mode:
            try:
                devices = _jax.devices()
                final_cmd.metadata["gpu_device_count"] = len(devices)
                final_cmd.metadata["gpu_devices"] = [str(d) for d in devices]
                # GPU memory tracking (if available)
                for i, d in enumerate(devices):
                    if hasattr(d, "memory_stats"):
                        try:
                            stats = d.memory_stats()
                            final_cmd.metadata[f"gpu_{i}_memory_used_bytes"] = stats.get("bytes_in_use", 0)
                        except Exception:
                            pass
            except Exception:
                final_cmd.metadata["gpu_devices"] = "unknown"

        return final_cmd

    def _generate_default_reference(
        self, current_state: VesselState, n_steps: int,
    ) -> list[VesselState]:
        """Generate a default reference trajectory via path planner.

        Uses the path_planner module which handles:
        - Channel centerline following
        - Bend navigation
        - Obstacle/bridge pier avoidance
        - Falls back to straight-line when no env context is available.
        """
        # Build obstacle list from enc_layer bridge piers if available
        obstacles = None
        env = None

        # Try to extract environment from recent context
        if hasattr(self, "_last_env") and self._last_env is not None:
            env = self._last_env

        if hasattr(self, "_last_enc_layer") and self._last_enc_layer is not None:
            enc = self._last_enc_layer
            if hasattr(enc, "bridge_piers") and enc.bridge_piers is not None:
                obstacles = []
                for p in enc.bridge_piers:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        obstacles.append({"x": p[0], "y": p[1], "radius": 5.0})
                    elif hasattr(p, "x") and hasattr(p, "y"):
                        obstacles.append({"x": p.x, "y": p.y, "radius": getattr(p, "radius", 5.0)})

        return generate_reference_trajectory(
            current_state=current_state,
            n_steps=n_steps,
            dt=self.tube_mpc.dt,
            env=env,
            obstacles=obstacles,
        )

    def reset(self):
        """Reset controller state for a new episode."""
        self.previous_command = None
        self.reference_trajectory = []
        self.fallback.reset()
        self.tube_mpc.mpc.reset_warm_start()
        self._last_env = None
        self._last_enc_layer = None
