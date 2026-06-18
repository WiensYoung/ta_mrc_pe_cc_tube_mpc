"""Core simulation engine.

Advances ownship dynamics, target ship trajectories, and environment
at each time step. Manages the simulation loop and records history.
"""

import time
from typing import Optional

import numpy as np

from ..models.mmg_3dof import MMG3DOFModel
from ..models.target_ship import get_noisy_target_state, propagate_target, update_target_covariance
from ..types import ControlCommand, EnvironmentState, EpisodeConfig, EpisodeResult, TargetShip, VesselState
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Mapping from FallbackLevel name strings to severity ordering for aggregation.
# Must match the FallbackLevel enum values in control/fallback.py.
_FALLBACK_LEVEL_ORDER = {"NONE": 0, "CAUTION": 1, "REDUCE_SPEED": 2, "STOP": 3, "EMERGENCY": 4}


class Simulator:
    """Closed-loop ship maneuvering simulator."""

    def __init__(self, config: dict = None):
        """
        Args:
            config: Simulation configuration dict.
        """
        cfg = config or {}
        self.dt = cfg.get("dt", 0.5)
        self.max_duration = cfg.get("closed_loop_duration", 600.0)
        self.rng_seed = cfg.get("random_seed", 42)
        self.runtime_deadline = cfg.get("fallback", {}).get("runtime_deadline", 0.1)

    def run_episode(
        self,
        episode: EpisodeConfig,
        controller,
        rng: np.random.Generator = None,
        method: str = "Proposed",
    ) -> EpisodeResult:
        """Run a single simulation episode.

        Args:
            episode: Episode configuration.
            controller: Controller object with compute_control method.
            rng: Seeded random generator for reproducibility.
            method: Method identifier for the result (e.g. "Proposed", "B3").

        Returns:
            EpisodeResult with full history, metrics, and metadata.
        """
        if rng is None:
            rng = np.random.default_rng(episode.random_seed)

        # BUG #8 fix (2026-06-10): Initialize last_valid_command with a
        # situational fallback based on initial state (maintain heading and
        # minimal speed) instead of hardcoded zero rudder + 0.5 prop.
        init_speed = episode.ownship_initial_state.speed()
        last_valid_command = ControlCommand(
            rudder=0.0,
            propeller=max(0.2, min(0.4, init_speed / 15.0)),
            source="init_situational",
        )

        # Initialize
        own_state = episode.ownship_initial_state
        own_params = episode.ownship_params
        targets = episode.targets
        env_seq = episode.environment_sequence

        # Build MMG model for ownship
        mmg = MMG3DOFModel({
            "length": own_params.length,
            "beam": own_params.beam,
            "draught": own_params.draught,
            "mass": own_params.mass,
            "Iz": own_params.Iz,
            "max_rudder": own_params.max_rudder,
            "max_rudder_rate": own_params.max_rudder_rate,
            "max_yaw_rate": own_params.max_yaw_rate,
            "min_turning_radius": own_params.min_turning_radius_multiplier,
        })

        # Pre-compute target trajectories
        target_trajectories = []
        for tgt in targets:
            traj = propagate_target(tgt, episode.dt, episode.duration)
            target_trajectories.append(traj)

        # BUG #7 fix (2026-06-10): Remove +1 to avoid wasted iteration.
        # The break guard `t >= episode.duration` already handles termination.
        n_steps = int(episode.duration / episode.dt)

        # History
        state_history = [own_state]
        command_history = []
        target_histories = [[] for _ in targets]
        runtime_stats = []
        controller_failure_count = 0

        # Run loop
        for k in range(n_steps):
            t = k * episode.dt
            if t >= episode.duration:
                break

            # Get environment at current time
            if env_seq:
                env = env_seq[min(k, len(env_seq) - 1)]
            else:
                env = EnvironmentState()

            # Get current target states (possibly noisy)
            current_targets = []
            for i, tgt in enumerate(targets):
                true_state = target_trajectories[i][min(k, len(target_trajectories[i]) - 1)]
                noisy_state = get_noisy_target_state(true_state, tgt, t, rng)

                # ── BUG #2 fix (2026-06-10): AIS dropout must NOT fall back to
                #    true_state — that defeats the purpose of AIS robustness
                #    testing.  Use last known position with inflated covariance.
                # ── BUG #3 fix (2026-06-10): covariance must reflect actual AIS
                #    quality (delay / dropout) via update_target_covariance().
                dropout_occurred = noisy_state is None
                ais_delay = tgt.ais_quality.get("delay", 0.0)

                if dropout_occurred:
                    # Use last known observed state (from previous steps in
                    # target_histories) if available; otherwise fall back to
                    # the initial target state (which is what the controller
                    # would have received before dropout began).
                    if i < len(target_histories) and len(target_histories[i]) > 0:
                        last_known = target_histories[i][-1]
                    else:
                        last_known = true_state
                    observed_state = last_known
                else:
                    observed_state = noisy_state

                # Compute properly inflated covariance
                # Temporarily update target covariance on a copy so
                # update_target_covariance sees the correct delay/dropout.
                tgt_copy = TargetShip(
                    mmsi=tgt.mmsi,
                    state=observed_state,
                    length=tgt.length, beam=tgt.beam, draught=tgt.draught,
                    vessel_type=tgt.vessel_type, nav_status=tgt.nav_status,
                    covariance=tgt.covariance.copy() if tgt.covariance is not None else np.eye(2) * 100.0,
                    ais_quality=tgt.ais_quality,
                )
                inflated_cov = update_target_covariance(tgt_copy, ais_delay, dropout_occurred)

                observed = TargetShip(
                    mmsi=tgt.mmsi,
                    state=observed_state,
                    length=tgt.length,
                    beam=tgt.beam,
                    draught=tgt.draught,
                    vessel_type=tgt.vessel_type,
                    nav_status=tgt.nav_status,
                    covariance=inflated_cov,
                    ais_quality=tgt.ais_quality,
                )
                current_targets.append(observed)
                target_histories[i].append(true_state)

            # Compute control
            t_start = time.perf_counter()
            try:
                command = controller.compute_control(
                    ownstate=own_state,
                    ownship_params=own_params,
                    target_ships=current_targets,
                    environment=env,
                    rule_context={
                        "enc_layer": episode.enc_layer,
                        "vts_zone": episode.vts_zone,
                        "coast_pilot_rules": episode.coast_pilot_rules,
                    },
                )
                # Guard against NaN/Inf in control output
                if np.isnan(command.rudder) or np.isinf(command.rudder):
                    command.rudder = 0.0
                if np.isnan(command.propeller) or np.isinf(command.propeller):
                    command.propeller = 0.2
                last_valid_command = command  # remember for situational fallback
            except Exception as e:
                logger.warning(f"Controller failed at t={t:.1f}s: {e}")
                # Use last valid command instead of blind hardcoded defaults;
                # this preserves situational awareness (e.g., if the ship was
                # already turning to avoid an obstacle).
                command = ControlCommand(
                    rudder=last_valid_command.rudder,
                    propeller=max(0.2, last_valid_command.propeller * 0.5),
                    source="error",
                    fallback_active=True,
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "controller_failed": True,
                        "failure_time": t,
                    },
                )
                # Propagate failure to result tracking
                controller_failure_count += 1
            runtime = time.perf_counter() - t_start
            runtime_stats.append(runtime)

            # Step dynamics with disturbance force from controller
            # The controller computes physics-enhanced disturbances (shallow water,
            # bank effect, ship interaction, wind/current) and stores them in
            # command metadata so the simulator applies the same forces that
            # the MPC predicted internally — eliminating model mismatch.
            disturbance_raw = command.metadata.get("total_disturbance")
            if disturbance_raw is not None and len(disturbance_raw) == 3:
                disturbance = np.array(disturbance_raw, dtype=float)
                if np.any(np.isnan(disturbance)) or np.any(np.isinf(disturbance)):
                    disturbance = np.zeros(3)
            else:
                disturbance = np.zeros(3)
            own_state = mmg.step(own_state, command, env, episode.dt, disturbance)

            # Early termination on numerical blow-up
            if any(np.isnan(getattr(own_state, attr)) or np.isinf(getattr(own_state, attr))
                   for attr in ("x", "y", "u", "v", "r", "psi")):
                logger.warning(f"Numerical blow-up at t={t:.1f}s — terminating episode early")
                own_state = VesselState(
                    x=state_history[-1].x, y=state_history[-1].y,
                    psi=state_history[-1].psi,
                    u=0.0, v=0.0, r=0.0,
                    timestamp=t,
                )
                state_history.append(own_state)
                command_history.append(command)
                break

            state_history.append(own_state)
            command_history.append(command)

        # Aggregate per-step metadata from ControlCommand objects
        agg = _aggregate_step_metadata(command_history, controller_failure_count)

        # Build result
        result = EpisodeResult(
            episode_id=f"{episode.scenario_id}_seed{episode.random_seed}",
            scenario_id=episode.scenario_id,
            method=method,
            seed=episode.random_seed,
            state_history=state_history,
            command_history=command_history,
            target_histories=target_histories,
            runtime_stats={
                "mean_runtime": float(np.mean(runtime_stats)) if runtime_stats else 0.0,
                "p95_runtime": float(np.percentile(runtime_stats, 95)) if runtime_stats else 0.0,
                "max_runtime": float(np.max(runtime_stats)) if runtime_stats else 0.0,
                "deadline_misses": sum(1 for r in runtime_stats if r > self.runtime_deadline),
            },
            metadata=agg,
        )
        return result


def _aggregate_step_metadata(
    command_history: list,
    controller_failure_count: int = 0,
) -> dict:
    """Aggregate per-step ControlCommand.metadata into episode-level metadata.

    This bridges the gap between per-step controller outputs and the
    episode-level metadata expected by failure_detector.py and metrics.py.

    Args:
        command_history: List of ControlCommand objects from the episode.
        controller_failure_count: Number of steps where the controller raised.

    Returns:
        dict with aggregated metadata keys.
    """
    n_steps = max(len(command_history), 1)

    # -- mpc_info & feasibility --
    any_mpc_infeasible = False
    mpc_info_agg: dict = {}
    for cmd in command_history:
        mi = cmd.metadata.get("mpc_info", {})
        if mi and isinstance(mi, dict):
            if not mpc_info_agg:
                mpc_info_agg = mi
        if not cmd.metadata.get("mpc_feasible", True):
            any_mpc_infeasible = True

    # -- cbf_info --
    any_over_intervention = False
    cbf_info_agg: dict = {}
    for cmd in command_history:
        ci = cmd.metadata.get("cbf_info", {})
        if ci and isinstance(ci, dict):
            if not cbf_info_agg:
                cbf_info_agg = ci
            if ci.get("over_intervention", False):
                any_over_intervention = True

    # -- fallback --
    any_fallback = any(cmd.metadata.get("fallback_active", False) for cmd in command_history)
    fallback_levels = [cmd.metadata.get("fallback_level", "none") for cmd in command_history
                       if cmd.metadata.get("fallback_active")]

    # -- rule_assessment --
    rule_assessment = None
    for cmd in reversed(command_history):
        ra = cmd.metadata.get("rule_assessment")
        if ra is not None:
            rule_assessment = ra
            break

    # -- domain_results --
    domain_results: list = []
    for cmd in command_history:
        dr = cmd.metadata.get("domain_results")
        if dr and isinstance(dr, list):
            domain_results.extend(dr)

    # -- chance constraint violations --
    cc_violations = 0
    for cmd in command_history:
        cc = cmd.metadata.get("chance_constraint_results", [])
        if cc and isinstance(cc, list):
            cc_violations += sum(1 for r in cc if not r.get("constraint_satisfied", True))

    # -- tube components (aggregate mean) --
    tube_components: dict = {}
    tube_count = 0
    for cmd in command_history:
        tc = cmd.metadata.get("tube_components")
        if tc and isinstance(tc, dict):
            for k, v in tc.items():
                if isinstance(v, (int, float)):
                    tube_components[k] = tube_components.get(k, 0.0) + v
            tube_count += 1
    if tube_count > 0:
        tube_components = {k: v / tube_count for k, v in tube_components.items()}

    # -- safety_distance_breakdown (from the last step, where it's most relaxed) --
    safety_breakdown: list = []
    for cmd in reversed(command_history):
        sbd = cmd.metadata.get("safety_distance_breakdown")
        if sbd:
            safety_breakdown = sbd
            break

    # ── Accumulated episode-level counters ──────────────────────────────
    # The controller accumulates these across steps and stores them in each
    # command's metadata. The LAST command has the final totals.
    last_cmd_meta = command_history[-1].metadata if command_history else {}
    cbf_intervention_total = last_cmd_meta.get("cbf_intervention_total", 0)
    cbf_infeasible_total = last_cmd_meta.get("cbf_infeasible_total", 0)
    fallback_step_total = last_cmd_meta.get("fallback_step_total", 0)
    episode_has_fallback = last_cmd_meta.get("episode_has_fallback", any_fallback)

    # Also count from command history for robustness (if controller doesn't track)
    if cbf_intervention_total == 0:
        cbf_intervention_total = sum(
            1 for cmd in command_history
            if cmd.metadata.get("cbf_info", {}).get("cbf_active", False)
        )
    if cbf_infeasible_total == 0:
        cbf_infeasible_total = sum(
            1 for cmd in command_history
            if cmd.metadata.get("cbf_info", {}).get("cbf_infeasible", False)
        )
    if fallback_step_total == 0:
        fallback_step_total = sum(
            1 for cmd in command_history
            if cmd.metadata.get("fallback_active", False)
        )

    # Extract timing breakdown from last step
    timing_breakdown = last_cmd_meta.get("timing_breakdown", {})

    return {
        "controller_failure_count": controller_failure_count,
        "controller_failure_rate": controller_failure_count / n_steps,
        "mpc_info": mpc_info_agg,
        "mpc_feasible": not any_mpc_infeasible,
        "cbf_info": {**cbf_info_agg, "over_intervention": any_over_intervention},
        "cbf_intervention_total": cbf_intervention_total,
        "cbf_infeasible_total": cbf_infeasible_total,
        "fallback_active": any_fallback,
        "fallback_step_total": fallback_step_total,
        "episode_has_fallback": episode_has_fallback,
        "fallback_level": max(fallback_levels, key=lambda x: _FALLBACK_LEVEL_ORDER.get(x, 0)) if fallback_levels else "NONE",
        "rule_assessment": rule_assessment,
        "domain_results": domain_results,
        "chance_constraint_violations": cc_violations,
        "tube_components": tube_components,
        "safety_distance_breakdown": safety_breakdown,
        "timing_breakdown": timing_breakdown,
    }
