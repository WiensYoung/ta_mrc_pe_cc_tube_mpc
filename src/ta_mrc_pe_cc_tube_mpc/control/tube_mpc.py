"""Tube-MPC implementation.

Combines nominal MPC with a tube boundary that robustifies constraints
against disturbances. The tube radius rho_tube expands the safety margin
in the chance constraint formulation.
"""

from typing import Callable, Optional

import numpy as np

from ..physics.tube_boundary import compute_tube_radius
from ..risk.intent_predictor import predict_target_trajectories
from ..types import ControlCommand, EnvironmentState, TargetShip, VesselParams, VesselState
from .mpc_problem import MPCProblem, sampling_based_mpc


class TubeMPC:
    """Tube-based Model Predictive Control for ship collision avoidance.

    Wraps the nominal MPC with tube radius augmentation of safety constraints.
    """

    def __init__(
        self,
        dynamics: Callable,
        vessel_params: VesselParams,
        horizon: int = 20,
        dt: float = 0.5,
        config: dict = None,
    ):
        """
        Args:
            dynamics: State transition function.
            vessel_params: Ownship parameters.
            horizon: MPC prediction horizon steps.
            dt: Time step [s].
            config: Controller configuration dict.
        """
        self.dynamics = dynamics
        self.vessel_params = vessel_params
        self.N = horizon
        self.dt = dt

        cfg = config or {}
        mpc_cfg = dict(cfg.get("mpc", cfg))
        mpc_cfg["vessel_length"] = vessel_params.length
        # Propagate vessel physics so the CasADi surrogate uses matching
        # mass, inertia, and draught instead of hardcoded defaults.
        mpc_cfg.setdefault("mass", vessel_params.mass)
        mpc_cfg.setdefault("Iz", vessel_params.Iz)
        mpc_cfg.setdefault("vessel_draught", vessel_params.draught)
        self.mpc = MPCProblem(
            dynamics=dynamics, horizon=horizon, dt=dt, config=mpc_cfg,
        )

        self.tube_cfg = cfg.get("tube", {})
        self.rho_0 = self.tube_cfg.get("rho_0", 5.0)
        self.rho_maneuver = self.tube_cfg.get("rho_maneuver", 2.0)
        self.use_adaptive_tube = self.tube_cfg.get("use_adaptive", True)

        # Backend selection: unified via mpc.backend (or legacy use_sampling)
        mpc_backend = cfg.get("mpc", {}).get("backend",
                       cfg.get("mpc", {}).get("solver", ""))
        # BUG #8 fix (2026-06-10): normalise empty / unknown backend strings
        # to "casadi" so ``self.backend`` is always set.
        mpc_backend = str(mpc_backend).lower().strip()
        if cfg.get("use_sampling"):
            self.backend = "sampling"
        elif mpc_backend in ("casadi", "jax", "scipy", "sampling"):
            self.backend = mpc_backend
        else:
            self.backend = "casadi"  # default
        # Legacy compat flag
        self.use_sampling = (self.backend == "sampling")
        self.sampling_cfg = cfg.get("sampling_mpc", {})

    def _get_exploration_std(self, index: int, default: float) -> float:
        """Safely extract exploration_std element, handling both list and scalar config."""
        val = self.sampling_cfg.get("exploration_std", [5.0, 0.5])
        if isinstance(val, (list, tuple)) and len(val) > index:
            return float(val[index])
        if isinstance(val, (int, float)):
            return float(val)
        return default

    def _predict_targets(self, targets: list[TargetShip]) -> list[np.ndarray]:
        """Predict target positions via constant velocity over the horizon.

        Backward-compatible wrapper. For multi-modal prediction, use
        ``predict_target_trajectories`` from ``intent_predictor`` directly.
        """
        n_steps = self.N + 1
        if not targets:
            return [np.empty((0, 2)) for _ in range(n_steps)]
        preds = []
        for k in range(n_steps):
            t = k * self.dt
            positions = np.zeros((len(targets), 2))
            for i, tgt in enumerate(targets):
                c = np.cos(tgt.state.psi)
                s = np.sin(tgt.state.psi)
                # Full body-to-world: include sway velocity v
                positions[i, 0] = tgt.state.x + (tgt.state.u * c - tgt.state.v * s) * t
                positions[i, 1] = tgt.state.y + (tgt.state.u * s + tgt.state.v * c) * t
            preds.append(positions)
        return preds

    def compute_control(
        self,
        own_state: VesselState,
        own_Sigma: np.ndarray,
        targets: list[TargetShip],
        env: EnvironmentState,
        reference_trajectory: list[VesselState],
        d_safe_eff_list: list[float],
        previous_command: ControlCommand = None,
        rng: np.random.Generator = None,
        forbidden_zones: list = None,
        hard_constraints: list = None,
        enc_layer=None,
        rho_tube_override: float = None,
        enable_chance_constraint: bool = True,
    ) -> tuple[Optional[ControlCommand], dict]:
        """Compute tube-MPC control action.

        Args:
            own_state: Current ownship state.
            own_Sigma: Ownship position covariance (2x2).
            targets: Target ships.
            env: Environment state.
            reference_trajectory: Reference trajectory for N+1 steps.
            d_safe_eff_list: Effective safe distances [m] per target.
            previous_command: Previous control command.
            rng: Random state for sampling-based MPC.
            forbidden_zones: List of (geometry, label) tuples from rule engine.
            hard_constraints: List of hard constraint dicts from rule engine.
            enc_layer: Optional ENC layer for navigability checks.
            rho_tube_override: Pre-computed tube radius (skips internal computation
                               when set). Pass None for internal computation.
            enable_chance_constraint: When False, disables the chance-constraint
                               margin (kappa*sqrt(lambda_max)) in the MPC safety
                               constraints. Used for ablation A9.

        Returns:
            (control_command, metadata) or (None, metadata) if infeasible.
        """
        # Compute tube radius (use override if caller already computed it)
        if rho_tube_override is not None:
            rho_tube = rho_tube_override
            tube_components = {"total": rho_tube, "source": "caller_override"}
        else:
            tube_components = compute_tube_radius(
                own_state=own_state,
                own_length=self.vessel_params.length,
                own_beam=self.vessel_params.beam,
                own_draught=self.vessel_params.draught,
                targets=targets,
                env=env,
                rho_0=self.rho_0,
                rho_maneuver=self.rho_maneuver,
                config=self.tube_cfg,
                use_adaptive=self.use_adaptive_tube,
            )
            rho_tube = tube_components["radius"]

        # Extract target covariances for chance-constrained formulation.
        # Default: std=5m (consistent with chance_constraints.py).
        # Old value was 100.0 (std=10m) which overinflated the margin.
        target_covariances = [t.covariance if t.covariance is not None else np.eye(2) * 25.0
                             for t in targets]

        # ── Safe-distance stack (additive, conservative by design) ───────
        #
        # The total safe distance used in MPC constraints is:
        #
        #   d_safe_total = d_safe_eff                    ← DynamicShipDomain
        #                + rho_tube                       ← compute_tube_radius
        #                + kappa · sqrt(λ_max(Σ_rel))     ← chance constraint
        #                                                  (added inside MPCProblem.solve)
        #
        # Layer 1 — d_safe_eff (geometry / kinematics):
        #   d_base + Δd_speed + Δd_size + Δd_rule + Δd_unc
        #   + Δd_env + Δd_bank + Δd_ship + Δd_waterway
        #   → scalar, additive, per-target, NO direction dependence
        #
        # Layer 2 — rho_tube (robust disturbance buffer, Section 8.6):
        #   rho_0 + rho_AIS + rho_current + rho_wind
        #   + rho_shallow + rho_bank + rho_ship + rho_maneuver
        #   → conservative sum-of-bounds (assumes worst-case alignment)
        #
        # Layer 3 — chance margin (probabilistic, Section 9):
        #   κ_ε · sqrt(λ_max(Σ_j + Σ_i))
        #   → chi-squared confidence ellipse, 2-DOF
        #   → added inside MPCProblem.solve() via d_safe_cc_list
        #
        # IMPORTANT: Δd_bank / Δd_ship (geometric proximity expansion) and
        # ρ_bank / ρ_ship (disturbance-uncertainty buffer) are DISTINCT
        # quantities — they model different physical phenomena and are
        # intentionally additive.  See ASSUMPTIONS.md Section 7 and 13.
        d_safe_augmented = [d + rho_tube for d in d_safe_eff_list]

        # Predict target positions over horizon (multi-modal intent prediction)
        target_positions = predict_target_trajectories(
            targets=targets,
            own_state=own_state,
            horizon_steps=self.N + 1,
            dt=self.dt,
        )

        # When chance constraints are disabled (ablation A9), set epsilon=1.0
        # so that kappa_epsilon = sqrt(chi2.ppf(1-1.0, df=2)) = sqrt(chi2.ppf(0,2)) = 0,
        # effectively removing the stochastic safety margin from the MPC optimization.
        epsilon = self.mpc.epsilon_default if enable_chance_constraint else 1.0

        # Solve MPC with chance-constrained safety margins
        if self.backend == "sampling":
            cmd, info = sampling_based_mpc(
                current_state=own_state,
                dynamics=self.dynamics,
                reference_trajectory=reference_trajectory,
                target_positions=target_positions,
                d_safe_eff_list=d_safe_augmented,
                env=env,
                n_samples=self.sampling_cfg.get("n_samples", 500),
                rudder_std=self._get_exploration_std(0, 5.0),
                prop_std=self._get_exploration_std(1, 0.5),
                horizon=self.N,
                dt=self.dt,
                rng=rng,
                target_covariances=target_covariances,
                own_covariance=own_Sigma,
                rho_tube=rho_tube,
                epsilon=epsilon,
                enc_layer=enc_layer,
            )
        else:
            cmd, info = self.mpc.solve(
                current_state=own_state,
                reference_trajectory=reference_trajectory,
                target_positions=target_positions,
                d_safe_eff_list=d_safe_augmented,
                env=env,
                previous_command=previous_command,
                target_covariances=target_covariances,
                own_covariance=own_Sigma,
                rho_tube=rho_tube,
                epsilon=epsilon,
                forbidden_zones=forbidden_zones,
                hard_constraints=hard_constraints,
                enc_layer=enc_layer,
            )

        metadata = {
            "backend": info.get("actual_backend", self.backend),
            "requested_backend": info.get("requested_backend", self.backend),
            "actual_backend": info.get("actual_backend", self.backend),
            "degraded_backend": info.get("degraded_backend"),
            "fallback_reason": info.get("fallback_reason"),
            "solver_status": info.get("success", False),
            "objective_value": info.get("cost", info.get("best_cost", float("inf"))),
            "infeasible": not info.get("success", False),
            "fallback_triggered": False,
            "cbf_triggered": False,
            "rule_context_used": info.get("rule_context_used", False),
            "enc_used": info.get("enc_used", enc_layer is not None),
            "vts_used": False,
            "chance_constraint_used": enable_chance_constraint,
            "tube_radius": rho_tube,
            "tube_components": tube_components,
            "safe_distance_components": d_safe_augmented,
            "warnings": info.get("warnings", []),
            "mpc_info": info,
        }

        return cmd, metadata

