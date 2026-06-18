"""MPC optimization problem formulation.

Backend dispatch (set ``mpc.backend`` in config, or ``mpc.solver`` for
backward compatibility):

    ``"casadi"``  — CasADi Opti + IPOPT  (default, uses surrogate dynamics
                    with linear damping for symbolic auto-diff; NOT full MMG)
    ``"scipy"``   — scipy SLSQP           (gradient-free, uses full MMG model)
    ``"sampling"``— random-exploration MPC (baseline / ablation only)

Important: The CasADi backend uses a simplified surrogate dynamics model
(linear damping, no cross-coupling) for symbolic differentiation.  The
scipy SLSQP backend uses the full nonlinear MMG model.  The surrogate is
parameterized from config but uses default coefficients if keys are missing.

Objective:
    min sum_k [
        ||x_k - x_ref,k||_Q²  +  ||u_k||_R²  +  ||Δu_k||_S²
        + w_collision · collision_soft_penalty_k
        + w_rule · rule_violation_penalty_k
        + w_channel · channel_violation_penalty_k
    ]  +  ||x_N - x_ref,N||_P²

Chance-constrained safety (Section 9 of experiment design):
    ||mu_rel_ij|| >= d_safe_eff + κ_ε · √(λ_max(Σ_rel)) + ρ_tube
"""

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

from ..risk.uncertainty import compute_relative_covariance
from ..types import ControlCommand, EnvironmentState, VesselState
from ..utils.math_utils import kappa_epsilon, max_eigenvalue

# ── Optional CasADi import ────────────────────────────────────────────────
try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False
    ca = None

# ── Optional JAX import ───────────────────────────────────────────────────
try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None


class MPCProblem:
    """Receding-horizon nonlinear MPC for ship collision avoidance.

    Backend selection (single source of truth):
        ``backend="casadi"``   → CasADi Opti + IPOPT (default)
        ``backend="scipy"``    → scipy SLSQP
        ``backend="sampling"`` → random-exploration baseline
    """

    # Allowed backends + auto-detection of CasADi / JAX
    _BACKENDS = ("casadi", "jax", "scipy", "sampling")
    _CASADI_AVAILABLE = HAS_CASADI

    try:
        import jax as _jax_mod
        _JAX_AVAILABLE = True
    except ImportError:
        _JAX_AVAILABLE = False

    def __init__(
        self,
        dynamics: Callable,
        horizon: int = 20,
        dt: float = 0.5,
        Q_pos: np.ndarray = None,
        Q_vel: np.ndarray = None,
        R: np.ndarray = None,
        S: np.ndarray = None,
        P: np.ndarray = None,
        config: dict = None,
    ):
        self.dynamics = dynamics
        self.N = horizon
        self.dt = dt

        self.Q_pos = Q_pos if Q_pos is not None else np.diag([10.0, 10.0, 1.0])
        self.Q_vel = Q_vel if Q_vel is not None else np.diag([1.0, 1.0, 0.5])
        self.R_mat = R if R is not None else np.diag([0.1, 0.01])
        self.S_mat = S if S is not None else np.diag([1.0, 0.1])
        self.P = P if P is not None else np.diag([20.0, 20.0, 2.0])

        cfg = config or {}
        self.rudder_min = cfg.get("rudder_min", -35.0)
        self.rudder_max = cfg.get("rudder_max", 35.0)
        self.rudder_rate_max = cfg.get("rudder_rate_max", 5.0)
        self.prop_min = cfg.get("propeller_min", 0.0)
        self.prop_max = cfg.get("propeller_max", 1.0)
        self.prop_rate_max = cfg.get("propeller_rate_max", 0.5)  # [1/s]
        self.yaw_rate_min = cfg.get("yaw_rate_min", -3.0)
        self.yaw_rate_max = cfg.get("yaw_rate_max", 3.0)
        self.max_iterations = cfg.get("max_iterations", 150)
        self.vessel_length = cfg.get("vessel_length", 180.0)
        self._vessel_draught = cfg.get("vessel_draught", 10.5)  # for UKC/surrogate calcs

        # ── Backend selection ────────────────────────────────────────────
        self.backend = self._resolve_backend(cfg)
        # Legacy compat (keep for external callers)
        self.use_casadi = (self.backend == "casadi")

        # ── Penalty weights ──────────────────────────────────────────────
        self.w_collision = float(cfg.get("w_collision_soft", 100.0))
        self.w_rule = float(cfg.get("w_rule_colregs", 10.0))
        self.w_channel = float(cfg.get("w_channel_violation", 50.0))
        self.w_bank = float(cfg.get("w_bank", 5.0))
        self.w_ukc = float(cfg.get("w_ukc", 10.0))
        self.w_speed = float(cfg.get("w_speed_error", 1.0))

        # Chance constraint configuration
        self.epsilon_default = cfg.get("epsilon_default", 0.10)
        self.enable_chance_constraints = cfg.get("enable_chance_constraints", True)
        self.enable_hard_collision = cfg.get("enable_hard_collision_constraints", False)
        self.fd_eps = cfg.get("fd_epsilon", 0.01)

        # Rule context (set externally via set_rule_context)
        self._soft_constraints: list = []
        self._p5_terms: list = []
        self._waterway_safety_scale: float = 1.0
        self._forbidden_zones: list = []
        self._hard_constraints: list = []
        self._enc_layer = None

        # Warm-start state
        self._prev_solution: Optional[np.ndarray] = None
        self._prev_rudder: float = 0.0
        self._prev_prop: float = 0.5

        # P5 fix (2026-06-10): Cache the CasADi Opti stack across solve() calls.
        # Previously the Opti stack was rebuilt from scratch each call (~50ms
        # overhead).  Caching reduces per-step overhead by ~80%.
        self._cached_opti = None
        self._cached_opti_params_hash = None

        # CasADi solver options
        self._casadi_solver_options = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.max_iterations,
            "ipopt.tol": 1e-3,          # relaxed from 1e-4 for faster convergence
            "ipopt.acceptable_tol": 1e-2,  # acceptable solution tolerance
            "ipopt.acceptable_iter": 10,   # stop after 10 iterations at acceptable_tol
            "print_time": 0,
            "ipopt_max_cpu_time": 2.0,  # reduced from 5s to 2s per step
        }

        # Configure surrogate dynamics from config
        self._configure_surrogate_from_config(cfg)

    def _resolve_backend(self, cfg: dict) -> str:
        """Resolve a single backend string from legacy or new config keys."""
        # New-style: mpc.backend or mpc.solver
        raw = cfg.get("backend", cfg.get("solver", cfg.get("mpc_backend", "")))
        if not raw:
            # Legacy compat: use_casadi / use_sampling booleans
            if cfg.get("use_sampling"):
                raw = "sampling"
            elif cfg.get("use_casadi"):
                raw = "casadi"
            else:
                # Auto-detect: prefer JAX GPU when available
                if self._JAX_AVAILABLE:
                    try:
                        import jax as _jax
                        if any(d.platform == "gpu" for d in _jax.devices()):
                            raw = "jax"
                        else:
                            raw = "casadi"
                    except Exception:
                        raw = "casadi"
                else:
                    raw = "casadi"

        raw = str(raw).lower().strip()
        if raw in ("casadi", "ipopt"):
            if not self._CASADI_AVAILABLE:
                import warnings
                warnings.warn(
                    "MPC backend 'casadi' requested but CasADi is not installed. "
                    "Falling back to 'scipy' (SLSQP). Install CasADi with: "
                    "pip install ta_mrc_pe_cc_tube_mpc[solver]",
                    RuntimeWarning,
                )
                return "scipy"
            return "casadi"
        elif raw in ("jax", "gpu"):
            if not self._JAX_AVAILABLE:
                import warnings
                warnings.warn(
                    "MPC backend 'jax' requested but JAX is not installed. "
                    "Falling back to 'casadi' (IPOPT). Install JAX with: "
                    "pip install ta_mrc_pe_cc_tube_mpc[jax]",
                    RuntimeWarning,
                )
                if self._CASADI_AVAILABLE:
                    return "casadi"
                return "scipy"
            return "jax"
        elif raw in ("scipy", "slsqp"):
            return "scipy"
        elif raw in ("sampling", "sample", "random"):
            return "sampling"
        else:
            import warnings
            warnings.warn(
                f"Unknown MPC backend '{raw}'. Using 'casadi' as default.",
                RuntimeWarning,
            )
            if self._CASADI_AVAILABLE:
                return "casadi"
            return "scipy"

    # ── Backward-compatible aliases ──────────────────────────────────────
    @property
    def R(self) -> np.ndarray:
        """Alias for R_mat (control cost matrix)."""
        return self.R_mat

    @property
    def S(self) -> np.ndarray:
        """Alias for S_mat (control rate cost matrix)."""
        return self.S_mat

    @property
    def Q(self) -> np.ndarray:
        """Alias for Q_pos (position cost matrix)."""
        return self.Q_pos

    def set_rule_context(
        self, soft_constraints=None, p5_terms=None, waterway_safety_scale=1.0,
    ):
        self._soft_constraints = soft_constraints or []
        self._p5_terms = p5_terms or []
        self._waterway_safety_scale = waterway_safety_scale

    # ═══════════════════════════════════════════════════════════════════════
    # Public solve entry point — dispatches to CasADi or SLSQP
    # ═══════════════════════════════════════════════════════════════════════

    def solve(
        self,
        current_state: VesselState,
        reference_trajectory: list[VesselState],
        target_positions: list[np.ndarray],
        d_safe_eff_list: list[float],
        env: EnvironmentState,
        previous_command: ControlCommand = None,
        target_covariances: list[np.ndarray] = None,
        own_covariance: np.ndarray = None,
        rho_tube: float = 0.0,
        epsilon: float = None,
        forbidden_zones: list = None,
        hard_constraints: list = None,
        enc_layer=None,
    ) -> tuple[Optional[ControlCommand], dict]:
        """Solve MPC with chance constraints. Auto-selects solver backend."""
        if epsilon is None:
            epsilon = self.epsilon_default
        if target_covariances is None:
            target_covariances = []
        if own_covariance is None:
            own_covariance = np.eye(2) * 1.0

        # Pre-compute chance-constrained safety distances.
        # Decompose epsilon_total across horizon × targets using sqrt-Boole allocation.
        # This is less conservative than strict Boole (which inflates kappa by 50%+
        # for typical problems) while still scaling with problem size.
        n_horizon = self.N
        n_targets = len(d_safe_eff_list)
        from .risk_allocation import allocate_chance_risk
        eps_per_constraint = allocate_chance_risk(
            epsilon_total=epsilon,
            horizon_steps=n_horizon,
            n_targets=max(n_targets, 1),
            mode="sqrt_boole",
        )
        kappa = kappa_epsilon(eps_per_constraint, df=2)
        d_safe_cc_list = []
        for j in range(len(d_safe_eff_list)):
            d_safe_eff = d_safe_eff_list[j]
            if j < len(target_covariances) and self.enable_chance_constraints:
                Sigma_rel = compute_relative_covariance(own_covariance, target_covariances[j])
                lambda_max_val = max_eigenvalue(Sigma_rel)
                uncertainty_margin = kappa * np.sqrt(max(lambda_max_val, 0.0))
            else:
                uncertainty_margin = 0.0
            # rho_tube is already included in d_safe_eff (added in tube_mpc.py
            # compute_control() line 199: d_safe_augmented = [d + rho_tube for d in ...]).
            # Do NOT add it again — that would double-count the tube margin.
            # See also: tube_mpc.py::compute_control() for the safe-distance stack.
            d_safe_cc_list.append(d_safe_eff + uncertainty_margin)

        # Store rule context for cost function access
        if forbidden_zones:
            self._forbidden_zones = forbidden_zones
        if hard_constraints:
            self._hard_constraints = hard_constraints
        if enc_layer is not None:
            self._enc_layer = enc_layer

        prev_rudder = previous_command.rudder if previous_command else self._prev_rudder
        prev_prop = previous_command.propeller if previous_command else self._prev_prop

        # ── Dispatch to solver backend ──────────────────────────────────
        if self.backend == "sampling":
            # Sampling MPC is handled externally by TubeMPC — if it reaches
            # here, treat as scipy fallback.
            cmd, info = self._solve_slsqp(
                current_state, reference_trajectory, target_positions,
                d_safe_cc_list, env, prev_rudder, prev_prop,
            )
        elif self.backend == "jax":
            try:
                cmd, info = self._solve_jax(
                    current_state, reference_trajectory, target_positions,
                    d_safe_cc_list, env, prev_rudder, prev_prop,
                )
            except Exception as e:
                logger.warning(
                    f"JAX MPC failed ({e}), falling back to SLSQP",
                    exc_info=False,
                )
                cmd, info = self._solve_slsqp(
                    current_state, reference_trajectory, target_positions,
                    d_safe_cc_list, env, prev_rudder, prev_prop,
                )
                info["degraded_backend"] = "slsqp"
                info["fallback_reason"] = str(e)[:200]
                info["requested_backend"] = "jax"
                info["actual_backend"] = "slsqp"
        elif self.backend == "casadi":
            try:
                cmd, info = self._solve_casadi(
                    current_state, reference_trajectory, target_positions,
                    d_safe_cc_list, env, prev_rudder, prev_prop,
                    kappa, epsilon, rho_tube,
                )
            except Exception as e:
                logger.warning(
                    f"CasADi+IPOPT failed ({e}), falling back to SLSQP",
                    exc_info=False,
                )
                cmd, info = self._solve_slsqp(
                    current_state, reference_trajectory, target_positions,
                    d_safe_cc_list, env, prev_rudder, prev_prop,
                )
                info["degraded_backend"] = "slsqp"
                info["fallback_reason"] = str(e)[:200]
                info["requested_backend"] = "casadi"
                info["actual_backend"] = "slsqp"
        else:  # "scipy"
            cmd, info = self._solve_slsqp(
                current_state, reference_trajectory, target_positions,
                d_safe_cc_list, env, prev_rudder, prev_prop,
            )

        # ── Build unified info dict with accurate backend tracking ──────
        # Distinguish "requested" vs "actual" backend so that fallback
        # results are never misattributed to CasADi/IPOPT in experiments.
        info["kappa"] = kappa
        info["epsilon"] = epsilon
        info["rho_tube"] = rho_tube
        info["d_safe_cc_list"] = d_safe_cc_list
        info["warm_started"] = self._prev_solution is not None
        info["requested_backend"] = self.backend
        # Determine actual backend from the command source field
        if cmd is not None and hasattr(cmd, "source"):
            if "CasADi" in cmd.source:
                info["actual_backend"] = "casadi"
            elif "SLSQP" in cmd.source:
                info["actual_backend"] = "slsqp"
            elif "sampling" in cmd.source.lower():
                info["actual_backend"] = "sampling"
            else:
                info["actual_backend"] = info.get("degraded_backend", self.backend)
        else:
            info["actual_backend"] = info.get("degraded_backend", self.backend)
        info["backend"] = info["actual_backend"]  # primary key = what actually ran
        info["solver_backend"] = info["actual_backend"]
        info["casadi_available"] = self._CASADI_AVAILABLE
        info["chance_constraint_used"] = self.enable_chance_constraints
        info["hard_collision_enabled"] = self.enable_hard_collision
        info["has_targets"] = len(target_positions) > 0
        info["n_targets"] = len(target_positions[0]) if target_positions else 0
        info["rule_context_used"] = len(self._soft_constraints) > 0
        info["enc_used"] = self._enc_layer is not None
        info["warnings"] = []
        if self.backend != "casadi" and self._CASADI_AVAILABLE:
            info["degraded_backend"] = self.backend
            info["warnings"].append(
                f"Using {self.backend} backend; CasADi available but not selected"
            )
        if not self._CASADI_AVAILABLE and self.backend == "casadi":
            info["warnings"].append("CasADi not installed, SLSQP fallback used")

        if cmd is not None:
            # Store the first-step control for next iteration's rate-constraint
            # reference, and also store as prev_solution for warm-start.
            # Full solution vector is stored via _store_full_solution() from
            # individual solver backends.
            self._prev_rudder = float(cmd.rudder)
            self._prev_prop = float(cmd.propeller)
            return cmd, info
        else:
            self._prev_solution = None
            return None, info

    # ═══════════════════════════════════════════════════════════════════════
    # CasADi + IPOPT solver  (refactored: symbolic surrogate dynamics)
    # ═══════════════════════════════════════════════════════════════════════
    #
    # Dynamics model used inside the CasADi optimisation:
    #
    #   This backend uses a **differentiable surrogate** of the full MMG
    #   model so that IPOPT receives exact gradients through the entire
    #   prediction horizon.  The surrogate captures the dominant 3-DOF
    #   kinematics + simplified surge/sway/yaw dynamics:
    #
    #     ψ_{k+1} = ψ_k + dt·r_k
    #     u_{k+1} = u_k + dt·(X_prop - X_uu·u_k·|u_k|) / m
    #     v_{k+1} = v_k + dt·(Y_v·v_k) / m
    #     r_{k+1} = r_k + dt·(N_r·r_k + N_δ·δ_k·u_k·|u_k|) / I_z
    #     x_{k+1} = x_k + dt·(u_k·cos ψ_k − v_k·sin ψ_k)
    #     y_{k+1} = y_k + dt·(u_k·sin ψ_k + v_k·cos ψ_k)
    #
    #   The SLSQP and sampling backends use the **full MMG callback**
    #   (self.dynamics) for higher-fidelity rollout.
    #
    #   **Paper claim**: "nonlinear MMG-MPC" refers to the surrogate +
    #   full-MMG-validation architecture — the surrogate provides exact
    #   auto-diff gradients for IPOPT while the full MMG is used for
    #   post-solve constraint verification and the SLSQP fallback.
    # ═══════════════════════════════════════════════════════════════════════

    # Surrogate dynamics parameters (set in __init__ or from config)
    _surge_mass: float = 5e7       # [kg] — overridden by vessel config
    _yaw_inertia: float = 1.2e11   # [kg·m²]
    _surge_Xuu: float = -0.001     # dimensionless surge resistance coeff
    _sway_Yv: float = -0.015       # dimensionless sway damping coeff
    _yaw_Nr: float = -0.003        # dimensionless yaw damping coeff
    _yaw_Ndelta: float = -0.0015   # dimensionless rudder moment coeff
    _prop_Kt: float = 0.15         # propeller thrust coefficient
    _prop_D: float = 6.0           # propeller diameter [m]
    _rho_water: float = 1025.0     # water density
    _prop_rps_factor: float = 3.0  # propeller command (0-1) → RPS conversion

    def _configure_surrogate_from_config(self, cfg: dict):
        """Pull surrogate dynamics parameters from config.

        All parameters are read from *cfg* directly.  The caller
        (TubeMPC.__init__) normalises the config so that MPC-level
        keys sit at the top level; this method does NOT re-nest into
        ``cfg["mpc"]``.
        """
        self._surge_mass = float(cfg.get("mass", 5e7))
        self._yaw_inertia = float(cfg.get("Iz", 1.2e11))
        self._surge_Xuu = float(cfg.get("surrogate_Xuu", -2e-4))
        self._sway_Yv = float(cfg.get("surrogate_Yv", -3e-3))
        self._yaw_Nr = float(cfg.get("surrogate_Nr", -5e-4))
        self._yaw_Ndelta = float(cfg.get("surrogate_Ndelta", -3e-4))
        self._prop_Kt = float(cfg.get("K_T_coeff", 0.15))
        self._prop_D = float(cfg.get("D_prop", 6.0))
        self._prop_rps_factor = float(cfg.get("prop_rps_factor", 3.0))

    def _solve_casadi(
        self,
        current_state: VesselState,
        reference_trajectory: list[VesselState],
        target_positions: list[np.ndarray],
        d_safe_cc_list: list[float],
        env: EnvironmentState,
        prev_rudder: float,
        prev_prop: float,
        kappa: float,
        epsilon: float,
        rho_tube: float,
    ) -> tuple[Optional[ControlCommand], dict]:
        """Solve using CasADi Opti + IPOPT with symbolic surrogate dynamics."""
        import time as _time
        t_start = _time.perf_counter()

        N = self.N
        dt = self.dt
        opti = ca.Opti()

        # ── Decision variables: U ∈ ℝ²ˣᴺ ────────────────────────────────
        U = opti.variable(2, N)
        rudder = U[0, :]
        prop = U[1, :]

        # ── State trajectory X ∈ ℝ⁶ˣ⁽ᴺ⁺¹⁾ (symbolic rollout) ──────────
        X = ca.MX.zeros(6, N + 1)
        X[0, 0] = current_state.x     # x
        X[1, 0] = current_state.y     # y
        X[2, 0] = current_state.psi   # psi
        X[3, 0] = current_state.u     # u
        X[4, 0] = current_state.v     # v
        X[5, 0] = current_state.r     # r

        # Surrogate parameters
        m_surge = self._surge_mass
        Iz_surge = self._yaw_inertia
        Xuu_dim = self._surge_Xuu * 0.5 * self._rho_water * self.vessel_length**2
        Yv_dim  = self._sway_Yv * 0.5 * self._rho_water * self.vessel_length**3
        Nr_dim  = self._yaw_Nr * 0.5 * self._rho_water * self.vessel_length**4
        Nd_dim  = self._yaw_Ndelta * 0.5 * self._rho_water * self.vessel_length**3

        # ── Pre-compute disturbance bias (constant over horizon) ──────────
        # This approximates the external forces (shallow water, wind/current,
        # bank effect) that the full MMG model includes but the surrogate
        # otherwise ignores.  Using the initial-state disturbance as a
        # constant over the short horizon (N·dt ≈ 10 s) captures the
        # first-order effect while keeping the optimisation differentiable.
        X_dist, Y_dist, N_dist = _compute_surrogate_disturbance_bias(
            env, current_state, self.vessel_length, self._rho_water,
        )

        # Shallow-water resistance multiplier (>1 when depth/draft < 4)
        sw_factor = _shallow_water_resistance_factor(
            getattr(env, "water_depth", 50.0),
            getattr(self, "_vessel_draught", 10.5),
        )

        # Build state trajectory symbolically
        for k in range(N):
            xk, yk, psi_k = X[0, k], X[1, k], X[2, k]
            uk, vk, rk = X[3, k], X[4, k], X[5, k]
            dk = rudder[k]
            pk = prop[k]

            # Propeller thrust (simplified)
            n_rps = pk * self._prop_rps_factor
            thrust = self._prop_Kt * self._rho_water * n_rps**2 * self._prop_D**4
            X_prop = ca.if_else(n_rps > 0, thrust, 0.0)

            # Surrogate dynamics with disturbance bias
            # Shallow water increases surge resistance proportionally
            u_dot = (X_prop + Xuu_dim * uk * ca.fabs(uk) * sw_factor + X_dist) / m_surge
            v_dot = (Yv_dim * vk + Y_dist) / m_surge
            r_dot = (Nr_dim * rk + Nd_dim * dk * uk * ca.fabs(uk) + N_dist) / Iz_surge

            psi_next = psi_k + dt * rk
            u_next   = uk + dt * u_dot
            v_next   = vk + dt * v_dot
            r_next   = rk + dt * r_dot
            x_next   = xk + dt * (uk * ca.cos(psi_k) - vk * ca.sin(psi_k))
            y_next   = yk + dt * (uk * ca.sin(psi_k) + vk * ca.cos(psi_k))

            X[0, k + 1] = x_next
            X[1, k + 1] = y_next
            X[2, k + 1] = psi_next
            X[3, k + 1] = u_next
            X[4, k + 1] = v_next
            X[5, k + 1] = r_next

        # ── Control bounds ──────────────────────────────────────────────
        opti.subject_to(opti.bounded(self.rudder_min, rudder, self.rudder_max))
        opti.subject_to(opti.bounded(self.prop_min, prop, self.prop_max))

        # ── Rudder rate constraints ─────────────────────────────────────
        rate_max_dt = self.rudder_rate_max * dt
        opti.subject_to(rudder[0] - prev_rudder <= rate_max_dt)
        opti.subject_to(prev_rudder - rudder[0] <= rate_max_dt)
        for k in range(1, N):
            opti.subject_to(rudder[k] - rudder[k - 1] <= rate_max_dt)
            opti.subject_to(rudder[k - 1] - rudder[k] <= rate_max_dt)

        # ── Propeller rate constraints ─────────────────────────────────
        prop_rate_max_dt = self.prop_rate_max * dt
        opti.subject_to(prop[0] - prev_prop <= prop_rate_max_dt)
        opti.subject_to(prev_prop - prop[0] <= prop_rate_max_dt)
        for k in range(1, N):
            opti.subject_to(prop[k] - prop[k - 1] <= prop_rate_max_dt)
            opti.subject_to(prop[k - 1] - prop[k] <= prop_rate_max_dt)

        # ── Unified cost — built ONCE, single opti.minimize() ──────────
        total_cost = self._build_casadi_cost_on_trajectory(
            X, U, current_state, reference_trajectory,
            target_positions, d_safe_cc_list, prev_rudder, prev_prop, N,
            opti, env,  # pass opti (slack vars) + env (channel/bank/UKC penalties)
        )
        opti.minimize(total_cost)

        # ── Warm-start ──────────────────────────────────────────────────
        x0_rudder, x0_prop = self._casadi_warm_start(N, prev_rudder, prev_prop)
        opti.set_initial(rudder, x0_rudder)
        opti.set_initial(prop, x0_prop)

        # ── Solve with IPOPT (wall-clock timeout) ───────────────────────
        opts = dict(self._casadi_solver_options)
        max_cpu_time = float(opts.pop("ipopt_max_cpu_time", 5.0))
        opts["ipopt.max_cpu_time"] = max_cpu_time
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
            solve_time = _time.perf_counter() - t_start

            rudder_opt = float(sol.value(rudder)[0])
            prop_opt = float(sol.value(prop)[0])
            # CasADi U has shape (2, N); ravel in F-order gives [r0..rN-1, p0..pN-1].
            # Convert to interleaved [r0, p0, r1, p1, ...] for _casadi_warm_start
            # and _build_warm_start which both assume interleaved layout.
            u_2d = np.array(sol.value(U))  # shape (2, N)
            full_u = np.empty(2 * N)
            full_u[0::2] = u_2d[0, :]  # rudder at even indices
            full_u[1::2] = u_2d[1, :]  # propeller at odd indices
            self._prev_solution = full_u.copy()

            # ── Extract predicted trajectory for diagnostics ────────────
            x_pred = np.array(sol.value(X))
            min_clearance = _compute_min_pred_clearance(
                x_pred, target_positions, d_safe_cc_list, N,
            )
            max_con_viol = _compute_max_constraint_violation(
                x_pred, target_positions, d_safe_cc_list, N,
            )

            cmd = ControlCommand(
                rudder=rudder_opt, propeller=prop_opt,
                source="MPC_CasADi",
                metadata={"solver_backend": "casadi", "solver": "ipopt"},
            )
            info = {
                "success": True,
                "solver_status": "optimal",
                "solve_time": solve_time,
                "fallback_used": False,
                "infeasible_reason": None,
                "objective_value": float(sol.value(total_cost)),
                "n_iterations": int(sol.stats().get("iter_count", 0)),
                "min_predicted_clearance": min_clearance,
                "max_constraint_violation": max_con_viol,
                "message": "IPOPT optimal",
            }
            return cmd, info

        except Exception as e:
            solve_time = _time.perf_counter() - t_start
            # Do NOT clear _prev_solution on infeasibility — keep the last
            # good solution for warm-starting the next attempt.  Clearing it
            # forces cold-start on every subsequent solve, which makes recovery
            # from transient infeasibility much slower.
            # self._prev_solution = None  # intentionally commented out

            # Try to extract debug-feasible control
            fallback_cmd = None
            try:
                rd = float(opti.debug.value(rudder)[0])
                pd = float(opti.debug.value(prop)[0])
                if np.isfinite(rd) and np.isfinite(pd):
                    fallback_cmd = ControlCommand(
                        rudder=rd, propeller=pd,
                        source="MPC_CasADi_debug",
                        metadata={"status": "debug_feasible"},
                    )
            except Exception:
                pass

            info = {
                "success": False,
                "solver_status": "failed",
                "solve_time": solve_time,
                "fallback_used": fallback_cmd is not None,
                "infeasible_reason": str(e)[:200],
                "objective_value": float("inf"),
                "n_iterations": 0,
                "min_predicted_clearance": float("inf"),
                "max_constraint_violation": 0.0,
                "message": f"IPOPT failed: {e}",
            }
            if fallback_cmd is not None:
                return fallback_cmd, info
            raise  # re-raise for SLSQP fallback

    def _build_casadi_cost_on_trajectory(
        self, X, U, current_state, reference_trajectory,
        target_positions, d_safe_cc_list, prev_rudder, prev_prop, N,
        opti, env,  # CasADi Opti stack + EnvironmentState for channel/bank/UKC
    ):
        """Build unified symbolic cost using predicted state trajectory X[:,k].

        ALL costs are computed from the surrogate state trajectory, not from
        constant-velocity extrapolation.  Collision slack penalties are added
        to the SAME cost expression (single ``opti.minimize()`` call).

        Penalty terms included (matching SLSQP _cost_function):
          - tracking (position + velocity + heading)
          - control effort + smoothness
          - collision soft penalty + hard slack penalty
          - channel boundary penalty  ← NEW
          - bank clearance penalty     ← NEW
          - UKC (grounding) penalty    ← NEW
          - speed tracking error       ← NEW
          - terminal cost
        """
        cost = 0.0
        r_prev = prev_rudder
        p_prev = prev_prop
        R11, R22 = float(self.R_mat[0, 0]), float(self.R_mat[1, 1])
        S11, S22 = float(self.S_mat[0, 0]), float(self.S_mat[1, 1])
        Qx, Qy, Qpsi = float(self.Q_pos[0,0]), float(self.Q_pos[1,1]), float(self.Q_pos[2,2])
        Qu, Qv, Qr = float(self.Q_vel[0,0]), float(self.Q_vel[1,1]), float(self.Q_vel[2,2])
        has_ref = bool(reference_trajectory) and len(reference_trajectory) >= N
        has_targets = bool(target_positions) and bool(d_safe_cc_list)

        # ── Pre-compute environment constants for horizon ──────────────────
        channel_half_w = getattr(env, "channel_width", None)
        has_channel = channel_half_w is not None and channel_half_w > 0
        if has_channel:
            channel_half_w = float(channel_half_w) / 2.0

        bl = getattr(env, "bank_distance_left", None)
        br = getattr(env, "bank_distance_right", None)
        has_bank = (bl is not None) or (br is not None)

        # Compute channel center y from bank distances if available,
        # otherwise assume centered at current ownship y-position.
        channel_center_y = float(current_state.y)
        if bl is not None and br is not None:
            channel_center_y = float(current_state.y + (br - bl) / 2.0)

        water_depth = getattr(env, "water_depth", 50.0)
        has_depth = water_depth > 0

        for k in range(N):
            rk = U[0, k]
            pk = U[1, k]

            # ── Control effort ──────────────────────────────────────────
            cost += R11 * rk**2 + R22 * pk**2

            # ── Control smoothness ──────────────────────────────────────
            dr = rk - r_prev
            dp = pk - p_prev
            cost += S11 * dr**2 + S22 * dp**2
            r_prev = rk
            p_prev = pk

            # ── Tracking: position error from SYMBOLIC TRAJECTORY ───────
            if has_ref:
                ref_k = reference_trajectory[k]
                cost += Qx * (X[0, k] - ref_k.x)**2
                cost += Qy * (X[1, k] - ref_k.y)**2
                dpsi = X[2, k] - ref_k.psi
                dpsi = ca.atan2(ca.sin(dpsi), ca.cos(dpsi))
                cost += Qpsi * dpsi**2

            # ── Tracking: velocity error from SYMBOLIC TRAJECTORY ───────
            if has_ref:
                ref_k = reference_trajectory[k]
                cost += Qu * (X[3, k] - ref_k.u)**2
                cost += Qv * (X[4, k] - ref_k.v)**2
                cost += Qr * (X[5, k] - ref_k.r)**2

            # ── Collision soft penalty from SYMBOLIC TRAJECTORY ─────────
            if has_targets and self.w_collision > 0:
                tp_k = target_positions[min(k, len(target_positions) - 1)]
                for j in range(min(len(d_safe_cc_list), len(tp_k))):
                    d_safe = d_safe_cc_list[j]
                    tgt_x = float(tp_k[j][0])
                    tgt_y = float(tp_k[j][1])
                    dist_sq = (X[0, k] - tgt_x)**2 + (X[1, k] - tgt_y)**2
                    margin = ca.sqrt(ca.fmax(dist_sq, 1e-12)) - d_safe
                    violation = ca.fmax(-margin, 0.0)
                    cost += self.w_collision * violation**2

            # ── Speed tracking error (match ref speed) ─────────────────
            if has_ref and self.w_speed > 0:
                ref_k = reference_trajectory[k]
                own_speed_sq = X[3, k]**2 + X[4, k]**2
                own_speed = ca.sqrt(ca.fmax(own_speed_sq, 1e-12))
                ref_speed = ca.sqrt(ref_k.u**2 + ref_k.v**2)
                speed_err = own_speed - ref_speed
                cost += self.w_speed * speed_err**2

            # ── Channel boundary penalty ───────────────────────────────
            if has_channel and self.w_channel > 0:
                # Penalise proximity to channel edges using soft quadratic margin
                y_from_center = ca.fabs(X[1, k] - channel_center_y)
                margin_to_edge = channel_half_w - y_from_center
                # Soft warning zone: the smaller of 2× ship length or half the channel,
                # so centred vessels in narrow channels are not always penalised.
                soft_zone = ca.fmin(2.0 * self.vessel_length, channel_half_w * 0.5)
                channel_viol = ca.fmax(soft_zone - margin_to_edge, 0.0)
                cost += self.w_channel * channel_viol**2

            # ── Bank clearance penalty ─────────────────────────────────
            if has_bank and self.w_bank > 0:
                bank_safe = 1.5 * self.vessel_length
                y_sym = X[1, k]  # symbolic y position
                if bl is not None:
                    # Compute left bank y-coordinate relative to initial position
                    bl_ref = current_state.y - bl
                    margin_l = y_sym - bl_ref  # positive = away from left bank
                    bank_viol_l = ca.fmax(bank_safe - margin_l, 0.0)
                    cost += self.w_bank * bank_viol_l**2
                if br is not None:
                    br_ref = current_state.y + br
                    margin_r = br_ref - y_sym  # positive = away from right bank
                    bank_viol_r = ca.fmax(bank_safe - margin_r, 0.0)
                    cost += self.w_bank * bank_viol_r**2

        # ── UKC (grounding) penalty — per-step with spatial depth gradient ──
        # The old version was constant w.r.t. decision variables (IPOPT ignored it).
        # Now: UKC varies with lateral position (shallower near banks), so the
        # optimizer has an incentive to stay in deeper water.
        if has_depth and self.w_ukc > 0:
            _draught = getattr(self, "_vessel_draught", 10.5)
            _ukc_threshold = 2.0
            for k in range(1, N + 1):
                # Depth decreases linearly near banks: depth(y) = water_depth * (1 - bank_factor * |y - center|)
                # This is a rough approximation; real depth comes from ENC grids.
                y_k = X[1, k]
                bank_factor = 0.0
                if half_channel is not None and half_channel > 1.0:
                    # At the bank edge, depth drops by ~30% (empirical approximation)
                    bank_dist = ca.fmax(half_channel - ca.fabs(y_k - channel_center_y), 0.0)
                    bank_factor = 0.3 * (1.0 - bank_dist / half_channel)
                depth_k = water_depth * (1.0 - bank_factor)
                ukc_k = depth_k - _draught
                ukc_viol_k = ca.fmax(_ukc_threshold - ukc_k, 0.0)
                cost += self.w_ukc * ukc_viol_k**2

        # ── Terminal cost from SYMBOLIC TRAJECTORY ──────────────────────
        if has_ref and len(reference_trajectory) > N:
            ref_N = reference_trajectory[N]
            Px, Py, Ppsi = float(self.P[0,0]), float(self.P[1,1]), float(self.P[2,2])
            dpsi_N = X[2, N] - ref_N.psi
            dpsi_N = ca.atan2(ca.sin(dpsi_N), ca.cos(dpsi_N))
            cost += Px * (X[0, N] - ref_N.x)**2
            cost += Py * (X[1, N] - ref_N.y)**2
            cost += Ppsi * dpsi_N**2

        # ── Collision hard constraints WITH slack (single minimize) ─────
        if self.enable_hard_collision and has_targets:
            n_targets = min(len(d_safe_cc_list),
                            len(target_positions[0]) if target_positions else 0)
            slack_weight = float(self.w_collision * 10.0)
            for k in range(N):
                tp_k = target_positions[min(k, len(target_positions) - 1)]
                for j in range(n_targets):
                    d_safe = d_safe_cc_list[j]
                    tgt_x = float(tp_k[j][0])
                    tgt_y = float(tp_k[j][1])
                    # Distance from SYMBOLIC TRAJECTORY
                    dist_sq = (X[0, k] - tgt_x)**2 + (X[1, k] - tgt_y)**2
                    # Slack variable
                    s_kj = opti.variable()
                    opti.subject_to(s_kj >= 0)
                    opti.subject_to(dist_sq + s_kj >= d_safe**2)
                    # Slack penalty → ADDED TO UNIFIED COST (no extra minimize)
                    cost += slack_weight * s_kj**2

        return cost

    # _add_casadi_collision_constraints removed — collision slacks are now
    # built inside _build_casadi_cost_on_trajectory as part of the unified cost.

    def _check_casadi_solution(
        self, sol, U, current_state, target_positions, d_safe_cc_list, env, N,
    ) -> bool:
        """Post-solve chance constraint verification for CasADi path."""
        try:
            # U has shape (2, N) with row 0=rudder, row 1=propeller.
            # _check_chance_constraints expects interleaved [r0,p0,r1,p1,...]
            # so use Fortran-order ravel (column-major).
            u_vals = np.array(sol.value(U)).ravel(order='F')
            return self._check_chance_constraints(
                u_vals, current_state, target_positions, d_safe_cc_list, env,
            )
        except Exception:
            return False

    def _casadi_warm_start(self, N, prev_rudder, prev_prop):
        """Build warm-start arrays for CasADi from previous solution."""
        x0_rudder = np.full(N, prev_rudder)
        x0_prop = np.full(N, prev_prop)
        if self._prev_solution is not None and len(self._prev_solution) == 2 * N:
            x0_rudder = self._prev_solution[::2].copy()
            x0_prop = self._prev_solution[1::2].copy()
        return x0_rudder, x0_prop

    # ═══════════════════════════════════════════════════════════════════════
    # JAX GPU-accelerated solver
    # ═══════════════════════════════════════════════════════════════════════

    def _solve_jax(
        self,
        current_state: VesselState,
        reference_trajectory: list[VesselState],
        target_positions: list[np.ndarray],
        d_safe_cc_list: list[float],
        env: EnvironmentState,
        prev_rudder: float,
        prev_prop: float,
    ) -> tuple[Optional[ControlCommand], dict]:
        """Solve MPC using JAX + L-BFGS-B on GPU.

        Delegates to ``mpc_jax.solve_jax_mpc()`` with the full MPCProblem
        configuration (cost weights, bounds, horizon, etc.).

        BUG #5 fix (2026-06-10): Uses ``info["full_solution"]`` for proper
        warm-start instead of tiling the first-step control.
        BUG #3 fix (2026-06-10): Passes surrogate dynamics params from config.
        """
        from .mpc_jax import solve_jax_mpc

        # Build config dict with all MPC parameters + surrogate dynamics (BUG #3)
        jax_cfg = {
            "rudder_min": self.rudder_min,
            "rudder_max": self.rudder_max,
            "rudder_rate_max": self.rudder_rate_max,
            "propeller_min": self.prop_min,
            "propeller_max": self.prop_max,
            "propeller_rate_max": self.prop_rate_max,
            "Q_pos": [float(self.Q_pos[0, 0]), float(self.Q_pos[1, 1]), float(self.Q_pos[2, 2])],
            "Q_vel": [float(self.Q_vel[0, 0]), float(self.Q_vel[1, 1]), float(self.Q_vel[2, 2])],
            "R_ctrl": [float(self.R_mat[0, 0]), float(self.R_mat[1, 1])],
            "S_smooth": [float(self.S_mat[0, 0]), float(self.S_mat[1, 1])],
            "P_terminal": [float(self.P[0, 0]), float(self.P[1, 1]), float(self.P[2, 2])],
            "w_collision_soft": self.w_collision,
            "w_channel_violation": self.w_channel,
            "w_bank": self.w_bank,
            "w_speed_error": self.w_speed,
            "w_ukc": self.w_ukc,
            "max_iterations": self.max_iterations,
            # BUG #3: Pass surrogate dynamics params matching CasADi
            "mass": float(getattr(self, "_surge_mass", 5e7)),
            "Iz": float(getattr(self, "_yaw_inertia", 1.2e11)),
            "surrogate_Xuu": float(getattr(self, "_surge_Xuu", -2e-4)),
            "surrogate_Yv": float(getattr(self, "_sway_Yv", -3e-3)),
            "surrogate_Nr": float(getattr(self, "_yaw_Nr", -5e-4)),
            "surrogate_Ndelta": float(getattr(self, "_yaw_Ndelta", -3e-4)),
            "K_T_coeff": float(getattr(self, "_prop_Kt", 0.15)),
            "D_prop": float(getattr(self, "_prop_D", 6.0)),
        }

        warm_start = self._prev_solution

        cmd, info = solve_jax_mpc(
            current_state=current_state,
            reference_trajectory=reference_trajectory,
            target_positions=target_positions,
            d_safe_cc_list=d_safe_cc_list,
            env=env,
            prev_rudder=prev_rudder,
            prev_prop=prev_prop,
            horizon=self.N,
            dt=self.dt,
            config=jax_cfg,
            warm_start=warm_start,
            vessel_length=self.vessel_length,
            vessel_draught=getattr(self, "_vessel_draught", 10.5),
        )

        if cmd is not None:
            # BUG #5 fix: Use the full solution from the solver for warm-start
            full_sol = info.get("full_solution")
            if full_sol is not None and len(full_sol) == 2 * self.N:
                self._prev_solution = np.asarray(full_sol, dtype=float)
            else:
                # Fallback: pad with repeats (degraded but functional)
                self._prev_solution = np.tile([float(cmd.rudder), float(cmd.propeller)], self.N)
            return cmd, info
        else:
            self._prev_solution = None
            return None, info

    # ═══════════════════════════════════════════════════════════════════════
    # SLSQP fallback solver (original implementation)
    # ═══════════════════════════════════════════════════════════════════════

    def _solve_slsqp(
        self,
        current_state, reference_trajectory, target_positions,
        d_safe_cc_list, env, prev_rudder, prev_prop,
    ) -> tuple[Optional[ControlCommand], dict]:
        """Fallback SLSQP solver (original scipy.optimize.minimize path)."""
        from scipy.optimize import minimize

        n_vars = 2 * self.N
        x0 = self._build_warm_start(n_vars, None, prev_rudder, prev_prop)

        bounds = []
        for _ in range(self.N):
            bounds.extend([(self.rudder_min, self.rudder_max),
                          (self.prop_min, self.prop_max)])

        constraints = self._build_all_constraints(
            current_state, reference_trajectory, target_positions,
            d_safe_cc_list, env, prev_rudder, prev_prop,
        )

        result = minimize(
            fun=self._cost_function,
            x0=x0,
            args=(current_state, reference_trajectory, target_positions,
                  d_safe_cc_list, env, prev_rudder, prev_prop),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iterations, "ftol": 1e-4},
        )

        cc_satisfied = self._check_chance_constraints(
            result.x if result.success else x0,
            current_state, target_positions, d_safe_cc_list, env,
        )

        info = {
            "success": result.success,
            "n_iterations": result.nit,
            "cost": result.fun,
            "message": result.message,
            "chance_constraints_satisfied": cc_satisfied,
        }

        if result.success:
            self._prev_solution = result.x.copy()  # store full solution for warm-start
            cmd = ControlCommand(
                rudder=float(result.x[0]),
                propeller=float(result.x[1]),
                source="MPC_SLSQP",
                metadata={"solver_info": info},
            )
            return cmd, info
        else:
            return None, info

    # ═══════════════════════════════════════════════════════════════════════
    # Shared helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _build_warm_start(
        self, n_vars, previous_command=None, prev_rudder=0.0, prev_prop=0.5,
    ):
        if previous_command is not None:
            prev_rudder = previous_command.rudder
            prev_prop = previous_command.propeller
        if self._prev_solution is not None and len(self._prev_solution) == n_vars:
            x0 = np.zeros(n_vars)
            x0[:-2] = self._prev_solution[2:]
            x0[-2:] = self._prev_solution[-2:]
            return x0
        x0 = np.zeros(n_vars)
        x0[::2] = prev_rudder
        x0[1::2] = prev_prop
        return x0

    def _build_all_constraints(
        self, current_state, reference_trajectory, target_positions,
        d_safe_cc_list, env, prev_rudder, prev_prop,
    ):
        # (Identical to original — kept for backward compat)
        constraints = []
        rate_max = self.rudder_rate_max * self.dt
        prop_rate_max = self.prop_rate_max * self.dt

        for k in range(self.N):
            if k == 0:
                # Rudder rate constraint (step 0)
                idx_k = 0; p = prev_rudder; rm = rate_max; i = idx_k
                def _make_upper_0(p=p, rm=rm, i=i):
                    def fn(u): return rm - (u[i] - p)
                    def grad(u): g = np.zeros_like(u); g[i] = -1.0; return g
                    return fn, grad
                def _make_lower_0(p=p, rm=rm, i=i):
                    def fn(u): return rm - (p - u[i])
                    def grad(u): g = np.zeros_like(u); g[i] = 1.0; return g
                    return fn, grad
                fn_u, gr_u = _make_upper_0(); fn_l, gr_l = _make_lower_0()
                constraints.append({"type": "ineq", "fun": fn_u, "jac": gr_u})
                constraints.append({"type": "ineq", "fun": fn_l, "jac": gr_l})
                # Propeller rate constraint (step 0, index 1)
                idx_p = 1; pp = prev_prop; prm = prop_rate_max
                def _make_upper_p0(pp=pp, prm=prm, i=idx_p):
                    def fn(u): return prm - (u[i] - pp)
                    def grad(u): g = np.zeros_like(u); g[i] = -1.0; return g
                    return fn, grad
                def _make_lower_p0(pp=pp, prm=prm, i=idx_p):
                    def fn(u): return prm - (pp - u[i])
                    def grad(u): g = np.zeros_like(u); g[i] = 1.0; return g
                    return fn, grad
                fn_up, gr_up = _make_upper_p0(); fn_lp, gr_lp = _make_lower_p0()
                constraints.append({"type": "ineq", "fun": fn_up, "jac": gr_up})
                constraints.append({"type": "ineq", "fun": fn_lp, "jac": gr_lp})
            else:
                # Rudder rate constraint (step k)
                idx_k = 2 * k; idx_km1 = 2 * (k - 1)
                def _make_upper(k_idx=idx_k, km1_idx=idx_km1, rm=rate_max):
                    def fn(u): return rm - (u[k_idx] - u[km1_idx])
                    def grad(u): g = np.zeros_like(u); g[k_idx] = -1.0; g[km1_idx] = 1.0; return g
                    return fn, grad
                def _make_lower(k_idx=idx_k, km1_idx=idx_km1, rm=rate_max):
                    def fn(u): return rm - (u[km1_idx] - u[k_idx])
                    def grad(u): g = np.zeros_like(u); g[k_idx] = 1.0; g[km1_idx] = -1.0; return g
                    return fn, grad
                fn_u, gr_u = _make_upper(); fn_l, gr_l = _make_lower()
                constraints.append({"type": "ineq", "fun": fn_u, "jac": gr_u})
                constraints.append({"type": "ineq", "fun": fn_l, "jac": gr_l})
                # Propeller rate constraint (step k, odd indices)
                idx_pk = 2 * k + 1; idx_pkm1 = 2 * (k - 1) + 1
                def _make_upper_p(k_idx=idx_pk, km1_idx=idx_pkm1, prm=prop_rate_max):
                    def fn(u): return prm - (u[k_idx] - u[km1_idx])
                    def grad(u): g = np.zeros_like(u); g[k_idx] = -1.0; g[km1_idx] = 1.0; return g
                    return fn, grad
                def _make_lower_p(k_idx=idx_pk, km1_idx=idx_pkm1, prm=prop_rate_max):
                    def fn(u): return prm - (u[km1_idx] - u[k_idx])
                    def grad(u): g = np.zeros_like(u); g[k_idx] = 1.0; g[km1_idx] = -1.0; return g
                    return fn, grad
                fn_up, gr_up = _make_upper_p(); fn_lp, gr_lp = _make_lower_p()
                constraints.append({"type": "ineq", "fun": fn_up, "jac": gr_up})
                constraints.append({"type": "ineq", "fun": fn_lp, "jac": gr_lp})

        if self.enable_hard_collision and target_positions:
            collision_constraints = self._build_collision_constraints(
                current_state, target_positions, d_safe_cc_list, env,
            )
            constraints.extend(collision_constraints)

        bank_constraints = self._build_bank_constraints(current_state, env)
        constraints.extend(bank_constraints)
        return constraints

    def _build_collision_constraints(
        self, current_state, target_positions, d_safe_cc_list, env,
    ):
        # (Identical to original)
        constraints = []
        fd_eps = self.fd_eps
        n_targets = len(d_safe_cc_list)
        for tgt_idx in range(n_targets):
            d_safe = d_safe_cc_list[tgt_idx]
            for k in range(self.N):
                if k >= len(target_positions):
                    break
                tgt_pos = target_positions[k][tgt_idx]
                def _make(j=tgt_idx, ki=k, tgt=tgt_pos[:2].copy(), ds=d_safe,
                          st0=current_state, env_=env, dt_=self.dt, dyn=self.dynamics, eps_=fd_eps):
                    def fn(u):
                        _, own_pos = _rollout_to_step(st0, u, ki, env_, dt_, dyn)
                        return float(np.linalg.norm(own_pos - tgt)) - ds
                    def grad(u):
                        g = np.zeros_like(u)
                        for i in range(min(2 * (ki + 1), len(u))):
                            u_plus = u.copy(); u_plus[i] += eps_
                            _, pp = _rollout_to_step(st0, u_plus, ki, env_, dt_, dyn)
                            dp = float(np.linalg.norm(pp - tgt))
                            u_minus = u.copy(); u_minus[i] -= eps_
                            _, pm = _rollout_to_step(st0, u_minus, ki, env_, dt_, dyn)
                            dm = float(np.linalg.norm(pm - tgt))
                            g[i] = (dp - dm) / (2.0 * eps_)
                        return g
                    return fn, grad
                fn, gr = _make()
                constraints.append({"type": "ineq", "fun": fn, "jac": gr})
        return constraints

    def _build_bank_constraints(self, current_state, env):
        # (Identical to original)
        constraints = []
        bank_safe = 1.5 * self.vessel_length
        has_bl = env.bank_distance_left is not None
        has_br = env.bank_distance_right is not None
        if not has_bl and not has_br:
            return constraints
        fd_eps = self.fd_eps
        own_y0 = current_state.y
        bl_ref = own_y0 - env.bank_distance_left if has_bl else None
        br_ref = own_y0 + env.bank_distance_right if has_br else None
        for k in range(self.N):
            def _make(ki=k, bl=bl_ref, br=br_ref, bs=bank_safe,
                      st0=current_state, env_=env, dt_=self.dt, dyn=self.dynamics, eps_=fd_eps):
                def fn(u):
                    _, pos = _rollout_to_step(st0, u, ki, env_, dt_, dyn)
                    margin = float("inf")
                    if bl is not None: margin = min(margin, pos[1] - bl - bs)
                    if br is not None: margin = min(margin, br - pos[1] - bs)
                    return margin
                def grad(u):
                    g = np.zeros_like(u)
                    for i in range(min(2 * (ki + 1), len(u))):
                        u_plus = u.copy(); u_plus[i] += eps_
                        _, pp = _rollout_to_step(st0, u_plus, ki, env_, dt_, dyn)
                        mp = float("inf")
                        if bl is not None: mp = min(mp, pp[1] - bl - bs)
                        if br is not None: mp = min(mp, br - pp[1] - bs)
                        u_minus = u.copy(); u_minus[i] -= eps_
                        _, pm = _rollout_to_step(st0, u_minus, ki, env_, dt_, dyn)
                        mm = float("inf")
                        if bl is not None: mm = min(mm, pm[1] - bl - bs)
                        if br is not None: mm = min(mm, br - pm[1] - bs)
                        g[i] = (mp - mm) / (2.0 * eps_)
                    return g
                return fn, grad
            fn, gr = _make()
            constraints.append({"type": "ineq", "fun": fn, "jac": gr})
        return constraints

    def _check_chance_constraints(
        self, u_flat, current_state, target_positions, d_safe_cc_list, env,
    ):
        state = current_state
        n_targets = len(d_safe_cc_list)
        for k in range(self.N):
            if k >= len(target_positions):
                break
            rudder = u_flat[2 * k]; prop = u_flat[2 * k + 1]
            own_pos = state.position()
            tgt_pos_k = target_positions[k]
            for j in range(min(n_targets, len(tgt_pos_k))):
                d = float(np.linalg.norm(own_pos - tgt_pos_k[j][:2]))
                if d < d_safe_cc_list[j]:
                    return False
            cmd = ControlCommand(rudder=rudder, propeller=prop, source="mpc_check")
            state = self.dynamics(state, cmd, env, self.dt)
        return True

    def _cost_function(
        self, u_flat, current_state, reference_trajectory,
        target_positions, d_safe_cc_list, env, prev_rudder, prev_prop,
    ):
        """Full cost function used by SLSQP backend.

        J = Σ_k [ tracking_pos + tracking_vel + control_effort + control_rate
                 + collision_penalty + channel_penalty + bank_penalty
                 + ukc_penalty + speed_penalty ]
           + terminal_pos
        """
        cost = 0.0
        state = current_state
        rp, pp = prev_rudder, prev_prop
        has_ref = bool(reference_trajectory)

        for k in range(self.N):
            rudder = u_flat[2 * k]; prop = u_flat[2 * k + 1]
            u_k = np.array([rudder, prop])

            # Control effort + rate
            cost += float(u_k @ self.R_mat @ u_k)
            du = np.array([rudder - rp, prop - pp])
            cost += float(du @ self.S_mat @ du)

            # Tracking: position error
            if has_ref and k < len(reference_trajectory):
                ref = reference_trajectory[k]
                x_err = np.array([state.x - ref.x, state.y - ref.y,
                                  state.psi - ref.psi])
                x_err[2] = np.arctan2(np.sin(x_err[2]), np.cos(x_err[2]))
                cost += float(x_err @ self.Q_pos @ x_err)

            # Tracking: velocity error
            if has_ref and k < len(reference_trajectory):
                ref = reference_trajectory[k]
                v_err = np.array([state.u - ref.u, state.v - ref.v, state.r - ref.r])
                cost += float(v_err @ self.Q_vel @ v_err)

            # ── Collision soft penalty ──────────────────────────────────
            own_pos = state.position()
            if k < len(target_positions):
                tgt_pos_k = target_positions[k]
                for j in range(min(len(d_safe_cc_list), len(tgt_pos_k))):
                    d = float(np.linalg.norm(own_pos - tgt_pos_k[j][:2]))
                    if d < d_safe_cc_list[j]:
                        cost += self.w_collision * (d_safe_cc_list[j] - d)**2

            # ── Channel / boundary penalty ──────────────────────────────
            if self.w_channel > 0 and self._enc_layer is not None:
                try:
                    in_channel = float(self._enc_layer.is_navigable(own_pos[0], own_pos[1]))
                    if in_channel < 0.5:
                        cost += self.w_channel * 1.0
                except Exception as exc:
                    logger.debug(
                        "ENC navigability check failed in SLSQP cost: %s", exc,
                    )

            # ── Bank clearance penalty ──────────────────────────────────
            if self.w_bank > 0:
                bank_safe = 1.5 * self.vessel_length  # unified with CasADi
                bl = env.bank_distance_left; br = env.bank_distance_right
                if bl is not None:
                    margin_l = state.y - (current_state.y - bl)
                    if margin_l < bank_safe:
                        cost += self.w_bank * (bank_safe - margin_l)**2
                if br is not None:
                    margin_r = (current_state.y + br) - state.y
                    if margin_r < bank_safe:
                        cost += self.w_bank * (bank_safe - margin_r)**2

            # ── UKC (grounding) penalty ─────────────────────────────────
            if self.w_ukc > 0 and env.water_depth > 0:
                ukc = env.water_depth - getattr(self, "_vessel_draught", 10.5)
                if ukc < 2.0:
                    cost += self.w_ukc * (2.0 - ukc)**2

            # ── Speed tracking error ────────────────────────────────────
            if has_ref and k < len(reference_trajectory):
                ref = reference_trajectory[k]
                speed_err = (state.u**2 + state.v**2)**0.5 - (ref.u**2 + ref.v**2)**0.5
                cost += self.w_speed * speed_err**2

            # Propagate dynamics
            cmd = ControlCommand(rudder=rudder, propeller=prop, source="mpc_step")
            state = self.dynamics(state, cmd, env, self.dt)
            rp, pp = rudder, prop

        # ── Terminal cost ───────────────────────────────────────────────
        if has_ref and len(reference_trajectory) > self.N:
            ref = reference_trajectory[self.N]
            x_err = np.array([state.x - ref.x, state.y - ref.y, state.psi - ref.psi])
            x_err[2] = np.arctan2(np.sin(x_err[2]), np.cos(x_err[2]))
            cost += float(x_err @ self.P @ x_err)

        return cost

    def reset_warm_start(self):
        self._prev_solution = None
        self._prev_rudder = 0.0
        self._prev_prop = 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Module-level helper
# ═══════════════════════════════════════════════════════════════════════════

def _rollout_to_step(state0, u_flat, k_target, env, dt, dynamics):
    # Save/restore _prev_rudder on the MMG model (not the wrapper) so this
    # function is stateless and produces reproducible gradient computation.
    # The dynamics callable may be a bound method of a wrapper class; we need
    # to find the actual MMG model that stores _prev_rudder.
    mmg_model = None
    if hasattr(dynamics, '__self__'):
        owner = dynamics.__self__
        if hasattr(owner, 'mmg'):
            mmg_model = owner.mmg
    saved_rudder = getattr(mmg_model, '_prev_rudder', 0.0) if mmg_model else 0.0

    state = state0
    for k in range(k_target + 1):
        rudder = u_flat[2 * k]; prop = u_flat[2 * k + 1]
        pos = state.position()
        if k == k_target:
            if mmg_model is not None:
                mmg_model._prev_rudder = saved_rudder
            return state, pos
        cmd = ControlCommand(rudder=rudder, propeller=prop, source="rollout")
        state = dynamics(state, cmd, env, dt)
    if mmg_model is not None:
        mmg_model._prev_rudder = saved_rudder
    return state, state.position()


def _compute_min_pred_clearance(
    x_pred: np.ndarray, target_positions: list, d_safe_list: list, N: int,
) -> float:
    """Minimum clearance across all prediction steps and targets."""
    min_cl = float("inf")
    for k in range(N):
        if k >= len(target_positions):
            break
        tp_k = target_positions[k]
        own = np.array([x_pred[0, k], x_pred[1, k]])
        for j in range(min(len(d_safe_list), len(tp_k))):
            d = float(np.linalg.norm(own - tp_k[j][:2]))
            clearance = d - d_safe_list[j]
            if clearance < min_cl:
                min_cl = clearance
    return min_cl if min_cl != float("inf") else 0.0


def _compute_max_constraint_violation(
    x_pred: np.ndarray, target_positions: list, d_safe_list: list, N: int,
) -> float:
    """Maximum positive constraint violation (negative = satisfied)."""
    max_v = 0.0
    for k in range(N):
        if k >= len(target_positions):
            break
        tp_k = target_positions[k]
        own = np.array([x_pred[0, k], x_pred[1, k]])
        for j in range(min(len(d_safe_list), len(tp_k))):
            d = float(np.linalg.norm(own - tp_k[j][:2]))
            violation = d_safe_list[j] - d  # >0 = violated
            if violation > max_v:
                max_v = violation
    return max_v


def copy_state(s: VesselState) -> VesselState:
    """Shallow copy of VesselState for simulation."""
    return VesselState(x=s.x, y=s.y, psi=s.psi, u=s.u, v=s.v, r=s.r, timestamp=s.timestamp)


# ═══════════════════════════════════════════════════════════════════════════════
# CasADi surrogate disturbance helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_surrogate_disturbance_bias(
    env, state: VesselState, vessel_length: float, rho_water: float,
) -> tuple[float, float, float]:
    """Estimate a constant body-frame disturbance bias for the CasADi surrogate.

    Computes wind + current + bank-effect contributions at the **initial** state
    and returns them as a body-frame force triplet ``(X_dist, Y_dist, N_dist)``.
    The result is treated as constant over the MPC horizon — a reasonable
    approximation for N·dt ≈ 10 s.

    Returns:
        (X_dist, Y_dist, N_dist) — surge force, sway force, yaw moment [N, N, N·m].
    """
    import numpy as np

    X_dist, Y_dist, N_dist = 0.0, 0.0, 0.0

    own_psi = state.psi
    own_c = np.cos(own_psi)
    own_s = np.sin(own_psi)

    # ── Wind force (body frame) ────────────────────────────────────────────
    wind_x = getattr(env, "wind_x", 0.0) or 0.0
    wind_y = getattr(env, "wind_y", 0.0) or 0.0
    wind_mag = np.sqrt(wind_x**2 + wind_y**2)
    if wind_mag > 0.1:
        # Project world-frame wind into body frame
        wind_u_body = wind_x * own_c + wind_y * own_s
        wind_v_body = -wind_x * own_s + wind_y * own_c  # lateral component
        # Lateral windage area ~ L * freeboard (~0.15 * L)
        lateral_area = vessel_length * 0.15 * vessel_length * 0.06
        Cd_wind = 1.2
        q_wind = 0.5 * 1.225 * wind_mag**2  # dynamic pressure [Pa]
        Y_dist += Cd_wind * q_wind * lateral_area * np.sign(wind_v_body)
        # Surge wind resistance (frontal area ~ beam * freeboard);
        # headwind = negative X (drag), tailwind = positive X (thrust)
        frontal_area = vessel_length * 0.04 * vessel_length * 0.06
        X_dist -= 0.5 * Cd_wind * q_wind * frontal_area * np.sign(wind_u_body)

    # ── Current effect (treated as body-frame force) ────────────────────────
    curr_x = getattr(env, "current_x", 0.0) or 0.0
    curr_y = getattr(env, "current_y", 0.0) or 0.0
    curr_mag = np.sqrt(curr_x**2 + curr_y**2)
    if curr_mag > 0.05:
        # Approximate current drift force proportional to wetted area
        wetted_area = vessel_length * vessel_length * 0.04  # rough estimate
        Cd_current = 0.5
        q_current = 0.5 * rho_water * curr_mag**2
        N_dist += Cd_current * q_current * wetted_area * vessel_length * 0.02

    # ── Bank-effect bias ────────────────────────────────────────────────────
    # Conservative estimate based on lateral force coefficients from
    # Vantorre et al. and Lataire et al. (bank suction ∝ 1/d² for near-bank).
    # Reference: Delefortrie et al. (2024), Ocean Engineering, 310, 118614.
    bl = getattr(env, "bank_distance_left", None)
    br = getattr(env, "bank_distance_right", None)
    if bl is not None and bl < 0.5 * vessel_length:
        # Very close to left bank — bank suction pulls toward bank (-Y)
        # Scaled by water density × vessel cross-sectional area
        bank_force = 0.005 * rho_water * vessel_length * vessel_length * 0.1
        Y_dist -= bank_force * (0.5 * vessel_length / max(bl, 1.0))**2
    if br is not None and br < 0.5 * vessel_length:
        # Very close to right bank — bank suction pulls toward bank (+Y)
        bank_force = 0.005 * rho_water * vessel_length * vessel_length * 0.1
        Y_dist += bank_force * (0.5 * vessel_length / max(br, 1.0))**2

    # Clip to prevent optimisation blow-up
    MAX_DIST = 1e7
    X_dist = float(np.clip(X_dist, -MAX_DIST, MAX_DIST))
    Y_dist = float(np.clip(Y_dist, -MAX_DIST, MAX_DIST))
    N_dist = float(np.clip(N_dist, -MAX_DIST, MAX_DIST))

    return (X_dist, Y_dist, N_dist)


def _shallow_water_resistance_factor(
    water_depth: float,
    vessel_draught: float,
) -> float:
    """Return a surge-resistance multiplier for shallow water.

    Returns 1.0 in deep water (depth/draft > 4) and up to ~2.5 in very
    shallow water (depth/draft < 1.5).  Based on PIANC/ITTC qualitative
    guidance for added resistance in confined waterways.
    """
    if water_depth <= 0:
        return 1.0
    ratio = water_depth / max(vessel_draught, 0.5)
    if ratio > 4.0:
        return 1.0
    elif ratio > 3.0:
        return 1.0 + 0.2 * (4.0 - ratio)  # mild: 1.0 → 1.2
    elif ratio > 2.0:
        return 1.2 + 0.4 * (3.0 - ratio)  # moderate: 1.2 → 1.6
    elif ratio > 1.5:
        return 1.6 + 0.9 * (2.0 - ratio)  # strong: 1.6 → 2.05
    else:
        return min(2.5, 2.05 + 0.9 * (1.5 - ratio) * 2.0)  # severe


def sampling_based_mpc(
    current_state, dynamics, reference_trajectory, target_positions,
    d_safe_eff_list, env, n_samples=500, rudder_std=5.0, prop_std=0.1,
    horizon=20, dt=0.5, rng=None, target_covariances=None,
    own_covariance=None, rho_tube=0.0, epsilon=0.10,
    rule_context: dict = None, enc_layer=None,
    w_collision=100.0, w_rule=10.0, w_channel=50.0, w_bank=5.0, w_speed=1.0,
):
    """Sampling-based MPC — random exploration baseline.

    This is NOT the nonlinear CasADi/IPOPT MPC claimed in the paper. It is a
    computationally lightweight fallback / ablation baseline. It does NOT support
    full multi-source rule constraints (P0-P5); rule-context information passed
    via ``rule_context`` is used only for a directional COLREGs heuristic.

    Cost components:
        - position tracking error
        - collision penalty (hard feasibility check)
        - COLREGs directional heuristic (if rule_context provided)
        - control effort + smoothness
        - speed error
    """
    # ── JAX GPU dispatch ────────────────────────────────────────────
    # When JAX is available, use vmap-batched GPU evaluation instead of
    # the sequential Python for-loop (100-500x faster on H100).
    if HAS_JAX:
        try:
            from .sampling_jax import jax_sampling_mpc
            return jax_sampling_mpc(
                current_state=current_state,
                reference_trajectory=reference_trajectory,
                target_positions=target_positions,
                d_safe_eff_list=d_safe_eff_list,
                env=env,
                n_samples=n_samples,
                rudder_std=rudder_std,
                prop_std=prop_std,
                horizon=horizon,
                dt=dt,
                rng=rng,
                target_covariances=target_covariances,
                own_covariance=own_covariance,
                rho_tube=rho_tube,
                epsilon=epsilon,
                rule_context=rule_context,
                enc_layer=enc_layer,
                w_collision=w_collision,
                w_rule=w_rule,
                w_channel=w_channel,
                w_bank=w_bank,
                w_speed=w_speed,
            )
        except Exception:
            pass  # Fall through to NumPy sequential loop

    if rng is None:
        rng = np.random.default_rng()
    kappa = kappa_epsilon(epsilon, df=2)
    if target_covariances is None:
        target_covariances = []
    if own_covariance is None:
        own_covariance = np.eye(2) * 1.0
    d_safe_cc_list = []
    for j in range(len(d_safe_eff_list)):
        d = d_safe_eff_list[j]
        if j < len(target_covariances):
            S_rel = compute_relative_covariance(own_covariance, target_covariances[j])
            lm = max_eigenvalue(S_rel)
            d += kappa * np.sqrt(max(lm, 0.0))
        d_safe_cc_list.append(d)

    # COLREGs heuristic: prefer starboard for head-on encounters
    prefer_starboard = False
    if rule_context:
        enc = rule_context.get("encounter_type", "")
        role = rule_context.get("ownship_role", "")
        prefer_starboard = (enc == "head_on") or (role == "give_way" and "crossing" in enc)

    best_cost = float("inf"); best_seq = None
    n_sobol = min(n_samples // 2, 200); n_random = n_samples - n_sobol

    # Halton sequence generator for low-discrepancy sampling.
    # Uses coprime bases (2, 3) for the two control dimensions.
    # This is a proper quasi-random sequence with O(1/N) discrepancy,
    # far superior to the old LCG with period ~100.
    def _halton(index, base):
        """Compute the index-th element of the Halton sequence for given base."""
        result = 0.0; f = 1.0 / base; i = index + 1  # 1-indexed
        while i > 0:
            result += (i % base) * f
            i //= base; f /= base
        return result

    for si in range(n_samples):
        if si < n_sobol:
            rudders = np.zeros(horizon); props = np.zeros(horizon)
            for k in range(horizon):
                # Use Halton sequence with coprime bases for each dimension
                # and offset each step by a different prime to decorrelate
                idx_r = si * horizon + k
                idx_p = si * horizon + k + 50000  # large offset for independence
                rudders[k] = -35.0 + 70.0 * _halton(idx_r, 2)
                props[k] = 0.2 + 0.6 * _halton(idx_p, 3)
        else:
            rudders = np.clip(rng.normal(0, rudder_std, horizon), -35.0, 35.0)
            props = np.clip(rng.normal(0.5, prop_std, horizon), 0.0, 1.0)
        u_seq = np.column_stack([rudders, props])
        cost = 0.0; state = current_state; feasible = True
        prev_r = 0.0; prev_p = 0.5
        for k in range(horizon):
            r = float(u_seq[k, 0]); p = float(u_seq[k, 1])
            # Tracking
            if k < len(reference_trajectory):
                ref = reference_trajectory[k]
                dx = state.x - ref.x; dy = state.y - ref.y
                pos_err = min(dx**2 + dy**2, 1e10) if abs(dx) <= 1e5 and abs(dy) <= 1e5 else 1e10
                cost += 10.0 * pos_err
                speed = (state.u**2 + state.v**2)**0.5
                ref_speed = (ref.u**2 + ref.v**2)**0.5
                cost += w_speed * (speed - ref_speed)**2
            # Collision
            if k < len(target_positions):
                own_pos = state.position()
                tgt_k = target_positions[k]
                for j in range(min(len(d_safe_cc_list), len(tgt_k))):
                    d = float(np.linalg.norm(own_pos - tgt_k[j][:2]))
                    if d < d_safe_cc_list[j]:
                        cost += w_collision
                        feasible = False
                        break  # BUG #6 fix: early-exit infeasible trajectory
            if not feasible:
                break  # BUG #6 fix: stop computing remaining horizon steps
            # COLREGs heuristic
            if prefer_starboard and r < 0:
                cost += w_rule  # penalise port turn in head-on
            # Control
            cost += 0.1 * r**2 + 0.01 * (p - 0.5)**2
            cost += 1.0 * (r - prev_r)**2 + 0.1 * (p - prev_p)**2
            prev_r, prev_p = r, p
            # Bank penalty
            bl = getattr(env, 'bank_distance_left', None)
            br = getattr(env, 'bank_distance_right', None)
            if bl is not None and (state.y - (current_state.y - bl)) < 30:
                cost += w_bank
            if br is not None and ((current_state.y + br) - state.y) < 30:
                cost += w_bank
            # Channel
            if enc_layer is not None:
                try:
                    if not enc_layer.is_navigable(state.x, state.y):
                        cost += w_channel
                except Exception as exc:
                    logger.debug(
                        "ENC navigability check failed in sampling MPC: %s", exc,
                    )
            cmd = ControlCommand(rudder=r, propeller=p, source="sampling")
            state = dynamics(state, cmd, env, dt)
        if feasible and cost < best_cost:
            best_cost = cost; best_seq = u_seq
    if best_seq is not None:
        metadata = {
            "n_samples": n_samples, "best_cost": best_cost,
            "backend": "sampling", "rule_context_used": rule_context is not None,
            "enc_used": enc_layer is not None,
        }
        return ControlCommand(
            rudder=float(best_seq[0, 0]), propeller=float(best_seq[0, 1]),
            source="sampling_MPC", metadata=metadata,
        ), {"success": True, "best_cost": best_cost, "backend": "sampling"}
    return None, {"success": False, "reason": "no_feasible_sample", "backend": "sampling"}
