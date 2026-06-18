"""CBF-QP safety filter.

Implements the quadratic program:
    u_safe = argmin_u ||u - u_MPC||^2
    s.t.
        h_m(x) >= 0
        hdot_m(x, u) + alpha * h_m(x) >= 0
        u in U

Safety functions:
    1. h_ship_ij = d_ij - d_safe_eff_ij
    2. h_bank = d_bank - d_bank_safe
    3. h_UKC = UKC - UKC_min
    4. h_lane = lane_boundary_margin

Uses either MMG finite-difference gradients (accurate) or fixed-gain
linear approximation (fallback) for barrier Jacobians.

Solves via cvxpy (OSQP) or scipy SLSQP.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
from scipy.optimize import minimize

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

from ..types import ControlCommand, EnvironmentState, TargetShip, VesselParams, VesselState

# ── CBF-QP output dict key name constants (B10 fix — 2026-06-10) ───────────
# Single source of truth for CBF filter output keys.  All consumers
# (controller.py fallback check, safety_evaluator.py, failure_detector.py)
# MUST import these constants rather than using bare string literals, which
# are fragile to spelling variations.
CBF_OUTPUT_KEYS = {
    "ACTIVE": "cbf_active",
    "INFEASIBLE": "cbf_infeasible",
    "OVER_INTERVENTION": "over_intervention",
    "ACTIVE_CONSTRAINTS": "active_constraints",
    "CORRECTION_MAGNITUDE": "correction_magnitude",
    "N_ACTIVE": "n_active_constraints",
}


class CBFQPSafetyFilter:
    """Control Barrier Function Quadratic Program safety filter.

    Acts as a runtime safety layer that minimally modifies the MPC command
    to guarantee constraint satisfaction via a proper QP.

    Supports MMG-based accurate barrier gradients for improved precision.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.alpha = cfg.get("alpha_cbf", 1.0)
        self.tolerance = cfg.get("active_constraint_tolerance", 2.0)  # 2.0 m for maritime navigation
        self.max_correction_norm = cfg.get("max_correction_norm", 0.5)
        self.use_cvxpy = cfg.get("qp_solver", "scipy") == "cvxpy" and HAS_CVXPY
        self.use_mmg_gradients = cfg.get("cbf_use_mmg_gradients", True)
        self.fd_eps = cfg.get("cbf_fd_epsilon", 0.5)  # deg rudder
        # Separate perturbation for propeller (range [0,1] vs rudder [-35,35])
        self.fd_eps_prop = cfg.get("cbf_fd_epsilon_prop", 0.02)  # ~2% of propeller range
        # Rate constraints for CBF safety filter
        self.rudder_rate_limit = cfg.get("cbf_rudder_rate_limit",
                                         cfg.get("rudder_rate_max", 5.0) * cfg.get("cbf_dt", 0.5))
        self.prop_rate_limit = cfg.get("cbf_prop_rate_limit",
                                        cfg.get("propeller_rate_max", 0.5) * cfg.get("cbf_dt", 0.5))

        self.rudder_min = cfg.get("rudder_min", -35.0)
        self.rudder_max = cfg.get("rudder_max", 35.0)
        self.prop_min = cfg.get("propeller_min", 0.0)
        self.prop_max = cfg.get("propeller_max", 1.0)

        # Fixed-gain fallback parameters
        self.yaw_gain = cfg.get("yaw_rate_gain", 0.015)
        self.prop_gain = cfg.get("propulsion_gain", 2.0)
        self.dt_cbf = cfg.get("cbf_prediction_dt", 0.5)

        # MMG model reference (set via set_mmg_model)
        self._mmg_model = None

    def set_mmg_model(self, mmg_model):
        """Set the MMG model for accurate barrier gradient computation."""
        self._mmg_model = mmg_model

    def filter(
        self,
        u_mpc: ControlCommand,
        own_state: VesselState,
        own_params: VesselParams,
        targets: list[TargetShip],
        env: EnvironmentState,
        d_safe_eff_list: list[float],
        d_bank_safe_list: list[float] = None,
        ukc_min: float = 1.0,
    ) -> tuple[ControlCommand, dict]:
        """Apply CBF-QP safety filter to MPC command.

        Returns (filtered_command, diagnostic_info).

        .. note::
           **Safety distance consistency with MPC**: The ``d_safe_eff_list``
           passed to this method contains the **dynamic ship domain** safety
           distance (base + dynamic expansion + rule scale).  It does NOT
           include the tube margin or chance-constraint margin because those
           are disturbance-inflation terms for the MPC horizon optimization,
           not geometric barrier thresholds.  The CBF acts as a "last metre"
           runtime correction — using a smaller barrier distance than the MPC
           prediction makes it less conservative while still preventing
           imminent collision.

           To use the full ``SafetyDistanceBreakdown.total`` (including tube
           and chance margins), pass ``d_safe_eff_list`` inflated by those
           terms from the caller's breakdown.

        Args:
            u_mpc: Nominal MPC command.
            own_state: Current ownship state.
            own_params: Ownship physical parameters.
            targets: Target ships.
            env: Environment state.
            d_safe_eff_list: Effective safe distances [m] per target.
            d_bank_safe_list: Bank clearance safe distances [m].
            ukc_min: Minimum under-keel clearance [m].

        Returns:
            (filtered_command, diagnostic_info).
        """
        own_pos = own_state.position()

        # Extract nominal control early for MMG gradient computation
        u_nom = np.array([u_mpc.rudder, u_mpc.propeller], dtype=float)

        # Evaluate barrier functions and record violated ones
        h_violated = []
        barrier_ok = True

        # 1. Ship collision barriers
        for i, target in enumerate(targets):
            d_safe = d_safe_eff_list[i] if i < len(d_safe_eff_list) else 500.0
            tgt_pos = target.state.position()
            d_ij = float(np.linalg.norm(own_pos - tgt_pos))
            h_ship = d_ij - d_safe

            if h_ship < self.tolerance:
                barrier_ok = False
                hdot, dhdot = self._linearize_ship_barrier(
                    own_state, target, d_ij, own_pos, tgt_pos, d_safe,
                    own_params, u_nom, env,
                )
                h_violated.append(("ship", target.mmsi, h_ship, hdot, dhdot))

        # 2. Bank clearance barrier — check each side independently.
        # Using `or` instead of `and`: a ship near one bank with open water on
        # the other side still needs bank-clearance protection (BUG #2 fix).
        if env.bank_distance_left is not None or env.bank_distance_right is not None:
            d_bank_safe = d_bank_safe_list[0] if d_bank_safe_list else 2.0 * own_params.beam
            # Check left bank
            if env.bank_distance_left is not None:
                h_bank_l = env.bank_distance_left - d_bank_safe
                if h_bank_l < self.tolerance:
                    barrier_ok = False
                    hdot, dhdot = self._linearize_bank_barrier(own_state, env.bank_distance_left, d_bank_safe, side="left")
                    h_violated.append(("bank_clearance_left", None, h_bank_l, hdot, dhdot))
            # Check right bank
            if env.bank_distance_right is not None:
                h_bank_r = env.bank_distance_right - d_bank_safe
                if h_bank_r < self.tolerance:
                    barrier_ok = False
                    hdot, dhdot = self._linearize_bank_barrier(own_state, env.bank_distance_right, d_bank_safe, side="right")
                    h_violated.append(("bank_clearance_right", None, h_bank_r, hdot, dhdot))

        # 3. UKC barrier
        if env.water_depth is None:
            h_ukc = float("inf")  # no depth info → skip UKC check
        else:
            ukc = env.water_depth - own_params.draught
            h_ukc = ukc - ukc_min
        if h_ukc < self.tolerance:
            barrier_ok = False
            hdot, dhdot = self._linearize_ukc_barrier(ukc, ukc_min)
            h_violated.append(("ukc", None, h_ukc, hdot, dhdot))

        # 4. Lane boundary barrier
        if env.channel_width is not None:
            half_w = env.channel_width / 2
            h_lane = half_w - abs(own_state.y)
            if h_lane < self.tolerance:
                barrier_ok = False
                hdot, dhdot = self._linearize_lane_barrier(own_state, half_w)
                h_violated.append(("lane", None, h_lane, hdot, dhdot))

        if barrier_ok:
            return u_mpc, {"cbf_active": False, "active_constraints": [],
                           "over_intervention": False}

        # Solve QP for minimal control correction
        if self.use_cvxpy:
            u_safe, qp_info = self._solve_qp_cvxpy(u_nom, h_violated)
        else:
            u_safe, qp_info = self._solve_qp_scipy(u_nom, h_violated)

        correction_norm = float(np.linalg.norm(u_safe - u_nom))
        over_intervention = correction_norm > self.max_correction_norm

        active_labels = [f"{bt}_{idx}" if idx is not None else bt for bt, idx, *_ in h_violated]

        filtered_cmd = ControlCommand(
            rudder=float(u_safe[0]),
            propeller=float(u_safe[1]),
            source=u_mpc.source,
            cbf_modified=True,
            fallback_active=False,
            metadata={
                **u_mpc.metadata,
                "cbf": {
                    "active": True,
                    "active_constraints": active_labels,
                    "correction_norm": correction_norm,
                    "over_intervention": over_intervention,
                    "qp_info": qp_info,
                },
            },
        )

        return filtered_cmd, {
            "cbf_active": True,
            "active_constraints": active_labels,
            "correction_norm": correction_norm,
            "over_intervention": over_intervention,
        }

    # ------------------------------------------------------------------
    # Barrier linearisation helpers
    # ------------------------------------------------------------------

    def _linearize_ship_barrier(
        self, own_state, target, d_ij, own_pos, tgt_pos, d_safe,
        own_params=None, u_nom=None, env=None,
    ):
        """Linearize h_ship = d_ij - d_safe w.r.t. control.

        Uses MMG finite-difference gradients when available, otherwise
        falls back to the Nomoto + surge simplified model.

        hdot_current = (p_rel · v_rel) / d_ij
        dhdot/du computed via FD on MMG model.
        """
        p_rel = own_pos - tgt_pos
        v_own = np.array([
            own_state.u * np.cos(own_state.psi) - own_state.v * np.sin(own_state.psi),
            own_state.u * np.sin(own_state.psi) + own_state.v * np.cos(own_state.psi),
        ])
        v_tgt = np.array([
            target.state.u * np.cos(target.state.psi) - target.state.v * np.sin(target.state.psi),
            target.state.u * np.sin(target.state.psi) + target.state.v * np.cos(target.state.psi),
        ])

        if d_ij < 0.1:
            d_ij = 0.1

        v_rel = v_own - v_tgt
        hdot_current = float(np.dot(p_rel, v_rel) / d_ij)

        # Compute control gradient
        if self.use_mmg_gradients and self._mmg_model is not None:
            dhdot = self._mmg_barrier_gradient(
                own_state, target, d_ij, p_rel, True,
                u_nom if u_nom is not None else np.array([0.0, 0.5]),
                env,
            )
        else:
            dhdot = self._fixed_gain_ship_gradient(own_state, p_rel, d_ij)

        return hdot_current, dhdot

    def _mmg_barrier_gradient(
        self, own_state, target, d_ij, p_rel, is_ship_barrier,
        u_nom: np.ndarray = None, env: EnvironmentState = None,
    ) -> np.ndarray:
        """Compute ∂hdot/∂u using MMG finite-difference around the nominal command.

        Simulates one MMG step forward with perturbed controls to
        measure the effect on the barrier function.
        """
        dt = self.dt_cbf
        eps_rudder = self.fd_eps  # degrees for rudder
        eps_prop = self.fd_eps_prop  # fraction for propeller (~2% of range)
        mmg = self._mmg_model

        if u_nom is None:
            u_nom = np.array([0.0, 0.5])
        if env is None:
            env = EnvironmentState()

        rudder0 = float(u_nom[0])
        prop0 = float(u_nom[1])

        # Base simulation around the actual nominal command
        base_cmd = ControlCommand(rudder=rudder0, propeller=prop0, source="cbf_fd")
        base_next = mmg.step(own_state, base_cmd, env, dt)

        tgt_pos = target.state.position()

        def barrier_value(next_state):
            own_p = next_state.position()
            v_own_f = np.array([
                next_state.u * np.cos(next_state.psi) - next_state.v * np.sin(next_state.psi),
                next_state.u * np.sin(next_state.psi) + next_state.v * np.cos(next_state.psi),
            ])
            d_ij_f = max(float(np.linalg.norm(own_p - tgt_pos)), 0.1)
            p_rel_f = own_p - tgt_pos
            v_rel_f = v_own_f - np.array([
                target.state.u * np.cos(target.state.psi) - target.state.v * np.sin(target.state.psi),
                target.state.u * np.sin(target.state.psi) + target.state.v * np.cos(target.state.psi),
            ])
            return float(np.dot(p_rel_f, v_rel_f) / d_ij_f)

        # Central finite-difference for rudder (O(h²) accuracy, consistent with MPC)
        cmd_minus_r = ControlCommand(rudder=rudder0 - eps_rudder, propeller=prop0, source="cbf_fd")
        next_minus_r = mmg.step(own_state, cmd_minus_r, env, dt)
        cmd_plus_r = ControlCommand(rudder=rudder0 + eps_rudder, propeller=prop0, source="cbf_fd")
        next_plus_r = mmg.step(own_state, cmd_plus_r, env, dt)
        dhdot_d_delta = (barrier_value(next_plus_r) - barrier_value(next_minus_r)) / (2.0 * eps_rudder)

        # Central finite-difference for propeller (separate smaller epsilon)
        cmd_minus_p = ControlCommand(rudder=rudder0, propeller=prop0 - eps_prop, source="cbf_fd")
        next_minus_p = mmg.step(own_state, cmd_minus_p, env, dt)
        cmd_plus_p = ControlCommand(rudder=rudder0, propeller=prop0 + eps_prop, source="cbf_fd")
        next_plus_p = mmg.step(own_state, cmd_plus_p, env, dt)
        dhdot_d_n = (barrier_value(next_plus_p) - barrier_value(next_minus_p)) / (2.0 * eps_prop)

        grad = np.array([dhdot_d_delta, dhdot_d_n])
        # NaN guard: replace NaN/Inf entries with the fixed-gain fallback values
        if np.any(~np.isfinite(grad)):
            return self._fixed_gain_ship_gradient(own_state, p_rel, d_ij)
        return grad

    def _fixed_gain_ship_gradient(self, own_state, p_rel, d_ij) -> np.ndarray:
        """Fixed-gain ship barrier gradient (fallback)."""
        dt = self.dt_cbf
        own_heading = own_state.psi
        fwd_unit = np.array([np.cos(own_heading), np.sin(own_heading)])
        lat_unit = np.array([-np.sin(own_heading), np.cos(own_heading)])

        dhdot_d_delta = float(np.dot(p_rel, lat_unit) / d_ij * self.yaw_gain * dt * own_state.u)
        dhdot_d_n = -float(np.dot(p_rel, fwd_unit) / d_ij * self.prop_gain * dt)

        return np.array([dhdot_d_delta, dhdot_d_n])

    def _linearize_bank_barrier(self, own_state, d_bank, d_bank_safe, side: str = "right"):
        """Bank clearance barrier. Control effect via lateral velocity.

        h_bank = d_bank - d_bank_safe

        Sign convention (BUG #1 fix — 2026-06-10):
          - Left bank:  ship moving right (+v_lat) → away from left bank
            → d_bank_left INCREASES → h_bank_left INCREASES → hdot = +v_lat
            → starboard rudder → turns right → away from left bank → dhdot_d_delta = +gain
          - Right bank: ship moving right (+v_lat) → toward right bank
            → d_bank_right DECREASES → h_bank_right DECREASES → hdot = -v_lat
            → starboard rudder → turns right → toward right bank → dhdot_d_delta = -gain

        Args:
            own_state: Current ownship state.
            d_bank: Distance to bank [m].
            d_bank_safe: Safe bank clearance [m].
            side: "left" or "right" — which bank this barrier protects.
        """
        v_lat = own_state.v
        dt = self.dt_cbf

        if side == "left":
            # Moving right (+v_lat) → away from left bank → hdot positive
            hdot_current = float(v_lat)
            # Starboard rudder → turns right → away from left bank → positive gradient
            dhdot_d_delta = self.yaw_gain * dt * own_state.u
        else:
            # Moving right (+v_lat) → toward right bank → hdot negative
            hdot_current = -float(v_lat)
            # Starboard rudder → turns right → toward right bank → negative gradient
            dhdot_d_delta = -self.yaw_gain * dt * own_state.u

        dhdot_d_n = 0.0
        return hdot_current, np.array([dhdot_d_delta, dhdot_d_n])

    def _linearize_ukc_barrier(self, ukc, ukc_min):
        """UKC barrier — controls speed to reduce grounding risk.

        The ship cannot directly change water depth, but CAN reduce speed
        to limit forward motion into shallower water.  The propeller gradient
        is set negative so the QP reduces throttle when UKC is low.

        h_ukc = (depth - draught) - ukc_min
        dh/dt ≈ depth_gradient · v_forward  (depth varies with position)
        ∂(dh/dt)/∂n ≈ -prop_gain  (more throttle → more forward motion → lower UKC)
        """
        h = ukc - ukc_min
        # hdot is approximated by depth change rate — we don't have the spatial
        # gradient, so use a conservative heuristic.
        hdot = 0.0  # no direct velocity measurement of depth change
        dhdot_d_delta = 0.0  # rudder has no direct UKC effect
        # Propeller gradient: continuous ramp from 0 (at h=ukc_min) to full
        # gain (at h=0, i.e., draught = depth).  The old step function at h=0.5
        # created a dead zone where the CBF provided no throttle control.
        # Now: proportional response — closer to grounding → stronger deceleration.
        dhdot_d_n = -self.prop_gain * max(0.0, 1.0 - max(h, 0.0) / max(ukc_min, 0.1))
        return hdot, np.array([dhdot_d_delta, dhdot_d_n])

    def _linearize_lane_barrier(self, own_state, half_w):
        """Lane boundary barrier."""
        sign_y = 1.0 if own_state.y >= 0 else -1.0
        v_lat = own_state.v
        hdot_current = -sign_y * v_lat

        dt = self.dt_cbf
        dhdot_d_delta = -sign_y * self.yaw_gain * dt * own_state.u
        dhdot_d_n = 0.0

        return hdot_current, np.array([dhdot_d_delta, dhdot_d_n])

    # ------------------------------------------------------------------
    # QP solvers
    # ------------------------------------------------------------------

    def _solve_qp_cvxpy(self, u_nom, h_violated):
        """Solve CBF-QP using cvxpy with SLSQP fallback. Returns (u_safe, info)."""
        n_u = 2
        u = cp.Variable(n_u)
        objective = cp.Minimize(cp.sum_squares(u - u_nom))

        constraints = [
            u[0] >= self.rudder_min,
            u[0] <= self.rudder_max,
            u[1] >= self.prop_min,
            u[1] <= self.prop_max,
        ]

        # Actuator rate constraints: CBF correction must not exceed physical
        # steering gear / engine rate limits.  Without these, the QP can command
        # instantaneous jumps from -35° to +35° rudder, which is infeasible.
        if u_nom is not None and len(u_nom) >= 2:
            constraints.append(u[0] - u_nom[0] >= -self.rudder_rate_limit)
            constraints.append(u[0] - u_nom[0] <= self.rudder_rate_limit)
            constraints.append(u[1] - u_nom[1] >= -self.prop_rate_limit)
            constraints.append(u[1] - u_nom[1] <= self.prop_rate_limit)

        for btype, idx, h_val, hdot, dhdot in h_violated:
            constraints.append(dhdot[0] * u[0] + dhdot[1] * u[1] + hdot + self.alpha * h_val >= 0)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return u.value, {"status": prob.status, "n_constraints": len(constraints) - 4}
        except Exception:
            logger.debug("cvxpy QP solve failed, falling back to SLSQP", exc_info=True)

        # Fallback to SLSQP when cvxpy fails
        try:
            u_safe, slsqp_info = self._solve_qp_scipy(u_nom, h_violated)
            return u_safe, {"status": f"cvxpy_failed_fallback_slsqp_{slsqp_info.get('status', 'unknown')}"}
        except Exception:
            return u_nom, {"status": "both_solvers_failed"}

    def _solve_qp_scipy(self, u_nom, h_violated):
        """Solve CBF-QP using SLSQP via scipy.optimize.minimize."""
        def objective(u):
            return np.sum((u - u_nom) ** 2)

        def gradient(u):
            return 2.0 * (u - u_nom)

        bounds = [
            (self.rudder_min, self.rudder_max),
            (self.prop_min, self.prop_max),
        ]

        constraints = []
        for btype, idx, h_val, hdot, dhdot in h_violated:
            def make_constraint(dh=dhdot, hd=hdot, hv=h_val):
                def cbf_fn(u):
                    return float(np.dot(dh, u) + hd + self.alpha * hv)
                def cbf_grad(u):
                    return dh
                return {"type": "ineq", "fun": cbf_fn, "jac": cbf_grad}
            constraints.append(make_constraint())

        if not constraints:
            return np.clip(u_nom, [bounds[0][0], bounds[1][0]], [bounds[0][1], bounds[1][1]]), {"status": "no_constraints"}

        result = minimize(
            objective,
            u_nom,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": 50, "ftol": 1e-6},
        )

        if result.success:
            return result.x, {"status": "optimal", "nit": result.nit}

        # Fallback: adaptive grid search around nominal
        n_grid = 30
        rudder_vals = np.linspace(
            max(self.rudder_min, u_nom[0] - 10),
            min(self.rudder_max, u_nom[0] + 10),
            n_grid,
        )
        prop_vals = np.linspace(
            max(self.prop_min, u_nom[1] - 0.3),
            min(self.prop_max, u_nom[1] + 0.3),
            n_grid,
        )
        best_cost = float("inf")
        best_u = np.clip(u_nom, [bounds[0][0], bounds[1][0]], [bounds[0][1], bounds[1][1]])

        for r in rudder_vals:
            for p in prop_vals:
                u = np.array([r, p])
                cost = np.sum((u - u_nom) ** 2)
                feasible = True
                for _bt, _idx, h_val, hdot, dhdot in h_violated:
                    cbf_val = float(np.dot(dhdot, u) + hdot + self.alpha * h_val)
                    if cbf_val < 0:
                        feasible = False
                        cost += 1e6 * cbf_val ** 2
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_u = u.copy()

        return best_u, {"status": "slsqp_failed_grid", "slsqp_message": result.message}
