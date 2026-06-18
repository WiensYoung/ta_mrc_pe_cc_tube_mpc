"""JAX-accelerated 3-DOF MMG ship maneuvering model.

Mirrors ``mmg_3dof.MMG3DOFModel`` with pure JAX NumPy operations,
enabling JIT compilation, auto-differentiation, and GPU acceleration.

Key differences from the NumPy version:
  - Immutable state dict (JAX pytree) instead of VesselState dataclass
  - ``jnp.where`` / ``jnp.clip`` instead of Python ``if`` / ``float()``
  - Rudder rate limiter state threaded explicitly (no ``self._prev_rudder``)
  - Exact auto-diff derivatives via ``jax.jacfwd`` instead of finite differences
  - ``vmap``-compatible for batch trajectory rollout

Usage::

    from ta_mrc_pe_cc_tube_mpc.models.mmg_jax import MMGJAXModel

    model = MMGJAXModel(params)
    state = model.initial_state(x=0., y=0., psi=0., u=0., v=0., r=0.)
    next_state = model.step(state, rudder_deg=5.0, propeller=0.5, env_current=(0., 0.), dt=0.5)
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# ── Constants (JAX-friendly, no np.pi dependency at trace time) ──────────
_DEG_TO_RAD = jnp.pi / 180.0
_RHO_WATER = 1025.0

# Clipping bounds for numerical stability
_MAX_FORCE = 1e8
_MAX_REL_VEL = 20.0
_MAX_POS = 1e6


class MMGJAXModel:
    """JAX-accelerated 3-DOF MMG ship dynamics model.

    Parameters are stored as JAX arrays in a pytree-compatible dict
    for JIT compatibility.  All methods are pure functions that return
    new state dicts.

    State dict keys::

        {"x", "y", "psi", "u", "v", "r", "prev_rudder", "timestamp"}
    """

    def __init__(self, params: dict = None):
        p = params or {}
        self.L = float(p.get("length", 180.0))
        self.B = float(p.get("beam", 30.0))
        self.T = float(p.get("draught", 10.5))
        self.mass = float(p.get("mass", 5e7))
        self.Iz = float(p.get("Iz", 1.2e11))
        self.D_prop = float(p.get("D_prop", 6.0))
        self.rho = float(p.get("rho_water", _RHO_WATER))

        # Added mass
        self.X_udot = -0.05 * self.mass
        self.Y_vdot = -0.5 * self.mass
        self.Y_rdot = 0.0
        self.N_vdot = 0.0
        self.N_rdot = -0.25 * self.Iz

        # Mass matrix elements
        m = self.mass
        Iz = self.Iz
        self._m11 = m - self.X_udot
        self._m22 = m - self.Y_vdot
        self._m23 = -self.Y_rdot
        self._m32 = -self.N_vdot
        self._m33 = Iz - self.N_rdot

        # Pre-compute mass matrix inverse factor
        self._det_22 = self._m22 * self._m33 - self._m23 * self._m32

        # Nondimensional derivatives
        mmg = p.get("mmg", {})
        is_standard = p.get("version") == "standard"

        if is_standard:
            U_design = float(p.get("design_speed", p.get("max_speed", 15.0) * 0.514444))
            d_corr = self.T / self.L
            rho_Ld = 0.5 * self.rho * self.L * self.T
            rho_L2d = 0.5 * self.rho * self.L**2 * self.T
            rho_L3d = 0.5 * self.rho * self.L**3 * self.T
            rho_L4d = 0.5 * self.rho * self.L**4 * self.T

            self.X_uu = float(mmg.get("X_uu", -0.001)) * rho_Ld
            self.Y_vv = float(mmg.get("Y_vv", -0.03)) * rho_Ld
            self.Y_rr = float(mmg.get("Y_rr", 0.001)) * rho_L3d
            self.Y_vr = float(mmg.get("Y_vr", -0.01)) * rho_L2d
            self.N_vv = float(mmg.get("N_vv", -0.005)) * rho_L2d
            self.N_rr = float(mmg.get("N_rr", -0.001)) * rho_L4d
            self.N_vr = float(mmg.get("N_vr", -0.003)) * rho_L3d

            rho_Ld_U = rho_Ld * U_design
            rho_L2d_U = rho_L2d * U_design
            rho_L3d_U = rho_L3d * U_design

            self.Y_v = float(mmg.get("Y_v", -0.015)) * rho_Ld_U
            self.Y_r = float(mmg.get("Y_r", 0.002)) * rho_L2d_U
            self.N_v = float(mmg.get("N_v", -0.005)) * rho_L2d_U
            self.N_r = float(mmg.get("N_r", -0.003)) * rho_L3d_U

            ctrl_corr = d_corr
        else:
            rho_L3 = 0.5 * self.rho * self.L**3
            rho_L4 = 0.5 * self.rho * self.L**4
            self.X_uu = float(mmg.get("X_uu", -0.001)) * rho_L3 / self.L
            self.Y_v = float(mmg.get("Y_v", -0.015)) * rho_L3
            self.Y_r = float(mmg.get("Y_r", 0.002)) * rho_L4
            self.Y_vv = float(mmg.get("Y_vv", -0.03)) * rho_L3 / self.L
            self.Y_rr = float(mmg.get("Y_rr", 0.001)) * rho_L4 * self.L
            self.Y_vr = float(mmg.get("Y_vr", -0.01)) * rho_L4
            self.N_v = float(mmg.get("N_v", -0.005)) * rho_L4
            self.N_r = float(mmg.get("N_r", -0.003)) * rho_L4 * self.L
            self.N_vv = float(mmg.get("N_vv", -0.005)) * rho_L4
            self.N_rr = float(mmg.get("N_rr", -0.001)) * rho_L4 * self.L**2
            self.N_vr = float(mmg.get("N_vr", -0.003)) * rho_L4 * self.L
            ctrl_corr = 1.0

        # Control derivatives
        self.Y_delta = float(mmg.get("Y_delta", 0.003)) * 0.5 * self.rho * self.L**2 * ctrl_corr
        self.N_delta = float(mmg.get("N_delta", -0.0015)) * 0.5 * self.rho * self.L**3 * ctrl_corr

        # Propeller
        self.K_T_coeff = float(mmg.get("K_T_coeff", 0.15))

        # Constraints
        max_rudder_deg = float(p.get("max_rudder", 35.0))
        max_rudder_rate_deg = float(p.get("max_rudder_rate", 5.0))
        max_yaw_rate_deg = float(p.get("max_yaw_rate", 3.0))
        self.max_rudder = max_rudder_deg * _DEG_TO_RAD
        self.max_rudder_rate = max_rudder_rate_deg * _DEG_TO_RAD
        self.max_yaw_rate = max_yaw_rate_deg * _DEG_TO_RAD

        raw_min_turn = float(p.get("min_turning_radius", 3.0))
        self.min_turning_radius = raw_min_turn * self.L if raw_min_turn < 20 else raw_min_turn

        # Speed bounds
        self.max_speed = float(p.get("max_speed", 50.0))
        self.max_reverse_speed = float(p.get("max_reverse_speed", 5.0))

    # ═══════════════════════════════════════════════════════════════════════
    # State helpers
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def initial_state(
        x: float = 0.0, y: float = 0.0, psi: float = 0.0,
        u: float = 0.0, v: float = 0.0, r: float = 0.0,
        timestamp: float = 0.0,
    ) -> dict:
        """Create an initial state dict (JAX pytree-compatible)."""
        return {
            "x": jnp.array(x, dtype=jnp.float64),
            "y": jnp.array(y, dtype=jnp.float64),
            "psi": jnp.array(psi, dtype=jnp.float64),
            "u": jnp.array(u, dtype=jnp.float64),
            "v": jnp.array(v, dtype=jnp.float64),
            "r": jnp.array(r, dtype=jnp.float64),
            "prev_rudder": jnp.array(0.0, dtype=jnp.float64),
            "timestamp": jnp.array(timestamp, dtype=jnp.float64),
        }

    @staticmethod
    def state_to_numpy(state: dict) -> "np.ndarray":
        """Convert JAX state dict to NumPy array [x, y, psi, u, v, r]."""
        return np.array([
            float(state["x"]), float(state["y"]), float(state["psi"]),
            float(state["u"]), float(state["v"]), float(state["r"]),
        ])

    @staticmethod
    def numpy_to_state(arr: np.ndarray, prev_rudder: float = 0.0, timestamp: float = 0.0) -> dict:
        """Convert NumPy array [x,y,psi,u,v,r] to JAX state dict."""
        return {
            "x": jnp.array(float(arr[0]), dtype=jnp.float64),
            "y": jnp.array(float(arr[1]), dtype=jnp.float64),
            "psi": jnp.array(float(arr[2]), dtype=jnp.float64),
            "u": jnp.array(float(arr[3]), dtype=jnp.float64),
            "v": jnp.array(float(arr[4]), dtype=jnp.float64),
            "r": jnp.array(float(arr[5]), dtype=jnp.float64),
            "prev_rudder": jnp.array(float(prev_rudder), dtype=jnp.float64),
            "timestamp": jnp.array(float(timestamp), dtype=jnp.float64),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Single-step dynamics (pure function, JIT-compatible)
    # ═══════════════════════════════════════════════════════════════════════

    def step(
        self,
        state: dict,
        rudder_deg: float | jnp.ndarray = 0.0,
        propeller: float | jnp.ndarray = 0.5,
        env_current_x: float | jnp.ndarray = 0.0,
        env_current_y: float | jnp.ndarray = 0.0,
        dt: float = 0.5,
        disturbance_force: Optional[tuple[float, float, float]] = None,
    ) -> dict:
        """Advance ship state by one time step.

        Args:
            state: State dict with keys x, y, psi, u, v, r, prev_rudder, timestamp.
            rudder_deg: Commanded rudder angle [deg].
            propeller: Commanded propeller [0, 1].
            env_current_x: Current velocity x-component [m/s] in world frame.
            env_current_y: Current velocity y-component [m/s] in world frame.
            dt: Time step [s].
            disturbance_force: Optional (X, Y, N) disturbance in body frame [N, N, N·m].

        Returns:
            New state dict after dt seconds.
        """
        # BUG #4 fix: unpack disturbance tuple into scalar JAX arrays
        _dist_X = jnp.array(disturbance_force[0] if disturbance_force else 0.0, dtype=jnp.float64)
        _dist_Y = jnp.array(disturbance_force[1] if disturbance_force else 0.0, dtype=jnp.float64)
        _dist_N = jnp.array(disturbance_force[2] if disturbance_force else 0.0, dtype=jnp.float64)

        return _jax_step_impl(
            L=self.L, B=self.B, T=self.T, mass=self.mass, Iz=self.Iz,
            D_prop=self.D_prop, rho=self.rho,
            X_uu=self.X_uu, Y_v=self.Y_v, Y_r=self.Y_r,
            Y_vv=self.Y_vv, Y_rr=self.Y_rr, Y_vr=self.Y_vr,
            N_v=self.N_v, N_r=self.N_r, N_vv=self.N_vv, N_rr=self.N_rr, N_vr=self.N_vr,
            Y_delta=self.Y_delta, N_delta=self.N_delta,
            K_T_coeff=self.K_T_coeff,
            max_rudder=self.max_rudder, max_rudder_rate=self.max_rudder_rate,
            max_yaw_rate=self.max_yaw_rate,
            min_turning_radius=self.min_turning_radius,
            max_speed=self.max_speed, max_reverse_speed=self.max_reverse_speed,
            m11=self._m11, m22=self._m22, m23=self._m23, m32=self._m32, m33=self._m33,
            det_22=self._det_22,
            state=state, rudder_deg=rudder_deg, propeller=propeller,
            current_x=env_current_x, current_y=env_current_y,
            dt=dt, dist_X=_dist_X, dist_Y=_dist_Y, dist_N=_dist_N,
        )

    def batch_step(
        self,
        states: dict,
        rudders_deg: jnp.ndarray,
        propellers: jnp.ndarray,
        env_current_x: float = 0.0,
        env_current_y: float = 0.0,
        dt: float = 0.5,
    ) -> dict:
        """Batch step: advance N states in parallel via vmap.

        Args:
            states: Dict of stacked state arrays (each value shape (N,)).
            rudders_deg: Array of rudder angles (N,).
            propellers: Array of propeller settings (N,).
            env_current_x: Scalar or (N,) current x.
            env_current_y: Scalar or (N,) current y.
            dt: Time step.

        Returns:
            Dict of stacked next-state arrays.
        """
        @jax.vmap
        def _step_one(s, rd, pd, cx, cy):
            return self.step(s, rd, pd, cx, cy, dt)

        return _step_one(states, rudders_deg, propellers, env_current_x, env_current_y)

    # ═══════════════════════════════════════════════════════════════════════
    # Auto-diff state derivatives (replaces finite-difference get_state_derivatives)
    # ═══════════════════════════════════════════════════════════════════════

    def state_derivatives(
        self, state: dict, rudder_deg: float, propeller: float,
        env_current_x: float = 0.0, env_current_y: float = 0.0, dt: float = 0.05,
    ) -> jnp.ndarray:
        """Compute [u_dot, v_dot, r_dot] via forward-mode auto-diff.

        This replaces the finite-difference ``get_state_derivatives()`` in the
        NumPy MMG model.  Uses ``jax.jacfwd`` over the step function to get
        exact derivatives of the state update w.r.t. time.
        """
        dt_arr = jnp.array(dt, dtype=jnp.float64)

        def _delta_state(_dt):
            next_s = self.step(state, rudder_deg, propeller, env_current_x, env_current_y, _dt)
            return jnp.array([next_s["u"] - state["u"],
                              next_s["v"] - state["v"],
                              next_s["r"] - state["r"]])

        # Forward-mode Jacobian at dt gives du/dt, dv/dt, dr/dt
        return jax.jacfwd(_delta_state)(dt_arr)


# ═══════════════════════════════════════════════════════════════════════════
# Pure-function dynamics implementation (module-level for JIT)
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def _jax_step_impl(
    # ── Vessel parameters ──
    L: float, B: float, T: float, mass: float, Iz: float,
    D_prop: float, rho: float,
    X_uu: float, Y_v: float, Y_r: float,
    Y_vv: float, Y_rr: float, Y_vr: float,
    N_v: float, N_r: float, N_vv: float, N_rr: float, N_vr: float,
    Y_delta: float, N_delta: float, K_T_coeff: float,
    # ── Constraints ──
    max_rudder: float, max_rudder_rate: float, max_yaw_rate: float,
    min_turning_radius: float, max_speed: float, max_reverse_speed: float,
    # ── Mass matrix ──
    m11: float, m22: float, m23: float, m32: float, m33: float, det_22: float,
    # ── Inputs ──
    state: dict,
    rudder_deg: jnp.ndarray,
    propeller: jnp.ndarray,
    current_x: jnp.ndarray,
    current_y: jnp.ndarray,
    dt: float,
    dist_X: jnp.ndarray = None,
    dist_Y: jnp.ndarray = None,
    dist_N: jnp.ndarray = None,
) -> dict:
    """Pure JAX implementation of the MMG 3-DOF step (JIT-compiled).

    See ``MMG3DOFModel._step_impl`` for the mathematical formulation.
    This version uses only JAX operations — no Python control flow.
    """
    # BUG #4 fix: Default disturbance to zero scalars if not provided
    if dist_X is None:
        dist_X = jnp.array(0.0, dtype=jnp.float64)
    if dist_Y is None:
        dist_Y = jnp.array(0.0, dtype=jnp.float64)
    if dist_N is None:
        dist_N = jnp.array(0.0, dtype=jnp.float64)

    # ── Unpack state ────────────────────────────────────────────────────
    x = state["x"]
    y_s = state["y"]
    psi = state["psi"]
    u = state["u"]
    v = state["v"]
    r = state["r"]
    prev_rudder = state["prev_rudder"]
    ts = state["timestamp"]

    # ── Rudder rate limit ───────────────────────────────────────────────
    delta_cmd = jnp.clip(rudder_deg, -jnp.rad2deg(max_rudder), jnp.rad2deg(max_rudder))
    delta_cmd_rad = delta_cmd * _DEG_TO_RAD
    delta_rate_max = max_rudder_rate * dt
    delta = jnp.clip(delta_cmd_rad, prev_rudder - delta_rate_max, prev_rudder + delta_rate_max)

    # ── Propeller ───────────────────────────────────────────────────────
    n_cmd = jnp.clip(propeller, 0.0, 1.0)
    n_rps = n_cmd * 3.0  # max ~3 rps

    # ── Current velocity in body frame ──────────────────────────────────
    cos_psi = jnp.cos(psi)
    sin_psi = jnp.sin(psi)
    u_current_body = current_x * cos_psi + current_y * sin_psi
    v_current_body = -current_x * sin_psi + current_y * cos_psi

    # ── Propeller thrust ────────────────────────────────────────────────
    J_val = jnp.where(
        jnp.abs(n_rps) > 1e-6,
        u / (n_rps * D_prop + 1e-6),
        0.0,
    )
    K_T = jnp.where(
        n_rps > 0,
        K_T_coeff * jnp.maximum(0.0, 1.0 - 0.5 * jnp.abs(J_val)),
        0.0,
    )
    X_prop = K_T * rho * n_rps**2 * D_prop**4

    # ── Relative velocity (clipped) ─────────────────────────────────────
    u_rel = jnp.clip(u - u_current_body, -_MAX_REL_VEL, _MAX_REL_VEL)
    v_rel = jnp.clip(v - v_current_body, -_MAX_REL_VEL, _MAX_REL_VEL)
    r_clipped = jnp.clip(r, -max_yaw_rate, max_yaw_rate)

    # ── Hull forces ─────────────────────────────────────────────────────
    X_hull = X_uu * u_rel * jnp.abs(u_rel)

    Y_hull = (Y_v * v_rel + Y_r * r_clipped
              + Y_vv * v_rel * jnp.abs(v_rel)
              + Y_rr * r_clipped * jnp.abs(r_clipped)
              + Y_vr * v_rel * r_clipped)

    N_hull = (N_v * v_rel + N_r * r_clipped
              + N_vv * v_rel * jnp.abs(v_rel)
              + N_rr * r_clipped * jnp.abs(r_clipped)
              + N_vr * v_rel * r_clipped)

    # ── Rudder forces ───────────────────────────────────────────────────
    U_sq = u_rel * jnp.abs(u_rel) + v_rel * jnp.abs(v_rel)
    X_r = 0.0
    Y_r_force = Y_delta * delta * U_sq
    N_r_force = N_delta * delta * U_sq

    # ── Total forces with clipping ──────────────────────────────────────
    X_total = jnp.clip(X_hull + X_prop + X_r, -_MAX_FORCE, _MAX_FORCE)
    Y_total = jnp.clip(Y_hull + Y_r_force, -_MAX_FORCE, _MAX_FORCE)
    N_total = jnp.clip(N_hull + N_r_force, -_MAX_FORCE, _MAX_FORCE)

    # ── External disturbance (BUG #4 fix — 2026-06-10) ────────────────
    # Disturbance forces [X, Y, N] in body frame are applied additively.
    # Previously accepted but silently ignored inside JIT; now they are
    # passed as three scalar float args (dist_X, dist_Y, dist_N) for JIT
    # compatibility.  Default is 0.0 (no disturbance).
    X_total = X_total + dist_X
    Y_total = Y_total + dist_Y
    N_total = N_total + dist_N

    # NaN guard
    X_total = jnp.where(jnp.isfinite(X_total), X_total, 0.0)
    Y_total = jnp.where(jnp.isfinite(Y_total), Y_total, 0.0)
    N_total = jnp.where(jnp.isfinite(N_total), N_total, 0.0)

    # ── External disturbance ────────────────────────────────────────────
    # disturbance_force is a tuple that can't be used inside JIT.
    # We handle it as three float arguments (0 if not provided).
    # This is a limitation of the current approach — see caller-side
    # workaround in the _wrap_step helper.

    # ── Mass matrix solve ───────────────────────────────────────────────
    # [m11   0    0  ] [u_dot]   [X_total]
    # [ 0   m22  m23 ] [v_dot] = [Y_total]
    # [ 0   m32  m33 ] [r_dot]   [N_total]

    u_dot = X_total / jnp.maximum(m11, 1e-12)

    # Solve 2x2 subsystem for v_dot, r_dot
    det_22_safe = jnp.where(jnp.abs(det_22) > 1e-12, det_22, 1.0)
    v_dot_main = (m33 * Y_total - m23 * N_total) / det_22_safe
    r_dot_main = (-m32 * Y_total + m22 * N_total) / det_22_safe

    # Fallback when det_22 is singular
    v_dot_fallback = jnp.where(jnp.abs(m22) > 1e-12, Y_total / m22, 0.0)
    r_dot_fallback = jnp.where(jnp.abs(m33) > 1e-12, N_total / m33, 0.0)

    v_dot = jnp.where(jnp.abs(det_22) > 1e-12, v_dot_main, v_dot_fallback)
    r_dot = jnp.where(jnp.abs(det_22) > 1e-12, r_dot_main, r_dot_fallback)

    # ── Integrate ───────────────────────────────────────────────────────
    u_new = jnp.clip(u + u_dot * dt, -max_reverse_speed, max_speed)
    v_new = jnp.clip(v + v_dot * dt, -max_speed, max_speed)
    r_new = jnp.clip(r + r_dot * dt, -max_yaw_rate * 2, max_yaw_rate * 2)

    # NaN guard after integration
    u_new = jnp.where(jnp.isfinite(u_new), u_new, 0.0)
    v_new = jnp.where(jnp.isfinite(v_new), v_new, 0.0)
    r_new = jnp.where(jnp.isfinite(r_new), r_new, 0.0)

    # Yaw rate constraint
    r_new = jnp.clip(r_new, -max_yaw_rate, max_yaw_rate)

    # Turning radius constraint
    speed_sq = u_new**2 + v_new**2
    speed = jnp.sqrt(jnp.maximum(speed_sq, 0.0))
    R_turn = jnp.where(jnp.abs(r_new) > 1e-6, speed / jnp.abs(r_new), 1e9)
    r_new = jnp.where(
        (jnp.abs(r_new) > 1e-6) & (speed > 0.1) & (R_turn < min_turning_radius),
        jnp.sign(r_new) * speed / min_turning_radius,
        r_new,
    )

    # ── Update position (trapezoidal yaw) ───────────────────────────────
    psi_new = psi + 0.5 * (r + r_new) * dt
    cos_psi_new = jnp.cos(psi_new)
    sin_psi_new = jnp.sin(psi_new)
    x_dot = u_new * cos_psi_new - v_new * sin_psi_new
    y_dot = u_new * sin_psi_new + v_new * cos_psi_new

    x_new = jnp.clip(x + x_dot * dt, -_MAX_POS, _MAX_POS)
    y_new = jnp.clip(y_s + y_dot * dt, -_MAX_POS, _MAX_POS)

    return {
        "x": x_new,
        "y": y_new,
        "psi": psi_new,
        "u": u_new,
        "v": v_new,
        "r": r_new,
        "prev_rudder": delta,
        "timestamp": ts + dt,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory rollout (vmap-batched for GPU)
# ═══════════════════════════════════════════════════════════════════════════

def rollout_trajectory(
    model: MMGJAXModel,
    init_state: dict,
    control_sequence: jnp.ndarray,  # shape (N_steps, 2): [rudder_deg, propeller]
    env_current_x: float = 0.0,
    env_current_y: float = 0.0,
    dt: float = 0.5,
) -> dict:
    """Roll out a trajectory over N steps given a control sequence.

    Uses ``jax.lax.scan`` for efficient loop unrolling on GPU.

    Args:
        model: MMGJAXModel instance.
        init_state: Initial state dict.
        control_sequence: (N_steps, 2) array of [rudder_deg, propeller].
        env_current_x/y: Constant current velocity.
        dt: Time step.

    Returns:
        Dict with stacked trajectory arrays:
          - ``states``: (N_steps+1, 8) array of [x,y,psi,u,v,r,prev_rudder,timestamp]
          - ``x_traj``: (N_steps+1, 6) array of [x, y, psi, u, v, r]
    """
    def _step_fn(state_dict, control):
        rudder, prop = control[0], control[1]
        next_state = model.step(state_dict, rudder, prop, env_current_x, env_current_y, dt)
        return next_state, next_state

    init = init_state
    # Prepend initial state
    all_states = jax.lax.scan(_step_fn, init, control_sequence)
    # all_states[0] = final state, all_states[1] = stacked intermediates

    # Build full trajectory: [init] + intermediate states
    state_stack = {
        "x": jnp.concatenate([init["x"][None], all_states[1]["x"]]),
        "y": jnp.concatenate([init["y"][None], all_states[1]["y"]]),
        "psi": jnp.concatenate([init["psi"][None], all_states[1]["psi"]]),
        "u": jnp.concatenate([init["u"][None], all_states[1]["u"]]),
        "v": jnp.concatenate([init["v"][None], all_states[1]["v"]]),
        "r": jnp.concatenate([init["r"][None], all_states[1]["r"]]),
        "prev_rudder": jnp.concatenate([init["prev_rudder"][None], all_states[1]["prev_rudder"]]),
        "timestamp": jnp.concatenate([init["timestamp"][None], all_states[1]["timestamp"]]),
    }

    x_traj = jnp.stack([
        state_stack["x"], state_stack["y"], state_stack["psi"],
        state_stack["u"], state_stack["v"], state_stack["r"],
    ], axis=-1)  # (N+1, 6)

    return {"states": state_stack, "x_traj": x_traj}


def batch_rollout_trajectory(
    model: MMGJAXModel,
    init_states: dict,  # each value shape (B,)
    control_sequences: jnp.ndarray,  # shape (B, N_steps, 2)
    env_current_x: float = 0.0,
    env_current_y: float = 0.0,
    dt: float = 0.5,
) -> jnp.ndarray:
    """Batch trajectory rollout via vmap.

    Evaluates B trajectories in parallel on GPU.

    Args:
        model: MMGJAXModel instance.
        init_states: Dict of stacked initial states (B,).
        control_sequences: (B, N_steps, 2) array.
        env_current_x/y: Scalar current.
        dt: Time step.

    Returns:
        (B, N_steps+1, 6) trajectory array.
    """
    def _single_rollout(single_state, single_controls):
        result = rollout_trajectory(model, single_state, single_controls,
                                     env_current_x, env_current_y, dt)
        return result["x_traj"]

    return jax.vmap(_single_rollout)(init_states, control_sequences)
