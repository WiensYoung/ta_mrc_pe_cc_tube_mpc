"""JAX-accelerated MPC solver backend.

Provides GPU-accelerated trajectory optimization as an alternative to the
CasADi+IPOPT backend.  Uses ``jaxopt.ScipyBoundedMinimize`` (L-BFGS-B) with
JAX-evaluated cost and auto-diff gradients on GPU.

Architecture:
    - ``build_jax_cost_fn()`` — JIT-compiled cost function with all penalties
    - ``jax_rollout()`` — ``jax.lax.scan`` trajectory rollout on GPU
    - ``solve_jax_mpc()`` — main entry point, same interface as ``_solve_casadi()``

BUG #1 fix (2026-06-10): All Python ``if`` in scan body replaced with ``jnp.where``.
BUG #3 fix (2026-06-10): Surrogate dynamics params passed via config dict.

Usage::

    from ta_mrc_pe_cc_tube_mpc.control.mpc_jax import solve_jax_mpc
    cmd, info = solve_jax_mpc(current_state, reference_trajectory,
                              target_positions, d_safe_cc_list, env,
                              prev_rudder, prev_prop, config)
"""

from __future__ import annotations

import time as _time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..types import ControlCommand, EnvironmentState, VesselState

try:
    from jaxopt import ScipyBoundedMinimize
    HAS_JAXOPT = True
except ImportError:
    HAS_JAXOPT = False
    ScipyBoundedMinimize = None


def solve_jax_mpc(
    current_state: VesselState,
    reference_trajectory: list[VesselState],
    target_positions: list[np.ndarray],
    d_safe_cc_list: list[float],
    env: EnvironmentState,
    prev_rudder: float,
    prev_prop: float,
    horizon: int = 20,
    dt: float = 0.5,
    config: dict = None,
    warm_start: Optional[np.ndarray] = None,
    vessel_length: float = 180.0,
    vessel_draught: float = 10.5,
) -> tuple[Optional[ControlCommand], dict]:
    """Solve MPC using JAX + L-BFGS-B on GPU.

    Returns (ControlCommand, info_dict).  info_dict["full_solution"] contains
    the complete optimized control sequence for warm-start (BUG #5 fix).
    """
    t_start = _time.perf_counter()
    cfg = config or {}
    N = horizon

    rudder_min = float(cfg.get("rudder_min", -35.0))
    rudder_max = float(cfg.get("rudder_max", 35.0))
    rudder_rate_max = float(cfg.get("rudder_rate_max", 5.0))
    prop_min = float(cfg.get("propeller_min", 0.0))
    prop_max = float(cfg.get("propeller_max", 1.0))
    prop_rate_max = float(cfg.get("propeller_rate_max", 0.5))

    Q_pos = np.array(cfg.get("Q_pos", [10.0, 10.0, 1.0]), dtype=np.float64)
    Q_vel = np.array(cfg.get("Q_vel", [1.0, 1.0, 0.5]), dtype=np.float64)
    R_w = np.array(cfg.get("R_ctrl", [0.1, 0.01]), dtype=np.float64)
    S_w = np.array(cfg.get("S_smooth", [1.0, 0.1]), dtype=np.float64)
    P_w = np.array(cfg.get("P_terminal", [20.0, 20.0, 2.0]), dtype=np.float64)

    w_collision = float(cfg.get("w_collision_soft", 100.0))
    w_channel = float(cfg.get("w_channel_violation", 50.0))
    w_bank = float(cfg.get("w_bank", 5.0))
    w_speed = float(cfg.get("w_speed_error", 1.0))
    w_ukc = float(cfg.get("w_ukc", 10.0))
    max_iterations = int(cfg.get("max_iterations", 300))

    # BUG #3 fix: Read surrogate dynamics from config (matching CasADi)
    mass = float(cfg.get("mass", 5e7))
    Iz_val = float(cfg.get("Iz", 1.2e11))
    rho_water = 1025.0
    s_Xuu = float(cfg.get("surrogate_Xuu", -2e-4))
    s_Yv = float(cfg.get("surrogate_Yv", -3e-3))
    s_Nr = float(cfg.get("surrogate_Nr", -5e-4))
    s_Ndelta = float(cfg.get("surrogate_Ndelta", -3e-4))
    Kt = float(cfg.get("K_T_coeff", 0.15))
    Dp = float(cfg.get("D_prop", 6.0))

    surge_mass = mass
    yaw_inertia = Iz_val
    Xuu_dim = s_Xuu * 0.5 * rho_water * vessel_length**2
    Yv_dim = s_Yv * 0.5 * rho_water * vessel_length**3
    Nr_dim = s_Nr * 0.5 * rho_water * vessel_length**4
    Nd_dim = s_Ndelta * 0.5 * rho_water * vessel_length**3
    prop_Kt = Kt
    prop_D = Dp

    # ── Prepare JAX arrays ──────────────────────────────────────────
    x0 = jnp.array([current_state.x, current_state.y, current_state.psi,
                     current_state.u, current_state.v, current_state.r], dtype=jnp.float64)

    has_ref = bool(reference_trajectory) and len(reference_trajectory) >= N
    ref_array = _build_ref_array(current_state, reference_trajectory, N, dt, has_ref)

    n_targets = len(d_safe_cc_list) if d_safe_cc_list else 0
    tgt_array = _build_target_array(target_positions, n_targets, N)
    d_safe_array = jnp.array(d_safe_cc_list[:n_targets] if n_targets > 0 else [], dtype=jnp.float64)

    channel_hw = float(env.channel_width) / 2.0 if getattr(env, "channel_width", None) is not None and env.channel_width > 0 else -1.0
    bl_val = float(env.bank_distance_left) if getattr(env, "bank_distance_left", None) is not None else -1.0
    br_val = float(env.bank_distance_right) if getattr(env, "bank_distance_right", None) is not None else -1.0
    wdepth = float(getattr(env, "water_depth", 50.0) or 50.0)

    env_params = {
        "channel_half_w": jnp.array(channel_hw, dtype=jnp.float64),
        "bank_left": jnp.array(bl_val, dtype=jnp.float64),
        "bank_right": jnp.array(br_val, dtype=jnp.float64),
        "water_depth": jnp.array(wdepth, dtype=jnp.float64),
        "vessel_length": jnp.array(vessel_length, dtype=jnp.float64),
        "vessel_draught": jnp.array(vessel_draught, dtype=jnp.float64),
    }

    rate_max = rudder_rate_max * dt
    prop_rate_max_val = prop_rate_max * dt

    lower_bounds = []
    upper_bounds = []
    for k in range(N):
        lower_bounds.extend([rudder_min, prop_min])
        upper_bounds.extend([rudder_max, prop_max])
    bounds = (jnp.array(lower_bounds), jnp.array(upper_bounds))

    if warm_start is not None and len(warm_start) == 2 * N:
        u0 = jnp.array(warm_start, dtype=jnp.float64)
    else:
        u0 = jnp.zeros(2 * N, dtype=jnp.float64)
        u0 = u0.at[::2].set(prev_rudder)
        u0 = u0.at[1::2].set(prev_prop)

    # Pack weights including surrogate dynamics params (BUG #3 fix)
    weights = {
        "Q_pos": jnp.array(Q_pos), "Q_vel": jnp.array(Q_vel),
        "R": jnp.array(R_w), "S": jnp.array(S_w), "P": jnp.array(P_w),
        "w_collision": jnp.array(w_collision), "w_channel": jnp.array(w_channel),
        "w_bank": jnp.array(w_bank), "w_speed": jnp.array(w_speed),
        "w_ukc": jnp.array(w_ukc),
        "surge_mass": jnp.array(surge_mass), "yaw_inertia": jnp.array(yaw_inertia),
        "surge_Xuu_dim": jnp.array(Xuu_dim), "sway_Yv_dim": jnp.array(Yv_dim),
        "yaw_Nr_dim": jnp.array(Nr_dim), "yaw_Ndelta_dim": jnp.array(Nd_dim),
        "prop_Kt": jnp.array(prop_Kt), "prop_D": jnp.array(prop_D),
    }

    cost_params = {
        "x0": x0, "ref": ref_array, "targets": tgt_array,
        "d_safe": d_safe_array, "env": env_params, "N": N, "dt": dt,
        "prev_rudder": jnp.array(prev_rudder), "prev_prop": jnp.array(prev_prop),
        "rudder_rate_max": jnp.array(rate_max), "prop_rate_max": jnp.array(prop_rate_max_val),
    }

    @jax.jit
    def cost_fn(u_flat):
        return _mpc_cost_function_jit_safe(u_flat, weights, cost_params)

    @jax.jit
    def grad_fn(u_flat):
        return jax.grad(_mpc_cost_function_jit_safe)(u_flat, weights, cost_params)

    # ── Solve ───────────────────────────────────────────────────────
    try:
        if not HAS_JAXOPT:
            raise ImportError("jaxopt is required for JAX MPC backend")

        solver = ScipyBoundedMinimize(
            fun=cost_fn, method="L-BFGS-B",
            maxiter=max_iterations, tol=1e-4, jac=grad_fn,
        )
        result = solver.run(u0, bounds=bounds)
        u_opt = result.params
        solve_time = _time.perf_counter() - t_start

        rudder_opt = float(u_opt[0])
        prop_opt = float(u_opt[1])

        min_clearance = _compute_min_pred_clearance_jax(u_opt, cost_params, weights)

        cmd = ControlCommand(
            rudder=float(np.clip(rudder_opt, rudder_min, rudder_max)),
            propeller=float(np.clip(prop_opt, prop_min, prop_max)),
            source="MPC_JAX",
            metadata={"solver_backend": "jax", "solver": "lbfgsb"},
        )

        info = {
            "success": result.state.success if hasattr(result.state, "success") else True,
            "solver_status": "optimal" if result.state.success else "converged",
            "solve_time": solve_time,
            "fallback_used": False, "infeasible_reason": None,
            "objective_value": float(result.state.fun_val) if hasattr(result.state, "fun_val") else float(cost_fn(u_opt)),
            "n_iterations": int(result.state.iter_num) if hasattr(result.state, "iter_num") else 0,
            "min_predicted_clearance": min_clearance,
            "max_constraint_violation": 0.0,
            "message": "L-BFGS-B via JAX GPU",
            # BUG #5 fix: Return FULL solution for warm-start
            "full_solution": np.array(u_opt, dtype=np.float64),
        }
        return cmd, info

    except Exception as e:
        solve_time = _time.perf_counter() - t_start
        info = {
            "success": False, "solver_status": "failed",
            "solve_time": solve_time, "fallback_used": False,
            "infeasible_reason": str(e)[:200],
            "objective_value": float("inf"), "n_iterations": 0,
            "min_predicted_clearance": float("inf"),
            "max_constraint_violation": 0.0,
            "message": f"JAX MPC failed: {e}",
            "full_solution": None,
        }
        return None, info


# ═══════════════════════════════════════════════════════════════════════════
# JIT-safe MPC cost function (BUG #1 fix — no Python `if` inside scan)
# ═══════════════════════════════════════════════════════════════════════════

def _mpc_cost_function_jit_safe(
    u_flat: jnp.ndarray,
    weights: dict,
    params: dict,
) -> jnp.ndarray:
    """JIT-safe MPC cost function.

    ALL conditionals use ``jnp.where`` — this is essential for JAX JIT
    correctness inside ``jax.lax.scan`` bodies (BUG #1 fix).

    Surrogate dynamics parameters are read from ``weights`` dict (BUG #3 fix).
    """
    x0 = params["x0"]
    ref = params["ref"]
    targets = params["targets"]
    d_safe = params["d_safe"]
    env = params["env"]
    N = params["N"]
    dt = params["dt"]
    prev_rudder = params["prev_rudder"]
    prev_prop = params["prev_prop"]
    rudder_rate_lim = params["rudder_rate_max"]
    prop_rate_lim = params["prop_rate_max"]

    Q_pos = weights["Q_pos"]; Q_vel = weights["Q_vel"]
    R = weights["R"]; S = weights["S"]; P_term = weights["P"]
    w_coll = weights["w_collision"]; w_chan = weights["w_channel"]
    w_bnk = weights["w_bank"]; w_spd = weights["w_speed"]
    w_ukc_val = weights["w_ukc"]

    n_targets = int(d_safe.shape[0])
    channel_hw = env["channel_half_w"]
    bank_l = env["bank_left"]
    bank_r = env["bank_right"]
    water_depth = env["water_depth"]
    vessel_length = env["vessel_length"]
    vessel_draught = env["vessel_draught"]

    # BUG #3 fix: surrogate params from weights
    surge_mass = weights.get("surge_mass", jnp.array(5e7))
    yaw_inertia = weights.get("yaw_inertia", jnp.array(1.2e11))
    Xuu_dim = weights.get("surge_Xuu_dim", jnp.array(0.0))
    Yv_dim = weights.get("sway_Yv_dim", jnp.array(0.0))
    Nr_dim = weights.get("yaw_Nr_dim", jnp.array(0.0))
    Nd_dim = weights.get("yaw_Ndelta_dim", jnp.array(0.0))
    prop_Kt = weights.get("prop_Kt", jnp.array(0.15))
    prop_D = weights.get("prop_D", jnp.array(6.0))

    # Pre-compute JAX scalars for jnp.where conditions
    has_ref_f = jnp.where(ref.shape[0] >= N, jnp.float64(1.0), jnp.float64(0.0))
    has_ch = jnp.where(channel_hw > 0, jnp.float64(1.0), jnp.float64(0.0))
    has_bl = jnp.where(bank_l > 0, jnp.float64(1.0), jnp.float64(0.0))
    has_br = jnp.where(bank_r > 0, jnp.float64(1.0), jnp.float64(0.0))
    has_wd = jnp.where(water_depth > 0, jnp.float64(1.0), jnp.float64(0.0))
    bank_safe = 1.5 * vessel_length
    soft_zone = jnp.minimum(2.0 * vessel_length, jnp.maximum(channel_hw * 0.5, 1.0))

    def _body(carry, k):
        xs, ys, psi, us, vs, rs, rp, pp, cost = carry

        rc = u_flat[2 * k]; pc = u_flat[2 * k + 1]
        dr = jnp.clip(rc - rp, -rudder_rate_lim, rudder_rate_lim)
        ra = rp + dr
        dp = jnp.clip(pc - pp, -prop_rate_lim, prop_rate_lim)
        pa = pp + dp

        cost += R[0] * ra**2 + R[1] * pa**2
        cost += S[0] * dr**2 + S[1] * dp**2

        # Kinematics
        n_rps = pa * 3.0
        thrust = jnp.where(n_rps > 0, prop_Kt * 1025.0 * n_rps**2 * prop_D**4, 0.0)
        u_dot = (thrust + Xuu_dim * us * jnp.abs(us)) / surge_mass
        v_dot = (Yv_dim * vs) / surge_mass
        r_dot = (Nr_dim * rs + Nd_dim * ra * us * jnp.abs(us)) / yaw_inertia

        psi_n = psi + dt * rs
        u_n = us + dt * u_dot; v_n = vs + dt * v_dot; r_n = rs + dt * r_dot
        x_n = xs + dt * (us * jnp.cos(psi) - vs * jnp.sin(psi))
        y_n = ys + dt * (us * jnp.sin(psi) + vs * jnp.cos(psi))

        # Tracking: only active when has_ref AND k < ref.shape[0]
        k_ok = jnp.where(k < ref.shape[0], jnp.float64(1.0), jnp.float64(0.0))
        act = has_ref_f * k_ok
        rk_x = jnp.where(act > 0, ref[k, 0], 0.0)
        rk_y = jnp.where(act > 0, ref[k, 1], 0.0)
        rk_psi = jnp.where(act > 0, ref[k, 2], 0.0)
        rk_u = jnp.where(act > 0, ref[k, 3], 0.0)
        rk_v = jnp.where(act > 0, ref[k, 4], 0.0)
        rk_r = jnp.where(act > 0, ref[k, 5], 0.0)

        dpsi = psi - rk_psi
        dpsi = jnp.arctan2(jnp.sin(dpsi), jnp.cos(dpsi))
        cost += act * (
            Q_pos[0]*(xs - rk_x)**2 + Q_pos[1]*(ys - rk_y)**2 + Q_pos[2]*dpsi**2
            + Q_vel[0]*(us - rk_u)**2 + Q_vel[1]*(vs - rk_v)**2 + Q_vel[2]*(rs - rk_r)**2
        )

        # Collision penalty (guard: skip if no targets)
        tgt_ok = jnp.where((n_targets > 0) & (w_coll > 0) & (k < targets.shape[0]),
                           jnp.float64(1.0), jnp.float64(0.0))
        n_iter = max(n_targets, 1)
        for j in range(n_iter):
            ds_j = d_safe[j] if (j < n_targets and n_targets > 0) else jnp.float64(0.0)
            tj_x = jnp.where((j < targets.shape[1]), targets[k, j, 0], 0.0)
            tj_y = jnp.where((j < targets.shape[1]), targets[k, j, 1], 0.0)
            t_exists = jnp.where((jnp.abs(tj_x) + jnp.abs(tj_y)) > 1e-9, jnp.float64(1.0), jnp.float64(0.0))
            dist_sq = (xs - tj_x)**2 + (ys - tj_y)**2
            dist = jnp.sqrt(jnp.maximum(dist_sq, 1e-12))
            viol = jnp.maximum(ds_j - dist, 0.0)
            cost += tgt_ok * t_exists * w_coll * viol**2

        # Speed tracking
        own_spd = jnp.sqrt(jnp.maximum(us**2 + vs**2, 1e-12))
        ref_spd = jnp.sqrt(jnp.maximum(rk_u**2 + rk_v**2, 1e-12))
        cost += act * jnp.where(w_spd > 0, jnp.float64(1.0), 0.0) * w_spd * (own_spd - ref_spd)**2

        # Channel
        ycd = jnp.abs(ys)
        m2e = channel_hw - ycd
        cv = jnp.maximum(soft_zone - m2e, 0.0)
        cost += has_ch * jnp.where(w_chan > 0, jnp.float64(1.0), 0.0) * w_chan * cv**2

        # Bank
        ba = jnp.where(w_bnk > 0, jnp.float64(1.0), 0.0)
        blr = x0[1] - bank_l; ml = ys - blr; vl = jnp.maximum(bank_safe - ml, 0.0)
        brr = x0[1] + bank_r; mr = brr - ys; vr = jnp.maximum(bank_safe - mr, 0.0)
        cost += ba * has_bl * w_bnk * vl**2 + ba * has_br * w_bnk * vr**2

        # UKC
        ukc = water_depth - vessel_draught
        uv = jnp.maximum(2.0 - ukc, 0.0)
        cost += has_wd * jnp.where(w_ukc_val > 0, jnp.float64(1.0), 0.0) * w_ukc_val * uv**2

        return (x_n, y_n, psi_n, u_n, v_n, r_n, ra, pa, cost), None

    init = (x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], prev_rudder, prev_prop, jnp.float64(0.0))
    (xs, ys, psi, us, vs, rs, rp, pp, cost), _ = jax.lax.scan(_body, init, jnp.arange(N))

    # Terminal cost
    term_ok = jnp.where((has_ref_f > 0) & (ref.shape[0] > N), jnp.float64(1.0), jnp.float64(0.0))
    rN_x = jnp.where(term_ok > 0, ref[N, 0], 0.0)
    rN_y = jnp.where(term_ok > 0, ref[N, 1], 0.0)
    rN_psi = jnp.where(term_ok > 0, ref[N, 2], 0.0)
    dpsi = psi - rN_psi
    dpsi = jnp.arctan2(jnp.sin(dpsi), jnp.cos(dpsi))
    cost += term_ok * (P_term[0]*(xs - rN_x)**2 + P_term[1]*(ys - rN_y)**2 + P_term[2]*dpsi**2)

    return cost


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_ref_array(current_state, reference_trajectory, N, dt, has_ref):
    if has_ref:
        return jnp.array([[r.x, r.y, r.psi, r.u, r.v, r.r]
                          for r in reference_trajectory[:N+1]], dtype=jnp.float64)
    heading = current_state.psi
    speed = current_state.speed()
    refs = [[current_state.x + speed * np.cos(heading) * k * dt,
             current_state.y + speed * np.sin(heading) * k * dt,
             heading, speed, 0.0, 0.0] for k in range(N + 1)]
    return jnp.array(refs, dtype=jnp.float64)


def _build_target_array(target_positions, n_targets, N):
    if n_targets == 0 or not target_positions:
        return jnp.zeros((N + 1, 0, 2), dtype=jnp.float64)
    max_k = min(len(target_positions), N + 1)
    tgt_list = []
    for k in range(N + 1):
        if k < max_k and k < len(target_positions):
            tp_k = target_positions[k]
            row = jnp.zeros((n_targets, 2), dtype=jnp.float64)
            for j in range(min(n_targets, len(tp_k))):
                row = row.at[j, 0].set(tp_k[j][0]); row = row.at[j, 1].set(tp_k[j][1])
            tgt_list.append(row)
        else:
            tgt_list.append(tgt_list[-1] if tgt_list else jnp.zeros((n_targets, 2)))
    return jnp.stack(tgt_list)


@jax.jit
def _rollout_for_check(u_flat, x0, N, dt):
    def _step(carry, k):
        xs, ys, psi, us, vs, rs = carry
        rc, pc = u_flat[2*k], u_flat[2*k+1]
        n_rps = pc * 3.0
        thrust = jnp.where(n_rps > 0, 0.15 * 1025.0 * n_rps**2 * 6.0**4, 0.0)
        u_dot = thrust / 5e7 - 0.001 * us * jnp.abs(us)
        v_dot = -0.001 * vs
        r_dot = (-0.001 * rs - 0.0005 * rc * us * jnp.abs(us)) / 1.2e11
        u_n = us + dt*u_dot; v_n = vs + dt*v_dot; r_n = rs + dt*r_dot
        psi_n = psi + dt*rs
        x_n = xs + dt*(us*jnp.cos(psi) - vs*jnp.sin(psi))
        y_n = ys + dt*(us*jnp.sin(psi) + vs*jnp.cos(psi))
        return (x_n, y_n, psi_n, u_n, v_n, r_n), (x_n, y_n)
    init = (x0[0], x0[1], x0[2], x0[3], x0[4], x0[5])
    _, positions = jax.lax.scan(_step, init, jnp.arange(N))
    return jnp.concatenate([x0[0:2][None], jnp.stack(positions, axis=0)], axis=0)


def _compute_min_pred_clearance_jax(u_flat, cost_params, weights):
    try:
        x_traj = _rollout_for_check(u_flat, cost_params["x0"], cost_params["N"], cost_params["dt"])
        targets = cost_params["targets"]; d_safe = cost_params["d_safe"]; n_tgt = int(d_safe.shape[0])
        if n_tgt == 0: return float("inf")
        min_cl = jnp.inf
        for k in range(x_traj.shape[0]):
            if k >= targets.shape[0]: break
            ox, oy = x_traj[k, 0], x_traj[k, 1]
            for j in range(n_tgt):
                tx, ty = targets[k, j, 0], targets[k, j, 1]
                if float(jnp.abs(tx) + jnp.abs(ty)) < 1e-6: continue
                dist = jnp.sqrt((ox - tx)**2 + (oy - ty)**2)
                min_cl = jnp.minimum(min_cl, dist - d_safe[j])
        return float(min_cl) if jnp.isfinite(min_cl) else 0.0
    except Exception:
        return 0.0
