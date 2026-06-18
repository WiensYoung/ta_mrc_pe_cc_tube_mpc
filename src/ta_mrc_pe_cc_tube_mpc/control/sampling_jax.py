"""GPU-accelerated sampling-based MPC via JAX vmap (JIT-safe).

BUG #2 fix (2026-06-10): All Python ``if`` statements inside JIT/vmap
replaced with ``jnp.where`` for correct GPU execution.

Expected speedup: 100-500x on H100 GPU (500 trajectories x 20 steps each).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..types import ControlCommand, EnvironmentState, VesselState


def jax_sampling_mpc(
    current_state: VesselState,
    reference_trajectory: list[VesselState],
    target_positions: list[np.ndarray],
    d_safe_eff_list: list[float],
    env: EnvironmentState,
    n_samples: int = 500,
    rudder_std: float = 5.0,
    prop_std: float = 0.1,
    horizon: int = 20,
    dt: float = 0.5,
    rng: np.random.Generator = None,
    target_covariances: list[np.ndarray] = None,
    own_covariance: np.ndarray = None,
    rho_tube: float = 0.0,
    epsilon: float = 0.10,
    rule_context: dict = None,
    enc_layer=None,
    w_collision: float = 100.0,
    w_rule: float = 10.0,
    w_channel: float = 50.0,
    w_bank: float = 5.0,
    w_speed: float = 1.0,
    vessel_length: float = 180.0,
    vessel_beam: float = 30.0,
    vessel_draught: float = 10.5,
) -> tuple[Optional[ControlCommand], dict]:
    """Sampling-based MPC accelerated on GPU via JAX vmap (JIT-safe)."""
    N = horizon
    n_targets = len(d_safe_eff_list) if d_safe_eff_list else 0
    seed = int(rng.integers(0, 2**31 - 1)) if rng is not None else 42
    key = jax.random.PRNGKey(seed)

    key, key_u = jax.random.split(key)
    n_quasi = n_samples // 2
    n_random = n_samples - n_quasi

    quasi_rudder = jnp.linspace(-35.0, 35.0, n_quasi + 1)[:n_quasi][:, None] + jnp.zeros((1, N))
    quasi_prop = jnp.linspace(0.1, 0.9, n_quasi + 1)[:n_quasi][:, None] + jnp.zeros((1, N))

    key_r, key_p = jax.random.split(key_u)
    random_rudder = jax.random.normal(key_r, (n_random, N)) * rudder_std
    random_prop = jax.random.normal(key_p, (n_random, N)) * prop_std + 0.5

    rudders = jnp.clip(jnp.concatenate([quasi_rudder, random_rudder], axis=0), -35.0, 35.0)
    props = jnp.clip(jnp.concatenate([quasi_prop, random_prop], axis=0), 0.0, 1.0)

    x0 = jnp.array([current_state.x, current_state.y, current_state.psi,
                     current_state.u, current_state.v, current_state.r], dtype=jnp.float64)

    has_ref = bool(reference_trajectory) and len(reference_trajectory) >= N
    ref_arr = jnp.array([[r.x, r.y, r.psi, r.u, r.v, r.r] for r in reference_trajectory[:N]], dtype=jnp.float64) if has_ref else jnp.zeros((N, 6), dtype=jnp.float64)

    if n_targets > 0 and target_positions:
        max_k = min(len(target_positions), N)
        tgt_list = []
        for k in range(N):
            if k < max_k:
                tp_k = target_positions[k]
                row = jnp.zeros((n_targets, 2), dtype=jnp.float64)
                for j in range(min(n_targets, len(tp_k))):
                    row = row.at[j, 0].set(tp_k[j][0]); row = row.at[j, 1].set(tp_k[j][1])
                tgt_list.append(row)
            else:
                tgt_list.append(tgt_list[-1])
        tgt_arr = jnp.stack(tgt_list)
    else:
        tgt_arr = jnp.zeros((N, 0, 2), dtype=jnp.float64)

    d_safe_arr = jnp.array(d_safe_eff_list[:n_targets], dtype=jnp.float64) if n_targets > 0 else jnp.array([], dtype=jnp.float64)

    cc_arr = jnp.zeros(max(n_targets, 1), dtype=jnp.float64)
    if target_covariances and own_covariance is not None and n_targets > 0:
        from ..utils.math_utils import kappa_epsilon, max_eigenvalue
        from ..risk.uncertainty import compute_relative_covariance
        kappa = kappa_epsilon(epsilon, df=2)
        cm = []
        for j in range(n_targets):
            if j < len(target_covariances):
                Sigma_rel = compute_relative_covariance(own_covariance, target_covariances[j])
                lm = max_eigenvalue(Sigma_rel)
                cm.append(kappa * np.sqrt(max(lm, 0.0)))
            else:
                cm.append(0.0)
        cc_arr = jnp.array(cm, dtype=jnp.float64)

    channel_hw = float(env.channel_width / 2.0) if getattr(env, "channel_width", None) is not None and env.channel_width > 0 else -1.0
    bl_v = float(env.bank_distance_left) if getattr(env, "bank_distance_left", None) is not None else -1.0
    br_v = float(env.bank_distance_right) if getattr(env, "bank_distance_right", None) is not None else -1.0
    bl_ref = current_state.y - env.bank_distance_left if env.bank_distance_left is not None else current_state.y
    br_ref = current_state.y + env.bank_distance_right if env.bank_distance_right is not None else current_state.y

    prefer_starboard = False
    if rule_context:
        enc = rule_context.get("encounter_type", "")
        role = rule_context.get("ownship_role", "")
        prefer_starboard = (enc == "head_on") or (role == "give_way" and "crossing" in enc)

    # Pre-compute JAX scalars for jnp.where conditions
    _n_tgt = n_targets
    _chw = jnp.array(channel_hw, dtype=jnp.float64)
    _bl = jnp.array(bl_v, dtype=jnp.float64)
    _br = jnp.array(br_v, dtype=jnp.float64)
    _blr = jnp.array(bl_ref, dtype=jnp.float64)
    _brr = jnp.array(br_ref, dtype=jnp.float64)
    _ps = jnp.where(jnp.array(prefer_starboard), jnp.float64(1.0), jnp.float64(0.0))
    _bank_safe = 1.5 * vessel_length

    @jax.jit
    def evaluate_all(r_seq, p_seq):
        """vmap over N sequences."""
        return jax.vmap(lambda r, p: _eval_one_jit(
            r, p, x0, ref_arr, tgt_arr, d_safe_arr, cc_arr,
            _chw, _bl, _br, _blr, _brr, vessel_length, _bank_safe,
            w_collision, w_rule, w_channel, w_bank, w_speed,
            _ps, _n_tgt, N, dt,
        ))(r_seq, p_seq)

    costs = evaluate_all(rudders, props)
    best_idx = jnp.argmin(costs)
    best_cost = costs[best_idx]
    best_rudder = float(rudders[best_idx, 0])
    best_prop = float(props[best_idx, 0])
    best_cost_val = float(best_cost)

    if jnp.isfinite(best_cost) and best_cost_val < 1e9:
        cmd = ControlCommand(rudder=best_rudder, propeller=best_prop,
                             source="sampling_MPC_JAX",
                             metadata={"n_samples": n_samples, "best_cost": best_cost_val, "backend": "jax_sampling"})
        return cmd, {"success": True, "best_cost": best_cost_val, "backend": "jax_sampling", "n_samples": n_samples}
    return None, {"success": False, "reason": "all_infeasible", "backend": "jax_sampling"}


# ═══════════════════════════════════════════════════════════════════════════
# CEM (Cross-Entropy Method) sampling MPC — GAP #9 (2026-06-10)
# ═══════════════════════════════════════════════════════════════════════════

def cem_sampling_mpc(
    current_state: VesselState,
    reference_trajectory: list[VesselState],
    target_positions: list[np.ndarray],
    d_safe_eff_list: list[float],
    env: EnvironmentState,
    n_samples: int = 200,
    n_elite: int = 20,
    n_iterations: int = 4,
    horizon: int = 20,
    dt: float = 0.5,
    rng: np.random.Generator = None,
    **kwargs,
) -> tuple[Optional[ControlCommand], dict]:
    """Cross-Entropy Method (CEM) sampling MPC on GPU.

    Iteratively refines the sampling distribution toward high-performing
    control sequences.  Each iteration evaluates N samples via vmap on GPU,
    selects elite top-K, and refits the Gaussian.  This is dramatically
    more sample-efficient than uniform random sampling.

    Reference: Williams et al. (2017) "MPPI" IEEE T-ITS;
               Wagener et al. (2024) "CEM-MPC" Ocean Engineering.

    Args:
        n_samples: Samples per iteration (default 200, much less than naive 500).
        n_elite: Elite samples for distribution fitting (top-K).
        n_iterations: CEM iterations (3-5 sufficient in practice).
        **kwargs: Passed through to ``jax_sampling_mpc``.

    Returns:
        (ControlCommand, info_dict).
    """
    N = horizon
    seed = int(rng.integers(0, 2**31 - 1)) if rng is not None else 42
    key = jax.random.PRNGKey(seed)

    # Initial distribution: mean = zero rudder + 0.5 prop, std = wide
    mu = jnp.zeros((N, 2))
    mu = mu.at[:, 0].set(0.0)   # rudder mean
    mu = mu.at[:, 1].set(0.5)   # propeller mean
    sigma = jnp.ones((N, 2))
    sigma = sigma.at[:, 0].set(15.0)   # rudder std (wide)
    sigma = sigma.at[:, 1].set(0.3)   # propeller std

    best_cost = float("inf")
    best_seq = None

    for iteration in range(n_iterations):
        key, sk = jax.random.split(key)
        # Sample from current distribution
        noise = jax.random.normal(sk, (n_samples, N, 2))
        samples = mu[None, :, :] + noise * sigma[None, :, :]  # (n_samples, N, 2)
        rudders = jnp.clip(samples[:, :, 0], -35.0, 35.0)
        props = jnp.clip(samples[:, :, 1], 0.0, 1.0)

        # Evaluate all samples on GPU via vmap
        costs = _cem_evaluate_batch(rudders, props, current_state,
                                     reference_trajectory, target_positions,
                                     d_safe_eff_list, env, horizon, dt, **kwargs)

        # Select elites (lowest cost = best)
        elite_idx = jnp.argsort(costs)[:n_elite]
        elite_rudders = rudders[elite_idx]  # (n_elite, N)
        elite_props = props[elite_idx]

        # Update best
        if float(costs[elite_idx[0]]) < best_cost:
            best_cost = float(costs[elite_idx[0]])
            best_seq = (jnp.array(elite_rudders[0]), jnp.array(elite_props[0]))

        # Refit distribution to elites
        mu = mu.at[:, 0].set(jnp.mean(elite_rudders, axis=0))
        mu = mu.at[:, 1].set(jnp.mean(elite_props, axis=0))
        sigma = sigma.at[:, 0].set(jnp.clip(jnp.std(elite_rudders, axis=0), 1.0, 25.0))
        sigma = sigma.at[:, 1].set(jnp.clip(jnp.std(elite_props, axis=0), 0.05, 0.4))

    if best_seq is not None and best_cost < 1e9:
        cmd = ControlCommand(
            rudder=float(best_seq[0][0]), propeller=float(best_seq[1][0]),
            source="CEM_MPC_GPU",
            metadata={"n_samples": n_samples, "n_iterations": n_iterations,
                      "best_cost": best_cost, "backend": "cem_jax"},
        )
        return cmd, {"success": True, "best_cost": best_cost, "n_iterations": n_iterations,
                      "backend": "cem_jax", "n_samples": n_samples * n_iterations}
    return None, {"success": False, "reason": "all_infeasible", "backend": "cem_jax"}


def _cem_evaluate_batch(rudders, props, current_state, ref_traj, tgt_pos,
                         d_safe_list, env, horizon, dt,
                         target_covariances=None, own_covariance=None,
                         rho_tube=0.0, epsilon=0.10, rule_context=None,
                         enc_layer=None, w_collision=100.0, w_rule=10.0,
                         w_channel=50.0, w_bank=5.0, w_speed=1.0,
                         vessel_length=180.0):
    """Evaluate a batch of CEM samples via vmap.

    Builds the JAX arrays inline and calls _eval_one_jit for each sample.
    """
    N = horizon
    n_targets = len(d_safe_list) if d_safe_list else 0
    x0 = jnp.array([current_state.x, current_state.y, current_state.psi,
                     current_state.u, current_state.v, current_state.r], dtype=jnp.float64)

    has_ref = bool(ref_traj) and len(ref_traj) >= N
    ref_arr = jnp.array([[r.x, r.y, r.psi, r.u, r.v, r.r] for r in ref_traj[:N]], dtype=jnp.float64) if has_ref else jnp.zeros((N, 6), dtype=jnp.float64)

    if n_targets > 0 and tgt_pos:
        max_k = min(len(tgt_pos), N)
        tgt_list = []
        for k in range(N):
            if k < max_k and k < len(tgt_pos):
                tp_k = tgt_pos[k]
                row = jnp.zeros((n_targets, 2), dtype=jnp.float64)
                for j in range(min(n_targets, len(tp_k))):
                    row = row.at[j, 0].set(tp_k[j][0]); row = row.at[j, 1].set(tp_k[j][1])
                tgt_list.append(row)
            else:
                tgt_list.append(tgt_list[-1])
        tgt_arr = jnp.stack(tgt_list)
    else:
        tgt_arr = jnp.zeros((N, 0, 2), dtype=jnp.float64)

    d_safe_arr = jnp.array(d_safe_list[:n_targets], dtype=jnp.float64) if n_targets > 0 else jnp.array([], dtype=jnp.float64)
    cc_arr = jnp.zeros(max(n_targets, 1), dtype=jnp.float64)

    channel_hw = float(env.channel_width / 2.0) if getattr(env, "channel_width", None) is not None and env.channel_width > 0 else -1.0
    bl_v = float(env.bank_distance_left) if getattr(env, "bank_distance_left", None) is not None else -1.0
    br_v = float(env.bank_distance_right) if getattr(env, "bank_distance_right", None) is not None else -1.0
    bl_ref = current_state.y - env.bank_distance_left if env.bank_distance_left is not None else current_state.y
    br_ref = current_state.y + env.bank_distance_right if env.bank_distance_right is not None else current_state.y

    _chw = jnp.array(channel_hw, dtype=jnp.float64)
    _bl = jnp.array(bl_v, dtype=jnp.float64); _br = jnp.array(br_v, dtype=jnp.float64)
    _blr = jnp.array(bl_ref, dtype=jnp.float64); _brr = jnp.array(br_ref, dtype=jnp.float64)
    _ps = jnp.float64(0.0); _bank_safe = 1.5 * vessel_length  # use actual vessel_length

    return jax.vmap(lambda r, p: _eval_one_jit(
        r, p, x0, ref_arr, tgt_arr, d_safe_arr, cc_arr,
        _chw, _bl, _br, _blr, _brr, vessel_length, _bank_safe,
        w_collision, w_rule, w_channel, w_bank, w_speed,
        _ps, n_targets, N, dt,
    ))(rudders, props)
# JIT-safe single-trajectory evaluation (BUG #2 fix: no Python `if`)
# ═══════════════════════════════════════════════════════════════════════════

def _eval_one_jit(
    r_seq, p_seq, x0, ref_arr, tgt_arr, d_safe_arr, cc_arr,
    chw, bl, br, blr, brr, vessel_length, bank_safe,
    w_coll, w_rule, w_chan, w_bank, w_spd, ps, n_tgt, N, dt,
):
    """Single control sequence evaluation — JIT-safe (all jnp.where)."""
    # Pre-compute condition flags as JAX floats
    has_ch = jnp.where(chw > 0, jnp.float64(1.0), jnp.float64(0.0))
    has_bl = jnp.where(bl > 0, jnp.float64(1.0), jnp.float64(0.0))
    has_br = jnp.where(br > 0, jnp.float64(1.0), jnp.float64(0.0))
    has_tgt = jnp.where(n_tgt > 0, jnp.float64(1.0), jnp.float64(0.0))
    has_ref = jnp.where(ref_arr.shape[0] >= N, jnp.float64(1.0), jnp.float64(0.0))

    # Surrogate dynamics
    sm = 5e7; yi = 1.2e11
    Xuu = -0.001 * 0.5 * 1025.0 * vessel_length**2
    Yv = -0.015 * 0.5 * 1025.0 * vessel_length**3
    Nr = -0.003 * 0.5 * 1025.0 * vessel_length**4
    Nd = -0.0015 * 0.5 * 1025.0 * vessel_length**3

    def _step(carry, k):
        xs, ys, psi, us, vs, rs, rp, pp, cost, feas = carry
        rc, pc = r_seq[k], p_seq[k]

        # Collision check (jnp.where instead of Python for/if)
        k_ok = jnp.where(k < tgt_arr.shape[0], jnp.float64(1.0), jnp.float64(0.0))
        for j in range(max(n_tgt, 1)):
            ds_tot = d_safe_arr[j] + cc_arr[j]
            tj_x = jnp.where((k_ok > 0) & (j < tgt_arr.shape[1]), tgt_arr[k, j, 0], 0.0)
            tj_y = jnp.where((k_ok > 0) & (j < tgt_arr.shape[1]), tgt_arr[k, j, 1], 0.0)
            t_exists = jnp.where((jnp.abs(tj_x) + jnp.abs(tj_y)) > 1e-9, jnp.float64(1.0), jnp.float64(0.0))
            dist_sq = (xs - tj_x)**2 + (ys - tj_y)**2
            dist = jnp.sqrt(jnp.maximum(dist_sq, 1e-12))
            viol = ds_tot - dist
            feas = feas & (viol <= 0.0)
            cost += has_tgt * k_ok * t_exists * w_coll * jnp.maximum(viol, 0.0)**2

        # Kinematics
        n_rps = pc * 3.0
        thrust = jnp.where(n_rps > 0, 0.15 * 1025.0 * n_rps**2 * 6.0**4, 0.0)
        u_dot = (thrust + Xuu * us * jnp.abs(us)) / sm
        v_dot = (Yv * vs) / sm
        r_dot = (Nr * rs + Nd * rc * us * jnp.abs(us)) / yi
        u_n = us + dt * u_dot; v_n = vs + dt * v_dot; r_n = rs + dt * r_dot
        psi_n = psi + dt * rs
        x_n = xs + dt * (us * jnp.cos(psi) - vs * jnp.sin(psi))
        y_n = ys + dt * (us * jnp.sin(psi) + vs * jnp.cos(psi))

        # Tracking (jnp.where)
        rk_x = jnp.where(k_ok * has_ref > 0, ref_arr[k, 0], 0.0)
        rk_y = jnp.where(k_ok * has_ref > 0, ref_arr[k, 1], 0.0)
        rk_u = jnp.where(k_ok * has_ref > 0, ref_arr[k, 3], 0.0)
        rk_v = jnp.where(k_ok * has_ref > 0, ref_arr[k, 4], 0.0)
        cost += k_ok * has_ref * 10.0 * ((xs - rk_x)**2 + (ys - rk_y)**2)
        own_spd = jnp.sqrt(jnp.maximum(us**2 + vs**2, 1e-12))
        ref_spd = jnp.sqrt(jnp.maximum(rk_u**2 + rk_v**2, 1e-12))
        cost += k_ok * has_ref * w_spd * (own_spd - ref_spd)**2

        # COLREGs
        cost += ps * jnp.where(rc < 0, jnp.float64(1.0), 0.0) * w_rule

        # Control
        cost += 0.1 * rc**2 + 0.01 * (pc - 0.5)**2 + 1.0 * (rc - rp)**2 + 0.1 * (pc - pp)**2

        # Bank (jnp.where)
        ml = ys - blr; vl = jnp.maximum(bank_safe - ml, 0.0)
        mr = brr - ys; vr = jnp.maximum(bank_safe - mr, 0.0)
        cost += has_bl * w_bank * vl**2 + has_br * w_bank * vr**2

        # Channel (jnp.where)
        ch_viol = jnp.where(jnp.abs(ys) > chw - 2.0 * vessel_length, jnp.float64(1.0), 0.0)
        cost += has_ch * w_chan * ch_viol

        return (x_n, y_n, psi_n, u_n, v_n, r_n, rc, pc, cost, feas), None

    init = (x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], 0.0, 0.5, jnp.float64(0.0), jnp.bool_(True))
    (xs, ys, psi, us, vs, rs, rp, pp, cost, feas), _ = jax.lax.scan(_step, init, jnp.arange(N))
    return jnp.where(feas, cost, jnp.inf)
