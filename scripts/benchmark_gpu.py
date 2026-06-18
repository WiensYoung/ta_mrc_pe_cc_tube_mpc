#!/usr/bin/env python
"""GPU acceleration benchmark: CasADi (CPU) vs JAX (GPU/CPU).

Usage:
    python scripts/benchmark_gpu.py
    python scripts/benchmark_gpu.py --n-runs 10 --horizon 20
"""

import argparse
import time

import numpy as np

from ta_mrc_pe_cc_tube_mpc.types import (
    ControlCommand, EnvironmentState, TargetShip, VesselParams, VesselState,
)
from ta_mrc_pe_cc_tube_mpc.control.controller import TAMRCPECCTubeMPCController


def make_vessel_params():
    return VesselParams(
        name="TestShip", mmsi="999999001",
        length=180.0, beam=30.0, draught=10.5,
        mass=5e7, Iz=1.2e11,
        max_rudder=35.0, max_rudder_rate=5.0,
        max_speed=15.0, min_speed=0.0,
        max_yaw_rate=3.0, min_turning_radius_multiplier=3.0,
        vessel_type="cargo",
    )


def make_config(backend, horizon=20):
    return {
        "mpc": {
            "backend": backend,
            "horizon": horizon,
            "dt": 0.5,
            "max_iterations": 100,
        },
        "controller": {
            "enable_multi_rule": True,
            "enable_chance_constraint": True,
            "enable_tube_mpc": True,
            "enable_cbf_filter": True,
            "enable_fallback": True,
        },
        "cbf": {},
        "fallback": {"max_consecutive_infeasible": 5},
        "domain": {},
        "rules": {},
        "tube": {"rho_0": 5.0, "rho_maneuver": 2.0},
        "uncertainty": {"epsilon_default": 0.10},
        "jax": {"enable_x64": True, "platform": "gpu", "precompile": True},
    }


def benchmark_single_solve(backend, n_runs=10, horizon=20):
    """Benchmark single MPC solve time."""
    cfg = make_config(backend, horizon)
    vp = make_vessel_params()
    ctrl = TAMRCPECCTubeMPCController(vp, cfg)

    own_state = VesselState(x=0, y=0, psi=0, u=7.0, v=0, r=0)
    tgt_state = VesselState(x=300, y=0, psi=np.pi, u=5.0, v=0, r=0)
    tgt = TargetShip(
        mmsi="TGT001", vessel_type="cargo",
        state=tgt_state, length=180.0, beam=30.0, draught=10.5,
        covariance=np.eye(2) * 100.0,
    )
    env = EnvironmentState()

    # Warm-up
    for _ in range(3):
        ctrl.compute_control(own_state, vp, [tgt], env)

    # Timed runs
    times = []
    for _ in range(n_runs):
        ctrl.reset()
        t0 = time.perf_counter()
        cmd = ctrl.compute_control(own_state, vp, [tgt], env)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "backend": backend,
        "n_runs": n_runs,
        "horizon": horizon,
        "mean_s": np.mean(times),
        "std_s": np.std(times),
        "min_s": np.min(times),
        "max_s": np.max(times),
        "p50_s": np.percentile(times, 50),
        "p95_s": np.percentile(times, 95),
    }


def benchmark_episode(backend, n_steps=50, horizon=20):
    """Benchmark a short closed-loop episode."""
    cfg = make_config(backend, horizon)
    vp = make_vessel_params()
    ctrl = TAMRCPECCTubeMPCController(vp, cfg)

    state = VesselState(x=0, y=0, psi=0, u=7.0, v=0, r=0)
    env = EnvironmentState()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        cmd = ctrl.compute_control(state, vp, [], env)
        # Simple kinematic update
        state = VesselState(
            x=state.x + state.u * 0.5,
            y=state.y + state.v * 0.5,
            psi=state.psi + state.r * 0.5,
            u=state.u, v=state.v, r=state.r,
        )
    t1 = time.perf_counter()

    return {
        "backend": backend,
        "n_steps": n_steps,
        "total_s": t1 - t0,
        "avg_per_step_ms": (t1 - t0) / n_steps * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="GPU acceleration benchmark")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of timed runs")
    parser.add_argument("--horizon", type=int, default=20, help="MPC prediction horizon")
    args = parser.parse_args()

    print("=" * 60)
    print("  TA-MRC-PE-CC-Tube-MPC GPU Benchmark")
    print("=" * 60)

    # Check JAX GPU availability
    try:
        import jax
        devices = jax.devices()
        gpu_available = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)
        print(f"\n  JAX devices: {devices}")
        print(f"  GPU available: {gpu_available}")
    except ImportError:
        print("\n  JAX not installed — skipping GPU benchmarks")
        return

    print(f"\n  Benchmark: {args.n_runs} runs, horizon={args.horizon}")
    print()

    # ── Single MPC solve benchmark ──
    print("--- Single MPC Solve ---")
    for backend in ["casadi", "jax"]:
        try:
            result = benchmark_single_solve(backend, args.n_runs, args.horizon)
            print(f"  {backend:8s}: mean={result['mean_s']*1000:.1f}ms  "
                  f"p95={result['p95_s']*1000:.1f}ms  "
                  f"min={result['min_s']*1000:.1f}ms")
        except Exception as e:
            print(f"  {backend:8s}: FAILED — {e}")

    # ── Episode benchmark ──
    print("\n--- Episode (50 steps) ---")
    for backend in ["casadi", "jax"]:
        try:
            result = benchmark_episode(backend, n_steps=50, horizon=args.horizon)
            print(f"  {backend:8s}: total={result['total_s']:.1f}s  "
                  f"avg={result['avg_per_step_ms']:.1f}ms/step")
        except Exception as e:
            print(f"  {backend:8s}: FAILED — {e}")

    # ── Sampling MPC benchmark (if JAX) ──
    print("\n--- Sampling MPC (500 samples) ---")
    try:
        from ta_mrc_pe_cc_tube_mpc.control.mpc_problem import sampling_based_mpc as sampling_cpu
        from ta_mrc_pe_cc_tube_mpc.control.sampling_jax import jax_sampling_mpc

        own_state = VesselState(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        env = EnvironmentState()
        ref = [own_state] * 21
        tgt_positions = [np.array([[500, 0]])] * 21

        # JAX sampling
        t0 = time.perf_counter()
        cmd_jax, _ = jax_sampling_mpc(
            current_state=own_state,
            reference_trajectory=ref,
            target_positions=tgt_positions,
            d_safe_eff_list=[500.0],
            env=env, n_samples=500, horizon=20, dt=0.5,
            rng=np.random.default_rng(42),
        )
        t_jax = time.perf_counter() - t0
        print(f"  jax_gpu: {t_jax*1000:.1f}ms  (with JAX vmap on {'GPU' if gpu_available else 'CPU'})")
    except Exception as e:
        print(f"  jax_gpu: FAILED — {e}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
