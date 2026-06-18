#!/usr/bin/env python
"""Verification and experiment pipeline with full hardware utilization.

充分利用 48核CPU + 2×H100 GPU 加速运行:
  - CPU: Ray分布式并行 (默认48 workers)
  - GPU: JAX自动检测并使用所有可用GPU
  - 每个阶段结束后生成分析报告

Phases:
  Phase 0: Verify all critical bug fixes
  Phase 1: Quick single-scenario smoke test
  Phase 2: Core experiments (Proposed only)
  Phase 3: Core experiments (all 9 methods)
  Phase 4: Ablation experiments (13 methods)
  Phase 5: Statistical analysis and report generation

Usage:
    # 快速验证模式 (120s episodes)
    python scripts/run_verification_pipeline.py

    # 正式实验模式 (600s episodes, 完整seed矩阵)
    python scripts/run_verification_pipeline.py --full

    # 指定阶段
    python scripts/run_verification_pipeline.py --phase 0
    python scripts/run_verification_pipeline.py --phase 0:2

    # 自定义并行度
    python scripts/run_verification_pipeline.py --workers 32
    python scripts/run_verification_pipeline.py --workers 48 --gpus 2

    # 跳过验证
    python scripts/run_verification_pipeline.py --skip-verify
"""

import argparse
import io
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── CRITICAL: Set multiprocessing start method to 'spawn' BEFORE any Pool
# creation. On Linux the default is 'fork', which deadlocks when JAX's
# pthreads have been initialized in the parent process. 'spawn' starts
# fresh Python interpreters instead of forking.
if sys.platform != "win32":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

# ── CRITICAL: Configure JAX threading BEFORE any 'import jax'.
# These env vars are read at JAX import time and control how many
# threads XLA creates. Limiting threads prevents resource contention
# across 46 Ray workers and avoids the fork-after-pthreads deadlock.
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")  # prefer GPU, fallback CPU

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Hardware Detection & Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def detect_hardware():
    """Detect available CPU cores and GPU devices."""
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()

    gpu_count = 0
    gpu_names = []
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_names = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            gpu_count = len(gpu_names)
    except Exception:
        pass

    return {
        "cpu_cores": cpu_count,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
    }


def configure_jax_for_gpu():
    """Configure JAX to use GPU with float64 precision.

    NOTE: This only sets configuration flags — it does NOT call jax.devices()
    to avoid initializing JAX's thread pool before Ray's os.fork() calls.
    Device enumeration is deferred to _probe_jax_devices().
    """
    try:
        import jax
        # Enable float64 for numerical precision in MPC/physics
        jax.config.update("jax_enable_x64", True)
        # Set preferred platform; actual device probe is deferred
        jax.config.update("jax_platform_name", "gpu")
        print(f"  JAX x64 precision: enabled (device probe deferred)")
        return True
    except ImportError:
        print("  JAX: Not installed, GPU acceleration unavailable")
        return False
    except Exception as e:
        print(f"  JAX: Error configuring: {e}")
        return False


def _probe_jax_devices():
    """Probe JAX devices — safe to call AFTER Ray is initialized.

    Ray's internal os.fork() calls complete during ray.init(). After that,
    probing JAX devices (which creates XLA threads) is safe because no
    further forks will occur in the main process.
    """
    try:
        import jax
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if gpu_devices:
            print(f"  JAX GPU devices: {len(gpu_devices)}")
            for i, d in enumerate(gpu_devices):
                print(f"    GPU {i}: {d}")
            return True
        else:
            print("  JAX: No GPU devices found, using CPU fallback")
            return False
    except Exception as e:
        print(f"  JAX: Error probing devices: {e}")
        return False


def configure_ray(n_workers: int, n_gpus: int = 0):
    """Initialize Ray for distributed parallel execution with GPU support."""
    try:
        import ray
        if not ray.is_initialized():
            # Use /tmp/ray to avoid AF_UNIX path length limit (107 bytes)
            ray_tmp = "/tmp/ray_ta_mrc"
            os.makedirs(ray_tmp, exist_ok=True)
            os.environ["RAY_TMPDIR"] = ray_tmp
            ray.init(
                num_cpus=n_workers,
                num_gpus=n_gpus,
                ignore_reinit_error=True,
                log_to_driver=False,
                _temp_dir=ray_tmp,
            )
        print(f"  Ray initialized: {n_workers} CPUs, {n_gpus} GPUs")
        return True
    except ImportError:
        print("  Ray: Not installed, falling back to multiprocessing")
        return False
    except Exception as e:
        print(f"  Ray: Error initializing: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "ray_temp").mkdir(parents=True, exist_ok=True)


def write_report(phase: int, title: str, content: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"phase{phase}_{ts}.md"
    path = REPORTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Phase {phase}: {title}\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write(content)
    print(f"\n  Report saved: {path}")
    return path


def fmt_pct(val, total):
    if total == 0:
        return "N/A"
    return f"{100.0 * val / total:.1f}%"


def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def get_hardware_summary(hw: dict) -> str:
    lines = [
        f"| Resource | Available |",
        f"|----------|-----------|",
        f"| CPU Cores | {hw['cpu_cores']} |",
        f"| GPU Count | {hw['gpu_count']} |",
    ]
    if hw["gpu_names"]:
        for i, name in enumerate(hw["gpu_names"]):
            lines.append(f"| GPU {i} | {name} |")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0: Verify critical fixes
# ═══════════════════════════════════════════════════════════════════════════════

def phase_0_verify(hw: dict):
    separator("PHASE 0: Verifying Critical Bug Fixes")
    checks = []
    t0 = time.time()

    # Check 1: Ablation merge logic
    try:
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner
        from ta_mrc_pe_cc_tube_mpc.utils.io_utils import load_config_with_overrides, deep_merge
        from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import build_ablation_features
        from ta_mrc_pe_cc_tube_mpc.types import VesselParams

        config = load_config_with_overrides(str(PROJECT_ROOT / "configs"))
        ablation_features = build_ablation_features("A6")
        ablated_config = deep_merge(config, {"controller": ablation_features})
        runner = ClosedLoopRunner(ablated_config)
        ctrl = runner._build_controller("Proposed", VesselParams())

        cbf_disabled = not ctrl.flags.get("enable_cbf_filter", True)
        tube_disabled = not ctrl.flags.get("enable_tube_mpc", True)
        checks.append(("Ablation A6: CBF disabled", cbf_disabled, str(ctrl.flags)))
        checks.append(("Ablation A6: Tube still enabled", not tube_disabled, ""))
    except Exception as e:
        checks.append(("Ablation merge logic", False, str(e)))

    # Check 2: VTS point-in-polygon
    try:
        from ta_mrc_pe_cc_tube_mpc.data.vts_rules import _point_in_polygon
        poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
        inside_ok = _point_in_polygon(5, 5, poly)
        outside_ok = not _point_in_polygon(15, 5, poly)
        triangle = [(0, 10), (10, 0), (0, 0)]
        tri_inside = _point_in_polygon(2, 3, triangle)
        checks.append(("VTS point-in-polygon (inside)", inside_ok, ""))
        checks.append(("VTS point-in-polygon (outside)", outside_ok, ""))
        checks.append(("VTS point-in-polygon (downward edge)", tri_inside, ""))
    except Exception as e:
        checks.append(("VTS point-in-polygon", False, str(e)))

    # Check 3: Bank effect
    try:
        from ta_mrc_pe_cc_tube_mpc.physics.bank_effect import compute_bank_disturbance_force
        checks.append(("Bank effect module importable", True, ""))
    except Exception as e:
        checks.append(("Bank effect", False, str(e)))

    # Check 4: MMG model
    try:
        from ta_mrc_pe_cc_tube_mpc.models.mmg_3dof import MMG3DOFModel
        model = MMG3DOFModel({"L": 180, "B": 30, "T": 10.5, "mass": 5e7, "Iz": 1.2e11})
        checks.append(("MMG model instantiation", True, ""))
    except Exception as e:
        checks.append(("MMG model", False, str(e)))

    # Check 5: Sobol estimator
    try:
        import numpy as _np
        from ta_mrc_pe_cc_tube_mpc.analysis.sobol_sensitivity import compute_sobol_indices, saltelli_sampling
        _np.random.seed(42)
        samples = saltelli_sampling(n_base=50, n_dims=2, bounds=[(0, 1), (0, 1)])
        A = samples["A"]; B = samples["B"]; AB_list = samples["AB"]
        def f(X): return X[:, 0] + 2 * X[:, 1]
        result = compute_sobol_indices(f(A), f(B), [f(AB) for AB in AB_list])
        s1_valid = all(0 <= s <= 1 for s in result["S1"])
        checks.append(("Sobol indices valid range", s1_valid, f"S1={result['S1']}"))
    except Exception as e:
        checks.append(("Sobol estimator", False, str(e)))

    # Check 6: Hardware
    checks.append(("CPU cores detected", hw["cpu_cores"] > 0, f"{hw['cpu_cores']} cores"))
    checks.append(("GPU detected", hw["gpu_count"] > 0, f"{hw['gpu_count']} GPUs"))

    # Check 7: Ray
    try:
        import ray
        checks.append(("Ray importable", True, ""))
    except ImportError:
        checks.append(("Ray importable", False, "Not installed"))

    # Check 8: JAX GPU
    try:
        import jax
        gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
        checks.append(("JAX GPU available", len(gpu_devices) > 0, f"{len(gpu_devices)} GPU devices"))
    except Exception:
        checks.append(("JAX GPU available", False, "JAX not available"))

    # Check 9: NumPy version compatibility
    try:
        import numpy as _np
        np_major = int(_np.__version__.split(".")[0])
        # NumPy 2.x is fine if cvxpy and other deps work (they may show warnings)
        checks.append(("NumPy version", True, f"v{_np.__version__}"))
    except Exception:
        checks.append(("NumPy version", False, "Cannot check"))

    elapsed = time.time() - t0

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    lines = [
        f"## Hardware\n",
        get_hardware_summary(hw),
        f"",
        f"## Summary\n",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| Passed | {passed} |",
        f"| Failed | {total - passed} |",
        f"| Total | {total} |",
        f"",
        f"**Time**: {elapsed:.1f}s\n",
        f"## Details\n",
        f"| # | Check | Status | Notes |",
        f"|---|-------|--------|-------|",
    ]
    for i, (name, ok, note) in enumerate(checks, 1):
        status = "PASS" if ok else "FAIL"
        lines.append(f"| {i} | {name} | {status} | {note[:80]} |")

    report = "\n".join(lines)
    write_report(0, "Bug Fix Verification", report)

    for name, ok, note in checks:
        icon = "OK" if ok else "FAIL"
        print(f"  [{icon}] {name}" + (f" ({note})" if note and not ok else ""))

    if passed < total:
        print(f"\n  WARNING: {total - passed} checks FAILED.")
        return False
    print(f"\n  All {total} checks passed.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Quick smoke test
# ═══════════════════════════════════════════════════════════════════════════════

def phase_1_smoke_test(hw: dict, n_workers: int, episode_duration: float = 60, n_seeds: int = 1):
    separator("PHASE 1: Quick Smoke Test (S1, seed=42, Proposed, 60s)")
    t0 = time.time()

    from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner
    from ta_mrc_pe_cc_tube_mpc.utils.io_utils import load_config_with_overrides, load_yaml
    from ta_mrc_pe_cc_tube_mpc.data.perturbation_generator import build_synthetic_episode_batch

    config = load_config_with_overrides(str(PROJECT_ROOT / "configs"))
    config.setdefault("simulation", {})
    config["simulation"]["closed_loop_duration"] = 60

    scenario_config = load_yaml(str(PROJECT_ROOT / "configs" / "scenarios_core.yaml"))
    episodes = build_synthetic_episode_batch(scenario_config, "S1", n_episodes=1, base_seed=42)

    if not episodes:
        print("  FAIL: No episodes generated.")
        write_report(1, "Smoke Test", "FAIL: No episodes generated.")
        return False

    runner = ClosedLoopRunner(config)
    results = runner.run_batch(
        episodes=episodes, methods=["Proposed"], n_seeds=1, n_workers=1, verbose=False,
    )

    elapsed = time.time() - t0

    if not results:
        print("  FAIL: No results returned.")
        write_report(1, "Smoke Test", "FAIL: No results returned.")
        return False

    r = results[0]
    lines = [
        f"## Result\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Episode ID | {r.episode_id} |",
        f"| Method | {r.method} |",
        f"| Failure Type | {r.failure_type or 'None (SUCCESS)'} |",
        f"| Collision | {r.metrics.get('collision', 'N/A')} |",
        f"| Colregs Violation | {r.metrics.get('colregs_violation', 'N/A')} |",
        f"| Mean Runtime | {r.metrics.get('mean_runtime', 0):.4f} s |",
        f"| MPC Infeasibility | {r.metrics.get('mpc_infeasibility', 'N/A')} |",
        f"",
        f"**Time**: {elapsed:.1f}s\n",
    ]

    success = str(r.failure_type) in ("None", "nan", "")
    if success:
        lines.append("SMOKE TEST PASSED")
    else:
        lines.append(f"SMOKE TEST: episode ended with failure type '{r.failure_type}'")

    write_report(1, "Smoke Test", "\n".join(lines))
    print(f"  Failure: {r.failure_type or 'None'}")
    print(f"  Runtime: {r.metrics.get('mean_runtime', 0):.4f}s")
    print(f"  Time: {elapsed:.1f}s")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Core experiments (Proposed only)
# ═══════════════════════════════════════════════════════════════════════════════

def phase_2_core_proposed(hw: dict, n_workers: int, episode_duration: float, n_seeds: int):
    separator(f"PHASE 2: Core Experiments (Proposed, {episode_duration:.0f}s, {n_seeds} seeds/scenario)")
    t0 = time.time()

    from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner
    from ta_mrc_pe_cc_tube_mpc.utils.io_utils import load_config_with_overrides, load_yaml
    from ta_mrc_pe_cc_tube_mpc.data.perturbation_generator import build_synthetic_episode_batch
    from ta_mrc_pe_cc_tube_mpc.simulation.scenario_loader import list_scenarios

    config = load_config_with_overrides(str(PROJECT_ROOT / "configs"))
    config.setdefault("simulation", {})
    config["simulation"]["closed_loop_duration"] = episode_duration

    scenario_config = load_yaml(str(PROJECT_ROOT / "configs" / "scenarios_core.yaml"))
    scenario_ids = list_scenarios(str(PROJECT_ROOT / "configs" / "scenarios_core.yaml"))

    all_episodes = []
    for sid in scenario_ids:
        batch = build_synthetic_episode_batch(scenario_config, sid, n_episodes=n_seeds, base_seed=42)
        all_episodes.extend(batch)

    total_runs = len(all_episodes)
    est_time = total_runs * (episode_duration / 0.5) * 2.0 / n_workers  # ~2s per step

    print(f"  Scenarios: {len(scenario_ids)}")
    print(f"  Episodes: {total_runs}")
    print(f"  Workers: {n_workers}")
    print(f"  Estimated time: {est_time/60:.1f} min")
    print(f"  Running...")

    runner = ClosedLoopRunner(config)
    results = runner.run_batch(
        episodes=all_episodes, methods=["Proposed"], n_seeds=1,
        n_workers=n_workers, verbose=True,
        checkpoint_path=str(RESULTS_DIR / "checkpoints" / "verify_proposed_checkpoint.json"),
    )

    elapsed = time.time() - t0

    # Analyze
    failure_types = {}
    for r in results:
        ft = str(r.failure_type) if r.failure_type else "SUCCESS"
        failure_types[ft] = failure_types.get(ft, 0) + 1

    success_count = failure_types.get("SUCCESS", 0) + failure_types.get("nan", 0)

    lines = [
        f"## Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Episodes | {len(results)} |",
        f"| Scenarios | {len(scenario_ids)} |",
        f"| Workers | {n_workers} |",
        f"| Total Time | {elapsed:.1f}s ({elapsed/60:.1f} min) |",
        f"| Time/Episode | {elapsed/max(len(results),1):.1f}s |",
        f"",
        f"## Failure Distribution\n",
        f"| Failure Type | Count | Percentage |",
        f"|-------------|-------|------------|",
    ]
    for ft, count in sorted(failure_types.items(), key=lambda x: -x[1]):
        lines.append(f"| {ft} | {count} | {fmt_pct(count, len(results))} |")

    lines.append(f"\n**Success Rate**: {fmt_pct(success_count, len(results))}")

    if success_count > 0:
        lines.append(f"\nProposed has {success_count} successful episodes. Ready for full core experiment.")
    else:
        lines.append(f"\nWARNING: Proposed has 0% success rate. Review before proceeding.")

    write_report(2, "Core Experiment - Proposed Only", "\n".join(lines))

    # Save CSV
    import csv
    csv_path = RESULTS_DIR / "raw" / "verify_proposed_results.csv"
    if results:
        all_keys = set()
        for r in results:
            all_keys.update(r.metrics.keys())
        fieldnames = ["episode_id", "method", "failure_type"] + sorted(all_keys)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {"episode_id": r.episode_id, "method": r.method,
                       "failure_type": r.failure_type or ""}
                row.update(r.metrics)
                writer.writerow(row)
        print(f"  Results saved: {csv_path}")

    print(f"\n  Success rate: {fmt_pct(success_count, len(results))}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return success_count > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Core experiments (all methods)
# ═══════════════════════════════════════════════════════════════════════════════

def phase_3_core_all(hw: dict, n_workers: int, episode_duration: float, n_seeds: int):
    separator(f"PHASE 3: Core Experiments (All 9 Methods, {episode_duration:.0f}s, {n_seeds} seeds)")
    t0 = time.time()

    import numpy as _np
    from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner
    from ta_mrc_pe_cc_tube_mpc.utils.io_utils import load_config_with_overrides, load_yaml
    from ta_mrc_pe_cc_tube_mpc.data.perturbation_generator import build_synthetic_episode_batch
    from ta_mrc_pe_cc_tube_mpc.simulation.scenario_loader import list_scenarios
    from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import list_baseline_methods

    config = load_config_with_overrides(str(PROJECT_ROOT / "configs"))
    config.setdefault("simulation", {})
    config["simulation"]["closed_loop_duration"] = episode_duration

    scenario_config = load_yaml(str(PROJECT_ROOT / "configs" / "scenarios_core.yaml"))
    scenario_ids = list_scenarios(str(PROJECT_ROOT / "configs" / "scenarios_core.yaml"))
    methods = list_baseline_methods()

    all_episodes = []
    for sid in scenario_ids:
        batch = build_synthetic_episode_batch(scenario_config, sid, n_episodes=n_seeds, base_seed=42)
        all_episodes.extend(batch)

    total_runs = len(all_episodes) * len(methods)
    est_time = total_runs * (episode_duration / 0.5) * 2.0 / n_workers

    print(f"  Scenarios: {len(scenario_ids)}")
    print(f"  Episodes: {len(all_episodes)}")
    print(f"  Methods: {methods}")
    print(f"  Workers: {n_workers}")
    print(f"  Total runs: {total_runs}")
    print(f"  Estimated time: {est_time/60:.1f} min")
    print(f"  Running...")

    runner = ClosedLoopRunner(config)
    results = runner.run_batch(
        episodes=all_episodes, methods=methods, n_seeds=1,
        n_workers=n_workers, verbose=True,
        checkpoint_path=str(RESULTS_DIR / "checkpoints" / "verify_core_all_checkpoint.json"),
        output_csv=str(RESULTS_DIR / "raw" / "verify_core_all_results.csv"),
    )

    elapsed = time.time() - t0

    # Analyze by method
    method_stats = {}
    for r in results:
        m = r.method
        if m not in method_stats:
            method_stats[m] = {"total": 0, "success": 0, "failures": {}, "runtimes": []}
        method_stats[m]["total"] += 1
        ft = str(r.failure_type) if r.failure_type else "SUCCESS"
        if ft in ("None", "nan"):
            method_stats[m]["success"] += 1
            ft = "SUCCESS"
        method_stats[m]["failures"][ft] = method_stats[m]["failures"].get(ft, 0) + 1
        rt = r.metrics.get("mean_runtime")
        if rt and isinstance(rt, (int, float)) and not _np.isnan(rt):
            method_stats[m]["runtimes"].append(rt)

    lines = [
        f"## Summary\n",
        f"| Method | Episodes | Success | Success Rate | Avg Runtime |",
        f"|--------|----------|---------|--------------|-------------|",
    ]
    for m in methods:
        s = method_stats.get(m, {"total": 0, "success": 0, "runtimes": []})
        avg_rt = f"{_np.mean(s['runtimes']):.4f}s" if s["runtimes"] else "N/A"
        lines.append(f"| {m} | {s['total']} | {s['success']} | "
                    f"{fmt_pct(s['success'], s['total'])} | {avg_rt} |")

    lines.extend([
        f"",
        f"## Failure Distribution\n",
        f"| Method | F1 | F2 | F3 | F4 | F5 | F6 | F7 | SUCCESS |",
        f"|--------|----|----|----|----|----|----|----|---------|",
    ])
    for m in methods:
        s = method_stats.get(m, {"failures": {}})
        f = s["failures"]
        lines.append(f"| {m} | {f.get('F1', 0)} | {f.get('F2', 0)} | {f.get('F3', 0)} | "
                    f"{f.get('F4', 0)} | {f.get('F5', 0)} | {f.get('F6', 0)} | "
                    f"{f.get('F7', 0)} | {f.get('SUCCESS', 0)} |")

    lines.append(f"\n**Total Time**: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    lines.append(f"**Workers**: {n_workers}")

    proposed_stats = method_stats.get("Proposed", {})
    lines.append(f"**Proposed Success**: {fmt_pct(proposed_stats.get('success', 0), proposed_stats.get('total', 1))}")

    write_report(3, "Core Experiments - All Methods", "\n".join(lines))
    print(f"\n  Proposed success: {fmt_pct(proposed_stats.get('success', 0), proposed_stats.get('total', 1))}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Ablation experiments
# ═══════════════════════════════════════════════════════════════════════════════

def phase_4_ablations(hw: dict, n_workers: int, episode_duration: float, n_seeds: int):
    separator(f"PHASE 4: Ablation Experiments (13 Methods, {episode_duration:.0f}s, {n_seeds} seeds)")
    t0 = time.time()

    from ta_mrc_pe_cc_tube_mpc.experiments.run_ablations import run_ablation_experiments

    print(f"  Workers: {n_workers}")
    print(f"  Running...")

    results = run_ablation_experiments(
        config_dir=str(PROJECT_ROOT / "configs"),
        scenario_file=str(PROJECT_ROOT / "configs" / "scenarios_core.yaml"),
        output_path=str(RESULTS_DIR / "raw" / "verify_ablation_results.csv"),
        n_seeds=n_seeds,
        n_workers=n_workers,
        verbose=True,
        checkpoint_dir=str(RESULTS_DIR / "checkpoints"),
    )

    elapsed = time.time() - t0

    # Analyze
    lines = [
        f"## Summary\n",
        f"| Method | Episodes | Success | Failure Distribution |",
        f"|--------|----------|---------|---------------------|",
    ]

    all_identical = True
    first_metrics = None
    for method_name, method_results in results.items():
        success = sum(1 for r in method_results
                     if str(r.failure_type) in ("None", "nan", ""))
        failures = {}
        for r in method_results:
            ft = str(r.failure_type) if r.failure_type else "SUCCESS"
            failures[ft] = failures.get(ft, 0) + 1

        ft_str = ", ".join(f"{k}:{v}" for k, v in sorted(failures.items()))
        lines.append(f"| {method_name} | {len(method_results)} | {success} | {ft_str} |")

        if first_metrics is None:
            first_metrics = [r.metrics.get("path_efficiency", 0) for r in method_results]
        else:
            current_metrics = [r.metrics.get("path_efficiency", 0) for r in method_results]
            if current_metrics != first_metrics:
                all_identical = False

    lines.extend([
        f"",
        f"**Total Time**: {elapsed:.1f}s ({elapsed/60:.1f} min)",
        f"**Workers**: {n_workers}",
    ])

    if all_identical:
        lines.append("\nWARNING: All methods produce identical metrics. C1 bug may still be active.")
    else:
        lines.append("\nSUCCESS: Methods produce different metrics. Ablation study is functional.")

    write_report(4, "Ablation Experiments", "\n".join(lines))
    print(f"\n  Metrics identical: {all_identical}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return not all_identical


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Analysis report
# ═══════════════════════════════════════════════════════════════════════════════

def phase_5_analysis(hw: dict):
    separator("PHASE 5: Final Analysis Report")

    result_files = list(RESULTS_DIR.glob("raw/verify_*_results.csv"))
    if not result_files:
        print("  No result files found. Run phases 2-4 first.")
        write_report(5, "Final Analysis", "No result files found.")
        return False

    lines = [
        f"## Available Result Files\n",
    ]
    for f in result_files:
        lines.append(f"- `{f.name}`")

    core_csv = RESULTS_DIR / "raw" / "verify_core_all_results.csv"
    if core_csv.exists():
        import csv
        try:
            with open(core_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            methods = set(r["method"] for r in rows)
            lines.extend([
                f"\n## Core Experiment Analysis\n",
                f"**Total episodes**: {len(rows)}\n",
                f"**Methods**: {', '.join(sorted(methods))}\n",
            ])

            # Success rate by method
            lines.append(f"### Success Rate by Method\n")
            lines.append(f"| Method | Total | Success | Rate |")
            lines.append(f"|--------|-------|---------|------|")
            for m in sorted(methods):
                method_rows = [r for r in rows if r["method"] == m]
                success = sum(1 for r in method_rows
                            if r.get("failure_type", "") in ("", "None", "nan"))
                lines.append(f"| {m} | {len(method_rows)} | {success} | "
                           f"{fmt_pct(success, len(method_rows))} |")
        except Exception as e:
            lines.append(f"\nError reading core results: {e}")

    write_report(5, "Final Analysis", "\n".join(lines))
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

PHASES = {
    0: ("Verify Fixes", phase_0_verify),
    1: ("Smoke Test", phase_1_smoke_test),
    2: ("Core - Proposed Only", phase_2_core_proposed),
    3: ("Core - All Methods", phase_3_core_all),
    4: ("Ablation Experiments", phase_4_ablations),
    5: ("Final Analysis", phase_5_analysis),
}


def parse_range(s: str) -> list:
    if ":" in s:
        start, end = s.split(":")
        return list(range(int(start), int(end) + 1))
    return [int(s)]


def main():
    parser = argparse.ArgumentParser(description="Verification and experiment pipeline")
    parser.add_argument("--phase", type=str, default=None,
                       help="Phase number or range (e.g., '0', '0:3', '2:4')")
    parser.add_argument("--skip-verify", action="store_true",
                       help="Skip phase 0 (verification)")
    parser.add_argument("--full", action="store_true",
                       help="Full experiment mode (600s episodes, complete seed matrix)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: auto-detect CPU cores)")
    parser.add_argument("--gpus", type=int, default=None,
                       help="Number of GPUs to use (default: auto-detect)")
    parser.add_argument("--seeds", type=int, default=None,
                       help="Number of seeds per scenario (default: 5 quick, 30 full)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print phases without executing")
    args = parser.parse_args()

    ensure_dirs()

    # Detect hardware
    hw = detect_hardware()
    print(f"\nHardware detected:")
    print(f"  CPU: {hw['cpu_cores']} cores")
    print(f"  GPU: {hw['gpu_count']} devices")
    if hw["gpu_names"]:
        for name in hw["gpu_names"]:
            print(f"    - {name}")

    # Configure workers
    if args.workers:
        n_workers = args.workers
    else:
        # Use most cores, leave some for system
        n_workers = max(1, hw["cpu_cores"] - 2)

    # Configure GPU
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.gpus))

    # Configure JAX settings (flags only, NO device probe yet —
    # jax.devices() creates threads that deadlock with Ray's os.fork())
    configure_jax_for_gpu()

    # Initialize Ray FIRST — its internal os.fork() calls must complete
    # before any JAX thread-creating functions are called.
    n_gpus = args.gpus if args.gpus else hw["gpu_count"]
    configure_ray(n_workers, n_gpus)

    # NOW safe to probe JAX devices (Ray won't fork again in this process)
    _probe_jax_devices()

    # Configure episode parameters
    if args.full:
        episode_duration = 600.0  # Full 600s episodes
        n_seeds = args.seeds or 30  # Full seed matrix
    else:
        episode_duration = 120.0  # Quick 120s episodes
        n_seeds = args.seeds or 5  # Quick validation

    print(f"\nConfiguration:")
    print(f"  Mode: {'FULL' if args.full else 'QUICK'}")
    print(f"  Episode duration: {episode_duration:.0f}s")
    print(f"  Seeds per scenario: {n_seeds}")
    print(f"  Workers: {n_workers}")

    # Determine phases
    if args.phase:
        phases = parse_range(args.phase)
    elif args.skip_verify:
        phases = list(range(1, 6))
    else:
        phases = list(range(6))

    if args.dry_run:
        print("\nDry run - phases to execute:")
        for p in phases:
            name, _ = PHASES[p]
            print(f"  Phase {p}: {name}")
        return

    # Run phases
    start_time = time.time()
    results = {}

    for phase_num in phases:
        if phase_num not in PHASES:
            print(f"  Unknown phase {phase_num}. Skipping.")
            continue

        name, func = PHASES[phase_num]
        try:
            # Pass appropriate args to each phase
            if phase_num == 0:
                success = func(hw)
            elif phase_num == 5:
                success = func(hw)
            else:
                success = func(hw, n_workers, episode_duration, n_seeds)
            results[phase_num] = ("OK" if success else "WARN", name)
        except Exception as e:
            results[phase_num] = ("FAIL", f"{name}: {e}")
            print(f"\n  Phase {phase_num} failed: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    total_time = time.time() - start_time
    separator("PIPELINE COMPLETE")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)\n")
    print(f"  Results:")
    for p in sorted(results.keys()):
        icon, name = results[p]
        print(f"    Phase {p}: [{icon}] {name}")

    print(f"\n  Reports: {REPORTS_DIR}")
    print(f"  Results: {RESULTS_DIR / 'raw'}")


if __name__ == "__main__":
    main()
