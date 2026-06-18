#!/usr/bin/env python
"""Master experiment runner — runs all phases with checkpoint/resume support.

Usage:
    # Run everything (all 6 phases in order):
    python scripts/run_all.py

    # Run specific phases only:
    python scripts/run_all.py --phases 1 2 3

    # Dry run (print what would be done without running):
    python scripts/run_all.py --dry-run

    # Custom settings:
    python scripts/run_all.py --n-seeds 5 --n-workers 4

Phases:
    0 — Unit tests (validation)
    1 — Ablation experiments
    2 — Core experiments (main results)
    3 — Extended experiments (stress tests)
    4 — Sensitivity experiments
    5 — Statistical analysis + Report generation
"""

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Track whether we're in the middle of a run for graceful shutdown
_cleanup_done = False
_current_phase_started = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
PROCESSED_DIR = RESULTS_DIR / "processed"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

# ── Phase definitions ──────────────────────────────────────────────────────
PHASES = {
    0: {
        "name": "Unit Tests",
        "description": "Run all unit tests to verify environment",
        "output": None,
        "command": "pytest",
        "args": ["-m", "pytest", "tests/", "--tb=short", "-q"],
        "estimated_time": "~10 seconds",
        "critical": True,  # failure stops the pipeline
    },
    1: {
        "name": "Ablation Experiments",
        "description": "12 ablation studies (A1-A12) — module on/off comparison",
        "output": RAW_DIR / "ablation_results.csv",
        "command": "python",
        "args": ["scripts/run_all_ablations.py",
                 "--n-seeds", "{n_seeds}", "--n-workers", "{n_workers}",
                 "--checkpoint", str(CHECKPOINT_DIR)],
        "estimated_time": "~15-30 minutes",
        "critical": False,
        "resume_id": "ablations",
    },
    2: {
        "name": "Core Experiments",
        "description": "8 scenarios x 8 methods x N seeds — main results",
        "output": RAW_DIR / "core" / "metrics_by_episode.csv",
        "command": "python",
        "args": ["scripts/run_all_core.py",
                 "--output", str(RAW_DIR / "core"),
                 "--n-seeds", "{n_seeds}", "--n-workers", "{n_workers}",
                 "--checkpoint", str(CHECKPOINT_DIR / "core_checkpoint.json")],
        "estimated_time": "~3-4 hours (Proposed method uses 80% of time)",
        "critical": True,
        "resume_id": "core",
    },
    3: {
        "name": "Extended Experiments",
        "description": "Stress tests across 3 challenging scenarios (E1-E3)",
        "output": RAW_DIR / "extended_results.csv",
        "command": "python",
        "args": ["scripts/run_all_extended.py",
                 "--n-seeds", "{n_seeds}", "--n-workers", "{n_workers}",
                 "--checkpoint", str(CHECKPOINT_DIR / "extended_checkpoint.json")],
        "estimated_time": "~1-2 hours",
        "critical": False,
        "resume_id": "extended",
    },
    4: {
        "name": "Sensitivity Experiments",
        "description": "6 parameter dimensions x N conditions each",
        "output": RAW_DIR / "sensitivity_results.csv",
        "command": "python",
        "args": ["scripts/run_all_sensitivity.py",
                 "--n-seeds", "{n_seeds_sensitivity}", "--n-workers", "{n_workers}",
                 "--checkpoint", str(CHECKPOINT_DIR)],
        "estimated_time": "~2-4 hours",
        "critical": False,
        "resume_id": "sensitivity",
    },
    5: {
        "name": "Statistical Analysis & Report",
        "description": "Pairwise tests, effect sizes, bootstrap, tables, plots, LaTeX export",
        "output": PROCESSED_DIR / "pairwise_comparisons.csv",
        "command": "python",
        "args": ["scripts/make_report.py",
                 "--input", str(RAW_DIR / "core" / "metrics_by_episode.csv"),
                 "--output_dir", str(PROCESSED_DIR)],
        "estimated_time": "~2-5 minutes",
        "critical": False,
        "post_hook": "_run_enhanced_figures",
    },
}

def _graceful_shutdown(signum, frame):
    """Handle termination signals — print status and let Python unwind."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    print(f"\n\n[{sig_name}] Shutdown signal received.", flush=True)
    if _current_phase_started:
        print("Partial results may be available in results/checkpoints/ and results/raw/.", flush=True)
        print("Re-run with the same --checkpoint path to resume.", flush=True)
    # Let the KeyboardInterrupt / SystemExit propagate naturally for cleanup


def main():
    # Register signal handlers for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _graceful_shutdown)
        except (ValueError, AttributeError):
            pass  # signal not available on this platform
    if hasattr(signal, "SIGBREAK"):
        try:
            signal.signal(signal.SIGBREAK, _graceful_shutdown)
        except (ValueError, AttributeError):
            pass

    parser = argparse.ArgumentParser(
        description="TA-MRC-PE-CC-Tube-MPC — Run All Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "Phases:\n" +
            "\n".join(f"  {k} — {v['name']:40s} {v['estimated_time']}"
                      for k, v in PHASES.items())
        ),
    )
    parser.add_argument("--phases", type=int, nargs="*", default=None,
                        help="Phases to run (default: all). Example: --phases 0 1 2")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python interpreter path.")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Seeds per scenario/method for core/extended (default: 5).")
    parser.add_argument("--n-seeds-sensitivity", type=int, default=3,
                        help="Seeds per condition for sensitivity (default: 3).")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Parallel workers (default: 4). Use 0 for cpu_count.")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                        help="Skip phases whose output already exists (default: True).")
    parser.add_argument("--no-skip", action="store_false", dest="skip_completed",
                        help="Re-run even if output exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing.")
    parser.add_argument("--start-phase", type=int, default=0,
                        help="Start from this phase (default: 0).")
    parser.add_argument("--stop-phase", type=int, default=5,
                        help="Stop after this phase (default: 5).")
    parser.add_argument("--backend", type=str, default=None,
                        help="MPC backend: casadi (default), jax (GPU), scipy, sampling.")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU mode (sets --backend=jax, requires JAX).")
    args = parser.parse_args()

    # Resolve phases to run
    phase_ids = args.phases if args.phases is not None else list(range(6))
    phase_ids = [p for p in phase_ids if args.start_phase <= p <= args.stop_phase]
    phase_ids = sorted(phase_ids)

    # Resolve backend
    backend = args.backend
    if args.gpu:
        backend = "jax"
    if backend:
        print(f"  MPC Backend override: {backend}")

    # Resolve Python interpreter
    python_exe = args.python
    if args.n_workers <= 0:
        import multiprocessing
        args.n_workers = max(1, multiprocessing.cpu_count() - 1)

    # Ensure output directories exist
    for d in [RAW_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, CHECKPOINT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Print banner ───────────────────────────────────────────────────
    print("=" * 72)
    print("  TA-MRC-PE-CC-Tube-MPC — Master Experiment Runner")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {python_exe}")
    print(f"  Phases: {phase_ids}")
    print(f"  Seeds:  core/extended={args.n_seeds}, sensitivity={args.n_seeds_sensitivity}")
    print(f"  Workers: {args.n_workers}")
    print("=" * 72)

    # Verify environment first
    _verify_environment(python_exe)

    # ── Determine which phases to skip ──────────────────────────────────
    plan = []
    for pid in phase_ids:
        info = PHASES[pid]
        output_file = info.get("output")
        completed = output_file and Path(output_file).exists() and args.skip_completed
        plan.append((pid, info, completed))

    if args.dry_run:
        print("\n>>> DRY RUN — no experiments will be executed <<<\n")
        for pid, info, completed in plan:
            status = "[SKIP — output exists]" if completed else "[RUN]"
            print(f"  Phase {pid}: {info['name']:40s} {status}")
            print(f"            {info['description']}")
            print(f"            Est. time: {info['estimated_time']}")
            print()
        return

    # ── Execute phases ──────────────────────────────────────────────────
    overall_start = time.time()
    phase_results = {}
    failures = []

    for pid, info, completed in plan:
        _print_phase_header(pid, info)

        if completed:
            print(f"  [SKIP] Output already exists: {info['output']}")
            phase_results[pid] = "skipped"
            continue

        phase_start = time.time()

        try:
            global _current_phase_started
            _current_phase_started = True
            success = _run_phase(pid, info, python_exe, args, backend)
            _current_phase_started = False
            elapsed = time.time() - phase_start

            if success:
                phase_results[pid] = f"passed ({_fmt_duration(elapsed)})"
                print(f"\n  [OK] Phase {pid} completed in {_fmt_duration(elapsed)}")
            else:
                phase_results[pid] = f"FAILED ({_fmt_duration(elapsed)})"
                failures.append((pid, info["name"]))
                print(f"\n  [FAIL] Phase {pid} failed after {_fmt_duration(elapsed)}")
                if info["critical"]:
                    print(f"  [STOP] Critical phase failed — stopping pipeline.")
                    break
        except KeyboardInterrupt:
            _current_phase_started = False
            print(f"\n  [INTERRUPTED] Phase {pid} stopped by user.")
            phase_results[pid] = "interrupted"
            break
        except Exception as e:
            _current_phase_started = False
            elapsed = time.time() - phase_start
            phase_results[pid] = f"ERROR: {e}"
            failures.append((pid, info["name"]))
            print(f"\n  [ERROR] Phase {pid}: {e}")
            if info["critical"]:
                print(f"  [STOP] Critical phase errored — stopping pipeline.")
                break

    # ── Summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    _print_summary(phase_results, failures, total_elapsed, args)

    # Print output locations
    print("\nOutput files:")
    for label, path in [
        ("Ablation results", RAW_DIR / "ablation_results.csv"),
        ("Core results", RAW_DIR / "core" / "metrics_by_episode.csv"),
        ("Extended results", RAW_DIR / "extended_results.csv"),
        ("Sensitivity results", RAW_DIR / "sensitivity_results.csv"),
        ("Statistics", PROCESSED_DIR),
        ("Tables", TABLES_DIR),
        ("Figures", FIGURES_DIR),
    ]:
        exists = " [EXISTS]" if Path(path).exists() else ""
        print(f"  {label:25s}: {path}{exists}")

    return 0 if not failures else 1


# ── Helpers ─────────────────────────────────────────────────────────────────

def _verify_environment(python_exe: str):
    """Check that Python and key modules are available."""
    print("\n--- Environment Check ---")

    # Check Python version
    result = subprocess.run(
        [python_exe, "--version"], capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    print(f"  Python: {result.stdout.strip()}")

    # Check project imports
    result = subprocess.run(
        [python_exe, "-c",
         "import sys; sys.path.insert(0, 'src'); "
         "from ta_mrc_pe_cc_tube_mpc.control.controller import TAMRCPECCTubeMPCController; "
         "print('  Controller import: OK')"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"  [WARN] Controller import failed:\n{result.stderr[:500]}"
              f"\n  Did you run 'pip install -e .' ?")
    else:
        print(result.stdout.strip())

    # Check critical dependencies
    for module in ["numpy", "scipy", "cvxpy", "pandas", "matplotlib", "yaml"]:
        r = subprocess.run(
            [python_exe, "-c", f"import {module}; print('  {module}:', {module}.__version__)"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        if r.returncode == 0:
            print(r.stdout.strip())
        else:
            print(f"  {module}: MISSING")

    # Check config files
    for cfg in ["default.yaml", "scenarios_core.yaml", "scenarios_extended.yaml"]:
        exists = (CONFIG_DIR / cfg).exists()
        print(f"  configs/{cfg}: {'OK' if exists else 'MISSING'}")

    print()


def _run_phase(pid: int, info: dict, python_exe: str, args, backend: str | None = None) -> bool:
    """Run a single experimental phase. Returns True on success."""
    if pid == 0:
        full_cmd = [python_exe, "-m", "pytest", "tests/", "--tb=short", "-q"]
    elif pid == 5:
        full_cmd = [
            python_exe, str(PROJECT_ROOT / info["args"][0]),
            "--input", str(RAW_DIR / "core" / "metrics_by_episode.csv"),
            "--output_dir", str(PROCESSED_DIR),
        ]
    else:
        # Build: python script.py --arg1 val1 --arg2 val2 ...
        # info["args"] = ["scripts/run_all_xxx.py", "--flag1", "{n_seeds}", ...]
        full_cmd = [python_exe, str(PROJECT_ROOT / info["args"][0])]
        for i in range(1, len(info["args"])):
            arg_val = info["args"][i].format(
                n_seeds=args.n_seeds,
                n_seeds_sensitivity=args.n_seeds_sensitivity,
                n_workers=args.n_workers,
            )
            full_cmd.append(arg_val)

    print(f"\n  Command: {' '.join(full_cmd)}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_PATH) + os.pathsep + env.get("PYTHONPATH", "")
    if backend:
        env["MPC_BACKEND"] = backend
        env["JAX_PLATFORM"] = "gpu" if backend == "jax" else "cpu"

    result = subprocess.run(
        full_cmd,
        cwd=PROJECT_ROOT,
        env=env,
    )

    if result.returncode == 0 and info.get("post_hook"):
        hook_name = info["post_hook"]
        hook_fn = globals().get(hook_name)
        if hook_fn:
            hook_fn()

    return result.returncode == 0


def _run_enhanced_figures():
    """Run statistics, enhanced paper figures, and LaTeX table generation."""
    # First run existing statistics
    stats_script = PROJECT_ROOT / "scripts" / "run_statistical_tests.py"
    if stats_script.exists():
        core_csv = RAW_DIR / "core" / "metrics_by_episode.csv"
        if core_csv.exists():
            cmd = [
                sys.executable, str(stats_script),
                "--input", str(core_csv),
                "--output_dir", str(PROCESSED_DIR),
            ]
            subprocess.run(cmd, cwd=PROJECT_ROOT)
        else:
            print("  [SKIP] No core metrics CSV found for statistical analysis.")

    # Enhanced paper figures (delegates to pub_plots)
    paper_fig_script = PROJECT_ROOT / "scripts" / "plot_paper_figures.py"
    if paper_fig_script.exists():
        core_csv = RAW_DIR / "core" / "metrics_by_episode.csv"
        if core_csv.exists():
            cmd = [
                sys.executable, str(paper_fig_script),
                "--input", str(RAW_DIR / "core"),
                "--output", str(FIGURES_DIR),
            ]
            print(f"\n  Generating enhanced paper figures → {FIGURES_DIR}/")
            subprocess.run(cmd, cwd=PROJECT_ROOT)
        else:
            print("  [SKIP] No core metrics CSV for paper figures.")

    # Generate LaTeX tables
    ablation_csv = RAW_DIR / "ablation_results.csv"
    core_csv = RAW_DIR / "core" / "metrics_by_episode.csv"
    extended_csv = RAW_DIR / "extended_results.csv"
    sensitivity_csv = RAW_DIR / "sensitivity_results.csv"

    # Generate LaTeX tables if core results exist
    if core_csv.exists():
        print(f"\n  Generating LaTeX tables → {TABLES_DIR}/")
        try:
            latex_cmd = [
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, '{SRC_PATH}'); "
                f"import pandas as pd; "
                f"from ta_mrc_pe_cc_tube_mpc.evaluation.report_tables import save_all_tables_latex; "
                f"df = pd.read_csv('{core_csv}'); "
                # Convert DataFrame rows to EpisodeResult-like dicts for the table builders.
                # The table functions expect EpisodeResult objects, but we reconstruct
                # approximate versions from CSV columns for LaTeX generation.
                f"results = df.to_dict('records'); "
                f"outputs = save_all_tables_latex(results, '{TABLES_DIR}'); "
                f"print(f'LaTeX tables generated: ' + ', '.join(outputs.keys()))",
            ]
            subprocess.run(latex_cmd, cwd=PROJECT_ROOT, env={
                **os.environ, "PYTHONPATH": str(SRC_PATH),
            })
        except Exception as e:
            print(f"  [WARN] LaTeX table generation failed: {e}")


def _print_phase_header(pid: int, info: dict):
    print(f"\n{'─' * 72}")
    print(f"  Phase {pid}: {info['name']}")
    print(f"  {info['description']}")
    print(f"  Est. time: {info['estimated_time']}")
    if "resume_id" in info:
        ckpt = CHECKPOINT_DIR / f"{info['resume_id']}_checkpoint.json"
        if ckpt.exists():
            print(f"  Checkpoint found: {ckpt}")
    print(f"{'─' * 72}")


def _print_summary(phase_results: dict, failures: list, total_elapsed: float, args):
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total elapsed: {_fmt_duration(total_elapsed)}")
    print(f"{'=' * 72}")

    for pid, info in PHASES.items():
        if pid in phase_results:
            status = phase_results[pid]
            print(f"  Phase {pid} ({info['name']:35s}): {status}")

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for pid, name in failures:
            print(f"    - Phase {pid}: {name}")
            if PHASES[pid].get("resume_id"):
                print(f"      Re-run to resume from checkpoint.")
        print(f"\n  To resume from phase {failures[0][0]}, run:")
        print(f"    python scripts/run_all.py --start-phase {failures[0][0]}")
    else:
        print(f"\n  All phases completed successfully.")


def _fmt_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        td = timedelta(seconds=int(seconds))
        return str(td)


if __name__ == "__main__":
    sys.exit(main())
