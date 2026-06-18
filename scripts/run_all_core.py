#!/usr/bin/env python
"""Authoritative core-experiment entry point — fully reproducible.

Usage:
    # Dry-run (print plan, no simulation)
    python scripts/run_all_core.py --dry-run

    # Quick smoke test (1 episode × 1 seed, B3 + Proposed)
    python scripts/run_all_core.py --quick --output results/quick

    # Full experiment
    python scripts/run_all_core.py --n-seeds 5 --output results/core

Output directory (--output) will contain:
    resolved_config.yaml          ← merged config actually used
    method_feature_table.csv      ← per-method feature flags
    metrics_by_episode.csv        ← raw episode results
    runtime_summary.csv           ← timing per method
    failure_cases.csv             ← episodes with failures
    statistical_tests.csv         ← pairwise comparisons
    README_run.md                 ← run metadata
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
    BASELINE_REGISTRY,
    list_baseline_methods,
    validate_registry,
)
from ta_mrc_pe_cc_tube_mpc.utils.io_utils import (
    deep_merge,
    load_config_with_overrides,
    load_yaml,
    save_yaml,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run all core collision avoidance experiments.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to default config YAML.")
    parser.add_argument("--output", default="results/core",
                        help="Output directory.")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Seeds per (episode, method).")
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Perturbed episodes per scenario.")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Methods (default: all 8 from registry).")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Scenarios (default: all from scenarios_core.yaml).")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--checkpoint", default=None)

    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test: 1 episode × 1 seed, B3 + Proposed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment plan and exit (no simulation).")

    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, args.output)

    # ── Resolve config ──────────────────────────────────────────────────
    config_path = os.path.join(project_root, args.config)
    if os.path.isfile(config_path):
        config_dir = os.path.dirname(config_path)
    else:
        config_dir = os.path.join(project_root, "configs")

    config = load_config_with_overrides(config_dir)

    # ── Validate registry before anything else ──────────────────────────
    issues = validate_registry()
    if issues:
        print("FATAL: Baseline registry validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    # ── Quick mode overrides ────────────────────────────────────────────
    if args.quick:
        args.n_seeds = 1
        args.n_episodes = 1
        args.methods = args.methods or ["B3", "Proposed"]
        args.scenarios = args.scenarios or ["S1", "S2"]
        print("=" * 60)
        print("QUICK SMOKE-TEST MODE")
        print(f"  Seeds per run:    {args.n_seeds}")
        print(f"  Episodes per scenario: {args.n_episodes}")
        print(f"  Methods:          {args.methods}")
        print(f"  Scenarios:        {args.scenarios}")
        total = len(args.scenarios) * args.n_episodes * len(args.methods) * args.n_seeds
        print(f"  Total runs:       {total}")
        print("=" * 60)

    methods = args.methods or list_baseline_methods()
    scenario_file = os.path.join(project_root, "configs", "scenarios_core.yaml")
    scenario_config = load_yaml(scenario_file)
    all_scenario_ids = args.scenarios or list(scenario_config.get("scenarios", {}).keys())

    # ── Dry-run ─────────────────────────────────────────────────────────
    if args.dry_run:
        _print_dry_run(
            methods, all_scenario_ids, args.n_episodes, args.n_seeds,
            config, output_dir,
        )
        return

    # ── Ensure output directory ─────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # ── Save resolved config ────────────────────────────────────────────
    resolved_path = os.path.join(output_dir, "resolved_config.yaml")
    save_yaml(dict(config), resolved_path)
    print(f"Resolved config → {resolved_path}")

    # ── Save method feature table ───────────────────────────────────────
    _save_method_feature_table(output_dir)
    print(f"Method feature table → {output_dir}/method_feature_table.csv")

    # ── Save git info ───────────────────────────────────────────────────
    _save_git_info(output_dir, project_root)

    # ── Save dependency versions for reproducibility ────────────────────
    _save_dependency_versions(output_dir)

    # ── Build episode batch ─────────────────────────────────────────────
    from ta_mrc_pe_cc_tube_mpc.data.perturbation_generator import (
        build_synthetic_episode_batch,
    )
    from ta_mrc_pe_cc_tube_mpc.experiments.run_core_experiments import (
        _deterministic_hash,
        get_scenario_weights,
        compute_weighted_episode_counts,
    )

    # Load scenario weights for balanced multi-ship coverage (GAP #10)
    scenario_weights = get_scenario_weights(
        os.path.join(project_root, "configs", "statistics.yaml")
    )
    weighted_counts = compute_weighted_episode_counts(
        all_scenario_ids, args.n_episodes, scenario_weights,
    )

    all_episodes = []
    for sid in all_scenario_ids:
        if sid not in scenario_config.get("scenarios", {}):
            warnings.warn(f"Scenario '{sid}' not found in {scenario_file}, skipping")
            continue
        n_ep = weighted_counts.get(sid, args.n_episodes)
        batch = build_synthetic_episode_batch(
            scenario_config, sid,
            n_episodes=n_ep,
            base_seed=args.seed_offset + _deterministic_hash(sid),
        )
        all_episodes.extend(batch)
        if batch:
            weight_note = f" (weight={scenario_weights.get(sid, 1.0):.1f}x)" if scenario_weights.get(sid, 1.0) != 1.0 else ""
            print(f"  {sid}: {len(batch)} episodes{weight_note}")

    print(f"Total: {len(all_episodes)} episodes × {len(methods)} methods × "
          f"{args.n_seeds} seeds = "
          f"{len(all_episodes) * len(methods) * args.n_seeds} runs")

    # ── Run experiments ─────────────────────────────────────────────────
    from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner

    t_start = time.time()
    runner = ClosedLoopRunner(config)
    results = runner.run_batch(
        episodes=all_episodes,
        methods=methods,
        n_seeds=args.n_seeds,
        seed_offset=args.seed_offset,
        verbose=True,
        checkpoint_path=args.checkpoint,
        n_workers=args.n_workers,
        output_csv=os.path.join(output_dir, "metrics_by_episode.csv"),
    )
    elapsed = time.time() - t_start

    # ── Post-process outputs ────────────────────────────────────────────
    _save_runtime_summary(results, output_dir)
    _save_failure_cases(results, output_dir)
    _save_metrics_by_scenario_method(results, output_dir)
    _save_readme_run(output_dir, args, methods, all_scenario_ids, elapsed, len(results))

    print(f"\nCompleted {len(results)} episodes in {elapsed:.0f}s")
    print(f"Output → {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        print(f"  {f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _print_dry_run(methods, scenarios, n_ep, n_seeds, config, output_dir):
    """Print experiment plan without running."""
    print("=" * 60)
    print("DRY-RUN — Experiment Plan")
    print("=" * 60)
    print(f"Backend:  {config.get('mpc', {}).get('backend', '?')}")
    print(f"Horizon:  {config.get('mpc', {}).get('horizon', '?')}")
    print(f"dt:       {config.get('mpc', {}).get('dt', '?')}")
    print(f"Scenarios: {len(scenarios)} — {scenarios}")
    print(f"Methods:   {len(methods)} — {methods}")
    print(f"Episodes per scenario: {n_ep}")
    print(f"Seeds per run:        {n_seeds}")
    total = len(scenarios) * n_ep * len(methods) * n_seeds
    print(f"Total runs:           {total}")
    print(f"Output dir:           {output_dir}")
    print()
    print("Feature flags per method:")
    for m in methods:
        feats = BASELINE_REGISTRY.get(m, {}).get("features", {})
        enabled = [k for k, v in feats.items() if v]
        print(f"  {m:10s}: {len(enabled)} enabled — {sorted(enabled)[:5]}...")
    print("=" * 60)


def _save_method_feature_table(output_dir: str):
    """Save CSV of method → feature flags."""
    rows = []
    for method in list_baseline_methods():
        entry = BASELINE_REGISTRY[method]
        row = {
            "method": method,
            "label": entry["label"],
            "controller": entry["controller"],
            "backend_override": entry.get("backend_override") or "",
        }
        row.update(entry["features"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "method_feature_table.csv"), index=False)


def _save_git_info(output_dir: str, project_root: str):
    """Save git commit / diff if available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        with open(os.path.join(output_dir, "git_commit.txt"), "w") as f:
            f.write(f"{commit}\n")
    except Exception:
        pass


def _save_dependency_versions(output_dir: str):
    """Save Python and key dependency versions for reproducibility."""
    versions = {
        "python_version": sys.version,
    }
    for pkg in ["numpy", "scipy", "pandas", "matplotlib", "cvxpy", "shapely"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    try:
        import casadi
        versions["casadi"] = casadi.__version__
    except ImportError:
        versions["casadi"] = "not_installed"
    try:
        import ray
        versions["ray"] = ray.__version__
    except ImportError:
        versions["ray"] = "not_installed"

    save_yaml(versions, os.path.join(output_dir, "dependency_versions.yaml"))


def _save_runtime_summary(results: list, output_dir: str):
    """Aggregate runtime stats per method, including backend tracking."""
    rows = []
    for r in results:
        is_dict = isinstance(r, dict)
        rt = r.get("runtime_stats", {}) if is_dict else (r.runtime_stats if hasattr(r, "runtime_stats") else {})
        meta = r.get("metadata", {}) if is_dict else (r.metadata if hasattr(r, "metadata") else {})
        mpc_info = meta.get("mpc_info", {}) if isinstance(meta, dict) else {}
        rows.append({
            "method": r.get("method", "?") if is_dict else (r.method if hasattr(r, "method") else "?"),
            "mean_runtime": rt.get("mean_runtime", np.nan),
            "p95_runtime": rt.get("p95_runtime", np.nan),
            "p99_runtime": rt.get("p99_runtime", np.nan),
            "deadline_miss_rate": rt.get("deadline_miss_rate", np.nan),
            "actual_backend": mpc_info.get("actual_backend", "unknown"),
            "requested_backend": mpc_info.get("requested_backend", "unknown"),
            "degraded_backend": mpc_info.get("degraded_backend", ""),
            "fallback_reason": mpc_info.get("fallback_reason", ""),
        })
    if rows:
        df = pd.DataFrame(rows)
        summary = df.groupby("method").agg(
            mean_runtime=("mean_runtime", "mean"),
            p95_runtime=("p95_runtime", "mean"),
            p99_runtime=("p99_runtime", "mean"),
            deadline_miss_rate=("deadline_miss_rate", "mean"),
            backend_used=("actual_backend", "first"),
            backend_fallback_count=("degraded_backend", lambda x: (x != "").sum()),
        ).reset_index()
        summary.to_csv(os.path.join(output_dir, "runtime_summary.csv"), index=False)


def _save_failure_cases(results: list, output_dir: str):
    """Save episodes with failures to a separate CSV."""
    failures = []
    for r in results:
        ft = r.failure_type if hasattr(r, "failure_type") else r.get("failure_type", "")
        if ft:
            row = {
                "episode_id": r.episode_id if hasattr(r, "episode_id") else r.get("episode_id", ""),
                "scenario_id": r.scenario_id if hasattr(r, "scenario_id") else r.get("scenario_id", ""),
                "method": r.method if hasattr(r, "method") else r.get("method", ""),
                "seed": r.seed if hasattr(r, "seed") else r.get("seed", 0),
                "failure_type": ft,
            }
            flags = r.failure_flags if hasattr(r, "failure_flags") else r.get("failure_flags", {})
            row.update(flags)
            failures.append(row)
    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(output_dir, "failure_cases.csv"), index=False,
        )


def _save_metrics_by_scenario_method(results: list, output_dir: str):
    """Aggregate key metrics by (scenario, method)."""
    rows = []
    for r in results:
        sid = r.scenario_id if hasattr(r, "scenario_id") else r.get("scenario_id", "?")
        m = r.method if hasattr(r, "method") else r.get("method", "?")
        metrics = r.metrics if hasattr(r, "metrics") else r.get("metrics", {})
        row = {"scenario_id": sid, "method": m}
        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float, bool))})
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        agg = df.groupby(["scenario_id", "method"]).mean(numeric_only=True).reset_index()
        agg.to_csv(os.path.join(output_dir, "metrics_by_scenario_method.csv"), index=False)


def _save_readme_run(output_dir, args, methods, scenarios, elapsed, n_results):
    """Save run metadata as markdown."""
    lines = [
        f"# Experiment Run — {datetime.now().isoformat()}",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Backend | {args.config} |",
        f"| Scenarios | {len(scenarios)} — {', '.join(scenarios[:5])}... |",
        f"| Methods | {len(methods)} — {', '.join(methods)} |",
        f"| Episodes per scenario | {args.n_episodes} |",
        f"| Seeds | {args.n_seeds} |",
        f"| Quick mode | {args.quick} |",
        f"| Dry run | {args.dry_run} |",
        f"| Results | {n_results} episodes |",
        f"| Wall time | {elapsed:.0f} s |",
        "",
        "## Output Files",
    ]
    for f in sorted(os.listdir(output_dir)):
        lines.append(f"- `{f}`")
    with open(os.path.join(output_dir, "README_run.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
