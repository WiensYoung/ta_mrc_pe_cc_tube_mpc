"""Run core experiments comparing all methods across core scenarios.

Methods sourced from ``experiments.baseline_registry`` (single source of truth):

    B1: COLREGs + VO/OZT          (reactive heuristic)
    B2: Rule-based DWA             (local search)
    B3: Deterministic MMG-MPC      (multi-rule, no CC/PE/Tube/CBF/fallback)
    B4: CC-MPC                     (+ chance constraint)
    B5: PE-CC-MPC                  (+ physics enhancement)
    B6: PE-CC-Tube-MPC             (+ robust tube + adaptive)
    B7: PE-CC-Tube-MPC + CBF       (+ CBF-QP, no fallback)
    Proposed: TA-MRC-PE-CC-Tube-MPC (B7 + fallback, full system)

Each base scenario is perturbed N times to produce ≥100 episodes per scenario
for statistical validity (per Section 12.1 of experiment design).
"""

import hashlib
import os
from typing import Optional

import numpy as np
import pandas as pd

from ..data.perturbation_generator import build_synthetic_episode_batch
from ..simulation.closed_loop_runner import ClosedLoopRunner
from ..simulation.scenario_loader import list_scenarios
from ..utils.io_utils import load_config_with_overrides, load_yaml


def run_core_experiments(
    config_dir: str = "configs",
    scenario_file: str = "configs/scenarios_core.yaml",
    output_path: str = "results/raw/core_results.csv",
    n_episodes_per_scenario: int = 100,
    n_seeds: int = 3,
    seed_offset: int = 0,
    methods: list[str] = None,
    verbose: bool = True,
    checkpoint_path: str = None,
    n_workers: int = 4,
) -> list:
    """Run all core experiments with perturbation-enhanced episodes.

    Args:
        config_dir: Path to configuration directory.
        scenario_file: Path to core scenarios YAML.
        output_path: Path to save results CSV.
        n_episodes_per_scenario: Number of perturbed episodes per scenario.
        n_seeds: Number of random seeds per (episode, method).
            Total runs = n_episodes_per_scenario x n_methods x n_seeds.
        seed_offset: Starting seed offset.
        methods: Methods to evaluate (default: all baselines + Proposed).
        verbose: Show progress bar.
        n_workers: Number of parallel workers (1=serial).

    Returns:
        List of EpisodeResult objects.
    """
    if methods is None:
        from .baseline_registry import list_baseline_methods
        methods = list_baseline_methods()

    # Load config
    config = load_config_with_overrides(config_dir)

    # Load scenario template and generate perturbed episodes
    scenario_config = load_yaml(scenario_file)
    scenario_ids = list_scenarios(scenario_file)

    # Build perturbation-enhanced episodes per scenario
    all_episodes = []
    for sid in scenario_ids:
        batch = build_synthetic_episode_batch(
            scenario_config,
            sid,
            n_episodes=n_episodes_per_scenario,
            base_seed=seed_offset + _deterministic_hash(sid) % 10000,
        )
        all_episodes.extend(batch)
        if verbose:
            print(f"  {sid}: generated {len(batch)} perturbed episodes")

    if verbose:
        print(f"Total episodes: {len(all_episodes)} across {len(scenario_ids)} scenarios")
        total_runs = len(all_episodes) * len(methods) * n_seeds
        print(f"Total experiment runs: {total_runs}")

    # Run experiments
    runner = ClosedLoopRunner(config)
    results = runner.run_batch(
        episodes=all_episodes,
        methods=methods,
        n_seeds=n_seeds,
        seed_offset=seed_offset,
        verbose=verbose,
        checkpoint_path=checkpoint_path,
        n_workers=n_workers,
        output_csv=output_path,
    )

    # Save results
    if output_path:
        _save_results_csv(results, output_path)

    return results


def _save_results_csv(results: list, output_path: str):
    """Save episode results to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for r in results:
        row = {
            "episode_id": r.episode_id,
            "scenario_id": r.scenario_id,
            "method": r.method,
            "seed": r.seed,
            "failure_type": r.failure_type or "",
            **{f"fail_F{i}": r.failure_flags.get(f"F{i}", False) for i in range(1, 11)},
            **r.metrics,
            **r.runtime_stats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def _deterministic_hash(s: str) -> int:
    """Deterministic integer hash from a string (cross-run reproducible).

    Uses MD5 to avoid Python's `hash()` which is salted by PYTHONHASHSEED
    and varies across interpreter runs. This ensures perturbation seeds
    and cluster-bootstrap cluster assignments are consistent.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def get_scenario_weights(statistics_config_path: str = "configs/statistics.yaml") -> dict:
    """Load scenario weighting for balanced multi-ship coverage.

    Returns a dict mapping scenario_id → weight (float, default 1.0).
    Weights are used to oversample multi-ship scenarios to ensure
    sufficient statistical power.

    GAP #10 — 2026-06-10: Ensures 3-4 ship scenarios are not
    underrepresented relative to pairwise scenarios.
    """
    import os
    try:
        path = statistics_config_path
        if not os.path.isabs(path):
            # Resolve relative to project root (2 levels up from experiments/)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )))
            path = os.path.join(project_root, statistics_config_path)
        if os.path.exists(path):
            stats_cfg = load_yaml(path)
            return stats_cfg.get("scenario_weights", {})
    except Exception:
        pass
    return {}


def compute_weighted_episode_counts(
    scenario_ids: list[str],
    base_n_episodes: int,
    weights: dict = None,
) -> dict:
    """Compute weighted episode counts per scenario.

    Args:
        scenario_ids: List of scenario identifiers.
        base_n_episodes: Default number of episodes per scenario.
        weights: Dict of scenario_id → weight (default 1.0 if missing).

    Returns:
        Dict of scenario_id → n_episodes (int, ≥1).
    """
    if weights is None:
        weights = {}
    counts = {}
    for sid in scenario_ids:
        w = weights.get(sid, 1.0)
        counts[sid] = max(1, int(base_n_episodes * w))
    return counts
