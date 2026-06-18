"""Run extended experiments (stress tests, challenging waterways)."""

import hashlib

from ..data.perturbation_generator import build_synthetic_episode_batch
from ..simulation.closed_loop_runner import ClosedLoopRunner
from ..simulation.scenario_loader import list_scenarios
from ..utils.io_utils import load_config_with_overrides, load_yaml


def run_extended_experiments(
    config_dir: str = "configs",
    scenario_file: str = "configs/scenarios_extended.yaml",
    output_path: str = "results/raw/extended_results.csv",
    n_episodes_per_scenario: int = 30,
    n_seeds: int = 3,
    seed_offset: int = 0,
    methods: list[str] = None,
    verbose: bool = True,
    checkpoint_path: str = None,
    n_workers: int = 4,
) -> list:
    """Run extended experiments on challenging scenarios.

    Uses perturbation augmentation (build_synthetic_episode_batch) to produce
    statistically meaningful results, consistent with core experiments.

    Args:
        config_dir, scenario_file: Config paths.
        output_path: Path to save results.
        n_episodes_per_scenario: Number of perturbed episodes per scenario.
        n_seeds: Number of seeds per (episode, method).
        seed_offset: Starting seed offset.
        methods: Methods to evaluate.
        verbose: Show progress bar.
        n_workers: Number of parallel workers.

    Returns:
        List of EpisodeResult objects.
    """
    if methods is None:
        methods = ["B4", "B6", "B7", "Proposed"]

    config = load_config_with_overrides(config_dir)
    scenario_config = load_yaml(scenario_file)
    scenario_ids = list_scenarios(scenario_file)

    # Build perturbation-enhanced episodes per scenario (consistent with core)
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
        print(f"Total experiment runs: {len(all_episodes) * len(methods) * n_seeds}")

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

    if output_path:
        from .run_core_experiments import _save_results_csv
        _save_results_csv(results, output_path)

    return results


def _deterministic_hash(s: str) -> int:
    """Deterministic integer hash (cross-run reproducible)."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
