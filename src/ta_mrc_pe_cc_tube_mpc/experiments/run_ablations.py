"""Run ablation experiments.

Ablation definitions are sourced from ``experiments.baseline_registry``
(single source of truth).  Each ablation removes ONE primary feature from
the Proposed method.
"""

from ..data.perturbation_generator import build_synthetic_episode_batch
from ..experiments.run_core_experiments import _save_results_csv
from ..simulation.closed_loop_runner import ClosedLoopRunner
from ..simulation.scenario_loader import list_scenarios
from ..utils.io_utils import deep_merge, load_config_with_overrides, load_yaml

from .baseline_registry import (
    ABLATION_REGISTRY,
    build_ablation_features,
    list_ablation_ids,
)

# Build the ablation table from the registry (lazy — evaluated at import)
def _build_ablations_from_registry() -> list:
    """Convert registry definitions to (name, override_dict) pairs."""
    result = []
    for aid in list_ablation_ids():
        entry = ABLATION_REGISTRY[aid]
        features = build_ablation_features(aid)
        # Use label as human-readable suffix
        label = entry["label"].replace(" ", "_").replace("/", "_")
        name = f"{aid}_{label}"
        result.append((name, {"controller": features}))
    return result

ABLATIONS = _build_ablations_from_registry()


def run_ablation_experiments(
    config_dir: str = "configs",
    scenario_file: str = "configs/scenarios_core.yaml",
    output_path: str = "results/raw/ablation_results.csv",
    n_seeds: int = 30,
    n_workers: int = 4,
    ablations: list = None,
    verbose: bool = True,
    checkpoint_dir: str = None,
) -> dict:
    """Run all ablation experiments.

    Each ablation removes one component. The full Proposed method
    is also run for comparison.

    Args:
        config_dir: Config directory path.
        scenario_file: Core scenarios file.
        output_path: Path to save results.
        n_seeds: Seeds per (scenario, ablation).
        ablations: List of (name, override_dict) tuples. Uses ABLATIONS if None.
        verbose: Show progress bar.

    Returns:
        dict mapping ablation name to list of EpisodeResult.
    """
    import os as _os

    if ablations is None:
        ablations = ABLATIONS

    config = load_config_with_overrides(config_dir)
    scenario_config = load_yaml(scenario_file)
    scenario_ids = list_scenarios(scenario_file)

    # Generate perturbed episodes for each scenario
    all_episodes = []
    for sid in scenario_ids:
        batch = build_synthetic_episode_batch(
            scenario_config, sid,
            n_episodes=n_seeds,
            base_seed=42,
        )
        all_episodes.extend(batch)

    if not all_episodes:
        raise ValueError(
            f"No episodes generated. Check scenario file '{scenario_file}' "
            f"and scenario IDs: {scenario_ids}. n_seeds={n_seeds} may be 0."
        )

    # First, run full Proposed
    runner = ClosedLoopRunner(config)
    all_results = {}

    print(f"Running full Proposed method ({len(all_episodes)} episodes)...")
    ckpt = _os.path.join(checkpoint_dir, "ablations_Proposed_full_checkpoint.json") if checkpoint_dir else None
    full_results = runner.run_batch(
        episodes=all_episodes,
        methods=["Proposed"],
        n_seeds=1,
        n_workers=n_workers,
        verbose=verbose,
        checkpoint_path=ckpt,
        output_csv=None,  # don't save incrementally — one write at end
    )
    all_results["Proposed_full"] = full_results

    # Run each ablation
    for ablation_name, override in ablations:
        print(f"Running ablation: {ablation_name}")
        ablated_config = deep_merge(config, override)

        # NOTE: The ablation config is already deep-merged into ablated_config.
        # The ClosedLoopRunner is constructed with ablated_config, so
        # _build_controller will use the correct feature flags without
        # needing controller_configs. We pass an empty dict to avoid
        # redundant top-level keys that the controller ignores anyway.
        ablation_runner = ClosedLoopRunner(ablated_config)
        ckpt = _os.path.join(checkpoint_dir, f"ablations_{ablation_name}_checkpoint.json") if checkpoint_dir else None
        results = ablation_runner.run_batch(
            episodes=all_episodes,
            methods=["Proposed"],
            n_seeds=1,
            n_workers=n_workers,
            controller_configs={},
            verbose=verbose,
            checkpoint_path=ckpt,
            output_csv=None,  # don't save incrementally — one write at end
        )
        # Rename method for tracking
        for r in results:
            r.method = ablation_name
        all_results[ablation_name] = results

    # Combine and save — SINGLE write at the end (no duplicates)
    if output_path:
        _os.makedirs(_os.path.dirname(output_path), exist_ok=True)

        all_flat = []
        for results_list in all_results.values():
            all_flat.extend(results_list)

        _save_results_csv(all_flat, output_path)

    return all_results
