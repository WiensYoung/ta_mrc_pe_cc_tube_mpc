#!/usr/bin/env python
"""Run the remaining ablation experiments A7-A12.

This script completes the ablation study by running the six ablations
that were not yet executed (A7-A12), building on the existing checkpoints
for A1-A6 and Proposed_full.

Each ablation removes exactly ONE component via a config flag override:

    A7:  Remove bank effect           (enable_bank_effect=False)
    A8:  Remove ship interaction      (enable_ship_interaction=False)
    A9:  Remove chance constraint     (enable_chance_constraint=False)
    A10: Remove tube-MPC              (enable_tube_mpc=False)
    A11: Remove CBF-QP filter         (enable_cbf_filter=False)
    A12: Remove fallback strategy     (enable_fallback=False)

Usage:
    python scripts/run_remaining_ablations.py

This will run all six ablations sequentially, saving checkpoints
to results/checkpoints/ and results to results/raw/ablation_results.csv
(appended to existing results if the file already exists).

After A7-A12 complete, run:
    python scripts/run_statistics.py
    python scripts/make_report.py
to generate the full statistical analysis and publication-ready tables.
"""

import os
import sys

# Ensure the src directory is on the path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_project_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from ta_mrc_pe_cc_tube_mpc.data.perturbation_generator import build_synthetic_episode_batch
from ta_mrc_pe_cc_tube_mpc.experiments.run_core_experiments import _save_results_csv
from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner
from ta_mrc_pe_cc_tube_mpc.simulation.scenario_loader import list_scenarios
from ta_mrc_pe_cc_tube_mpc.utils.io_utils import deep_merge, load_config_with_overrides, load_yaml


# ---------------------------------------------------------------------------
# Remaining ablations (A7-A12)
# ---------------------------------------------------------------------------

ABLATIONS_REMAINING = [
    ("A7_no_bank_effect",          {"controller": {"enable_bank_effect": False}}),
    ("A8_no_ship_interaction",     {"controller": {"enable_ship_interaction": False}}),
    ("A9_no_chance_constraint",    {"controller": {"enable_chance_constraint": False}}),
    ("A10_no_tube_mpc",            {"controller": {"enable_tube_mpc": False}}),
    ("A11_no_cbf_filter",          {"controller": {"enable_cbf_filter": False}}),
    ("A12_no_fallback",            {"controller": {"enable_fallback": False}}),
]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run remaining ablation experiments A7-A12"
    )
    parser.add_argument(
        "--config-dir", default=os.path.join(_project_root, "configs"),
        help="Config directory"
    )
    parser.add_argument(
        "--scenario-file",
        default=os.path.join(_project_root, "configs", "scenarios_core.yaml"),
        help="Core scenarios file"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_project_root, "results", "raw", "ablation_results_remaining.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.join(_project_root, "results", "checkpoints"),
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=3,
        help="Number of perturbed episodes per scenario"
    )
    parser.add_argument(
        "--n-workers", type=int, default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--ablations", nargs="+",
        choices=["A7", "A8", "A9", "A10", "A11", "A12"],
        default=["A7", "A8", "A9", "A10", "A11", "A12"],
        help="Specific ablations to run (default: all)"
    )
    parser.add_argument(
        "--no-verbose", action="store_true",
        help="Suppress progress bars"
    )
    args = parser.parse_args()

    # Load config
    print("Loading configuration...")
    config = load_config_with_overrides(args.config_dir)
    scenario_config = load_yaml(args.scenario_file)
    scenario_ids = list_scenarios(args.scenario_file)
    print(f"Scenarios: {scenario_ids}")

    # Generate perturbed episodes
    print(f"Generating {args.n_seeds} perturbed episodes per scenario...")
    all_episodes = []
    for sid in scenario_ids:
        batch = build_synthetic_episode_batch(
            scenario_config, sid,
            n_episodes=args.n_seeds,
            base_seed=42,
        )
        all_episodes.extend(batch)
    print(f"Total episodes: {len(all_episodes)}")

    if not all_episodes:
        print(
            "ERROR: No episodes generated. Check scenario file"
            f" '{args.scenario_file}' and n_seeds={args.n_seeds}."
        )
        sys.exit(1)

    # Filter ablations by user selection.
    # ab[0] is e.g. "A7_no_bank_effect" — extract the "A7" prefix.
    # Using split("_")[0] is robust against variable-length prefixes like "A10", "A12".
    selected = [ab for ab in ABLATIONS_REMAINING if ab[0].split("_")[0].upper() in args.ablations]
    print(f"Running {len(selected)} ablation(s): {[s[0] for s in selected]}")

    all_results = {}

    for ablation_name, override in selected:
        print(f"\n{'='*60}")
        print(f"Running ablation: {ablation_name}")
        print(f"Override: {override}")
        print(f"{'='*60}")

        # Deep-merge the ablation override into the base config
        ablated_config = deep_merge(config, override)

        # Create runner with ablated config
        ablation_runner = ClosedLoopRunner(ablated_config)

        # Checkpoint path
        ckpt = os.path.join(args.checkpoint_dir, f"ablations_{ablation_name}_checkpoint.json")

        # Run
        results = ablation_runner.run_batch(
            episodes=all_episodes,
            methods=["Proposed"],  # Method is still "Proposed" but with ablated config
            n_seeds=1,             # Seeds handled via perturbed episodes
            n_workers=args.n_workers,
            verbose=not args.no_verbose,
            checkpoint_path=ckpt,
            output_csv=args.output,
        )

        # Tag results with ablation name for downstream analysis
        for r in results:
            r.method = ablation_name

        all_results[ablation_name] = results
        print(f"  Completed: {len(results)} episodes for {ablation_name}")

    # Save combined results
    if args.output and all_results:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        all_flat = []
        for results_list in all_results.values():
            all_flat.extend(results_list)
        _save_results_csv(all_flat, args.output)
        print(f"\nResults saved to: {args.output}")
        print(f"Total completed episodes: {len(all_flat)}")

    print("\nRemaining ablation experiments (A7-A12) complete.")
    print("Next steps:")
    print("  1. python scripts/run_statistics.py")
    print("  2. python scripts/make_report.py")


if __name__ == "__main__":
    main()
