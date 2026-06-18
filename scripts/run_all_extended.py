#!/usr/bin/env python
"""Run all extended experiments (stress tests, cross-waterway generalization).

Usage:
    python scripts/run_all_extended.py --config configs/default.yaml --output results/raw/extended_results.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.experiments.run_extended_experiments import run_extended_experiments


def main():
    parser = argparse.ArgumentParser(description="Run extended collision avoidance experiments.", allow_abbrev=False)
    parser.add_argument("--config", default="configs/default.yaml", help="Default config file.")
    parser.add_argument("--output", default="results/raw/extended_results.csv", help="Output CSV path.")
    parser.add_argument("--n-seeds", type=int, default=5, help="Seeds per scenario/method.")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Methods to run (default: B4 B6 B7 Proposed).")
    parser.add_argument("--n-episodes", type=int, default=30,
                        help="Perturbed episodes per scenario (default: 30).")
    parser.add_argument("--n-workers", type=int, default=4, help="Parallel workers (4=default, 0=cpu_count-1).")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file path for resume support.")
    args = parser.parse_args()

    if args.n_workers <= 0:
        import multiprocessing
        args.n_workers = max(1, multiprocessing.cpu_count() - 1)

    project_root = os.path.join(os.path.dirname(__file__), "..")
    config_dir = os.path.join(project_root, "configs")
    scenario_file = os.path.join(project_root, "configs", "scenarios_extended.yaml")
    output_path = os.path.join(project_root, args.output)

    results = run_extended_experiments(
        config_dir=config_dir,
        scenario_file=scenario_file,
        output_path=output_path,
        n_episodes_per_scenario=args.n_episodes,
        n_seeds=args.n_seeds,
        methods=args.methods,
        checkpoint_path=args.checkpoint,
        n_workers=args.n_workers,
    )

    print(f"\nCompleted {len(results)} extended episodes. Results saved to {output_path}")


if __name__ == "__main__":
    main()
