#!/usr/bin/env python
"""Run all ablation experiments.

Usage:
    python scripts/run_all_ablations.py --config configs/default.yaml --output results/raw/ablation_results.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.experiments.run_ablations import run_ablation_experiments


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments.", allow_abbrev=False)
    parser.add_argument("--config", default="configs/default.yaml", help="Default config file.")
    parser.add_argument("--output", default="results/raw/ablation_results.csv", help="Output CSV path.")
    parser.add_argument("--n-seeds", type=int, default=3, help="Seeds per scenario/ablation.")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of parallel workers (4=default).")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory for resume support.")
    args = parser.parse_args()

    project_root = os.path.join(os.path.dirname(__file__), "..")
    config_dir = os.path.join(project_root, "configs")
    scenario_file = os.path.join(project_root, "configs", "scenarios_core.yaml")
    output_path = os.path.join(project_root, args.output)

    results = run_ablation_experiments(
        config_dir=config_dir,
        scenario_file=scenario_file,
        output_path=output_path,
        n_seeds=args.n_seeds,
        n_workers=args.n_workers,
        checkpoint_dir=args.checkpoint,
    )

    n_total = sum(len(v) for v in results.values())
    print(f"\nCompleted {n_total} ablation episodes. Results saved to {output_path}")


if __name__ == "__main__":
    main()
