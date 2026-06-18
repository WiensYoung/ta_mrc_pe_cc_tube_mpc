#!/usr/bin/env python
"""Run all sensitivity experiments across 6 dimensions.

Usage:
    python scripts/run_all_sensitivity.py --scenario S1 --output results/raw/sensitivity_results.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.experiments.run_sensitivity import run_all_sensitivity


def main():
    parser = argparse.ArgumentParser(description="Run sensitivity experiments.", allow_abbrev=False)
    parser.add_argument("--config", default="configs/default.yaml", help="Default config file.")
    parser.add_argument("--scenario", default="S1", help="Base scenario ID (default: S1).")
    parser.add_argument("--scenario-file", default="configs/scenarios_core.yaml",
                        help="Scenario YAML file.")
    parser.add_argument("--output-dir", default="results/raw", help="Output directory.")
    parser.add_argument("--n-seeds", type=int, default=3, help="Seeds per condition.")
    parser.add_argument("--n-workers", type=int, default=4, help="Parallel workers (4=default, 0=cpu_count-1).")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory for resume support.")
    args = parser.parse_args()

    if args.n_workers <= 0:
        import multiprocessing
        args.n_workers = max(1, multiprocessing.cpu_count() - 1)

    project_root = os.path.join(os.path.dirname(__file__), "..")
    config_dir = os.path.join(project_root, "configs")
    scenario_file = os.path.join(project_root, args.scenario_file)
    output_dir = os.path.join(project_root, args.output_dir)

    results = run_all_sensitivity(
        config_dir=config_dir,
        scenario_file=scenario_file,
        scenario_id=args.scenario,
        output_dir=output_dir,
        n_seeds=args.n_seeds,
        checkpoint_dir=args.checkpoint,
        n_workers=args.n_workers,
    )

    n_total = sum(
        len(v_list) for dim_results in results.values()
        for v_list in dim_results.values()
    )
    print(f"\nCompleted {n_total} sensitivity episodes across {len(results)} dimensions.")
    print(f"Results saved to {output_dir}/sensitivity_results.csv")


if __name__ == "__main__":
    main()
