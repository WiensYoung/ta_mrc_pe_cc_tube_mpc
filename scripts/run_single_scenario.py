#!/usr/bin/env python
"""Run a single scenario with a specified method and seed.

Usage:
    python scripts/run_single_scenario.py --config configs/default.yaml --scenario S2 --method Proposed --seed 1
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner
from ta_mrc_pe_cc_tube_mpc.simulation.scenario_loader import load_scenario
from ta_mrc_pe_cc_tube_mpc.utils.io_utils import load_config_with_overrides


def main():
    parser = argparse.ArgumentParser(description="Run a single collision avoidance scenario.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to default config.")
    parser.add_argument("--scenario", default="S2", help="Scenario ID (e.g., S1-S8).")
    parser.add_argument("--method", default="Proposed",
                        help="Method: Proposed, B3, B4, B5, B6, B7.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--scenario-file", default="configs/scenarios_core.yaml",
                        help="Path to scenario YAML file.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    # Paths relative to project root
    project_root = os.path.join(os.path.dirname(__file__), "..")
    config_path = os.path.join(project_root, args.config)
    if os.path.isfile(config_path):
        config = load_config_with_overrides(os.path.dirname(config_path))
    else:
        config = load_config_with_overrides(os.path.join(project_root, "configs"))
    scenario_file = os.path.join(project_root, args.scenario_file) if not os.path.isabs(args.scenario_file) else args.scenario_file
    episode = load_scenario(scenario_file, args.scenario)

    runner = ClosedLoopRunner(config)
    result = runner.run_single(episode, method=args.method, seed=args.seed)

    print(f"\n=== Episode Results ===")
    print(f"Scenario: {result.scenario_id}")
    print(f"Method:   {result.method}")
    print(f"Seed:     {result.seed}")
    print(f"Failure:  {result.failure_type or 'None'}")
    print(f"Runtime:  mean={result.runtime_stats.get('mean_runtime', 0):.4f}s "
          f"p95={result.runtime_stats.get('p95_runtime', 0):.4f}s")
    print(f"\nMetrics:")
    for key, value in sorted(result.metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    if args.output:
        import pandas as pd
        row = {
            "episode_id": result.episode_id,
            "scenario_id": result.scenario_id,
            "method": result.method,
            "seed": result.seed,
            "failure_type": result.failure_type or "",
            **result.metrics,
        }
        df = pd.DataFrame([row])
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
