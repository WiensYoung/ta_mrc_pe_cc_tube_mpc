#!/usr/bin/env python
"""Build simulation episodes from processed AIS data or scenario configs.

Usage:
    python scripts/build_episodes.py --scenarios configs/scenarios_core.yaml --output data/processed/episodes.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.data.episode_builder import build_episode_from_scenario
from ta_mrc_pe_cc_tube_mpc.utils.io_utils import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Build simulation episodes.")
    parser.add_argument("--scenarios", required=True, help="Scenario YAML file.")
    parser.add_argument("--output", required=True, help="Output file path (JSON).")
    args = parser.parse_args()

    # Load YAML once, build all scenarios from in-memory config.
    # Resolve relative paths against project root (consistent with other scripts).
    project_root = os.path.join(os.path.dirname(__file__), "..")
    scenario_path = os.path.join(project_root, args.scenarios) if not os.path.isabs(args.scenarios) else args.scenarios
    import yaml
    with open(scenario_path, "r") as f:
        config = yaml.safe_load(f)

    episodes = []
    for sid in config.get("scenarios", {}):
        ep = build_episode_from_scenario(config, sid)
        episodes.append({
            "scenario_id": ep.scenario_id,
            "waterway": ep.waterway,
            "duration": ep.duration,
            "dt": ep.dt,
            "num_targets": len(ep.targets),
        })

    ensure_dir(os.path.dirname(args.output))
    with open(args.output, "w") as f:
        json.dump(episodes, f, indent=2)
    print(f"Built {len(episodes)} episodes. Saved to {args.output}")


if __name__ == "__main__":
    main()
