#!/usr/bin/env python
"""Extract real AIS encounter episodes and save as EpisodeConfig JSON.

Usage:
    python scripts/extract_ais_episodes.py [--waterway san_francisco_bay] [--max 20]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.data.ais_episode_builder import build_episodes_from_csv


WATERWAY_CONFIGS = {
    "san_francisco_bay": {
        "ais_path": "data/processed/ais_san_francisco_bay.csv",
        "water_depth": 100.0,
        "ref_lon": -122.45,
        "ref_lat": 37.80,
    },
    "juan_de_fuca_puget_sound": {
        "ais_path": "data/processed/ais_juan_de_fuca_puget_sound.csv",
        "water_depth": 280.0,
        "ref_lon": -122.75,
        "ref_lat": 48.15,
    },
    "new_york_harbor": {
        "ais_path": "data/processed/ais_new_york_harbor.csv",
        "water_depth": 60.0,
        "ref_lon": -74.02,
        "ref_lat": 40.67,
    },
}


def episode_config_to_dict(ep: "EpisodeConfig") -> dict:
    """Convert EpisodeConfig to serializable dict."""
    own = ep.ownship_initial_state
    targets = []
    for tgt in ep.targets:
        tgt_dict = {
            "mmsi": tgt.mmsi,
            "state": {
                "x": tgt.state.x, "y": tgt.state.y,
                "psi": tgt.state.psi, "u": tgt.state.u,
                "v": tgt.state.v, "r": tgt.state.r,
            },
            "length": tgt.length,
            "beam": tgt.beam,
            "draught": tgt.draught,
            "vessel_type": tgt.vessel_type,
            "nav_status": tgt.nav_status,
            "cog": tgt.cog,
            "sog": tgt.sog,
            "heading": tgt.heading,
            "covariance": tgt.covariance.tolist() if tgt.covariance is not None else None,
            "ais_quality": tgt.ais_quality,
        }
        targets.append(tgt_dict)

    return {
        "scenario_id": ep.scenario_id,
        "waterway": ep.waterway,
        "duration": ep.duration,
        "dt": ep.dt,
        "ownship_initial_state": {
            "x": own.x, "y": own.y, "psi": own.psi,
            "u": own.u, "v": own.v, "r": own.r,
        },
        "targets": targets,
        "environment": {
            "water_depth": ep.environment_sequence[0].water_depth if ep.environment_sequence else 50.0,
            "waterway_id": ep.waterway,
        },
        "perturbation_config": ep.perturbation_config,
        "random_seed": ep.random_seed,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract AIS encounter episodes")
    parser.add_argument("--waterway", default="san_francisco_bay",
                        choices=list(WATERWAY_CONFIGS.keys()),
                        help="Waterway identifier")
    parser.add_argument("--ownship", default="",
                        help="Ownship MMSI (default: most frequent MMSI in dataset)")
    parser.add_argument("--max", type=int, default=20, dest="max_encounters",
                        help="Maximum encounter episodes to extract")
    parser.add_argument("--cpa", type=float, default=2000.0,
                        help="CPA threshold [m]")
    parser.add_argument("--output", default="",
                        help="Output JSON path (default: data/processed/ais_episodes_<waterway>.json)")
    args = parser.parse_args()

    ww = WATERWAY_CONFIGS[args.waterway]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ais_path = os.path.join(base_dir, ww["ais_path"])

    if not os.path.exists(ais_path):
        print(f"ERROR: AIS data not found at {ais_path}")
        print("Run AIS preprocessing first.")
        sys.exit(1)

    print(f"AIS data: {ais_path}")
    print(f"Waterway: {args.waterway}, water depth: {ww['water_depth']}m")
    print(f"CPA threshold: {args.cpa}m, max encounters: {args.max_encounters}")

    episodes = build_episodes_from_csv(
        csv_path=ais_path,
        ownship_mmsi=args.ownship,
        waterway_id=args.waterway,
        water_depth=ww["water_depth"],
        max_encounters=args.max_encounters,
        cpa_threshold_m=args.cpa,
        min_duration_s=120.0,
    )

    if not episodes:
        print("No encounters found.")
        sys.exit(1)

    print(f"\nFound {len(episodes)} unique encounters:")
    for ep in episodes:
        cfg = ep.perturbation_config or {}
        print(f"  {ep.scenario_id}: CPA={cfg.get('cpa_m', 0):.0f}m, dur={ep.duration:.0f}s")

    output_path = args.output or os.path.join(
        base_dir, "data", "processed", f"ais_episodes_{args.waterway}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    episodes_data = [episode_config_to_dict(ep) for ep in episodes]
    with open(output_path, "w") as f:
        json.dump(episodes_data, f, indent=2)
    print(f"\nSaved {len(episodes)} episodes to {output_path}")


if __name__ == "__main__":
    main()
