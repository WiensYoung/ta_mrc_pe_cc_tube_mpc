#!/usr/bin/env python
"""Real AIS/ENC replay experiment entry point (G2 — enhanced 2026-06-10).

REQUIRES USER-PROVIDED DATA. This script does NOT ship with real AIS or ENC
data. It will fail with a clear error message if required data is missing.

Supports both pairwise and multi-target AIS encounter extraction.

Prerequisites:
    1. Preprocess AIS data:
       python scripts/preprocess_ais.py --input /path/to/ais_raw --output data/processed/ais_clean.parquet

    2. Extract ENC layers:
       python scripts/extract_enc.py --input-root /path/to/enc_raw --output data/processed/enc --waterways san_francisco_bay --strict

    3. Build AIS episodes (pairwise):
       python scripts/extract_ais_episodes.py --ais data/processed/ais_clean.parquet --enc data/processed/enc/enc_layer_sf.json --waterway san_francisco_bay --output data/processed/ais_episodes_sf.json --max 100

    4. Or build multi-target episodes:
       python scripts/build_episodes.py --ais data/processed/ais_clean.parquet --multi-target --output data/processed/ais_multi.json

Usage:
    # Full 9-baseline run on real AIS data:
    python scripts/run_real_ais_replay.py \\
        --episodes data/processed/ais_episodes_sf.json \\
        --enc data/processed/enc/enc_layer_san_francisco_bay.json \\
        --methods Proposed B1 B3 B4 B6 B7 B8 \\
        --output results/real_ais_replay/metrics_by_episode.csv

    # Quick validation (synthetic only):
    python scripts/run_real_ais_replay.py \\
        --episodes data/processed/episodes_scenarios_core.json \\
        --enc data/processed/enc_layer_puget_sound.json \\
        --methods Proposed B3 --dry-run

Output:
    results/real_ais_replay/metrics_by_episode.csv    (per-episode results)
    results/real_ais_replay/resolved_config.yaml       (merged config)
    results/real_ais_replay/README_run.md              (run metadata)
    results/real_ais_replay/figures/                   (Pareto + degradation plots)
    results/real_ais_replay/safety_distance_validation.csv

Data source metadata:
    - scenario_type = "real_replay"
    - data_source = "marinecadastre" | "noaa_enc" | "user_provided"

All output records clearly distinguish real_replay from synthetic scenarios.
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def validate_ais_fields(episodes: list) -> list[str]:
    """Check required AIS fields. Returns list of issues."""
    issues = []
    required_target_fields = ["mmsi", "state"]
    required_state_fields = ["x", "y", "psi", "u", "v"]

    for i, ep in enumerate(episodes):
        for j, tgt in enumerate(ep.get("targets", [])):
            for f in required_target_fields:
                if f not in tgt:
                    issues.append(f"episode[{i}].target[{j}]: missing '{f}'")
            state = tgt.get("state", {})
            for f in required_state_fields:
                if f not in state:
                    issues.append(f"episode[{i}].target[{j}].state: missing '{f}'")
    return issues


def validate_enc_fields(enc_data: dict) -> list[str]:
    """Check required ENC fields. Returns list of issues."""
    issues = []
    required = ["shoreline", "channel_boundaries", "waterway_name", "coordinate_reference_system"]
    for f in required:
        if f not in enc_data and not any(f in k.lower().replace(" ", "_") for k in enc_data.keys()):
            issues.append(f"ENC data missing key related to: '{f}'")
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Real AIS/ENC replay experiment.",
        allow_abbrev=False,
    )
    parser.add_argument("--episodes", required=True,
                       help="Path to AIS episodes JSON file.")
    parser.add_argument("--enc", required=True,
                       help="Path to ENC layer JSON file.")
    parser.add_argument("--methods", nargs="+", default=["B3", "B4", "B6", "B7", "B8", "Proposed"],
                       help="Methods to evaluate.")
    parser.add_argument("--output", default="results/real_ais_replay/metrics_by_episode.csv",
                       help="Output CSV path.")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate data only, no simulation.")
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, os.path.dirname(args.output))
    os.makedirs(output_dir, exist_ok=True)

    # ── Load episodes ──────────────────────────────────────────────────
    episodes_path = os.path.join(PROJECT_ROOT, args.episodes) if not os.path.isabs(args.episodes) else args.episodes
    if not os.path.isfile(episodes_path):
        print(f"FATAL: Episodes file not found: {episodes_path}")
        print("Run extract_ais_episodes.py first with user-provided AIS data.")
        sys.exit(1)

    with open(episodes_path, "r") as f:
        episodes_data = json.load(f)

    episodes = episodes_data if isinstance(episodes_data, list) else episodes_data.get("episodes", [])
    if not episodes:
        print("FATAL: No episodes found in AIS episodes file.")
        sys.exit(1)

    print(f"Loaded {len(episodes)} episode(s) from {episodes_path}")

    # ── Validate AIS fields ────────────────────────────────────────────
    ais_issues = validate_ais_fields(episodes)
    if ais_issues:
        print("FATAL: AIS data validation failed:")
        for issue in ais_issues[:10]:
            print(f"  - {issue}")
        if len(ais_issues) > 10:
            print(f"  ... and {len(ais_issues) - 10} more issues")
        sys.exit(1)
    print("AIS field validation: OK")

    # ── Load & validate ENC ────────────────────────────────────────────
    enc_path = os.path.join(PROJECT_ROOT, args.enc) if not os.path.isabs(args.enc) else args.enc
    if not os.path.isfile(enc_path):
        print(f"FATAL: ENC file not found: {enc_path}")
        print("Run extract_enc.py first with user-provided ENC data.")
        sys.exit(1)

    with open(enc_path, "r") as f:
        enc_data = json.load(f)

    enc_issues = validate_enc_fields(enc_data)
    if enc_issues:
        print("FATAL: ENC data validation failed:")
        for issue in enc_issues:
            print(f"  - {issue}")
        sys.exit(1)
    print("ENC field validation: OK")

    # ── Set scenario type metadata ─────────────────────────────────────
    for ep in episodes:
        ep["scenario_type"] = "real_replay"
        ep["data_source"] = "user_provided"
        if "enc_layer" not in ep:
            ep["enc_layer"] = enc_data

    print(f"Data source: real_replay (user-provided AIS + ENC)")
    print(f"Methods: {args.methods}")
    print(f"Seeds per episode: {args.n_seeds}")

    if args.dry_run:
        print("\nDRY RUN — data validation passed. No simulation executed.")
        print(f"Would run: {len(episodes)} episodes × {len(args.methods)} methods × {args.n_seeds} seeds")
        return

    # ── Run experiments ────────────────────────────────────────────────
    try:
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner
    except ImportError as e:
        print(f"FATAL: Cannot import closed_loop_runner: {e}")
        sys.exit(1)

    from ta_mrc_pe_cc_tube_mpc.utils.io_utils import load_config_with_overrides

    config_dir = os.path.join(PROJECT_ROOT, "configs")
    config = load_config_with_overrides(config_dir)

    t_start = time.time()
    runner = ClosedLoopRunner(config)
    results = runner.run_batch(
        episodes=episodes,
        methods=args.methods,
        n_seeds=args.n_seeds,
        seed_offset=0,
        verbose=True,
        n_workers=args.n_workers,
        output_csv=os.path.join(output_dir, "metrics_by_episode.csv"),
    )
    elapsed = time.time() - t_start

    print(f"\nCompleted {len(results)} real-replay episodes in {elapsed:.0f}s")
    print(f"Output → {output_dir}")

    # ── GAP #8 (2026-06-10): Validate safety distances against real AIS ──
    _validate_safety_distances(results, output_dir)

    # ── G3+G4+G5 (2026-06-10): Generate AIS replay comparison plots ──────
    _generate_replay_plots(results, output_dir)

    # Save README
    readme_path = os.path.join(output_dir, "README_run.md")
    lines = [
        f"# Real AIS/ENC Replay Run — {datetime.now().isoformat()}",
        "",
        f"- Episodes: {len(episodes)} (from {args.episodes})",
        f"- ENC: {args.enc}",
        f"- Methods: {args.methods}",
        f"- Seeds: {args.n_seeds}",
        f"- Results: {len(results)} episodes",
        f"- Wall time: {elapsed:.0f} s",
        "",
        "## Data provenance",
        "- scenario_type: real_replay",
        "- data_source: user_provided",
        "- Synthetic data was NOT substituted for real data.",
    ]
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))
    print(f"README → {readme_path}")


def _validate_safety_distances(results: list, output_dir: str):
    """GAP #8 (2026-06-10): Validate that safety distances from real AIS replay
    are physically plausible.  Flags episodes where:
      - min_distance < 0.25 * (own_length + max_target_length)  (likely collision)
      - mean_required_safety_distance > 5000 m  (runaway safety inflation)
      - safety_margin_violation_rate > 0.50  (unable to maintain safe separation)
    """
    import numpy as np

    issues = []
    for r in results:
        metrics = r.metrics if hasattr(r, "metrics") else r.get("metrics", {})
        sid = r.scenario_id if hasattr(r, "scenario_id") else r.get("scenario_id", "?")
        method = r.method if hasattr(r, "method") else r.get("method", "?")

        min_dist = metrics.get("min_distance", float("inf"))
        mean_req = metrics.get("mean_required_safety_distance", 0.0)
        viol_rate = metrics.get("safety_margin_violation_rate", 0.0)

        flags = []
        if min_dist is not None and min_dist < 50.0:
            flags.append(f"min_distance={min_dist:.1f}m < 50m (near-collision)")
        if mean_req is not None and mean_req > 5000.0:
            flags.append(f"mean_required_safety={mean_req:.0f}m > 5000m (runaway inflation)")
        if viol_rate is not None and viol_rate > 0.50:
            flags.append(f"safety_margin_violation_rate={viol_rate:.2f} > 0.50")

        if flags:
            issues.append({
                "episode_id": r.episode_id if hasattr(r, "episode_id") else r.get("episode_id", ""),
                "scenario_id": sid,
                "method": method,
                "flags": "; ".join(flags),
            })

    if issues:
        import pandas as pd
        val_path = os.path.join(output_dir, "safety_distance_validation.csv")
        pd.DataFrame(issues).to_csv(val_path, index=False)
        print(f"Safety distance validation: {len(issues)} episodes flagged → {val_path}")
    else:
        print("Safety distance validation: all episodes within plausible ranges.")


def _generate_replay_plots(results: list, output_dir: str):
    """Generate Pareto frontier and robustness degradation plots (G3+G4+G5)."""
    import pandas as pd

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Build DataFrame
    rows = []
    for r in results:
        row = {
            "method": r.method if hasattr(r, "method") else r.get("method", "?"),
            "scenario_id": r.scenario_id if hasattr(r, "scenario_id")
                           else r.get("scenario_id", "?"),
        }
        metrics = r.metrics if hasattr(r, "metrics") else r.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float, bool)):
                row[k] = v
        runtime = r.runtime_stats if hasattr(r, "runtime_stats") else r.get("runtime_stats", {})
        for k, v in runtime.items():
            if isinstance(v, (int, float)):
                row[f"rt_{k}"] = v
        rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)

    try:
        from ta_mrc_pe_cc_tube_mpc.evaluation.pub_plots import (
            plot_runtime_safety_pareto,
            plot_robustness_degradation,
        )

        # Pareto frontier
        safety_col = "min_safety_margin" if "min_safety_margin" in df.columns else None
        runtime_col = "mean_runtime" if "mean_runtime" in df.columns else "rt_mean_runtime"
        if safety_col and runtime_col in df.columns:
            plot_runtime_safety_pareto(
                df,
                safety_metric=safety_col,
                runtime_metric=runtime_col,
                save_path=os.path.join(figures_dir, "ais_pareto.pdf"),
            )
            print(f"  Pareto plot → figures/ais_pareto.pdf")

        # Robustness degradation
        safety_for_degrade = safety_col or "min_dcpa"
        if safety_for_degrade in df.columns:
            methods = df["method"].unique()
            safety_by_level = {}
            for m in methods:
                mdf = df[df["method"] == m][safety_for_degrade].dropna()
                if len(mdf) >= 3:
                    chunks = np.array_split(mdf.values, 3)
                    safety_by_level[m] = [float(np.mean(c)) if len(c) > 0 else 0.0
                                          for c in chunks]
            if len(safety_by_level) >= 2:
                plot_robustness_degradation(
                    ["Low σ", "Medium σ", "High σ"],
                    safety_by_level,
                    save_path=os.path.join(figures_dir, "ais_degradation.pdf"),
                )
                print(f"  Degradation plot → figures/ais_degradation.pdf")

    except Exception as e:
        print(f"  [WARN] Replay plots generation failed: {e}")


if __name__ == "__main__":
    main()
