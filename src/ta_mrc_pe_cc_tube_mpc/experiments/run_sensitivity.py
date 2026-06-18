"""Run sensitivity experiments.

Tests sensitivity to:
    1. Target speed (3-6, 6-12, 12-18, 18-25 kn)
    2. Target size (0.8x, 1.0x, 1.2x default)
    3. AIS delay/dropout (delay: 1,3,5,10s; dropout: 5%,10%,20%)
    4. Bank distance (0.5L, 1L, 1.5L, 2L, 3L)
    5. Inter-ship lateral distance (1B, 2B, 3B, 5B, 8B)
    6. Cross-current (0, 1, 2, 3 kn)
"""

import copy
import os as _os
from typing import Optional

import numpy as np

from ..simulation.closed_loop_runner import ClosedLoopRunner
from ..simulation.scenario_loader import load_scenario
from ..types import EpisodeConfig
from ..utils.io_utils import load_config_with_overrides


def run_sensitivity_target_speed(
    base_episode: EpisodeConfig,
    runner: ClosedLoopRunner,
    speed_ranges: list = None,
    n_seeds: int = 3,
    n_workers: int = 4,
    checkpoint_dir: str = None,
) -> dict:
    """Sensitivity to target ship speed.

    Returns dict: speed_range -> list of EpisodeResult.
    """
    if speed_ranges is None:
        speed_ranges = [
            ("3-6kn", 1.5, 3.0),
            ("6-12kn", 3.0, 6.0),
            ("12-18kn", 6.0, 9.0),
            ("18-25kn", 9.0, 12.5),
        ]

    results = {}
    for label, speed_min, speed_max in speed_ranges:
        ep = copy.deepcopy(base_episode)
        for tgt in ep.targets:
            speed = (speed_min + speed_max) / 2.0
            tgt.state.u = speed
            tgt.sog = speed / 0.514444  # m/s to kn
        ckpt = _os.path.join(checkpoint_dir, f"sens_speed_{label}_checkpoint.json") if checkpoint_dir else None
        results[label] = runner.run_batch(
            episodes=[ep], methods=["Proposed"],
            n_seeds=n_seeds, verbose=False, n_workers=n_workers,
            checkpoint_path=ckpt,
        )
    return results


def run_sensitivity_target_size(
    base_episode: EpisodeConfig,
    runner: ClosedLoopRunner,
    scale_factors: list = None,
    n_seeds: int = 3,
    n_workers: int = 4,
    checkpoint_dir: str = None,
) -> dict:
    """Sensitivity to target ship size.

    Returns dict: scale_factor -> list of EpisodeResult.
    """
    if scale_factors is None:
        scale_factors = [0.8, 1.0, 1.2]

    results = {}
    for scale in scale_factors:
        ep = copy.deepcopy(base_episode)
        for tgt in ep.targets:
            tgt.length *= scale
            tgt.beam *= scale
        ckpt = _os.path.join(checkpoint_dir, f"sens_size_{scale}x_checkpoint.json") if checkpoint_dir else None
        results[f"{scale}x"] = runner.run_batch(
            episodes=[ep], methods=["Proposed"],
            n_seeds=n_seeds, verbose=False, n_workers=n_workers,
            checkpoint_path=ckpt,
        )
    return results


def run_sensitivity_ais(
    base_episode: EpisodeConfig,
    runner: ClosedLoopRunner,
    delays: list = None,
    dropout_probs: list = None,
    n_seeds: int = 3,
    n_workers: int = 4,
    checkpoint_dir: str = None,
) -> dict:
    """Sensitivity to AIS delay and dropout.

    Returns dict: "delay_X_dropout_Y" -> list of EpisodeResult.
    """
    if delays is None:
        delays = [1.0, 3.0, 5.0, 10.0]
    if dropout_probs is None:
        dropout_probs = [0.0, 0.05, 0.10, 0.20]

    results = {}
    for delay in delays:
        for dp in dropout_probs:
            ep = copy.deepcopy(base_episode)
            for tgt in ep.targets:
                if not isinstance(tgt.ais_quality, dict):
                    tgt.ais_quality = {}
                tgt.ais_quality["delay"] = delay
                tgt.ais_quality["dropout_prob"] = dp
            safe_label = f"delay{delay:.0f}s_drop{int(dp*100):d}pct"
            ckpt = _os.path.join(checkpoint_dir, f"sens_ais_{safe_label}_checkpoint.json") if checkpoint_dir else None
            results[safe_label] = runner.run_batch(
                episodes=[ep], methods=["Proposed"],
                n_seeds=n_seeds, verbose=False, n_workers=n_workers,
                checkpoint_path=ckpt,
            )
    return results


def run_sensitivity_bank_distance(
    base_episode: EpisodeConfig,
    runner: ClosedLoopRunner,
    distances_L: list = None,
    vessel_length: float = 180.0,
    n_seeds: int = 3,
    n_workers: int = 4,
    checkpoint_dir: str = None,
) -> dict:
    """Sensitivity to bank distance.

    Returns dict: distance_label -> list of EpisodeResult.
    """
    if distances_L is None:
        distances_L = [0.5, 1.0, 1.5, 2.0, 3.0]

    results = {}
    for dist_L in distances_L:
        ep = copy.deepcopy(base_episode)
        dist = dist_L * vessel_length
        for i, env in enumerate(ep.environment_sequence):
            env.bank_distance_left = dist
            env.bank_distance_right = dist * 2
            ep.environment_sequence[i] = env
        label = f"{dist_L}L"
        ckpt = _os.path.join(checkpoint_dir, f"sens_bank_{label}_checkpoint.json") if checkpoint_dir else None
        results[label] = runner.run_batch(
            episodes=[ep], methods=["Proposed"],
            n_seeds=n_seeds, verbose=False, n_workers=n_workers,
            checkpoint_path=ckpt,
        )
    return results


def run_sensitivity_cross_current(
    base_episode: EpisodeConfig,
    runner: ClosedLoopRunner,
    currents_kn: list = None,
    n_seeds: int = 3,
    n_workers: int = 4,
    checkpoint_dir: str = None,
) -> dict:
    """Sensitivity to cross-current magnitude.

    Returns dict: current_label -> list of EpisodeResult.
    """
    if currents_kn is None:
        currents_kn = [0, 1, 2, 3]
    KNOTS_TO_MPS = 0.514444

    results = {}
    for c_kn in currents_kn:
        ep = copy.deepcopy(base_episode)
        c_mps = c_kn * KNOTS_TO_MPS
        for i, env in enumerate(ep.environment_sequence):
            env.current_y = c_mps
            ep.environment_sequence[i] = env
        label = f"{c_kn}kn"
        ckpt = _os.path.join(checkpoint_dir, f"sens_current_{label}_checkpoint.json") if checkpoint_dir else None
        results[label] = runner.run_batch(
            episodes=[ep], methods=["Proposed"],
            n_seeds=n_seeds, verbose=False, n_workers=n_workers,
            checkpoint_path=ckpt,
        )
    return results


def run_sensitivity_inter_ship(
    base_episode: EpisodeConfig,
    runner: ClosedLoopRunner,
    distances_B: list = None,
    beam: float = 30.0,
    n_seeds: int = 3,
    n_workers: int = 4,
    checkpoint_dir: str = None,
) -> dict:
    """Sensitivity to inter-ship lateral separation.

    Returns dict: distance_label -> list of EpisodeResult.
    """
    if distances_B is None:
        distances_B = [1, 2, 3, 5, 8]  # multiples of beam

    results = {}
    for dist_B in distances_B:
        ep = copy.deepcopy(base_episode)
        lateral_offset = dist_B * beam
        # Adjust target ship lateral position
        for tgt in ep.targets:
            tgt.state.y += lateral_offset
        label = f"{dist_B}B"
        ckpt = _os.path.join(checkpoint_dir, f"sens_ship_{label}_checkpoint.json") if checkpoint_dir else None
        results[label] = runner.run_batch(
            episodes=[ep], methods=["Proposed"],
            n_seeds=n_seeds, verbose=False, n_workers=n_workers,
            checkpoint_path=ckpt,
        )
    return results


def run_all_sensitivity(
    config_dir: str = "configs",
    scenario_file: str = "configs/scenarios_core.yaml",
    scenario_id: str = "S1",
    output_dir: str = "results/raw",
    n_seeds: int = 3,
    checkpoint_dir: str = None,
    n_workers: int = 4,
) -> dict:
    """Run all sensitivity experiments on a base scenario.

    Returns a nested dict of all sensitivity results.
    """
    config = load_config_with_overrides(config_dir)
    base_episode = load_scenario(scenario_file, scenario_id)
    runner = ClosedLoopRunner(config)

    all_sens = {}

    print("Running target speed sensitivity...")
    all_sens["target_speed"] = run_sensitivity_target_speed(base_episode, runner, n_seeds=n_seeds, n_workers=n_workers, checkpoint_dir=checkpoint_dir)

    print("Running target size sensitivity...")
    all_sens["target_size"] = run_sensitivity_target_size(base_episode, runner, n_seeds=n_seeds, n_workers=n_workers, checkpoint_dir=checkpoint_dir)

    print("Running AIS delay/dropout sensitivity...")
    all_sens["ais"] = run_sensitivity_ais(base_episode, runner, n_seeds=n_seeds, n_workers=n_workers, checkpoint_dir=checkpoint_dir)

    print("Running bank distance sensitivity...")
    all_sens["bank_distance"] = run_sensitivity_bank_distance(base_episode, runner, n_seeds=n_seeds, n_workers=n_workers, checkpoint_dir=checkpoint_dir)

    print("Running cross-current sensitivity...")
    all_sens["cross_current"] = run_sensitivity_cross_current(base_episode, runner, n_seeds=n_seeds, n_workers=n_workers, checkpoint_dir=checkpoint_dir)

    print("Running inter-ship lateral separation sensitivity...")
    all_sens["inter_ship"] = run_sensitivity_inter_ship(base_episode, runner, n_seeds=n_seeds, n_workers=n_workers, checkpoint_dir=checkpoint_dir)

    # Save results to CSV
    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    all_flat = []
    for sens_name, sens_results in all_sens.items():
        for var_label, results_list in sens_results.items():
            for r in results_list:
                row = {
                    "sensitivity_dimension": sens_name,
                    "variable_level": var_label,
                    "episode_id": r.episode_id,
                    "method": r.method,
                    "seed": r.seed,
                    "failure_type": r.failure_type or "",
                }
                row.update(r.metrics)
                row.update(r.runtime_stats)
                all_flat.append(row)

    df = pd.DataFrame(all_flat)
    output_path = os.path.join(output_dir, "sensitivity_results.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved sensitivity results to {output_path}")

    return all_sens
