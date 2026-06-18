"""Sobol-sequence-based global sensitivity analysis for Tube-MPC parameters.

Replaces the grid-search approach in run_sensitivity.py with proper
variance-based sensitivity analysis using Saltelli's method:
  - Sobol' low-discrepancy sequence for efficient coverage
  - First-order (S1) and total-effect (ST) Sobol' indices
  - GP-based Bayesian Optimization for adaptive refinement
  - Fewer experiments needed: N*(d+2) vs d * grid_points^d

References:
    Saltelli et al. (2024), "Global Sensitivity Analysis: The Primer", 2nd ed.
    Owen (2020), "On dropping the first Sobol' point", arXiv:2008.08051.

Usage:
    from ta_mrc_pe_cc_tube_mpc.experiments.run_sensitivity_sobol import (
        run_sobol_sensitivity,
    )
    results = run_sobol_sensitivity(n_base=64, n_workers=8)
"""

import copy
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..analysis.sobol_sensitivity import (
    GPSurrogateModel,
    bayesian_optimization_step,
    compute_sobol_indices,
    saltelli_sampling,
    sobol_sequence,
)
from ..simulation.closed_loop_runner import ClosedLoopRunner
from ..simulation.scenario_loader import load_scenario
from ..types import EpisodeConfig
from ..utils.io_utils import load_config_with_overrides

# ── Sensitivity parameter definitions ────────────────────────────────────
# Each parameter: (name, lower_bound, upper_bound, unit, description)
SENSITIVITY_PARAMS = [
    ("tube_rho_0",            2.0,   15.0,  "m",    "Base tube radius"),
    ("tube_rho_maneuver",     0.5,   6.0,   "m",    "Maneuver uncertainty"),
    ("epsilon_default",       0.01,  0.30,  "-",    "Chance constraint epsilon"),
    ("cbf_alpha",             0.2,   3.0,   "-",    "CBF convergence rate"),
    ("cbf_tolerance",         0.5,   8.0,   "m",    "CBF activation margin"),
    ("d_safe_base",           100.0, 800.0, "m",    "Base safe distance"),
    ("mpc_horizon",           8,     30,    "steps","MPC prediction horizon"),
    ("rho_0_adaptive",        0.5,   2.0,   "-",    "Adaptive tube encounter scale"),
    ("speed_norm_factor",     0.5,   3.0,   "-",    "Speed-dependent tube factor"),
    ("fallback_emergency_fac", 0.3,  0.8,   "-",    "Emergency distance factor"),
]

# Metrics to track for sensitivity
SENSITIVITY_METRICS = [
    "min_dcpa",
    "collision_rate",
    "colregs_violation",
    "ship_domain_violation_rate",
    "mean_runtime",
    "fallback_activation_rate",
]


def _apply_params_to_config(
    base_config: dict,
    param_names: list[str],
    param_values: np.ndarray,
) -> dict:
    """Apply a parameter vector to the config dict.

    Maps the sensitivity parameter names to their corresponding
    nested config keys.
    """
    config = copy.deepcopy(base_config)

    # BUG #12 fix (2026-06-10): "d_safe_base" previously mapped to
    # ("domain", "d_safe_base") which does not exist in configs/vessel.yaml.
    # Changed to ("reference", "d_ref") which is the actual reference distance.
    param_map = {
        "tube_rho_0":           ("tube", "rho_0"),
        "tube_rho_maneuver":    ("tube", "rho_maneuver"),
        "epsilon_default":      ("uncertainty", "epsilon_default"),
        "cbf_alpha":            ("cbf", "alpha_cbf"),
        "cbf_tolerance":        ("cbf", "active_constraint_tolerance"),
        "d_safe_base":          ("reference", "d_ref"),
        "mpc_horizon":          ("mpc", "horizon"),
        "rho_0_adaptive":       ("tube_adaptive", "alpha_encounter_max"),
        "speed_norm_factor":    ("tube_adaptive", "alpha_speed_max"),
        "fallback_emergency_fac": ("fallback", "emergency_distance_factor"),
    }

    for i, name in enumerate(param_names):
        val = float(param_values[i])
        section, key = param_map.get(name, (None, None))
        if section and key:
            if section not in config:
                config[section] = {}
            if key == "horizon":
                config[section][key] = int(val)
            else:
                config[section][key] = val

    return config


def _run_single_config(
    config: dict,
    episode: EpisodeConfig,
    n_seeds: int = 3,
    n_workers: int = 4,
) -> dict:
    """Run a single config and return aggregated metrics."""
    runner = ClosedLoopRunner(config)
    results = runner.run_batch(
        episodes=[episode],
        methods=["Proposed"],
        n_seeds=n_seeds,
        n_workers=min(n_workers, n_seeds),
        verbose=False,
    )
    if not results:
        return {}

    # Aggregate metrics across seeds
    agg = {}
    for metric in SENSITIVITY_METRICS:
        values = []
        for r in results:
            # Guard against None metrics/metadata on incomplete results (BUG #9 fix)
            mets = getattr(r, "metrics", None) or {}
            meta = getattr(r, "metadata", None) or {}
            if metric in mets:
                values.append(mets[metric])
            elif metric == "fallback_activation_rate":
                values.append(float(meta.get("fallback_active", False)))
        if values:
            agg[metric] = float(np.mean(values))
        else:
            agg[metric] = 0.0

    # Safety score (higher = better, for minimization in BO)
    agg["safety_cost"] = (
        agg.get("collision_rate", 0.0) * 100.0 +
        agg.get("colregs_violation", 0.0) * 10.0 +
        agg.get("ship_domain_violation_rate", 0.0) * 5.0 +
        agg.get("fallback_activation_rate", 0.0) * 2.0
    ) - agg.get("min_dcpa", 0.0) * 0.01

    return agg


def run_sobol_sensitivity(
    config_dir: str = "configs",
    scenario_file: str = "configs/scenarios_core.yaml",
    scenario_id: str = "S1",
    output_dir: str = "results/raw",
    n_base: int = 64,
    n_seeds: int = 3,
    n_workers: int = 8,
    params: list = None,
    bayesian_refinement: bool = True,
    n_bo_steps: int = 10,
    verbose: bool = True,
) -> dict:
    """Run global sensitivity analysis using Sobol' sequences.

    Args:
        config_dir: Config directory.
        scenario_file: Scenario YAML file.
        scenario_id: Base scenario to analyze.
        output_dir: Output directory for results.
        n_base: Base sample size N (total runs ≈ N*(d+2)).
        n_seeds: Seeds per parameter configuration.
        n_workers: Parallel workers for simulation runs.
        params: Override parameter definitions [(name, lo, hi, unit, desc), ...].
        bayesian_refinement: Enable GP-BO adaptive refinement.
        n_bo_steps: Number of Bayesian Optimization refinement steps.
        verbose: Print progress.

    Returns:
        dict with Sobol' indices, experiment data, and BO results.
    """
    if params is None:
        params = SENSITIVITY_PARAMS

    param_names = [p[0] for p in params]
    param_bounds = [(p[1], p[2]) for p in params]
    n_dims = len(params)

    # ── Load config and scenario ───────────────────────────────────────
    config = load_config_with_overrides(config_dir)
    episode = load_scenario(scenario_file, scenario_id)

    if verbose:
        print(f"Sobol' sensitivity analysis: {n_dims} parameters")
        print(f"  Base samples N={n_base}, total runs ≈ {n_base * (n_dims + 2)}")
        print(f"  Scenario: {scenario_id}, seeds per config: {n_seeds}")

    # ── Generate Saltelli sampling matrices ────────────────────────────
    sampling = saltelli_sampling(n_base, n_dims, param_bounds)
    if verbose:
        print(f"  Generated {sampling['N'] * (n_dims + 2)} parameter vectors")

    # ── Evaluate model at all sampling points ──────────────────────────
    all_X = np.vstack([sampling["A"], sampling["B"]] + sampling["AB"])
    n_total = len(all_X)

    if verbose:
        print(f"  Evaluating {n_total} configurations...")

    all_metrics = {}
    metric_names = SENSITIVITY_METRICS + ["safety_cost"]

    for i in range(n_total):
        cfg = _apply_params_to_config(config, param_names, all_X[i])
        metrics = _run_single_config(cfg, episode, n_seeds, n_workers)
        all_metrics[i] = metrics
        if verbose and (i + 1) % max(1, n_total // 10) == 0:
            print(f"    {i + 1}/{n_total} done")

    # ── Compute Sobol' indices for each metric ─────────────────────────
    sobol_results = {}
    for metric in metric_names:
        y = np.array([all_metrics[i].get(metric, 0.0) for i in range(n_total)])

        # Extract f_A, f_B, f_AB
        f_A = y[:n_base]
        f_B = y[n_base:2*n_base]
        f_AB = []
        for d in range(n_dims):
            start = 2 * n_base + d * n_base
            f_AB.append(y[start:start + n_base])

        indices = compute_sobol_indices(f_A, f_B, f_AB)
        sobol_results[metric] = {
            "S1": indices["S1"].tolist(),
            "ST": indices["ST"].tolist(),
            "var_Y": float(indices["var_Y"]),
            "param_names": param_names,
        }
        if verbose:
            top_params = np.argsort(indices["ST"])[::-1][:3]
            top_str = ", ".join(
                f"{param_names[j]}(ST={indices['ST'][j]:.3f})"
                for j in top_params
            )
            print(f"  {metric:30s}: top params: {top_str}")

    # ── Bayesian Optimization refinement (optional) ────────────────────
    bo_results = None
    if bayesian_refinement:
        if verbose:
            print(f"\n  Bayesian Optimization refinement: {n_bo_steps} steps")

        surrogate = GPSurrogateModel(
            length_scale=0.5,
            signal_variance=1.0,
            noise_variance=1e-4,
        )

        # Use safety_cost as the objective (minimize)
        y_safety = np.array([all_metrics[i].get("safety_cost", 1000.0)
                             for i in range(n_total)])

        # Initial training: best 20% of Sobol' samples
        n_init = max(20, n_total // 5)
        best_indices = np.argsort(y_safety)[:n_init]
        X_init = all_X[best_indices]
        y_init = y_safety[best_indices]

        surrogate.fit(X_init, y_init / np.max(np.abs(y_init) + 1e-6))
        y_best = float(np.min(y_init))

        bo_history = []
        for step in range(n_bo_steps):
            x_next = bayesian_optimization_step(
                surrogate, param_bounds, y_best / np.max(np.abs(y_init) + 1e-6),
            )
            cfg_next = _apply_params_to_config(config, param_names, x_next)
            metrics_next = _run_single_config(cfg_next, episode, n_seeds, n_workers)
            y_next = metrics_next.get("safety_cost", 1000.0)

            bo_history.append({
                "step": step,
                "params": x_next.tolist(),
                "safety_cost": y_next,
                "min_dcpa": metrics_next.get("min_dcpa", 0.0),
            })

            # Update surrogate
            X_aug = np.vstack([X_init, x_next.reshape(1, -1)])
            y_aug = np.append(y_init, y_next)
            surrogate.fit(X_aug, y_aug / np.max(np.abs(y_aug) + 1e-6))
            X_init = X_aug
            y_init = y_aug
            y_best = min(y_best, y_next)

            if verbose:
                print(f"    BO step {step+1}/{n_bo_steps}: "
                      f"safety_cost={y_next:.3f}, best={y_best:.3f}")

        bo_results = {
            "history": bo_history,
            "best_params": bo_history[np.argmin([h["safety_cost"] for h in bo_history])]
            if bo_history else None,
            "best_safety_cost": y_best,
        }

    # ── Save results ──────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # Save Sobol' indices
    import json
    sobol_path = os.path.join(output_dir, "sobol_sensitivity_indices.json")
    with open(sobol_path, "w") as f:
        json.dump(sobol_results, f, indent=2)
    if verbose:
        print(f"\n  Sobol' indices saved to: {sobol_path}")

    # Save experiment data
    rows = []
    for i in range(n_total):
        row = {"config_id": i}
        for d, name in enumerate(param_names):
            row[name] = float(all_X[i, d])
        row.update(all_metrics[i])
        rows.append(row)
    df = pd.DataFrame(rows)
    data_path = os.path.join(output_dir, "sobol_experiment_data.csv")
    df.to_csv(data_path, index=False)
    if verbose:
        print(f"  Experiment data saved to: {data_path}")

    # Save BO results
    if bo_results:
        bo_path = os.path.join(output_dir, "bayesian_optimization_results.json")
        with open(bo_path, "w") as f:
            json.dump(bo_results, f, indent=2)
        if verbose:
            print(f"  BO results saved to: {bo_path}")

    return {
        "sobol_indices": sobol_results,
        "experiment_data": df,
        "bo_results": bo_results,
        "param_names": param_names,
    }
