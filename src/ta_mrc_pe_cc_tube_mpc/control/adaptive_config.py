"""Adaptive controller configuration and sensitivity feedback.

1. Scenario-adaptive parameter switching:
   Automatically adjusts controller parameters based on:
   - Water depth (shallow water → more conservative)
   - Traffic density and vessel types
   - Channel width constraints
   - Environmental conditions (current, wind)

2. Sensitivity-to-calibration feedback loop:
   Feeds sensitivity analysis results back into parameter calibration
   to tighten or relax bounds based on observed impact.
"""

from typing import Optional

import numpy as np

from ..types import EnvironmentState, TargetShip


def compute_adaptive_parameters(
    env: EnvironmentState,
    targets: list[TargetShip],
    own_draught: float,
    own_length: float,
    base_config: dict = None,
    sensitivity_feedback: dict = None,
) -> dict:
    """Compute scenario-adaptive controller parameters.

    Args:
        env: Current environment state.
        targets: List of target ships.
        own_draught: Ownship draught [m].
        own_length: Ownship length [m].
        base_config: Base controller configuration.
        sensitivity_feedback: Optional sensitivity analysis results for calibration.

    Returns:
        dict with adjusted parameters:
            - epsilon: Chance constraint violation probability
            - rho_0: Base tube radius [m]
            - alpha_cbf: CBF convergence rate
            - horizon: MPC horizon [steps]
            - bank_safety_factor: Multiplier for bank safety margin
            - safety_domain_scale: Overall safety margin scale
            - max_speed_factor: Speed limit as fraction of max
    """
    cfg = base_config or {}
    sensitivity = sensitivity_feedback or {}

    # Base parameters — read from nested config sections matching default.yaml
    epsilon = cfg.get("uncertainty", {}).get("epsilon_default", 0.10)
    rho_0 = cfg.get("tube", {}).get("rho_0", 5.0)
    alpha_cbf = cfg.get("cbf", {}).get("alpha_cbf", 1.0)
    horizon = cfg.get("simulation", {}).get("mpc_horizon_steps", 20)
    bank_safety_factor = 1.0
    safety_domain_scale = 1.0
    max_speed_factor = 1.0

    # --- Water depth adaptation ---
    depth_ratio = env.water_depth / max(own_draught, 1e-3)

    if depth_ratio < 2.0:  # strong shallow
        epsilon *= 0.5         # stricter chance constraint
        rho_0 *= 1.5           # larger tube radius
        bank_safety_factor *= 1.5
        safety_domain_scale *= 1.3
        max_speed_factor *= 0.7
    elif depth_ratio < 3.0:  # moderate shallow
        epsilon *= 0.7
        rho_0 *= 1.2
        bank_safety_factor *= 1.2
        max_speed_factor *= 0.85
    elif depth_ratio < 4.0:  # mild shallow
        epsilon *= 0.85
        rho_0 *= 1.1

    # --- Channel width adaptation ---
    if env.channel_width is not None:
        narrow_ratio = env.channel_width / (10 * own_length)

        if narrow_ratio < 1.0:  # extremely narrow
            epsilon *= 0.5
            rho_0 *= 1.3
            bank_safety_factor *= 2.0
            safety_domain_scale *= 1.5
            horizon = min(horizon + 10, 40)  # longer look-ahead
            max_speed_factor *= 0.5
        elif narrow_ratio < 2.0:  # narrow channel
            epsilon *= 0.7
            rho_0 *= 1.15
            bank_safety_factor *= 1.3
            safety_domain_scale *= 1.2
            horizon = min(horizon + 5, 35)
            max_speed_factor *= 0.7
        elif narrow_ratio < 5.0:
            epsilon *= 0.9
            bank_safety_factor *= 1.1

    # --- Bank distance adaptation — check each side independently (BUG #2 pattern: use OR not AND) ---
    min_bank = float("inf")
    if env.bank_distance_left is not None:
        min_bank = env.bank_distance_left
    if env.bank_distance_right is not None:
        min_bank = min(min_bank, env.bank_distance_right)
    if min_bank != float("inf"):
        bank_ratio = min_bank / own_length

        if bank_ratio < 0.5:
            epsilon *= 0.5
            alpha_cbf *= 1.5
            bank_safety_factor *= 2.0
        elif bank_ratio < 1.0:
            epsilon *= 0.7
            bank_safety_factor *= 1.3
        elif bank_ratio < 2.0:
            bank_safety_factor *= 1.1

    # --- Traffic density adaptation ---
    n_targets = len(targets)
    if n_targets >= 3:
        epsilon *= 0.6
        rho_0 *= 1.2
        safety_domain_scale *= 1.15
        horizon = min(horizon + 5, 40)
    elif n_targets == 2:
        epsilon *= 0.8
        safety_domain_scale *= 1.05

    # --- Vessel type risk adaptation ---
    for tgt in targets:
        if tgt.vessel_type in ("tanker",):
            epsilon *= 0.7
            safety_domain_scale *= 1.3
        elif tgt.vessel_type in ("ferry",):
            epsilon *= 0.6  # ferries are fast, unpredictable
            safety_domain_scale *= 1.2
        elif tgt.vessel_type in ("tug_barge",):
            safety_domain_scale *= 1.15

        # Restricted maneuverability
        if tgt.nav_status in ("restricted_maneuverability", "not_under_command"):
            epsilon *= 0.5
            safety_domain_scale *= 1.5
            max_speed_factor *= 0.6

        # AIS quality degradation
        delay = tgt.ais_quality.get("delay", 0.0)
        dropout = tgt.ais_quality.get("dropout_prob", 0.0)
        if delay > 5.0 or dropout > 0.1:
            rho_0 *= 1.2
            epsilon *= 0.8

    # --- Environmental conditions ---
    cx = env.current_x if env.current_x is not None else 0.0
    cy = env.current_y if env.current_y is not None else 0.0
    wx = env.wind_x if env.wind_x is not None else 0.0
    wy = env.wind_y if env.wind_y is not None else 0.0
    current_mag = np.sqrt(cx**2 + cy**2)
    wind_mag = np.sqrt(wx**2 + wy**2)
    env_severity = (current_mag / 3.0 + wind_mag / 15.0)  # normalized

    if env_severity > 0.5:
        epsilon *= 0.7
        rho_0 *= 1.2
    elif env_severity > 0.2:
        epsilon *= 0.85
        rho_0 *= 1.1

    # --- Sensitivity feedback calibration ---
    if sensitivity:
        # If sensitivity analysis found a parameter has high impact on safety,
        # adjust it conservatively
        for param_name, impact in sensitivity.items():
            if abs(impact) > 0.3:  # high-impact parameter
                # Tighten the most impactful parameter
                if "bank" in param_name.lower():
                    bank_safety_factor *= (1.0 + abs(impact) * 0.5)
                elif "epsilon" in param_name.lower():
                    epsilon *= (1.0 - abs(impact) * 0.3)
                elif "tube" in param_name.lower() or "rho" in param_name.lower():
                    rho_0 *= (1.0 + abs(impact) * 0.3)
                elif "horizon" in param_name.lower():
                    horizon = min(int(horizon * (1.0 + abs(impact) * 0.5)), 50)

    # --- Clamp all parameters to reasonable ranges ---
    epsilon = np.clip(epsilon, 0.005, 0.20)
    rho_0 = np.clip(rho_0, 2.0, 25.0)  # upper bound prevents excessive conservatism
    alpha_cbf = np.clip(alpha_cbf, 0.5, 3.0)
    horizon = int(np.clip(horizon, 10, 50))
    bank_safety_factor = np.clip(bank_safety_factor, 0.8, 3.0)
    safety_domain_scale = np.clip(safety_domain_scale, 0.8, 3.0)
    max_speed_factor = np.clip(max_speed_factor, 0.3, 1.0)

    return {
        "epsilon": float(epsilon),
        "rho_0": float(rho_0),
        "alpha_cbf": float(alpha_cbf),
        "horizon": horizon,
        "bank_safety_factor": float(bank_safety_factor),
        "safety_domain_scale": float(safety_domain_scale),
        "max_speed_factor": float(max_speed_factor),
        "diagnostics": {
            "depth_ratio": float(depth_ratio),
            "n_targets": n_targets,
            "env_severity": float(env_severity),
            "active_adaptations": _list_active_adaptations(
                depth_ratio, n_targets, env, targets, env_severity,
                own_length=own_length,
            ),
        },
    }


def _list_active_adaptations(
    depth_ratio, n_targets, env, targets, env_severity,
    own_length: float = 180.0,
) -> list[str]:
    """List which adaptations are active for diagnostics."""
    active = []
    if depth_ratio < 4.0:
        active.append("shallow_water")
    if env.channel_width is not None and env.channel_width < 10 * own_length:
        active.append("narrow_channel")
    if n_targets >= 2:
        active.append("high_traffic")
    for tgt in targets:
        if tgt.vessel_type in ("tanker", "ferry"):
            active.append("high_risk_vessel")
            break
    if env_severity > 0.2:
        active.append("adverse_environment")
    return active


def apply_adaptive_config(
    controller,
    env: EnvironmentState,
    targets: list[TargetShip],
    own_draught: float,
    own_length: float,
    base_config: dict = None,
    sensitivity_feedback: dict = None,
) -> dict:
    """Apply adaptive parameters to a controller instance.

    Modifies the controller's internal parameters in-place based on
    current scenario conditions, and returns the applied parameters.

    Args:
        controller: TAMRCPECCTubeMPCController instance.
        env: Current environment state.
        targets: Target ships.
        own_draught: Ownship draught.
        own_length: Ownship length.
        base_config: Base configuration.
        sensitivity_feedback: Optional calibration feedback.

    Returns:
        dict of applied adaptive parameters.
    """
    params = compute_adaptive_parameters(
        env, targets, own_draught, own_length,
        base_config, sensitivity_feedback,
    )

    # Apply to tube MPC
    controller.tube_mpc.rho_0 = params["rho_0"]
    controller.tube_mpc.mpc.epsilon_default = params["epsilon"]
    controller.tube_mpc.mpc.max_iterations = max(
        controller.tube_mpc.mpc.max_iterations,
        300 if params["horizon"] > 25 else 200,
    )

    # Apply to CBF filter
    controller.cbf_filter.alpha = params["alpha_cbf"]

    # Apply to chance constraint evaluator
    controller.chance_evaluator.epsilon_default = params["epsilon"]

    # Store for reference
    controller._adaptive_params = params

    return params


def compute_sensitivity_feedback(
    sensitivity_results: list[dict],
    baseline_performance: dict,
    tolerance_threshold: float = 0.05,
) -> dict:
    """Compute calibration feedback from sensitivity experiment results.

    Analyzes sensitivity experiment outputs and produces parameter
    impact scores that guide the adaptive configuration.

    Args:
        sensitivity_results: List of dicts with keys:
            param_name, param_value, metric_name, metric_value.
        baseline_performance: Dict of metric_name -> baseline_value.
        tolerance_threshold: Acceptable performance degradation fraction.

    Returns:
        dict of param_name -> impact_score (0 = no impact, 1 = critical).
    """
    impacts = {}

    for result in sensitivity_results:
        param = result.get("param_name", "")
        metric = result.get("metric_name", "")
        value = result.get("metric_value", 0.0)
        baseline = baseline_performance.get(metric, value + 1e-6)

        # Relative change from baseline
        rel_change = abs(value - baseline) / max(abs(baseline), 1e-6)
        impact = min(1.0, rel_change)

        if param not in impacts:
            impacts[param] = []
        impacts[param].append(impact)

    # Aggregate: mean impact across metrics
    aggregated = {p: float(np.mean(scores)) for p, scores in impacts.items()}

    # Only return parameters with significant impact
    significant = {p: s for p, s in aggregated.items()
                   if s > tolerance_threshold}

    return significant
