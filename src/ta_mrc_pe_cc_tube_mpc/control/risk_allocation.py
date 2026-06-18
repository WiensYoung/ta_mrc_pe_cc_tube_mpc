"""Chance constraint risk allocation — Boole's inequality for horizon × targets.

Implements risk budget allocation from total epsilon to per-step epsilon
using Boole's inequality: epsilon_step = epsilon_total / (horizon * n_targets).

Reference:
    Mesbah (2016), "Stochastic Model Predictive Control: An Overview and
    Perspectives for Future Research", IEEE Control Systems, 36(6).
"""

import numpy as np


def allocate_chance_risk(
    epsilon_total: float,
    horizon_steps: int,
    n_targets: int,
    mode: str = "sqrt_boole",
) -> float:
    """Allocate per-constraint epsilon from a total risk budget.

    Args:
        epsilon_total: Total allowed violation probability (e.g., 0.10).
        horizon_steps: MPC prediction horizon length.
        n_targets: Number of target ships.
        mode: Allocation mode.

            - "none": No allocation — return epsilon_total directly.
              The claim is "pairwise per-step chance margin; no joint
              horizon-level probability guarantee."

            - "boole": Conservative Boole risk allocation:
              epsilon_step = epsilon_total / max(1, horizon_steps * n_targets).
              Under independence assumptions, this provides a conservative
              bound on the joint violation probability.  Very conservative
              for large horizons or many targets.

            - "sqrt_boole": Square-root Boole allocation (default):
              epsilon_step = epsilon_total / sqrt(horizon_steps * n_targets).
              Less conservative than strict Boole while still scaling with
              problem size.  For epsilon=0.10, N=2, T=12: kappa≈2.75
              vs strict Boole kappa≈3.3.

            - "per_target": Per-target allocation ignoring horizon:
              epsilon_step = epsilon_total / max(1, n_targets).

            - "per_horizon": Per-horizon allocation ignoring targets:
              epsilon_step = epsilon_total / max(1, horizon_steps).

    Returns:
        epsilon_step: Allocated per-constraint violation probability.

    Raises:
        ValueError: If epsilon_total is not in (0, 1] or mode is unknown.
    """
    if not (0.0 < epsilon_total <= 1.0):
        raise ValueError(f"epsilon_total must be in (0, 1], got {epsilon_total}")

    N = max(1, horizon_steps * n_targets)

    if mode == "none":
        return epsilon_total
    elif mode == "boole":
        return epsilon_total / float(N)
    elif mode == "sqrt_boole":
        return epsilon_total / float(np.sqrt(N))
    elif mode == "per_target":
        return epsilon_total / float(max(1, n_targets))
    elif mode == "per_horizon":
        return epsilon_total / float(max(1, horizon_steps))
    else:
        raise ValueError(
            f"Unknown allocation_mode '{mode}'. "
            f"Valid: none, boole, sqrt_boole, per_target, per_horizon."
        )


def format_claim_statement(mode: str) -> str:
    """Return the appropriate claim statement for the allocation mode.

    This is used in README, paper draft, and audit to ensure the claim
    matches the actual implementation.

    Args:
        mode: Allocation mode string.

    Returns:
        One-sentence claim statement.
    """
    if mode == "none":
        return (
            "Pairwise per-step chance margin; "
            "no joint horizon-level probability guarantee."
        )
    elif mode == "boole":
        return (
            "Conservative Boole risk allocation for pairwise horizon "
            "constraints under model assumptions."
        )
    elif mode == "sqrt_boole":
        return (
            "Square-root Boole risk allocation for pairwise horizon "
            "constraints — less conservative than strict Boole while "
            "still scaling with problem size."
        )
    elif mode == "per_target":
        return (
            "Per-target risk allocation (no horizon decomposition); "
            "no joint horizon-level guarantee."
        )
    elif mode == "per_horizon":
        return (
            "Per-horizon risk allocation (no target decomposition); "
            "no multi-target joint guarantee."
        )
    return "Unknown allocation mode."
