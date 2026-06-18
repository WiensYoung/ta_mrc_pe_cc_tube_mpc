"""Uncertainty modeling for target ship position estimation.

Uses relative covariance: Sigma_rel = Sigma_j + Sigma_i.
Supports AIS delay, dropout, and vessel-specific uncertainty inflation.
"""

import numpy as np

from ..utils.math_utils import kappa_epsilon, max_eigenvalue


def compute_relative_covariance(
    Sigma_own: np.ndarray,
    Sigma_tgt: np.ndarray,
) -> np.ndarray:
    """Compute relative position covariance.

    Sigma_rel = Sigma_j + Sigma_i

    Args:
        Sigma_own: Ownship position covariance (2x2).
        Sigma_tgt: Target ship position covariance (2x2).

    Returns:
        Sigma_rel: Relative covariance (2x2).
    """
    return Sigma_own + Sigma_tgt


def inflate_covariance_delay(
    Sigma: np.ndarray,
    velocity: np.ndarray,
    delay: float,
) -> np.ndarray:
    """Inflate covariance due to AIS reporting delay.

    Position uncertainty grows with delay: Sigma_delay ≈ v*v^T * delay^2
    This is a conservative approximation.

    Args:
        Sigma: Base covariance (2x2).
        velocity: Target velocity [vx, vy] in world frame.
        delay: AIS delay [s].

    Returns:
        Inflated covariance (2x2).
    """
    if delay <= 0:
        return Sigma
    v = velocity[:2].reshape(2, 1)
    delay_inflation = v @ v.T * delay**2
    return Sigma + delay_inflation


def inflate_covariance_dropout(
    Sigma: np.ndarray,
    n_missing: int = 1,
) -> np.ndarray:
    """Inflate covariance when AIS dropout occurs.

    Inflates covariance proportionally to dropout duration.

    Args:
        Sigma: Current covariance (2x2).
        n_missing: Number of consecutive missing observations.

    Returns:
        Inflated covariance (2x2).
    """
    # Growth factor: 2× for 1-step dropout, growing with sqrt(duration)
    # to model accumulating process noise during data gaps.
    factor = 2.0 * np.sqrt(max(n_missing, 1))
    return Sigma * factor


def compute_chance_constraint_margin(
    mu_rel: np.ndarray,
    Sigma_rel: np.ndarray,
    d_safe_eff: float,
    epsilon: float,
    rho_tube: float = 0.0,
) -> dict:
    """Compute the chance-constrained safety margin.

    Uses the conservative deterministic reformulation:
        ||mu_rel|| >= d_safe_eff + kappa_epsilon * sqrt(lambda_max(Sigma_rel)) + rho_tube

    where kappa_epsilon = sqrt(chi2.ppf(1 - epsilon, df=2))

    Args:
        mu_rel: Mean relative position vector [2].
        Sigma_rel: Relative covariance (2x2).
        d_safe_eff: Effective safe distance [m].
        epsilon: Constraint violation probability.
        rho_tube: Tube radius [m].

    Returns:
        dict with:
            margin: Safety margin (positive = safe) [m].
            required_distance: Required separation [m].
            current_distance: ||mu_rel|| [m].
            kappa: Chance constraint scaling factor.
            lambda_max: Max eigenvalue of Sigma_rel.
            constraint_satisfied: Boolean.
    """
    kappa = kappa_epsilon(epsilon, df=2)
    lambda_max_val = max_eigenvalue(Sigma_rel)
    current_distance = float(np.linalg.norm(mu_rel))
    uncertainty_margin = kappa * np.sqrt(max(lambda_max_val, 0.0))
    required_distance = d_safe_eff + uncertainty_margin + rho_tube
    margin = current_distance - required_distance

    return {
        "margin": margin,
        "required_distance": required_distance,
        "current_distance": current_distance,
        "kappa": kappa,
        "lambda_max": lambda_max_val,
        "uncertainty_margin": uncertainty_margin,
        "rho_tube": rho_tube,
        "constraint_satisfied": margin >= 0,
    }


def get_epsilon_for_waterway(waterway_type: str) -> float:
    """Return the appropriate epsilon for the waterway context.

    Args:
        waterway_type: One of 'normal', 'restricted', 'tss_ferry', 'tanker_lowvis'.

    Returns:
        epsilon value.
    """
    mapping = {
        "normal": 0.10,
        "restricted": 0.05,
        "tss_ferry": 0.025,
        "tanker_lowvis": 0.01,
    }
    return mapping.get(waterway_type, 0.10)
