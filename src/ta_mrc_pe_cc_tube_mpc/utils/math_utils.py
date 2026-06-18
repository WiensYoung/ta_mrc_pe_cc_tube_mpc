"""Mathematical utility functions."""

import numpy as np
from scipy.stats import chi2


def clip_value(value: float, lower: float, upper: float) -> float:
    """Clip value to [lower, upper]."""
    return max(lower, min(upper, value))


def safe_divide(a: float, b: float, eps: float = 1e-12) -> float:
    """Safe division: returns a/b or 0 if denominator is near zero."""
    if abs(b) < eps:
        return 0.0
    return a / b


def kappa_epsilon(epsilon: float, df: int = 2) -> float:
    """Compute the chance constraint scaling factor from chi-squared quantile.

    kappa_epsilon = sqrt(chi2.ppf(1 - epsilon, df=2))

    Args:
        epsilon: Probability of constraint violation.
        df: Degrees of freedom (default 2 for 2D position).

    Returns:
        kappa: Scaling factor for covariance ellipsoid.
    """
    return float(np.sqrt(chi2.ppf(1.0 - epsilon, df=df)))


def max_eigenvalue(Sigma: np.ndarray) -> float:
    """Maximum eigenvalue of a 2x2 or nxn covariance matrix.

    Args:
        Sigma: Covariance matrix.

    Returns:
        lambda_max: Maximum eigenvalue.
    """
    if Sigma.shape == (2, 2):
        a = Sigma[0, 0]
        b = Sigma[0, 1]
        d = Sigma[1, 1]
        trace = a + d
        det = a * d - b * b
        return float(0.5 * (trace + np.sqrt(max(0, trace**2 - 4 * det))))
    w = np.linalg.eigvalsh(Sigma)
    return float(np.max(w))


def sigmoid(x: float, scale: float = 1.0) -> float:
    """Smooth sigmoid for soft constraints."""
    return 1.0 / (1.0 + np.exp(-scale * x))


def clamp_angle_diff(target: float, current: float) -> float:
    """Shortest angular difference target - current, in radians."""
    return np.arctan2(np.sin(target - current), np.cos(target - current))
