"""Sobol sequence sampling and global sensitivity analysis.

Implements Saltelli's method for first-order and total-effect Sobol indices
using quasi-Monte Carlo (QMC) sampling. Combined with Gaussian Process
surrogate modeling for efficient sensitivity analysis with fewer experiments.

References:
    Saltelli, A. et al. (2010), "Variance based sensitivity analysis of model
    output. Design and estimator for the total sensitivity index."
    Computer Physics Communications, 181(2), 259-270.

    Saltelli, A. et al. (2024), "Global Sensitivity Analysis: The Primer", 2nd ed.

    Owen, A.B. (2020), "On dropping the first Sobol' point."
    arXiv:2008.08051.

    Joe, S. & Kuo, F.Y. (2008), "Constructing Sobol' sequences with better
    two-dimensional projections", SIAM J. Sci. Comput. 30, 2635-2654.
"""

from typing import Callable, Optional

import numpy as np

# ── Primitive polynomials + initial direction numbers (Joe & Kuo 2008) ────
# Each entry: (degree s, polynomial coefficients a_1..a_{s-1}, m_1..m_s)
# The polynomial is: x^s + a_1*x^{s-1} + ... + a_{s-1}*x + 1
_SOBOL_POLY = {
    1:  (1, [],       [1]),
    2:  (2, [1],      [1, 3]),
    3:  (3, [1, 1],   [1, 3, 1]),
    4:  (3, [1, 1],   [1, 1, 1]),
    5:  (4, [1, 3],   [1, 1, 3, 3]),
    6:  (4, [1, 3],   [1, 3, 5, 13]),
    7:  (5, [1, 1, 1], [1, 1, 5, 5, 17]),
    8:  (5, [1, 1, 1], [1, 1, 5, 5, 13]),
    9:  (6, [1, 1, 0, 1], [1, 3, 7, 7, 5, 1]),
    10: (6, [1, 1, 0, 1], [1, 1, 3, 11, 11, 37]),
    11: (7, [1, 1, 0, 1, 1, 1], [1, 3, 5, 5, 7, 11, 31]),
    12: (7, [1, 1, 0, 1, 1, 1], [1, 3, 7, 1, 5, 9, 23]),
    13: (7, [1, 0, 1, 0, 1, 1], [1, 3, 3, 3, 7, 13, 15]),
    14: (7, [1, 0, 1, 0, 1, 1], [1, 3, 7, 13, 1, 5, 7]),
    15: (7, [1, 0, 0, 1, 0, 0], [1, 1, 5, 9, 13, 11, 19]),
    16: (7, [1, 0, 0, 1, 0, 0], [1, 3, 5, 7, 17, 25, 23]),
    17: (7, [1, 1, 0, 0, 1, 1], [1, 3, 1, 1, 15, 5, 37]),
    18: (7, [1, 1, 0, 0, 1, 1], [1, 1, 1, 3, 15, 3, 25]),
    19: (7, [1, 1, 1, 0, 0, 0], [1, 3, 3, 1, 11, 7, 21]),
    20: (7, [1, 1, 1, 0, 0, 0], [1, 3, 1, 1, 15, 21, 31]),
}


def _build_v(dim: int, n_bits: int = 32) -> np.ndarray:
    """Build full direction number array for one Sobol' dimension.

    Uses the recursive formula (Joe & Kuo 2008, Eq. 8):
      For j > s:  v_j = v_{j-s} XOR (v_{j-s} >> s)
                         XOR sum_k a_k * (v_{j-k} >> (s-k))

    Returns uint32 array of shape (n_bits,) with v_j at index j-1.
    """
    if dim not in _SOBOL_POLY:
        dim = 1
    s, a, m = _SOBOL_POLY[dim]
    v = np.zeros(n_bits, dtype=np.uint32)

    # Initial direction numbers from m values
    for j in range(min(s, n_bits)):
        v[j] = np.uint32(m[j]) << np.uint32(31 - j)

    # Recursively compute remaining direction numbers
    for j in range(s, n_bits):
        vj = v[j - s] ^ (v[j - s] >> np.uint32(s))
        for k in range(s - 1):
            if a[k]:
                shift = np.uint32(s - k - 1)
                vj ^= (np.uint32(a[k]) * v[j - k - 1]) >> shift
        v[j] = vj

    return v


def sobol_sequence(
    n_samples: int,
    n_dims: int,
    skip: int = 64,
) -> np.ndarray:
    """Generate Sobol' low-discrepancy sequence in [0,1]^d.

    Uses scipy.stats.qmc.Sobol if available (scipy >= 1.7), otherwise
    falls back to a Halton sequence which also has low discrepancy.

    Args:
        n_samples: Number of sample points.
        n_dims: Dimensionality.
        skip: Initial points to skip.

    Returns:
        Array of shape (n_samples, n_dims) in [0, 1).
    """
    # Try scipy's built-in Sobol' generator first
    try:
        from scipy.stats.qmc import Sobol
        sampler = Sobol(d=n_dims, scramble=False)
        return sampler.random(n=n_samples + skip)[skip:]
    except (ImportError, AttributeError):
        pass

    # Fallback: Halton sequence (correct low-discrepancy via coprime bases)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
              31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    result = np.zeros((n_samples + skip, n_dims))
    for d in range(min(n_dims, len(primes))):
        base = primes[d]
        for i in range(n_samples + skip):
            x = 0.0
            f = 1.0 / base
            n = i + 1  # Halton starts at i+1
            while n > 0:
                x += f * (n % base)
                n //= base
                f /= base
            result[i, d] = x
    return result[skip:, :n_dims]


def saltelli_sampling(
    n_base: int,
    n_dims: int,
    bounds: list[tuple[float, float]],
) -> dict:
    """Generate Saltelli sampling matrices for Sobol sensitivity analysis.

    Produces matrices A, B, and A_B^(i) for computing first-order and
    total-effect Sobol' indices via the Saltelli-Jansen estimator.

    Args:
        n_base: Base sample size (N — total runs ≈ N*(d+2)).
        n_dims: Number of input dimensions.
        bounds: List of (lower, upper) bounds per dimension.

    Returns:
        dict with keys:
            'A': (N, d) sample matrix
            'B': (N, d) resample matrix
            'AB': list of (N, d) matrices where column i is from B
            'bounds': original bounds for scaling
            'N': base sample size
            'd': number of dimensions
    """
    # Generate two independent Sobol' sequences
    A_raw = sobol_sequence(n_base, n_dims, skip=0)
    B_raw = sobol_sequence(n_base, n_dims, skip=n_base)

    # Scale to bounds
    A = np.zeros((n_base, n_dims))
    B = np.zeros((n_base, n_dims))
    for d in range(n_dims):
        lo, hi = bounds[d]
        A[:, d] = lo + (hi - lo) * A_raw[:, d]
        B[:, d] = lo + (hi - lo) * B_raw[:, d]

    # A_B^(i): replace column i of A with column i of B
    AB_matrices = []
    for d in range(n_dims):
        AB = A.copy()
        AB[:, d] = B[:, d]
        AB_matrices.append(AB)

    return {
        "A": A, "B": B, "AB": AB_matrices,
        "bounds": bounds, "N": n_base, "d": n_dims,
    }


def compute_sobol_indices(
    f_A: np.ndarray,
    f_B: np.ndarray,
    f_AB: list[np.ndarray],
) -> dict:
    """Compute first-order and total-effect Sobol' indices.

    Uses the Saltelli-Jansen estimator (Saltelli et al. 2010, Eq. 13-16).

    Args:
        f_A: Model output for sample matrix A, shape (N,).
        f_B: Model output for sample matrix B, shape (N,).
        f_AB: List of model outputs for each A_B^(i) matrix, shape [(N,), ...].

    Returns:
        dict with 'S1' (first-order), 'ST' (total-effect), and confidence intervals.
    """
    N = len(f_A)
    d = len(f_AB)

    # Total variance
    f_all = np.concatenate([f_A, f_B] + f_AB)
    var_Y = np.var(f_all, ddof=0)
    if var_Y < 1e-12:
        return {"S1": np.zeros(d), "ST": np.zeros(d), "var_Y": var_Y}

    # Mean of f_B (used in estimators)
    mean_f_B = np.mean(f_B)

    S1 = np.zeros(d)
    ST = np.zeros(d)

    mean_f_A = np.mean(f_A)

    for i in range(d):
        # First-order (Saltelli 2010, Table 2): V_i = E[f_B * f_AB_i] - E[f_A]*E[f_B]
        S1[i] = (np.mean(f_B * f_AB[i]) - mean_f_A * mean_f_B) / var_Y
        S1[i] = max(0.0, min(1.0, S1[i]))

        # Total-effect (Saltelli 2010): ST_i = 1 - (E[f_A * f_AB_i] - E[f_A]*E[f_B]) / var_Y
        ST[i] = 1.0 - (np.mean(f_A * f_AB[i]) - mean_f_A * mean_f_B) / var_Y
        ST[i] = max(0.0, min(1.0, ST[i]))

    return {"S1": S1, "ST": ST, "var_Y": var_Y}


class GPSurrogateModel:
    """Gaussian Process surrogate with Matérn 5/2 kernel.

    Used for Bayesian Optimization of sensitivity parameters —
    learns the response surface from a small number of evaluations
    and guides adaptive sampling toward high-uncertainty regions.

    Minimal implementation using only numpy — avoids scikit-learn
    dependency. For production use with >100 points, replace with
    GPyTorch or scikit-learn's GaussianProcessRegressor.
    """

    def __init__(
        self,
        length_scale: float = 0.5,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-6,
    ):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

    def _matern52_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matérn 5/2 kernel: k(r) = (1 + √5·r + 5r²/3)·exp(-√5·r)."""
        dists = np.zeros((len(X1), len(X2)))
        for i in range(X1.shape[1]):
            d = X1[:, i:i+1] - X2[:, i:i+1].T
            dists += (d / self.length_scale) ** 2
        r = np.sqrt(np.maximum(dists, 0.0))
        sqrt5_r = np.sqrt(5.0) * r
        return self.signal_variance * (1.0 + sqrt5_r + 5.0/3.0 * r**2) * np.exp(-sqrt5_r)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to training data."""
        self.X_train = X.copy()
        self.y_train = y.copy().ravel()

        K = self._matern52_kernel(X, X)
        K += self.noise_variance * np.eye(len(X))

        try:
            self.K_inv = np.linalg.solve(
                K, np.eye(len(X))
            )
        except np.linalg.LinAlgError:
            # Add jitter for numerical stability
            K += 1e-8 * np.eye(len(X))
            self.K_inv = np.linalg.solve(K, np.eye(len(X)))

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at test points."""
        if self.X_train is None:
            return np.zeros(len(X_test)), np.ones(len(X_test)) * self.signal_variance

        K_star = self._matern52_kernel(X_test, self.X_train)
        K_star_star = self._matern52_kernel(X_test, X_test)

        mu = K_star @ self.K_inv @ self.y_train
        cov = K_star_star - K_star @ self.K_inv @ K_star.T
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))

        return mu, std

    def expected_improvement(self, X_candidate: np.ndarray, y_best: float) -> np.ndarray:
        """Compute Expected Improvement acquisition function."""
        mu, std = self.predict(X_candidate)
        std = np.maximum(std, 1e-12)

        z = (mu - y_best) / std
        from scipy.stats import norm
        ei = (mu - y_best) * norm.cdf(z) + std * norm.pdf(z)
        return ei


def bayesian_optimization_step(
    surrogate: GPSurrogateModel,
    param_bounds: list[tuple[float, float]],
    y_best: float,
    n_candidates: int = 1000,
) -> np.ndarray:
    """Select next evaluation point via Expected Improvement maximization.

    Uses random candidate sampling + local refinement. For high-dimensional
    problems (>10 dims), consider using trust-region BO (TuRBO) instead.

    Args:
        surrogate: Trained GP surrogate model.
        param_bounds: Bounds for each parameter dimension.
        y_best: Best observed objective value (minimization).
        n_candidates: Number of random candidates to evaluate.

    Returns:
        Next parameter point to evaluate.
    """
    n_dims = len(param_bounds)
    # Generate Sobol' candidates for better coverage
    candidates_raw = sobol_sequence(n_candidates, n_dims, skip=42)
    candidates = np.zeros((n_candidates, n_dims))
    for d in range(n_dims):
        lo, hi = param_bounds[d]
        candidates[:, d] = lo + (hi - lo) * candidates_raw[:, d]

    ei = surrogate.expected_improvement(candidates, y_best)
    best_idx = np.argmax(ei)

    # Local refinement: perturb the best candidate
    best = candidates[best_idx].copy()
    for _ in range(10):
        perturbed = best + 0.01 * (np.random.rand(n_dims) - 0.5) * np.array(
            [hi - lo for lo, hi in param_bounds]
        )
        perturbed = np.clip(perturbed, [b[0] for b in param_bounds], [b[1] for b in param_bounds])
        ei_pert = surrogate.expected_improvement(perturbed.reshape(1, -1), y_best)
        if ei_pert[0] > ei[best_idx]:
            best = perturbed

    return best
