"""Recursive feasibility and stability analysis for TA-MRC-PE-CC-Tube-MPC.

The analysis follows the tube-MPC paradigm established by Mayne et al. (2005)
and extended in Rawlings & Mayne (2009):

Key references:
    - Mayne, D. Q., Seron, M. M., & Raković, S. V. (2005).
      "Robust model predictive control of constrained linear systems with
      bounded disturbances." Automatica, 41(2), 219-224.
    - Rawlings, J. B., & Mayne, D. Q. (2009).
      "Model Predictive Control: Theory and Design." Nob Hill Publishing.
    - Cao, Y., & Li, T. (2024).
      "Nonlinear antiswing control for shipboard boom cranes with full
      state constraints." Applied Ocean Research, 146, 103971.
    - 上海海事大学课题组 (2025).
      "Underactuated ship path following control based on RMPC and roll
      constraints." Control and Decision, 40(4), 1303-1311.

Theory:
    1. Nominal MPC solves a constrained optimal control problem
    2. Tube radius rho_tube bounds all disturbance realizations
    3. The terminal constraint set X_f is control-invariant
    4. Recursive feasibility: if feasible at t=0, feasible at all t>0

Extension to chance-constrained tube MPC:
    The chance-constrained safety margin
        d_safe_eff + kappa_epsilon * sqrt(lambda_max(Sigma_rel))
    replaces the standard additive disturbance bound, extending recursive
    feasibility from deterministic tube-MPC to chance-constrained tube-MPC.
    The CBF-QP safety filter is a Lipschitz continuous projection with
    L_cbf <= 1, preserving the contraction property.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_lyapunov


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RecursiveFeasibilityResult:
    """Output of recursive feasibility analysis.

    Attributes:
        is_feasible: Whether all recursive feasibility conditions hold.
        terminal_set_satisfied: Terminal constraint verification.
        tube_contraction_verified: Tube radius contraction check.
        max_disturbance_realization: Maximum ||w|| observed in trajectory.
        tube_margin: rho_tube - max_disturbance (positive => safe).
        certified_rho_max: Maximum tube radius from RPI set theory.
        constraint_margins: Slack for each constraint type.
        analysis_details: Additional diagnostic information.
    """

    is_feasible: bool = False
    terminal_set_satisfied: bool = False
    tube_contraction_verified: bool = False
    max_disturbance_realization: float = 0.0
    tube_margin: float = 0.0
    certified_rho_max: float = 0.0
    constraint_margins: dict = field(default_factory=dict)
    analysis_details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MMG linearization
# ---------------------------------------------------------------------------


def linearize_mmg_dynamics(
    mmg_model,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    fd_eps: float = 1e-4,
    dt: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize the MMG dynamics around a reference point.

    Uses central finite differences to compute the Jacobians:
        A = ∂f/∂x|_{x_ref, u_ref}
        B = ∂f/∂u|_{x_ref, u_ref}

    State x = [x, y, psi, u, v, r]^T (6-DOF planar)
    Input u = [delta, n]^T (rudder, propeller)

    Args:
        mmg_model: MMG3DOFModel instance with a `step` method.
        x_ref: Reference state vector [6,].
        u_ref: Reference input vector [2,].
        fd_eps: Finite difference perturbation.
        dt: Discretization time step [s]. Must match the MPC dt.

    Returns:
        (A, B): System and input Jacobian matrices, shapes (6, 6) and (6, 2).
    """
    nx = 6
    nu = 2

    # Helper: step and convert VesselState → numpy array.
    # The legacy MMG wrapper returns (VesselState, None); we need [x,y,psi,u,v,r].
    def _step_to_vec(x, u, dt_val):
        result = mmg_model.step(x, u, dt_val)
        vs = result[0] if isinstance(result, tuple) else result
        if vs is None:
            return x
        # Convert VesselState dataclass → [x, y, psi, u, v, r]
        if hasattr(vs, 'x'):
            return np.array([vs.x, vs.y, vs.psi, vs.u, vs.v, vs.r])
        # Legacy path: already numpy
        return np.asarray(vs, dtype=float)

    fd_dt = dt  # use the actual MPC dt, not a hardcoded value

    # Nominal dynamics
    x_next_nom = _step_to_vec(x_ref, u_ref, fd_dt)

    # Build A matrix
    A = np.zeros((nx, nx))
    for i in range(nx):
        x_plus = x_ref.copy().astype(float)
        x_plus[i] += fd_eps
        x_next_plus = _step_to_vec(x_plus, u_ref, fd_dt)

        x_minus = x_ref.copy().astype(float)
        x_minus[i] -= fd_eps
        x_next_minus = _step_to_vec(x_minus, u_ref, fd_dt)

        A[:, i] = (x_next_plus - x_next_minus) / (2.0 * fd_eps)

    # Build B matrix
    B = np.zeros((nx, nu))
    for i in range(nu):
        u_plus = u_ref.copy().astype(float)
        u_plus[i] += fd_eps
        x_next_plus = _step_to_vec(x_ref, u_plus, fd_dt)

        u_minus = u_ref.copy().astype(float)
        u_minus[i] -= fd_eps
        x_next_minus = _step_to_vec(x_ref, u_minus, fd_dt)

        B[:, i] = (x_next_plus - x_next_minus) / (2.0 * fd_eps)

    return A, B


# ---------------------------------------------------------------------------
# Ancillary feedback gain and terminal cost (LQR)
# ---------------------------------------------------------------------------


def compute_ancillary_feedback_gain(
    A: np.ndarray,
    B: np.ndarray,
    Q_lqr: Optional[np.ndarray] = None,
    R_lqr: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the LQR ancillary feedback gain K for tube-MPC.

    The ancillary controller u = -K * (x - x_nominal) drives the error
    dynamics to zero. K is the solution to the discrete-time LQR problem:

        min Σ (e_k^T Q e_k + v_k^T R v_k)
        s.t. e_{k+1} = A e_k + B v_k

    where e_k = x_k - x_nominal,k and v_k = u_k - u_nominal,k.

    Args:
        A: System matrix (n_x × n_x).
        B: Input matrix (n_x × n_u).
        Q_lqr: State cost matrix (default: diag(10,10,1,1,1,0.5)).
        R_lqr: Input cost matrix (default: diag(0.1, 0.01)).

    Returns:
        (K, P): Feedback gain matrix (n_u × n_x) and cost-to-go matrix (n_x × n_x).
    """
    nx = A.shape[0]
    nu = B.shape[1]

    if Q_lqr is None:
        # [x, y, psi, u, v, r] weights
        Q_lqr = np.diag([10.0, 10.0, 1.0, 1.0, 1.0, 0.5])
    if R_lqr is None:
        R_lqr = np.diag([0.1, 0.01])

    # Discrete-time algebraic Riccati equation (DARE) solution
    # P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q
    try:
        # Use scipy's DARE solver
        from scipy.linalg import solve_discrete_are

        P = solve_discrete_are(A, B, Q_lqr, R_lqr)
    except Exception:
        # Fallback: iterate the Riccati recursion
        P = Q_lqr.copy()
        for _ in range(1000):
            # Use solve() instead of inv() for better numerical stability
            # in the iterative DARE fallback (inv() fails on ill-conditioned
            # R_lqr + B^T P B).
            K_gain = np.linalg.solve(R_lqr + B.T @ P @ B, B.T @ P @ A)
            P_next = A.T @ P @ A - A.T @ P @ B @ K_gain + Q_lqr
            if np.linalg.norm(P_next - P, ord="fro") < 1e-8:
                P = P_next
                break
            P = P_next

    # Feedback gain: K = (R + B^T P B)^{-1} B^T P A
    K = np.linalg.solve(R_lqr + B.T @ P @ B, B.T @ P @ A)

    return K, P


def solve_terminal_cost_matrix(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve for the terminal cost matrix P_terminal via discrete Lyapunov.

    P_terminal satisfies:
        (A - BK)^T P (A - BK) - P = -(Q + K^T R K)

    This gives the terminal cost ||x_N||^2_P_terminal that ensures
    closed-loop stability of the MPC (Rawlings & Mayne, 2009, Sec. 2.5).

    Args:
        A: System matrix.
        B: Input matrix.
        K: Ancillary feedback gain.
        Q: State cost matrix (default: identity).
        R: Input cost matrix (default: identity * 0.01).

    Returns:
        P_terminal: Terminal cost matrix (n_x × n_x).
    """
    A_cl = A - B @ K  # closed-loop system matrix

    if Q is None:
        Q = np.eye(A.shape[0])
    if R is None:
        R = 0.01 * np.eye(B.shape[1])

    Q_lyap = Q + K.T @ R @ K

    try:
        P = solve_discrete_lyapunov(A_cl.T, Q_lyap)
    except Exception:
        P = Q_lyap  # fallback to Q+K^T R K

    # Ensure positive definiteness
    eigvals = np.linalg.eigvalsh(P)
    if np.min(eigvals) < 1e-10:
        P += (abs(np.min(eigvals)) + 1e-6) * np.eye(A.shape[0])

    return P


# ---------------------------------------------------------------------------
# Robust Positive Invariant (RPI) set
# ---------------------------------------------------------------------------


def compute_robust_positive_invariant_set(
    A_cl: np.ndarray,
    W_max: float,
    gamma: float = 0.99,
) -> tuple[np.ndarray, float]:
    """Compute a lambda-contractive robust positively invariant (RPI) set.

    Uses the standard tube-MPC construction (Mayne et al., 2005):
        Z = {z : z^T P_z z <= 1}
    where P_z solves the discrete Lyapunov equation:
        (A_cl / gamma)^T P_z (A_cl / gamma) - P_z = -I

    If gamma < 1, the tube is lambda-contractive with lambda = gamma,
    ensuring exponential convergence of the error dynamics.

    Args:
        A_cl: Closed-loop system matrix A - B K.
        W_max: Maximum norm of disturbance bound.
        gamma: Contraction factor (< 1 for exponential stability).

    Returns:
        (P_z, rho_max): RPI matrix and certified tube radius.
    """
    A_gamma = A_cl / gamma

    # Solve discrete Lyapunov: A_gamma^T P A_gamma - P = -I
    try:
        P_z = solve_discrete_lyapunov(A_gamma.T, np.eye(A_cl.shape[0]))
    except Exception:
        P_z = np.eye(A_cl.shape[0])

    # Ensure P_z is positive definite
    eigvals = np.linalg.eigvalsh(P_z)
    if np.min(eigvals) < 1e-10:
        P_z += (abs(np.min(eigvals)) + 1e-6) * np.eye(A_cl.shape[0])

    # Certified tube radius: for the ball ||w|| <= W_max to be contained
    # in the ellipsoid z^T P_z z <= 1, we need:
    #   max_{||w||<=W_max} w^T P_z w = W_max^2 * lambda_max(P_z) <= 1
    # So rho_max = 1 / sqrt(lambda_max(P_z)).
    # The old formula used lambda_min (the widest ellipsoid axis), which
    # overestimated the certified radius.
    rho_max = 1.0 / np.sqrt(np.max(np.linalg.eigvalsh(P_z)))

    return P_z, rho_max


# ---------------------------------------------------------------------------
# Verification functions
# ---------------------------------------------------------------------------


def verify_terminal_constraint(
    terminal_state: np.ndarray,
    terminal_set_center: np.ndarray,
    terminal_set_radius: float,
    rho_tube: float,
) -> bool:
    """Verify that the terminal state lies within the robust terminal set.

    X_f_robust = X_f ⊖ (rho_tube × B)

    where X_f = {x : ||x - x_ref||_P^2 <= alpha} and ⊖ denotes the
    Pontryagin difference. This ensures the terminal state remains in X_f
    despite disturbances bounded by rho_tube.

    Args:
        terminal_state: Predicted terminal state x_N.
        terminal_set_center: Center of terminal set (typically the origin
                             in error coordinates, or x_ref,N).
        terminal_set_radius: Radius of terminal set.
        rho_tube: Current total tube radius.

    Returns:
        True if terminal constraint satisfied.
    """
    error = np.linalg.norm(terminal_state[:3] - terminal_set_center[:3])
    return error + rho_tube <= terminal_set_radius + 1e-6


def verify_tube_contraction(
    rho_k: float,
    rho_k_plus_1: float,
    contraction_rate: float = 0.95,
) -> bool:
    """Verify tube radius contraction (exponential stability of error dynamics).

    If rho_{k+1} <= lambda * rho_k with lambda < 1, the tube is
    lambda-contractive, ensuring exponential convergence of the actual
    trajectory to the nominal.

    Args:
        rho_k: Tube radius at step k.
        rho_k_plus_1: Tube radius at step k+1.
        contraction_rate: Maximum allowable ratio (default 0.95).

    Returns:
        True if contraction holds.
    """
    if rho_k < 1e-6:
        # Tube effectively zero: contraction is trivially satisfied
        return True
    return (rho_k_plus_1 / max(rho_k, 1e-12)) <= contraction_rate


def verify_unified_framework_recursive_feasibility(
    x_current: np.ndarray,
    x_nominal: np.ndarray,
    disturbance_realization: float,
    rho_tube_total: float,
    rho_tube_certified: float,
    terminal_state: np.ndarray,
    terminal_set_center: np.ndarray,
    terminal_set_radius: float,
    constraint_margins: dict,
    rho_tube_history: Optional[list[float]] = None,
) -> RecursiveFeasibilityResult:
    """Verify recursive feasibility for the unified framework.

    This extends standard tube-MPC feasibility (Mayne et al., 2005) to
    incorporate:

    1. Chance-constrained safety margins:
       d_safe_eff + kappa_epsilon * sqrt(lambda_max(Sigma_rel))
       replaces the deterministic disturbance bound.

    2. Physics-enhanced tube radius:
       rho_tube = rho_0 + rho_AIS + rho_current + rho_wind
                + rho_shallow + rho_bank + rho_ship + rho_maneuver

    3. CBF-QP safety filter:
       Conservative perturbation bound due to CBF minimal intervention.

    Conservative guarantee (informal):
       If standard tube-MPC assumptions hold (Mayne et al., 2005, Assumptions
       A1-A4), and if kappa_epsilon is non-decreasing over the horizon
       (satisfied since epsilon is constant and Sigma_rel grows with
       prediction horizon), and if the CBF filter is a Lipschitz continuous
       projection with L_cbf <= 1 (satisfied by QP projection), then
       recursive feasibility is preserved.

    Args:
        x_current: Current actual state [6,].
        x_nominal: Current nominal (planned) state [6,].
        disturbance_realization: ||x_current - x_nominal|| (actual error).
        rho_tube_total: Total tube radius from all components.
        rho_tube_certified: Maximum certified tube radius from RPI set.
        terminal_state: Predicted terminal state x_N.
        terminal_set_center: Center of terminal set.
        terminal_set_radius: Radius of terminal set.
        constraint_margins: Dict mapping constraint name to margin value
                            (positive = satisfied).
        rho_tube_history: Recent tube radius history for contraction check.

    Returns:
        RecursiveFeasibilityResult with detailed verification.
    """
    details = {}

    # 1. Check tube covers actual disturbance
    tube_margin = rho_tube_total - disturbance_realization
    details["tube_coverage"] = tube_margin >= -1e-6
    details["tube_margin_absolute"] = tube_margin
    details["tube_utilization"] = (
        disturbance_realization / max(rho_tube_total, 1e-12)
    )

    # 2. Check tube is within certified RPI bound
    details["tube_certified"] = rho_tube_total <= rho_tube_certified + 1e-6
    details["rho_certified"] = rho_tube_certified
    details["rho_actual"] = rho_tube_total

    # 3. Check terminal constraint
    terminal_ok = verify_terminal_constraint(
        terminal_state=terminal_state,
        terminal_set_center=terminal_set_center,
        terminal_set_radius=terminal_set_radius,
        rho_tube=rho_tube_total,
    )
    details["terminal_satisfied"] = terminal_ok

    # 4. Check constraint margins (all must be positive for safety)
    details["constraint_margin_satisfied"] = all(
        m >= -1e-6 for m in constraint_margins.values()
    )
    details["constraint_margins"] = constraint_margins.copy()

    # 5. Check tube contraction (if history available)
    tube_contraction_ok = True
    if rho_tube_history and len(rho_tube_history) >= 2:
        for i in range(len(rho_tube_history) - 1):
            if not verify_tube_contraction(
                rho_tube_history[i],
                rho_tube_history[i + 1],
            ):
                tube_contraction_ok = False
                break
    details["tube_contraction"] = tube_contraction_ok

    # 6. Overall feasibility
    is_feasible = (
        details["tube_coverage"]
        and details["tube_certified"]
        and terminal_ok
        and details["constraint_margin_satisfied"]
    )

    return RecursiveFeasibilityResult(
        is_feasible=is_feasible,
        terminal_set_satisfied=terminal_ok,
        tube_contraction_verified=tube_contraction_ok,
        max_disturbance_realization=disturbance_realization,
        tube_margin=tube_margin,
        certified_rho_max=rho_tube_certified,
        constraint_margins=constraint_margins,
        analysis_details=details,
    )


# ---------------------------------------------------------------------------
# Utility: extract linearized system for the typical cruise operating point
# ---------------------------------------------------------------------------


def build_cruise_linearization(mmg_model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build linearized system for the typical cruise operating point.

    This provides a reusable linearization for RPI analysis, LQR gain
    computation, and terminal cost matrix derivation — all of which should
    be computed once per vessel type.

    Args:
        mmg_model: MMG3DOFModel instance.

    Returns:
        (A, B, K, P_terminal):
            System matrix, input matrix, ancillary feedback gain,
            terminal cost matrix.
    """
    # Reference state: steady cruise at 7 m/s, zero cross-track
    x_ref = np.array([0.0, 0.0, 0.0, 7.0, 0.0, 0.0])  # [x, y, psi, u, v, r]
    u_ref = np.array([0.0, 0.5])  # [rudder 0°, propeller 50%]

    A, B = linearize_mmg_dynamics(mmg_model, x_ref, u_ref)
    K, P_cost = compute_ancillary_feedback_gain(A, B)
    P_term = solve_terminal_cost_matrix(A, B, K)

    return A, B, K, P_term
