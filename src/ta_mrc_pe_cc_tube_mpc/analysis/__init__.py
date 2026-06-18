"""Analysis module for TA-MRC-PE-CC-Tube-MPC.

Provides:
    - stability.py: Recursive feasibility analysis (Mayne et al., 2005 framework)
    - conservatism.py: Additive tube conservatism quantification and calibration
"""

from .conservatism import (
    ConservatismAnalysis,
    calibrate_reduction_factor,
    component_correlation_analysis,
    monte_carlo_joint_bound,
    tube_utilization_analysis,
)
from .stability import (
    RecursiveFeasibilityResult,
    compute_robust_positive_invariant_set,
    verify_terminal_constraint,
    verify_tube_contraction,
    verify_unified_framework_recursive_feasibility,
    linearize_mmg_dynamics,
    compute_ancillary_feedback_gain,
    solve_terminal_cost_matrix,
)

__all__ = [
    # Stability
    "RecursiveFeasibilityResult",
    "compute_robust_positive_invariant_set",
    "verify_terminal_constraint",
    "verify_tube_contraction",
    "verify_unified_framework_recursive_feasibility",
    "linearize_mmg_dynamics",
    "compute_ancillary_feedback_gain",
    "solve_terminal_cost_matrix",
    # Conservatism
    "ConservatismAnalysis",
    "calibrate_reduction_factor",
    "component_correlation_analysis",
    "monte_carlo_joint_bound",
    "tube_utilization_analysis",
]
