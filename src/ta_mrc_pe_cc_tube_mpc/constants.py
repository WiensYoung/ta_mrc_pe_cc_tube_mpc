"""Physical and algorithmic constants for the project."""

import numpy as np

# Earth / physics
GRAVITY = 9.81               # [m/s^2]
RHO_WATER = 1025.0            # [kg/m^3]
KNOTS_TO_MPS = 0.514444       # 1 kn = 0.514444 m/s
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

# Numerical
EPS_V = 1.0e-6                # small velocity for TCPA
EPS_D = 1.0e-3                # small distance
INF = 1e12                    # large number for constraints

# Shallow water classification thresholds
SHALLOW_DEEP = 4.0
SHALLOW_MILD = 3.0
SHALLOW_MODERATE = 2.0
SHALLOW_STRONG = 1.5

# Ship interaction activation
K_L_INTERACTION = 2.0         # longitudinal factor
K_B_INTERACTION = 5.0         # lateral factor
T_CRIT_INTERACTION = 60.0     # [s] TCPA threshold

# Default reference values
U_REF_DEFAULT = 7.0           # [m/s]
D_REF_DEFAULT = 1000.0        # [m]
D_MIN_DEFAULT = 10.0          # [m]

# COLREGS
# Bearing ranges use half-open intervals [min, max) to avoid boundary ambiguity.
# Exact boundary values (5.0, 112.5, 247.5, 355.0) belong to the range that
# starts at that value, not the one that ends at it.
HEAD_ON_BEARING_RANGE = (355.0, 5.0)    # [deg] — wraps through 0/360
CROSSING_GIVEWAY_RANGE = (5.0, 112.5)   # [deg] — [5.0, 112.5)
CROSSING_STANDON_RANGE = (247.5, 355.0) # [deg] — [247.5, 355.0)
OVERTAKING_RANGE = (112.5, 247.5)       # [deg] — [112.5, 247.5)

# Failure taxonomy codes
FAILURE_CODES = {
    "F1": "Collision",
    "F2": "Near-miss",
    "F3": "Rule violation",
    "F4": "Channel boundary violation",
    "F5": "Bank clearance violation",
    "F6": "Inter-ship clearance violation",
    "F7": "MPC infeasibility",
    "F8": "CBF infeasible / over-intervention",
    "F9": "Fallback failure",
    "F10": "Runtime deadline miss",
}

# Baseline method identifiers
BASELINE_METHODS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "Proposed"]
PRIORITY_BASELINES = ["B3", "B4", "B5", "B6", "B7", "B8", "Proposed"]

# ── Navigation status constants (BUG #14 fix — 2026-06-10) ──────────────
# Single source of truth for nav_status string values.  All modules that
# compare against nav_status MUST import these constants rather than using
# bare string literals, which are fragile to spelling variations
# (e.g. British "manoeuvrability" vs American "maneuverability").
NAV_STATUS = {
    "UNDERWAY": "underway",
    "AT_ANCHOR": "at_anchor",
    "MOORED": "moored",
    "AGROUND": "aground",
    "NOT_UNDER_COMMAND": "not_under_command",
    "RESTRICTED_MANOEUVRABILITY": "restricted_maneuverability",
    "CONSTRAINED_BY_DRAUGHT": "constrained_by_draught",
    "FISHING": "fishing",
    "SAILING": "sailing",
    "TOWING": "towing",
}
