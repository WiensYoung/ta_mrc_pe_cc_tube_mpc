"""Coordinate transformations: ENU/NED, geodetic conversion, local frames."""

import numpy as np

from ..constants import DEG_TO_RAD, RAD_TO_DEG


def body_to_world(u: float, v: float, psi: float) -> np.ndarray:
    """Convert body-frame velocity [u, v] to world-frame velocity.

    Args:
        u: Surge velocity [m/s].
        v: Sway velocity [m/s].
        psi: Heading angle [rad].

    Returns:
        world_vel: 2-element array [vx, vy] in world frame.
    """
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([u * c - v * s, u * s + v * c])


def world_to_body(vx: float, vy: float, psi: float) -> np.ndarray:
    """Convert world-frame velocity to body-frame.

    Args:
        vx, vy: World-frame velocity components [m/s].
        psi: Heading angle [rad].

    Returns:
        body_vel: 2-element array [u, v] in body frame.
    """
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([vx * c + vy * s, -vx * s + vy * c])


def relative_bearing(own_pos: np.ndarray, own_psi: float,
                     target_pos: np.ndarray) -> float:
    """Compute relative bearing from ownship to target.

    Args:
        own_pos: Ownship position [x, y].
        own_psi: Ownship heading [rad].
        target_pos: Target position [x, y].

    Returns:
        rel_bearing: Relative bearing [deg] in [0, 360).
    """
    dx = target_pos[0] - own_pos[0]
    dy = target_pos[1] - own_pos[1]
    true_bearing = np.arctan2(dy, dx)
    rel = (true_bearing - own_psi) * RAD_TO_DEG
    return rel % 360.0


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.linalg.norm(p2[:2] - p1[:2]))


def normalize_angle_rad(angle: float) -> float:
    """Normalize angle to [-pi, pi)."""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def normalize_angle_deg(angle: float) -> float:
    """Normalize angle to [-180, 180)."""
    return ((angle + 180.0) % 360.0) - 180.0


def rotation_matrix_2d(psi: float) -> np.ndarray:
    """2D rotation matrix for angle psi [rad]."""
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[c, -s], [s, c]])
