"""Oriented bounding-box collision detection for vessel hull geometry.

Uses Separating Axis Theorem (SAT) to detect overlaps between two
oriented rectangles representing ship hulls. This replaces the
isotropic centroid-distance threshold with proper geometry-aware
collision detection.

References:
    Ericson (2005), "Real-Time Collision Detection", Morgan Kaufmann.
"""

import numpy as np


def oriented_box_corners(
    x: float,
    y: float,
    psi: float,
    length: float,
    beam: float,
) -> np.ndarray:
    """Compute the four corners of an oriented rectangular box.

    The box is centered at (x, y), oriented with heading psi [rad],
    with the long axis along the heading direction.

    Args:
        x, y: Center position [m].
        psi: Heading angle [rad], 0 = East (x-axis).
        length: Box length (along heading) [m].
        beam: Box width (perpendicular to heading) [m].

    Returns:
        Array of shape (4, 2) with corner coordinates in order:
        [front-right, front-left, aft-left, aft-right].
    """
    c = np.cos(psi)
    s = np.sin(psi)
    half_len = length / 2.0
    half_beam = beam / 2.0

    # Local coordinates of corners: (l, b) where l is along heading, b is to port
    local_corners = np.array([
        [ half_len, -half_beam],   # front-right (starboard bow)
        [ half_len,  half_beam],   # front-left (port bow)
        [-half_len,  half_beam],   # aft-left (port quarter)
        [-half_len, -half_beam],   # aft-right (starboard quarter)
    ])

    # Rotate to world frame
    R = np.array([[c, -s], [s, c]])
    world_corners = np.dot(local_corners, R.T) + np.array([x, y])

    return world_corners


def _project_polygon(poly: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """Project a polygon onto an axis, return (min, max)."""
    dots = np.dot(poly, axis)
    return float(np.min(dots)), float(np.max(dots))


def polygons_overlap_sat(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Check if two convex polygons overlap using Separating Axis Theorem.

    Projects both polygons onto each edge normal of both polygons.
    If any axis separates them (projections don't overlap), returns False.
    Otherwise returns True (overlap detected).

    Args:
        poly_a: Polygon A vertices as (N, 2) array.
        poly_b: Polygon B vertices as (M, 2) array.

    Returns:
        True if polygons overlap, False otherwise.
    """
    # Get edge normals for both polygons
    def _edge_normals(poly):
        n = len(poly)
        normals = []
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            # Perpendicular to edge: (-y, x) rotated 90 deg
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normals.append(normal / norm)
        return normals

    axes = _edge_normals(poly_a) + _edge_normals(poly_b)

    for axis in axes:
        min_a, max_a = _project_polygon(poly_a, axis)
        min_b, max_b = _project_polygon(poly_b, axis)

        if max_a < min_b or max_b < min_a:
            return False  # Separating axis found

    return True  # No separating axis → overlap


def hulls_overlap_oriented_rect(
    own_x: float,
    own_y: float,
    own_psi: float,
    own_length: float,
    own_beam: float,
    target_x: float,
    target_y: float,
    target_psi: float,
    target_length: float,
    target_beam: float,
    safety_padding: float = 0.0,
) -> bool:
    """Check if two vessel hulls overlap using oriented rectangle SAT.

    Safety_padding can be added to expand hulls for near-miss detection,
    but collision=TRUE must use safety_padding=0.0 (actual hull overlap).

    Args:
        own_x, own_y: Ownship center position [m].
        own_psi: Ownship heading [rad].
        own_length: Ownship length [m].
        own_beam: Ownship beam [m].
        target_x, target_y: Target center position [m].
        target_psi: Target heading [rad].
        target_length: Target length [m].
        target_beam: Target beam [m].
        safety_padding: Extra padding around hulls [m] (0 for collision).

    Returns:
        True if hulls overlap, False otherwise.
    """
    own_corners = oriented_box_corners(
        own_x, own_y, own_psi,
        own_length + 2 * safety_padding,
        own_beam + 2 * safety_padding,
    )
    target_corners = oriented_box_corners(
        target_x, target_y, target_psi,
        target_length + 2 * safety_padding,
        target_beam + 2 * safety_padding,
    )
    return polygons_overlap_sat(own_corners, target_corners)


def centroid_distance(
    own_x: float, own_y: float,
    target_x: float, target_y: float,
) -> float:
    """Compute centroid-to-centroid distance."""
    return float(np.sqrt((own_x - target_x) ** 2 + (own_y - target_y) ** 2))
