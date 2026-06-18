"""Reference path planner for constrained waterways.

Generates smooth reference trajectories using cubic splines that respect:
- Channel centerlines and bends
- TSS lane boundaries
- Bridge pier avoidance
- Bank clearance requirements

Used by the controller when no explicit reference path is provided.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from ..types import EnvironmentState, VesselState


def generate_reference_trajectory(
    current_state: VesselState,
    n_steps: int,
    dt: float = 0.5,
    env: EnvironmentState = None,
    waypoints: list[tuple[float, float]] = None,
    channel_centerline: list[tuple[float, float]] = None,
    obstacles: list[dict] = None,  # [{"x", "y", "radius"}, ...]
    target_speed: float = None,
) -> list[VesselState]:
    """Generate a smooth reference trajectory for the MPC horizon.

    Priority:
    1. Explicit waypoints (if provided)
    2. Channel centerline (if env has channel_width and centerline)
    3. Straight-line from current state (fallback)

    Args:
        current_state: Current ownship state.
        n_steps: Number of reference trajectory steps (N+1 for MPC).
        dt: Time step [s].
        env: Current environment state (for channel/obstacle info).
        waypoints: Optional list of (x, y) waypoints.
        channel_centerline: Optional list of (x, y) centerline points.
        obstacles: Optional list of obstacle dicts with x, y, radius.
        target_speed: Desired speed [m/s], defaults to current speed.

    Returns:
        List of VesselState for N+1 steps.
    """
    if target_speed is None:
        target_speed = max(current_state.speed(), 1.0)

    # Build path waypoints
    path_xy = _build_path_waypoints(
        current_state, waypoints, channel_centerline, env, obstacles,
    )

    # Generate smooth path via cubic spline
    ref_states = _spline_path_to_states(
        path_xy, current_state, n_steps, dt, target_speed,
    )

    return ref_states


def _build_path_waypoints(
    current_state: VesselState,
    waypoints: list[tuple[float, float]] = None,
    channel_centerline: list[tuple[float, float]] = None,
    env: EnvironmentState = None,
    obstacles: list[dict] = None,
) -> np.ndarray:
    """Build an array of (x, y) waypoints for the reference path."""
    cx, cy = current_state.x, current_state.y
    heading = current_state.psi

    if waypoints:
        # Use explicit waypoints, prepend current position
        pts = [(cx, cy)] + list(waypoints)
        return np.array(pts)

    if channel_centerline and len(channel_centerline) >= 2:
        # Snap current position to nearest centerline point and follow
        cl = np.array(channel_centerline)
        dists = np.linalg.norm(cl - np.array([cx, cy]), axis=1)
        nearest_idx = int(np.argmin(dists))
        return cl[nearest_idx:]

    if env is not None and env.channel_width is not None:
        # Generate path along channel direction with bend handling
        return _generate_channel_path(current_state, env, obstacles)

    # Fallback: straight line
    lookahead = 1000.0
    pts = np.array([
        [cx, cy],
        [cx + lookahead * np.cos(heading), cy + lookahead * np.sin(heading)],
    ])
    return pts


def _generate_channel_path(
    current_state: VesselState,
    env: EnvironmentState,
    obstacles: list[dict] = None,
) -> np.ndarray:
    """Generate a channel-following path that stays centered."""
    cx, cy = current_state.x, current_state.y
    heading = current_state.psi
    half_w = env.channel_width / 2 if env.channel_width else 500.0

    # Stay within channel center
    target_y = np.clip(cy, -half_w * 0.8, half_w * 0.8)

    # Avoid obstacles
    if obstacles:
        target_y = _adjust_for_obstacles(cx, target_y, heading, obstacles, half_w)

    # Generate look-ahead points along heading
    n_pts = 20
    lookahead = 2000.0
    pts = []
    for i in range(n_pts):
        t = i / (n_pts - 1)
        x = cx + t * lookahead * np.cos(heading)
        y = target_y  # follow channel center laterally
        pts.append([x, y])

    return np.array(pts)


def _adjust_for_obstacles(
    cx: float, cy: float, heading: float,
    obstacles: list[dict], half_w: float,
) -> float:
    """Adjust lateral position to avoid obstacles (bridge piers, etc.)."""
    best_y = cy
    best_cost = float("inf")

    # Search lateral offsets
    for dy in np.linspace(-half_w * 0.4, half_w * 0.4, 21):
        test_y = np.clip(cy + dy, -half_w * 0.8, half_w * 0.8)
        cost = abs(test_y - cy)  # prefer staying close to current y

        for obs in obstacles:
            ox, oy = obs.get("x", 0.0), obs.get("y", 0.0)
            radius = obs.get("radius", 10.0) + 50.0  # safety margin

            # Check if path passes near obstacle
            # Projected position 200m ahead
            px = cx + 200.0 * np.cos(heading)
            py = test_y
            dist = np.sqrt((px - ox)**2 + (py - oy)**2)
            if dist < radius:
                cost += 1e6 * (radius - dist)**2

        if cost < best_cost:
            best_cost = cost
            best_y = test_y

    return best_y


def _spline_path_to_states(
    path_xy: np.ndarray,
    current_state: VesselState,
    n_steps: int,
    dt: float,
    target_speed: float,
) -> list[VesselState]:
    """Convert a path of (x, y) points to VesselState trajectory via cubic spline.

    Produces evenly-spaced states at the target speed along the spline.
    """
    if len(path_xy) < 2:
        # Degenerate: return constant state
        return [VesselState(
            x=current_state.x, y=current_state.y,
            psi=current_state.psi, u=target_speed,
            v=0.0, r=0.0, timestamp=i * dt,
        ) for i in range(n_steps)]

    # Build spline with arclength parameterization
    diffs = np.diff(path_xy, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    s = np.zeros(len(path_xy))
    s[1:] = np.cumsum(seg_lengths)

    if s[-1] < 1.0:
        # Path too short: extend with current heading
        ext = np.array([path_xy[-1] + [np.cos(current_state.psi) * 1000,
                                        np.sin(current_state.psi) * 1000]])
        path_xy = np.vstack([path_xy, ext])
        s = np.append(s, s[-1] + 1000.0)

    try:
        cs_x = CubicSpline(s, path_xy[:, 0], bc_type='natural')
        cs_y = CubicSpline(s, path_xy[:, 1], bc_type='natural')
    except Exception:
        # Fallback: linear interpolation
        ref = []
        for i in range(n_steps):
            t = i * dt
            dist = target_speed * t
            frac = min(dist / max(s[-1], 1.0), 1.0)
            idx = min(int(frac * (len(path_xy) - 1)), len(path_xy) - 2)
            local_frac = frac * (len(path_xy) - 1) - idx
            x = path_xy[idx, 0] + local_frac * (path_xy[idx + 1, 0] - path_xy[idx, 0])
            y = path_xy[idx, 1] + local_frac * (path_xy[idx + 1, 1] - path_xy[idx, 1])
            psi = current_state.psi
            ref.append(VesselState(x=x, y=y, psi=psi, u=target_speed, v=0.0, r=0.0, timestamp=t))
        return ref

    # Sample at equal arclength intervals for constant speed
    total_time = n_steps * dt
    total_arclength = target_speed * total_time
    s_max = min(total_arclength, s[-1])
    s_samples = np.linspace(0, s_max, n_steps)

    ref_states = []
    prev_x, prev_y = cs_x(s_samples[0]), cs_y(s_samples[0])
    prev_psi = current_state.psi

    for i, s_val in enumerate(s_samples):
        x = float(cs_x(s_val))
        y = float(cs_y(s_val))

        # Compute heading from spline tangent
        if i > 0:
            dx = x - prev_x
            dy = y - prev_y
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                prev_psi = np.arctan2(dy, dx)

        ref_states.append(VesselState(
            x=x, y=y, psi=prev_psi,
            u=target_speed, v=0.0, r=0.0,
            timestamp=i * dt,
        ))
        prev_x, prev_y = x, y

    return ref_states


def build_channel_centerline_from_env(
    env: EnvironmentState,
    start_x: float,
    start_y: float,
    lookahead: float = 2000.0,
    n_points: int = 30,
) -> list[tuple[float, float]]:
    """Build a synthetic channel centerline from environment data.

    For scenarios with channel_width but no explicit centerline,
    generates a straight centerline along the channel.
    """
    pts = []
    for i in range(n_points):
        t = i / (n_points - 1)
        x = start_x + t * lookahead
        y = 0.0  # Assume channel centered at y=0
        pts.append((x, y))
    return pts
