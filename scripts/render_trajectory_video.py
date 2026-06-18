#!/usr/bin/env python
"""Render animated trajectory video showing ship collision avoidance in action.

Output:
    MP4 video (if ffmpeg available) or GIF fallback.
    Each frame shows: ship positions, dynamic domains, safety margins,
    rule modes, CBF/fallback status, and solver timing.

Usage:
    python scripts/render_trajectory_video.py \
        --trajectory results/trajectories/test_traj_seed1.npz \
        --output results/videos/scenario_S2.mp4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ta_mrc_pe_cc_tube_mpc.types import VesselState, ControlCommand


def load_trajectory(path: str) -> dict:
    """Load trajectory data from .npz file."""
    data = np.load(path, allow_pickle=False)
    result = {}
    if "own_state" in data:
        arr = data["own_state"]
        result["own_states"] = [
            VesselState(x=r[0], y=r[1], psi=r[2], u=r[3], v=r[4], r=r[5], timestamp=r[6])
            for r in arr
        ]
    if "own_command" in data:
        arr = data["own_command"]
        result["commands"] = [
            ControlCommand(rudder=r[0], propeller=r[1],
                          cbf_modified=bool(r[2]), fallback_active=bool(r[3]),
                          source="")
            for r in arr
        ]
    target_keys = sorted([k for k in data if k.startswith("target_") and k.endswith("_state")])
    result["target_states"] = []
    for tk in target_keys:
        arr = data[tk]
        result["target_states"].append([
            VesselState(x=r[0], y=r[1], psi=r[2], u=r[3], v=r[4], r=r[5], timestamp=r[6])
            for r in arr
        ])
    return result


def render_video(traj: dict, output_path: str, fps: int = 2, dpi: int = 100):
    """Render trajectory animation to MP4 or GIF."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib not available. Install: pip install matplotlib")
        return False

    own = traj.get("own_states", [])
    targets = traj.get("target_states", [])
    commands = traj.get("commands", [])

    if not own:
        print("No ownship state data found.")
        return False

    n_frames = len(own)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax_map, ax_margin), (ax_control, ax_time) = axes

    # Pre-compute bounds
    all_x = [s.x for s in own]
    all_y = [s.y for s in own]
    for tgt in targets:
        all_x += [s.x for s in tgt]
        all_y += [s.y for s in tgt]
    x_range = max(all_x) - min(all_x) + 100
    y_range = max(all_y) - min(all_y) + 100
    pad = 0.1

    safety_margins = []
    for i, s in enumerate(own):
        min_d = float("inf")
        for tgt in targets:
            if i < len(tgt):
                d = np.sqrt((s.x - tgt[i].x) ** 2 + (s.y - tgt[i].y) ** 2)
                min_d = min(min_d, d)
        safety_margins.append(min_d if min_d != float("inf") else 0)

    solve_times = []
    for c in commands:
        solve_times.append(c.metadata.get("runtime", 0.0) if c.metadata else 0.0)

    def update(frame):
        for ax in axes.flat:
            ax.clear()

        # Map
        ax_map.set_xlim(min(all_x) - pad * x_range, max(all_x) + pad * x_range)
        ax_map.set_ylim(min(all_y) - pad * y_range, max(all_y) + pad * y_range)
        ax_map.set_aspect("equal")
        ax_map.set_title(f"Trajectory — t={frame * 0.5:.1f}s")

        # Ownship trail
        end = frame + 1
        ax_map.plot([s.x for s in own[:end]], [s.y for s in own[:end]], "b-", lw=1.5, label="Ownship")
        # Ownship hull (rectangle)
        if frame < n_frames:
            s = own[frame]
            L, B = 180, 30
            c, sn = np.cos(s.psi), np.sin(s.psi)
            corners = np.array([[-L / 2, -B / 2], [L / 2, -B / 2],
                                [L / 2, B / 2], [-L / 2, B / 2]])
            R = np.array([[c, -sn], [sn, c]])
            hull = corners @ R.T + np.array([s.x, s.y])
            ax_map.fill(hull[:, 0], hull[:, 1], "blue", alpha=0.3)
            ax_map.plot(hull[:, 0], hull[:, 1], "blue", lw=1)

            # Safety circle
            circle = plt.Circle((s.x, s.y), 500, fill=False, color="blue", ls="--", alpha=0.4)
            ax_map.add_patch(circle)

        # Targets
        for j, tgt in enumerate(targets):
            if frame < len(tgt):
                ts = tgt[frame]
                ax_map.plot(ts.x, ts.y, "o", color=colors[j % len(colors)], ms=8,
                            label=f"Target {j + 1}")
                L, B = 100, 15
                c, sn = np.cos(ts.psi), np.sin(ts.psi)
                corners = np.array([[-L / 2, -B / 2], [L / 2, -B / 2],
                                    [L / 2, B / 2], [-L / 2, B / 2]])
                R = np.array([[c, -sn], [sn, c]])
                hull = corners @ R.T + np.array([ts.x, ts.y])
                ax_map.fill(hull[:, 0], hull[:, 1], colors[j % len(colors)], alpha=0.2)

        ax_map.legend(fontsize=6, loc="upper right")

        # Safety margin
        ax_margin.plot(safety_margins[:end], "b-", lw=1)
        ax_margin.axhline(y=500, color="r", ls="--", alpha=0.5, label="d_safe")
        ax_margin.set_ylabel("Min Distance [m]")
        ax_margin.set_xlabel("Step")
        ax_margin.legend(fontsize=7)

        # Controls
        if frame < len(commands):
            c_hist = commands[:end]
            ax_control.plot([c.rudder for c in c_hist], "r-", lw=1, label="Rudder")
            ax_control.plot([c.propeller * 35 for c in c_hist], "g-", lw=1, label="Prop×35")
        ax_control.set_ylabel("Command")
        ax_control.legend(fontsize=7)
        # Fallback + CBF indicators
        if frame < len(commands):
            c = commands[frame]
            status = []
            if c.fallback_active:
                status.append("FALLBACK")
            if c.cbf_modified:
                status.append("CBF")
            ax_control.set_title(" | ".join(status) if status else "Nominal", fontsize=8)

        # Solver time
        ax_time.plot(solve_times[:end], "k-", lw=1)
        ax_time.axhline(y=0.1, color="r", ls="--", alpha=0.5, label="0.1s deadline")
        ax_time.set_ylabel("Solve Time [s]")
        ax_time.set_xlabel("Step")
        ax_time.legend(fontsize=7)

        fig.tight_layout()

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".mp4":
        try:
            ani.save(output_path, writer="ffmpeg", fps=fps, dpi=dpi)
            print(f"Video saved: {output_path}")
            return True
        except (RuntimeError, FileNotFoundError):
            print("ffmpeg not available — falling back to GIF.")

    # GIF fallback
    gif_path = output_path.replace(".mp4", ".gif")
    ani.save(gif_path, writer="pillow", fps=fps, dpi=dpi)
    print(f"GIF saved: {gif_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Render trajectory animation video.")
    parser.add_argument("--trajectory", required=True, help="Path to .npz trajectory file.")
    parser.add_argument("--output", default="results/videos/trajectory.mp4", help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second.")
    parser.add_argument("--dpi", type=int, default=100, help="Render DPI.")
    args = parser.parse_args()

    traj = load_trajectory(args.trajectory)
    success = render_video(traj, args.output, fps=args.fps, dpi=args.dpi)
    if not success:
        print("Rendering failed. Check dependencies: pip install matplotlib pillow")
        sys.exit(1)


if __name__ == "__main__":
    main()
