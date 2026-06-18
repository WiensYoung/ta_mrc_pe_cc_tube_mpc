"""Trajectory I/O — save, load, and validate simulation trajectories.

Provides stable, self-contained functions for persisting and restoring
state / command / target histories independently of the closed-loop runner.
Supports .npz (default, compressed numpy) format.

Schema (per file):
    own_state:     (T, 7)  float32  [x, y, psi, u, v, r, timestamp]
    own_command:   (T, 4)  float32  [rudder, propeller, cbf_modified, fallback_active]
    target_N_state:(T_N, 7) float32  one per target ship (key = "target_0_state", ...)

All functions are importable from ``ta_mrc_pe_cc_tube_mpc.simulation.trajectory_io``
and do not depend on the closed-loop runner internals.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

import numpy as np

# ── Column indices (own_state) ───────────────────────────────────────────────
OWN_STATE_COLS = ["x", "y", "psi", "u", "v", "r", "timestamp"]
OWN_CMD_COLS  = ["rudder", "propeller", "cbf_modified", "fallback_active"]

# ═══════════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════════

def save_trajectory(
    result,          # EpisodeResult
    output_dir: str,
    fmt: str = "npz",
) -> str:
    """Save episode trajectory to disk.

    Args:
        result: EpisodeResult with state_history, command_history,
                target_histories populated.
        output_dir: Directory to write the trajectory file.
        fmt: Format string — ``"npz"`` only for now.

    Returns:
        Absolute path to the saved file.

    Raises:
        ValueError: if *result* has empty histories or unsupported format.
    """
    if fmt != "npz":
        raise ValueError(f"Unsupported format '{fmt}'. Use 'npz'.")

    if not result.state_history:
        raise ValueError("EpisodeResult has empty state_history — nothing to save.")

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{result.episode_id or 'trajectory'}.npz"
    path = os.path.join(output_dir, fname)

    arrays = {}
    # Ownship state: (T, 7)
    arrays["own_state"] = np.array(
        [[s.x, s.y, s.psi, s.u, s.v, s.r, s.timestamp]
         for s in result.state_history],
        dtype=np.float32,
    )
    # Ownship command: (T, 4)
    if result.command_history:
        arrays["own_command"] = np.array(
            [[c.rudder, c.propeller, int(c.cbf_modified), int(c.fallback_active)]
             for c in result.command_history],
            dtype=np.float32,
        )
    # Target states
    for i, th in enumerate(result.target_histories):
        if th:
            arrays[f"target_{i}_state"] = np.array(
                [[s.x, s.y, s.psi, s.u, s.v, s.r, s.timestamp]
                 for s in th],
                dtype=np.float32,
            )

    np.savez_compressed(path, **arrays)
    return os.path.abspath(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Load
# ═══════════════════════════════════════════════════════════════════════════════

def load_trajectory(path: str) -> dict[str, np.ndarray]:
    """Load a saved trajectory file.

    Args:
        path: Path to a .npz file written by :func:`save_trajectory`.

    Returns:
        Dict mapping array names (``"own_state"``, ``"own_command"``,
        ``"target_0_state"``, ...) to float32 numpy arrays.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


# ═══════════════════════════════════════════════════════════════════════════════
# Validate
# ═══════════════════════════════════════════════════════════════════════════════

def validate_trajectory_schema(data: dict[str, np.ndarray]) -> list[str]:
    """Validate that loaded trajectory data conforms to the expected schema.

    Args:
        data: Dict from :func:`load_trajectory`.

    Returns:
        List of issue strings (empty → valid).
    """
    issues = []

    if "own_state" not in data:
        issues.append("missing required key 'own_state'")
    else:
        arr = data["own_state"]
        if arr.ndim != 2:
            issues.append(f"'own_state' has {arr.ndim} dims, expected 2")
        elif arr.shape[1] != 7:
            issues.append(f"'own_state' has {arr.shape[1]} cols, expected 7")

    if "own_command" in data:
        arr = data["own_command"]
        if arr.ndim != 2:
            issues.append(f"'own_command' has {arr.ndim} dims, expected 2")
        elif arr.shape[1] != 4:
            issues.append(f"'own_command' has {arr.shape[1]} cols, expected 4")

    for key in data:
        if key.startswith("target_") and key.endswith("_state"):
            arr = data[key]
            if arr.ndim != 2:
                issues.append(f"'{key}' has {arr.ndim} dims, expected 2")
            elif arr.shape[1] != 7:
                issues.append(f"'{key}' has {arr.shape[1]} cols, expected 7")

    # Check for NaN / Inf contamination
    for key, arr in data.items():
        if np.any(np.isnan(arr)):
            issues.append(f"'{key}' contains NaN values")
        if np.any(np.isinf(arr)):
            issues.append(f"'{key}' contains Inf values")

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Shape metadata helper
# ═══════════════════════════════════════════════════════════════════════════════

def trajectory_shape_meta(result) -> dict:
    """Return shape metadata dict suitable for inclusion in result metadata.

    Args:
        result: EpisodeResult with populated histories.

    Returns:
        Dict with ``state_history_shape``, ``command_history_shape``,
        ``target_history_keys``, ``episode_id``, ``scenario_id``,
        ``method_id``, ``seed``.
    """
    return {
        "state_history_shape": (
            len(result.state_history),
            len(OWN_STATE_COLS),
        ) if result.state_history else (0, 0),
        "command_history_shape": (
            len(result.command_history),
            len(OWN_CMD_COLS),
        ) if result.command_history else (0, 0),
        "target_history_keys": [
            f"target_{i}_state"
            for i, th in enumerate(result.target_histories) if th
        ],
        "episode_id": result.episode_id,
        "scenario_id": result.scenario_id,
        "method_id": result.method,
        "seed": result.seed,
    }
