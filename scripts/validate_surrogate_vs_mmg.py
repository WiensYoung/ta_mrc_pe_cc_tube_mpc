#!/usr/bin/env python
"""Validate CasADi surrogate rollout against full MMG simulation.

Generates random initial states and control sequences, rolls out both
the CasADi surrogate and the full MMG model, and reports error statistics.

Output:
    results/analysis/surrogate_vs_mmg.csv
    results/analysis/surrogate_vs_mmg_summary.json

Usage:
    python scripts/validate_surrogate_vs_mmg.py --n-samples 5 --horizon 5
    python scripts/validate_surrogate_vs_mmg.py --n-samples 100 --horizon 20 --output results/analysis
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ta_mrc_pe_cc_tube_mpc.models.mmg_3dof import MMG3DOFModel
from ta_mrc_pe_cc_tube_mpc.types import ControlCommand, EnvironmentState, VesselState


def random_vessel_state(rng: np.random.RandomState) -> VesselState:
    """Generate a random physically plausible vessel state."""
    return VesselState(
        x=rng.uniform(-500, 500),
        y=rng.uniform(-500, 500),
        psi=rng.uniform(0, 2 * np.pi),
        u=rng.uniform(2, 12),  # surge speed [m/s]
        v=rng.uniform(-1, 1),  # sway speed [m/s]
        r=rng.uniform(-0.05, 0.05),  # yaw rate [rad/s]
    )


def random_control(rng: np.random.RandomState) -> ControlCommand:
    """Generate a random feasible control command."""
    return ControlCommand(
        rudder=rng.uniform(-35, 35),  # [deg]
        propeller=rng.uniform(0.2, 1.0),
    )


def full_mmg_rollout(
    mmg: MMG3DOFModel,
    state: VesselState,
    controls: list[ControlCommand],
    dt: float,
) -> list[VesselState]:
    """Roll out full MMG model with control sequence."""
    states = [state]
    env = EnvironmentState()
    for cmd in controls:
        next_state = mmg.step(states[-1], cmd, env, dt)
        states.append(next_state)
    return states


def surrogate_rollout(
    state: VesselState,
    controls: list[ControlCommand],
    dt: float,
) -> list[VesselState]:
    """Roll out the CasADi surrogate dynamics (simplified 3-DOF kinematics).

    This matches the surrogate used in MPCProblem._solve_casadi (mpc_problem.py),
    NOT the full MMG model. Key simplifications:
      - No rudder rate limiter (instant rudder response)
      - No current/wind disturbance
      - Constant added mass (m=5e7, Iz=1.2e11)
      - Simplified linear+quadratic damping (no cross-flow drag)
      - Simplified propeller thrust (linear with command)
    """
    import numpy as _np

    m_surge = 5e7
    Iz_surge = 1.2e11
    vessel_length = 180.0
    rho_water = 1025.0
    # Dimensionalised coefficients matching mpc_problem.py defaults
    Xuu = -2e-4 * 0.5 * rho_water * vessel_length ** 2
    Yv = -3e-3 * 0.5 * rho_water * vessel_length ** 3
    Nr = -5e-4 * 0.5 * rho_water * vessel_length ** 4
    Nd = -3e-4 * 0.5 * rho_water * vessel_length ** 3
    Kt = 0.15
    D_prop = 6.0

    states = [state]
    for cmd in controls:
        prev = states[-1]
        dk = _np.radians(cmd.rudder)   # deg → rad (no rate limiter)
        pk = _np.clip(cmd.propeller, 0.0, 1.0)

        # Propeller thrust (simplified)
        n_rps = pk * 3.0
        thrust = Kt * rho_water * (n_rps ** 2) * (D_prop ** 4) if n_rps > 0 else 0.0

        u_dot = (thrust + Xuu * prev.u * abs(prev.u)) / m_surge
        v_dot = (Yv * prev.v) / m_surge
        r_dot = (Nr * prev.r + Nd * dk * prev.u * abs(prev.u)) / Iz_surge

        psi_next = prev.psi + dt * prev.r
        u_next = prev.u + dt * u_dot
        v_next = prev.v + dt * v_dot
        r_next = prev.r + dt * r_dot
        x_next = prev.x + dt * (prev.u * _np.cos(prev.psi) - prev.v * _np.sin(prev.psi))
        y_next = prev.y + dt * (prev.u * _np.sin(prev.psi) + prev.v * _np.cos(prev.psi))

        states.append(VesselState(
            x=x_next, y=y_next, psi=psi_next,
            u=u_next, v=v_next, r=r_next,
            timestamp=prev.timestamp + dt,
        ))
    return states


def compute_error_stats(
    full_states: list[VesselState],
    surr_states: list[VesselState],
) -> dict:
    """Compute error statistics between two state trajectories."""
    pos_errors = []
    heading_errors = []
    speed_errors = []

    for fs, ss in zip(full_states, surr_states):
        pos_err = np.sqrt((fs.x - ss.x) ** 2 + (fs.y - ss.y) ** 2)
        heading_err = abs((fs.psi - ss.psi + np.pi) % (2 * np.pi) - np.pi)
        speed_err = abs(fs.speed() - ss.speed())
        pos_errors.append(pos_err)
        heading_errors.append(heading_err)
        speed_errors.append(speed_err)

    pos_arr = np.array(pos_errors)
    heading_arr = np.array(heading_errors)
    speed_arr = np.array(speed_errors)

    return {
        "mean_position_error": float(np.mean(pos_arr)),
        "p95_position_error": float(np.percentile(pos_arr, 95)),
        "max_position_error": float(np.max(pos_arr)),
        "mean_heading_error_deg": float(np.rad2deg(np.mean(heading_arr))),
        "p95_heading_error_deg": float(np.rad2deg(np.percentile(heading_arr, 95))),
        "max_heading_error_deg": float(np.rad2deg(np.max(heading_arr))),
        "mean_speed_error": float(np.mean(speed_arr)),
        "p95_speed_error": float(np.percentile(speed_arr, 95)),
        "max_speed_error": float(np.max(speed_arr)),
        "n_samples": len(pos_errors),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate CasADi surrogate vs full MMG model."
    )
    parser.add_argument("--n-samples", type=int, default=100,
                       help="Number of random trajectories to sample (>=100 for statistical validity).")
    parser.add_argument("--horizon", type=int, default=20,
                       help="Control sequence length (match MPC horizon, default 20).")
    parser.add_argument("--dt", type=float, default=0.5,
                       help="Time step [s].")
    parser.add_argument("--n-repeats", type=int, default=3,
                       help="Repeats per sample with different random seeds.")
    parser.add_argument("--output", default="results/analysis",
                       help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, args.output)
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    # Initialize MMG model (full MMG)
    vessel_params = {
        "length": 180.0,
        "beam": 30.0,
        "draught": 10.5,
        "mass": 5e7,
        "Iz": 1.2e11,
        "max_rudder": 35.0,
        "max_rudder_rate": 5.0,
        "max_yaw_rate": 3.0,
        "min_turning_radius": 3.0,
    }
    mmg = MMG3DOFModel(vessel_params)

    all_errors = []
    for i in range(args.n_samples):
        state = random_vessel_state(rng)
        controls = [random_control(rng) for _ in range(args.horizon)]

        # Full MMG rollout
        full_states = full_mmg_rollout(mmg, state, controls, args.dt)

        # CasADi surrogate: uses simplified kinematics without full MMG
        # hydrodynamics, matching what the MPC's CasADi backend predicts.
        # Key differences from full MMG: no rate limiter, simplified rudder
        # forces, no current/wind/shallow-water effects, constant added mass.
        mmg.reset_rudder_state(0.0)
        surr_states = surrogate_rollout(state, controls, args.dt)

        errors = compute_error_stats(full_states, surr_states)
        errors["sample_id"] = i
        all_errors.append(errors)

    # Save CSV
    import pandas as pd
    csv_path = os.path.join(output_dir, "surrogate_vs_mmg.csv")
    pd.DataFrame(all_errors).to_csv(csv_path, index=False)
    print(f"CSV → {csv_path}")

    # Save summary JSON
    summary = {
        "n_samples": args.n_samples,
        "horizon": args.horizon,
        "dt": args.dt,
        "aggregate": {
            "mean_position_error": float(np.mean([e["mean_position_error"] for e in all_errors])),
            "p95_position_error": float(np.percentile([e["p95_position_error"] for e in all_errors], 95)),
            "max_position_error": float(np.max([e["max_position_error"] for e in all_errors])),
            "mean_heading_error_deg": float(np.mean([e["mean_heading_error_deg"] for e in all_errors])),
            "p95_heading_error_deg": float(np.percentile([e["p95_heading_error_deg"] for e in all_errors], 95)),
            "mean_speed_error": float(np.mean([e["mean_speed_error"] for e in all_errors])),
        },
    }
    json_path = os.path.join(output_dir, "surrogate_vs_mmg_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON → {json_path}")

    print("\nValidation complete.")
    print(f"  Mean position error: {summary['aggregate']['mean_position_error']:.3f} m")
    print(f"  P95 position error:  {summary['aggregate']['p95_position_error']:.3f} m")
    print(f"  Mean heading error:  {summary['aggregate']['mean_heading_error_deg']:.2f} deg")


if __name__ == "__main__":
    main()
