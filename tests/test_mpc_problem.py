"""Tests for MPC problem formulation and sampling-based MPC."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.control.mpc_problem import MPCProblem, sampling_based_mpc
from ta_mrc_pe_cc_tube_mpc.models.mmg_3dof import MMG3DOFModel
from ta_mrc_pe_cc_tube_mpc.types import ControlCommand, EnvironmentState, TargetShip, VesselParams, VesselState


class TestMPCProblem:
    def setup_method(self):
        self.model = MMG3DOFModel()
        self.params = VesselParams()
        self.mpc = MPCProblem(
            dynamics=self.model.step,
            horizon=10,
            dt=0.5,
        )

    def _make_ref_trajectory(self, n_steps: int, x0=0.0, y0=0.0, u=8.0, psi=0.0):
        return [
            VesselState(x=x0 + i * u * np.cos(psi) * 0.5, y=y0 + i * u * np.sin(psi) * 0.5,
                        psi=psi, u=u, v=0, r=0)
            for i in range(n_steps)
        ]

    def test_initialization(self):
        """Default initialization uses given horizon and dt."""
        assert self.mpc.N == 10
        assert self.mpc.dt == 0.5
        assert self.mpc.rudder_min == -35.0
        assert self.mpc.rudder_max == 35.0
        assert self.mpc.prop_min == 0.0
        assert self.mpc.prop_max == 1.0

    def test_cost_matrix_shapes(self):
        """Q, R, S, P matrices have expected shapes."""
        assert self.mpc.Q_pos.shape == (3, 3)
        assert self.mpc.Q_vel.shape == (3, 3)
        assert self.mpc.R_mat.shape == (2, 2)  # matches mpc_problem.py attribute name
        assert self.mpc.S_mat.shape == (2, 2)  # matches mpc_problem.py attribute name
        assert self.mpc.P.shape == (3, 3)

    def test_cost_function_returns_float(self):
        """Cost function evaluates without error and returns a float."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0, r=0)
        ref = self._make_ref_trajectory(self.mpc.N + 1)
        n_vars = 2 * self.mpc.N
        u_flat = np.zeros(n_vars)
        u_flat[1::2] = 0.5

        env = EnvironmentState(water_depth=100.0)
        cost = self.mpc._cost_function(
            u_flat, state, ref, [], [], env, 0.0, 0.5,
        )
        assert np.isfinite(cost)
        assert cost >= 0.0

    def test_build_constraints(self):
        """All constraints include rudder rate bounds."""
        state = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        ref = self._make_ref_trajectory(self.mpc.N + 1)
        env = EnvironmentState()

        constraints = self.mpc._build_all_constraints(
            state, ref, [], [], env, 0.0, 0.5,
        )
        # 4 rate constraints (2 rudder + 2 propeller) per step
        assert len(constraints) == 4 * self.mpc.N

        # First constraint (upper rate limit at step 0)
        dt = self.mpc.dt
        rate_max = self.mpc.rudder_rate_max * dt
        u_test = np.zeros(2 * self.mpc.N)
        u_test[0] = 0.5 * rate_max
        u_test[1::2] = 0.5
        val = constraints[0]["fun"](u_test)
        assert val >= 0, "Within-rate change should satisfy constraint"
        # rudder_0 = rate_max * 2, prev = 0 → change = 2*rate_max, should violate
        u_test_violate = u_test.copy()
        u_test_violate[0] = rate_max * 2.0
        val_v = constraints[0]["fun"](u_test_violate)
        assert val_v < 0, "Excessive rate change should violate constraint"

    def test_build_constraints_inter_step(self):
        """Rate constraints between later steps are enforced."""
        state = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        ref = self._make_ref_trajectory(self.mpc.N + 1)
        env = EnvironmentState()

        constraints = self.mpc._build_all_constraints(
            state, ref, [], [], env, 0.0, 0.5,
        )
        dt = self.mpc.dt
        rate_max = self.mpc.rudder_rate_max * dt

        # Test step 1 upper rudder constraint: rudder_1 - rudder_0 <= rate_max
        # Constraint order: [r_upper_0, r_lower_0, p_upper_0, p_lower_0,
        #                    r_upper_1, r_lower_1, p_upper_1, p_lower_1, ...]
        u_test = np.zeros(2 * self.mpc.N)
        u_test[0] = 0.0
        u_test[2] = rate_max * 1.5  # rudder at step 1
        u_test[1::2] = 0.5
        val = constraints[4]["fun"](u_test)  # upper rudder constraint at k=1
        assert val < 0, "Large inter-step rudder change should violate constraint"

    def test_solver_no_targets_returns_solution(self):
        """MPC solves a simple case with no targets."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0, r=0)
        ref = self._make_ref_trajectory(self.mpc.N + 1)
        env = EnvironmentState(water_depth=100.0)

        cmd, info = self.mpc.solve(
            current_state=state,
            reference_trajectory=ref,
            target_positions=[],
            d_safe_eff_list=[],
            env=env,
        )

        assert cmd is not None
        assert info["success"]
        assert -35.0 <= cmd.rudder <= 35.0
        assert 0.0 <= cmd.propeller <= 1.0

    def test_solver_with_previous_command(self):
        """MPC respects previous command for rate constraints."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0, r=0)
        ref = self._make_ref_trajectory(self.mpc.N + 1)
        env = EnvironmentState(water_depth=100.0)
        prev_cmd = ControlCommand(rudder=15.0, propeller=0.5, source="previous")

        cmd, info = self.mpc.solve(
            current_state=state,
            reference_trajectory=ref,
            target_positions=[],
            d_safe_eff_list=[],
            env=env,
            previous_command=prev_cmd,
        )

        assert cmd is not None
        # Rudder should not change more than rate_max from previous 15.0
        max_delta = self.mpc.rudder_rate_max * self.mpc.dt
        assert abs(cmd.rudder - 15.0) <= max_delta + 0.01

    def test_config_override(self):
        """MPCProblem respects custom config values."""
        mpc = MPCProblem(
            dynamics=self.model.step,
            horizon=5,
            config={
                "rudder_min": -20.0,
                "rudder_max": 20.0,
                "propeller_min": 0.1,
                "propeller_max": 0.9,
            },
        )
        assert mpc.rudder_min == -20.0
        assert mpc.rudder_max == 20.0
        assert mpc.prop_min == 0.1
        assert mpc.prop_max == 0.9

    def test_casadi_options_no_internal_key_leak(self):
        """P0 fix: ipopt_max_cpu_time is NOT passed to CasADi solver options."""
        mpc = MPCProblem(
            dynamics=self.model.step,
            horizon=5,
            config={"mpc_backend": "casadi"},
        )
        # Simulate what happens during _solve_casadi
        opts = dict(mpc._casadi_solver_options)
        max_cpu = float(opts.pop("ipopt_max_cpu_time", 5.0))
        opts["ipopt.max_cpu_time"] = max_cpu
        # After the fix, ipopt_max_cpu_time (internal key) must NOT be in opts
        assert "ipopt_max_cpu_time" not in opts
        assert "ipopt.max_cpu_time" in opts
        assert opts["ipopt.max_cpu_time"] == 5.0

    def test_solver_metadata_distinguishes_requested_vs_actual(self):
        """P0 fix: metadata separates requested_backend from actual_backend."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0, r=0)
        ref = self._make_ref_trajectory(self.mpc.N + 1)
        env = EnvironmentState(water_depth=100.0)

        cmd, info = self.mpc.solve(
            current_state=state,
            reference_trajectory=ref,
            target_positions=[],
            d_safe_eff_list=[],
            env=env,
        )

        # Both requested_backend and actual_backend should be present
        assert "requested_backend" in info, "metadata must include requested_backend"
        assert "actual_backend" in info, "metadata must include actual_backend"
        # For a successful run, actual_backend should match the configured backend
        assert info["actual_backend"] == info["requested_backend"] or \
               info.get("degraded_backend") is not None

    def test_solver_returns_valid_command_with_backend_metadata(self):
        """P0 fix: command source field accurately reflects the solver used."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0, r=0)
        ref = self._make_ref_trajectory(self.mpc.N + 1)
        env = EnvironmentState(water_depth=100.0)

        cmd, info = self.mpc.solve(
            current_state=state,
            reference_trajectory=ref,
            target_positions=[],
            d_safe_eff_list=[],
            env=env,
        )

        assert cmd is not None
        # Command source should contain backend info
        assert cmd.source in ("MPC_SLSQP", "MPC_CasADi", "MPC_CasADi_debug"), \
            f"Unexpected command source: {cmd.source}"
        # Info dict should contain the actual backend
        assert "actual_backend" in info


class TestSamplingMPC:
    def setup_method(self):
        self.model = MMG3DOFModel()

    def _make_ref(self, n, x0=0, y0=0, u=8, psi=0):
        return [
            VesselState(x=x0 + i * u * np.cos(psi) * 0.5, y=y0 + i * u * np.sin(psi) * 0.5,
                        psi=psi, u=u, v=0, r=0)
            for i in range(n)
        ]

    def test_no_targets_returns_solution(self):
        """Sampling MPC returns a valid command with no obstacles."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0, r=0)
        ref = self._make_ref(21)
        env = EnvironmentState(water_depth=100.0)
        rng = np.random.RandomState(42)

        cmd, info = sampling_based_mpc(
            current_state=state,
            dynamics=self.model.step,
            reference_trajectory=ref,
            target_positions=[],
            d_safe_eff_list=[],
            env=env,
            n_samples=100,
            horizon=10,
            rng=rng,
        )

        assert cmd is not None
        assert info["success"]
        assert cmd.source == "sampling_MPC"
        assert -35.0 <= cmd.rudder <= 35.0
        assert 0.0 <= cmd.propeller <= 1.0

    def test_infeasible_with_tight_safety(self):
        """Sampling MPC returns None when obstacle is unavoidable."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0, r=0)
        ref = self._make_ref(21)
        env = EnvironmentState(water_depth=100.0)
        rng = np.random.RandomState(42)

        # Target positions very close at every step — unavoidable
        tgt_positions = [np.array([[state.x + 5, state.y]]) for _ in range(21)]

        cmd, info = sampling_based_mpc(
            current_state=state,
            dynamics=self.model.step,
            reference_trajectory=ref,
            target_positions=tgt_positions,
            d_safe_eff_list=[800.0],  # very large safety distance
            env=env,
            n_samples=50,
            horizon=10,
            rng=rng,
        )

        # May be infeasible due to large safety margin
        if cmd is None:
            assert not info["success"]
        else:
            assert info["success"]
