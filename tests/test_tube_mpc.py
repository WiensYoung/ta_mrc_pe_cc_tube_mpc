"""Tests for Tube-MPC wrapper."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.control.tube_mpc import TubeMPC
from ta_mrc_pe_cc_tube_mpc.models.mmg_3dof import MMG3DOFModel
from ta_mrc_pe_cc_tube_mpc.types import ControlCommand, EnvironmentState, TargetShip, VesselParams, VesselState


class TestTubeMPC:
    def setup_method(self):
        self.params = VesselParams()
        self.dynamics = MMG3DOFModel().step
        self.tube_mpc = TubeMPC(
            dynamics=self.dynamics,
            vessel_params=self.params,
            horizon=10,
            dt=0.5,
        )

    def test_initialization(self):
        """TubeMPC initializes with valid internal state."""
        assert self.tube_mpc.N == 10
        assert self.tube_mpc.dt == 0.5
        assert self.tube_mpc.rho_0 == 5.0
        assert self.tube_mpc.mpc is not None

    def test_predict_targets_single(self):
        """Single target is predicted at constant velocity."""
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=0, y=0, psi=0, u=5, v=0, r=0),
        )
        preds = self.tube_mpc._predict_targets([tgt])
        assert len(preds) == self.tube_mpc.N + 1
        # At time 0, positions match initial
        np.testing.assert_allclose(preds[0][0], [0, 0])
        # After N steps, x has advanced
        t_last = self.tube_mpc.N * self.tube_mpc.dt
        expected_x = 5.0 * t_last
        assert preds[-1][0, 0] == pytest.approx(expected_x, rel=1e-10)
        assert preds[-1][0, 1] == pytest.approx(0.0)

    def test_predict_targets_multiple(self):
        """Multiple targets are predicted independently."""
        targets = [
            TargetShip(
                mmsi="T001",
                state=VesselState(x=0, y=0, psi=0, u=5, v=0, r=0),
            ),
            TargetShip(
                mmsi="T002",
                state=VesselState(x=100, y=50, psi=np.pi / 2, u=3, v=0, r=0),
            ),
        ]
        preds = self.tube_mpc._predict_targets(targets)
        assert len(preds) == self.tube_mpc.N + 1
        # Each step has shape (2, 2)
        for step_pos in preds:
            assert step_pos.shape == (2, 2)

    def test_predict_targets_heading(self):
        """Target moving with non-zero heading advances correctly."""
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=0, y=0, psi=np.pi / 2, u=5, v=0, r=0),
        )
        preds = self.tube_mpc._predict_targets([tgt])
        # At step N, should move North (+y direction)
        t_last = self.tube_mpc.N * self.tube_mpc.dt
        assert preds[-1][0, 1] == pytest.approx(5.0 * t_last, rel=1e-10)

    def test_compute_control_no_targets(self):
        """With no targets, TubeMPC returns a valid command."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        own_Sigma = np.eye(2)
        env = EnvironmentState()
        ref = [VesselState(x=i * 4, y=0, psi=0, u=8, v=0, r=0) for i in range(self.tube_mpc.N + 1)]

        cmd, meta = self.tube_mpc.compute_control(
            own_state=own,
            own_Sigma=own_Sigma,
            targets=[],
            env=env,
            reference_trajectory=ref,
            d_safe_eff_list=[],
        )

        assert cmd is not None, "TubeMPC should return a command with no targets"
        assert -self.params.max_rudder <= cmd.rudder <= self.params.max_rudder
        assert 0.0 <= cmd.propeller <= 1.0
        assert "tube_radius" in meta
        assert "tube_components" in meta

    def test_config_not_mutated(self):
        """Verifies P0.4 fix: original config dict is not mutated."""
        original_cfg = {
            "mpc": {"horizon": 15, "custom_key": "value"},
            "tube": {"rho_0": 10.0},
        }
        cfg_copy = dict(original_cfg)
        cfg_copy["mpc"] = dict(original_cfg["mpc"])

        TubeMPC(
            dynamics=self.dynamics,
            vessel_params=self.params,
            config=original_cfg,
        )

        # Original config should be unchanged
        assert "vessel_length" not in original_cfg["mpc"]
        assert original_cfg["mpc"]["custom_key"] == "value"
        assert original_cfg["mpc"]["horizon"] == 15

    def test_sampling_mode(self):
        """TubeMPC with use_sampling flag falls back to sampling MPC."""
        tube = TubeMPC(
            dynamics=self.dynamics,
            vessel_params=self.params,
            horizon=5,
            config={"use_sampling": True, "sampling_mpc": {"n_samples": 50}},
        )
        assert tube.use_sampling is True

        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        env = EnvironmentState()
        ref = [VesselState(x=i * 4, y=0, psi=0, u=8, v=0, r=0) for i in range(6)]

        cmd, meta = tube.compute_control(
            own_state=own,
            own_Sigma=np.eye(2),
            targets=[],
            env=env,
            reference_trajectory=ref,
            d_safe_eff_list=[],
        )

        assert cmd is not None
        assert cmd.source == "sampling_MPC"
