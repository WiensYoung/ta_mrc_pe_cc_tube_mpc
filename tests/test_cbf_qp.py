"""Tests for CBF-QP safety filter."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.control.cbf_qp import CBFQPSafetyFilter
from ta_mrc_pe_cc_tube_mpc.types import (
    ControlCommand,
    EnvironmentState,
    TargetShip,
    VesselParams,
    VesselState,
)


class TestCBFQPSafetyFilter:
    """Test CBF-QP safety filter behavior."""

    def setup_method(self):
        self.cbf = CBFQPSafetyFilter()
        self.own_params = VesselParams()

    def test_safe_state_no_modification(self):
        """When all barriers satisfied, command passes through unchanged."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        u_mpc = ControlCommand(rudder=5.0, propeller=0.5, source="MPC")
        env = EnvironmentState(water_depth=50.0, channel_width=1000.0)
        targets = []

        filtered, info = self.cbf.filter(
            u_mpc, own, self.own_params, targets, env,
            d_safe_eff_list=[],
        )

        assert not filtered.cbf_modified
        assert info["cbf_active"] == False

    def test_close_target_triggers_cbf(self):
        """Close target violating safety barrier triggers CBF."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=20, y=0, psi=np.pi, u=3, v=0, r=0),
            length=100, beam=20, draught=5,
        )
        u_mpc = ControlCommand(rudder=10.0, propeller=0.8, source="MPC")
        env = EnvironmentState(water_depth=50.0)

        filtered, info = self.cbf.filter(
            u_mpc, own, self.own_params, [tgt], env,
            d_safe_eff_list=[500.0],  # safe distance >> current distance
        )

        assert filtered.cbf_modified
        assert info["cbf_active"] == True
        assert "ship_T001" in info["active_constraints"]

    def test_cbf_modifies_command_when_unsafe(self):
        """CBF modifies command when barriers are violated and recovery is feasible.

        Uses MMG finite-difference gradients for accurate barrier Jacobians.
        The scenario places a target off the starboard bow: close enough to
        violate the safety barrier, but with geometry that gives the CBF
        gradient-based control authority to steer away.
        """
        from ta_mrc_pe_cc_tube_mpc.models.mmg_3dof import MMG3DOFModel

        mmg = MMG3DOFModel({
            "length": 180.0, "beam": 30.0, "draught": 10.0,
            "mass": 5e7, "Iz": 1e11, "max_rudder": 35.0,
            "max_rudder_rate": 5.0, "max_yaw_rate": 3.0,
            "min_turning_radius": 200.0,
        })
        self.cbf.set_mmg_model(mmg)

        # Own at origin heading East; target ahead and slightly to starboard,
        # heading roughly toward own — barrier is violated (d=250 << d_safe=500).
        own = VesselState(x=0, y=0, psi=0, u=6, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=250, y=40, psi=np.deg2rad(200), u=4, v=0, r=0),
            length=120, beam=18, draught=7,
        )
        u_mpc = ControlCommand(rudder=0.0, propeller=0.8, source="MPC")
        env = EnvironmentState(water_depth=60.0)

        filtered, info = self.cbf.filter(
            u_mpc, own, self.own_params, [tgt], env,
            d_safe_eff_list=[500.0],
        )

        assert filtered.cbf_modified, "CBF should mark command as modified"
        assert info["cbf_active"] is True
        assert -35.0 <= filtered.rudder <= 35.0
        assert 0.0 <= filtered.propeller <= 1.0
        # CBF must attempt a correction; it may adjust rudder, propeller, or both.
        # Check that the filter ran and produced a valid command.
        assert "ship_T001" in info["active_constraints"]

    def test_cbf_respects_control_limits(self):
        """Filtered command respects actuator limits AND enforces barriers."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=10, y=0, psi=np.pi, u=3, v=0, r=0),
            length=100, beam=20, draught=5,
        )
        u_mpc = ControlCommand(rudder=40.0, propeller=1.5, source="MPC")  # beyond limits
        env = EnvironmentState(water_depth=50.0)

        filtered, info = self.cbf.filter(
            u_mpc, own, self.own_params, [tgt], env,
            d_safe_eff_list=[200.0],
        )

        # Must respect actuator limits
        assert -35.0 <= filtered.rudder <= 35.0
        assert 0.0 <= filtered.propeller <= 1.0
        # CBF must be active (both limits and barriers violated)
        assert filtered.cbf_modified
        assert info["cbf_active"] == True

    def test_bank_clearance_barrier(self):
        """Bank clearance violation triggers CBF."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        u_mpc = ControlCommand(rudder=0.0, propeller=0.5, source="MPC")
        env = EnvironmentState(
            water_depth=50.0,
            bank_distance_left=10.0,  # very close to bank
            bank_distance_right=500.0,
        )

        filtered, info = self.cbf.filter(
            u_mpc, own, self.own_params, [], env,
            d_safe_eff_list=[],
        )

        assert filtered.cbf_modified
        # BUG #2 fix: each side produces separate tags ("bank_clearance_left"/"bank_clearance_right")
        assert any("bank_clearance" in c for c in info["active_constraints"])
