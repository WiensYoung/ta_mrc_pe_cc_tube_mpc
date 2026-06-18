"""Tests for the main TA-MRC-PE-CC-Tube-MPC controller."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.control.controller import TAMRCPECCTubeMPCController
from ta_mrc_pe_cc_tube_mpc.types import ControlCommand, EnvironmentState, TargetShip, VesselParams, VesselState


class TestTAMRCPECCTubeMPCController:
    def setup_method(self):
        self.params = VesselParams()
        # Disable expensive features for unit tests
        self.cfg = {
            "controller": {
                "enable_multi_rule": False,
                "enable_target_size": False,
                "enable_speed_sensitive_domain": False,
                "enable_vessel_type_weight": False,
                "enable_behavior_uncertainty": False,
                "enable_shallow_water": False,
                "enable_wind_current": False,
                "enable_bank_effect": False,
                "enable_ship_interaction": False,
                "enable_chance_constraint": False,
                "enable_tube_mpc": False,
                "enable_cbf_filter": False,
                "enable_fallback": False,
            },
            "mpc": {"horizon": 5, "dt": 0.5},
        }
        self.controller = TAMRCPECCTubeMPCController(self.params, self.cfg)

    def test_initialization(self):
        """Controller initializes all sub-components."""
        ctrl = self.controller
        assert ctrl.mmg is not None
        assert ctrl.rule_engine is not None
        assert ctrl.ship_domain is not None
        assert ctrl.chance_evaluator is not None
        assert ctrl.tube_mpc is not None
        assert ctrl.cbf_filter is not None
        assert ctrl.fallback is not None

    def test_feature_flags_off(self):
        """When all features are disabled, flags are False."""
        assert not self.controller.enable_multi_rule
        assert not self.controller.enable_tube_mpc
        assert not self.controller.enable_cbf_filter
        assert not self.controller.enable_fallback

    def test_reset(self):
        """Reset clears controller state."""
        self.controller.previous_command = ControlCommand(rudder=10, propeller=0.8)
        self.controller.reference_trajectory = [VesselState()]
        self.controller.reset()
        assert self.controller.previous_command is None
        assert self.controller.reference_trajectory == []

    def test_generate_default_reference(self):
        """Default reference is a straight-line constant-speed path."""
        state = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        ref = self.controller._generate_default_reference(state, 10)
        assert len(ref) == 10
        # All waypoints should have same heading and speed
        for r in ref:
            assert r.psi == pytest.approx(state.psi)
            assert r.u == pytest.approx(state.u)
        # First waypoint is at origin
        assert ref[0].x == pytest.approx(state.x)
        assert ref[0].y == pytest.approx(state.y)
        # Last waypoint should be forward
        assert ref[-1].x > state.x

    def test_generate_default_reference_diagonal(self):
        """Reference path respects non-zero heading."""
        state = VesselState(x=0, y=0, psi=np.pi / 4, u=10, v=0, r=0)
        ref = self.controller._generate_default_reference(state, 5)
        # Diagonal heading: x and y should both increase
        assert ref[-1].x > 0
        assert ref[-1].y > 0

    def test_compute_control_no_targets(self):
        """Basic control computation with no targets returns valid command."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        cmd = self.controller.compute_control(
            ownstate=own,
            ownship_params=self.params,
            target_ships=[],
            environment=EnvironmentState(),
        )
        assert cmd is not None
        assert -self.params.max_rudder <= cmd.rudder <= self.params.max_rudder
        assert 0.0 <= cmd.propeller <= 1.0
        assert "runtime" in cmd.metadata

    def test_compute_control_with_targets(self):
        """Control computation with a target ship succeeds."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=500, y=0, psi=np.pi, u=5, v=0, r=0),
        )
        cmd = self.controller.compute_control(
            ownstate=own,
            ownship_params=self.params,
            target_ships=[tgt],
            environment=EnvironmentState(),
        )
        assert cmd is not None
        assert "rule_assessment" in cmd.metadata

    def test_mpc_command_produces_valid_output(self):
        """When MPC returns any solution, controller provides valid control command."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)

        # Force a config that will likely make MPC infeasible
        cfg = {
            "controller": {
                "enable_multi_rule": False,
                "enable_tube_mpc": True,
                "enable_cbf_filter": False,
                "enable_fallback": True,
                "enable_shallow_water": False,
                "enable_wind_current": False,
                "enable_bank_effect": False,
                "enable_ship_interaction": False,
                "enable_chance_constraint": False,
                "enable_target_size": False,
                "enable_speed_sensitive_domain": False,
                "enable_vessel_type_weight": False,
                "enable_behavior_uncertainty": False,
            },
            "mpc": {"horizon": 5, "dt": 0.5},
        }
        ctrl = TAMRCPECCTubeMPCController(self.params, cfg)
        cmd = ctrl.compute_control(
            ownstate=own,
            ownship_params=self.params,
            target_ships=[],
            environment=EnvironmentState(),
        )
        assert cmd is not None
        assert -self.params.max_rudder <= cmd.rudder <= self.params.max_rudder
        assert 0.0 <= cmd.propeller <= 1.0
