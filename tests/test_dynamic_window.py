"""Tests for Dynamic Window Approach (B2 baseline) controller."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.control.dynamic_window import DynamicWindowController
from ta_mrc_pe_cc_tube_mpc.types import ControlCommand, EnvironmentState, TargetShip, VesselParams, VesselState


class TestDynamicWindowController:
    def setup_method(self):
        self.params = VesselParams()
        self.dwa = DynamicWindowController(self.params)

    def test_no_targets_returns_valid_command(self):
        """Without targets, DWA returns a valid control command."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        cmd = self.dwa.compute_control(own, self.params, [], EnvironmentState())
        assert cmd.source == "DWA"
        assert -self.params.max_rudder <= cmd.rudder <= self.params.max_rudder
        assert 0.0 <= cmd.propeller <= 1.0

    def test_rudder_rate_limiting(self):
        """Rudder change respects max_rudder_rate constraint."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=300, y=0, psi=np.pi, u=5, v=0, r=0),
        )
        cmd1 = self.dwa.compute_control(own, self.params, [tgt], EnvironmentState())
        prev = self.dwa.previous_rudder
        cmd2 = self.dwa.compute_control(own, self.params, [tgt], EnvironmentState())
        delta = abs(cmd2.rudder - prev)
        assert delta <= self.params.max_rudder_rate + 0.01

    def test_reset(self):
        """Reset clears internal state to defaults."""
        self.dwa.previous_rudder = 15.0
        self.dwa.previous_propeller = 0.8
        self.dwa.reset()
        assert self.dwa.previous_rudder == 0.0
        assert self.dwa.previous_propeller == 0.5

    def test_unsupported_features_derived(self):
        """unsupported_features list is computed, not hardcoded."""
        assert "chance_constraint" in self.dwa.unsupported_features
        assert "tube_mpc" in self.dwa.unsupported_features
        assert "dynamic_feasibility" not in self.dwa.unsupported_features
        assert "obstacle_avoidance" not in self.dwa.unsupported_features

    def test_config_overrides(self):
        """Constructor picks up config values."""
        dwa = DynamicWindowController(
            self.params,
            {"dwa_safety_radius": 800.0, "dwa_n_rudder": 15, "dwa_predict_time": 30.0},
        )
        assert dwa.safety_radius == 800.0
        assert dwa.n_rudder == 15
        assert dwa.predict_time == 30.0

    def test_command_with_target(self):
        """With a target, DWA produces a valid command with metadata."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=300, y=50, psi=np.pi, u=5, v=0, r=0),
        )
        cmd = self.dwa.compute_control(own, self.params, [tgt], EnvironmentState())
        assert cmd.source == "DWA"
        assert "best_cost" in cmd.metadata
        assert "encounter_type" in cmd.metadata
        assert "unsupported_features" in cmd.metadata

    def test_command_bounds(self):
        """Output rudder and propeller respect actuator limits."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=50, y=0, psi=np.pi, u=8, v=0, r=0),
        )
        cmd = self.dwa.compute_control(own, self.params, [tgt], EnvironmentState())
        assert -self.params.max_rudder <= cmd.rudder <= self.params.max_rudder
        assert 0.0 <= cmd.propeller <= 1.0
