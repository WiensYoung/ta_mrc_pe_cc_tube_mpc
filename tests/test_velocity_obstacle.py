"""Tests for Velocity Obstacle (B1 baseline) controller."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.control.velocity_obstacle import VelocityObstacleController
from ta_mrc_pe_cc_tube_mpc.types import ControlCommand, EnvironmentState, TargetShip, VesselParams, VesselState


class TestVelocityObstacleController:
    def setup_method(self):
        self.params = VesselParams()
        self.vo = VelocityObstacleController(self.params)

    def test_no_targets_returns_default(self):
        """Without targets, returns straight-ahead cruise command."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        cmd = self.vo.compute_control(own, self.params, [], EnvironmentState())
        assert cmd.source == "VO"
        assert abs(cmd.rudder) < 0.01
        assert cmd.propeller == pytest.approx(0.5)
        assert cmd.metadata["vo_state"] == "no_targets"

    def test_rudder_rate_limiting(self):
        """Rudder change is bounded by max_rudder_rate."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=200, y=0, psi=np.pi, u=5, v=0, r=0),
        )
        # First call sets previous_rudder
        self.vo.compute_control(own, self.params, [tgt], EnvironmentState())
        prev = self.vo.previous_rudder

        # Second call: rudder change should be <= max_rudder_rate
        cmd2 = self.vo.compute_control(own, self.params, [tgt], EnvironmentState())
        delta = abs(cmd2.rudder - prev)
        assert delta <= self.params.max_rudder_rate + 0.01

    def test_reset(self):
        """Reset clears internal state."""
        self.vo.previous_rudder = 20.0
        self.vo.reset()
        assert self.vo.previous_rudder == 0.0
        assert self.vo.previous_propeller == 0.5

    def test_unsupported_features_derived(self):
        """unsupported_features list is computed at init."""
        assert "chance_constraint" in self.vo.unsupported_features
        assert "tube_mpc" in self.vo.unsupported_features
        assert "colregs_rule_selection" not in self.vo.unsupported_features
        assert "velocity_obstacle" not in self.vo.unsupported_features

    def test_head_on_encounter_turns_starboard(self):
        """Head-on encounter triggers starboard rudder."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=300, y=0, psi=np.pi, u=8, v=0, r=0),
        )
        cmd = self.vo.compute_control(own, self.params, [tgt], EnvironmentState())
        # Head-on → starboard turn (positive rudder)
        assert cmd.rudder > 0, f"Expected starboard turn, got rudder={cmd.rudder}"
        assert cmd.propeller <= 0.5  # speed not increased

    def test_close_range_reduces_speed(self):
        """At close range (< 0.5 * vo_radius), propeller is reduced."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=50, y=0, psi=np.pi, u=8, v=0, r=0),
        )
        cmd = self.vo.compute_control(own, self.params, [tgt], EnvironmentState())
        assert cmd.propeller < 0.5

    def test_config_respected(self):
        """Constructor respects config overrides."""
        vo = VelocityObstacleController(self.params, {"vo_safety_radius": 800.0})
        assert vo.safety_radius == 800.0
        assert vo.time_horizon == 600.0  # default

    def test_command_bounds(self):
        """Output rudder and propeller stay within actuator limits."""
        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=100, y=0, psi=np.pi, u=8, v=0, r=0),
        )
        cmd = self.vo.compute_control(own, self.params, [tgt], EnvironmentState())
        assert -self.params.max_rudder <= cmd.rudder <= self.params.max_rudder
        assert 0.0 <= cmd.propeller <= 1.0
