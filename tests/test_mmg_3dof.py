"""Tests for MMG 3-DOF ship dynamics model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.models.mmg_3dof import MMG3DOFModel
from ta_mrc_pe_cc_tube_mpc.types import ControlCommand, EnvironmentState, VesselState


class TestMMG3DOFModel:
    def setup_method(self):
        self.model = MMG3DOFModel()
        self.env = EnvironmentState(water_depth=100.0)
        self.cmd_zero = ControlCommand(rudder=0.0, propeller=0.0, source="test")
        self.cmd_cruise = ControlCommand(rudder=0.0, propeller=0.5, source="test")

    def test_straight_line_propulsion(self):
        """Propeller thrust increases forward speed from rest."""
        state = VesselState(x=0, y=0, psi=0, u=0.1, v=0, r=0)
        s1 = self.model.step(state, self.cmd_cruise, self.env, dt=10.0)
        # After 10s of half throttle, speed should increase
        assert s1.u > state.u, (
            f"Speed should increase with propeller; u went from {state.u} to {s1.u}"
        )

    def test_no_control_speed_decay(self):
        """Without propulsion, speed decays due to hull resistance."""
        state = VesselState(x=0, y=0, psi=0, u=10.0, v=0, r=0)
        s1 = self.model.step(state, self.cmd_zero, self.env, dt=10.0)
        assert s1.u < state.u, (
            f"Speed should decay without thrust; u went from {state.u} to {s1.u}"
        )

    def test_rudder_causes_yaw(self):
        """Rudder input changes heading."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0, r=0)
        cmd = ControlCommand(rudder=10.0, propeller=0.3, source="test")
        s1 = self.model.step(state, cmd, self.env, dt=10.0)
        # Yaw rate should be non-zero with rudder
        assert abs(s1.r) > 1e-6, "Rudder should produce non-zero yaw rate"

    def test_rudder_sign_consistency(self):
        """Positive rudder (starboard) turns the ship to starboard (negative r)."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0, r=0)
        cmd = ControlCommand(rudder=10.0, propeller=0.3, source="test")
        s1 = self.model.step(state, cmd, self.env, dt=10.0)
        # Positive rudder → negative yaw moment → negative yaw rate
        assert s1.r < 0, (
            f"Positive rudder should turn to starboard (negative r), got r={s1.r}"
        )

    def test_yaw_rate_clamped(self):
        """Yaw rate respects max_yaw_rate constraint."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0, r=0)
        cmd = ControlCommand(rudder=35.0, propeller=0.8, source="test")
        s1 = self.model.step(state, cmd, self.env, dt=60.0)  # long step
        max_rate = self.model.max_yaw_rate
        assert abs(s1.r) <= max_rate * 1.01, (
            f"Yaw rate {s1.r} exceeds max {max_rate}"
        )

    def test_state_derivatives(self):
        """get_state_derivatives returns correct shape and non-NaN values."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0.5, r=0.01)
        derivs = self.model.get_state_derivatives(state, rudder=5.0, propeller=0.4)
        assert len(derivs) == 3
        assert not np.any(np.isnan(derivs)), f"Derivatives contain NaN: {derivs}"

    def test_zero_control_straight(self):
        """Zero controls produce no lateral motion from symmetric initial state."""
        state = VesselState(x=0, y=0, psi=0, u=8.0, v=0, r=0)
        s1 = self.model.step(state, ControlCommand(rudder=0.0, propeller=0.3, source="test"),
                             self.env, dt=1.0)
        # Sway (v) should stay near zero due to symmetry
        assert abs(s1.v) < 0.1, f"Sway velocity should stay near zero, got v={s1.v}"

    def test_position_update_direction(self):
        """Ship moves forward along heading when going straight."""
        state = VesselState(x=0, y=0, psi=0, u=5.0, v=0, r=0)
        cmd = ControlCommand(rudder=0.0, propeller=0.3, source="test")
        s1 = self.model.step(state, cmd, self.env, dt=2.0)
        # With psi=0 (East), ship should move in +x direction
        assert s1.x > 0, f"Ship should move East (+x), got x={s1.x}"
        assert abs(s1.y) < 0.5, f"Ship should not drift North/South, got y={s1.y}"

    def test_with_current_disturbance(self):
        """External disturbance force affects motion."""
        state = VesselState(x=0, y=0, psi=0, u=5.0, v=0, r=0)
        cmd = ControlCommand(rudder=0.0, propeller=0.3, source="test")
        # Current pushing sideways
        dist = np.array([0.0, 1e5, 0.0])  # strong lateral force
        s_dist = self.model.step(state, cmd, self.env, dt=1.0, disturbance_force=dist)
        s_no = self.model.step(state, cmd, self.env, dt=1.0)
        # With lateral disturbance, v should change
        assert s_dist.v != s_no.v, "Disturbance should affect sway velocity"
