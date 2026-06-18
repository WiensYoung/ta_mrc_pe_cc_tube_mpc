"""Tests for fallback strategy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.control.fallback import (
    FallbackController,
    FallbackLevel,
    FallbackState,
)
from ta_mrc_pe_cc_tube_mpc.types import (
    ControlCommand,
    EnvironmentState,
    TargetShip,
    VesselState,
)


class TestFallbackController:
    """Test fallback strategy trigger conditions and actions."""

    def setup_method(self):
        self.fallback = FallbackController()

    def test_initial_state_none(self):
        """Initial fallback state is NONE."""
        assert self.fallback.state.level == FallbackLevel.NONE
        assert not self.fallback.is_active()

    def test_mpc_infeasible_triggers_reduce_speed(self):
        """Consecutive MPC infeasibility triggers fallback."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        env = EnvironmentState(water_depth=50.0)
        cbf_info = {"over_intervention": False}

        # Accumulate infeasibilities
        for _ in range(6):  # > max_consecutive_infeasible (5)
            cmd = self.fallback.update(
                mpc_feasible=False, cbf_info=cbf_info,
                own_state=own, targets=[], env=env,
                d_safe_eff_list=[],
            )

        assert self.fallback.is_active()
        assert cmd.fallback_active

    def test_critical_distance_triggers_emergency(self):
        """Very close target triggers emergency fallback."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=10, y=0, psi=np.pi, u=3, v=0, r=0),
            length=100, beam=20, draught=5,
        )
        env = EnvironmentState(water_depth=50.0)
        cbf_info = {"over_intervention": False}

        cmd = self.fallback.update(
            mpc_feasible=True, cbf_info=cbf_info,
            own_state=own, targets=[tgt], env=env,
            d_safe_eff_list=[200.0],  # emergency threshold = 0.5 * 0.5 * 200 = 50m
        )

        # d=10m < 0.5 * (0.5 * 200) = 50m → emergency
        assert cmd.fallback_active
        # Emergency should reduce speed
        assert cmd.propeller < 0.3

    def test_fallback_respects_control_limits(self):
        """Fallback commands respect actuator constraints."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        env = EnvironmentState(water_depth=50.0)
        cbf_info = {"over_intervention": False}

        # Trigger fallback
        for _ in range(6):
            cmd = self.fallback.update(
                mpc_feasible=False, cbf_info=cbf_info,
                own_state=own, targets=[], env=env,
                d_safe_eff_list=[],
            )

        assert -35.0 <= cmd.rudder <= 35.0
        assert 0.0 <= cmd.propeller <= 1.0

    def test_reset_clears_state(self):
        """Reset returns fallback to NONE."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        env = EnvironmentState(water_depth=50.0)
        cbf_info = {"over_intervention": False}

        for _ in range(6):
            self.fallback.update(
                mpc_feasible=False, cbf_info=cbf_info,
                own_state=own, targets=[], env=env,
                d_safe_eff_list=[],
            )

        assert self.fallback.is_active()

        self.fallback.reset()
        assert not self.fallback.is_active()
        assert self.fallback.state.level == FallbackLevel.NONE

    def test_recovery_from_caution(self):
        """When conditions improve, fallback should eventually de-escalate."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        env = EnvironmentState(water_depth=50.0)
        cbf_info = {"over_intervention": False}

        # First, trigger caution via AIS delay
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=500, y=0, psi=np.pi, u=3, v=0, r=0),
            length=100, beam=20, draught=5,
            ais_quality={"delay": 60.0},  # > ais_loss_timeout
        )
        self.fallback.update(
            mpc_feasible=True, cbf_info=cbf_info,
            own_state=own, targets=[tgt], env=env,
            d_safe_eff_list=[300.0],
        )
        assert self.fallback.state.level == FallbackLevel.CAUTION

        # Now provide good conditions -> should eventually recover
        tgt_good = TargetShip(
            mmsi="T001",
            state=VesselState(x=500, y=0, psi=np.pi, u=3, v=0, r=0),
            length=100, beam=20, draught=5,
            ais_quality={"delay": 0.0},
        )
        for _ in range(15):
            self.fallback.update(
                mpc_feasible=True, cbf_info=cbf_info,
                own_state=own, targets=[tgt_good], env=env,
                d_safe_eff_list=[300.0],
            )
        assert not self.fallback.is_active()
