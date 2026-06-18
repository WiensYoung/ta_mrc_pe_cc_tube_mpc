"""Integration tests for JAX-accelerated control pipeline.

Verifies end-to-end JAX mode produces safe trajectories equivalent
to the CasADi baseline.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from ta_mrc_pe_cc_tube_mpc.control.controller import TAMRCPECCTubeMPCController
from ta_mrc_pe_cc_tube_mpc.types import (
    ControlCommand, EnvironmentState, TargetShip, VesselParams, VesselState,
)


@pytest.fixture
def jax_config():
    """Minimal config with JAX backend."""
    return {
        "mpc": {
            "backend": "jax",
            "horizon": 10,   # shorter horizon for fast tests
            "dt": 0.5,
            "max_iterations": 50,
        },
        "controller": {
            "enable_multi_rule": True,
            "enable_chance_constraint": True,
            "enable_tube_mpc": True,
            "enable_cbf_filter": True,
            "enable_fallback": True,
            "enable_bank_effect": False,
            "enable_ship_interaction": False,
            "enable_shallow_water": False,
            "enable_wind_current": False,
        },
        "cbf": {"alpha_cbf": 1.0, "active_constraint_tolerance": 2.0},
        "fallback": {"max_consecutive_infeasible": 5},
        "domain": {},
        "rules": {},
        "tube": {"rho_0": 5.0, "rho_maneuver": 2.0, "use_adaptive": False},
        "uncertainty": {"epsilon_default": 0.10},
        "jax": {"enable_x64": True, "platform": "cpu", "precompile": True},
    }


@pytest.fixture
def casadi_config(jax_config):
    """Same config but with CasADi backend for comparison."""
    cfg = dict(jax_config)
    cfg["mpc"] = dict(jax_config["mpc"])
    cfg["mpc"]["backend"] = "casadi"
    return cfg


@pytest.fixture
def vessel_params():
    return VesselParams(
        name="TestShip",
        mmsi="999999001",
        length=180.0,
        beam=30.0,
        draught=10.5,
        mass=5e7,
        Iz=1.2e11,
        max_rudder=35.0,
        max_rudder_rate=5.0,
        max_speed=15.0,
        min_speed=0.0,
        max_yaw_rate=3.0,
        min_turning_radius_multiplier=3.0,
        vessel_type="cargo",
    )


class TestJAXIntegration:
    """End-to-end tests for JAX control pipeline."""

    def test_jax_controller_construction(self, jax_config, vessel_params):
        """Controller should initialize successfully in JAX mode."""
        ctrl = TAMRCPECCTubeMPCController(vessel_params, jax_config)
        assert ctrl._jax_mode is True
        assert ctrl.tube_mpc.backend == "jax"

    def test_jax_compute_control_no_targets(self, jax_config, vessel_params):
        """Compute control with no targets — should return safe command."""
        ctrl = TAMRCPECCTubeMPCController(vessel_params, jax_config)
        own_state = VesselState(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        env = EnvironmentState()

        cmd = ctrl.compute_control(own_state, vessel_params, [], env)

        assert cmd is not None
        assert -35.0 <= cmd.rudder <= 35.0
        assert 0.0 <= cmd.propeller <= 1.0
        assert "jax_mode" in cmd.metadata
        assert cmd.metadata["jax_mode"] is True

    def test_jax_compute_control_one_target(self, jax_config, vessel_params):
        """Compute control with one crossing target — should avoid collision."""
        ctrl = TAMRCPECCTubeMPCController(vessel_params, jax_config)
        own_state = VesselState(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        tgt_state = VesselState(x=500, y=100, psi=np.pi, u=5.0, v=0, r=0)
        tgt = TargetShip(
            mmsi="TGT001", vessel_type="cargo",
            state=tgt_state, length=180.0, beam=30.0, draught=10.5,
            covariance=np.eye(2) * 100.0,
        )
        env = EnvironmentState()

        cmd = ctrl.compute_control(own_state, vessel_params, [tgt], env)

        assert cmd is not None
        assert -35.0 <= cmd.rudder <= 35.0
        assert 0.0 <= cmd.propeller <= 1.0

    def test_jax_vs_casadi_equivalence(self, jax_config, casadi_config, vessel_params):
        """JAX and CasADi should produce similar control commands."""
        jax_ctrl = TAMRCPECCTubeMPCController(vessel_params, jax_config)
        casadi_ctrl = TAMRCPECCTubeMPCController(vessel_params, casadi_config)

        own_state = VesselState(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        tgt_state = VesselState(x=300, y=0, psi=np.pi, u=5.0, v=0, r=0)
        tgt = TargetShip(
            mmsi="TGT001", vessel_type="cargo",
            state=tgt_state, length=180.0, beam=30.0, draught=10.5,
            covariance=np.eye(2) * 100.0,
        )
        env = EnvironmentState()

        jax_cmd = jax_ctrl.compute_control(own_state, vessel_params, [tgt], env)
        casadi_cmd = casadi_ctrl.compute_control(own_state, vessel_params, [tgt], env)

        # Both should produce safe commands
        assert jax_cmd is not None
        assert casadi_cmd is not None

        # Rudder direction should agree (both should turn to avoid head-on collision)
        # Allow tolerance for different solver paths
        if abs(jax_cmd.rudder) > 0.5 and abs(casadi_cmd.rudder) > 0.5:
            assert np.sign(jax_cmd.rudder) == np.sign(casadi_cmd.rudder), \
                f"JAX rudder={jax_cmd.rudder}, CasADi rudder={casadi_cmd.rudder}"

    def test_jax_multiple_steps(self, jax_config, vessel_params):
        """Multiple control steps should remain stable."""
        ctrl = TAMRCPECCTubeMPCController(vessel_params, jax_config)
        state = VesselState(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        env = EnvironmentState()

        for _ in range(20):
            cmd = ctrl.compute_control(state, vessel_params, [], env)
            # Simple kinematic update
            state = VesselState(
                x=state.x + state.u * 0.5,
                y=state.y + state.v * 0.5,
                psi=state.psi + state.r * 0.5,
                u=state.u, v=state.v, r=state.r,
            )
            assert cmd is not None

    def test_jax_reset(self, jax_config, vessel_params):
        """Reset should clear state correctly."""
        ctrl = TAMRCPECCTubeMPCController(vessel_params, jax_config)
        state = VesselState(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        env = EnvironmentState()

        # Run one step
        cmd1 = ctrl.compute_control(state, vessel_params, [], env)
        assert cmd1 is not None

        # Reset
        ctrl.reset()
        assert ctrl.previous_command is None

        # Run again — should work after reset
        cmd2 = ctrl.compute_control(state, vessel_params, [], env)
        assert cmd2 is not None
