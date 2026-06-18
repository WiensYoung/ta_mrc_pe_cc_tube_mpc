"""Tests for JAX MMG 3-DOF model — equivalence with NumPy version."""

import numpy as np
import pytest

# Skip all tests if JAX is not installed
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from ta_mrc_pe_cc_tube_mpc.models.mmg_jax import MMGJAXModel
from ta_mrc_pe_cc_tube_mpc.models.mmg_3dof import MMG3DOFModel
from ta_mrc_pe_cc_tube_mpc.types import ControlCommand, EnvironmentState, VesselState


@pytest.fixture
def default_params():
    """Default vessel parameters matching configs/vessel.yaml ownship."""
    return {
        "length": 180.0,
        "beam": 30.0,
        "draught": 10.5,
        "mass": 5e7,
        "Iz": 1.2e11,
        "max_rudder": 35.0,
        "max_rudder_rate": 5.0,
        "max_yaw_rate": 3.0,
        "min_turning_radius": 3.0,
        "mmg": {
            "X_uu": -0.001,
            "Y_v": -0.015,
            "Y_r": 0.002,
            "Y_vv": -0.03,
            "Y_rr": 0.001,
            "Y_vr": -0.01,
            "N_v": -0.005,
            "N_r": -0.003,
            "N_vv": -0.005,
            "N_rr": -0.001,
            "N_vr": -0.003,
            "Y_delta": 0.003,
            "N_delta": -0.0015,
            "K_T_coeff": 0.15,
        },
    }


@pytest.fixture
def jax_model(default_params):
    return MMGJAXModel(default_params)


@pytest.fixture
def numpy_model(default_params):
    return MMG3DOFModel(default_params)


class TestMMGJAXEquivalence:
    """Verify JAX MMG produces equivalent results to NumPy MMG."""

    def test_straight_sailing(self, jax_model, numpy_model):
        """Ship sailing straight at 7 m/s — should continue straight."""
        # NumPy
        np_state = VesselState(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        np_cmd = ControlCommand(rudder=0, propeller=0.5, source="test")
        np_env = EnvironmentState()
        np_next = numpy_model.step(np_state, np_cmd, np_env, dt=0.5)

        # JAX
        jx_state = jax_model.initial_state(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        jx_next = jax_model.step(jx_state, rudder_deg=0.0, propeller=0.5,
                                  env_current_x=0.0, env_current_y=0.0, dt=0.5)

        # Compare
        np.testing.assert_allclose(jx_next["x"], np_next.x, rtol=1e-3)
        np.testing.assert_allclose(jx_next["y"], np_next.y, rtol=1e-3)
        np.testing.assert_allclose(jx_next["psi"], np_next.psi, rtol=1e-3)
        np.testing.assert_allclose(jx_next["u"], np_next.u, rtol=1e-2)
        np.testing.assert_allclose(jx_next["v"], np_next.v, rtol=1e-2)
        np.testing.assert_allclose(jx_next["r"], np_next.r, rtol=1e-2)

    def test_rudder_turn(self, jax_model, numpy_model):
        """Hard rudder turn at speed — both models should turn similarly."""
        # NumPy
        np_state = VesselState(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        np_cmd = ControlCommand(rudder=15, propeller=0.5, source="test")
        np_env = EnvironmentState()

        np_states = [np_state]
        for _ in range(20):
            np_next = numpy_model.step(np_states[-1], np_cmd, np_env, dt=0.5)
            np_states.append(np_next)

        # JAX
        jx_state = jax_model.initial_state(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        jx_states = [jx_state]
        for _ in range(20):
            jx_next = jax_model.step(jx_states[-1], rudder_deg=15.0, propeller=0.5,
                                      env_current_x=0.0, env_current_y=0.0, dt=0.5)
            jx_states.append(jx_next)

        # Compare final positions — turning radius should be similar
        np.testing.assert_allclose(jx_states[-1]["x"], np_states[-1].x, rtol=0.05)
        np.testing.assert_allclose(jx_states[-1]["y"], np_states[-1].y, rtol=0.05)
        np.testing.assert_allclose(jx_states[-1]["psi"], np_states[-1].psi, rtol=0.05)

    def test_current_effect(self, jax_model, numpy_model):
        """Cross-current should push ship laterally."""
        # NumPy (1 m/s cross-current from port)
        np_state = VesselState(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        np_cmd = ControlCommand(rudder=0, propeller=0.5, source="test")
        np_env = EnvironmentState(current_x=0.0, current_y=1.0)

        np_states = [np_state]
        for _ in range(20):
            np_next = numpy_model.step(np_states[-1], np_cmd, np_env, dt=0.5)
            np_states.append(np_next)

        # JAX
        jx_state = jax_model.initial_state(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        jx_states = [jx_state]
        for _ in range(20):
            jx_next = jax_model.step(jx_states[-1], rudder_deg=0.0, propeller=0.5,
                                      env_current_x=0.0, env_current_y=1.0, dt=0.5)
            jx_states.append(jx_next)

        np.testing.assert_allclose(jx_states[-1]["x"], np_states[-1].x, rtol=0.05)
        np.testing.assert_allclose(jx_states[-1]["y"], np_states[-1].y, rtol=0.05)

    def test_speed_bounds(self, jax_model):
        """Speed should not exceed max_speed."""
        state = jax_model.initial_state(x=0, y=0, psi=0, u=0, v=0, r=0)
        for _ in range(100):
            state = jax_model.step(state, rudder_deg=0.0, propeller=1.0,
                                    env_current_x=0.0, env_current_y=0.0, dt=1.0)
        speed = float(jnp.sqrt(state["u"]**2 + state["v"]**2))
        assert speed <= jax_model.max_speed + 1.0  # 1.0 tolerance for overshoot

    def test_rudder_rate_limit(self, jax_model):
        """Rudder should not change faster than max_rudder_rate."""
        state = jax_model.initial_state(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        # Try to command full rudder from zero
        state = jax_model.step(state, rudder_deg=35.0, propeller=0.5,
                                env_current_x=0.0, env_current_y=0.0, dt=0.5)
        # After 0.5s at 5 deg/s, rudder should be at most 2.5 deg
        assert abs(float(state["prev_rudder"])) <= jnp.deg2rad(2.5) + 1e-6

    def test_deterministic(self, jax_model):
        """Same inputs should produce same outputs."""
        state = jax_model.initial_state(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        s1 = jax_model.step(state, rudder_deg=10.0, propeller=0.5,
                             env_current_x=0.0, env_current_y=0.0, dt=0.5)
        s2 = jax_model.step(state, rudder_deg=10.0, propeller=0.5,
                             env_current_x=0.0, env_current_y=0.0, dt=0.5)
        for key in ["x", "y", "psi", "u", "v", "r"]:
            assert float(s1[key]) == float(s2[key])


class TestMMGJAXRollout:
    """Test trajectory rollout functionality."""

    def test_rollout_trajectory(self, jax_model):
        """Rollout should produce correct-length trajectory."""
        from ta_mrc_pe_cc_tube_mpc.models.mmg_jax import rollout_trajectory

        state = jax_model.initial_state(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        controls = jnp.zeros((10, 2))
        controls = controls.at[:, 0].set(2.0)  # 2 deg rudder
        controls = controls.at[:, 1].set(0.5)  # 0.5 propeller

        result = rollout_trajectory(jax_model, state, controls)
        x_traj = result["x_traj"]

        assert x_traj.shape == (11, 6)  # N_steps+1, 6-DOF
        assert float(x_traj[0, 0]) == 0.0  # initial x preserved

    def test_batch_rollout(self, jax_model):
        """Batch rollout should handle multiple trajectories."""
        from ta_mrc_pe_cc_tube_mpc.models.mmg_jax import batch_rollout_trajectory

        # Create 4 identical initial states
        s = jax_model.initial_state(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        batched_states = {k: jnp.stack([v, v, v, v]) for k, v in s.items()}

        # 4 different control sequences
        controls = jnp.zeros((4, 10, 2))
        controls = controls.at[:, :, 1].set(0.5)  # all at 0.5 propeller
        controls = controls.at[0, :, 0].set(0.0)   # straight
        controls = controls.at[1, :, 0].set(10.0)  # right turn
        controls = controls.at[2, :, 0].set(-10.0) # left turn
        controls = controls.at[3, :, 0].set(5.0)   # mild right

        trajs = batch_rollout_trajectory(jax_model, batched_states, controls)
        assert trajs.shape == (4, 11, 6)

        # Different controls should produce different trajectories
        assert float(trajs[0, -1, 1]) != float(trajs[1, -1, 1])  # different final y

    @pytest.mark.gpu
    def test_state_derivatives(self, jax_model):
        """Auto-diff derivatives should be finite and reasonable."""
        state = jax_model.initial_state(x=0, y=0, psi=0, u=7.0, v=0, r=0)
        derivs = jax_model.state_derivatives(state, rudder_deg=5.0, propeller=0.5, dt=0.01)
        derivs_np = np.array(derivs)
        assert derivs_np.shape == (3,)
        assert np.all(np.isfinite(derivs_np))
        # Surge acceleration should be small and positive for 0.5 propeller
        assert derivs_np[0] > -1.0
        assert derivs_np[0] < 10.0
