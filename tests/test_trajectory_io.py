"""Tests for trajectory_io — save, load, validate, and shape metadata."""

import os
import tempfile

import numpy as np
import pytest


@pytest.fixture
def sample_result():
    """Build a minimal EpisodeResult with state/command/target histories."""
    from ta_mrc_pe_cc_tube_mpc.types import (
        ControlCommand,
        EpisodeResult,
        VesselState,
    )

    n_steps = 10
    state_history = [
        VesselState(x=float(i), y=0.0, psi=0.0, u=7.0, v=0.0, r=0.0, timestamp=i * 0.5)
        for i in range(n_steps)
    ]
    command_history = [
        ControlCommand(rudder=0.0, propeller=0.5, source="test")
        for _ in range(n_steps)
    ]
    target_histories = [
        [
            VesselState(x=100.0 + i, y=10.0, psi=np.pi, u=4.0, v=0.0, r=0.0, timestamp=i * 0.5)
            for i in range(n_steps)
        ],
    ]

    return EpisodeResult(
        episode_id="test_ep_001",
        scenario_id="S1",
        method="Proposed",
        seed=1,
        state_history=state_history,
        command_history=command_history,
        target_histories=target_histories,
    )


class TestTrajectorySaveLoad:
    """Round-trip save → load → validate."""

    def test_save_and_load_roundtrip(self, sample_result):
        from ta_mrc_pe_cc_tube_mpc.simulation.trajectory_io import (
            load_trajectory,
            save_trajectory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_trajectory(sample_result, tmpdir)
            assert os.path.exists(path)
            assert path.endswith(".npz")

            data = load_trajectory(path)
            assert "own_state" in data
            assert data["own_state"].shape == (10, 7)
            assert "own_command" in data
            assert data["own_command"].shape == (10, 4)
            assert "target_0_state" in data
            assert data["target_0_state"].shape == (10, 7)

    def test_save_empty_state_raises(self, sample_result):
        from ta_mrc_pe_cc_tube_mpc.simulation.trajectory_io import save_trajectory

        sample_result.state_history = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="empty state_history"):
                save_trajectory(sample_result, tmpdir)

    def test_load_missing_file_raises(self):
        from ta_mrc_pe_cc_tube_mpc.simulation.trajectory_io import load_trajectory

        with pytest.raises(FileNotFoundError):
            load_trajectory("/nonexistent/path.npz")

    def test_validate_passes_on_valid_data(self, sample_result):
        from ta_mrc_pe_cc_tube_mpc.simulation.trajectory_io import (
            load_trajectory,
            save_trajectory,
            validate_trajectory_schema,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_trajectory(sample_result, tmpdir)
            data = load_trajectory(path)
            issues = validate_trajectory_schema(data)
            assert issues == [], f"Unexpected issues: {issues}"

    def test_validate_detects_missing_key(self):
        from ta_mrc_pe_cc_tube_mpc.simulation.trajectory_io import (
            validate_trajectory_schema,
        )

        issues = validate_trajectory_schema({})
        assert len(issues) >= 1
        assert any("own_state" in i for i in issues)

    def test_validate_detects_wrong_dims(self):
        from ta_mrc_pe_cc_tube_mpc.simulation.trajectory_io import (
            validate_trajectory_schema,
        )

        data = {"own_state": np.zeros((5,))}  # 1-D, should be 2-D
        issues = validate_trajectory_schema(data)
        assert any("dims" in i for i in issues)


class TestTrajectoryShapeMeta:
    """Shape metadata is correct."""

    def test_shape_meta_values(self, sample_result):
        from ta_mrc_pe_cc_tube_mpc.simulation.trajectory_io import (
            trajectory_shape_meta,
        )

        meta = trajectory_shape_meta(sample_result)
        assert meta["state_history_shape"] == (10, 7)
        assert meta["command_history_shape"] == (10, 4)
        assert meta["target_history_keys"] == ["target_0_state"]
        assert meta["episode_id"] == "test_ep_001"
        assert meta["scenario_id"] == "S1"
        assert meta["method_id"] == "Proposed"
        assert meta["seed"] == 1

    def test_shape_meta_empty(self):
        from ta_mrc_pe_cc_tube_mpc.simulation.trajectory_io import (
            trajectory_shape_meta,
        )
        from ta_mrc_pe_cc_tube_mpc.types import EpisodeResult

        r = EpisodeResult(episode_id="e", scenario_id="s", method="B3", seed=0)
        meta = trajectory_shape_meta(r)
        assert meta["state_history_shape"] == (0, 0)
        assert meta["command_history_shape"] == (0, 0)


class TestIntegrationTrajectoryAccess:
    """Integration: real run preserves trajectory or trajectory_file."""

    def test_serial_runner_preserves_state_history(self):
        """Serial ClosedLoopRunner keeps state_history in result."""
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import (
            ClosedLoopRunner,
        )
        from ta_mrc_pe_cc_tube_mpc.types import (
            EnvironmentState,
            EpisodeConfig,
            VesselParams,
            VesselState,
        )

        ep = EpisodeConfig(
            scenario_id="test_traj", waterway="test",
            duration=5.0, dt=0.5,
            ownship_initial_state=VesselState(x=0, y=0, psi=0, u=7, v=0, r=0),
            ownship_params=VesselParams(), targets=[],
            environment_sequence=[EnvironmentState(water_depth=50.0)],
            random_seed=1,
        )
        runner = ClosedLoopRunner({"use_sampling": True,
            "controller": {"enable_multi_rule": True}})
        result = runner.run_single(ep, method="B3", seed=1)

        # Serial mode → state_history preserved directly
        assert len(result.state_history) > 0, "state_history should be non-empty"
        assert len(result.command_history) > 0, "command_history should be non-empty"
        # _trajectory_meta is populated only via the pickle path (trajectory_dir set).
        # In serial mode (trajectory_dir=None), _trajectory_meta is not stored.
        if result.metadata.get("_trajectory_meta"):
            assert result.metadata["_trajectory_meta"]["n_steps"] > 0

    def test_runner_batch_preserves_trajectory_file(self):
        """Batch runner either embeds state_history or records trajectory_file."""
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import (
            ClosedLoopRunner,
        )
        from ta_mrc_pe_cc_tube_mpc.types import (
            EnvironmentState,
            EpisodeConfig,
            VesselParams,
            VesselState,
        )

        ep = EpisodeConfig(
            scenario_id="test_traj_batch", waterway="test",
            duration=5.0, dt=0.5,
            ownship_initial_state=VesselState(x=0, y=0, psi=0, u=7, v=0, r=0),
            ownship_params=VesselParams(), targets=[],
            environment_sequence=[EnvironmentState(water_depth=50.0)],
            random_seed=1,
        )

        # Batch with 1 method, 1 seed — goes through serial path
        runner = ClosedLoopRunner({"use_sampling": True,
            "controller": {"enable_multi_rule": True}})
        results = runner.run_batch(
            episodes=[ep], methods=["B3"], n_seeds=1, verbose=False,
        )
        assert len(results) == 1
        r = results[0]
        # After run_batch with n_workers=4, goes parallel → trajectory_file
        # With n_workers=1, goes serial → embedded state_history
        has_traj = len(r.state_history) > 0
        has_file = bool(r.metadata.get("trajectory_file"))
        assert has_traj or has_file, (
            "Result must have state_history or trajectory_file"
        )
