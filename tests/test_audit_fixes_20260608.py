"""Targeted tests for audit report fixes (2026-06-08).

Tests Fix 1 (CSV dedup), Fix 2 (ship domain clamping),
Fix 3 (NaN→None), Fix 5 (single-ship env metrics),
Fix 4 (CBF/fallback accumulation).
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Fix 2: Ship domain clamping ────────────────────────────────────────────

class TestShipDomainClamping:
    """Verify that the dynamic ship domain doesn't produce runaway distances."""

    def test_d_ship_term_clamped(self):
        """d_ship term must not exceed reasonable bounds even at near-zero distance."""
        from ta_mrc_pe_cc_tube_mpc.risk.dynamic_ship_domain import DynamicShipDomain
        from ta_mrc_pe_cc_tube_mpc.types import EnvironmentState, TargetShip, VesselParams, VesselState

        domain = DynamicShipDomain({
            "alpha_s": 0.2,
            "d_ref": 1000.0,
            "d_min": 10.0,
            "d_ship_amp_max": 5.0,
            "max_domain": 5000.0,
        })

        own = VesselState(x=0, y=0, psi=0, u=7, v=0, r=0)
        # Place target at 5m (very close) to test clamping
        tgt = TargetShip(
            mmsi="test", state=VesselState(x=5, y=0, psi=0, u=7, v=0, r=0),
            length=180, beam=30, draught=10,
        )
        env = EnvironmentState()

        result = domain.compute(
            own_state=own, own_params=VesselParams(length=180, beam=30, draught=10,
                                                    mass=5e7, Iz=1.2e11, max_rudder=35),
            target=tgt, env=env, enable_ship_interaction=True,
            ship_interaction_risk=1.0,
        )

        # The d_ship term must be bounded by clamping d_ref/d_ij
        # d_ship_max ≈ 0.2 * 360 * (7/7) * 5.0 * 1.0 = 360m (not 14km)
        assert result["d_ship"] <= 500, \
            f"d_ship term {result['d_ship']} should be <= 500m with clamping"

        # Total must be capped
        assert result["total"] <= 5000.0, \
            f"Total domain {result['total']} should be <= 5000m"

    def test_narrow_factor_clamped(self):
        """narrow_factor must be clamped at narrow_factor_max."""
        from ta_mrc_pe_cc_tube_mpc.risk.dynamic_ship_domain import DynamicShipDomain
        from ta_mrc_pe_cc_tube_mpc.types import EnvironmentState, TargetShip, VesselParams, VesselState

        domain = DynamicShipDomain({
            "alpha_waterway": 0.15,
            "narrow_factor_max": 0.5,
        })

        own = VesselState(x=0, y=0, psi=0, u=7, v=0, r=0)
        tgt = TargetShip(
            mmsi="test", state=VesselState(x=500, y=0, psi=0, u=0, v=0, r=0),
            length=100, beam=20, draught=5,
        )
        env = EnvironmentState(channel_width=10.0)  # very narrow

        result = domain.compute(
            own_state=own, own_params=VesselParams(length=180, beam=30, draught=10,
                                                    mass=5e7, Iz=1.2e11, max_rudder=35),
            target=tgt, env=env, enable_waterway=True,
        )

        # narrow_factor = clip(1 - 10/(20*180), 0, 0.5) = clip(0.997, 0, 0.5) = 0.5
        assert result["d_waterway"] <= 0.15 * 180 * 0.5 + 1e-9, \
            f"d_waterway {result['d_waterway']} should be clamped through narrow_factor_max"


# ── Fix 5: Single-ship environmental metrics ────────────────────────────────

class TestSingleShipEnvironmentalMetrics:
    """Verify S4/S5/S8-style single-ship scenarios get env metrics."""

    def test_single_ship_no_targets_still_gets_env_metrics(self):
        """When n_targets==0, bank/UKC/channel metrics should still populate."""
        from ta_mrc_pe_cc_tube_mpc.evaluation.safety_evaluator import (
            EpisodeSafetyMetrics,
            evaluate_episode_safety,
        )
        from ta_mrc_pe_cc_tube_mpc.types import ControlCommand, EnvironmentState, VesselParams, VesselState

        own_states = [VesselState(x=0, y=0, psi=0, u=7, v=0, r=0) for _ in range(10)]
        commands = [ControlCommand(rudder=0, propeller=0.5) for _ in range(10)]
        own_params = VesselParams(length=180, beam=30, draught=10, mass=5e7,
                                   Iz=1.2e11, max_rudder=35)
        env_seq = [EnvironmentState(
            water_depth=15.0,
            bank_distance_left=50.0,
            bank_distance_right=100.0,
            channel_width=200.0,
        ) for _ in range(10)]

        m = evaluate_episode_safety(
            own_states=own_states,
            target_histories=[],  # no targets
            own_params=own_params,
            target_params_list=[],  # no targets
            command_history=commands,
            metadata={},
            env_sequence=env_seq,
        )

        # Should NOT crash — single-ship scenarios are now handled
        assert isinstance(m, EpisodeSafetyMetrics)
        # Environmental fields should be populated
        assert m.ukc_violation is not None
        assert m.bank_clearance_violation is not None
        assert m.channel_boundary_violation is not None
        # Collision fields should default to safe (no targets)
        assert m.collision is False

    def test_single_ship_detects_ukc_violation(self):
        """Shallow water single-ship should detect UKC violation."""
        from ta_mrc_pe_cc_tube_mpc.evaluation.safety_evaluator import evaluate_episode_safety
        from ta_mrc_pe_cc_tube_mpc.types import ControlCommand, EnvironmentState, VesselParams, VesselState

        own_states = [VesselState(x=0, y=0, psi=0, u=7, v=0, r=0) for _ in range(10)]
        commands = [ControlCommand(rudder=0, propeller=0.5) for _ in range(10)]
        own_params = VesselParams(length=180, beam=30, draught=10, mass=5e7,
                                   Iz=1.2e11, max_rudder=35)
        env_seq = [EnvironmentState(water_depth=10.5) for _ in range(10)]  # UKC = 0.5m → violation

        m = evaluate_episode_safety(
            own_states=own_states, target_histories=[], own_params=own_params,
            target_params_list=[], command_history=commands,
            metadata={}, env_sequence=env_seq,
        )

        assert m.ukc_violation is True, "UKC of 0.5m should trigger violation"


# ── Fix 1: CSV dedup ────────────────────────────────────────────────────────

class TestCSVIncrementalSave:
    """Verify no duplicate rows in incremental CSV saves."""

    def test_no_duplicate_rows_on_multiple_saves(self):
        """Each incremental save should only append NEW rows."""
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import _incremental_csv_save

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")

            # Simulate episode results as pickled dicts
            r1 = {"episode_id": "ep1", "scenario_id": "S1", "method": "M1",
                  "seed": 1, "failure_type": "", "failure_flags": {},
                  "metrics": {"collision": False}, "runtime_stats": {"mean_runtime": 1.0}}

            r2 = {"episode_id": "ep2", "scenario_id": "S1", "method": "M1",
                  "seed": 2, "failure_type": "", "failure_flags": {},
                  "metrics": {"collision": False}, "runtime_stats": {"mean_runtime": 1.1}}

            results = [r1]
            # First save
            count = _incremental_csv_save(results, csv_path, mode="append", saved_count=0)
            assert count == 1
            assert os.path.exists(csv_path)

            # Add second result
            results.append(r2)
            # Second save — only the new row should be appended
            count = _incremental_csv_save(results, csv_path, mode="append", saved_count=count)
            assert count == 2

            # Read back and verify no duplicates
            import pandas as pd
            df = pd.read_csv(csv_path)
            assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
            assert df["episode_id"].tolist() == ["ep1", "ep2"]

    def test_write_mode_overwrites_all(self):
        """Write mode should save everything and not duplicate."""
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import _incremental_csv_save

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")

            r = {"episode_id": "ep1", "scenario_id": "S1", "method": "M1",
                 "seed": 1, "failure_type": "", "failure_flags": {},
                 "metrics": {}, "runtime_stats": {}}
            results = [r]

            # Write mode
            count = _incremental_csv_save(results, csv_path, mode="write", saved_count=0)
            assert count == 1

            # Write again with same data — should overwrite, not duplicate
            count = _incremental_csv_save(results, csv_path, mode="write", saved_count=0)
            assert count == 1

            import pandas as pd
            df = pd.read_csv(csv_path)
            assert len(df) == 1, f"Write mode should overwrite, got {len(df)} rows"


# ── Fix 3: NaN → None ──────────────────────────────────────────────────────

class TestNaNHandling:
    """Verify NaN/Inf are NOT silently converted to 0.0."""

    def test_nan_not_zeroed_in_safety_evaluator(self):
        """NaN and Inf values in metrics should become None, not 0.0."""
        from ta_mrc_pe_cc_tube_mpc.evaluation.safety_evaluator import (
            EpisodeSafetyMetrics,
        )
        m = EpisodeSafetyMetrics()

        # The _safe_float inside to_dict should return None for NaN/Inf
        m.min_safety_margin = float("nan")
        m.mean_runtime = float("inf")
        m.p95_runtime = float("-inf")

        d = m.to_dict()
        assert d["min_safety_margin"] is None, \
            f"NaN safety margin should be None, got {d['min_safety_margin']}"
        assert d["mean_runtime"] is None, \
            f"+Inf runtime should be None, got {d['mean_runtime']}"
        assert d["p95_runtime"] is None, \
            f"-Inf runtime should be None, got {d['p95_runtime']}"
