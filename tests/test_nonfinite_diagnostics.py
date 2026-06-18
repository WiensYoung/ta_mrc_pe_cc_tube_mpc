"""Tests for NaN/Inf diagnostics."""

import numpy as np
import pytest
from ta_mrc_pe_cc_tube_mpc.evaluation.safety_evaluator import EpisodeSafetyMetrics


def test_nonfinite_force_count_in_metrics():
    m = EpisodeSafetyMetrics()
    m.nonfinite_force_count = 3
    d = m.to_dict()
    assert d["nonfinite_force_count"] == 3


def test_nonfinite_state_count_in_metrics():
    m = EpisodeSafetyMetrics()
    m.nonfinite_state_count = 1
    d = m.to_dict()
    assert d["nonfinite_state_count"] == 1


def test_nonfinite_control_count_in_metrics():
    m = EpisodeSafetyMetrics()
    m.nonfinite_control_count = 0
    d = m.to_dict()
    assert d["nonfinite_control_count"] == 0


def test_nonfinite_solver_output_count_in_metrics():
    m = EpisodeSafetyMetrics()
    m.nonfinite_solver_output_count = 5
    d = m.to_dict()
    assert d["nonfinite_solver_output_count"] == 5


def test_all_nonfinite_fields_default_to_zero():
    m = EpisodeSafetyMetrics()
    d = m.to_dict()
    for field in ["nonfinite_force_count", "nonfinite_state_count",
                  "nonfinite_control_count", "nonfinite_solver_output_count"]:
        assert d[field] == 0, f"Field '{field}' should default to 0"


def test_nan_inf_never_in_metrics():
    """NaN or Inf should never appear in metrics output — converted to None (JSON null).

    The _safe_float method in EpisodeSafetyMetrics returns None for NaN/Inf,
    which is detectable downstream (unlike silently returning 0.0, which was
    a statistical error that distorted safety metrics and hypothesis tests).
    NaN/Inf as actual float values MUST never appear in the output dict.
    """
    m = EpisodeSafetyMetrics()
    m.min_distance = float("nan")
    m.min_safety_margin = float("inf")
    m.mean_runtime = float("-inf")
    m.p95_runtime = float("nan")
    d = m.to_dict()
    for k, v in d.items():
        if isinstance(v, float):
            assert np.isfinite(v), (
                f"Field '{k}' contains NaN/Inf as float: {v}. "
                f"NaN/Inf must be converted to None (JSON null), not silently zeroed."
            )
        # Note: None is acceptable — it signals "no valid data" rather than
        # silently pretending a value exists.


def test_vessel_state_speed_handles_nan():
    """VesselState.speed() returns 0.0 for NaN/Inf velocities."""
    from ta_mrc_pe_cc_tube_mpc.types import VesselState
    s = VesselState(u=float("nan"), v=float("inf"))
    assert s.speed() == 0.0
    s2 = VesselState(u=10.0, v=0.0)
    assert s2.speed() > 0.0
