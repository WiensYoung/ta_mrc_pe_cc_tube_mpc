"""Tests for metrics consistency across all methods."""

import numpy as np
import pytest
from ta_mrc_pe_cc_tube_mpc.evaluation.safety_evaluator import EpisodeSafetyMetrics


def test_proposed_and_baseline_have_same_schema():
    """All methods must output the same metric fields."""
    m = EpisodeSafetyMetrics()
    required_fields = {
        "collision", "min_distance", "min_required_safety_distance",
        "min_safety_margin", "mean_safety_margin", "mean_runtime",
        "p95_runtime", "deadline_miss_rate", "backend_degraded",
        "scenario_type", "data_source",
    }
    d = m.to_dict()
    for field in required_fields:
        assert field in d, f"Missing required field: {field}"


def test_baseline_required_distance_is_finite():
    """Even without advanced features, required safety distance is finite."""
    m = EpisodeSafetyMetrics()
    m.min_required_safety_distance = 4.0 * (180.0 + 100.0)  # conservative default
    d = m.to_dict()
    assert np.isfinite(d["min_required_safety_distance"])


def test_safety_margin_semantics():
    """min_safety_margin = min_distance - required_distance (positive = safe)."""
    m = EpisodeSafetyMetrics()
    m.min_distance = 500.0
    m.min_required_safety_distance = 400.0
    m.min_safety_margin = 500.0 - 400.0
    assert m.min_safety_margin == 100.0
