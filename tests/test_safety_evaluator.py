"""Tests for unified safety evaluator."""

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.evaluation.safety_evaluator import (
    EpisodeSafetyMetrics,
    evaluate_episode_safety,
    SafetyEvaluationBreakdown,
)
from ta_mrc_pe_cc_tube_mpc.types import (
    ControlCommand, VesselParams, VesselState, TargetShip,
)


def _make_state(x, y, psi=0.0, u=10.0, v=0.0, r=0.0):
    return VesselState(x=x, y=y, psi=psi, u=u, v=v, r=r)


def _make_params(length=180.0, beam=30.0):
    return VesselParams(length=length, beam=beam)


def test_all_fields_finite():
    """All EpisodeSafetyMetrics.to_dict() fields are finite."""
    m = EpisodeSafetyMetrics()
    d = m.to_dict()
    for k, v in d.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"Field '{k}' is not finite: {v}"


def test_evaluate_no_targets_returns_no_collision():
    """No targets → no collision."""
    own = [_make_state(0, 0)]
    m = evaluate_episode_safety(
        own_states=own,
        target_histories=[],
        own_params=_make_params(),
        target_params_list=[],
        command_history=[],
        metadata={},
    )
    assert not m.collision


def test_evaluate_distant_targets_no_collision():
    """Distant targets → no collision."""
    own = [_make_state(0, 0)]
    tgt = [_make_state(500, 0)]
    tgt_params = TargetShip(length=100, beam=15)
    m = evaluate_episode_safety(
        own_states=own,
        target_histories=[tgt],
        own_params=_make_params(),
        target_params_list=[tgt_params],
        command_history=[],
        metadata={},
    )
    assert not m.collision


def test_collision_with_overlapping_hulls():
    """Overlapping hulls → collision."""
    own = [_make_state(0, 0)]
    tgt = [_make_state(5, 0, psi=np.pi)]  # head-on, very close
    tgt_params = TargetShip(length=100, beam=20)
    m = evaluate_episode_safety(
        own_states=own,
        target_histories=[tgt],
        own_params=_make_params(100, 20),
        target_params_list=[tgt_params],
        command_history=[],
        metadata={},
    )
    assert m.collision


def test_safety_margin_from_metadata():
    """Safety margin uses metadata breakdown when available."""
    own = [_make_state(0, 0), _make_state(10, 0)]
    tgt = [_make_state(100, 0), _make_state(90, 0)]
    tgt_params = TargetShip(length=100, beam=15)
    metadata = {
        "safety_distance_breakdown": [
            {"base_domain": 500, "dynamic_margin": 100, "chance_margin": 50,
             "tube_margin": 30, "rule_margin": 20, "extra_margin": 0,
             "total": 700},
        ],
    }
    m = evaluate_episode_safety(
        own_states=own,
        target_histories=[tgt],
        own_params=_make_params(),
        target_params_list=[tgt_params],
        command_history=[],
        metadata=metadata,
    )
    assert m.min_required_safety_distance > 0
    assert np.isfinite(m.min_required_safety_distance)


def test_required_safety_distance_falls_back_to_default():
    """Missing metadata → conservative default (not NaN)."""
    own = [_make_state(0, 0)]
    tgt = [_make_state(100, 0)]
    tgt_params = TargetShip(length=100, beam=15)
    m = evaluate_episode_safety(
        own_states=own,
        target_histories=[tgt],
        own_params=_make_params(),
        target_params_list=[tgt_params],
        command_history=[],
        metadata={},
    )
    assert np.isfinite(m.min_required_safety_distance)
    assert m.min_required_safety_distance > 0
    assert len(m.missing_context_reasons) > 0


def test_collision_not_equal_safety_margin_violation():
    """Collision (bool) and safety_margin_violation_rate (float) are distinct fields."""
    m = EpisodeSafetyMetrics()
    m.collision = True
    m.safety_margin_violation_rate = 0.5
    d = m.to_dict()
    assert "collision" in d
    assert "safety_margin_violation_rate" in d
    # collision is bool, safety_margin_violation_rate is float — different semantics
    assert isinstance(d["collision"], bool)
    assert isinstance(d["safety_margin_violation_rate"], float)
    # They should differ in value (True != 0.5)
    assert d["collision"] != d["safety_margin_violation_rate"]


def test_nan_inf_sanitized():
    """NaN/Inf values are sanitized to 0.0."""
    m = EpisodeSafetyMetrics()
    m.min_distance = float("nan")
    m.min_safety_margin = float("inf")
    m.mean_runtime = float("-inf")
    d = m.to_dict()
    for k, v in d.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"Field '{k}' leaked NaN/Inf: {v}"


class TestSafetyBreakdown:
    def test_breakdown_creation(self):
        bd = SafetyEvaluationBreakdown(
            base_domain=500,
            dynamic_domain_margin=100,
            chance_margin=50,
            total_required_distance=680,
        )
        assert bd.total_required_distance == 680
