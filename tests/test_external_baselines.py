"""Tests for external baseline registry and status."""

import pytest

from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
    BASELINE_REGISTRY,
    ABLATION_REGISTRY,
    list_baseline_methods,
    validate_registry,
    get_baseline,
)


def test_registry_distinguishes_implemented_from_planned():
    """BASELINE_REGISTRY contains only implemented methods.

    G1 (2026-06-10): B8 (CBF-only) is now IMPLEMENTED and registered.
    B0, B9, B10, B11, B12 remain PLANNED.
    """
    methods = list_baseline_methods()
    for planned in ["B0", "B9", "B10", "B11", "B12"]:
        assert planned not in methods, (
            f"Planned baseline '{planned}' should NOT be in BASELINE_REGISTRY. "
            f"It must be implemented before registration."
        )


def test_b8_cbf_only_is_registered():
    """B8 (CBF-only) is now an implemented baseline (G1 — 2026-06-10)."""
    methods = set(list_baseline_methods())
    assert "B8" in methods, "B8 (CBF-only) must now be in registry per G1 implementation"


def test_proposed_method_in_registry():
    assert "Proposed" in BASELINE_REGISTRY


def test_each_baseline_has_required_keys():
    for method in list_baseline_methods():
        entry = get_baseline(method)
        for key in ["label", "controller", "description", "features"]:
            assert key in entry, f"{method} missing key: {key}"


def test_validate_registry_no_issues():
    issues = validate_registry()
    assert len(issues) == 0, f"Registry validation issues: {issues}"


def test_ablation_registry_has_required_ablations():
    required = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A11", "A12"]
    for a_id in required:
        assert a_id in ABLATION_REGISTRY, f"Missing required ablation: {a_id}"


def test_b0_apf_has_velocity_obstacle_controller():
    """B1 (VO/OZT) serves as the APF-inspired heuristic baseline."""
    b1 = get_baseline("B1")
    assert b1["controller"] == "velocity_obstacle"


def test_b3_uses_sampling_backend():
    b3 = get_baseline("B3")
    assert b3.get("backend_override") == "sampling"
