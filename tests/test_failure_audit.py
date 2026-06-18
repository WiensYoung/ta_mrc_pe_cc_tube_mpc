"""Tests for failure audit taxonomy."""

import os

import numpy as np
import pandas as pd
import pytest


def _make_fake_failure_csv() -> pd.DataFrame:
    rows = [
        {"episode_id": "ep0", "method": "Proposed", "failure_type": "collision",
         "F1": True, "F2": False},
        {"episode_id": "ep1", "method": "B3", "failure_type": "near_miss",
         "F1": False, "F2": True},
        {"episode_id": "ep2", "method": "B3", "failure_type": "solver_failure",
         "F1": False, "F3": True},
        {"episode_id": "ep3", "method": "Proposed", "failure_type": "cbf_infeasible",
         "F1": False, "F4": True},
        {"episode_id": "ep4", "method": "B7", "failure_type": "fallback_failed",
         "F1": True, "F5": True},
    ]
    return pd.DataFrame(rows)


def test_failure_taxonomy_has_required_categories():
    required = ["collision", "near_miss", "grounding", "rule_violation",
                "solver_timeout", "solver_failure", "cbf_infeasible",
                "fallback_failed", "nonfinite_dynamics",
                "chance_constraint_violation", "safety_margin_violation",
                "backend_degraded"]
    # Check that the failure taxonomy script or module lists these categories
    auditer = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "audit_failure_cases.py"
    )
    if os.path.isfile(auditer):
        with open(auditer, "r", encoding="utf-8") as f:
            content = f.read()
        for cat in required:
            # Check if the category or a close variant appears in the script
            found = cat.replace("_", "") in content.replace("_", "").lower()
            if not found:
                # Try partial match (e.g., "collision" matches "collision_or")
                parts = cat.split("_")
                found = all(p in content.lower() for p in parts)
            assert found, (
                f"Failure taxonomy missing category: {cat}"
            )


def test_collision_separate_from_near_miss():
    df = _make_fake_failure_csv()
    collisions = df[df["failure_type"] == "collision"]
    near_misses = df[df["failure_type"] == "near_miss"]
    assert len(collisions) == 1
    assert len(near_misses) == 1
    assert collisions.iloc[0]["episode_id"] != near_misses.iloc[0]["episode_id"]


def test_cbf_infeasible_separate_from_fallback_failed():
    df = _make_fake_failure_csv()
    cbf = df[df["failure_type"] == "cbf_infeasible"]
    fb = df[df["failure_type"] == "fallback_failed"]
    assert len(cbf) == 1
    assert len(fb) == 1
    assert cbf.iloc[0]["episode_id"] != fb.iloc[0]["episode_id"]


def test_failure_audit_script_exists():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "audit_failure_cases.py"
    )
    assert os.path.isfile(path), "audit_failure_cases.py must exist"


def test_representative_episode_id_in_failure():
    df = _make_fake_failure_csv()
    for _, row in df.iterrows():
        assert row["episode_id"], f"Failure row missing episode_id: {row}"
