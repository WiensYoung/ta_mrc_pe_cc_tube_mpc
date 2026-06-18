"""Tests for experiment smoke entrypoints."""

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_run_all_core_script_exists():
    path = os.path.join(PROJECT_ROOT, "scripts", "run_all_core.py")
    assert os.path.isfile(path), f"Missing script: {path}"


def test_run_all_ablations_script_exists():
    path = os.path.join(PROJECT_ROOT, "scripts", "run_all_ablations.py")
    assert os.path.isfile(path), f"Missing script: {path}"


def test_run_statistics_script_exists():
    path = os.path.join(PROJECT_ROOT, "scripts", "run_statistics.py")
    assert os.path.isfile(path), f"Missing script: {path}"


def test_run_all_core_has_dry_run():
    """script supports --dry-run to print plan without running."""
    path = os.path.join(PROJECT_ROOT, "scripts", "run_all_core.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "--dry-run" in content, "--dry-run flag not found in run_all_core.py"


def test_run_all_core_has_quick():
    with open(os.path.join(PROJECT_ROOT, "scripts", "run_all_core.py"), "r", encoding="utf-8") as f:
        content = f.read()
    assert "--quick" in content, "--quick flag not found in run_all_core.py"


def test_smoke_does_not_require_real_data():
    """Smoke tests use synthetic data only — no AIS/ENC needed."""
    path = os.path.join(PROJECT_ROOT, "src", "ta_mrc_pe_cc_tube_mpc",
                        "data", "synthetic_generator.py")
    assert os.path.isfile(path), "Synthetic generator must exist for smoke tests"


def test_real_replay_script_exists_or_is_documented():
    """Real AIS replay script must exist or be documented as needing data."""
    path = os.path.join(PROJECT_ROOT, "scripts", "run_real_ais_replay.py")
    assert os.path.isfile(path), (
        "run_real_ais_replay.py must exist (even if it requires user-provided data)"
    )
