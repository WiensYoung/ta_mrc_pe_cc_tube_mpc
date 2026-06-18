"""Tests for chance constraint risk allocation."""

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.control.risk_allocation import (
    allocate_chance_risk,
    format_claim_statement,
)


class TestRiskAllocationBoole:
    def test_epsilon_step_equal_split(self):
        """epsilon_step = epsilon_total / (N * M) with Boole allocation."""
        eps_total = 0.10
        horizon = 20
        n_targets = 3
        eps_step = allocate_chance_risk(eps_total, horizon, n_targets, mode="boole")
        expected = eps_total / (horizon * n_targets)
        assert abs(eps_step - expected) < 1e-10

    def test_single_target_single_step(self):
        """Single target, single horizon step."""
        eps_step = allocate_chance_risk(0.10, 1, 1, mode="boole")
        assert abs(eps_step - 0.10) < 1e-10

    def test_zero_targets_uses_one(self):
        """Zero targets fall back to 1."""
        eps_step = allocate_chance_risk(0.10, 10, 0, mode="boole")
        assert eps_step > 0

    def test_none_mode_returns_total(self):
        """Mode='none' returns epsilon_total unchanged."""
        eps_step = allocate_chance_risk(0.10, 10, 5, mode="none")
        assert abs(eps_step - 0.10) < 1e-10


class TestRiskAllocationModes:
    def test_per_target_mode(self):
        eps = allocate_chance_risk(0.10, 20, 5, mode="per_target")
        assert abs(eps - 0.10 / 5) < 1e-10

    def test_per_horizon_mode(self):
        eps = allocate_chance_risk(0.10, 20, 5, mode="per_horizon")
        assert abs(eps - 0.10 / 20) < 1e-10

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            allocate_chance_risk(0.10, 10, 3, mode="invalid")

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            allocate_chance_risk(0.0, 10, 3, mode="boole")
        with pytest.raises(ValueError):
            allocate_chance_risk(1.5, 10, 3, mode="boole")


class TestClaimStatements:
    def test_none_claim_no_joint_guarantee(self):
        claim = format_claim_statement("none")
        assert "no joint" in claim.lower()

    def test_boole_claim_mentions_boole(self):
        claim = format_claim_statement("boole")
        assert "boole" in claim.lower()
        assert "conservative" in claim.lower()

    def test_all_modes_have_claims(self):
        for mode in ["none", "boole", "per_target", "per_horizon"]:
            claim = format_claim_statement(mode)
            assert len(claim) > 0
