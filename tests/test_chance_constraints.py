"""Tests for chance-constrained safety formulation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.risk.uncertainty import (
    compute_chance_constraint_margin,
    compute_relative_covariance,
    get_epsilon_for_waterway,
    inflate_covariance_delay,
    inflate_covariance_dropout,
)
from ta_mrc_pe_cc_tube_mpc.utils.math_utils import kappa_epsilon
from ta_mrc_pe_cc_tube_mpc.control.chance_constraints import ChanceConstraintEvaluator


class TestRelativeCovariance:
    """Test Sigma_rel = Sigma_j + Sigma_i."""

    def test_additive_covariance(self):
        """Relative covariance is the sum of individual covariances."""
        Sigma_i = np.array([[4.0, 0.0], [0.0, 4.0]])
        Sigma_j = np.array([[9.0, 0.0], [0.0, 9.0]])

        Sigma_rel = compute_relative_covariance(Sigma_i, Sigma_j)

        np.testing.assert_array_almost_equal(Sigma_rel, np.array([[13.0, 0.0], [0.0, 13.0]]))

    def test_zero_own_covariance(self):
        """Zero ownship covariance still correct."""
        Sigma_i = np.zeros((2, 2))
        Sigma_j = np.eye(2) * 100.0

        Sigma_rel = compute_relative_covariance(Sigma_i, Sigma_j)

        np.testing.assert_array_almost_equal(Sigma_rel, Sigma_j)

    def test_delay_inflates_covariance(self):
        """AIS delay increases covariance."""
        Sigma = np.eye(2) * 100.0
        vel = np.array([5.0, 0.0])

        Sigma_delayed = inflate_covariance_delay(Sigma, vel, delay=10.0)

        assert np.trace(Sigma_delayed) > np.trace(Sigma)

    def test_dropout_doubles_covariance(self):
        """Dropout doubles covariance."""
        Sigma = np.eye(2) * 100.0

        Sigma_drop = inflate_covariance_dropout(Sigma)

        np.testing.assert_array_almost_equal(Sigma_drop, Sigma * 2.0)


class TestKappaEpsilon:
    """Test kappa_epsilon = sqrt(chi2.ppf(1-epsilon, df=2))."""

    def test_epsilon_smaller_kappa_larger(self):
        """Smaller epsilon -> larger kappa (more conservative)."""
        k_10 = kappa_epsilon(0.10)
        k_05 = kappa_epsilon(0.05)
        k_01 = kappa_epsilon(0.01)

        assert k_01 > k_05 > k_10

    def test_kappa_positive(self):
        """Kappa is always positive."""
        for eps in [0.10, 0.05, 0.025, 0.01]:
            assert kappa_epsilon(eps) > 0


class TestChanceConstraintMargin:
    """Test chance constraint margin computation."""

    def test_margin_computation(self):
        """Margin = ||mu_rel|| - d_safe - kappa * sigma - rho_tube."""
        mu_rel = np.array([500.0, 0.0])
        Sigma_rel = np.eye(2) * 100.0  # 10m std
        d_safe = 200.0
        epsilon = 0.10
        rho_tube = 10.0

        result = compute_chance_constraint_margin(
            mu_rel, Sigma_rel, d_safe, epsilon, rho_tube,
        )

        assert "margin" in result
        assert "constraint_satisfied" in result
        assert result["current_distance"] == pytest.approx(500.0)
        assert result["required_distance"] > d_safe

    def test_zero_target_covariance_deterministic(self):
        """Zero target covariance -> degenerate to deterministic constraint."""
        mu_rel = np.array([300.0, 0.0])
        Sigma_rel = np.zeros((2, 2))
        d_safe = 200.0
        epsilon = 0.10

        result = compute_chance_constraint_margin(
            mu_rel, Sigma_rel, d_safe, epsilon, 0.0,
        )

        # With zero covariance, kappa * sqrt(0) = 0
        assert result["uncertainty_margin"] == pytest.approx(0.0)
        assert result["required_distance"] == pytest.approx(d_safe)

    def test_constraint_satisfied_far(self):
        """Far-away target satisfies constraint."""
        mu_rel = np.array([2000.0, 0.0])
        Sigma_rel = np.eye(2) * 100.0
        d_safe = 500.0
        epsilon = 0.10

        result = compute_chance_constraint_margin(
            mu_rel, Sigma_rel, d_safe, epsilon, 0.0,
        )

        assert result["constraint_satisfied"]

    def test_constraint_violated_close(self):
        """Close target violates constraint."""
        mu_rel = np.array([50.0, 0.0])
        Sigma_rel = np.eye(2) * 100.0
        d_safe = 500.0
        epsilon = 0.10

        result = compute_chance_constraint_margin(
            mu_rel, Sigma_rel, d_safe, epsilon, 0.0,
        )

        assert not result["constraint_satisfied"]
        assert result["margin"] < 0


class TestEpsilonSelection:
    """Test epsilon selection for different waterway types."""

    def test_waterway_epsilon_values(self):
        """Each waterway type maps to correct epsilon."""
        assert get_epsilon_for_waterway("normal") == 0.10
        assert get_epsilon_for_waterway("restricted") == 0.05
        assert get_epsilon_for_waterway("tss_ferry") == 0.025
        assert get_epsilon_for_waterway("tanker_lowvis") == 0.01

    def test_unknown_waterway_default(self):
        """Unknown waterway type uses default epsilon (0.10)."""
        assert get_epsilon_for_waterway("unknown_type") == 0.10
