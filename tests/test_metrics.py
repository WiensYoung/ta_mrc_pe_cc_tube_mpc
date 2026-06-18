"""Tests for evaluation metrics and failure taxonomy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.evaluation.metrics import compute_episode_metrics
from ta_mrc_pe_cc_tube_mpc.evaluation.failure_taxonomy import (
    build_failure_taxonomy_table,
    compute_failure_rates_by_method,
)
from ta_mrc_pe_cc_tube_mpc.evaluation.statistics import (
    cohens_d,
    cliffs_delta,
    compute_summary_statistics,
    holm_bonferroni,
    benjamini_hochberg,
    paired_ttest,
    wilcoxon_test,
    cluster_bootstrap,
)
from ta_mrc_pe_cc_tube_mpc.simulation.failure_detector import detect_failures, summarize_failures
from ta_mrc_pe_cc_tube_mpc.types import (
    EpisodeConfig,
    EpisodeResult,
    TargetShip,
    VesselParams,
    VesselState,
)


class TestStatistics:
    """Test statistical analysis functions."""

    def test_summary_statistics(self):
        """Compute basic descriptive statistics."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        stats = compute_summary_statistics(values)

        assert stats["mean"] == pytest.approx(5.5)
        assert stats["median"] == pytest.approx(5.5)
        assert "ci_95_lower" in stats
        assert "ci_95_upper" in stats
        assert "worst_5_percent" in stats

    def test_worst_5_percent(self):
        """Worst 5% correctly identifies the worst values."""
        values = np.array([1.0, 2.0, 3.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                          10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        stats = compute_summary_statistics(values)
        # Worst 5% ~= mean of one worst value
        assert stats["worst_5_percent"] < 5.0

    def test_paired_ttest(self):
        """Paired t-test between two groups."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.2, 2.3, 3.1, 4.4, 5.2])
        result = paired_ttest(x, y)
        assert "p_value" in result
        assert "mean_diff" in result

    def test_wilcoxon(self):
        """Wilcoxon signed-rank test."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = wilcoxon_test(x, y)
        assert "p_value" in result

    def test_cohens_d(self):
        """Cohen's d effect size."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(x, y)
        assert d == pytest.approx(0.0)

    def test_cliffs_delta(self):
        """Cliff's delta effect size."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        delta = cliffs_delta(x, y)
        # All x < y → delta = -1.0
        assert delta == pytest.approx(-1.0)

    def test_holm_bonferroni(self):
        """Holm-Bonferroni correction."""
        p_values = [0.001, 0.01, 0.03, 0.5]
        significant = holm_bonferroni(p_values, alpha=0.05)

        # p=0.001 should be significant; p=0.5 should not
        assert significant[0]  # 0.001 < 0.05/4 = 0.0125
        assert not significant[-1]

    def test_benjamini_hochberg(self):
        """Benjamini-Hochberg FDR correction."""
        p_values = [0.001, 0.01, 0.03, 0.5]
        significant = benjamini_hochberg(p_values, alpha=0.05)

        assert significant[0]  # 0.001 < 1/4 * 0.05
        assert not significant[-1]

    def test_cluster_bootstrap(self):
        """Cluster bootstrap respects cluster structure."""
        np.random.seed(42)
        n_clusters = 10
        data = np.random.randn(n_clusters * 5)
        cluster_ids = np.repeat(np.arange(n_clusters), 5)

        result = cluster_bootstrap(
            data, cluster_ids,
            statistic_fn=np.mean,
            n_bootstrap=100,
            seed=42,
        )
        assert "mean" in result
        assert "ci_95_lower" in result
        assert "ci_95_upper" in result
        assert result["ci_95_lower"] <= result["mean"] <= result["ci_95_upper"]


class TestFailureTaxonomy:
    """Test failure taxonomy and detection."""

    def test_failure_rates_by_method(self):
        """Compute failure rates per method."""
        results = []
        for i in range(10):
            r = EpisodeResult(
                episode_id=f"ep_{i}",
                scenario_id="S1",
                method="Proposed" if i < 5 else "B3",
                seed=i,
            )
            if i == 2:
                r.failure_flags["F1"] = True
                r.failure_type = "F1"
            results.append(r)

        rates = compute_failure_rates_by_method(results)
        assert "Proposed" in rates
        assert "F1" in rates["Proposed"]
        assert rates["Proposed"]["F1"] == 0.2


class TestNaNInfHandling:
    """Verify NaN/Inf are never silently converted to zero."""

    def test_summary_stats_preserves_nan_count(self):
        """NaN values are counted, not zero-filled."""
        values = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
        stats = compute_summary_statistics(values)
        assert stats["count"] == 5
        assert stats["valid_count"] == 3
        assert stats["nan_count"] == 2
        assert stats["inf_count"] == 0
        assert stats["mean"] == pytest.approx((1 + 2 + 4) / 3)

    def test_summary_stats_preserves_inf_count(self):
        """Inf values are counted separately."""
        values = np.array([1.0, np.inf, 3.0, -np.inf, 5.0])
        stats = compute_summary_statistics(values)
        assert stats["count"] == 5
        assert stats["valid_count"] == 3
        assert stats["inf_count"] == 2

    def test_summary_stats_all_nan_returns_zero_valid(self):
        stats = compute_summary_statistics(np.array([np.nan, np.nan, np.nan]))
        assert stats["count"] == 3
        assert stats["valid_count"] == 0
        assert stats["nan_count"] == 3
        assert "mean" not in stats  # no valid data → no mean

    def test_paired_ttest_filters_nan(self):
        """Paired t-test excludes NaN pairs and reports n_excluded."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.1, 2.1, 3.0, np.nan, 5.1])
        result = paired_ttest(x, y)
        # Valid pairs: (1.0,1.1), (2.0,2.1) → n=2 + wait, (5.0,5.1) → n=3
        # Actually: index 2 nan in x, index 3 nan in y → filtered
        assert result["n"] == 3
        assert result["n_excluded"] == 2
        assert result["n_total"] == 5

    def test_cohens_d_nan_input_returns_nan(self):
        """Cohen's d with all-NaN returns NaN, not 0."""
        d = cohens_d(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
        assert np.isnan(d), f"Expected NaN, got {d}"

    def test_safe_float_nan_returns_none(self):
        """_safe_float with NaN returns None, not 0.0."""
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import _safe_float
        assert _safe_float(float("nan")) is None
        assert _safe_float(float("inf")) is None
        assert _safe_float(float("-inf")) is None

    def test_safe_float_finite_unchanged(self):
        """_safe_float preserves finite values."""
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import _safe_float
        assert _safe_float(3.14) == 3.14
        assert _safe_float(0.0) == 0.0

    def test_safe_float_bool(self):
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import _safe_float
        assert _safe_float(True) == 1.0
        assert _safe_float(False) == 0.0

    def test_failure_status_present_in_metadata(self):
        """EpisodeResult metadata includes failure_status struct."""
        from ta_mrc_pe_cc_tube_mpc.simulation.closed_loop_runner import ClosedLoopRunner
        from ta_mrc_pe_cc_tube_mpc.types import (
            EnvironmentState, EpisodeConfig, VesselParams, VesselState,
        )

        ep = EpisodeConfig(
            scenario_id="test_fail_status", waterway="test",
            duration=5.0, dt=0.5,
            ownship_initial_state=VesselState(x=0, y=0, psi=0, u=7, v=0, r=0),
            ownship_params=VesselParams(), targets=[],
            environment_sequence=[EnvironmentState(water_depth=50.0)],
            random_seed=1,
        )
        runner = ClosedLoopRunner({"use_sampling": True,
            "controller": {"enable_multi_rule": True}})
        result = runner.run_single(ep, method="B3", seed=1)
        fs = result.metadata.get("failure_status", {})
        assert isinstance(fs, dict)
        for key in ("success", "failure_reason", "collision", "grounding",
                     "infeasible", "solver_failure", "timeout", "fallback_triggered"):
            assert key in fs, f"failure_status missing key '{key}'"
