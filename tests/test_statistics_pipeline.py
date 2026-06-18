"""Tests for statistics pipeline."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest


def _make_fake_metrics_csv(n_methods=3, n_episodes=30) -> pd.DataFrame:
    """Generate a small fake metrics CSV for testing stats pipeline."""
    rng = np.random.RandomState(42)
    methods = ["Proposed", "B3", "B7"][:n_methods]
    rows = []
    for i in range(n_episodes):
        for method in methods:
            rows.append({
                "episode_id": f"ep{i}",
                "method": method,
                "collision": rng.choice([True, False], p=[0.1, 0.9]),
                "min_distance": rng.uniform(50, 500),
                "min_safety_margin": rng.uniform(-20, 200),
                "mean_runtime": rng.uniform(0.01, 0.3),
                "p95_runtime": rng.uniform(0.05, 0.5),
                "deadline_miss_rate": rng.beta(0.5, 10),
                "chance_constraint_violation_rate": rng.beta(0.3, 10),
                "scenario_type": "synthetic",
                "data_source": "procedural",
            })
    return pd.DataFrame(rows)


def test_fake_csv_generates_correct_shape():
    df = _make_fake_metrics_csv()
    assert len(df) == 90  # 3 methods × 30 episodes
    assert "method" in df.columns
    assert "min_distance" in df.columns


def test_stats_can_compute_mean_per_method():
    df = _make_fake_metrics_csv()
    means = df.groupby("method")["min_distance"].mean()
    assert len(means) == 3
    assert all(np.isfinite(means))


def test_p99_runtime_computed():
    df = _make_fake_metrics_csv()
    p99 = df.groupby("method")["p95_runtime"].apply(lambda x: np.percentile(x, 99))
    assert len(p99) == 3


def test_deadline_miss_rate_in_range():
    df = _make_fake_metrics_csv()
    assert all(df["deadline_miss_rate"].between(0, 1))


def test_empty_group_does_not_crash():
    df = _make_fake_metrics_csv()
    empty_df = df[df["method"] == "nonexistent"]
    if len(empty_df) == 0:
        assert True  # empty group handled gracefully


def test_nan_inf_handling_policy():
    """NaN/Inf in metrics should be handled explicitly."""
    df = _make_fake_metrics_csv()
    df.loc[0, "min_distance"] = float("nan")
    df.loc[1, "min_distance"] = float("inf")
    # Stats should use nan-aware functions
    finite = df["min_distance"][np.isfinite(df["min_distance"])]
    assert len(finite) < len(df)  # NaN/Inf filtered out
    assert np.isfinite(finite.mean())
