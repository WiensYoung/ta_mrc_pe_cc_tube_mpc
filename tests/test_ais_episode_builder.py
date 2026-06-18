"""Tests for AIS episode builder — utility functions and encounter extraction."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from ta_mrc_pe_cc_tube_mpc.data.ais_episode_builder import (
    _build_ais_quality,
    _find_contiguous_segments,
    _infer_covariance,
    _map_nav_status,
    _map_vessel_type,
    build_target_from_track,
    extract_encounters,
)
from ta_mrc_pe_cc_tube_mpc.types import TargetShip


def _make_ais_df(
    mmsis: list,
    timestamps: list,
    xs: list,
    ys: list,
    sogs: list = None,
    cogs: list = None,
    lengths: list = None,
    beams: list = None,
    draughts: list = None,
    vessel_types: list = None,
    nav_statuses: list = None,
) -> pd.DataFrame:
    """Build a minimal AIS DataFrame for testing."""
    n = len(mmsis)
    data = {
        "mmsi": mmsis,
        "timestamp": timestamps,
        "x": xs,
        "y": ys,
        "sog": sogs if sogs is not None else [10.0] * n,
        "cog": cogs if cogs is not None else [0.0] * n,
        "heading": [np.nan] * n,
        "vessel_type": vessel_types if vessel_types is not None else [0] * n,
        "nav_status": nav_statuses if nav_statuses is not None else [0] * n,
        "length": lengths if lengths is not None else [100.0] * n,
        "beam": beams if beams is not None else [15.0] * n,
        "draught": draughts if draughts is not None else [5.0] * n,
    }
    return pd.DataFrame(data)


class TestContiguousSegments:
    def test_empty_mask(self):
        assert _find_contiguous_segments(np.array([], dtype=bool)) == []

    def test_all_false(self):
        segs = _find_contiguous_segments(np.array([False, False, False]))
        assert segs == []

    def test_all_true(self):
        segs = _find_contiguous_segments(np.array([True, True, True]))
        assert segs == [(0, 2)]

    def test_single_true(self):
        segs = _find_contiguous_segments(np.array([False, True, False]))
        assert segs == [(1, 1)]

    def test_multiple_segments(self):
        mask = np.array([True, True, False, True, False, True, True])
        segs = _find_contiguous_segments(mask)
        assert segs == [(0, 1), (3, 3), (5, 6)]


class TestMapVesselType:
    def test_known_types(self):
        assert _map_vessel_type(30) == "engaged_in_fishing"
        assert _map_vessel_type(31) == "tug_barge"
        assert _map_vessel_type(40) == "ferry"
        assert _map_vessel_type(70) == "cargo"
        assert _map_vessel_type(80) == "tanker"
        assert _map_vessel_type(36) == "small_craft"

    def test_unknown_defaults_to_cargo(self):
        assert _map_vessel_type(999) == "cargo"
        assert _map_vessel_type(0) == "cargo"

    def test_string_input(self):
        assert _map_vessel_type("70") == "cargo"
        assert _map_vessel_type("not_a_number") == "cargo"


class TestMapNavStatus:
    def test_known_statuses(self):
        assert _map_nav_status(0) == "underway_engine"
        assert _map_nav_status(1) == "at_anchor"
        assert _map_nav_status(2) == "not_under_command"
        assert _map_nav_status(5) == "moored"

    def test_unknown_defaults_to_underway(self):
        assert _map_nav_status(99) == "underway"
        assert _map_nav_status(15) == "underway"


class TestInferCovariance:
    def test_minimal_track(self):
        """Single-row track gets base covariance."""
        track = _make_ais_df(
            mmsis=[1], timestamps=[0], xs=[0], ys=[0],
        )
        cov = _infer_covariance(track)
        assert cov.shape == (2, 2)
        assert np.allclose(cov, np.eye(2) * 100.0)  # default 10m std^2

    def test_covariance_clamped(self):
        """Output covariance is within expected bounds [25, 2500]."""
        track = _make_ais_df(
            mmsis=[1, 1], timestamps=[0, 10], xs=[0, 0], ys=[0, 0],
        )
        cov = _infer_covariance(track)
        assert 25.0 <= cov[0, 0] <= 2500.0  # 5m to 50m std squared

    def test_is_diagonal(self):
        """Covariance is diagonal (independent x, y)."""
        track = _make_ais_df(
            mmsis=[1, 1, 1], timestamps=[0, 10, 20], xs=[0, 10, 20], ys=[0, 5, 10],
        )
        cov = _infer_covariance(track)
        assert cov[0, 1] == pytest.approx(0.0)
        assert cov[0, 0] == pytest.approx(cov[1, 1])


class TestBuildAISQuality:
    def test_minimal_track(self):
        track = _make_ais_df(mmsis=[1], timestamps=[0], xs=[0], ys=[0])
        q = _build_ais_quality(track)
        assert q["message_count"] == 1
        assert q["position_accuracy"] == 1
        assert q["reporting_interval_s"] == 10.0

    def test_multi_point_track(self):
        track = _make_ais_df(
            mmsis=[1, 1, 1, 1],
            timestamps=[0, 10, 20, 30],
            xs=[0, 100, 200, 300],
            ys=[0, 0, 0, 0],
            sogs=[8, 8, 9, 9],
            cogs=[0, 1, 0, -1],
        )
        q = _build_ais_quality(track)
        assert q["message_count"] == 4
        assert q["track_duration_s"] == pytest.approx(30.0)
        assert 8.0 <= q["reporting_interval_s"] <= 12.0
        assert 0.5 <= q["message_completeness"] <= 1.0


class TestBuildTargetFromTrack:
    def test_basic_construction(self):
        track = _make_ais_df(
            mmsis=[1], timestamps=[100], xs=[500], ys=[300],
            sogs=[10], cogs=[90], lengths=[120], beams=[20], draughts=[6],
            vessel_types=[70], nav_statuses=[0],
        )
        target = build_target_from_track(track, "1")
        assert isinstance(target, TargetShip)
        assert target.mmsi == "1"
        assert target.state.x == pytest.approx(500.0)
        assert target.state.y == pytest.approx(300.0)
        assert target.length == 120.0
        assert target.beam == 20.0
        assert target.draught == 6.0
        assert target.vessel_type == "cargo"
        assert target.covariance is not None

    def test_at_time_interpolation(self):
        """Selects row nearest to the specified time."""
        track = _make_ais_df(
            mmsis=[1, 1, 1],
            timestamps=[0, 50, 100],
            xs=[0, 500, 1000],
            ys=[0, 0, 0],
        )
        target = build_target_from_track(track, "1", at_time=55)
        assert target.state.x == pytest.approx(500.0)

    def test_ais_quality_present(self):
        track = _make_ais_df(
            mmsis=[1, 1],
            timestamps=[0, 10],
            xs=[0, 100],
            ys=[0, 50],
            sogs=[10, 11],
            cogs=[0, 5],
        )
        target = build_target_from_track(track, "1")
        assert "reporting_interval_s" in target.ais_quality
        assert "message_count" in target.ais_quality

    def test_target_uses_defaults_for_nan(self):
        """NaN dimensions fall back to defaults."""
        track = _make_ais_df(
            mmsis=[1], timestamps=[0], xs=[0], ys=[0],
            lengths=[np.nan], beams=[np.nan], draughts=[np.nan],
        )
        target = build_target_from_track(track, "1")
        assert target.length == 100.0
        assert target.beam == 15.0
        assert target.draught == 5.0


class TestExtractEncounters:
    def test_no_other_vessels(self):
        """Single-vessel DataFrame returns empty list."""
        df = _make_ais_df(
            mmsis=[1, 1, 1],
            timestamps=[0, 10, 20],
            xs=[0, 100, 200],
            ys=[0, 0, 0],
        )
        encounters = extract_encounters(df, "1")
        assert encounters == []

    def test_vessels_never_close(self):
        """Vessels far apart produce no encounters."""
        df = _make_ais_df(
            mmsis=[1, 1, 2, 2],
            timestamps=[0, 10, 0, 10],
            xs=[0, 100, 10000, 10100],
            ys=[0, 0, 0, 0],
        )
        encounters = extract_encounters(df, "1")
        assert encounters == []

    def test_close_encounter_detected(self):
        """Vessels that come close produce an encounter."""
        n = 20
        timestamps = []
        mmsis = []
        xs = []
        ys = []
        for i in range(n):
            t = i * 10
            timestamps.extend([t, t])
            # Use integer MMSIs to match code's int() cast
            mmsis.extend([111111111, 222222222])
            # Ownship moves East, target moves West — rendezvous at t=100
            xs.extend([0 + 5 * t, 1000 - 5 * t])
            ys.extend([0, 0])
        df = _make_ais_df(mmsis=mmsis, timestamps=timestamps, xs=xs, ys=ys)
        encounters = extract_encounters(df, "111111111", cpa_threshold_m=500.0)
        assert len(encounters) > 0
        enc = encounters[0]
        assert enc["cpa_m"] <= 500.0
        assert "ownship_track" in enc
        assert "target_track" in enc

    def test_encounter_sorted_by_cpa(self):
        """Multiple encounters are sorted by CPA (closest first)."""
        n = 15
        timestamps = []
        mmsis = []
        xs = []
        ys = []
        # vessel 2 passes at 200m, vessel 3 at 400m
        for i in range(n):
            t = i * 10
            timestamps.extend([t, t, t])
            mmsis.extend([111111111, 222222222, 333333333])
            xs.extend([0 + 5 * t, 800 - 5 * t, 600 - 5 * t])
            ys.extend([0, 0, 0])
        df = _make_ais_df(mmsis=mmsis, timestamps=timestamps, xs=xs, ys=ys)
        encounters = extract_encounters(df, "111111111", cpa_threshold_m=600.0)
        assert len(encounters) >= 1
        if len(encounters) >= 2:
            assert encounters[0]["cpa_m"] <= encounters[1]["cpa_m"]

    def test_min_duration_filter(self):
        """Encounters shorter than min_duration_s are excluded."""
        n = 5
        timestamps = []
        mmsis = []
        xs = []
        ys = []
        for i in range(n):
            t = i * 30  # 120s total
            timestamps.extend([t, t])
            mmsis.extend([111111111, 222222222])
            xs.extend([0 + 5 * t, 500 - 5 * t])
            ys.extend([0, 0])
        df = _make_ais_df(mmsis=mmsis, timestamps=timestamps, xs=xs, ys=ys)
        # min_duration = 200s, but data spans only ~120s
        encounters = extract_encounters(df, "111111111", min_duration_s=200.0)
        assert encounters == []

    def test_ownship_too_few_points(self):
        """Ownship with < 2 points returns empty list."""
        df = _make_ais_df(
            mmsis=[1, 2],
            timestamps=[0, 0],
            xs=[0, 10],
            ys=[0, 0],
        )
        encounters = extract_encounters(df, "1")
        assert encounters == []
