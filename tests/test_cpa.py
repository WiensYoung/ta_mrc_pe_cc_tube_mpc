"""Tests for TCPA/DCPA computation with boundary conditions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.risk.cpa import (
    compute_cpa_for_anchored,
    compute_cpa_for_vessel_pair,
    compute_tcpa_dcpa,
)
from ta_mrc_pe_cc_tube_mpc.types import VesselState


class TestTCPADCPA:
    """Test TCPA/DCPA computation with boundary cases."""

    def test_zero_relative_velocity(self):
        """When v_rel ≈ 0, TCPA should be clamped to horizon and DCPA = current distance."""
        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([5.0, 0.0])
        tgt_pos = np.array([100.0, 0.0])
        tgt_vel = np.array([5.0, 0.0])  # same velocity

        result = compute_tcpa_dcpa(own_pos, own_vel, tgt_pos, tgt_vel, horizon=600.0)

        assert result["v_rel_norm"] < 1e-3  # ~zero relative speed
        assert result["tcpa"] == 600.0  # clamped to horizon
        assert result["dcpa"] == pytest.approx(100.0, rel=1e-2)
        assert not result["is_converging"]

    def test_negative_tcpa(self):
        """TCPA_raw < 0 should be clipped to 0 (vessels are separating)."""
        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([5.0, 0.0])
        tgt_pos = np.array([-100.0, 0.0])   # target behind ownship
        tgt_vel = np.array([0.0, 0.0])       # stationary

        result = compute_tcpa_dcpa(own_pos, own_vel, tgt_pos, tgt_vel, horizon=600.0)

        assert result["tcpa_raw"] < 0  # raw is negative (ownship moving away)
        assert result["tcpa"] == 0.0  # clipped to 0
        assert not result["is_converging"]

    def test_tcpa_exceeds_horizon(self):
        """TCPA_raw > T_H should be clipped to T_H."""
        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([1.0, 0.0])
        tgt_pos = np.array([10000.0, 0.0])
        tgt_vel = np.array([0.0, 0.0])

        result = compute_tcpa_dcpa(own_pos, own_vel, tgt_pos, tgt_vel, horizon=600.0)

        assert result["tcpa_raw"] > 600.0
        assert result["tcpa"] == 600.0

    def test_normal_crossing(self):
        """Normal crossing: vessels approaching at 90 degrees."""
        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([5.0, 0.0])
        tgt_pos = np.array([500.0, 100.0])
        tgt_vel = np.array([0.0, -5.0])

        result = compute_tcpa_dcpa(own_pos, own_vel, tgt_pos, tgt_vel, horizon=600.0)

        assert result["is_converging"]
        assert result["tcpa"] > 0
        assert result["tcpa"] < result["current_distance"] / result["v_rel_norm"] + 1
        assert result["dcpa"] >= 0.0

    def test_overtaking(self):
        """Overtaking: ownship catches up from behind."""
        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([8.0, 0.0])
        tgt_pos = np.array([200.0, 10.0])
        tgt_vel = np.array([4.0, 0.0])

        result = compute_tcpa_dcpa(own_pos, own_vel, tgt_pos, tgt_vel, horizon=600.0)

        assert result["is_converging"]
        assert result["tcpa"] > 0
        # DCPA should be small (nearly parallel)
        assert result["dcpa"] < 20.0

    def test_anchored_target(self):
        """Anchored target treated as static obstacle with dimensions."""
        own_pos = np.array([0.0, 0.0])
        own_vel = np.array([5.0, 0.0])
        tgt_pos = np.array([500.0, 0.0])

        result = compute_cpa_for_anchored(
            own_pos, own_vel, tgt_pos,
            target_length=100.0, target_beam=20.0,
        )

        assert "target_extent" in result
        assert result["dcpa_effective"] < result["dcpa"]
        assert result["target_extent"] > 100.0

    def test_cpa_vessel_pair(self):
        """CPA between two VesselState objects."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        tgt = VesselState(x=500, y=100, psi=np.pi, u=3, v=0, r=0)

        result = compute_cpa_for_vessel_pair(own, tgt)
        assert result["tcpa"] >= 0
        assert result["dcpa"] >= 0
