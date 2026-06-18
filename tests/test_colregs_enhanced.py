"""Tests for enhanced COLREGs — Rules 6, 8, 18, multi-vessel consistency."""

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.rules.colregs_enhanced import (
    resolve_responsibility,
    verify_action_substantial,
    assess_safe_speed,
    check_multi_vessel_consistency,
    VESSEL_RESPONSIBILITY_RANK,
    MINIMUM_SUBSTANTIAL_COURSE_CHANGE,
)
from ta_mrc_pe_cc_tube_mpc.types import VesselParams


# ═══════════════════════════════════════════════════════════════════════════
# Rule 18 — Vessel Responsibility Hierarchy
# ═══════════════════════════════════════════════════════════════════════════

class TestRule18ResponsibilityHierarchy:

    def test_nuc_has_highest_priority(self):
        """NUC vessel must be given way by all others."""
        resp = resolve_responsibility("underway", "cargo", "not_under_command", "cargo")
        assert resp["own_must_give_way"], "Ownship must give way to NUC"
        assert not resp["own_is_privileged"]

    def test_ram_over_power_driven(self):
        """RAM vessel has priority over power-driven vessel."""
        resp = resolve_responsibility("underway", "cargo", "restricted_maneuverability", "cargo")
        assert resp["own_must_give_way"]

    def test_power_driven_over_sailing(self):
        """In narrow channels, this can vary. Default: power gives way to sailing."""
        resp = resolve_responsibility("underway", "cargo", "underway", "sailing")
        assert resp["own_must_give_way"], "Power-driven gives way to sailing"

    def test_equal_rank_no_override(self):
        """Equal rank → no Rule 18 override (standard COLREGs apply)."""
        resp = resolve_responsibility("underway", "cargo", "underway", "tanker")
        assert not resp["own_must_give_way"]
        assert not resp["own_is_privileged"]
        assert "Equal rank" in resp["rule18_basis"]

    def test_anchored_is_static_obstacle(self):
        """Anchored vessel is not in Rule 18 hierarchy — static obstacle."""
        resp = resolve_responsibility("underway", "cargo", "at_anchor", "cargo")
        assert resp["target_rank"] == 0

    def test_own_nuc_privileged_over_others(self):
        """Ownship as NUC is privileged over all others."""
        resp = resolve_responsibility("not_under_command", "cargo", "underway", "cargo")
        assert resp["own_is_privileged"]
        assert not resp["own_must_give_way"]

    def test_full_hierarchy_order(self):
        """Verify the complete Rule 18 hierarchy ordering."""
        # NUC (1) < RAM (2) < CBD (3) < fishing (4) < sailing (5) < power (6)
        expected_order = [
            "not_under_command", "restricted_maneuverability",
            "constrained_by_draught", "fishing", "sailing", "power_driven"
        ]
        ranks = [VESSEL_RESPONSIBILITY_RANK[s] for s in expected_order]
        for i in range(len(ranks) - 1):
            assert ranks[i] < ranks[i + 1], (
                f"{expected_order[i]} (rank {ranks[i]}) should have higher priority "
                f"than {expected_order[i+1]} (rank {ranks[i+1]})"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Rule 8 — Action Substantiality
# ═══════════════════════════════════════════════════════════════════════════

class TestRule8ActionSubstantial:

    def test_head_on_30_deg_is_substantial(self):
        r = verify_action_substantial("head_on", 30.0, 0.0, own_speed=8.0)
        assert r["all_substantial"]
        assert r["course_substantial"]

    def test_head_on_10_deg_not_substantial(self):
        r = verify_action_substantial("head_on", 10.0, 0.0, own_speed=8.0)
        assert not r["all_substantial"]
        assert not r["course_substantial"]
        assert len(r["issues"]) > 0

    def test_crossing_20_deg_is_substantial(self):
        r = verify_action_substantial("crossing_giveway", 20.0, 0.0, own_speed=8.0)
        assert r["course_substantial"]

    def test_overtaking_10_deg_is_substantial(self):
        r = verify_action_substantial("overtaking_giveway", 10.0, 0.0, own_speed=8.0)
        assert r["course_substantial"]

    def test_stand_on_no_minimum_required(self):
        r = verify_action_substantial("crossing_standon", 0.0, 0.0, own_speed=8.0)
        assert r["all_substantial"], "Stand-on vessel has no minimum course change"

    def test_speed_only_change_not_radar_apparent(self):
        """Speed-only alteration with no course change is not radar-apparent."""
        r = verify_action_substantial("head_on", 0.0, -0.3, own_speed=8.0)
        assert not r["radar_apparent"]
        assert len(r["issues"]) > 0

    def test_substantial_speed_change_20_percent(self):
        """20% speed change meets minimum threshold."""
        r = verify_action_substantial("head_on", 30.0, 0.20, own_speed=8.0)
        assert r["speed_substantial"]

    def test_all_minimum_thresholds_defined(self):
        """Every encounter type has a minimum course change defined."""
        for etype in ["head_on", "crossing_giveway", "crossing_standon",
                       "overtaking_giveway", "overtaking_standon", "none"]:
            assert etype in MINIMUM_SUBSTANTIAL_COURSE_CHANGE, f"Missing: {etype}"


# ═══════════════════════════════════════════════════════════════════════════
# Rule 6 — Safe Speed Assessment
# ═══════════════════════════════════════════════════════════════════════════

class TestRule6SafeSpeed:

    @pytest.fixture
    def vessel_params(self):
        return VesselParams(length=180, beam=30, draught=10.5, max_speed=15)

    def test_good_conditions_full_speed(self, vessel_params):
        r = assess_safe_speed(vessel_params, visibility_nm=10, traffic_density=0,
                              water_depth_m=50)
        assert r.safe_speed_factor == 1.0
        assert len(r.speed_reduction_reasons) == 0

    def test_restricted_visibility_reduces_speed(self, vessel_params):
        r = assess_safe_speed(vessel_params, visibility_nm=0.3, traffic_density=0,
                              water_depth_m=50)
        assert r.safe_speed_factor < 1.0
        assert any("Visibility" in reason for reason in r.speed_reduction_reasons)

    def test_high_traffic_reduces_speed(self, vessel_params):
        r = assess_safe_speed(vessel_params, visibility_nm=10, traffic_density=0.9,
                              water_depth_m=50, num_nearby_targets=5)
        assert r.safe_speed_factor < 0.8
        assert any("traffic" in reason.lower() for reason in r.speed_reduction_reasons)

    def test_minimal_ukc_reduces_speed(self, vessel_params):
        r = assess_safe_speed(vessel_params, visibility_nm=10, traffic_density=0,
                              water_depth_m=12, ukc_m=1.5)
        assert r.safe_speed_factor < 1.0
        assert any("UKC" in reason for reason in r.speed_reduction_reasons)

    def test_bank_proximity_reduces_speed(self, vessel_params):
        r = assess_safe_speed(vessel_params, visibility_nm=10, traffic_density=0,
                              water_depth_m=50, bank_proximity=True)
        assert r.safe_speed_factor < 1.0
        assert any("bank" in reason.lower() for reason in r.speed_reduction_reasons)

    def test_speed_factor_never_below_20_percent(self, vessel_params):
        """Safe speed factor never drops below 0.2 (never fully stop)."""
        r = assess_safe_speed(vessel_params, visibility_nm=0.1, traffic_density=1.0,
                              water_depth_m=11, ukc_m=0.5, bank_proximity=True,
                              wind_speed_kn=40, current_speed_kn=5, num_nearby_targets=10)
        assert r.safe_speed_factor >= 0.2

    def test_stopping_distance_computed(self, vessel_params):
        r = assess_safe_speed(vessel_params)
        assert r.stopping_distance_m > 0

    def test_multiple_factors_cumulative(self, vessel_params):
        """Multiple risk factors accumulate speed reduction."""
        r = assess_safe_speed(vessel_params, visibility_nm=0.3, traffic_density=0.7,
                              water_depth_m=20, bank_proximity=True)
        assert r.safe_speed_factor < 0.6
        assert len(r.speed_reduction_reasons) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# Multi-vessel Consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestMultiVesselConsistency:

    def test_single_target_always_consistent(self):
        r = check_multi_vessel_consistency(
            own_state=None, target_ships=[None],
            colregs_actions=[{"required_course_change": 30, "encounter_type": "head_on"}],
        )
        assert r["consistent"]

    def test_same_direction_actions_consistent(self):
        """Both require starboard turn — consistent."""
        r = check_multi_vessel_consistency(
            own_state=None, target_ships=[None, None],
            colregs_actions=[
                {"required_course_change": 30, "encounter_type": "head_on"},
                {"required_course_change": 20, "encounter_type": "crossing_giveway"},
            ],
        )
        assert r["consistent"]

    def test_opposite_course_changes_conflict(self):
        """One requires starboard, other requires port — conflict."""
        r = check_multi_vessel_consistency(
            own_state=None, target_ships=[None, None],
            colregs_actions=[
                {"required_course_change": 30, "encounter_type": "head_on"},
                {"required_course_change": -25, "encounter_type": "crossing_giveway"},
            ],
        )
        assert not r["consistent"]
        assert len(r["conflicts"]) > 0
        assert "contradictory course" in r["conflicts"][0].lower()

    def test_small_changes_not_flagged(self):
        """Insignificant course changes (<5 deg) not flagged as conflicts."""
        r = check_multi_vessel_consistency(
            own_state=None, target_ships=[None, None],
            colregs_actions=[
                {"required_course_change": 30, "encounter_type": "head_on"},
                {"required_course_change": -3, "encounter_type": "crossing_standon"},
            ],
        )
        assert r["consistent"]  # -3 deg is too small to be a conflict

    def test_contradictory_speed_changes_conflict(self):
        r = check_multi_vessel_consistency(
            own_state=None, target_ships=[None, None],
            colregs_actions=[
                {"required_course_change": 0, "required_speed_change": 5, "encounter_type": "none"},
                {"required_course_change": 0, "required_speed_change": -5, "encounter_type": "none"},
            ],
        )
        assert not r["consistent"]
        assert any("speed" in c.lower() for c in r["conflicts"])
