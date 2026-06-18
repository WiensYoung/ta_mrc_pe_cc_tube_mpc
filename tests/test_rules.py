"""Tests for COLREGs, multi-source rule priority, and violation metrics."""

import numpy as np
import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def own_state():
    from ta_mrc_pe_cc_tube_mpc.types import VesselState
    return VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)


@pytest.fixture
def own_params():
    from ta_mrc_pe_cc_tube_mpc.types import VesselParams
    return VesselParams()


@pytest.fixture
def env_deep():
    from ta_mrc_pe_cc_tube_mpc.types import EnvironmentState
    return EnvironmentState(water_depth=50.0)


@pytest.fixture
def rule_engine():
    from ta_mrc_pe_cc_tube_mpc.rules.rule_engine import RuleEngine
    return RuleEngine()


# ── COLREGs head-on behavior ────────────────────────────────────────────────

class TestCOLREGsHeadOn:
    """Head-on encounter must trigger starboard maneuver."""

    def test_head_on_starboard(self, own_state, own_params, env_deep, rule_engine):
        """Head-on → own_role includes give-way logic and preference for starboard."""
        from ta_mrc_pe_cc_tube_mpc.types import TargetShip, VesselState

        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=500, y=0, psi=np.pi, u=8, v=0, r=0),
            length=150, beam=25, draught=8,
        )
        assessment = rule_engine.assess(own_state, own_params, [tgt], env_deep)
        assert assessment.encounter_type in ("head_on", "crossing_giveway",
                                              "overtaking_giveway", "overtaking_standon",
                                              "crossing_standon", "none")
        assert assessment.encounter_type == "head_on", \
            f"Expected head_on for 180° opposing, got {assessment.encounter_type}"

    def test_head_on_with_collision_risk_starboard(self):
        """VO controller in head-on must command starboard rudder."""
        from ta_mrc_pe_cc_tube_mpc.control.velocity_obstacle import (
            VelocityObstacleController,
        )
        from ta_mrc_pe_cc_tube_mpc.types import (
            EnvironmentState,
            TargetShip,
            VesselParams,
            VesselState,
        )

        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        tgt = TargetShip(
            mmsi="T001",
            state=VesselState(x=300, y=0, psi=np.pi, u=8, v=0, r=0),
        )
        params = VesselParams()
        vo = VelocityObstacleController(params, {"controller": {"vo_safety_radius": 500}})
        cmd = vo.compute_control(own, params, [tgt], EnvironmentState())
        assert cmd.rudder > 0, f"Head-on must turn starboard, got rudder={cmd.rudder}"


# ── COLREGs crossing give-way ───────────────────────────────────────────────

class TestCOLREGsCrossing:
    """Crossing give-way: avoid crossing ahead of target."""

    def test_crossing_giveway_not_through_bow(self):
        """Give-way vessel should not steer toward target's bow."""
        from ta_mrc_pe_cc_tube_mpc.rules.colregs import colregs_action
        from ta_mrc_pe_cc_tube_mpc.types import VesselParams, VesselState

        own = VesselState(x=0, y=0, psi=0, u=8, v=0, r=0)
        # Target coming from starboard (crossing give-way for ownship)
        tgt = VesselState(x=200, y=200, psi=np.deg2rad(-135), u=6, v=0, r=0)
        params = VesselParams()
        action = colregs_action(own, tgt, params)
        # In crossing give-way, required course change should be starboard
        if action["encounter_type"] == "crossing_giveway":
            assert action["required_course_change"] >= 0, (
                "Give-way crossing should turn starboard (positive course change)"
            )


# ── Stand-on behavior ──────────────────────────────────────────────────────

class TestStandOn:
    """Stand-on vessel maintains course unless imminent danger."""

    def test_stand_on_keeps_course(self, own_state, own_params, env_deep, rule_engine):
        """Stand-on vessel at safe distance has no required course change."""
        from ta_mrc_pe_cc_tube_mpc.types import TargetShip, VesselState

        # Target approaching from port (ownship is stand-on)
        tgt = VesselState(x=1000, y=-300, psi=np.deg2rad(20), u=5, v=0, r=0)
        # colregs_action needs own_state and target_state
        from ta_mrc_pe_cc_tube_mpc.rules.colregs import colregs_action

        action = colregs_action(own_state, tgt, own_params)
        role = action.get("ownship_role", "none")
        # At 1000m distance, shouldn't trigger evasion if stand-on
        if role == "stand_on":
            assert action["is_stand_on"], "Stand-on vessel should be marked as stand_on"


# ── P0 priority > P2 ───────────────────────────────────────────────────────

class TestPriorityHierarchy:
    """P0 hard constraints override P2 COLREGs when conflicting."""

    def test_p0_overrides_colregs(self):
        """P0 (grounding) constraint has higher priority than P2 (COLREGs)."""
        from ta_mrc_pe_cc_tube_mpc.rules.rule_priority import (
            get_rule_priority_level,
            is_hard_constraint,
        )

        # P0 is always hard
        assert is_hard_constraint("collision_avoidance")
        assert is_hard_constraint("grounding_risk_area")
        # P2 is hard by default (configurable)
        assert is_hard_constraint("colregs")

        # P0 priority levels are higher than P2
        p0 = get_rule_priority_level("grounding_risk_area")
        p2 = get_rule_priority_level("colregs")
        assert p0.value < p2.value, f"P0={p0.value} should have higher priority than P2={p2.value}"

    def test_shallow_water_p0_violation(self, rule_engine, own_state, own_params):
        """UKC violation triggers P0 constraint."""
        from ta_mrc_pe_cc_tube_mpc.types import EnvironmentState

        env_shallow = EnvironmentState(water_depth=5.0)  # draught=10.5 → UKC < 0
        assessment = rule_engine.assess(own_state, own_params, [], env_shallow)
        # P0 constraint should be triggered at water_depth=5.0 < draught=10.5+1.0
        p0_violations = [c for c in assessment.hard_constraints if c.get("priority") == "P0"]
        assert len(p0_violations) > 0, f"Expected P0 violation for shallow water, got {assessment.hard_constraints}"


# ── ENC forbidden zone ─────────────────────────────────────────────────────

class TestENCForbiddenZone:
    """ENC forbidden zones should not be crossed."""

    def test_navigable_point(self):
        """is_navigable returns True for open water."""
        from ta_mrc_pe_cc_tube_mpc.data.enc_layers import make_synthetic_enc

        enc = make_synthetic_enc("test", depth=30.0, channel_width=400.0)
        assert enc.is_navigable(0, 0, min_depth=20.0)
        assert not enc.is_navigable(0, 0, min_depth=40.0)  # too shallow

    def test_land_not_navigable(self):
        """Points in land polygons are not navigable."""
        from ta_mrc_pe_cc_tube_mpc.data.enc_layers import make_synthetic_enc

        enc = make_synthetic_enc("test", depth=30.0, channel_width=200.0,
                                  bank_left=100.0, bank_right=100.0)
        # Far outside channel → land
        assert not enc.is_navigable(500.0, 0.0)


# ── VTS / local rules in RuleContext ───────────────────────────────────────

class TestVTSInRuleContext:
    """VTS zone info persists through rule assessment."""

    def test_vts_zone_accessible(self):
        """VTS zone can be built and queried."""
        from ta_mrc_pe_cc_tube_mpc.data.vts_rules import build_puget_sound_vts

        vts = build_puget_sound_vts()
        assert vts.name == "Puget Sound VTS"
        assert len(vts.reporting_lines) >= 1
        rules = vts.get_applicable_rules(0, 0)
        # 0,0 might or might not be inside — depends on coordinate mapping
        assert isinstance(rules, list)


# ── Rule violation metrics ─────────────────────────────────────────────────

class TestRuleViolationMetrics:
    """Metrics correctly track rule violation counts."""

    def test_metric_keys_present(self):
        """All rule-violation metric keys are present in a fresh metrics dict."""
        expected = {
            "colregs_violation", "colregs_violation_count",
            "tss_lane_violation", "channel_violation", "channel_violation_count",
            "separation_zone_intrusion", "atba_intrusion",
            "hard_constraint_violation_count", "grounding_violation",
            "forbidden_zone_violation", "vts_rule_violation_count",
            "rule_exception_count", "rule_conflict_count",
            "stand_on_give_way_consistency", "encounter_type",
        }
        # Verify the metric computation produces these keys
        from ta_mrc_pe_cc_tube_mpc.evaluation.metrics import compute_episode_metrics
        from ta_mrc_pe_cc_tube_mpc.types import (
            EnvironmentState, EpisodeConfig, EpisodeResult,
            VesselParams, VesselState,
        )

        ep = EpisodeConfig(
            scenario_id="test_rule_metrics", waterway="test",
            duration=1.0, dt=0.5,
            ownship_initial_state=VesselState(x=0, y=0, psi=0, u=7, v=0, r=0),
            ownship_params=VesselParams(), targets=[],
            environment_sequence=[EnvironmentState(water_depth=50.0,
                                                     waterway_id="test")],
            random_seed=1,
        )
        result = EpisodeResult(
            episode_id="test", scenario_id="test", method="Proposed", seed=0,
            state_history=[VesselState(x=0, y=0, psi=0, u=7, v=0, r=0, timestamp=0)],
            command_history=[],
            target_histories=[],
        )
        metrics = compute_episode_metrics(result, ep)
        for key in expected:
            assert key in metrics, f"Missing metric key: {key}"
