"""Tests for rule priority system and rule engine."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ta_mrc_pe_cc_tube_mpc.rules.rule_priority import (
    PriorityLevel,
    get_rule_priority_level,
    is_hard_constraint,
    resolve_priority,
)
from ta_mrc_pe_cc_tube_mpc.rules.rule_engine import RuleEngine
from ta_mrc_pe_cc_tube_mpc.types import EnvironmentState, TargetShip, VesselParams, VesselState


class TestRulePriority:
    """Test P0-P5 rule priority system."""

    def test_p0_overrides_all(self):
        """P0 (land/forbidden zone) always has highest priority."""
        winner = resolve_priority("land", "colregs")
        # "land" should map to P0
        land_level = get_rule_priority_level("land")
        colregs_level = get_rule_priority_level("colregs")
        assert land_level is not None
        assert land_level.value < colregs_level.value

    def test_p1_immediate_safety_over_colregs(self):
        """P1 (collision avoidance, safety domain) overrides COLREGs."""
        p1 = get_rule_priority_level("collision_avoidance")
        p2 = get_rule_priority_level("colregs")
        assert p1 is not None
        assert p1.value < p2.value

    def test_hard_constraint_classification(self):
        """P0 and P1 are always hard constraints."""
        assert is_hard_constraint("land")
        assert is_hard_constraint("collision_avoidance")
        assert is_hard_constraint("dynamic_safety_domain")

    def test_resolve_conflict(self):
        """Higher priority rule wins in conflict resolution."""
        assert resolve_priority("land", "colregs") == "land"
        assert resolve_priority("colregs", "smoothness") == "colregs"

    def test_unknown_rule(self):
        """Unknown rule names return None priority."""
        assert get_rule_priority_level("nonexistent_rule") is None


class TestRuleEngine:
    """Test multi-source rule engine."""

    def setup_method(self):
        self.engine = RuleEngine()
        self.own_params = VesselParams()

    def test_rule_assessment_no_targets(self):
        """Rule assessment with no targets should be safe."""
        own = VesselState()
        env = EnvironmentState(water_depth=50.0)

        assessment = self.engine.assess(own, self.own_params, [], env)

        assert assessment.encounter_type == "none"
        assert assessment.ownship_role == "none"

    def test_anchored_target_handling(self):
        """Anchored/moored targets treated as static obstacles."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        tgt_state = VesselState(x=200, y=0, psi=0, u=0, v=0, r=0)
        target = TargetShip(
            mmsi="ANC001", state=tgt_state,
            length=100, beam=20, draught=5,
            nav_status="at_anchor",
        )
        env = EnvironmentState(water_depth=50.0)

        assessment = self.engine.assess(own, self.own_params, [target], env)

        # Anchored target should generate a P1 hard constraint
        p1_constraints = [c for c in assessment.hard_constraints if c["priority"] == "P1"]
        assert len(p1_constraints) > 0

    def test_not_under_command_target(self):
        """NUC targets force give-way role."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        tgt_state = VesselState(x=200, y=0, psi=np.pi, u=3, v=0, r=0)
        target = TargetShip(
            mmsi="NUC001", state=tgt_state,
            length=100, beam=20, draught=5,
            nav_status="not_under_command",
        )
        env = EnvironmentState(water_depth=50.0)

        assessment = self.engine.assess(own, self.own_params, [target], env)
        assert assessment.ownship_role == "give_way"

    def test_colregs_head_on(self):
        """Head-on encounter classified correctly."""
        own = VesselState(x=0, y=0, psi=0, u=5, v=0, r=0)
        tgt_state = VesselState(x=1000, y=0, psi=np.pi, u=5, v=0, r=0)
        target = TargetShip(
            mmsi="HO001", state=tgt_state,
            length=100, beam=20, draught=5,
            nav_status="underway",
        )
        env = EnvironmentState(water_depth=50.0)

        assessment = self.engine.assess(own, self.own_params, [target], env)

        assert assessment.encounter_type == "head_on"
        assert assessment.ownship_role == "give_way"

    def test_deep_water_no_p0_violation(self):
        """Deep water should not trigger depth-related P0 violation."""
        own = VesselState()
        env = EnvironmentState(water_depth=100.0)

        assessment = self.engine.assess(own, self.own_params, [], env)

        p0_violations = [
            c for c in assessment.hard_constraints
            if c["priority"] == "P0" and c.get("violated")
        ]
        assert len(p0_violations) == 0

    def test_shallow_water_ukc_violation(self):
        """Shallow water with insufficient UKC triggers P0."""
        own = VesselState()
        self.own_params.draught = 10.0
        env = EnvironmentState(water_depth=10.5)  # only 0.5m UKC

        assessment = self.engine.assess(own, self.own_params, [], env)

        p0_violations = [
            c for c in assessment.hard_constraints
            if c["priority"] == "P0" and c.get("violated")
        ]
        assert len(p0_violations) > 0
