"""Rule priority system P0-P5.

P0: Inadmissible space constraints (HARD)
P1: Immediate safety constraints (HARD)
P2: Encounter responsibility rules (COLREGs, Inland)
P3: Traffic organization constraints (TSS, ATBA, channel)
P4: Local traffic management advice (VTS, Coast Pilot)
P5: Performance optimization (smoothness, energy, comfort)
"""

from enum import Enum
from typing import Optional


class PriorityLevel(Enum):
    P0 = 0  # Highest: inadmissible space
    P1 = 1  # Immediate safety
    P2 = 2  # Encounter rules
    P3 = 3  # Traffic organization
    P4 = 4  # Local advice
    P5 = 5  # Performance


# Priority definitions
PRIORITY_DEFINITIONS = {
    PriorityLevel.P0: {
        "name": "Inadmissible Space Constraints",
        "rules": [
            "land", "bridge_pier", "non_navigable_depth",
            "forbidden_zone", "grounding_risk_area",
        ],
        "is_hard": True,
        "can_override_lower": True,
    },
    PriorityLevel.P1: {
        "name": "Immediate Safety Constraints",
        "rules": [
            "collision_avoidance", "dynamic_safety_domain",
            "cbf_safety_set", "minimum_bank_clearance",
        ],
        "is_hard": True,
        "can_override_lower": True,
    },
    PriorityLevel.P2: {
        "name": "Encounter Responsibility Rules",
        "rules": [
            "colregs", "us_inland_rules",
            "give_way_stand_on_role", "head_on", "crossing", "overtaking",
        ],
        "is_hard": True,   # configurable
        "can_override_lower": True,
    },
    PriorityLevel.P3: {
        "name": "Traffic Organization Constraints",
        "rules": [
            "tss_lane", "separation_zone", "atba", "channel_boundary",
        ],
        "is_hard": True,   # configurable
        "can_override_lower": True,
    },
    PriorityLevel.P4: {
        "name": "Local Traffic Management Advice",
        "rules": [
            "vts_vmrs", "coast_pilot", "local_reporting_line", "recommended_route",
        ],
        "is_hard": False,
        "can_override_lower": True,
    },
    PriorityLevel.P5: {
        "name": "Performance Optimization",
        "rules": [
            "smoothness", "energy", "comfort", "route_tracking",
        ],
        "is_hard": False,
        "can_override_lower": False,
    },
}


class RuleConflictLogger:
    """Logs rule conflicts for transparency and debugging.

    Conflicts are recorded rather than silently overwritten.
    """

    def __init__(self):
        self.conflicts: list[dict] = []

    def log(self, priority_a: str, rule_a: str,
            priority_b: str, rule_b: str,
            resolution: str, reason: str = ""):
        """Record a rule conflict and its resolution."""
        self.conflicts.append({
            "priority_a": priority_a,
            "rule_a": rule_a,
            "priority_b": priority_b,
            "rule_b": rule_b,
            "resolution": resolution,
            "reason": reason,
        })

    def get_conflicts(self) -> list[dict]:
        return self.conflicts

    def clear(self):
        self.conflicts.clear()


def resolve_priority(rule_a: str, rule_b: str) -> str:
    """Determine which rule has higher priority.

    Returns the higher-priority rule name.
    """
    pa = get_rule_priority_level(rule_a)
    pb = get_rule_priority_level(rule_b)
    if pa is None and pb is None:
        return rule_a  # tie
    if pa is None:
        return rule_b
    if pb is None:
        return rule_a
    if pa.value < pb.value:
        return rule_a
    elif pb.value < pa.value:
        return rule_b
    return rule_a  # same level


def get_rule_priority_level(rule_name: str) -> Optional[PriorityLevel]:
    """Get the priority level for a named rule."""
    rule_lower = rule_name.lower()
    for level, info in PRIORITY_DEFINITIONS.items():
        for r in info["rules"]:
            # Exact match or prefix match (rule_name starts with registered rule)
            # Avoids false matches from substring containment (e.g. "my_energy"
            # matching P5's "energy" entry).
            if rule_lower == r or rule_lower.startswith(r + "_") or rule_lower.startswith(r + "."):
                return level
    return None


def is_hard_constraint(rule_name: str, config: dict = None) -> bool:
    """Check if a rule should be treated as a hard constraint."""
    level = get_rule_priority_level(rule_name)
    if level is None:
        return False
    info = PRIORITY_DEFINITIONS[level]
    # P0 and P1 are always hard
    if level in (PriorityLevel.P0, PriorityLevel.P1):
        return True
    # P2-P4 can be configured
    if config and f"P{level.value}" in config:
        val = config[f"P{level.value}"]
        if isinstance(val, dict):
            return val.get("default_hard", info["is_hard"])
    return info["is_hard"]


def get_rule_weight(rule_name: str, base_weights: dict = None) -> float:
    """Get the cost weight for a rule (used for soft constraints and P5)."""
    default_weights = {
        "colregs": 10.0,
        "tss_lane": 5.0,
        "channel_boundary": 5.0,
        "vts_vmrs": 2.0,
        "smoothness": 0.1,
        "energy": 0.05,
        "route_tracking": 0.5,
    }
    weights = {**default_weights, **(base_weights or {})}
    rule_lower = rule_name.lower()
    for key, weight in weights.items():
        if rule_lower == key or rule_lower.startswith(key + "_") or rule_lower.startswith(key + "."):
            return weight
    return 1.0
