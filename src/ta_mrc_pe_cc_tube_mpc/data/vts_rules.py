"""VTS/VMRS zone data structures and Coast Pilot rule integration.

Defines geographic zones for Vessel Traffic Service (VTS) areas, Vessel Movement
Reporting System (VMRS) zones, and Coast Pilot local navigation rules.

These feed into the rule engine as soft constraints (P4/P5 priority levels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Geometric primitives ──────────────────────────────────────────────────────

def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _distance_to_line_segment(
    x: float, y: float, x1: float, y1: float, x2: float, y2: float,
) -> float:
    """Minimum distance from point (x, y) to line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return float(np.sqrt((x - x1) ** 2 + (y - y1) ** 2))
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return float(np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2))


def _line_crossed(
    x_prev: float, y_prev: float,
    x_curr: float, y_curr: float,
    x1: float, y1: float, x2: float, y2: float,
) -> bool:
    """Check if the path from prev to curr crosses the line segment."""
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    p1 = (x_prev, y_prev)
    p2 = (x_curr, y_curr)
    p3 = (x1, y1)
    p4 = (x2, y2)

    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    # Collinear edge cases
    eps = 1e-9
    if abs(d1) < eps:
        return _point_on_segment(p1, p3, p4)
    if abs(d2) < eps:
        return _point_on_segment(p2, p3, p4)
    if abs(d3) < eps:
        return _point_on_segment(p3, p1, p2)
    if abs(d4) < eps:
        return _point_on_segment(p4, p1, p2)
    return False


def _point_on_segment(p, a, b):
    """Check if point p lies on segment a-b."""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class VTSReportingLine:
    """A VTS reporting line that vessels must call when crossing.

    Attributes:
        name: VTS call sign or reporting point name (e.g. "TANGO", "WHISKEY").
        x1, y1, x2, y2: Line segment endpoints in local ENU coordinates [m].
        vhf_channel: VHF radio channel for reporting.
        call_direction: Direction that triggers reporting:
            "both" (default), "inbound", "outbound".
        reporting_rule: Description of what to report (e.g. "position_and_intent").
        priority: COLREGS rule priority level (P3, P4, P5).
    """
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    vhf_channel: int = 13
    call_direction: str = "both"
    reporting_rule: str = "position_and_intent"
    priority: str = "P4"

    def to_dict(self) -> dict:
        return {
            "name": self.name, "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2, "vhf_channel": self.vhf_channel,
            "call_direction": self.call_direction,
            "reporting_rule": self.reporting_rule, "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VTSReportingLine":
        return cls(**{k: d[k] for k in [
            "name", "x1", "y1", "x2", "y2", "vhf_channel",
            "call_direction", "reporting_rule", "priority",
        ] if k in d})

    def check_crossing(
        self,
        x_prev: float, y_prev: float,
        x_curr: float, y_curr: float,
        heading_deg: float = 0.0,
    ) -> Optional[dict]:
        """Check if ownship crossed this reporting line in this time step.

        Returns dict with report info if crossed, None otherwise.
        """
        if not _line_crossed(x_prev, y_prev, x_curr, y_curr,
                             self.x1, self.y1, self.x2, self.y2):
            return None

        # Check direction constraint using line normal
        if self.call_direction in ("inbound", "outbound"):
            # Line direction vector and normal (rotated 90 deg CCW)
            dx = self.x2 - self.x1
            dy = self.y2 - self.y1
            # Motion vector
            mx = x_curr - x_prev
            my = y_curr - y_prev
            # Cross product: positive = moving to left of directed line
            cross = dx * my - dy * mx
            desired_positive = (self.call_direction == "inbound")
            if (cross > 0) != desired_positive:
                return None  # wrong crossing direction

        return {
            "line": self.name,
            "vhf_channel": self.vhf_channel,
            "rule": self.reporting_rule,
            "priority": self.priority,
        }


@dataclass
class VTSZone:
    """A VTS or VMRS operational zone.

    Attributes:
        name: Zone name (e.g. "Puget Sound VTS", "San Francisco VTS").
        waterway_id: Parent waterway.
        boundary: List of (x, y) polygon vertices [m] in local coordinates.
        speed_limit_kn: Speed limit within zone [kn], if any.
        min_cpa_m: Minimum CPA for encounters within zone [m].
        mandatory_pilotage: Whether pilotage is mandatory.
        restricted_areas: List of (name, polygon) tuples for no-go sub-zones.
        reporting_lines: List of VTSReportingLine objects.
        priority: COLREGS rule priority level.
    """
    name: str
    waterway_id: str
    boundary: list[tuple[float, float]] = field(default_factory=list)
    speed_limit_kn: Optional[float] = None
    min_cpa_m: Optional[float] = None
    mandatory_pilotage: bool = False
    restricted_areas: list[tuple[str, list[tuple[float, float]]]] = field(default_factory=list)
    reporting_lines: list[VTSReportingLine] = field(default_factory=list)
    priority: str = "P4"

    def to_dict(self) -> dict:
        return {
            "schema_version": "vts_zone_1.0",
            "name": self.name, "waterway_id": self.waterway_id,
            "boundary": [[float(x), float(y)] for x, y in self.boundary],
            "speed_limit_kn": self.speed_limit_kn,
            "min_cpa_m": self.min_cpa_m,
            "mandatory_pilotage": self.mandatory_pilotage,
            "restricted_areas": [
                [name, [[float(x), float(y)] for x, y in poly]]
                for name, poly in self.restricted_areas
            ],
            "reporting_lines": [rl.to_dict() for rl in self.reporting_lines],
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VTSZone":
        return cls(
            name=d["name"], waterway_id=d.get("waterway_id", ""),
            boundary=[(float(p[0]), float(p[1])) for p in d.get("boundary", [])],
            speed_limit_kn=d.get("speed_limit_kn"),
            min_cpa_m=d.get("min_cpa_m"),
            mandatory_pilotage=d.get("mandatory_pilotage", False),
            restricted_areas=[
                (name, [(float(p[0]), float(p[1])) for p in poly])
                for name, poly in d.get("restricted_areas", [])
            ],
            reporting_lines=[VTSReportingLine.from_dict(r) for r in d.get("reporting_lines", [])],
            priority=d.get("priority", "P4"),
        )

    def contains(self, x: float, y: float) -> bool:
        """Check if point (x, y) is inside the VTS zone."""
        if not self.boundary:
            return False
        return _point_in_polygon(x, y, self.boundary)

    def in_restricted_area(self, x: float, y: float) -> Optional[str]:
        """Return restricted area name if point is inside one, else None."""
        for name, poly in self.restricted_areas:
            if _point_in_polygon(x, y, poly):
                return name
        return None

    def get_applicable_rules(self, x: float, y: float) -> list[dict]:
        """Return list of applicable VTS rules at position (x, y)."""
        rules = []
        if not self.contains(x, y):
            return rules

        rules.append({
            "type": "vts_zone",
            "zone": self.name,
            "priority": self.priority,
        })

        if self.speed_limit_kn is not None:
            rules.append({
                "type": "speed_limit",
                "limit_kn": self.speed_limit_kn,
                "priority": self.priority,
            })

        if self.min_cpa_m is not None:
            rules.append({
                "type": "min_cpa",
                "distance_m": self.min_cpa_m,
                "priority": self.priority,
            })

        restricted = self.in_restricted_area(x, y)
        if restricted:
            rules.append({
                "type": "restricted_area",
                "area": restricted,
                "priority": "P3",
            })

        return rules


@dataclass
class CoastPilotRule:
    """A local navigation rule from Coast Pilot publications.

    These are "soft" constraints that apply only in specific geographic areas
    or under specific conditions.

    Attributes:
        rule_id: Unique identifier.
        description: Human-readable rule description.
        waterway_id: Parent waterway.
        zone: List of (x, y) polygon where the rule applies [m].
        rule_type: One of: speed_limit, no_anchoring, bridge_schedule,
            sound_signal, right_of_way_override, tow_restriction, etc.
        value: Rule-specific value (e.g. speed in knots, time window).
        active_condition: Optional condition string (e.g. "daylight_only",
            "ebb_tide", "visibility<1nm").
        priority: COLREGS priority level (P4 or P5).
    """
    rule_id: str
    description: str
    waterway_id: str
    zone: list[tuple[float, float]] = field(default_factory=list)
    rule_type: str = "speed_limit"
    value: float = 0.0
    active_condition: Optional[str] = None
    priority: str = "P4"

    def applies_at(self, x: float, y: float) -> bool:
        """Check if this rule applies at position (x, y)."""
        if not self.zone:
            return True  # global rule
        return _point_in_polygon(x, y, self.zone)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        return {
            "schema_version": "coast_pilot_1.0",
            "type": self.rule_type,
            "rule_id": self.rule_id,
            "description": self.description,
            "waterway_id": self.waterway_id,
            "zone": [[float(x), float(y)] for x, y in self.zone],
            "value": self.value,
            "active_condition": self.active_condition,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoastPilotRule":
        return cls(
            rule_id=d["rule_id"],
            description=d.get("description", ""),
            waterway_id=d.get("waterway_id", ""),
            zone=[(float(p[0]), float(p[1])) for p in d.get("zone", [])],
            rule_type=d.get("type", d.get("rule_type", "speed_limit")),
            value=float(d.get("value", 0.0)),
            active_condition=d.get("active_condition"),
            priority=d.get("priority", "P4"),
        )


# ── Predefined VTS zones ──────────────────────────────────────────────────────

def build_puget_sound_vts() -> VTSZone:
    """Build the Puget Sound VTS zone with reporting lines.

    Puget Sound VTS covers the Strait of Juan de Fuca, Puget Sound, and
    adjacent waters. Operated by USCG Sector Puget Sound.
    """
    # Approximate VTS boundary (local meters from ref -122.75, 48.15)
    # The actual VTS area covers a large region; these are simplified polygons
    boundary = [
        (-40000, 60000),   # Approx: Strait of Juan de Fuca entrance
        (10000, 60000),
        (10000, -40000),   # Southern Puget Sound
        (-40000, -40000),
    ]

    # Key reporting lines
    reporting_lines = [
        VTSReportingLine(
            name="TANGO", x1=-35000, y1=55000, x2=-30000, y2=55000,
            vhf_channel=14, call_direction="inbound",
            reporting_rule="position_and_intent",
        ),
        VTSReportingLine(
            name="WHISKEY", x1=-35000, y1=48000, x2=-30000, y2=48000,
            vhf_channel=14, call_direction="inbound",
            reporting_rule="position_and_intent",
        ),
    ]

    return VTSZone(
        name="Puget Sound VTS",
        waterway_id="puget_sound",
        boundary=boundary,
        reporting_lines=reporting_lines,
        priority="P4",
    )


def build_san_francisco_vts() -> VTSZone:
    """Build the San Francisco VTS zone.

    San Francisco VTS covers the offshore approach, Golden Gate, and bay.
    Operated by USCG Sector San Francisco (VTS San Francisco).
    """
    boundary = [
        (-30000, 30000),
        (10000, 30000),
        (10000, -20000),
        (-30000, -20000),
    ]

    reporting_lines = [
        VTSReportingLine(
            name="SEA_BUOY", x1=-25000, y1=25000, x2=-20000, y2=25000,
            vhf_channel=12, call_direction="inbound",
            reporting_rule="position_and_eta",
        ),
        VTSReportingLine(
            name="GOLDEN_GATE", x1=-5000, y1=10000, x2=-5000, y2=15000,
            vhf_channel=12, call_direction="both",
            reporting_rule="position_and_intent",
        ),
    ]

    return VTSZone(
        name="San Francisco VTS",
        waterway_id="san_francisco_bay",
        boundary=boundary,
        reporting_lines=reporting_lines,
        priority="P4",
    )


def build_new_york_vts() -> VTSZone:
    """Build the New York Harbor VTS zone.

    New York VTS covers the harbor approaches, Ambrose Channel, and
    Kill Van Kull / East River.
    """
    boundary = [
        (-25000, 25000),
        (15000, 25000),
        (15000, -15000),
        (-25000, -15000),
    ]

    reporting_lines = [
        VTSReportingLine(
            name="AMBROSE", x1=-20000, y1=20000, x2=-15000, y2=20000,
            vhf_channel=11, call_direction="inbound",
            reporting_rule="position_and_eta",
        ),
    ]

    return VTSZone(
        name="New York VTS",
        waterway_id="new_york_harbor",
        boundary=boundary,
        speed_limit_kn=12.0,
        reporting_lines=reporting_lines,
        priority="P4",
    )


# ── Predefined Coast Pilot rules ─────────────────────────────────────────────

def build_coast_pilot_rules(waterway_id: str) -> list[CoastPilotRule]:
    """Return predefined Coast Pilot rules for a waterway.

    These are simplified representations of actual US Coast Pilot regulations.
    """
    rules: list[CoastPilotRule] = []

    if waterway_id == "puget_sound":
        rules.extend([
            CoastPilotRule(
                rule_id="PS-CP-001",
                description="Vessels >300 GT must participate in Puget Sound VTS",
                waterway_id=waterway_id,
                zone=[],
                rule_type="vts_participation",
                value=300.0,
                priority="P4",
            ),
            CoastPilotRule(
                rule_id="PS-CP-002",
                description="Speed limit 12 kn in Rosario Strait narrows",
                waterway_id=waterway_id,
                zone=[(-15000, 30000), (-10000, 30000), (-10000, 35000), (-15000, 35000)],
                rule_type="speed_limit",
                value=12.0,
                active_condition="daylight_only",
                priority="P4",
            ),
            CoastPilotRule(
                rule_id="PS-CP-003",
                description="No anchoring in ferry lanes (Seattle-Bainbridge, Edmonds-Kingston)",
                waterway_id=waterway_id,
                zone=[(-5000, -5000), (5000, -5000), (5000, 5000), (-5000, 5000)],
                rule_type="no_anchoring",
                value=0.0,
                priority="P4",
            ),
            CoastPilotRule(
                rule_id="PS-CP-004",
                description="Washington State Ferries have right of way in designated crossing areas",
                waterway_id=waterway_id,
                zone=[],
                rule_type="right_of_way_override",
                value=0.0,
                priority="P4",
            ),
        ])

    elif waterway_id in ("new_york_harbor", "new_york_harbor_nj"):
        rules.extend([
            CoastPilotRule(
                rule_id="NY-CP-001",
                description="Speed limit 10 kn in Kill Van Kull and East River",
                waterway_id=waterway_id,
                zone=[(-5000, -5000), (5000, -5000), (5000, 5000), (-5000, 5000)],
                rule_type="speed_limit",
                value=10.0,
                priority="P4",
            ),
            CoastPilotRule(
                rule_id="NY-CP-002",
                description="Mandatory pilotage for foreign-flag vessels >1000 GT",
                waterway_id=waterway_id,
                zone=[],
                rule_type="mandatory_pilotage",
                value=1000.0,
                priority="P4",
            ),
            CoastPilotRule(
                rule_id="NY-CP-003",
                description="No overtaking in Ambrose Channel (narrows section)",
                waterway_id=waterway_id,
                zone=[(-15000, 15000), (-10000, 15000), (-10000, 20000), (-15000, 20000)],
                rule_type="no_overtaking",
                value=0.0,
                priority="P4",
            ),
        ])

    elif waterway_id == "san_francisco_bay":
        rules.extend([
            CoastPilotRule(
                rule_id="SF-CP-001",
                description="Speed limit 15 kn within SF Bay (inside Golden Gate)",
                waterway_id=waterway_id,
                zone=[(-5000, -5000), (10000, -5000), (10000, 15000), (-5000, 15000)],
                rule_type="speed_limit",
                value=15.0,
                priority="P4",
            ),
            CoastPilotRule(
                rule_id="SF-CP-002",
                description="Mandatory pilotage for vessels >300 GT in SF Bay",
                waterway_id=waterway_id,
                zone=[],
                rule_type="mandatory_pilotage",
                value=300.0,
                priority="P4",
            ),
            CoastPilotRule(
                rule_id="SF-CP-003",
                description="VTS participation required for vessels >300 GT in offshore precautionary area",
                waterway_id=waterway_id,
                zone=[],
                rule_type="vts_participation",
                value=300.0,
                priority="P4",
            ),
        ])

    return rules


# ── Rule evaluation helpers ───────────────────────────────────────────────────

def evaluate_vts_rules(
    x: float, y: float,
    vts_zone: Optional[VTSZone],
    coast_pilot_rules: Optional[list[CoastPilotRule]] = None,
) -> list[dict]:
    """Evaluate all VTS and Coast Pilot rules applicable at position (x, y).

    Args:
        x, y: Ownship position in local coordinates [m].
        vts_zone: Active VTS zone (or None).
        coast_pilot_rules: List of Coast Pilot rules (or None).

    Returns:
        List of applicable rule dicts, each with:
            - type: Rule type string.
            - priority: P3, P4, or P5.
            - (type-specific fields): e.g. limit_kn, distance_m, description.
    """
    applicable: list[dict] = []

    if vts_zone is not None:
        applicable.extend(vts_zone.get_applicable_rules(x, y))

    if coast_pilot_rules:
        for rule in coast_pilot_rules:
            if rule.applies_at(x, y):
                applicable.append(rule.to_dict())

    return applicable


def check_reporting_line_crossings(
    x_prev: float, y_prev: float,
    x_curr: float, y_curr: float,
    heading_deg: float,
    vts_zone: Optional[VTSZone],
) -> list[dict]:
    """Check for VTS reporting line crossings during this time step.

    Returns list of crossing reports (empty if no crossings).
    """
    if vts_zone is None or not vts_zone.reporting_lines:
        return []

    reports = []
    for rl in vts_zone.reporting_lines:
        result = rl.check_crossing(x_prev, y_prev, x_curr, y_curr, heading_deg)
        if result is not None:
            reports.append(result)
    return reports
