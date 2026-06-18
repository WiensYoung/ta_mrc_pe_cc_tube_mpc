"""Core dataclass types for the TA-MRC-PE-CC-Tube-MPC framework."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class VesselState:
    """3-DOF vessel state in local ENU-like coordinates.

    Attributes:
        x, y: Position [m].
        psi: Heading angle [rad], 0 = East/North convention per scenario.
        u, v: Surge and sway velocities [m/s].
        r: Yaw rate [rad/s].
        timestamp: Simulation time [s].
    """
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    u: float = 0.0
    v: float = 0.0
    r: float = 0.0
    timestamp: float = 0.0

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def velocity(self) -> np.ndarray:
        """Return velocity vector in world frame."""
        c = np.cos(self.psi)
        s = np.sin(self.psi)
        return np.array([
            self.u * c - self.v * s,
            self.u * s + self.v * c,
        ])

    def speed(self) -> float:
        if np.isinf(self.u) or np.isinf(self.v) or np.isnan(self.u) or np.isnan(self.v):
            return 0.0
        val = self.u**2 + self.v**2
        if np.isinf(val) or np.isnan(val):
            return 0.0
        return float(np.sqrt(max(val, 0.0)))


@dataclass
class VesselParams:
    """Vessel physical parameters and operational limits.

    Attributes:
        length: Overall length L_oa [m].
        beam: Beam B [m].
        draught: Draught T [m].
        mass: Displacement mass [kg].
        Iz: Yaw moment of inertia [kg*m^2].
        max_rudder: Maximum rudder angle [deg].
        max_rudder_rate: Maximum rudder rate [deg/s].
        max_speed: Maximum speed [kn].
        min_speed: Minimum speed [kn].
        max_yaw_rate: Maximum yaw rate [deg/s].
        min_turning_radius_multiplier: Minimum turning radius as multiple of L (actual radius = this * L).
    """
    length: float = 180.0
    beam: float = 30.0
    draught: float = 10.5
    mass: float = 5e7
    Iz: float = 1.2e11
    max_rudder: float = 35.0
    max_rudder_rate: float = 5.0
    max_speed: float = 15.0
    min_speed: float = 0.0
    max_yaw_rate: float = 3.0
    min_turning_radius_multiplier: float = 3.0  # multiplier of L (actual radius = this * L)


@dataclass
class TargetShip:
    """Target ship data including state, parameters, and uncertainty.

    Attributes:
        mmsi: Unique identifier.
        state: Current state estimate.
        length, beam, draught: Dimensions [m].
        vessel_type: One of cargo, tanker, ferry, tug_barge, small_craft.
        nav_status: Navigation status string.
        cog: Course over ground [deg].
        sog: Speed over ground [kn].
        heading: Heading [deg], optional.
        covariance: Position covariance matrix [m^2] (2x2 in ENU).
        ais_quality: AIS quality metadata dict.
    """
    mmsi: str = ""
    state: VesselState = field(default_factory=VesselState)
    length: float = 100.0
    beam: float = 15.0
    draught: float = 5.0
    vessel_type: str = "cargo"
    nav_status: str = "underway"
    cog: float = 0.0        # Course Over Ground [deg]
    sog: float = 0.0        # Speed Over Ground [kn]
    heading: Optional[float] = None  # [deg]
    covariance: Optional[np.ndarray] = None
    ais_quality: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.covariance is None:
            self.covariance = np.eye(2) * 100.0  # default 10m std


@dataclass
class EnvironmentState:
    """Environmental and waterway state.

    Attributes:
        water_depth: Local water depth h [m].
        current_x, current_y: Current velocity components [m/s].
        wind_x, wind_y: Wind velocity components [m/s].
        bank_distance_left: Distance to left bank [m], if applicable.
        bank_distance_right: Distance to right bank [m], if applicable.
        channel_width: Channel width [m], if applicable.
        waterway_id: Waterway identifier string.
    """
    water_depth: float = 50.0
    current_x: float = 0.0
    current_y: float = 0.0
    wind_x: float = 0.0
    wind_y: float = 0.0
    bank_distance_left: Optional[float] = None
    bank_distance_right: Optional[float] = None
    channel_width: Optional[float] = None
    waterway_id: str = ""


@dataclass
class RuleAssessment:
    """Output of the multi-source rule engine.

    Attributes:
        encounter_type: e.g. head_on, crossing, overtaking, none.
        ownship_role: give_way, stand_on, or none.
        target_priority: Priority weight for each target ship.
        rule_priority: Highest active rule priority (P0-P5).
        hard_constraints: List of hard constraint descriptors.
        soft_constraints: List of soft constraint descriptors.
        forbidden_zones: List of (geometry, label) tuples.
        preferred_maneuver: Recommended maneuver, if any.
        safety_domain_scale: Multiplier for base safety domain.
        p5_terms: Performance optimization terms (comfort, efficiency, etc.).
    """
    encounter_type: str = "none"
    ownship_role: str = "none"
    target_priority: float = 1.0
    rule_priority: str = "P5"
    hard_constraints: list = field(default_factory=list)
    soft_constraints: list = field(default_factory=list)
    forbidden_zones: list = field(default_factory=list)
    preferred_maneuver: Optional[str] = None
    safety_domain_scale: float = 1.0
    p5_terms: list = field(default_factory=list)


@dataclass
class ControlCommand:
    """Control command output from the controller.

    Attributes:
        rudder: Rudder angle [deg], positive = starboard.
        propeller: Propeller command, normalized [0, 1].
        source: Which controller produced this command.
        cbf_modified: Whether CBF-QP modified the command.
        fallback_active: Whether fallback is active.
        metadata: Additional diagnostic data.
    """
    rudder: float = 0.0
    propeller: float = 0.5
    source: str = ""
    cbf_modified: bool = False
    fallback_active: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class RuleContext:
    """Unified rule context passed from episode through controller to metrics.

    Captures all rule-relevant information at each control step so that
    rule-violation metrics can be computed from actual rule-engine outputs
    rather than post-hoc heuristics.
    """

    # ── Core encounter info ──────────────────────────────────────────────
    own_state: Optional[VesselState] = None
    target_ships: list = field(default_factory=list)
    encounter_type: str = "none"
    own_role: str = "none"               # "give_way", "stand_on", "none"

    # ── Environment / waterway ───────────────────────────────────────────
    enc_layer: Optional[object] = None
    vts_zone: Optional[object] = None
    coast_pilot_rules: Optional[list] = None
    channel_boundary: Optional[Tuple[float, float]] = None  # (left, right) [m]
    shallow_water_info: Optional[dict] = None
    traffic_lane_info: Optional[dict] = None
    waterway_id: str = ""

    # ── Rule-engine outputs ──────────────────────────────────────────────
    active_priorities: list[str] = field(default_factory=list)  # e.g. ["P0","P2","P4"]
    hard_constraint_count: int = 0
    soft_constraint_count: int = 0
    colregs_applied: bool = False
    inland_rules_applied: bool = False
    vts_rules_applied: bool = False
    coast_pilot_applied: bool = False
    rule_conflicts: list[str] = field(default_factory=list)
    rule_exceptions: list[str] = field(default_factory=list)
    forbidden_zone_encroached: bool = False
    channel_boundary_violation: bool = False

    # ── Metadata ─────────────────────────────────────────────────────────
    timestamp: float = 0.0
    scenario_id: str = ""
    visibility_nm: float = 10.0


@dataclass
class EpisodeConfig:
    """Full configuration for a single simulation episode.

    Attributes:
        scenario_id: Scenario identifier.
        waterway: Waterway name.
        duration: Episode duration [s].
        dt: Time step [s].
        ownship_initial_state: Initial ownship state.
        ownship_params: Ownship parameters.
        targets: List of target ships.
        environment_sequence: Time-varying environment states.
        random_seed: Random seed for reproducibility.
        perturbation_config: Perturbation injection config.
    """
    scenario_id: str = ""
    waterway: str = ""
    duration: float = 600.0
    dt: float = 0.5
    ownship_initial_state: VesselState = field(default_factory=VesselState)
    ownship_params: VesselParams = field(default_factory=VesselParams)
    targets: list = field(default_factory=list)   # list[TargetShip]
    environment_sequence: list = field(default_factory=list)  # list[EnvironmentState]
    random_seed: int = 42
    perturbation_config: dict = field(default_factory=dict)
    enc_layer: Optional[object] = None
    vts_zone: Optional[object] = None
    coast_pilot_rules: Optional[list] = None


@dataclass
class EpisodeResult:
    """Output of a completed simulation episode.

    Attributes:
        episode_id: Unique episode identifier.
        scenario_id: Scenario identifier.
        method: Method name (e.g. Proposed, B3, B4).
        seed: Random seed used.
        real_episode_id: ID of the real-AIS-derived parent episode, if synthetic.
        perturbation_id: Perturbation variant index.
        state_history: List of ownship VesselState snapshots.
        command_history: List of ControlCommand snapshots.
        target_histories: List of lists of target VesselState snapshots.
        metrics: Computed performance metrics dict.
        failure_flags: Dict of failure flag booleans.
        failure_type: Primary failure type (F1-F10) or None.
        runtime_stats: Dict of timing information.
        metadata: Additional metadata.
    """
    episode_id: str = ""
    scenario_id: str = ""
    method: str = ""
    seed: int = 0
    real_episode_id: str = ""
    perturbation_id: int = 0
    state_history: list = field(default_factory=list)
    command_history: list = field(default_factory=list)
    target_histories: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    failure_flags: dict = field(default_factory=dict)
    failure_type: Optional[str] = None
    runtime_stats: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class SafetyDistanceBreakdown:
    """Centralized safety distance breakdown for MPC, CBF, and metrics.

    All safety-aware components (MPC collision constraints, CBF barrier
    functions, and evaluation metrics) MUST derive their total safe distance
    from this single source of truth.  Each component is additive.

    The total is:
        total = base_domain + dynamic_margin + chance_margin
              + tube_margin + rule_margin + extra_margin

    The ``total`` property is CAPPED at ``max_total_cap`` (= 2 × base_domain)
    to prevent constraint explosion in close-quarters scenarios.  The uncapped
    raw value is available as ``total_raw`` for evaluation / diagnostics.

    This replaces ad-hoc per-module arithmetic that previously caused
    inconsistent safety sets between MPC, CBF, and metrics reporting.
    """

    base_domain: float = 100.0
    """Base static safety distance (e.g., 4 × ship length)."""

    dynamic_margin: float = 0.0
    """Speed-sensitive, vessel-type, behaviour-uncertainty expansion."""

    chance_margin: float = 0.0
    """Stochastic margin kappa * sqrt(lambda_max(Sigma_rel))."""

    tube_margin: float = 0.0
    """Robust tube radius from physics-enhanced disturbance bounds."""

    rule_margin: float = 0.0
    """Rule-driven safety domain scale factor (P0-P5 priority overlay)."""

    extra_margin: float = 0.0
    """Reserved for scenario-specific or fallback safety expansion."""

    @property
    def max_total_cap(self) -> float:
        """Hard cap on total safety distance = 2 × base_domain.

        Prevents multiplicative stacking from inflating the safety distance
        beyond what is physically feasible in close-quarters encounters.
        """
        return 2.0 * self.base_domain

    @property
    def total_raw(self) -> float:
        """Uncapped total — for evaluation / diagnostics only."""
        return (
            self.base_domain
            + self.dynamic_margin
            + self.chance_margin
            + self.tube_margin
            + self.rule_margin
            + self.extra_margin
        )

    @property
    def total(self) -> float:
        """Capped total — used for MPC / CBF constraints.

        Capped at 2 × base_domain to prevent constraint infeasibility
        in close-quarters scenarios where multiplicative stacking would
        otherwise produce unreachable safety distances (e.g. 1440m for
        180m vessels in 200-400m encounter geometries).
        """
        return min(self.total_raw, self.max_total_cap)

    def to_dict(self) -> dict[str, float]:
        return {
            "base_domain": self.base_domain,
            "dynamic_margin": self.dynamic_margin,
            "chance_margin": self.chance_margin,
            "tube_margin": self.tube_margin,
            "rule_margin": self.rule_margin,
            "extra_margin": self.extra_margin,
            "total_raw": self.total_raw,
            "total": self.total,
            "max_total_cap": self.max_total_cap,
        }
