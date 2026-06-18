"""Multi-source rule engine.

Integrates COLREGs (Rules 6, 8, 13-19), Inland Rules, waterway-specific rules,
and multi-rule priority hierarchy P0-P5 into a single RuleAssessment.

Enhanced with:
- Rule 18: Complete vessel responsibility hierarchy (NUC > RAM > CBD > fishing > sailing > power)
- Rule 8:  Verification of "substantial and readily apparent" avoiding action
- Rule 6:  Structured safe-speed assessment framework
- Multi-vessel pairwise consistency check
- Auto-detection of Inland Rules via ENC geographic data

Key behaviors:
- P0/P1 are always hard constraints
- P2-P4 can be hard or soft based on configuration
- P5 contributes to objective function only
- Conflicts are logged, not silently overwritten
"""

from typing import Optional

import numpy as np

from ..risk.encounter_classifier import classify_encounter
from ..types import EnvironmentState, RuleAssessment, TargetShip, VesselParams, VesselState
from .colregs import colregs_action
from .colregs_enhanced import (
    assess_safe_speed,
    check_multi_vessel_consistency,
    resolve_responsibility,
    verify_action_substantial,
)
from .inland_rules import is_inland_waterway
from .rule_priority import (
    PriorityLevel,
    RuleConflictLogger,
    get_rule_priority_level,
    get_rule_weight,
    is_hard_constraint,
)
from .waterway_rules import (
    check_atba,
    check_channel_boundary,
    check_precautionary_area,
    check_tss_compliance,
    compute_waterway_safety_scale,
)

# Optional VTS/Coast Pilot integration
try:
    from ..data.vts_rules import evaluate_vts_rules, check_reporting_line_crossings
    _HAS_VTS = True
except ImportError:
    _HAS_VTS = False


class RuleEngine:
    """Multi-source rule assessment engine.

    Processes ownship state, target ships, environment, and waterway
    context to produce a comprehensive RuleAssessment.
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: Rules configuration (from configs/rules.yaml).
        """
        self.config = config or {}
        self.conflict_logger = RuleConflictLogger()
        self.priorities_config = self.config.get("priorities", {})
        self._last_enc_layer = None  # stored by assess() for _should_use_inland_rules

    def assess(
        self,
        own_state: VesselState,
        own_params: VesselParams,
        target_ships: list[TargetShip],
        env: EnvironmentState,
        enc_layer=None,
        visibility: float = 10.0,   # [nm] visibility
        tss_config: dict = None,
        vts_zone=None,
        coast_pilot_rules: list = None,
    ) -> RuleAssessment:
        """Perform full multi-rule assessment.

        Args:
            own_state: Current ownship state.
            own_params: Ownship parameters.
            target_ships: List of observed target ships.
            env: Current environment state.
            enc_layer: Optional ENC layer for zone queries.
            visibility: Visibility range [nm].
            tss_config: Optional TSS configuration.
            vts_zone: Optional VTSZone for VTS/VMRS rules.
            coast_pilot_rules: Optional list of CoastPilotRule objects.

        Returns:
            RuleAssessment with all constraints and recommendations.
        """
        # Store enc_layer for _should_use_inland_rules
        self._last_enc_layer = enc_layer
        assessment = RuleAssessment()
        hard = []
        soft = []
        forbidden = []

        # --- P0: Inadmissible space constraints ---
        self._assess_p0(own_state, own_params, env, enc_layer, hard, forbidden)

        # --- P1: Immediate safety constraints ---
        self._assess_p1(own_state, own_params, target_ships, env, hard)

        # --- P2: Encounter rules ---
        encounter_type = "none"
        ownship_role = "none"
        safety_domain_scale = 1.0

        if target_ships:
            encounter_type, ownship_role, rule_constraints = self._assess_p2(
                own_state, own_params, target_ships, env, hard, soft
            )
            if rule_constraints:
                assessment.target_priority = self._compute_target_priority(own_state, target_ships)

        # --- P3: Traffic organization ---
        self._assess_p3(own_state, env, tss_config, hard, soft)

        # --- P4: Local management advice ---
        self._assess_p4(own_state, env, vts_zone, coast_pilot_rules, soft)

        # --- P5: Performance optimization (cost terms only) ---
        p5_terms = self._assess_p5(own_state)
        assessment.p5_terms = p5_terms

        # --- Aggregate ---
        # Waterway safety scale (additive, capped at 1.5 inside compute_waterway_safety_scale)
        safety_domain_scale = compute_waterway_safety_scale(
            env.waterway_id, env.channel_width,
        )

        # Check for restricted visibility — additive increment
        if visibility < 2.0:
            safety_domain_scale += 0.10

        # Check for restricted maneuverability targets — additive increment
        for tgt in target_ships:
            if tgt.nav_status in ("restricted_maneuverability", "not_under_command"):
                safety_domain_scale += 0.10

        # Final cap to prevent constraint infeasibility
        safety_domain_scale = min(safety_domain_scale, 1.5)

        assessment.encounter_type = encounter_type
        assessment.ownship_role = ownship_role
        assessment.rule_priority = self._highest_active_priority(hard)
        assessment.hard_constraints = hard
        assessment.soft_constraints = soft
        assessment.forbidden_zones = forbidden
        assessment.safety_domain_scale = safety_domain_scale

        # Determine preferred maneuver
        assessment.preferred_maneuver = self._determine_preferred_maneuver(
            encounter_type, ownship_role, hard, soft
        )

        return assessment

    def _assess_p0(self, own_state, own_params, env, enc_layer, hard, forbidden):
        """P0: Inadmissible space constraints."""
        # Land check
        if enc_layer is not None:
            if not enc_layer.is_navigable(own_state.x, own_state.y, min_depth=0.5):
                hard.append({
                    "priority": "P0",
                    "rule": "land_or_non_navigable",
                    "type": "hard",
                    "violated": True,
                })
                forbidden.append(("position", "land"))

        # Depth check
        if env.water_depth < own_params.draught + 1.0:  # UKC < 1m
            hard.append({
                "priority": "P0",
                "rule": "non_navigable_depth",
                "type": "hard",
                "violated": True,
                "ukc": env.water_depth - own_params.draught,
            })

    def _assess_p1(self, own_state, own_params, target_ships, env, hard):
        """P1: Immediate safety constraints."""
        # Bank clearance check
        if env.bank_distance_left is not None:
            min_bank_clearance = 2.0 * own_params.length  # conservative
            if env.bank_distance_left < min_bank_clearance:
                hard.append({
                    "priority": "P1",
                    "rule": "minimum_bank_clearance",
                    "type": "hard",
                    "violated": True,
                    "bank": "left",
                    "distance": env.bank_distance_left,
                    "required": min_bank_clearance,
                })
        if env.bank_distance_right is not None:
            min_bank_clearance = 2.0 * own_params.length
            if env.bank_distance_right < min_bank_clearance:
                hard.append({
                    "priority": "P1",
                    "rule": "minimum_bank_clearance",
                    "type": "hard",
                    "violated": True,
                    "bank": "right",
                    "distance": env.bank_distance_right,
                    "required": min_bank_clearance,
                })

    def _assess_p2(self, own_state, own_params, target_ships, env, hard, soft):
        """P2: Encounter responsibility rules.

        Enhanced with:
        - Rule 18: Complete vessel responsibility hierarchy
        - Rule 8:  Action substantiality verification
        - Rule 6:  Safe speed assessment
        - Multi-vessel pairwise consistency check
        """
        encounter_type = "none"
        ownship_role = "none"
        constraints = []
        all_colregs_actions = []  # for multi-vessel consistency check

        for target in target_ships:
            # Skip anchored / moored targets - handled separately
            if target.nav_status in ("at_anchor", "moored", "aground"):
                hard.append({
                    "priority": "P1",
                    "rule": "static_obstacle_avoidance",
                    "type": "hard",
                    "target_mmsi": target.mmsi,
                    "target_status": target.nav_status,
                })
                continue

            # ── Rule 18: Check vessel responsibility hierarchy ──────────
            resp = resolve_responsibility(
                own_nav_status="underway",
                own_vessel_type="cargo",  # ownship is a cargo vessel by default
                target_nav_status=target.nav_status,
                target_vessel_type=target.vessel_type,
            )

            # If Rule 18 requires ownship to give way unconditionally
            if resp["own_must_give_way"]:
                ownship_role = "give_way"
                hard.append({
                    "priority": "P2",
                    "rule": "rule18_give_way_hierarchy",
                    "type": "hard",
                    "target_mmsi": target.mmsi,
                    "target_status": target.nav_status,
                    "own_rank": resp["own_rank"],
                    "target_rank": resp["target_rank"],
                    "rule18_basis": resp["rule18_basis"],
                    "action": {
                        "required_course_change": 20.0,
                        "encounter_type": "rule18_giveway",
                        "action_description": (
                            "Rule 18 give-way: alter course to keep clear of "
                            f"{target.nav_status} vessel"
                        ),
                    },
                })
                # Do NOT continue — still run standard COLREGs for directional
                # guidance.  The hard constraint above takes priority; the
                # COLREGs action below provides a fallback direction.

            # If ownship is privileged by Rule 18
            if resp["own_is_privileged"]:
                # Ownship is stand-on but must still monitor (Rule 17)
                soft.append({
                    "priority": "P2",
                    "rule": "rule18_privileged_vessel",
                    "type": "soft",
                    "target_mmsi": target.mmsi,
                    "own_rank": resp["own_rank"],
                    "target_rank": resp["target_rank"],
                    "rule18_basis": resp["rule18_basis"],
                    "note": "Ownship has priority but must maintain proper lookout (Rule 5) and take action if collision risk develops (Rule 17(b))",
                })
                # Don't skip standard COLREGs — may still have encounter rules

            # ── Auto-detect Inland Rules via ENC geographic data ────────
            is_inland = self._should_use_inland_rules(env, own_state)

            # ── Standard COLREGs classification ─────────────────────────
            if is_inland:
                from .colregs import inland_rules_action
                # Attach target length to state copy so inland_rules_action
                # can read it via getattr (VesselState lacks length/beam).
                import copy as _copy
                tgt_state = _copy.copy(target.state)
                object.__setattr__(tgt_state, 'length', target.length)
                object.__setattr__(tgt_state, 'beam', target.beam)
                action = inland_rules_action(
                    own_state, tgt_state, own_params,
                    env=env,
                    channel_width=env.channel_width,
                    downbound="western_rivers" in env.waterway_id.lower(),
                )
            else:
                action = colregs_action(own_state, target.state, own_params)

            etype = action["encounter_type"]
            role = action["ownship_role"]

            if etype != "none":
                # Track the most-critical encounter (head_on > crossing > overtaking)
                rank_order = {"head_on": 4, "crossing_giveway": 3, "crossing_standon": 3,
                             "overtaking_giveway": 2, "overtaking_standon": 1, "none": 0}
                if rank_order.get(etype, 0) > rank_order.get(encounter_type, 0):
                    encounter_type = etype
                    ownship_role = role  # update role when encounter type changes
                elif rank_order.get(etype, 0) == rank_order.get(encounter_type, 0):
                    # Same rank: prefer "give_way" (more conservative) over "stand_on" or "none"
                    if role == "give_way":
                        ownship_role = "give_way"
                    elif role == "stand_on" and ownship_role not in ("give_way",):
                        ownship_role = "stand_on"

                # ── Rule 8: Verify action is substantial ────────────────
                course_change = action.get("required_course_change", 0.0)
                speed_change_frac = 0.0
                if action.get("required_speed_change", 0.0) != 0:
                    speed_change_frac = action["required_speed_change"] / max(own_state.speed(), 0.1)
                rule8_check = verify_action_substantial(
                    encounter_type=etype,
                    course_change_deg=course_change,
                    speed_change_fraction=speed_change_frac,
                    own_speed=own_state.speed(),
                )
                action["rule8_verification"] = rule8_check

                if not rule8_check["all_substantial"] and role != "stand_on":
                    soft.append({
                        "priority": "P2",
                        "rule": "rule8_action_not_substantial",
                        "type": "soft",
                        "target_mmsi": target.mmsi,
                        "issues": rule8_check["issues"],
                    })

                all_colregs_actions.append(action)

            if action.get("required_course_change", 0) != 0:
                constraint = {
                    "priority": "P2",
                    "rule": f"colregs_{etype}",
                    "type": "hard" if is_hard_constraint("colregs", self.priorities_config) else "soft",
                    "target_mmsi": target.mmsi,
                    "action": action,
                }
                if constraint["type"] == "hard":
                    hard.append(constraint)
                else:
                    soft.append(constraint)
                constraints.append(constraint)

        # ── Multi-vessel consistency check ──────────────────────────
        if len(all_colregs_actions) >= 2:
            mv_check = check_multi_vessel_consistency(
                own_state, target_ships, all_colregs_actions
            )
            if not mv_check["consistent"]:
                for conflict in mv_check["conflicts"]:
                    soft.append({
                        "priority": "P2",
                        "rule": "multi_vessel_colregs_conflict",
                        "type": "soft",
                        "conflict": conflict,
                        "resolution": mv_check.get("recommended_resolution", ""),
                    })

        # ── Rule 6: Safe speed assessment ────────────────────────────
        safe_speed = assess_safe_speed(
            own_params=own_params,
            visibility_nm=getattr(env, "visibility_nm", 10.0),
            traffic_density=min(len(target_ships) / 10.0, 1.0),
            water_depth_m=env.water_depth,
            bank_proximity=(env.bank_distance_left is not None and env.bank_distance_left < 5 * own_params.length) or
                          (env.bank_distance_right is not None and env.bank_distance_right < 5 * own_params.length),
            num_nearby_targets=len([t for t in target_ships
                                   if np.linalg.norm(t.state.position() - own_state.position()) < 3704]),  # 2nm
            ukc_m=env.water_depth - own_params.draught,
        )
        if safe_speed.safe_speed_factor < 1.0:
            soft.append({
                "priority": "P2",
                "rule": "rule6_safe_speed",
                "type": "soft",
                "safe_speed_factor": safe_speed.safe_speed_factor,
                "reasons": safe_speed.speed_reduction_reasons,
            })

        return encounter_type, ownship_role, constraints

    def _should_use_inland_rules(self, env: EnvironmentState, own_state=None) -> bool:
        """Determine whether Inland Rules should apply.

        Uses ENC geographic data when available, falling back to waterway_id
        string matching. The ENC layer can provide precise geographic boundaries
        for COLREGs demarcation lines (where International Rules transition to
        Inland Rules, per 33 CFR 80).

        Args:
            env: Environment state with waterway_id.
            own_state: Optional ownship state for position-based check via ENC.

        Returns:
            True if Inland Rules apply.
        """
        # Method 1: ENC-based geographic check
        if own_state is not None and hasattr(self, "_last_enc_layer") and self._last_enc_layer is not None:
            enc = self._last_enc_layer
            if hasattr(enc, "is_inland_water") and callable(enc.is_inland_water):
                try:
                    return enc.is_inland_water(own_state.x, own_state.y)
                except Exception:
                    pass

        # Method 2: Waterway ID string matching
        return is_inland_waterway(env.waterway_id)

    @staticmethod
    def _normalize_tss_lane(lane):
        """Convert a raw ENC lane (list of [x,y] coords) to dict with center/half_width."""
        if isinstance(lane, dict):
            return lane
        if isinstance(lane, list) and len(lane) > 0 and isinstance(lane[0], (list, tuple)):
            ys = [pt[1] for pt in lane]
            return {
                "center": float(np.mean(ys)),
                "half_width": float((max(ys) - min(ys)) / 2),
                "separation_half_width": 100.0,
            }
        return None

    @staticmethod
    def _normalize_tss_zone(zone):
        """Convert a raw ENC zone (list of [x,y] coords) to dict with center/radius."""
        if isinstance(zone, dict):
            return zone
        if isinstance(zone, list) and len(zone) > 0 and isinstance(zone[0], (list, tuple)):
            pts = np.array(zone)
            center = pts.mean(axis=0)
            radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
            return {"center": center.tolist(), "radius": radius}
        return None

    def _assess_p3(self, own_state, env, tss_config, hard, soft):
        """P3: Traffic organization constraints."""
        own_pos = np.array([own_state.x, own_state.y])

        # Channel boundary check
        if env.channel_width is not None:
            half_w = env.channel_width / 2
            channel_check = check_channel_boundary(
                own_state.x, own_state.y, -half_w, half_w
            )
            if not channel_check["within_channel"]:
                hard.append({
                    "priority": "P3",
                    "rule": "channel_boundary",
                    "type": "hard",
                    "violated": True,
                    "detail": channel_check,
                })
            elif channel_check["min_margin"] < 50.0:
                soft.append({
                    "priority": "P3",
                    "rule": "channel_boundary_approach",
                    "type": "soft",
                    "margin": channel_check["min_margin"],
                })

        # TSS lane compliance
        if tss_config:
            for raw_lane in tss_config.get("lanes", []):
                lane = self._normalize_tss_lane(raw_lane)
                if lane is None:
                    continue
                tss_check = check_tss_compliance(
                    own_state.y,
                    lane_center=lane["center"],
                    lane_half_width=lane["half_width"],
                    separation_zone_half_width=lane.get("separation_half_width", 100.0),
                )
                if not tss_check["in_lane"] or tss_check["in_separation_zone"]:
                    hard.append({
                        "priority": "P3",
                        "rule": "tss_lane_compliance",
                        "type": "hard",
                        "violated": True,
                        "detail": tss_check,
                    })
                elif tss_check["margin_to_lane_boundary"] < 100.0:
                    soft.append({
                        "priority": "P3",
                        "rule": "tss_lane_approach",
                        "type": "soft",
                        "margin": tss_check["margin_to_lane_boundary"],
                    })

            # ATBA (Area To Be Avoided) check
            for raw_atba in tss_config.get("atba_zones", []):
                atba = self._normalize_tss_zone(raw_atba)
                if atba is None:
                    continue
                atba_check = check_atba(
                    own_pos,
                    atba_center=np.array(atba["center"]),
                    atba_radius=atba["radius"],
                )
                if atba_check["intruding"]:
                    hard.append({
                        "priority": "P3",
                        "rule": "atba_intrusion",
                        "type": "hard",
                        "violated": True,
                        "detail": atba_check,
                    })

            # Precautionary area check
            for raw_area in tss_config.get("precautionary_areas", []):
                area = self._normalize_tss_zone(raw_area)
                if area is None:
                    continue
                pa_check = check_precautionary_area(
                    own_pos,
                    area_center=np.array(area["center"]),
                    area_radius=area["radius"],
                )
                if pa_check["in_area"]:
                    soft.append({
                        "priority": "P3",
                        "rule": "precautionary_area_presence",
                        "type": "soft",
                        "detail": pa_check,
                    })

    def _assess_p4(self, own_state, env, vts_zone, coast_pilot_rules, soft):
        """P4: Local traffic management advice — VTS, Coast Pilot."""
        # Evaluate VTS/Coast Pilot rules if available
        if _HAS_VTS and vts_zone is not None:
            vts_rules = evaluate_vts_rules(
                own_state.x, own_state.y, vts_zone, coast_pilot_rules,
            )
            for rule in vts_rules:
                soft.append({
                    "priority": rule.get("priority", "P4"),
                    "rule": rule.get("type", "vts_rule"),
                    "type": "soft",
                    "description": rule.get("description", ""),
                    "value": rule.get("value", 0.0),
                    "zone": rule.get("zone", ""),
                })
        else:
            # Fallback: generic VTS monitoring advisory
            soft.append({
                "priority": "P4",
                "rule": "vts_monitoring",
                "type": "soft",
                "waterway": env.waterway_id,
            })

    def _assess_p5(self, own_state) -> list:
        """P5: Performance optimization terms."""
        return [
            {"priority": "P5", "rule": "smoothness", "weight": get_rule_weight("smoothness")},
            {"priority": "P5", "rule": "energy", "weight": get_rule_weight("energy")},
            {"priority": "P5", "rule": "comfort", "weight": get_rule_weight("comfort")},
            {"priority": "P5", "rule": "route_tracking", "weight": get_rule_weight("route_tracking")},
        ]

    def _compute_target_priority(
        self, own_state: VesselState, targets: list[TargetShip]
    ) -> float:
        """Compute aggregated target priority score."""
        if not targets:
            return 0.0
        # Based on closest CPA and encounter type
        scores = []
        for tgt in targets:
            d = float(np.linalg.norm(tgt.state.position() - own_state.position()))
            score = 1.0 / max(d, 10.0)
            if tgt.nav_status in ("restricted_maneuverability", "not_under_command"):
                score *= 2.0
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0

    def _highest_active_priority(self, hard_constraints: list) -> str:
        """Determine the highest active priority level."""
        if not hard_constraints:
            return "P5"
        priorities = []
        for c in hard_constraints:
            p_name = str(c.get("priority", "P5"))
            try:
                if p_name.startswith("P") and len(p_name) >= 2:
                    priorities.append(int(p_name[1]))
                else:
                    priorities.append(5)  # default to lowest
            except (ValueError, IndexError):
                priorities.append(5)
        return f"P{min(priorities)}" if priorities else "P5"

    def _determine_preferred_maneuver(
        self, encounter_type: str, ownship_role: str,
        hard: list, soft: list,
    ) -> Optional[str]:
        """Determine preferred maneuver from rule assessment."""
        if encounter_type in ("head_on",):
            return "starboard_turn"
        if encounter_type in ("crossing_giveway",):
            return "starboard_turn"
        if encounter_type in ("overtaking_giveway",):
            return "alter_course_away"
        if encounter_type == "none" and ownship_role == "give_way":
            return "give_way_general"
        return None
