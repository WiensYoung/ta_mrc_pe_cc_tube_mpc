"""Target-vessel-aware dynamic ship domain — additive scalar model.

Implements the additive effective safety distance:
    d_safe_eff_ij = d_base_ij + Δd_speed_ij + Δd_size_ij + Δd_rule_ij
                  + Δd_unc_ij + Δd_env_ij + Δd_bank_i + Δd_ship_ij
                  + Δd_waterway_i

.. important::

   This is a **scalar additive** model — it computes a single isotropic
   safety distance per target ship.  It is NOT a full elliptical,
   quaternion, or heading-dependent ship domain (Fujii, Goodwin, etc.).
   The output is a scalar radius in meters, used as a circular constraint
   in the MPC optimization.

   For papers and documentation, refer to this as the "additive scalar
   safe distance" or "scalar dynamic safety domain" — do NOT claim
   directional / elliptical domain unless that feature is implemented.

All terms are dimensionally consistent (output in meters).
Configurable via YAML parameters.
"""

from typing import Optional

import numpy as np

from ..types import EnvironmentState, TargetShip, VesselParams, VesselState
from ..utils.coordinates import distance


class DynamicShipDomain:
    """Computes the effective safety distance between ownship and a target ship.

    All additive terms are individually tracked for sensitivity analysis.
    """

    def __init__(self, config: dict = None):
        """
        Args:
            config: Domain parameters dict (from configs/vessel.yaml -> domain).
        """
        cfg = config or {}
        # Base domain factors
        self.base_length_factor = cfg.get("base_length_factor", 2.0)
        self.base_width_factor = cfg.get("base_width_factor", 4.0)
        # Speed sensitivity
        self.alpha_v = cfg.get("alpha_v", 0.15)
        # Size sensitivity
        self.alpha_sz = cfg.get("alpha_sz", 0.1)
        # Rule sensitivity
        self.alpha_rule_giveway = cfg.get("alpha_rule_giveway", 0.5)
        # Stand-on vessel: no safety distance reduction.  The old default (-0.2)
        # reduced d_safe by up to 72m for two 180m vessels, which could produce
        # dangerously small margins when combined with other terms.
        self.alpha_rule_standon = cfg.get("alpha_rule_standon", 0.0)
        # Uncertainty
        self.alpha_unc = cfg.get("alpha_unc", 0.3)
        # Environment
        self.alpha_env = cfg.get("alpha_env", 0.2)
        # Bank effect
        self.beta_0 = cfg.get("beta_0", 0.5)
        self.beta_1 = cfg.get("beta_1", 2.0)
        self.beta_2 = cfg.get("beta_2", 0.1)
        self.beta_3 = cfg.get("beta_3", 1.0)
        # Ship interaction
        self.alpha_s = cfg.get("alpha_s", 0.2)
        # Waterway
        self.alpha_waterway = cfg.get("alpha_waterway", 0.15)
        # Vessel type weights (override from config)
        self.vessel_weights = cfg.get("vessel_weights", {
            "cargo": 1.0, "tanker": 1.3, "ferry": 0.9,
            "tug_barge": 1.2, "small_craft": 0.5,
        })
        # Reference values
        self.U_ref = cfg.get("U_ref", 7.0)
        self.d_ref = cfg.get("d_ref", 1000.0)
        self.d_min = cfg.get("d_min", 10.0)
        # Safety clamping — prevents runaway amplification when ships are close
        self.d_ship_amp_max = cfg.get("d_ship_amp_max", 5.0)  # max d_ref/d_ij ratio
        self.narrow_factor_max = cfg.get("narrow_factor_max", 0.5)  # max waterway narrow_factor
        self.max_domain = cfg.get("max_domain", 5000.0)  # hard cap on total [m]

    def compute(
        self,
        own_state: VesselState,
        own_params: VesselParams,
        target: TargetShip,
        env: EnvironmentState,
        encounter_role: str = "none",
        ais_delay: float = 0.0,
        ais_dropout: bool = False,
        ship_interaction_risk: float = 0.0,
        bank_effect_active: bool = False,
        enable_target_size: bool = True,
        enable_speed_sensitive: bool = True,
        enable_vessel_type_weight: bool = True,
        enable_behavior_uncertainty: bool = True,
        enable_environment: bool = True,
        enable_bank_effect: bool = True,
        enable_ship_interaction: bool = True,
        enable_waterway: bool = True,
    ) -> dict:
        """Compute d_safe_eff_ij with all additive terms.

        Returns a dict with each term and the total, enabling per-term
        sensitivity analysis and ablation.

        Args:
            own_state: Ownship state.
            own_params: Ownship parameters.
            target: Target ship.
            env: Environment state.
            encounter_role: 'give_way', 'stand_on', or 'none'.
            ais_delay: AIS reporting delay [s].
            ais_dropout: Whether dropout occurred.
            ship_interaction_risk: Ship interaction risk score [0, 1].
            bank_effect_active: Whether bank effect is relevant.
            enable_*: Feature flags for ablation.

        Returns:
            dict with keys for each term and 'total' for d_safe_eff_ij.
        """
        L_i = own_params.length
        B_i = own_params.beam
        L_j = target.length
        B_j = target.beam
        T_i = own_params.draught

        # Current distance
        d_ij = distance(own_state.position(), target.state.position())

        # Relative speed (vector velocity difference)
        own_speed = own_state.speed()
        tgt_speed = target.state.speed()
        own_vel = own_state.velocity()
        tgt_vel = target.state.velocity()
        U_rel_ij = np.linalg.norm(own_vel - tgt_vel)
        # Heading difference for head-on detection
        dpsi = own_state.psi - target.state.psi
        dpsi = np.arctan2(np.sin(dpsi), np.cos(dpsi))  # normalize to [-pi, pi]

        # Closing speed: for head-on encounters (180° opposing), closing ≈ own+tgt.
        # Detection: heading difference > 150° means near-opposite headings.
        # IMPORTANT: encounter_role values are "give_way", "stand_on", "none" —
        # the "head_on" encounter TYPE maps to role "give_way" in the classifier.
        # We detect head-on geometry by role=give_way AND large heading difference.
        is_head_on = (encounter_role == "give_way" and abs(dpsi) > np.deg2rad(150))
        U_closing_ij = own_speed + tgt_speed if is_head_on else U_rel_ij

        # Vessel type weight
        w_type = self.vessel_weights.get(target.vessel_type, 1.0)

        # --- Compute each term ---
        terms = {}

        # d_base: base domain from ship dimensions
        terms["d_base"] = self.base_length_factor * (L_i + L_j) if enable_target_size else self.base_length_factor * L_i

        # Δd_speed: speed-dependent inflation
        terms["d_speed"] = 0.0
        if enable_speed_sensitive:
            terms["d_speed"] = self.alpha_v * (L_i + L_j) * (U_closing_ij / self.U_ref)

        # Δd_size: additional size-based margin
        terms["d_size"] = 0.0
        if enable_target_size:
            terms["d_size"] = self.alpha_sz * (B_i + B_j)

        # Δd_rule: rule-based adjustment
        terms["d_rule"] = 0.0
        if encounter_role == "give_way":
            terms["d_rule"] = self.alpha_rule_giveway * (L_i + L_j)
        elif encounter_role == "stand_on":
            terms["d_rule"] = self.alpha_rule_standon * (L_i + L_j)

        # Δd_unc: uncertainty inflation
        terms["d_unc"] = 0.0
        if enable_behavior_uncertainty:
            delay_factor = 1.0 + ais_delay / 30.0
            dropout_factor = 2.0 if ais_dropout else 1.0
            terms["d_unc"] = self.alpha_unc * (L_i + L_j) * delay_factor * dropout_factor

        # Δd_env: environment (wind/current)
        terms["d_env"] = 0.0
        if enable_environment:
            env_mag = np.sqrt(env.current_x**2 + env.current_y**2
                              + env.wind_x**2 + env.wind_y**2)
            terms["d_env"] = self.alpha_env * (L_i + L_j) * (env_mag / self.U_ref)

        # Δd_bank: bank effect safety margin
        terms["d_bank"] = 0.0
        if enable_bank_effect and bank_effect_active:
            # Use explicit None checks — env.bank_distance_* can be 0.0
            # (ship exactly at the bank), which is a valid real value
            # that must not be conflated with None via falsy evaluation.
            d_left = env.bank_distance_left if env.bank_distance_left is not None else float("inf")
            d_right = env.bank_distance_right if env.bank_distance_right is not None else float("inf")
            d_bank = min(d_left, d_right)
            I_shallow = 1.0 if env.water_depth / max(T_i, 1e-3) < 4.0 else 0.0
            d_bank_safe = L_i * (
                self.beta_0
                + self.beta_1 * (B_i / L_i)
                + self.beta_2 * (own_speed / self.U_ref)**2
                + self.beta_3 * I_shallow
            )
            terms["d_bank"] = max(0.0, d_bank_safe - min(d_bank, d_bank_safe))

        # Δd_ship: ship-ship interaction margin
        # IMPORTANT: d_ref/d_ij is clamped to [0.1, d_ship_amp_max] to prevent
        # runaway amplification when ships approach closely (prevents km-scale
        # safety distances for 180m vessels — see Issue #5 in audit report).
        terms["d_ship"] = 0.0
        if enable_ship_interaction:
            proximity_ratio = np.clip(self.d_ref / max(d_ij, self.d_min),
                                      0.1, self.d_ship_amp_max)
            terms["d_ship"] = (
                self.alpha_s * (L_i + L_j)
                * (U_rel_ij / self.U_ref)
                * proximity_ratio
                * ship_interaction_risk
            )

        # Δd_waterway: waterway-specific adjustment
        terms["d_waterway"] = 0.0
        if enable_waterway and env.channel_width is not None:
            narrow_factor = np.clip(1.0 - env.channel_width / (20 * L_i),
                                    0.0, self.narrow_factor_max)
            terms["d_waterway"] = self.alpha_waterway * L_i * narrow_factor

        # Vessel type weight multiplier (applied as additive to base, not multiplicative on total)
        terms["d_vessel_type"] = 0.0
        if enable_vessel_type_weight:
            terms["d_vessel_type"] = (w_type - 1.0) * terms["d_base"]

        # --- Total ---
        total = sum(terms.values())
        terms["total_raw"] = max(0.0, total)
        terms["total"] = terms["total_raw"]

        # Cap at 2× base domain to prevent constraint explosion.
        # The base domain already accounts for ship sizes (2×(L_i+L_j)),
        # so 2× base = 4×(L_i+L_j) is a generous upper bound for all
        # dynamic margins combined.  This prevents the multiplicative
        # stacking of speed/uncertainty/rule/interaction margins from
        # producing infeasible safety distances (e.g. 1440m for 180m vessels
        # in 200-400m close-quarters encounter geometries).
        cap = 2.0 * terms["d_base"]
        if terms["total"] > cap:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.debug(
                "DynamicShipDomain total %.0f m exceeds 2×base cap %.0f m — clamping. "
                "Components: %s",
                terms["total"], cap,
                {k: round(v, 1) for k, v in terms.items()
                 if k not in ("total", "total_raw")},
            )
            terms["total"] = cap

        return terms

    def compute_for_all_targets(
        self,
        own_state: VesselState,
        own_params: VesselParams,
        targets: list[TargetShip],
        env: EnvironmentState,
        encounter_roles: list[str] = None,
        **kwargs,
    ) -> list[dict]:
        """Compute d_safe_eff for each target ship.

        Returns a list of per-target result dicts.
        """
        if encounter_roles is None:
            encounter_roles = ["none"] * len(targets)
        results = []
        for i, target in enumerate(targets):
            role = encounter_roles[i] if i < len(encounter_roles) else "none"
            result = self.compute(own_state, own_params, target, env, role, **kwargs)
            result["target_mmsi"] = target.mmsi
            results.append(result)
        return results
