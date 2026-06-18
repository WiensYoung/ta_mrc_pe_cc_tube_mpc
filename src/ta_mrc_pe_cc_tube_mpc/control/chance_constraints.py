"""Chance-constrained safety formulation.

Implements the conservative deterministic reformulation:
    ||mu_rel_ij(t+k)|| >= d_safe_eff_ij(t+k)
                         + kappa_epsilon * sqrt(lambda_max(Sigma_rel_ij(t+k)))
                         + rho_tube,k

where:
    Sigma_rel = Sigma_j + Sigma_i
    kappa_epsilon = sqrt(chi2.ppf(1 - epsilon, df=2))
"""

from typing import Optional

import numpy as np

from ..risk.uncertainty import (
    compute_chance_constraint_margin,
    compute_relative_covariance,
    get_epsilon_for_waterway,
    inflate_covariance_delay,
)
from ..types import EnvironmentState, TargetShip, VesselState
from ..utils.coordinates import body_to_world


class ChanceConstraintEvaluator:
    """Evaluates chance-constrained safety margins for all target ships."""

    def __init__(self, config: dict = None):
        """
        Args:
            config: Controller configuration dict.
        """
        cfg = config or {}
        self.epsilon_default = cfg.get("epsilon_default", 0.10)
        self.epsilon_restricted = cfg.get("epsilon_restricted", 0.05)
        self.epsilon_tss_ferry = cfg.get("epsilon_tss_ferry", 0.025)
        self.epsilon_tanker_lowvis = cfg.get("epsilon_tanker_lowvis", 0.01)
        self.Sigma_own = cfg.get("own_position_std", 1.0)**2 * np.eye(2)

    def evaluate(
        self,
        own_state: VesselState,
        own_Sigma: Optional[np.ndarray],
        targets: list[TargetShip],
        d_safe_eff_list: list[float],
        rho_tube: float,
        env: EnvironmentState,
        waterway_type: str = "normal",
    ) -> list[dict]:
        """Evaluate chance constraints for all target ships.

        Args:
            own_state: Ownship state.
            own_Sigma: Ownship position covariance (2x2), if None uses default.
            targets: List of target ships.
            d_safe_eff_list: Effective safe distance for each target [m].
            rho_tube: Total tube radius [m].
            env: Environment state.
            waterway_type: Waterway classification for epsilon selection.

        Returns:
            list of per-target evaluation dicts.
        """
        if own_Sigma is None:
            own_Sigma = self.Sigma_own

        # Use instance epsilon values (configurable via adaptive_config) rather
        # than the hardcoded get_epsilon_for_waterway() mapping, so that
        # apply_adaptive_config's epsilon modifications actually take effect.
        mapping = {
            "normal": self.epsilon_default,
            "restricted": self.epsilon_restricted,
            "tss_ferry": self.epsilon_tss_ferry,
            "tanker_lowvis": self.epsilon_tanker_lowvis,
        }
        epsilon = mapping.get(waterway_type, self.epsilon_default)
        own_pos = own_state.position()

        results = []
        for i, target in enumerate(targets):
            # Get observed (noisy) target state or true state
            tgt_pos = target.state.position()

            # Relative position
            mu_rel = tgt_pos - own_pos

            # Relative covariance
            # Default covariance: std=5m (reasonable for AIS-tracked targets).
            # Old value was 100.0 (std=10m) which overinflated the uncertainty margin.
            tgt_cov = target.covariance if target.covariance is not None else np.eye(2) * 25.0
            Sigma_rel = compute_relative_covariance(own_Sigma, tgt_cov)

            # Inflate for AIS delay / dropout
            ais_cfg = target.ais_quality
            delay = ais_cfg.get("delay", 0.0)
            if delay > 0:
                tgt_vel = body_to_world(target.state.u, target.state.v, target.state.psi)
                Sigma_rel = inflate_covariance_delay(Sigma_rel, tgt_vel, delay)

            d_safe = d_safe_eff_list[i] if i < len(d_safe_eff_list) else 500.0

            # Compute margin
            margin = compute_chance_constraint_margin(
                mu_rel, Sigma_rel, d_safe, epsilon, rho_tube,
            )
            margin["target_mmsi"] = target.mmsi
            margin["epsilon"] = epsilon
            results.append(margin)

        return results

    def evaluate_single(
        self,
        own_pos: np.ndarray,
        own_Sigma: np.ndarray,
        target_pos: np.ndarray,
        target_Sigma: np.ndarray,
        d_safe_eff: float,
        rho_tube: float,
        epsilon: float = None,
    ) -> dict:
        """Evaluate chance constraint for a single target.

        Args:
            own_pos: Ownship position [x, y].
            own_Sigma: Ownship covariance (2x2).
            target_pos: Target position [x, y].
            target_Sigma: Target covariance (2x2).
            d_safe_eff: Effective safe distance [m].
            rho_tube: Tube radius [m].
            epsilon: Violation probability (uses default if None).

        Returns:
            Margin evaluation dict.
        """
        if epsilon is None:
            epsilon = self.epsilon_default
        mu_rel = target_pos[:2] - own_pos[:2]
        Sigma_rel = compute_relative_covariance(own_Sigma, target_Sigma)
        return compute_chance_constraint_margin(
            mu_rel, Sigma_rel, d_safe_eff, epsilon, rho_tube,
        )
