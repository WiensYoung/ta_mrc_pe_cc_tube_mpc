"""Perturbation-enhanced episode generator.

Takes real AIS encounter fragments and generates perturbation variants
by injecting controlled noise into initial states, target parameters,
environmental conditions, and AIS quality metadata.

This is critical for statistical validity: one real encounter produces
N perturbed episodes, and statistical tests must cluster by real episode ID.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np

from ..types import (
    EnvironmentState,
    EpisodeConfig,
    TargetShip,
    VesselParams,
    VesselState,
)


def perturb_episode(
    episode: EpisodeConfig,
    perturbation_seed: int,
    position_noise_std: float = 50.0,
    speed_noise_std: float = 0.5,
    heading_noise_deg: float = 3.0,
    ais_delay_range: tuple[float, float] = (0.0, 10.0),
    current_noise_std: float = 0.2,
    wind_variation_kn: float = 3.0,
) -> EpisodeConfig:
    """Create a perturbed variant of an episode.

    Applies controlled noise to:
      - Ownship initial position and velocity
      - Target initial positions and velocities
      - AIS quality parameters (delay, dropout probability)
      - Environment current and wind fields
      - Covariance matrices

    Args:
        episode: Base EpisodeConfig (typically from real AIS data).
        perturbation_seed: Seed for reproducible perturbation.
        position_noise_std: Std dev of position perturbation [m].
        speed_noise_std: Std dev of speed perturbation [m/s].
        heading_noise_deg: Std dev of heading perturbation [deg].
        ais_delay_range: (min, max) AIS delay range [s].
        current_noise_std: Std dev of current component perturbation [m/s].
        wind_variation_kn: Range of wind speed variation [kn].

    Returns:
        New EpisodeConfig with perturbations applied.
    """
    rng = np.random.default_rng(perturbation_seed)
    new_ep = deepcopy(episode)

    # Perturb ownship initial state
    own = new_ep.ownship_initial_state
    noise_x = rng.normal(0, position_noise_std)
    noise_y = rng.normal(0, position_noise_std)
    noise_psi = np.deg2rad(rng.normal(0, heading_noise_deg))
    noise_u = rng.normal(0, speed_noise_std)
    noise_v = rng.normal(0, speed_noise_std * 0.5)

    own.x += noise_x
    own.y += noise_y
    own.psi += noise_psi
    own.u = max(0.1, own.u + noise_u)
    own.v += noise_v

    # Perturb target ships
    for tgt in new_ep.targets:
        tgt.state.x += rng.normal(0, position_noise_std)
        tgt.state.y += rng.normal(0, position_noise_std)
        tgt.state.psi += np.deg2rad(rng.normal(0, heading_noise_deg))
        tgt.state.u = max(0.1, tgt.state.u + rng.normal(0, speed_noise_std))
        tgt.state.v += rng.normal(0, speed_noise_std * 0.5)

        # Perturb AIS quality
        if tgt.ais_quality:
            delay_min, delay_max = ais_delay_range
            tgt.ais_quality["delay"] = rng.uniform(delay_min, delay_max)
            tgt.ais_quality["dropout_prob"] = rng.uniform(0.0, 0.2)
            tgt.ais_quality["position_noise_std"] = rng.uniform(5.0, 20.0)
        else:
            tgt.ais_quality = {
                "delay": rng.uniform(*ais_delay_range),
                "dropout_prob": rng.uniform(0.0, 0.2),
                "position_noise_std": rng.uniform(5.0, 20.0),
            }

        # Perturb covariance
        if tgt.covariance is not None:
            cov_scale = rng.uniform(0.5, 2.0)
            tgt.covariance = tgt.covariance * cov_scale

    # Perturb environment with a constant offset per episode
    # (ocean currents and wind change on timescales of hours, not seconds)
    current_dx = rng.normal(0, current_noise_std)
    current_dy = rng.normal(0, current_noise_std)
    wind_kn = wind_variation_kn * 0.514444
    wind_dx = rng.normal(0, wind_kn)
    wind_dy = rng.normal(0, wind_kn)
    for env in new_ep.environment_sequence:
        env.current_x += current_dx
        env.current_y += current_dy
        env.wind_x += wind_dx
        env.wind_y += wind_dy

    # Update metadata
    new_ep.random_seed = perturbation_seed
    new_ep.perturbation_config = {
        "source": "perturbation",
        "parent_scenario_id": episode.scenario_id,
        "perturbation_seed": perturbation_seed,
        "position_noise_std": position_noise_std,
        "speed_noise_std": speed_noise_std,
    }
    # Preserve navigation rule data from parent episode
    new_ep.enc_layer = episode.enc_layer
    new_ep.vts_zone = episode.vts_zone
    new_ep.coast_pilot_rules = episode.coast_pilot_rules

    return new_ep


def generate_perturbed_episodes(
    base_episodes: list[EpisodeConfig],
    perturbations_per_episode: int = 10,
    base_seed: int = 1000,
    position_noise_std: float = 50.0,
    speed_noise_std: float = 0.5,
) -> list[EpisodeConfig]:
    """Generate perturbation-enhanced episode set from base episodes.

    For N real base episodes and M perturbations per episode,
    produces N × M total episodes. Each perturbed episode records
    its parent (real_episode_id) for proper clustered statistics.

    Args:
        base_episodes: List of real AIS-derived base episodes.
        perturbations_per_episode: Number of perturbation variants per base.
        base_seed: Starting random seed.
        position_noise_std: Position perturbation std [m].
        speed_noise_std: Speed perturbation std [m/s].

    Returns:
        List of perturbed EpisodeConfig objects.
    """
    perturbed = []
    for i, base_ep in enumerate(base_episodes):
        for p in range(perturbations_per_episode):
            seed = base_seed + i * 1000 + p
            pert_ep = perturb_episode(
                base_ep,
                perturbation_seed=seed,
                position_noise_std=position_noise_std,
                speed_noise_std=speed_noise_std,
            )
            # Tag with parent for clustered statistics
            pert_ep.scenario_id = f"{base_ep.scenario_id}_P{p:02d}"
            pert_ep.perturbation_config["real_episode_id"] = base_ep.scenario_id
            pert_ep.perturbation_config["perturbation_index"] = p
            perturbed.append(pert_ep)
    return perturbed


def build_episode_batch_from_ais(
    ais_episodes: list[EpisodeConfig],
    target_count: int = 100,
    perturbations_per_real: int = 10,
    position_noise_std: float = 50.0,
    speed_noise_std: float = 0.5,
    random_seed: int = 42,
) -> list[EpisodeConfig]:
    """Build a batch of ≥target_count episodes from AIS encounters.

    If there aren't enough real encounters, reuses them with different
    perturbation seeds. Always produces at least target_count episodes.

    Args:
        ais_episodes: List of real AIS EpisodeConfig objects.
        target_count: Desired minimum number of episodes.
        perturbations_per_real: Max perturbations per real episode.
        position_noise_std, speed_noise_std: Perturbation magnitudes.
        random_seed: Base random seed.

    Returns:
        List of at least target_count EpisodeConfig objects.
    """
    rng = np.random.default_rng(random_seed)

    if len(ais_episodes) == 0:
        return []

    episodes = []
    real_idx = 0
    pert_idx = 0
    seed_counter = random_seed * 1000

    while len(episodes) < target_count:
        base_ep = ais_episodes[real_idx % len(ais_episodes)]
        pert_ep = perturb_episode(
            base_ep,
            perturbation_seed=seed_counter,
            position_noise_std=position_noise_std,
            speed_noise_std=speed_noise_std,
        )
        pert_ep.scenario_id = f"{base_ep.scenario_id}_P{pert_idx:03d}"
        pert_ep.perturbation_config["real_episode_id"] = base_ep.scenario_id
        pert_ep.perturbation_config["perturbation_index"] = pert_idx
        episodes.append(pert_ep)

        real_idx += 1
        pert_idx += 1
        seed_counter += 1

    return episodes


def build_synthetic_episode_batch(
    scenario_config: dict,
    scenario_id: str,
    n_episodes: int = 100,
    position_noise_std: float = 100.0,
    speed_noise_std: float = 0.5,
    heading_noise_deg: float = 5.0,
    base_seed: int = 42,
) -> list[EpisodeConfig]:
    """Build a batch of perturbed episodes from a synthetic scenario template.

    For scenarios defined in YAML configs (scenarios_core.yaml), this
    creates N perturbed variants by injecting noise into the template.

    Args:
        scenario_config: Parsed YAML scenario configuration dict.
        scenario_id: Scenario identifier (e.g. 'S1').
        n_episodes: Number of perturbed episodes to generate.
        position_noise_std, speed_noise_std, heading_noise_deg: Noise levels.
        base_seed: Base random seed.

    Returns:
        List of EpisodeConfig objects.
    """
    from .episode_builder import build_episode_from_scenario

    episodes = []
    for i in range(n_episodes):
        seed = base_seed + i
        ep = build_episode_from_scenario(scenario_config, scenario_id, seed)
        # L7 fix (2026-06-10): Validate target count matches YAML definition.
        # Perturbation should never change the number of target ships.
        expected_n = len(scenario_config["scenarios"][scenario_id].get("targets", []))
        actual_n = len(ep.targets)
        if actual_n != expected_n:
            import logging
            _log = logging.getLogger(__name__)
            _log.error(
                "Target count mismatch for %s: expected %d, got %d. "
                "Episode may be corrupted.",
                scenario_id, expected_n, actual_n,
            )
        ep = perturb_episode(
            ep,
            perturbation_seed=seed,
            position_noise_std=position_noise_std,
            speed_noise_std=speed_noise_std,
            heading_noise_deg=heading_noise_deg,
        )
        ep.scenario_id = f"{scenario_id}_seed{seed}"
        ep.perturbation_config["real_episode_id"] = scenario_id
        ep.perturbation_config["perturbation_index"] = i
        episodes.append(ep)

    return episodes
