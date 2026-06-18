"""Scenario loader: loads and manages scenario configurations."""

from typing import Optional

from ..data.episode_builder import build_episode_from_yaml
from ..types import EpisodeConfig
from ..utils.io_utils import load_yaml


def load_scenario(
    scenario_file: str,
    scenario_id: str,
    seed: int = 42,
) -> EpisodeConfig:
    """Load a scenario from a YAML file and build an EpisodeConfig.

    Args:
        scenario_file: Path to the scenario YAML (e.g. configs/scenarios_core.yaml).
        scenario_id: Scenario identifier (e.g. 'S1').
        seed: Random seed.

    Returns:
        EpisodeConfig.
    """
    return build_episode_from_yaml(scenario_file, scenario_id, seed)


def list_scenarios(scenario_file: str) -> list[str]:
    """List all scenario IDs in a scenario file."""
    config = load_yaml(scenario_file)
    return list(config.get("scenarios", {}).keys())


def load_all_scenarios(
    scenario_file: str,
    seed: int = 42,
) -> list[EpisodeConfig]:
    """Load all scenarios from a file.

    Returns a list of EpisodeConfig objects.
    """
    scenario_ids = list_scenarios(scenario_file)
    return [
        load_scenario(scenario_file, sid, seed + i)
        for i, sid in enumerate(scenario_ids)
    ]
