"""I/O utilities for loading configs and saving results."""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml


def read_csv_safe(path: str, **kwargs):
    """Read a CSV file with a large field size limit.

    Prevents ``_csv.Error: field larger than field limit (131072)`` when
    CSV files contain long string fields (e.g. embedded JSON or trajectory
    data from older pipeline versions).

    Args:
        path: Path to CSV file.
        **kwargs: Extra arguments passed to ``pd.read_csv``.

    Returns:
        pandas DataFrame.
    """
    import pandas as pd
    # Raise limit to handle very large fields (trajectory data, JSON blobs)
    csv.field_size_limit(sys.maxsize)
    return pd.read_csv(path, **kwargs)


def load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str) -> None:
    """Save a dict to a YAML file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def resolve_path(base_dir: str, relative: str) -> str:
    """Resolve a relative path against a base directory."""
    return str(Path(base_dir) / relative)


def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict.

    Uses copy.deepcopy so that mutable leaf values (lists, etc.) in the
    base are not shared with the caller's original dict.
    """
    import copy
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config_with_overrides(
    config_dir: str,
    default_name: str = "default.yaml",
    overrides: Optional[dict] = None,
) -> dict:
    """Load the full configuration with layered overrides.

    Loads default.yaml, then merges vessel.yaml (if present) for domain params,
    then merges any provided override dict.
    """
    config = load_yaml(os.path.join(config_dir, default_name))
    # Merge vessel-specific config for ship domain and physics parameters
    vessel_path = os.path.join(config_dir, "vessel.yaml")
    if os.path.isfile(vessel_path):
        config = deep_merge(config, load_yaml(vessel_path))
    if overrides:
        config = deep_merge(config, overrides)
    return config
