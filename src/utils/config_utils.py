"""
YAML configuration loader with key-presence validation.

Design decision: Config is loaded once by the pipeline orchestrator and
passed explicitly to each stage (dependency injection). No module loads
config at import time — that pattern breaks unit tests and makes
modules impossible to use outside the pipeline context.
"""

import os
from typing import Any

import yaml

from src.utils.exceptions import ConfigurationError


def load_config(config_path: str) -> dict[str, Any]:
    """
    Loads and returns a YAML config file as a dict.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed configuration dict.

    Raises:
        ConfigurationError: If the file is missing or unparseable.
    """
    if not os.path.exists(config_path):
        raise ConfigurationError(f"Config file not found: '{config_path}'")

    with open(config_path, "r") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse config '{config_path}': {e}") from e


def get_required(config: dict[str, Any], *keys: str) -> Any:
    """
    Traverses nested config keys and returns the value.
    Raises ConfigurationError if any key in the chain is missing.

    Args:
        config: The loaded config dict.
        *keys: Sequence of nested keys, e.g. get_required(cfg, "paths", "raw_data").

    Returns:
        The value at the specified key path.

    Raises:
        ConfigurationError: If any key is absent.

    Example:
        raw_dir = get_required(config, "paths", "raw_data")
    """
    node = config
    path = ""
    for key in keys:
        path = f"{path}.{key}" if path else key
        if not isinstance(node, dict) or key not in node:
            raise ConfigurationError(f"Missing required config key: '{path}'")
        node = node[key]
    return node
