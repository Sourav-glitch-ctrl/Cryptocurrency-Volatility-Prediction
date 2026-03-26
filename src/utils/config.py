"""
config.py
---------
Loads project configuration from a YAML file.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(path: str = None) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : Path to the YAML file. Defaults to ``config/config.yaml``
           relative to the project root.

    Returns
    -------
    dict  — parsed configuration.

    Raises
    ------
    FileNotFoundError  — if the config file does not exist.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config or {}
