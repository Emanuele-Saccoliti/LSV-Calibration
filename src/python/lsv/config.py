"""Validated, deterministic YAML configuration loading."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

import yaml

ConfigScalar: TypeAlias = str | int | float | bool | None
ConfigValue: TypeAlias = (
    ConfigScalar | tuple["ConfigValue", ...] | Mapping[str, "ConfigValue"]
)


def _freeze(value: object, location: str) -> ConfigValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return tuple(_freeze(item, f"{location}[]") for item in value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"Configuration keys at {location} must be strings")
        ordered = {
            key: _freeze(value[key], f"{location}.{key}") for key in sorted(value)
        }
        return MappingProxyType(ordered)
    raise ValueError(
        f"Unsupported configuration value at {location}: {type(value).__name__}"
    )


def load_config(path: str | Path) -> Mapping[str, ConfigValue]:
    """Load a YAML mapping into a recursively immutable, key-sorted structure.

    The loader accepts only YAML safe-loader scalar, sequence, and mapping values.
    Environment interpolation is deliberately unsupported so identical files yield
    identical configurations across repeated runs.
    """

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root in {config_path} must be a mapping")
    frozen = _freeze(raw, "root")
    if not isinstance(frozen, Mapping):
        raise AssertionError("Configuration freezing violated its mapping contract")
    return frozen
