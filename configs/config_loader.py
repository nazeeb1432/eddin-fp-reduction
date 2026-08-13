"""Utility for loading YAML config files into plain dicts or attribute-accessible objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_DIR / path
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Config:
    """Attribute-accessible wrapper around a config dict, with nested dict support."""

    _data: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def to_dict(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return its contents as a Config object."""
    return Config(load_yaml(path))
