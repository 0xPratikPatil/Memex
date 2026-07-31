"""YAML config loader with ${VAR} substitution and dot-notation access."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

_log = logging.getLogger(__name__)

_VAR_PATTERN: re.Pattern[str] = re.compile(r"\$\{([^}]+)\}")
_SUBST_MAX_DEPTH: int = 5


def _resolve_env_vars(value: str) -> str:
    """Substitute ``${VAR}`` references with environment variable values.

    Supports nested substitution up to ``_SUBST_MAX_DEPTH`` rounds.
    Missing variables are left as-is.
    """

    def _replace(m: re.Match[str]) -> str:
        env_key = m.group(1)
        return os.environ.get(env_key, m.group(0))

    result = value
    for _ in range(_SUBST_MAX_DEPTH):
        prev = result
        result = _VAR_PATTERN.sub(_replace, result)
        if result == prev:
            break
    return result


def _resolve_recursive(obj: Any) -> Any:
    """Recursively resolve ``${VAR}`` references in dicts, lists, and strings."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_recursive(item) for item in obj]
    return obj


class YamlConfig:
    """YAML config loader with ``${VAR}`` substitution and dot-notation access."""

    def __init__(self, path: str = "config.yaml") -> None:
        self._path = path
        self._data: dict[str, Any] = self._load(path)

    def _load(self, path: str) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            _log.debug("Config file %s not found, using empty config", path)
            return {}
        with p.open() as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            _log.warning("Config file %s did not parse as a mapping, using empty config", path)
            return {}
        resolved: dict[str, Any] = _resolve_recursive(raw)
        return resolved

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Retrieve a nested value using dot-notation (e.g. ``"embedding.model"``)."""
        keys = dotpath.split(".")
        node: Any = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def get_str(self, dotpath: str, default: str = "") -> str:
        val = self.get(dotpath, default)
        return str(val) if val is not None else default

    def get_int(self, dotpath: str, default: int = 0) -> int:
        val = self.get(dotpath)
        if val is None:
            return default
        try:
            return int(val)  # type: ignore[arg-type,no-any-return]
        except (TypeError, ValueError):
            return default

    def get_float(self, dotpath: str, default: float = 0.0) -> float:
        val = self.get(dotpath)
        if val is None:
            return default
        try:
            return float(val)  # type: ignore[arg-type,no-any-return]
        except (TypeError, ValueError):
            return default

    def get_bool(self, dotpath: str, default: bool = False) -> bool:
        val = self.get(dotpath, default)
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("1", "true", "yes")

    def get_list(self, dotpath: str, default: list[Any] | None = None) -> list[Any]:
        """Retrieve a value as a list, wrapping scalars if necessary."""
        val = self.get(dotpath, default)
        if val is None:
            return default or []
        if isinstance(val, list):
            return val
        return [val]

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"<YamlConfig path={self._path!r} keys={sorted(self._data.keys())}>"
