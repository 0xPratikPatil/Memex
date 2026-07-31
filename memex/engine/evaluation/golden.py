"""Golden set loader and matcher."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


def _basename(value: str) -> str:
    """Strip any directory prefix, tolerating both separators on any platform."""
    return value.replace("\\", "/").rsplit("/", 1)[-1]


@dataclass
class GoldenQuery:
    """A single query with expected retrieval results.

    Attributes:
        query: The search query text.
        expected_sources: Identifiers that count as a correct hit.
        expected_keywords: Keywords that should appear in retrieved content.
        category: Optional grouping label (e.g. "financial", "technical").
        difficulty: Optional difficulty level (e.g. "easy", "hard").
        filters: Optional metadata filters to pass to the search function.
    """

    query: str
    expected_sources: list[str]
    expected_keywords: list[str] = field(default_factory=list)
    category: str = ""
    difficulty: str = ""
    filters: dict | None = None


@dataclass
class GoldenSet:
    """A collection of golden queries for evaluation.

    Attributes:
        queries: The golden query entries.
    """

    queries: list[GoldenQuery]

    @classmethod
    def from_yaml(cls, path: str) -> GoldenSet:
        """Load a golden set from a YAML file.

        The file can be a bare list of entries, or a mapping with a ``queries``
        key alongside optional ``match_field`` and ``match_mode`` settings.

        Args:
            path: Path to a .yaml or .yml file.

        Returns:
            The loaded GoldenSet.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the structure is not recognised.
        """
        import yaml

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Golden set not found: {file_path}")

        data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        return cls._from_data(data)

    @classmethod
    def from_json(cls, path: str) -> GoldenSet:
        """Load a golden set from a JSON file.

        Args:
            path: Path to a .json file.

        Returns:
            The loaded GoldenSet.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the structure is not recognised.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Golden set not found: {file_path}")

        data = json.loads(file_path.read_text(encoding="utf-8"))
        return cls._from_data(data)

    @classmethod
    def _from_data(cls, data: object) -> GoldenSet:
        """Build a golden set from already-parsed data.

        Accepts either a bare list of entries or a mapping with a ``queries``
        key.
        """
        if isinstance(data, dict):
            entries = data.get("queries")
            if entries is None:
                raise ValueError("A golden set mapping must have a 'queries' key holding the list of query entries")
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError(
                f"A golden set must be a list or a mapping with a 'queries' key, got {type(data).__name__}"
            )

        queries: list[GoldenQuery] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Golden set entry {index} must be a mapping, got {type(entry).__name__}")

            expected = entry.get("expected_sources") or entry.get("expected")
            if expected is None:
                raise ValueError(f"Golden set entry {index} ({entry.get('query')!r}) has no 'expected_sources' key")
            if isinstance(expected, str):
                expected = [expected]

            queries.append(
                GoldenQuery(
                    query=entry.get("query", ""),
                    expected_sources=expected,
                    expected_keywords=entry.get("expected_keywords", []),
                    category=entry.get("category", ""),
                    difficulty=entry.get("difficulty", ""),
                    filters=entry.get("filters"),
                )
            )

        return cls(queries=queries)


def match_source(expected: str, actual: str, mode: str = "basename") -> bool:
    """Check if an expected source matches an actual source.

    Modes:
        - basename: compare file names only, case-insensitive
        - exact: full path match
        - contains: substring match (expected is a substring of actual)

    Args:
        expected: The expected source identifier.
        actual: The actual source identifier from retrieval.
        mode: Matching strategy.

    Returns:
        True if the sources match under the given mode.

    Raises:
        ValueError: If mode is not recognised.
    """
    if not actual:
        return False

    if mode == "basename":
        return _basename(actual).lower() == _basename(expected).lower()
    if mode == "exact":
        return actual == expected
    if mode == "contains":
        return expected.lower() in actual.lower()

    raise ValueError(f"Unknown match mode: {mode!r}. Available: basename, exact, contains")
