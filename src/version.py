"""Canonical DocMind release version resolution."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_DISTRIBUTION_NAME = "docmind_ai_llm"


def get_version() -> str:
    """Return installed metadata version or the source checkout version."""
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        try:
            project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
            return str(project["version"])
        except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError):
            return "0+unknown"


__all__ = ["get_version"]
