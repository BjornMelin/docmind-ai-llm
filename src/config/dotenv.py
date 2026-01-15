"""Dotenv path resolution for DocMind.

Pydantic Settings `env_file` lookup is limited to the current working directory.
DocMind resolves the repository root (via `pyproject.toml`) and uses an absolute
path when opting into dotenv loading, so `streamlit run` works from subdirs and
tests can isolate writes by changing the working directory.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path | None:
    """Return the nearest ancestor directory containing `pyproject.toml`."""
    try:
        current = (start or Path.cwd()).resolve()
    except OSError:
        current = start or Path.cwd()

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def resolve_dotenv_path(start: Path | None = None) -> Path:
    """Resolve the canonical `.env` path for this repo.

    If we can locate a repo root, return `<repo_root>/.env`. Otherwise, fall back
    to `<cwd>/.env`.
    """
    root = find_repo_root(start)
    if root is not None:
        return root / ".env"
    return (start or Path.cwd()) / ".env"
