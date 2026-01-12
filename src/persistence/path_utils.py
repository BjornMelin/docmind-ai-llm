"""Shared persistence path validation helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_path_under_data_dir(*, path: Path, data_dir: Path, label: str) -> Path:
    """Resolve a path anchored under data_dir."""
    if path.is_absolute():
        candidate = path.expanduser()
    else:
        # If the caller provides only a filename, anchor it under data_dir.
        # If the caller provides a relative path with directories (e.g. data/foo.db),
        # treat it as relative to the current working directory and validate that it
        # still resolves under data_dir.
        if path.parent == Path("."):
            candidate = (data_dir / path).expanduser()
        else:
            candidate = path.expanduser()
    if candidate.exists() and candidate.is_symlink():
        raise ValueError(f"{label} may not be a symlink (got {candidate})")
    resolved = candidate.resolve()
    data_dir_resolved = data_dir.resolve()
    if not resolved.is_relative_to(data_dir_resolved):
        raise ValueError(
            f"{label} must live under settings.data_dir "
            f"(got {resolved}, data_dir={data_dir_resolved})"
        )
    return resolved
