"""Shared persistence path validation helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_path_under_data_dir(*, path: Path, data_dir: Path, label: str) -> Path:
    """Resolve a path anchored under data_dir (keeps legacy data/ prefix support)."""
    if path.is_absolute():
        candidate = path.expanduser()
    else:
        base = data_dir
        if path.parts and path.parts[0] == data_dir.name:
            base = data_dir.parent
        candidate = (base / path).expanduser()
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
