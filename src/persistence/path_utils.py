"""Shared persistence path validation helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_path_under_data_dir(*, path: Path, data_dir: Path, label: str) -> Path:
    """Resolve a path anchored under data_dir.

    Validates that the final path and all components within data_dir are not symlinks.
    Absolute paths are expanded; relative paths are anchored under data_dir if they are
    simple filenames (parent == '.'), otherwise treated as relative to cwd and validated
    to still resolve under data_dir.

    Args:
        path: Path to resolve (absolute or relative).
        data_dir: Base directory; final path must be under this directory.
        label: Human-readable label for error messages.

    Returns:
        Resolved absolute path, guaranteed to be under data_dir and free of symlinks.

    Raises:
        ValueError: If the path is a symlink, contains symlink components, or escapes
        data_dir.

    Note:
        Symlink checks apply only to existing paths. Callers creating new paths should
        use safe file flags (O_NOFOLLOW) to prevent TOCTOU attacks.
    """
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

    data_dir_resolved = data_dir.resolve()
    resolved = candidate.resolve()

    # Check if the path itself is a symlink (even if dangling)
    if candidate.is_symlink():
        raise ValueError(f"{label} may not be a symlink (got {candidate})")

    # If policy is "no symlinks under data_dir", reject any symlink components.
    # This check is meaningful only when components exist.
    if resolved.exists():
        probe = resolved
        while True:
            if probe.is_symlink():
                raise ValueError(f"{label} may not be a symlink (got {probe})")
            if probe == data_dir_resolved:
                break
            if probe.parent == probe:
                break
            probe = probe.parent

    if not resolved.is_relative_to(data_dir_resolved):
        raise ValueError(
            f"{label} must live under data_dir "
            f"(got {resolved}, data_dir={data_dir_resolved})"
        )
    return resolved
