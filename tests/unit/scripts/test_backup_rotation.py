from __future__ import annotations

from pathlib import Path

import pytest

from src.persistence.backup_service import prune_backups

pytestmark = pytest.mark.unit


def test_prune_backups_keeps_most_recent(tmp_path: Path) -> None:
    """Remove older backups while keeping the newest entries.

    Args:
        tmp_path: Temporary directory for test artifacts.

    Returns:
        None.
    """
    root = tmp_path / "backups"
    root.mkdir()

    # Names sort lexicographically; these simulate timestamps.
    old = root / "backup_20260101_000000"
    mid = root / "backup_20260101_000001"
    new = root / "backup_20260101_000002"
    old.mkdir()
    mid.mkdir()
    new.mkdir()
    (root / "not_a_backup").mkdir()

    deleted = prune_backups(root, keep_last=2)

    assert old in deleted
    assert not old.exists()
    assert mid.exists()
    assert new.exists()
    assert (root / "not_a_backup").exists()


def test_prune_backups_rejects_invalid_keep_last(tmp_path: Path) -> None:
    """Reject keep_last values below the minimum of 1.

    Args:
        tmp_path: Temporary directory for test artifacts.

    Returns:
        None.
    """
    root = tmp_path / "backups"
    root.mkdir()

    with pytest.raises(ValueError, match="keep_last must be >= 1"):
        prune_backups(root, keep_last=0)
