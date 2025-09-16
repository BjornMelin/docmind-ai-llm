"""Unit tests for snapshot locking and workspace utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.persistence import snapshot


@pytest.fixture(autouse=True)
def _isolate_storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point snapshot storage to an isolated temp directory for each test."""
    from src.config.settings import settings as _settings

    _settings.data_dir = tmp_path


def test_snapshot_lock_acquire_and_release(tmp_path: Path) -> None:
    """SnapshotLock writes lease metadata and releases cleanly."""
    lock_path = tmp_path / "lockfile"
    lock = snapshot.SnapshotLock(lock_path, timeout=0.5)
    lock.acquire()
    try:
        meta_path = lock_path.with_suffix(lock_path.suffix + ".meta")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert "expires_at" in payload
    finally:
        lock.release()
    assert not lock_path.exists()
    assert not meta_path.exists()


def test_snapshot_lock_timeout(tmp_path: Path) -> None:
    """Second acquisition times out while the first holder retains the lock."""
    lock_path = tmp_path / "lockfile"
    primary = snapshot.SnapshotLock(lock_path, timeout=0.2)
    primary.acquire()
    try:
        contender = snapshot.SnapshotLock(lock_path, timeout=0.1)
        with pytest.raises(snapshot.SnapshotLockTimeout):
            contender.acquire()
    finally:
        primary.release()


def test_begin_snapshot_creates_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Temporary workspace contains expected subdirectories and enforces exclusivity."""
    workspace = snapshot.begin_snapshot()
    try:
        assert (workspace / "vector").is_dir()
        assert (workspace / "graph").is_dir()
        with pytest.raises(RuntimeError):
            snapshot.begin_snapshot()
    finally:
        snapshot.cleanup_tmp(workspace)
    assert not workspace.exists()
