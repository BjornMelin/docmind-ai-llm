"""Unit tests for snapshot locking and workspace utilities."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("portalocker")

from src.persistence import snapshot


@pytest.fixture(autouse=True)
def _force_snapshot_root(snapshot_storage_root: Path) -> None:
    """Ensure snapshot storage writes to an isolated test directory."""
    return snapshot_storage_root


def test_snapshot_lock_acquire_and_release(tmp_path: Path) -> None:
    """SnapshotLock writes lease metadata and releases cleanly."""
    lock_path = tmp_path / "lockfile"
    lock = snapshot.SnapshotLock(lock_path, timeout=0.5, ttl_seconds=0.2)
    lock.acquire()
    try:
        meta_path = lock_path.with_suffix(lock_path.suffix + ".meta.json")
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        assert payload["owner_id"]
        assert payload["ttl_seconds"] == pytest.approx(0.2, rel=0.1)
        assert payload["schema_version"] == 1
        assert "last_heartbeat" in payload
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


def test_snapshot_lock_takes_over_stale_lock(tmp_path: Path) -> None:
    """Stale lock files are rotated and a new lock acquires ownership."""
    lock_path = tmp_path / "lockfile"
    meta_path = lock_path.with_suffix(lock_path.suffix + ".meta.json")
    lock_path.write_text("stale", encoding="utf-8")
    stale_payload = {
        "owner_id": "123",
        "created_at": (datetime.now(UTC) - timedelta(seconds=3600)).isoformat(),
        "last_heartbeat": (datetime.now(UTC) - timedelta(seconds=3600)).isoformat(),
        "ttl_seconds": 1.0,
        "takeover_count": 0,
        "schema_version": 1,
    }
    meta_path.write_text(json.dumps(stale_payload), encoding="utf-8")

    lock = snapshot.SnapshotLock(
        lock_path, timeout=1.0, ttl_seconds=0.1, grace_seconds=0.0
    )
    lock.acquire()
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        assert payload["takeover_count"] == 1
        stale_files = list(tmp_path.glob("lockfile*.stale-" + "*"))
        assert stale_files
    finally:
        lock.release()


def test_snapshot_lock_respects_grace_period(tmp_path: Path) -> None:
    """Locks within the grace interval are not evicted prematurely."""
    lock_path = tmp_path / "lockfile"
    meta_path = lock_path.with_suffix(lock_path.suffix + ".meta.json")
    lock_path.write_text("held", encoding="utf-8")
    recent_payload = {
        "owner_id": "123",
        "created_at": (datetime.now(UTC) - timedelta(seconds=2)).isoformat(),
        "last_heartbeat": (datetime.now(UTC) - timedelta(seconds=2)).isoformat(),
        "ttl_seconds": 1.0,
        "takeover_count": 0,
        "schema_version": 1,
    }
    meta_path.write_text(json.dumps(recent_payload), encoding="utf-8")

    contender = snapshot.SnapshotLock(
        lock_path, timeout=0.2, ttl_seconds=1.0, grace_seconds=5.0
    )
    takeover = contender._evict_if_stale()
    assert takeover == 0
    assert lock_path.exists()
    assert meta_path.exists()
    contender.release()


def test_snapshot_lock_refresh_updates_metadata(tmp_path: Path) -> None:
    """Refreshing the lock updates the heartbeat timestamp."""
    lock_path = tmp_path / "lockfile"
    lock = snapshot.SnapshotLock(lock_path, timeout=0.5, ttl_seconds=0.2)
    lock.acquire()
    try:
        meta_path = lock_path.with_suffix(lock_path.suffix + ".meta.json")
        before = json.loads(meta_path.read_text(encoding="utf-8"))["last_heartbeat"]
        time.sleep(0.05)
        lock.refresh()
        after = json.loads(meta_path.read_text(encoding="utf-8"))["last_heartbeat"]
        assert after > before
    finally:
        lock.release()
