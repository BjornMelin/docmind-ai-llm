"""Unit tests for snapshot locking and workspace utilities."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("portalocker")

from src.persistence import lockfile, snapshot


@pytest.fixture(autouse=True)
def _force_snapshot_root(snapshot_storage_root: Path) -> None:
    """Ensure snapshot storage writes to an isolated test directory."""
    assert snapshot_storage_root.is_dir()


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
    assert lock_path.exists()
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
    """Temporary workspace exists and enforces process-local exclusivity."""
    workspace = snapshot.begin_snapshot()
    try:
        assert workspace.is_dir()
        with pytest.raises(RuntimeError):
            snapshot.begin_snapshot()
    finally:
        snapshot.cleanup_tmp(workspace)
    assert not workspace.exists()


def test_deleted_sentinel_cannot_clear_live_process_owner(tmp_path: Path) -> None:
    """External sentinel deletion never lets this process start a second writer."""
    workspace = snapshot.begin_snapshot(tmp_path)
    lock_path = tmp_path / ".lock"
    lock_path.unlink()
    try:
        with pytest.raises(RuntimeError, match="already held"):
            snapshot.begin_snapshot(tmp_path)
        assert workspace.is_dir()
    finally:
        snapshot.cleanup_tmp(workspace)


def test_stale_workspace_cleanup_cannot_release_successor(tmp_path: Path) -> None:
    """Duplicate cleanup is scoped to its workspace ownership token."""
    first = snapshot.begin_snapshot(tmp_path)
    snapshot.cleanup_tmp(first)
    second = snapshot.begin_snapshot(tmp_path)
    try:
        snapshot.cleanup_tmp(first)
        with pytest.raises(RuntimeError, match="already held"):
            snapshot.begin_snapshot(tmp_path)
        assert second.is_dir()
    finally:
        snapshot.cleanup_tmp(second)


def test_release_handoff_keeps_successor_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A waiting begin cannot register until predecessor release fully clears."""
    first = snapshot.begin_snapshot(tmp_path)
    release_entered = threading.Event()
    allow_release_return = threading.Event()
    original_release = snapshot.SnapshotLock.release

    def _paused_release(lock: snapshot.SnapshotLock) -> None:
        original_release(lock)
        release_entered.set()
        assert allow_release_return.wait(timeout=2.0)

    monkeypatch.setattr(snapshot.SnapshotLock, "release", _paused_release)
    cleanup_thread = threading.Thread(target=snapshot.cleanup_tmp, args=(first,))
    cleanup_thread.start()
    assert release_entered.wait(timeout=2.0)

    successor: list[Path] = []
    begin_thread = threading.Thread(
        target=lambda: successor.append(snapshot.begin_snapshot(tmp_path))
    )
    begin_thread.start()
    time.sleep(0.05)
    assert successor == []
    allow_release_return.set()
    cleanup_thread.join(timeout=2.0)
    begin_thread.join(timeout=2.0)

    assert len(successor) == 1
    try:
        with pytest.raises(RuntimeError, match="already held"):
            snapshot.begin_snapshot(tmp_path)
    finally:
        snapshot.cleanup_tmp(successor[0])


def test_snapshot_lock_fails_closed_without_portalocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing OS-lock support never falls back to an unfenced lease file."""
    monkeypatch.setattr(lockfile, "portalocker", None)

    with pytest.raises(snapshot.SnapshotLockError, match="portalocker is required"):
        snapshot.SnapshotLock(tmp_path / "lockfile", timeout=0.2)


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


def test_portalocker_stale_metadata_never_splits_lock_inode(
    tmp_path: Path,
) -> None:
    """Stale metadata cannot create concurrent writers on separate inodes."""
    lock_path = tmp_path / "lockfile"
    meta_path = lock_path.with_suffix(lock_path.suffix + ".meta.json")
    primary = snapshot.SnapshotLock(lock_path, timeout=1.0, ttl_seconds=60.0)
    primary.acquire()
    original_inode = lock_path.stat().st_ino
    stale_payload = {
        "owner_id": "stalled-holder",
        "created_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        "last_heartbeat": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        "ttl_seconds": 0.01,
        "takeover_count": 0,
        "schema_version": 1,
    }
    meta_path.write_text(json.dumps(stale_payload), encoding="utf-8")

    start = threading.Barrier(4)
    release_contenders = threading.Event()
    state_lock = threading.Lock()
    acquired = 0
    active = 0
    max_active = 0
    errors: list[BaseException] = []

    def _contend() -> None:
        nonlocal acquired, active, max_active
        contender = snapshot.SnapshotLock(
            lock_path,
            timeout=2.0,
            ttl_seconds=60.0,
        )
        start.wait()
        try:
            contender.acquire()
            with state_lock:
                acquired += 1
                active += 1
                max_active = max(max_active, active)
            release_contenders.wait(timeout=2.0)
            with state_lock:
                active -= 1
            contender.release()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    contenders = [threading.Thread(target=_contend) for _ in range(3)]
    for contender_thread in contenders:
        contender_thread.start()
    start.wait()
    time.sleep(0.25)
    with state_lock:
        acquired_while_primary_held = acquired
    inode_while_primary_held = lock_path.stat().st_ino

    primary.release()
    release_contenders.set()
    for contender_thread in contenders:
        contender_thread.join(timeout=3.0)

    assert not errors
    assert not any(contender_thread.is_alive() for contender_thread in contenders)
    assert acquired_while_primary_held == 0
    assert acquired == 3
    assert max_active == 1
    assert inode_while_primary_held == original_inode
    assert lock_path.stat().st_ino == original_inode


def test_metadata_setup_failure_releases_os_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-acquisition metadata failure leaves the lock immediately reusable."""
    lock_path = tmp_path / "lockfile"
    original_write = snapshot.SnapshotLock._write_metadata

    def _fail_metadata(_lock: snapshot.SnapshotLock) -> None:
        raise OSError("simulated metadata failure")

    monkeypatch.setattr(snapshot.SnapshotLock, "_write_metadata", _fail_metadata)
    primary = snapshot.SnapshotLock(lock_path, timeout=0.2)
    with pytest.raises(snapshot.SnapshotLockError, match="initialization failed"):
        primary.acquire()

    monkeypatch.setattr(snapshot.SnapshotLock, "_write_metadata", original_write)
    successor = snapshot.SnapshotLock(lock_path, timeout=0.2)
    successor.acquire()
    successor.release()

    assert lock_path.exists()
    assert not lock_path.with_suffix(".meta.json").exists()
