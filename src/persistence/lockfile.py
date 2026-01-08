"""Portalocker-backed file locks for snapshot persistence.

This module exposes a thin wrapper around ``portalocker`` to provide cross-platform
single-writer locking with heartbeat metadata and TTL-based takeover. Locks are
represented by a sentinel file plus a JSON sidecar recording ownership details.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final, cast

from loguru import logger

try:  # pragma: no cover - optional dependency guard
    import portalocker
    from portalocker.exceptions import LockException
except ImportError:  # pragma: no cover - handled at runtime
    portalocker = None  # type: ignore[assignment]
    LockException = RuntimeError  # sentinel fallback

__all__ = [
    "SnapshotLock",
    "SnapshotLockError",
    "SnapshotLockTimeoutError",
]

SCHEMA_VERSION: Final[int] = 1
_DEFAULT_SLEEP: Final[float] = 0.1


class SnapshotLockError(RuntimeError):
    """Raised when snapshot lock acquisition or refresh fails."""


class SnapshotLockTimeoutError(SnapshotLockError):
    """Raised when the lock cannot be acquired within the configured timeout."""


@dataclass(slots=True)
class _LockMetadata:
    """Structured representation of lock metadata persisted to JSON."""

    owner_id: str
    created_at: datetime
    last_heartbeat: datetime
    ttl_seconds: float
    takeover_count: int = 0
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> _LockMetadata:
        """Create metadata from a raw JSON dictionary."""
        try:
            created = datetime.fromisoformat(str(data["created_at"]))
            heartbeat = datetime.fromisoformat(str(data["last_heartbeat"]))
        except Exception as exc:  # pragma: no cover - defensive
            raise SnapshotLockError(
                "Lock metadata contains invalid timestamps"
            ) from exc
        return cls(
            owner_id=str(data.get("owner_id", "unknown")),
            created_at=created,
            last_heartbeat=heartbeat,
            ttl_seconds=float(data.get("ttl_seconds", 0.0)),
            takeover_count=int(data.get("takeover_count", 0)),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serialisable mapping for the metadata."""
        return {
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "takeover_count": self.takeover_count,
            "schema_version": self.schema_version,
        }


@dataclass
class SnapshotLock:
    """Portalocker-backed lock with TTL-based takeover semantics.

    Parameters
    ----------
    path:
        Filesystem path of the lock sentinel file.
    timeout:
        Maximum number of seconds to wait when attempting to acquire the lock.
    ttl_seconds:
        Time-to-live for an acquired lock. If the holder fails to refresh within
        this window, contenders can treat the lock as stale and forcibly take
        ownership by deleting the stale sentinel.
    """

    path: Path
    timeout: float
    ttl_seconds: float = 30.0
    grace_seconds: float = 0.0
    _lock: Any | None = field(init=False, default=None)
    _handle: object | None = field(init=False, default=None)
    _metadata: _LockMetadata | None = field(init=False, default=None)
    _pending_takeover_count: int = field(init=False, default=0)
    _use_portalocker: bool = field(init=False, default=True)
    _fd: int | None = field(init=False, default=None)
    _hb_stop: bool = field(init=False, default=False)
    _hb_thread: threading.Thread | None = field(init=False, default=None)
    _metadata_lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Prepare the lock file path and validate portalocker availability.

        Raises:
            SnapshotLockError: If the optional ``portalocker`` dependency is missing.
        """
        self.path = self.path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._use_portalocker = portalocker is not None
        if not self._use_portalocker:
            logger.warning(
                "portalocker is unavailable; using fallback O_EXCL file lock at %s",
                self.path,
            )

    def __enter__(self) -> SnapshotLock:
        """Enter the runtime context, acquiring the lock."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:  # pragma: no cover - simple
        """Exit the runtime context, releasing the lock."""
        del exc_type, exc, traceback
        self.release()

    def acquire(self) -> None:
        """Acquire the lock, evicting stale holders when necessary."""
        deadline = time.monotonic() + self.timeout
        while True:
            takeover_count, within_grace = self._evict_if_stale()
            self._pending_takeover_count = takeover_count
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SnapshotLockTimeoutError(
                    f"Timed out acquiring snapshot lock at {self.path} "
                    f"after {self.timeout:.1f}s"
                )
            if within_grace:
                time.sleep(min(_DEFAULT_SLEEP, max(0.0, remaining)))
                continue
            try:
                if self._use_portalocker:
                    portalocker_mod = cast(Any, portalocker)
                    lock = portalocker_mod.Lock(
                        str(self.path), mode="a+", timeout=remaining
                    )
                    handle = lock.acquire()
                    self._lock = lock
                    self._handle = handle
                else:
                    fd = os.open(
                        str(self.path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600
                    )
                    self._fd = fd
                    self._handle = os.fdopen(fd, "a+")
                now = datetime.now(UTC)
                self._metadata = _LockMetadata(
                    owner_id=_owner_id(),
                    created_at=now,
                    last_heartbeat=now,
                    ttl_seconds=self.ttl_seconds,
                    takeover_count=self._pending_takeover_count,
                )
                self._write_metadata()
                self._start_heartbeat()
                logger.debug("Acquired snapshot lock %s", self.path)
                return
            except LockException:  # pragma: no cover - depends on timing
                time.sleep(min(_DEFAULT_SLEEP, max(0.0, remaining)))
            except FileExistsError:  # pragma: no cover - depends on timing
                time.sleep(min(_DEFAULT_SLEEP, max(0.0, remaining)))

    def refresh(self) -> None:
        """Refresh lock heartbeat to prevent TTL expiry."""
        if self._metadata is None:
            raise SnapshotLockError("Cannot refresh snapshot lock before acquisition")
        self._metadata.last_heartbeat = datetime.now(UTC)
        self._write_metadata()

    def release(self) -> None:
        """Release the lock and clean up sidecar metadata."""
        try:
            self._stop_heartbeat()
            if self._lock is not None:
                self._lock.release()
            elif self._fd is not None:
                with contextlib.suppress(OSError):
                    os.close(self._fd)
        finally:
            if self._handle is not None:
                with contextlib.suppress(Exception):  # pragma: no cover - defensive
                    self._handle.close()  # type: ignore[attr-defined]
            self._lock = None
            self._handle = None
            self._metadata = None
            self._fd = None
            self._hb_thread = None
            _remove_if_exists(self._metadata_path())
            _remove_if_exists(self.path)
            logger.debug("Released snapshot lock %s", self.path)

    def _start_heartbeat(self) -> None:
        """Start a background thread that periodically refreshes the lock."""
        self._hb_stop = False
        interval = max(1.0, self.ttl_seconds / 2.0)

        def _worker() -> None:
            while not self._hb_stop:
                time.sleep(interval)
                if self._hb_stop:
                    break
                try:
                    self.refresh()
                except SnapshotLockError:
                    break

        self._hb_thread = threading.Thread(
            target=_worker, name=f"SnapshotLockHeartbeat[{self.path.name}]", daemon=True
        )
        self._hb_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stop the background heartbeat thread if running."""
        self._hb_stop = True
        thread = self._hb_thread
        if thread is None:
            return
        with contextlib.suppress(Exception):
            thread.join(timeout=1.0)

    def _metadata_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".meta.json")

    def _evict_if_stale(self) -> tuple[int, bool]:
        metadata_path = self._metadata_path()
        if not metadata_path.exists():
            return 0, False
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = _LockMetadata.from_json(data)
        except SnapshotLockError as exc:
            logger.warning("Lock metadata corrupted at %s: %s", metadata_path, exc)
            _remove_if_exists(metadata_path)
            _remove_if_exists(self.path)
            return 0, False
        now = datetime.now(UTC)
        elapsed = (now - metadata.last_heartbeat).total_seconds()
        effective_ttl = metadata.ttl_seconds + self.grace_seconds
        if elapsed <= effective_ttl:
            return metadata.takeover_count, True
        logger.warning(
            "Evicting stale snapshot lock %s held by %s (elapsed %.1fs > TTL %.1fs)",
            self.path,
            metadata.owner_id,
            elapsed,
            metadata.ttl_seconds,
        )
        new_takeover = metadata.takeover_count + 1
        _rotate_stale_lock(
            self.path, metadata_path, metadata.owner_id, metadata.takeover_count
        )
        return new_takeover, False

    def _write_metadata(self) -> None:
        if self._metadata is None:
            return
        payload = self._metadata.to_json()
        with self._metadata_lock:
            tmp_path = self._metadata_path().with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )
            _fsync_file(tmp_path)
            os.replace(tmp_path, self._metadata_path())
            _fsync_file(self._metadata_path())


def _owner_id() -> str:
    return f"{os.getpid()}@{socket.gethostname()}"


def _fsync_file(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError), path.open("rb") as handle:
        os.fsync(handle.fileno())


def _remove_if_exists(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _rotate_stale_lock(
    lock_path: Path, meta_path: Path, owner_id: str, takeover_count: int
) -> None:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    suffix = f".stale-{timestamp}-{owner_id.replace('@', '_')}-{takeover_count}"
    for target in (lock_path, meta_path):
        if target.exists():
            rotated = target.with_name(target.name + suffix)
            try:
                os.replace(target, rotated)
            except OSError:
                _remove_if_exists(target)
            else:
                _fsync_file(rotated)
