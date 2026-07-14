"""Cross-platform, OS-fenced single-writer locks for snapshot persistence."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final, cast

from loguru import logger

try:  # pragma: no cover - optional dependency guard
    import portalocker
    from portalocker.exceptions import LockException
except ImportError:  # pragma: no cover - fail-closed runtime guard
    portalocker = None  # type: ignore[assignment]
    LockException = RuntimeError

__all__ = [
    "SnapshotLock",
    "SnapshotLockError",
    "SnapshotLockTimeoutError",
]

SCHEMA_VERSION: Final[int] = 1


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
    """Single-writer lock with informational heartbeat metadata.

    Parameters
    ----------
    path:
        Filesystem path of the lock sentinel file.
    timeout:
        Maximum number of seconds to wait when attempting to acquire the lock.
    ttl_seconds:
        Heartbeat interval basis for informational lease metadata. The operating
        system lock remains the sole ownership authority.
    """

    path: Path
    timeout: float
    ttl_seconds: float = 30.0
    _lock: Any | None = field(init=False, default=None)
    _handle: object | None = field(init=False, default=None)
    _metadata: _LockMetadata | None = field(init=False, default=None)
    _hb_stop: threading.Event = field(init=False, default_factory=threading.Event)
    _hb_thread: threading.Thread | None = field(init=False, default=None)
    _metadata_lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Prepare the lock file path and validate portalocker availability.

        Raises:
            SnapshotLockError: If the optional ``portalocker`` dependency is missing.
        """
        self.path = self.path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if portalocker is None:
            raise SnapshotLockError(
                "portalocker is required for OS-fenced snapshot locking"
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
        """Acquire the operating-system lock and initialize lease metadata."""
        portalocker_mod = cast(Any, portalocker)
        try:
            lock = portalocker_mod.Lock(str(self.path), mode="a+", timeout=self.timeout)
            handle = lock.acquire()
        except LockException as exc:
            raise SnapshotLockTimeoutError(
                f"Timed out acquiring snapshot lock at {self.path} "
                f"after {self.timeout:.1f}s"
            ) from exc

        self._lock = lock
        self._handle = handle
        now = datetime.now(UTC)
        self._metadata = _LockMetadata(
            owner_id=_owner_id(),
            created_at=now,
            last_heartbeat=now,
            ttl_seconds=self.ttl_seconds,
        )
        try:
            self._write_metadata()
            self._start_heartbeat()
        except BaseException as exc:
            with contextlib.suppress(Exception):
                self.release()
            if isinstance(exc, SnapshotLockError):
                raise
            raise SnapshotLockError(
                "Snapshot lock metadata initialization failed"
            ) from exc
        logger.debug("Acquired snapshot lock {}", self.path.name)

    def refresh(self) -> None:
        """Refresh lock heartbeat to prevent TTL expiry."""
        if self._metadata is None:
            raise SnapshotLockError("Cannot refresh snapshot lock before acquisition")
        self._metadata.last_heartbeat = datetime.now(UTC)
        self._write_metadata()

    def release(self) -> None:
        """Release the OS lock while preserving its permanent sentinel inode."""
        try:
            self._stop_heartbeat()
            if self._lock is not None:
                _remove_if_exists(self._metadata_path())
            if self._lock is not None:
                self._lock.release()
        finally:
            if self._handle is not None:
                with contextlib.suppress(Exception):  # pragma: no cover - defensive
                    self._handle.close()  # type: ignore[attr-defined]
            self._lock = None
            self._handle = None
            self._metadata = None
            self._hb_thread = None
            logger.debug("Released snapshot lock {}", self.path.name)

    def _start_heartbeat(self) -> None:
        """Start a background thread that periodically refreshes the lock."""
        self._hb_stop.clear()
        interval = max(1.0, self.ttl_seconds / 2.0)

        def _worker() -> None:
            while not self._hb_stop.wait(interval):
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
        self._hb_stop.set()
        thread = self._hb_thread
        if thread is None:
            return
        with contextlib.suppress(Exception):
            thread.join(timeout=1.0)

    def _metadata_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".meta.json")

    def _write_metadata(self) -> None:
        if self._metadata is None:
            return
        payload = self._metadata.to_json()
        with self._metadata_lock:
            tmp_path = self._metadata_path().with_name(
                f".{self._metadata_path().name}.{uuid.uuid4().hex}.tmp"
            )
            tmp_path.write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )
            _fsync_file(tmp_path)
            os.replace(tmp_path, self._metadata_path())
            _fsync_file(self._metadata_path())


def _owner_id() -> str:
    return f"{os.getpid()}@{socket.gethostname()}-{uuid.uuid4().hex}"


def _fsync_file(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError), path.open("rb") as handle:
        os.fsync(handle.fileno())


def _remove_if_exists(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()
