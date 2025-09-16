"""SnapshotManager for index and property graph persistence.

Implements atomic, versioned snapshots with a manifest and single-writer lock.
The manifest carries corpus/config hashes for staleness detection.

Provides atomic snapshots of vector and graph indices with a manifest for
staleness detection. Uses a directory lock to ensure single-writer semantics
and an atomic rename to finalize snapshots.
"""

from __future__ import annotations

import json
import os
import posixpath
import shutil
import socket
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import uuid4

from filelock import FileLock, Timeout
from loguru import logger

from src.config.settings import settings

MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_FORMAT_VERSION = "1.0"


@dataclass(frozen=True)
class SnapshotPaths:
    """Resolved filesystem paths for snapshot persistence."""

    base_dir: Path
    lock_file: Path
    current_file: Path


class SnapshotLockTimeoutError(TimeoutError):
    """Raised when a snapshot lock cannot be acquired in time."""


SnapshotLockTimeout = SnapshotLockTimeoutError


class SnapshotLock:
    """Context manager wrapping :class:`filelock.FileLock` with metadata."""

    def __init__(
        self, path: Path, *, timeout: float, ttl_seconds: float = 30.0
    ) -> None:
        """Initialize a snapshot lock around ``path`` with the supplied timing."""
        self.path = path
        self.timeout = timeout
        self.ttl_seconds = ttl_seconds
        self._lock = FileLock(str(path))

    def acquire(self) -> None:
        """Acquire the underlying file lock, evicting stale holders when needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._evict_stale_lock()
        try:
            self._lock.acquire(timeout=self.timeout)
        except Timeout as exc:  # pragma: no cover - defensive
            raise SnapshotLockTimeoutError(str(exc)) from exc
        self._write_metadata()

    def refresh(self) -> None:
        """Refresh lock metadata to extend the lease TTL."""
        if not self._lock.is_locked:
            raise RuntimeError("Cannot refresh an unlocked snapshot lock")
        self._write_metadata()

    def release(self) -> None:
        """Release the lock and clean up associated metadata files."""
        if self._lock.is_locked:
            self._lock.release()
        with suppress(OSError):
            self.path.unlink()
        with suppress(OSError):
            self._meta_path().unlink()

    def __enter__(self) -> SnapshotLock:
        """Enter context manager and acquire lock."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the lock on context manager exit."""
        self.release()

    def _write_metadata(self) -> None:
        expires_at = datetime.now(UTC) + timedelta(seconds=self.ttl_seconds)
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at": datetime.now(UTC).isoformat(),
            "expires_at": expires_at.isoformat(),
        }
        meta_path = self._meta_path()
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())

    def _meta_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".meta")

    def _evict_stale_lock(self) -> None:
        meta_path = self._meta_path()
        if not meta_path.exists():
            return
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            expires = payload.get("expires_at")
            if not expires:
                return
            expiry_dt = datetime.fromisoformat(expires)
        except (OSError, json.JSONDecodeError, ValueError):  # pragma: no cover
            return
        if expiry_dt <= datetime.now(UTC):
            with suppress(OSError):
                self.path.unlink()
            with suppress(OSError):
                meta_path.unlink()
            logger.info("Removed stale snapshot lock at %s", self.path)


def _snapshot_paths(base_dir: Path | None = None) -> SnapshotPaths:
    base = Path(base_dir) if base_dir is not None else settings.data_dir / "storage"
    return SnapshotPaths(
        base_dir=base,
        lock_file=base / ".lock",
        current_file=base / "CURRENT",
    )


def _create_workspace(base_dir: Path) -> Path:
    workspace = base_dir / f"_tmp-{uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    for child in ("vector", "graph"):
        (workspace / child).mkdir(parents=True, exist_ok=True)
    return workspace


def _generate_version_name() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"


def _iter_payload_files(tmp_dir: Path) -> Iterator[Path]:
    for path in tmp_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(tmp_dir)
        if rel.name in {
            "manifest.jsonl",
            "manifest.checksum",
            "manifest.meta.json",
            "manifest.json",
        }:
            continue
        yield path


def _hash_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fsync_dir(path: Path) -> None:
    with suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _fsync_file(path: Path) -> None:
    with suppress(FileNotFoundError), path.open("rb") as handle:
        os.fsync(handle.fileno())


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _fsync_file(path)


def _enrich_manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    enriched = {
        **manifest,
        "schema_version": manifest.get("schema_version", MANIFEST_SCHEMA_VERSION),
        "persist_format_version": manifest.get(
            "persist_format_version", MANIFEST_FORMAT_VERSION
        ),
        "complete": manifest.get("complete", False),
    }
    enriched.setdefault("versions", {})
    return enriched


def _calculate_manifest_digest(
    entries: list[dict[str, Any]], manifest_payload: dict[str, Any]
) -> str:
    aggregate = sha256()
    for entry in entries:
        aggregate.update(entry["sha256"].encode("utf-8"))
    aggregate.update(
        json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    return aggregate.hexdigest()


def _write_manifest_checksum(
    snapshot_dir: Path, entries: list[dict[str, Any]], manifest_payload: dict[str, Any]
) -> None:
    checksum_payload = {
        "schema_version": 1,
        "manifest_sha256": _calculate_manifest_digest(entries, manifest_payload),
        "created_at": datetime.now(UTC).isoformat(),
    }
    checksum_path = _manifest_checksum_path(snapshot_dir)
    _dump_json(checksum_path, checksum_payload)


def _load_manifest_entries(snapshot_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    manifest_path = _manifest_path(snapshot_dir)
    if not manifest_path.exists():
        return entries
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:  # pragma: no cover - defensive
                continue
    return entries


def _mark_manifest_complete(snapshot_dir: Path) -> None:
    meta_path = snapshot_dir / "manifest.meta.json"
    if not meta_path.exists():
        return
    try:
        manifest_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - defensive
        return
    if manifest_payload.get("complete") is True:
        return
    manifest_payload["complete"] = True
    legacy_manifest = snapshot_dir / "manifest.json"
    _dump_json(meta_path, manifest_payload)
    if legacy_manifest.exists():
        _dump_json(legacy_manifest, manifest_payload)
    entries = _load_manifest_entries(snapshot_dir)
    _write_manifest_checksum(snapshot_dir, entries, manifest_payload)


_ACTIVE_LOCK: SnapshotLock | None = None


def begin_snapshot(base_dir: Path | None = None) -> Path:
    """Create a locked workspace for snapshot persistence."""
    global _ACTIVE_LOCK
    if _ACTIVE_LOCK is not None:
        raise RuntimeError("Snapshot lock already held; finalize or cleanup first.")

    paths = _snapshot_paths(base_dir)
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    config = settings.snapshots
    lock = SnapshotLock(
        paths.lock_file,
        timeout=float(config.lock_timeout_seconds),
        ttl_seconds=float(config.lock_ttl_seconds),
    )
    lock.acquire()
    _ACTIVE_LOCK = lock
    return _create_workspace(paths.base_dir)


def _release_active_lock() -> None:
    global _ACTIVE_LOCK
    if _ACTIVE_LOCK is None:
        return
    _ACTIVE_LOCK.release()
    _ACTIVE_LOCK = None


def cleanup_tmp(tmp_dir: Path) -> None:
    """Best-effort cleanup of workspace and lock release."""
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    finally:
        _release_active_lock()


def _manifest_path(snapshot_dir: Path) -> Path:
    return snapshot_dir / "manifest.jsonl"


def _manifest_checksum_path(snapshot_dir: Path) -> Path:
    return snapshot_dir / "manifest.checksum"


def write_manifest(tmp_dir: Path, manifest: dict[str, Any]) -> None:
    """Write snapshot manifest (JSONL + checksum + metadata)."""
    entries: list[dict[str, Any]] = []
    for file_path in sorted(_iter_payload_files(tmp_dir)):
        relative = file_path.relative_to(tmp_dir)
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": _hash_file(file_path),
                "size_bytes": file_path.stat().st_size,
                "content_type": "application/octet-stream",
            }
        )

    manifest_path = _manifest_path(tmp_dir)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    enriched_manifest = _enrich_manifest_payload(manifest)
    meta_path = tmp_dir / "manifest.meta.json"
    _dump_json(meta_path, enriched_manifest)

    legacy_manifest = tmp_dir / "manifest.json"
    _dump_json(legacy_manifest, enriched_manifest)

    _write_manifest_checksum(tmp_dir, entries, enriched_manifest)


def _write_current_pointer(paths: SnapshotPaths, version_name: str) -> None:
    tmp_pointer = paths.base_dir / "CURRENT.tmp"
    tmp_pointer.write_text(version_name, encoding="utf-8")
    _fsync_file(tmp_pointer)
    os.replace(tmp_pointer, paths.current_file)
    _fsync_file(paths.current_file)
    _fsync_dir(paths.base_dir)


def _garbage_collect(paths: SnapshotPaths) -> None:
    config = settings.snapshots
    keep = max(1, int(config.retention_count))
    grace = max(0, int(config.gc_grace_seconds))
    versions = sorted(
        [
            p
            for p in paths.base_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_tmp-")
        ]
    )
    if len(versions) <= keep:
        return
    cutoff = datetime.now(UTC) - timedelta(seconds=grace)
    for candidate in versions[:-keep]:
        if (
            grace > 0
            and datetime.fromtimestamp(candidate.stat().st_mtime, UTC) > cutoff
        ):
            continue
        shutil.rmtree(candidate, ignore_errors=True)


def finalize_snapshot(tmp_dir: Path, *, base_dir: Path | None = None) -> Path:
    """Rename workspace to versioned snapshot and update CURRENT."""
    from src.utils.monitoring import log_performance  # local import to avoid heavy deps

    start = time.perf_counter()
    paths = _snapshot_paths(base_dir)
    if not tmp_dir.exists():  # pragma: no cover - defensive
        _release_active_lock()
        raise FileNotFoundError(f"tmp snapshot dir missing: {tmp_dir}")

    version_name = _generate_version_name()
    destination = paths.base_dir / version_name

    try:
        _fsync_dir(tmp_dir)
        tmp_dir.rename(destination)
        _mark_manifest_complete(destination)
        _fsync_dir(paths.base_dir)
        _write_current_pointer(paths, version_name)
        _garbage_collect(paths)
        log_performance(
            operation="snapshot_finalize",
            success=True,
            duration_seconds=time.perf_counter() - start,
            version=version_name,
        )
        logger.info("Snapshot finalized at %s", destination)
        return destination
    except Exception as exc:
        log_performance(
            operation="snapshot_finalize",
            success=False,
            duration_seconds=time.perf_counter() - start,
            error=str(exc),
        )
        raise
    finally:
        _release_active_lock()


def latest_snapshot_dir(base_dir: Path | None = None) -> Path | None:
    """Return the most recent version directory or ``None``."""
    paths = _snapshot_paths(base_dir)
    if not paths.base_dir.exists():
        return None
    if paths.current_file.exists():
        with suppress(OSError):
            name = paths.current_file.read_text(encoding="utf-8").strip()
            if name:
                candidate = paths.base_dir / name
                if candidate.is_dir():
                    return candidate
    versions = [
        p
        for p in paths.base_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_tmp-")
    ]
    if not versions:
        return None
    return sorted(versions)[-1]


def load_manifest(
    snapshot_dir: Path | None = None, *, base_dir: Path | None = None
) -> dict[str, Any] | None:
    """Load snapshot metadata manifest."""
    snap = snapshot_dir or latest_snapshot_dir(base_dir)
    if not snap:
        return None
    meta = snap / "manifest.meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover
            logger.debug("Failed to parse manifest.meta.json in %s: %s", snap, exc)
            return None
    legacy = snap / "manifest.json"
    if legacy.exists():
        try:
            return json.loads(legacy.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover
            logger.debug("Failed to parse manifest.json in %s: %s", snap, exc)
            return None
    return None


# --- Remaining downstream helpers (persist/load/hash) adapted from legacy ---
#     implementation


def persist_vector_index(index: Any, out_dir: Path) -> None:
    """Persist a vector index into ``out_dir`` using the storage context API."""
    try:
        index.storage_context.persist(persist_dir=str(out_dir))  # type: ignore[attr-defined]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(f"Persist vector index failed: {exc}") from exc


def persist_graph_store(store: Any, out_dir: Path) -> None:
    """Persist a property graph store into ``out_dir`` in a library-first way."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            store.persist(persist_dir=str(out_dir))  # type: ignore[attr-defined]
        except TypeError:
            store.persist(str(out_dir))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError(f"Persist graph store failed: {exc}") from exc


def load_vector_index(snapshot_dir: Path | None = None) -> Any | None:
    """Load a persisted vector index from ``snapshot_dir`` when available."""
    try:
        from llama_index.core import StorageContext, load_index_from_storage
    except (
        ImportError,
        ModuleNotFoundError,
        AttributeError,
    ):  # pragma: no cover - import guard
        return None

    snap = snapshot_dir or latest_snapshot_dir()
    if not snap:
        return None
    vec_dir = snap / "vector"
    if not vec_dir.exists():
        return None
    try:
        storage = StorageContext.from_defaults(persist_dir=str(vec_dir))
        return load_index_from_storage(storage)
    except (
        OSError,
        RuntimeError,
        ValueError,
        AttributeError,
    ) as exc:  # pragma: no cover - defensive
        logger.debug("Unable to load vector index from %s: %s", vec_dir, exc)
        return None


def load_property_graph_index(snapshot_dir: Path | None = None) -> Any | None:
    """Load a persisted property graph index from ``snapshot_dir`` when available."""
    try:
        from llama_index.core import PropertyGraphIndex
        from llama_index.core.graph_stores import SimplePropertyGraphStore
    except (
        ImportError,
        ModuleNotFoundError,
        AttributeError,
    ):  # pragma: no cover - import guard
        return None

    snap = snapshot_dir or latest_snapshot_dir()
    if not snap:
        return None
    graph_dir = snap / "graph"
    if not graph_dir.exists():
        return None
    try:
        store = SimplePropertyGraphStore.from_persist_dir(str(graph_dir))
        return PropertyGraphIndex.from_existing(property_graph_store=store)
    except (
        OSError,
        RuntimeError,
        ValueError,
        AttributeError,
    ) as exc:  # pragma: no cover - defensive
        logger.debug("Unable to load property graph index from %s: %s", graph_dir, exc)
        return None


def compute_corpus_hash(
    upload: Path | list[Path] | None = None, *, base_dir: Path | None = None
) -> str:
    """Compute a stable SHA256 hash for the ingestion corpus."""
    files: list[Path] = []
    if upload is None:
        base = settings.data_dir / "uploads"
        if base.exists():
            files = [p for p in base.rglob("*") if p.is_file()]
    elif isinstance(upload, Path):
        base = upload
        if base.exists():
            files = [p for p in base.rglob("*") if p.is_file()]
    else:
        files = [p for p in upload if isinstance(p, Path) and p.is_file()]
    items = []
    for p in files:
        try:
            stat = p.stat()
            if base_dir is not None:
                try:
                    rel = p.relative_to(base_dir)
                except ValueError:
                    rel = p
                name = posixpath.join(*rel.parts)
            else:
                name = str(p)
            items.append((name, stat.st_size, stat.st_mtime_ns))
        except OSError:  # pragma: no cover
            continue
    hasher = sha256()
    for name, size, mtime in sorted(items):
        hasher.update(f"{name}|{size}|{mtime}".encode())
    return f"sha256:{hasher.hexdigest()}"


def compute_config_hash(cfg: dict[str, Any] | None = None) -> str:
    """Compute a SHA256 hash for the active retrieval/configuration payload."""
    if cfg is None:
        cfg = {
            "router": getattr(settings.retrieval, "router", None),
            "hybrid": getattr(settings.retrieval, "enable_server_hybrid", None),
            "graph_enabled": getattr(settings, "enable_graphrag", False),
            "chunk_size": getattr(settings.processing, "chunk_size", None),
            "chunk_overlap": getattr(settings.processing, "chunk_overlap", None),
        }
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return f"sha256:{sha256(blob.encode('utf-8')).hexdigest()}"


def is_stale(manifest: dict[str, Any] | None) -> bool:
    """Return ``True`` when current corpus/config hashes diverge from ``manifest``."""
    if not manifest:
        return True
    base = settings.data_dir / "uploads"
    from src.persistence import snapshot_utils as _snapshot_utils

    corpus_paths = _snapshot_utils.collect_corpus_paths(base)
    cfg = _snapshot_utils.current_config_dict(settings)
    cur = {
        "corpus_hash": compute_corpus_hash(corpus_paths, base_dir=base),
        "config_hash": compute_config_hash(cfg),
    }
    return (
        manifest.get("corpus_hash") != cur["corpus_hash"]
        or manifest.get("config_hash") != cur["config_hash"]
    )


def recover_snapshots(base_dir: Path | None = None) -> None:
    """Remove stale workspaces and repair CURRENT pointer if missing."""
    paths = _snapshot_paths(base_dir)
    if not paths.base_dir.exists():
        return

    for candidate in paths.base_dir.glob("_tmp-*"):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
            logger.debug("Removed stale workspace %s", candidate)

    current_target: Path | None = None
    if paths.current_file.exists():
        with suppress(OSError):
            name = paths.current_file.read_text(encoding="utf-8").strip()
            if name:
                candidate = paths.base_dir / name
                if candidate.exists():
                    current_target = candidate

    if current_target is not None:
        logger.debug("CURRENT pointer verified for %s", current_target)
        return

    versions = sorted(
        [
            p
            for p in paths.base_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_tmp-")
        ]
    )
    if not versions:
        with suppress(FileNotFoundError):
            paths.current_file.unlink()
        return

    latest = versions[-1].name
    _write_current_pointer(paths, latest)
    logger.info("Recovered CURRENT pointer -> %s", latest)


def verify_snapshot(snapshot_dir: Path) -> bool:
    """Verify manifest hashes and payload integrity for a snapshot directory."""
    manifest_path = _manifest_path(snapshot_dir)
    checksum_path = _manifest_checksum_path(snapshot_dir)
    meta_path = snapshot_dir / "manifest.meta.json"
    if not (manifest_path.exists() and checksum_path.exists() and meta_path.exists()):
        return False

    try:
        expected = json.loads(checksum_path.read_text(encoding="utf-8"))[
            "manifest_sha256"
        ]
    except (KeyError, json.JSONDecodeError):  # pragma: no cover
        return False

    entries = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    for entry in entries:
        target = snapshot_dir / entry.get("path", "")
        if not target.exists():
            return False
        if _hash_file(target) != entry.get("sha256"):
            return False

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    aggregate = sha256()
    for entry in entries:
        aggregate.update(entry["sha256"].encode("utf-8"))
    aggregate.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    return aggregate.hexdigest() == expected


__all__ = [
    "SnapshotLock",
    "SnapshotLockTimeout",
    "SnapshotLockTimeoutError",
    "begin_snapshot",
    "cleanup_tmp",
    "compute_config_hash",
    "compute_corpus_hash",
    "finalize_snapshot",
    "is_stale",
    "latest_snapshot_dir",
    "load_manifest",
    "load_property_graph_index",
    "load_vector_index",
    "persist_graph_store",
    "persist_vector_index",
    "recover_snapshots",
    "verify_snapshot",
    "write_manifest",
]


class SnapshotManager:
    """Compatibility wrapper for UI components expecting a class API."""

    def __init__(self, storage_dir: Path) -> None:
        """Initialize the manager, ensuring ``storage_dir`` exists."""
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def begin_snapshot(self) -> Path:
        """Create a locked workspace under ``storage_dir``."""
        return begin_snapshot(self.storage_dir)

    def persist_vector_index(self, index: Any, tmp_dir: Path) -> None:
        """Persist a vector index into the workspace ``tmp_dir``."""
        persist_vector_index(index, tmp_dir / "vector")

    def persist_graph_store(self, store: Any, tmp_dir: Path) -> None:
        """Persist a graph store into the workspace ``tmp_dir``."""
        persist_graph_store(store, tmp_dir / "graph")

    def write_manifest(
        self,
        tmp_dir: Path,
        *,
        index_id: str,
        graph_store_type: str,
        vector_store_type: str,
        corpus_hash: str,
        config_hash: str,
        versions: dict[str, Any] | None = None,
    ) -> None:
        """Write manifest metadata for the current workspace."""
        data = {
            "index_id": index_id,
            "graph_store_type": graph_store_type,
            "vector_store_type": vector_store_type,
            "corpus_hash": corpus_hash,
            "config_hash": config_hash,
            "created_at": datetime.now(UTC).isoformat(),
            "versions": versions or {},
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "persist_format_version": MANIFEST_FORMAT_VERSION,
            "complete": False,
        }
        write_manifest(tmp_dir, data)

    def finalize_snapshot(self, tmp_dir: Path) -> Path:
        """Finalize the workspace into an immutable snapshot directory."""
        return finalize_snapshot(tmp_dir, base_dir=self.storage_dir)

    def cleanup_tmp(self, tmp_dir: Path) -> None:
        """Remove temporary workspace and release the active lock."""
        cleanup_tmp(tmp_dir)


__all__ += ["SnapshotManager"]
