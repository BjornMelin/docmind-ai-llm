"""SnapshotManager for index and property graph persistence.

Provides atomic snapshots of vector and graph indices with a manifest for
staleness detection. Uses a single-writer filesystem lock and atomic renames to
finalize snapshots. Manifest generation and workspace management leverage helper
modules to ensure deterministic hashing and metadata emission.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from loguru import logger

from src.config.settings import settings
from src.persistence.hashing import compute_config_hash, compute_corpus_hash
from src.persistence.lockfile import (
    SnapshotLock,
    SnapshotLockError,
    SnapshotLockTimeoutError,
)
from src.persistence.snapshot_writer import (
    load_manifest_entries,
    mark_manifest_complete,
    start_workspace,
)
from src.persistence.snapshot_writer import (
    write_manifest as _writer_write_manifest,
)

try:  # pragma: no cover - optional monitoring dependencies
    from src.utils.monitoring import log_performance
except ImportError:  # pragma: no cover - defensive fallback

    def log_performance(*_args: Any, **_kwargs: Any) -> None:
        """No-op performance logger when monitoring stack is unavailable."""
        return None


try:  # pragma: no cover - optional instrumentation
    from opentelemetry import trace
except ImportError:  # pragma: no cover
    trace = None  # type: ignore[assignment]

if trace is not None:
    _tracer = trace.get_tracer(__name__)
else:  # pragma: no cover - fallback when OTel not installed

    class _NoopTracer:
        """Minimal tracer when OpenTelemetry instrumentation is unavailable."""

        def start_as_current_span(self, name: str):
            """Return a no-op context manager for span instrumentation."""
            del name
            return nullcontext()

    _tracer = _NoopTracer()

SnapshotLockTimeout = SnapshotLockTimeoutError


class SnapshotError(RuntimeError):
    """Base error for snapshot operations."""


class SnapshotManifestError(SnapshotError):
    """Raised when manifest generation fails."""


class SnapshotPromotionError(SnapshotError):
    """Raised when atomic promotion or CURRENT update fails."""


class SnapshotPersistenceError(SnapshotError):
    """Raised when persisting snapshot artifacts fails."""


class SnapshotLoadError(SnapshotError):
    """Raised when snapshot payloads cannot be loaded."""


@dataclass(frozen=True, slots=True)
class SnapshotPaths:
    """Resolved filesystem paths for snapshot persistence."""

    base_dir: Path
    lock_file: Path
    current_file: Path


def _snapshot_paths(base_dir: Path | None = None) -> SnapshotPaths:
    """Resolve snapshot paths for the provided base directory."""
    base = Path(base_dir) if base_dir is not None else settings.data_dir / "storage"
    return SnapshotPaths(
        base_dir=base,
        lock_file=base / ".lock",
        current_file=base / "CURRENT",
    )


def _generate_version_name() -> str:
    """Return a timestamped version identifier for finalized snapshots."""
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"


def _fsync_dir(path: Path) -> None:
    """Best-effort ``fsync`` of a directory descriptor."""
    with suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _fsync_file(path: Path) -> None:
    """Best-effort ``fsync`` of a regular file."""
    with suppress(FileNotFoundError), path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_text_atomic(path: Path, data: str) -> None:
    """Atomically replace ``path`` with the provided textual ``data``."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(data, encoding="utf-8")
    _fsync_file(tmp_path)
    os.replace(tmp_path, path)
    _fsync_file(path)


def _emit_snapshot_log(
    stage: str, *, status: str, snapshot_id: str | None = None, **extras: Any
) -> None:
    """Emit a structured log entry for snapshot operations."""
    logger.bind(
        snapshot_stage=stage, status=status, snapshot_id=snapshot_id, **extras
    ).info("snapshot.%s", stage)


def _append_error_record(base_dir: Path, payload: dict[str, Any]) -> None:
    """Append an error record to ``errors.jsonl`` under ``base_dir``."""
    errors_path = base_dir / "errors.jsonl"
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _pulse_lock() -> None:
    """Refresh the active snapshot lock heartbeat, ignoring failures."""
    lock = _get_active_lock()
    if lock is None:
        return
    with suppress(SnapshotLockError):
        lock.refresh()


def _manifest_path(snapshot_dir: Path) -> Path:
    return snapshot_dir / "manifest.jsonl"


def _manifest_checksum_path(snapshot_dir: Path) -> Path:
    return snapshot_dir / "manifest.checksum"


_LOCK_STATE: dict[str, SnapshotLock | None] = {"active": None}


def _get_active_lock() -> SnapshotLock | None:
    """Return the currently active snapshot lock, if any."""
    return cast(SnapshotLock | None, _LOCK_STATE.get("active"))


def _set_active_lock(lock: SnapshotLock | None) -> None:
    """Update the active lock reference."""
    _LOCK_STATE["active"] = lock


def begin_snapshot(base_dir: Path | None = None) -> Path:
    """Create a locked workspace for snapshot persistence."""
    active_lock = _get_active_lock()
    if active_lock is not None:
        # Clear stale in-memory lock references that outlive the lock file.
        if not active_lock.path.exists():
            _set_active_lock(None)
        else:
            raise RuntimeError("Snapshot lock already held; finalize or cleanup first.")

    paths = _snapshot_paths(base_dir)
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    config = settings.snapshots
    grace = max(1.0, float(config.lock_ttl_seconds) * 0.2)
    lock = SnapshotLock(
        paths.lock_file,
        timeout=float(config.lock_timeout_seconds),
        ttl_seconds=float(config.lock_ttl_seconds),
        grace_seconds=grace,
    )
    lock.acquire()
    _set_active_lock(lock)
    try:
        workspace = start_workspace(paths.base_dir)
    except (OSError, RuntimeError, ValueError, PermissionError):
        try:
            _release_active_lock()
        except SnapshotLockError as release_error:  # pragma: no cover - defensive
            logger.warning(
                "Failed to release snapshot lock after workspace error: %s",
                release_error,
            )
        raise
    return workspace.root


def _release_active_lock() -> None:
    """Release the active snapshot lock when held."""
    lock = _get_active_lock()
    if lock is None:
        return
    try:
        lock.release()
    finally:
        _set_active_lock(None)


def cleanup_tmp(tmp_dir: Path) -> None:
    """Best-effort cleanup of workspace and lock release."""
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    finally:
        _release_active_lock()


def write_manifest(tmp_dir: Path, manifest_meta: dict[str, Any]) -> None:
    """Write snapshot manifest (JSONL + checksum + metadata)."""
    workspace = Path(tmp_dir)
    snapshot_id = workspace.name
    _pulse_lock()
    with _tracer.start_as_current_span("snapshot.write_manifest"):
        try:
            _writer_write_manifest(workspace, manifest_meta)
        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:  # pragma: no cover - defensive
            _emit_snapshot_log(
                "write_manifest",
                status="failure",
                snapshot_id=snapshot_id,
                error=str(exc),
                error_code="manifest_write_failed",
                retryable=False,
            )
            _append_error_record(
                workspace.parent,
                {
                    "stage": "write_manifest",
                    "snapshot_id": snapshot_id,
                    "error": str(exc),
                    "error_code": "manifest_write_failed",
                },
            )
            raise SnapshotManifestError("Failed to write snapshot manifest") from exc
    _emit_snapshot_log("write_manifest", status="success", snapshot_id=snapshot_id)
    _pulse_lock()


def _write_current_pointer(paths: SnapshotPaths, version_name: str) -> None:
    """Update the ``CURRENT`` pointer to reference ``version_name``."""
    current_path = paths.base_dir / "CURRENT"
    _write_text_atomic(current_path, f"{version_name}\n")
    _fsync_dir(paths.base_dir)


def _garbage_collect(paths: SnapshotPaths) -> None:
    """Remove old snapshot versions respecting retention and grace period."""
    config = settings.snapshots
    keep = max(1, int(config.retention_count))
    grace = max(0, int(config.gc_grace_seconds))
    versions = sorted([
        p
        for p in paths.base_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_tmp-")
    ])
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
    """Rename workspace to versioned snapshot and update ``CURRENT``."""
    start = time.perf_counter()
    paths = _snapshot_paths(base_dir)
    if not tmp_dir.exists():  # pragma: no cover - defensive
        _release_active_lock()
        raise FileNotFoundError(f"tmp snapshot dir missing: {tmp_dir}")

    version_name = _generate_version_name()
    destination = paths.base_dir / version_name
    _pulse_lock()
    with _tracer.start_as_current_span("snapshot.finalize"):
        try:
            _fsync_dir(tmp_dir)
            os.replace(tmp_dir, destination)
            mark_manifest_complete(destination)
            _fsync_dir(destination)
            _fsync_dir(paths.base_dir)
            _write_current_pointer(paths, version_name)
            _garbage_collect(paths)
            duration = time.perf_counter() - start
            log_performance(
                operation="snapshot_finalize",
                success=True,
                duration_seconds=duration,
                version=version_name,
            )
            _emit_snapshot_log(
                "finalize",
                status="success",
                snapshot_id=version_name,
                duration_seconds=duration,
            )
            _pulse_lock()
            logger.info("Snapshot finalized at %s", destination)
            return destination
        except (
            OSError,
            SnapshotError,
            RuntimeError,
            ValueError,
        ) as exc:  # pragma: no cover - defensive
            duration = time.perf_counter() - start
            log_performance(
                operation="snapshot_finalize",
                success=False,
                duration_seconds=duration,
                error=str(exc),
            )
            _emit_snapshot_log(
                "finalize",
                status="failure",
                snapshot_id=version_name,
                error=str(exc),
                error_code="snapshot_finalize_failed",
                retryable=False,
            )
            _append_error_record(
                paths.base_dir,
                {
                    "stage": "finalize",
                    "snapshot_id": version_name,
                    "error": str(exc),
                    "error_code": "snapshot_finalize_failed",
                },
            )
            raise
        finally:
            _release_active_lock()


def latest_snapshot_dir(base_dir: Path | None = None) -> Path | None:
    """Return the most recent snapshot directory or ``None``."""
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
    target = snapshot_dir or latest_snapshot_dir(base_dir)
    if target is None:
        return None
    meta_path = target / "manifest.meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - defensive
        return None


def persist_vector_index(index: Any, out_dir: Path) -> None:
    """Persist a vector index into ``out_dir`` using LlamaIndex helpers."""
    snapshot_dir = out_dir.parent
    snapshot_id = snapshot_dir.name if snapshot_dir is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)
    _pulse_lock()
    with _tracer.start_as_current_span("snapshot.persist_vector"):
        try:
            index.storage_context.persist(  # type: ignore[attr-defined]
                persist_dir=str(out_dir)
            )
        except Exception as exc:
            _emit_snapshot_log(
                "persist_vector",
                status="failure",
                snapshot_id=snapshot_id,
                error=str(exc),
                error_code="vector_persist_failed",
                retryable=False,
            )
            _append_error_record(
                snapshot_dir or out_dir.parent,
                {
                    "stage": "persist_vector",
                    "snapshot_id": snapshot_id,
                    "error": str(exc),
                    "error_code": "vector_persist_failed",
                },
            )
            raise SnapshotPersistenceError(
                f"Persist vector index failed: {exc}"
            ) from exc
        _emit_snapshot_log("persist_vector", status="success", snapshot_id=snapshot_id)
        _pulse_lock()


def persist_graph_store(store: Any, out_dir: Path) -> None:
    """Persist a property graph store into ``out_dir`` in a library-first way."""
    snapshot_dir = out_dir.parent
    snapshot_id = snapshot_dir.name if snapshot_dir is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)
    _pulse_lock()
    with _tracer.start_as_current_span("snapshot.persist_graph"):
        try:
            try:
                store.persist(persist_dir=str(out_dir))  # type: ignore[attr-defined]
            except TypeError:
                store.persist(str(out_dir))
        except Exception as exc:
            _emit_snapshot_log(
                "persist_graph",
                status="failure",
                snapshot_id=snapshot_id,
                error=str(exc),
                error_code="graph_persist_failed",
                retryable=False,
            )
            _append_error_record(
                snapshot_dir or out_dir.parent,
                {
                    "stage": "persist_graph",
                    "snapshot_id": snapshot_id,
                    "error": str(exc),
                    "error_code": "graph_persist_failed",
                },
            )
            raise SnapshotPersistenceError(
                f"Persist graph store failed: {exc}"
            ) from exc
        _emit_snapshot_log("persist_graph", status="success", snapshot_id=snapshot_id)
        _pulse_lock()


def load_vector_index(snapshot_dir: Path | None = None) -> Any | None:
    """Load a persisted vector index from ``snapshot_dir`` when available."""
    try:
        from llama_index.core import StorageContext, load_index_from_storage
    except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
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
    ) as exc:  # pragma: no cover
        logger.debug("Unable to load vector index from %s: %s", vec_dir, exc)
        return None


def load_property_graph_index(snapshot_dir: Path | None = None) -> Any | None:
    """Load a persisted property graph index from ``snapshot_dir`` when available."""
    try:
        from llama_index.core import PropertyGraphIndex
        from llama_index.core.graph_stores import SimplePropertyGraphStore
    except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
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
    ) as exc:  # pragma: no cover
        logger.debug("Unable to load property graph index from %s: %s", graph_dir, exc)
        return None


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
    """Remove stale workspaces and repair the ``CURRENT`` pointer if missing."""
    paths = _snapshot_paths(base_dir)
    if not paths.base_dir.exists():
        return

    for candidate in paths.base_dir.glob("_tmp-*"):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
            logger.debug("Removed stale workspace %s", candidate)

    for stale_lock in paths.base_dir.glob(".lock.stale-*"):
        with suppress(FileNotFoundError):
            stale_lock.unlink()
    for stale_meta in paths.base_dir.glob(".lock.meta.json.stale-*"):
        with suppress(FileNotFoundError):
            stale_meta.unlink()

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

    versions = sorted([
        p
        for p in paths.base_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_tmp-")
    ])
    if not versions:
        with suppress(FileNotFoundError):
            paths.current_file.unlink()
        return

    latest = versions[-1].name
    _write_current_pointer(paths, latest)
    logger.info("Recovered CURRENT pointer -> %s", latest)


def _hash_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _entries_valid(snapshot_dir: Path, entries: list[dict[str, Any]]) -> bool:
    """Validate manifest entries against on-disk files and hashes."""
    for entry in entries:
        rel = entry.get("path")
        if not rel:
            return False
        target = snapshot_dir / Path(rel)
        if not target.exists():
            return False
        if _hash_file(target) != entry.get("sha256"):
            return False
    return True


def verify_snapshot(snapshot_dir: Path) -> bool:
    """Verify manifest hashes and payload integrity for a snapshot directory."""
    manifest_path = _manifest_path(snapshot_dir)
    checksum_path = _manifest_checksum_path(snapshot_dir)
    meta_path = snapshot_dir / "manifest.meta.json"
    if not (manifest_path.exists() and checksum_path.exists() and meta_path.exists()):
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checksum_payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - defensive
        return False

    entries = load_manifest_entries(snapshot_dir)
    expected = checksum_payload.get("manifest_sha256")
    if expected is None:
        return False

    if not _entries_valid(snapshot_dir, entries):
        return False

    aggregate = sha256()
    for entry in entries:
        aggregate.update(entry["sha256"].encode("utf-8"))
    aggregate.update(json.dumps(meta, sort_keys=True).encode("utf-8"))
    return aggregate.hexdigest() == expected


__all__ = [
    "SnapshotLock",
    "SnapshotLockTimeout",
    "SnapshotLockTimeoutError",
    "SnapshotPersistenceError",
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
    """Convenience wrapper for creating and persisting immutable snapshots.

    Snapshots store serialized vector indices, property graphs, and exports within
    an immutable directory structure. Per ADR-058, paths stored in manifest.jsonl
    are **snapshot-internal relative references only**—never absolute paths or
    stable external identifiers. These paths are confined to the snapshot boundary
    and are not exposed via the public API.

    Callers should use load_manifest_entries() to read manifest metadata, which
    validates all paths remain within the snapshot directory boundary.
    """

    def __init__(self, storage_dir: Path) -> None:
        """Initialize the manager with the provided storage directory.

        Args:
            storage_dir: Base directory used to store finalized snapshots.
        """
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        recover_snapshots(self.storage_dir)

    def begin_snapshot(self) -> Path:
        """Create and lock a workspace for snapshot persistence.

        Returns:
            Path: Root directory for the temporary snapshot workspace.
        """
        return begin_snapshot(self.storage_dir)

    def persist_vector_index(self, index: Any, tmp_dir: Path) -> None:
        """Persist the vector index artifact into the workspace.

        Args:
            index: In-memory vector index to serialize to disk.
            tmp_dir: Temporary snapshot workspace directory.
        """
        persist_vector_index(index, tmp_dir / "vector")

    def persist_graph_store(self, store: Any, tmp_dir: Path) -> None:
        """Persist the property graph store into the workspace.

        Args:
            store: Property graph store to export.
            tmp_dir: Temporary snapshot workspace directory.
        """
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
        graph_exports: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write manifest metadata describing the snapshot contents.

        Writes three files to tmp_dir: manifest.jsonl (entries), manifest.meta.json
        (metadata), and manifest.checksum (SHA256 of manifest.jsonl).

        **Important invariant (ADR-058):** Paths written to manifest.jsonl are
        snapshot-internal relative references (relative to tmp_dir), confined to
        the snapshot boundary, and are NOT stable external identifiers. These paths
        must never be persisted elsewhere or exposed outside the snapshot boundary.
        Use load_manifest_entries() to safely read entries with built-in path
        validation.

        Args:
            tmp_dir: Temporary snapshot workspace directory.
            index_id: Identifier representing the persisted index.
            graph_store_type: Storage backend used for graph data.
            vector_store_type: Storage backend used for vector data.
            corpus_hash: Content hash of the ingested corpus.
            config_hash: Hash summarizing configuration inputs.
            versions: Optional mapping of component versions.
            graph_exports: Optional metadata about packaged graph export files.
        """
        metadata = {
            "index_id": index_id,
            "graph_store_type": graph_store_type,
            "vector_store_type": vector_store_type,
            "corpus_hash": corpus_hash,
            "config_hash": config_hash,
            "versions": versions or {},
            "graph_exports": graph_exports or [],
        }
        write_manifest(tmp_dir, metadata)

    def finalize_snapshot(self, tmp_dir: Path) -> Path:
        """Promote the workspace to an immutable snapshot directory.

        Args:
            tmp_dir: Temporary snapshot workspace directory.

        Returns:
            Path: Path to the finalized snapshot directory.
        """
        return finalize_snapshot(tmp_dir, base_dir=self.storage_dir)

    def cleanup_tmp(self, tmp_dir: Path) -> None:
        """Remove the workspace directory and release the snapshot lock.

        Args:
            tmp_dir: Temporary snapshot workspace directory.
        """
        cleanup_tmp(tmp_dir)


__all__ += ["SnapshotManager"]
