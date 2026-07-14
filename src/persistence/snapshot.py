"""Atomic manifests and optional property-graph artifacts for live Qdrant data.

Application snapshots identify immutable physical Qdrant collections and package
graph artifacts for staleness-aware activation. Point-in-time vector data remains
owned by Qdrant backups.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from llama_index.core.graph_stores.types import DEFAULT_PG_PERSIST_FNAME
from loguru import logger
from opentelemetry import trace

from src.config.settings import settings
from src.persistence.hashing import compute_config_hash, compute_corpus_hash
from src.persistence.lockfile import (
    SnapshotLock,
    SnapshotLockError,
    SnapshotLockTimeoutError,
)
from src.persistence.snapshot_writer import (
    MANIFEST_SCHEMA_VERSION,
    hash_manifest,
    load_manifest_entries,
    mark_manifest_complete,
    start_workspace,
)
from src.persistence.snapshot_writer import (
    write_manifest as _writer_write_manifest,
)
from src.persistence.upload_journal import (
    cleanup_pending_uploads_after_restart,
    recover_upload_quarantines,
)
from src.utils.log_safety import build_pii_log_entry

_tracer = trace.get_tracer(__name__)

SnapshotLockTimeout = SnapshotLockTimeoutError
_SNAPSHOT_VERSION_RE = re.compile(r"\A\d{8}T\d{6}-[0-9a-f]{8}\Z")
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_ACTIVATION_JOURNAL_NAME = ".activation-transaction.json"
_ACTIVATION_JOURNAL_SCHEMA_VERSION = 1
_PROPERTY_GRAPH_NATIVE_PATHS = frozenset(
    {
        "graph/default__vector_store.json",
        "graph/docstore.json",
        "graph/graph_store.json",
        "graph/image__vector_store.json",
        "graph/index_store.json",
        f"graph/{DEFAULT_PG_PERSIST_FNAME}",
    }
)


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


def is_snapshot_version_name(name: str) -> bool:
    """Return whether ``name`` matches the generated snapshot directory format."""
    return _SNAPSHOT_VERSION_RE.fullmatch(name) is not None


def _fsync_dir(path: Path) -> None:
    """Durably flush a transaction directory or raise."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    """Durably flush a transaction file or raise."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _emit_snapshot_log(
    stage: str, *, status: str, snapshot_id: str | None = None, **extras: Any
) -> None:
    """Emit a structured log entry for snapshot operations."""
    logger.bind(
        snapshot_stage=stage, status=status, snapshot_id=snapshot_id, **extras
    ).info("snapshot.{}", stage)


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


@dataclass(slots=True)
class _SnapshotOwner:
    """Process-local ownership token binding one lock to one workspace."""

    lock: SnapshotLock
    workspace: Path | None = None


_LOCK_STATE_LOCK = threading.RLock()
_LOCK_STATE: dict[str, _SnapshotOwner | None] = {"active": None}


def _get_active_owner() -> _SnapshotOwner | None:
    """Return the active process-local owner under the state mutex."""
    with _LOCK_STATE_LOCK:
        return cast(_SnapshotOwner | None, _LOCK_STATE.get("active"))


def _get_active_lock() -> SnapshotLock | None:
    """Return the currently active snapshot lock, if any."""
    owner = _get_active_owner()
    return owner.lock if owner is not None else None


def _workspace_key(path: Path) -> Path:
    """Return a stable absolute key even after workspace promotion/removal."""
    return Path(path).resolve(strict=False)


def _new_snapshot_lock(paths: SnapshotPaths) -> SnapshotLock:
    """Build the canonical OS-fenced writer lock for one storage root."""
    config = settings.snapshots
    return SnapshotLock(
        paths.lock_file,
        timeout=float(config.lock_timeout_seconds),
        ttl_seconds=float(config.lock_ttl_seconds),
    )


def _recover_snapshot_transactions_locked(paths: SnapshotPaths) -> None:
    """Resolve filesystem and upload journals while the writer lock is held."""
    recover_snapshots(paths.base_dir)
    active_dir = _current_snapshot_dir(paths)
    active_manifest = (
        _load_complete_manifest(active_dir) if active_dir is not None else None
    )
    active_collections = (
        _manifest_collections(active_manifest) if active_manifest is not None else None
    )
    recover_upload_quarantines(
        data_dir=paths.base_dir.parent,
        active_collections=active_collections,
    )


def recover_snapshot_transactions(base_dir: Path | None = None) -> None:
    """Recover all activation/upload journals under the canonical writer lock."""
    paths = _snapshot_paths(base_dir)
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    with _new_snapshot_lock(paths):
        _recover_snapshot_transactions_locked(paths)
        cleanup_pending_uploads_after_restart(paths.base_dir.parent)


def begin_snapshot(base_dir: Path | None = None) -> Path:
    """Create a locked workspace for snapshot persistence."""
    with _LOCK_STATE_LOCK:
        if _LOCK_STATE.get("active") is not None:
            raise RuntimeError("Snapshot lock already held; finalize or cleanup first.")

        paths = _snapshot_paths(base_dir)
        paths.base_dir.mkdir(parents=True, exist_ok=True)
        lock = _new_snapshot_lock(paths)
        lock.acquire()
        owner = _SnapshotOwner(lock=lock)
        _LOCK_STATE["active"] = owner
        try:
            _recover_snapshot_transactions_locked(paths)
            _pulse_lock()
            workspace = start_workspace(paths.base_dir)
            owner.workspace = _workspace_key(workspace.root)
        except (OSError, RuntimeError, ValueError, PermissionError):
            try:
                _release_active_lock(expected_lock=lock)
            except SnapshotLockError as release_error:  # pragma: no cover - defensive
                redaction = build_pii_log_entry(
                    str(release_error), key_id="snapshot.release_lock"
                )
                logger.warning(
                    "Failed to release snapshot lock after workspace error "
                    "(error_type={}, error={})",
                    type(release_error).__name__,
                    redaction.redacted,
                )
            raise
        return workspace.root


def _release_active_lock(
    *,
    workspace: Path | None = None,
    expected_lock: SnapshotLock | None = None,
) -> None:
    """Release only the owner matching the supplied workspace or lock token."""
    with _LOCK_STATE_LOCK:
        owner = cast(_SnapshotOwner | None, _LOCK_STATE.get("active"))
        if owner is None:
            return
        if expected_lock is not None and owner.lock is not expected_lock:
            return
        if workspace is not None and owner.workspace != _workspace_key(workspace):
            return
        try:
            owner.lock.release()
        finally:
            if _LOCK_STATE.get("active") is owner:
                _LOCK_STATE["active"] = None


def cleanup_tmp(tmp_dir: Path) -> None:
    """Best-effort cleanup of workspace and lock release."""
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    finally:
        _release_active_lock(workspace=tmp_dir)


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
            redaction = build_pii_log_entry(str(exc), key_id="snapshot.write_manifest")
            _emit_snapshot_log(
                "write_manifest",
                status="failure",
                snapshot_id=snapshot_id,
                error=redaction.redacted,
                error_type=type(exc).__name__,
                error_code="manifest_write_failed",
                retryable=False,
            )
            _append_error_record(
                workspace.parent,
                {
                    "stage": "write_manifest",
                    "snapshot_id": snapshot_id,
                    "error_type": type(exc).__name__,
                    "error": redaction.redacted,
                    "error_code": "manifest_write_failed",
                },
            )
            raise SnapshotManifestError("Failed to write snapshot manifest") from exc
    _emit_snapshot_log("write_manifest", status="success", snapshot_id=snapshot_id)
    _pulse_lock()


def _write_current_pointer(paths: SnapshotPaths, version_name: str) -> None:
    """Update the ``CURRENT`` pointer to reference ``version_name``."""
    current_path = paths.base_dir / "CURRENT"
    tmp_path = current_path.with_suffix(current_path.suffix + ".tmp")
    tmp_path.write_text(f"{version_name}\n", encoding="utf-8")
    _fsync_file(tmp_path)
    os.replace(tmp_path, current_path)
    _fsync_file(current_path)
    _fsync_dir(paths.base_dir)


def _write_activation_journal(paths: SnapshotPaths, version_name: str) -> None:
    """Persist intent to promote one exact destination before its rename."""
    journal = paths.base_dir / _ACTIVATION_JOURNAL_NAME
    temporary = journal.with_suffix(".tmp")
    payload = {
        "schema_version": _ACTIVATION_JOURNAL_SCHEMA_VERSION,
        "destination": version_name,
    }
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, journal)
    _fsync_dir(paths.base_dir)


def _read_activation_journal(paths: SnapshotPaths) -> str | None:
    """Read one trusted pending activation destination."""
    journal = paths.base_dir / _ACTIVATION_JOURNAL_NAME
    if not journal.exists() and not journal.is_symlink():
        return None
    if journal.is_symlink() or not journal.is_file():
        raise SnapshotPersistenceError("Snapshot activation journal is unsafe")
    try:
        payload: object = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotPersistenceError(
            "Snapshot activation journal is unreadable"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _ACTIVATION_JOURNAL_SCHEMA_VERSION
        or not isinstance(payload.get("destination"), str)
        or not is_snapshot_version_name(payload["destination"])
    ):
        raise SnapshotPersistenceError("Snapshot activation journal is invalid")
    return cast(str, payload["destination"])


def _clear_activation_journal(paths: SnapshotPaths) -> None:
    """Retire activation intent after commit or complete rollback."""
    journal = paths.base_dir / _ACTIVATION_JOURNAL_NAME
    journal.unlink(missing_ok=True)
    journal.with_suffix(".tmp").unlink(missing_ok=True)
    _fsync_dir(paths.base_dir)


def _current_pointer_name(paths: SnapshotPaths) -> str | None:
    """Read a canonical plain-file CURRENT name without loading its snapshot."""
    if paths.current_file.is_symlink() or not paths.current_file.is_file():
        return None
    try:
        name = paths.current_file.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return name if is_snapshot_version_name(name) else None


def _retire_committed_upload_journals_locked(
    paths: SnapshotPaths,
    destination: Path,
) -> None:
    """Resolve source journals only after the durable CURRENT commit point."""
    manifest = _load_complete_manifest(destination)
    collections = _manifest_collections(manifest) if manifest is not None else None
    if collections is None:
        raise SnapshotPersistenceError(
            "Committed snapshot has no physical collection identity"
        )
    recover_upload_quarantines(
        data_dir=paths.base_dir.parent,
        active_collections=collections,
    )


def _recover_activation_journal(paths: SnapshotPaths) -> None:
    """Finish journal retirement or delete an uncommitted promoted destination."""
    journal = paths.base_dir / _ACTIVATION_JOURNAL_NAME
    temporary = journal.with_suffix(".tmp")
    if not journal.exists() and not journal.is_symlink():
        temporary.unlink(missing_ok=True)
        return
    version_name = _read_activation_journal(paths)
    if version_name is None:  # pragma: no cover - guarded above
        return
    destination = paths.base_dir / version_name
    if (
        _current_pointer_name(paths) == version_name
        and _load_complete_manifest(destination) is not None
    ):
        _clear_activation_journal(paths)
        return
    if destination.is_symlink():
        raise SnapshotPersistenceError("Pending snapshot destination is unsafe")
    if destination.exists():
        if not destination.is_dir():
            raise SnapshotPersistenceError("Pending snapshot destination is invalid")
        shutil.rmtree(destination)
        _fsync_dir(paths.base_dir)
    _clear_activation_journal(paths)


def _garbage_collect(
    paths: SnapshotPaths,
    *,
    protected_paths: tuple[Path, ...] = (),
) -> None:
    """Remove old snapshot versions respecting retention and grace period."""
    config = settings.snapshots
    keep = max(1, int(config.retention_count))
    grace = max(0, int(config.gc_grace_seconds))
    current_name: str | None = None
    if paths.current_file.exists():
        try:
            current_name = paths.current_file.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            logger.warning(
                "Snapshot retention skipped because CURRENT is unreadable "
                "(error_type={})",
                type(exc).__name__,
            )
            return
        if not is_snapshot_version_name(current_name):
            logger.warning("Snapshot retention skipped because CURRENT is invalid")
            return
    versions = sorted(
        p
        for p in paths.base_dir.iterdir()
        if p.is_dir()
        and not p.is_symlink()
        and is_snapshot_version_name(p.name)
        and _load_complete_manifest(p) is not None
    )
    if len(versions) <= keep:
        return
    if current_name is not None and not any(p.name == current_name for p in versions):
        logger.warning(
            "Snapshot retention skipped because CURRENT is not a complete snapshot"
        )
        return
    protected = set(versions[-keep:])
    protected.update(protected_paths)
    if current_name is not None:
        protected.add(paths.base_dir / current_name)
    cutoff = datetime.now(UTC) - timedelta(seconds=grace)
    with _tracer.start_as_current_span("snapshot.retention") as span:
        deleted = 0
        for candidate in versions:
            if candidate in protected:
                continue
            if (
                grace > 0
                and datetime.fromtimestamp(candidate.stat().st_mtime, UTC) > cutoff
            ):
                continue
            try:
                shutil.rmtree(candidate)
            except OSError as exc:
                redaction = build_pii_log_entry(
                    str(exc), key_id="snapshot.retention_delete"
                )
                _emit_snapshot_log(
                    "retention",
                    status="failure",
                    snapshot_id=candidate.name,
                    error_type=type(exc).__name__,
                    error=redaction.redacted,
                )
                continue
            deleted += 1
            _emit_snapshot_log(
                "retention",
                status="success",
                snapshot_id=candidate.name,
                action="deleted",
            )
        span.set_attribute("snapshot.retention.deleted_count", deleted)


def finalize_snapshot(  # noqa: PLR0915
    tmp_dir: Path,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Promote a workspace and commit activation by replacing ``CURRENT``.

    Args:
        tmp_dir: Temporary snapshot workspace directory.
        base_dir: Optional snapshot storage root.

    Returns:
        Path: Path to the finalized snapshot directory.
    """
    start = time.perf_counter()
    paths = _snapshot_paths(base_dir)
    if not tmp_dir.exists():  # pragma: no cover - defensive
        _release_active_lock(workspace=tmp_dir)
        raise FileNotFoundError(f"tmp snapshot dir missing: {tmp_dir}")

    version_name = _generate_version_name()
    destination = paths.base_dir / version_name
    committed = False
    _pulse_lock()
    try:
        with _tracer.start_as_current_span("snapshot.finalize"):
            _fsync_dir(tmp_dir)
            _write_activation_journal(paths, version_name)
            os.replace(tmp_dir, destination)
            mark_manifest_complete(destination)
            _fsync_dir(destination)
            _fsync_dir(paths.base_dir)
            if not verify_snapshot(destination):
                raise SnapshotPromotionError(
                    "Finalized snapshot failed manifest and payload verification"
                )
            _write_current_pointer(paths, version_name)
            committed = True
            _retire_committed_upload_journals_locked(paths, destination)
            with suppress(Exception):
                _clear_activation_journal(paths)
            with suppress(Exception):
                _garbage_collect(paths, protected_paths=(destination,))
            duration = time.perf_counter() - start
            with suppress(Exception):
                _emit_snapshot_log(
                    "finalize",
                    status="success",
                    snapshot_id=version_name,
                    duration_seconds=duration,
                )
                _pulse_lock()
                logger.info("Snapshot finalized at {}", destination.name)
        return destination
    except Exception as exc:  # pragma: no cover - defensive transaction boundary
        if committed:
            logger.warning(
                "Post-commit snapshot cleanup deferred (error_type={})",
                type(exc).__name__,
            )
            return destination
        if (
            _current_pointer_name(paths) == version_name
            and _load_complete_manifest(destination) is not None
        ):
            committed = True
            logger.warning(
                "CURRENT names a verified snapshot but durability confirmation "
                "failed; activation journals were retained for startup recovery"
            )
            return destination
        try:
            if destination.is_symlink():
                raise SnapshotPersistenceError(
                    "Uncommitted snapshot destination became unsafe"
                )
            if destination.exists():
                shutil.rmtree(destination)
                _fsync_dir(paths.base_dir)
            _clear_activation_journal(paths)
        except Exception as cleanup_error:
            logger.error(
                "Uncommitted snapshot rollback deferred (error_type={})",
                type(cleanup_error).__name__,
            )
        duration = time.perf_counter() - start
        redaction = build_pii_log_entry(str(exc), key_id="snapshot.finalize")
        with suppress(Exception):
            _emit_snapshot_log(
                "finalize",
                status="failure",
                snapshot_id=version_name,
                error=redaction.redacted,
                error_type=type(exc).__name__,
                error_code="snapshot_finalize_failed",
                retryable=False,
            )
            _append_error_record(
                paths.base_dir,
                {
                    "stage": "finalize",
                    "snapshot_id": version_name,
                    "error_type": type(exc).__name__,
                    "error": redaction.redacted,
                    "error_code": "snapshot_finalize_failed",
                },
            )
        raise
    finally:
        if committed:
            with suppress(Exception):
                _release_active_lock(workspace=tmp_dir)
        else:
            _release_active_lock(workspace=tmp_dir)


def _load_complete_manifest(snapshot_dir: Path) -> dict[str, Any] | None:
    """Load metadata only for a finalized, checksum-verified snapshot."""
    meta_path = snapshot_dir / "manifest.meta.json"
    try:
        payload: object = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        return None
    if not verify_snapshot(snapshot_dir):
        return None
    return cast(dict[str, Any], payload)


def _manifest_collections(manifest: dict[str, Any]) -> dict[str, str] | None:
    """Return canonical physical collection names from a manifest."""
    collections = manifest.get("collections")
    if not isinstance(collections, dict):
        return None
    text = collections.get("text")
    image = collections.get("image")
    if not isinstance(text, str) or not text or not isinstance(image, str) or not image:
        return None
    return {"text": text, "image": image}


def _manifest_semantics_valid(
    manifest: dict[str, Any],
    entries: list[dict[str, Any]] | None = None,
) -> bool:
    """Validate the canonical snapshot metadata contract in one place."""
    activation_config = manifest.get("activation_config")
    activation_config_hash = manifest.get("activation_config_hash")
    graph_store_type = manifest.get("graph_store_type")
    if (
        not isinstance(manifest.get("index_id"), str)
        or not manifest["index_id"]
        or graph_store_type not in {"none", "property_graph"}
        or manifest.get("vector_store_type") != "qdrant"
        or _manifest_collections(manifest) is None
        or not isinstance(manifest.get("corpus_hash"), str)
        or _SHA256_RE.fullmatch(manifest["corpus_hash"]) is None
        or not isinstance(manifest.get("config_hash"), str)
        or _SHA256_RE.fullmatch(manifest["config_hash"]) is None
        or not isinstance(manifest.get("versions"), dict)
        or not isinstance(manifest.get("graph_exports"), list)
        or not isinstance(manifest.get("collection_metadata"), dict)
        or not isinstance(activation_config, dict)
        or not isinstance(activation_config_hash, str)
        or _SHA256_RE.fullmatch(activation_config_hash) is None
        or compute_config_hash(activation_config) != activation_config_hash
    ):
        return False
    if entries is None:
        return True
    entry_by_path = {
        entry["path"]: entry for entry in entries if isinstance(entry.get("path"), str)
    }
    graph_paths = {path for path in entry_by_path if Path(path).parts[:1] == ("graph",)}
    if graph_store_type == "property_graph":
        export_paths: set[str] = set()
        exports_valid = True
        for export in manifest["graph_exports"]:
            if not isinstance(export, dict):
                exports_valid = False
                break
            filename = export.get("filename")
            size_bytes = export.get("size_bytes")
            checksum = export.get("sha256")
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or filename in {".", ".."}
                or not isinstance(size_bytes, int)
                or isinstance(size_bytes, bool)
                or size_bytes < 0
                or not isinstance(checksum, str)
                or _SHA256_RE.fullmatch(checksum) is None
            ):
                exports_valid = False
                break
            path = f"graph/{filename}"
            entry = entry_by_path.get(path)
            if (
                path in export_paths
                or path in _PROPERTY_GRAPH_NATIVE_PATHS
                or entry is None
                or entry.get("size_bytes") != size_bytes
                or entry.get("sha256") != checksum
            ):
                exports_valid = False
                break
            export_paths.add(path)
        return exports_valid and graph_paths == (
            _PROPERTY_GRAPH_NATIVE_PATHS | export_paths
        )
    return not graph_paths and not manifest["graph_exports"]


def _current_snapshot_dir(paths: SnapshotPaths) -> Path | None:
    """Resolve a verified snapshot only through the authoritative CURRENT file."""
    if not paths.current_file.is_file() or paths.current_file.is_symlink():
        return None
    try:
        name = paths.current_file.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not is_snapshot_version_name(name):
        return None
    candidate = paths.base_dir / name
    if candidate.is_symlink() or not candidate.is_dir():
        return None
    return candidate if _load_complete_manifest(candidate) is not None else None


def latest_snapshot_dir(base_dir: Path | None = None) -> Path | None:
    """Return the verified snapshot referenced by ``CURRENT``, or ``None``."""
    paths = _snapshot_paths(base_dir)
    if not paths.base_dir.exists():
        return None
    return _current_snapshot_dir(paths)


def load_manifest(
    snapshot_dir: Path | None = None, *, base_dir: Path | None = None
) -> dict[str, Any] | None:
    """Load metadata for a complete snapshot."""
    target = snapshot_dir or latest_snapshot_dir(base_dir)
    if target is None:
        return None
    return _load_complete_manifest(target)


def persist_graph_storage_context(storage_context: Any, out_dir: Path) -> None:
    """Persist the complete native property-graph context into ``out_dir``."""
    snapshot_dir = out_dir.parent
    snapshot_id = snapshot_dir.name if snapshot_dir is not None else None
    out_dir.mkdir(parents=True, exist_ok=True)
    _pulse_lock()
    with _tracer.start_as_current_span("snapshot.persist_graph"):
        try:
            from llama_index.core import StorageContext

            storage_context.persist(persist_dir=str(out_dir))  # type: ignore[attr-defined]
            loaded = StorageContext.from_defaults(persist_dir=str(out_dir))
            if loaded.property_graph_store is None or loaded.vector_store is None:
                raise ValueError("Persisted graph context is incomplete")
            native_paths = {
                path.relative_to(snapshot_dir).as_posix()
                for path in out_dir.iterdir()
                if path.is_file() and not path.is_symlink()
            }
            if native_paths != _PROPERTY_GRAPH_NATIVE_PATHS:
                raise ValueError(
                    "Persisted graph context has an unexpected native shape"
                )
        except Exception as exc:
            redaction = build_pii_log_entry(str(exc), key_id="snapshot.persist_graph")
            _emit_snapshot_log(
                "persist_graph",
                status="failure",
                snapshot_id=snapshot_id,
                error=redaction.redacted,
                error_type=type(exc).__name__,
                error_code="graph_persist_failed",
                retryable=False,
            )
            _append_error_record(
                snapshot_dir or out_dir.parent,
                {
                    "stage": "persist_graph",
                    "snapshot_id": snapshot_id,
                    "error_type": type(exc).__name__,
                    "error": redaction.redacted,
                    "error_code": "graph_persist_failed",
                },
            )
            raise SnapshotPersistenceError("Persist graph store failed") from exc
        _emit_snapshot_log("persist_graph", status="success", snapshot_id=snapshot_id)
        _pulse_lock()


def load_vector_index(snapshot_dir: Path | None = None) -> Any | None:
    """Activate the verified manifest against the canonical live Qdrant owner.

    Application snapshots own manifests and graph artifacts. Complete
    point-in-time vector history is owned by the backup service's Qdrant
    collection snapshots, so activation never pretends a local LlamaIndex
    payload contains Qdrant data.
    """
    snap = snapshot_dir or latest_snapshot_dir()
    manifest = _load_complete_manifest(snap) if snap is not None else None
    collections = _manifest_collections(manifest) if manifest is not None else None
    if snap is None or collections is None:
        return None

    try:
        from llama_index.core import VectorStoreIndex

        from src.utils.storage import (
            close_vector_store_clients,
            connect_vector_store,
            sparse_retrieval_enabled,
        )
    except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
        return None

    vector_store: Any | None = None
    try:
        vector_store = connect_vector_store(
            collections["text"],
            _dense_embedding_size=settings.embedding.dimension,
            enable_hybrid=sparse_retrieval_enabled(settings),
        )
        return VectorStoreIndex.from_vector_store(cast(Any, vector_store))
    except Exception as exc:  # pragma: no cover - external storage boundary
        if vector_store is not None:
            close_vector_store_clients(vector_store)
        redaction = build_pii_log_entry(str(exc), key_id="snapshot.load_vector")
        logger.debug(
            "Unable to activate vector index (snapshot={} error_type={} error={})",
            snap.name,
            type(exc).__name__,
            redaction.redacted,
        )
        return None


def load_property_graph_index(  # noqa: PLR0911
    snapshot_dir: Path | None = None,
) -> Any | None:
    """Load a persisted property graph index from ``snapshot_dir`` when available."""
    snap = snapshot_dir or latest_snapshot_dir()
    if snap is None:
        return None
    manifest = _load_complete_manifest(snap)
    if manifest is None or manifest.get("graph_store_type") != "property_graph":
        return None
    try:
        entries = load_manifest_entries(snap)
    except (OSError, UnicodeError, TypeError, ValueError):
        return None
    if not any(
        isinstance(entry.get("path"), str)
        and Path(entry["path"]).parts[:1] == ("graph",)
        for entry in entries
    ):
        return None

    try:
        from llama_index.core import PropertyGraphIndex, StorageContext
    except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover
        return None

    graph_dir = snap / "graph"
    if graph_dir.is_symlink() or not graph_dir.is_dir():
        return None
    try:
        storage_context = StorageContext.from_defaults(persist_dir=str(graph_dir))
        store = storage_context.property_graph_store
        vector_store = storage_context.vector_store
        if store is None or vector_store is None:
            return None
        return PropertyGraphIndex.from_existing(
            property_graph_store=store,
            vector_store=vector_store,
            storage_context=storage_context,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        AttributeError,
    ) as exc:  # pragma: no cover
        redaction = build_pii_log_entry(str(exc), key_id="snapshot.load_graph")
        logger.debug(
            "Unable to load property graph index (dir={} error_type={} error={})",
            graph_dir.name,
            type(exc).__name__,
            redaction.redacted,
        )
        return None


def recover_snapshots(base_dir: Path | None = None) -> None:
    """Remove crash debris and discard invalid ``CURRENT`` pointers."""
    paths = _snapshot_paths(base_dir)
    if not paths.base_dir.exists():
        return

    _recover_activation_journal(paths)

    raw_current_name: str | None = None
    if paths.current_file.is_file() and not paths.current_file.is_symlink():
        with suppress(OSError, UnicodeError):
            candidate_name = paths.current_file.read_text(encoding="utf-8").strip()
            if (
                candidate_name
                and candidate_name not in {".", ".."}
                and Path(candidate_name).name == candidate_name
            ):
                raw_current_name = candidate_name

    for candidate in paths.base_dir.glob("_tmp-*"):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
            logger.debug("Removed stale workspace {}", candidate.name)

    for stale_lock in paths.base_dir.glob(".lock.stale-*"):
        with suppress(FileNotFoundError):
            stale_lock.unlink()
    for stale_meta in paths.base_dir.glob(".lock.meta.json.stale-*"):
        with suppress(FileNotFoundError):
            stale_meta.unlink()
    with suppress(FileNotFoundError):
        (paths.base_dir / "CURRENT.tmp").unlink()

    cutoff = datetime.now(UTC) - timedelta(
        seconds=max(0, int(settings.snapshots.gc_grace_seconds))
    )
    for candidate in paths.base_dir.iterdir():
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or not is_snapshot_version_name(candidate.name)
            or candidate.name == raw_current_name
            or _load_complete_manifest(candidate) is not None
        ):
            continue
        with suppress(OSError):
            if datetime.fromtimestamp(candidate.stat().st_mtime, UTC) > cutoff:
                continue
            shutil.rmtree(candidate)
            logger.warning("Removed unreferenced invalid snapshot {}", candidate.name)

    current_target = _current_snapshot_dir(paths)
    if current_target is not None:
        logger.debug("CURRENT pointer verified for {}", current_target.name)
        return
    if paths.current_file.exists() or paths.current_file.is_symlink():
        with suppress(OSError):
            paths.current_file.unlink()
        logger.warning("Discarded invalid CURRENT pointer")


def _hash_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _entries_valid(  # noqa: PLR0911
    snapshot_dir: Path, entries: list[dict[str, Any]]
) -> bool:
    """Validate manifest entries against on-disk files and hashes.

    Enforces path boundary: all resolved entry paths must remain within snapshot_dir
    to prevent directory traversal attacks (e.g., ../../../etc/passwd).
    """
    base = snapshot_dir.resolve()
    declared_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        rel = entry.get("path")
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        content_type = entry.get("content_type")
        relative_path = Path(rel) if isinstance(rel, str) else None
        if (
            not isinstance(rel, str)
            or not rel
            or relative_path is None
            or relative_path.is_absolute()
            or relative_path == Path(".")
            or ".." in relative_path.parts
            or relative_path.as_posix() != rel
            or rel in declared_paths
            or not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or not isinstance(content_type, str)
            or not content_type
        ):
            return False
        declared_paths.add(rel)
        try:
            target = (snapshot_dir / Path(rel)).resolve()
        except (
            OSError,
            RuntimeError,
        ):  # pragma: no cover - defense against symlink loops
            return False
        # Enforce boundary: target must be inside the snapshot directory
        try:
            target.relative_to(base)
        except ValueError:
            # Path is outside snapshot_dir boundary
            return False
        if not target.is_file() or target.is_symlink():
            return False
        try:
            if target.stat().st_size != expected_size:
                return False
            actual_hash = _hash_file(target)
        except OSError:
            return False
        if actual_hash != expected_hash:
            return False

    actual_paths: set[str] = set()
    try:
        for path in snapshot_dir.rglob("*"):
            if path.is_symlink():
                return False
            if path.is_dir():
                continue
            if not path.is_file():
                return False
            relative = path.relative_to(snapshot_dir).as_posix()
            if relative in {
                "manifest.jsonl",
                "manifest.meta.json",
                "manifest.checksum",
            }:
                continue
            actual_paths.add(relative)
    except OSError:
        return False
    return actual_paths == declared_paths


def verify_snapshot(snapshot_dir: Path) -> bool:  # noqa: PLR0911
    """Verify manifest hashes and payload integrity for a snapshot directory."""
    manifest_path = _manifest_path(snapshot_dir)
    checksum_path = _manifest_checksum_path(snapshot_dir)
    meta_path = snapshot_dir / "manifest.meta.json"
    manifest_artifacts = (manifest_path, checksum_path, meta_path)
    if any(path.is_symlink() or not path.is_file() for path in manifest_artifacts):
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checksum_payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(meta, dict)
        or meta.get("complete") is not True
        or meta.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or not isinstance(meta.get("persist_format_version"), str)
        or not isinstance(meta.get("created_at"), str)
        or not isinstance(checksum_payload, dict)
        or checksum_payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
    ):
        return False

    try:
        entries = load_manifest_entries(snapshot_dir)
    except (OSError, UnicodeError, TypeError, ValueError):
        return False
    expected = checksum_payload.get("manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return False

    if not _entries_valid(snapshot_dir, entries):
        return False
    if not _manifest_semantics_valid(meta, entries):
        return False

    return hash_manifest(entries, meta) == expected


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
    "is_snapshot_version_name",
    "latest_snapshot_dir",
    "load_manifest",
    "load_property_graph_index",
    "load_vector_index",
    "persist_graph_storage_context",
    "recover_snapshot_transactions",
    "recover_snapshots",
    "verify_snapshot",
    "write_manifest",
]


class SnapshotManager:
    """Convenience wrapper for creating and persisting immutable snapshots.

    Snapshots store a manifest identifying physical Qdrant collections plus
    optional property graphs and exports in an immutable directory. Per ADR-058,
    paths stored in manifest.jsonl are **snapshot-internal relative references
    only**—never absolute paths or stable external identifiers. These paths are
    confined to the snapshot boundary and are not exposed via the public API.

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

    def begin_snapshot(self) -> Path:
        """Create and lock a workspace for snapshot persistence.

        Returns:
            Path: Root directory for the temporary snapshot workspace.
        """
        return begin_snapshot(self.storage_dir)

    def persist_graph_storage_context(
        self, storage_context: Any, tmp_dir: Path
    ) -> None:
        """Persist the complete property-graph context into the workspace.

        Args:
            storage_context: Property graph storage context to export.
            tmp_dir: Temporary snapshot workspace directory.
        """
        persist_graph_storage_context(storage_context, tmp_dir / "graph")

    def write_manifest(  # noqa: PLR0913
        self,
        tmp_dir: Path,
        *,
        index_id: str,
        graph_store_type: str,
        vector_store_type: str,
        text_collection: str,
        image_collection: str,
        corpus_hash: str,
        config_hash: str,
        versions: dict[str, Any] | None = None,
        graph_exports: list[dict[str, Any]] | None = None,
        collection_metadata: dict[str, Any] | None = None,
        activation_config: dict[str, Any] | None = None,
        activation_config_hash: str | None = None,
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
            text_collection: Immutable physical Qdrant text collection name.
            image_collection: Immutable physical Qdrant image collection name.
            corpus_hash: Content hash of the ingested corpus.
            config_hash: Hash summarizing configuration inputs.
            versions: Optional mapping of component versions.
            graph_exports: Optional metadata about packaged graph export files.
            collection_metadata: Optional immutable collection identity metadata.
            activation_config: Exact index-affecting configuration for this build.
            activation_config_hash: Integrity identity for build provenance.
        """
        if not text_collection or not image_collection:
            raise ValueError("Physical text and image collection names are required")
        metadata = {
            "index_id": index_id,
            "graph_store_type": graph_store_type,
            "vector_store_type": vector_store_type,
            "collections": {"text": text_collection, "image": image_collection},
            "collection_metadata": collection_metadata or {},
            "corpus_hash": corpus_hash,
            "config_hash": config_hash,
            "versions": versions or {},
            "graph_exports": graph_exports or [],
            "activation_config": activation_config or {},
            "activation_config_hash": activation_config_hash
            or compute_config_hash(activation_config or {}),
        }
        payload_entries = [
            {
                "path": path.relative_to(tmp_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
            for path in tmp_dir.rglob("*")
            if path.is_file()
            and path.name
            not in {"manifest.jsonl", "manifest.meta.json", "manifest.checksum"}
        ]
        if not _manifest_semantics_valid(metadata, payload_entries):
            raise ValueError("Snapshot manifest metadata contract is invalid")
        write_manifest(tmp_dir, metadata)

    def finalize_snapshot(
        self,
        tmp_dir: Path,
    ) -> Path:
        """Promote the workspace to an immutable snapshot directory.

        Args:
            tmp_dir: Temporary snapshot workspace directory.

        Returns:
            Path: Path to the finalized snapshot directory.
        """
        return finalize_snapshot(
            tmp_dir,
            base_dir=self.storage_dir,
        )

    def cleanup_tmp(self, tmp_dir: Path) -> None:
        """Remove the workspace directory and release the snapshot lock.

        Args:
            tmp_dir: Temporary snapshot workspace directory.
        """
        cleanup_tmp(tmp_dir)


__all__ += ["SnapshotManager"]
