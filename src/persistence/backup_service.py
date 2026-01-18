"""Local backup creation and rotation (SPEC-037).

This module provides the implementation used by `scripts/backup.py` and the
optional Settings UI entry point. Backups are local-only and should never log
secrets (notably `.env` contents or API keys).
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger
from qdrant_client.http import models as qmodels

from src.config.settings import DocMindSettings, settings
from src.utils.log_safety import build_pii_log_entry, safe_url_for_log
from src.utils.telemetry import log_jsonl


@dataclass(frozen=True, slots=True)
class QdrantSnapshotFile:
    """A single Qdrant collection snapshot captured into the backup.

    Args:
        collection: Name of the Qdrant collection.
        snapshot_name: Name of the snapshot file on the server.
        filename: Relative path to the snapshot file in the backup directory.
        size_bytes: Size of the snapshot in bytes.
        checksum: Optional checksum for integrity verification.
    """

    collection: str
    snapshot_name: str
    filename: str
    size_bytes: int
    checksum: str | None


@dataclass(frozen=True, slots=True)
class BackupResult:
    """Outcome of a backup run.

    Args:
        backup_dir: Path to the created backup directory.
        included: List of labels for artifacts included in the backup.
        bytes_written: Total bytes written to the backup directory.
        qdrant_snapshots: List of Qdrant collection snapshots captured.
        duration_ms: Total duration of the backup process in milliseconds.
        warnings: List of non-fatal warnings encountered during backup.
    """

    backup_dir: Path
    included: list[str]
    bytes_written: int
    qdrant_snapshots: list[QdrantSnapshotFile]
    duration_ms: float
    warnings: list[str]


def _utc_timestamp_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: Path) -> None:
    # Block symlink targets and symlink parents (existing dirs) to avoid writes
    # escaping the intended destination.
    for parent in (path, *path.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("Symlink destination blocked")
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> int:
    if src.is_symlink():
        raise ValueError(f"Symlink source blocked: {src}")
    _safe_mkdir(dst.parent)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def _copy_tree(src_dir: Path, dst_dir: Path) -> int:
    """Copy a directory tree without following symlinks.

    Returns:
        Total bytes written (best-effort based on destination sizes).
    """
    if src_dir.is_symlink():
        raise ValueError(f"Symlink source blocked: {src_dir}")
    bytes_written = 0
    for path in src_dir.rglob("*"):
        rel = path.relative_to(src_dir)
        dst = dst_dir / rel
        if path.is_symlink():
            raise ValueError(f"Symlink source blocked: {path}")
        if path.is_dir():
            _safe_mkdir(dst)
            continue
        if path.is_file():
            bytes_written += _copy_file(path, dst)
    return bytes_written


def _list_backup_dirs(backup_root: Path) -> list[Path]:
    if not backup_root.exists():
        return []
    out: list[Path] = []
    for child in backup_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("backup_"):
            out.append(child)
    return sorted(out, key=lambda p: p.name)


def prune_backups(backup_root: Path, *, keep_last: int) -> list[Path]:
    """Prune older backup directories beyond keep_last.

    Args:
        backup_root: Root directory containing backups.
        keep_last: Number of most recent backups to keep.

    Returns:
        A list of deleted backup directories (best-effort).
    """
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1")
    backups = _list_backup_dirs(backup_root)
    if len(backups) <= keep_last:
        return []

    to_delete = backups[: max(0, len(backups) - keep_last)]
    deleted: list[Path] = []
    for path in to_delete:
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        deleted.append(path)
    if deleted:
        _safe_log_jsonl(
            {
                "event": "backup_pruned",
                "deleted_count": len(deleted),
                "keep_last": keep_last,
            },
            key_id="backup.prune.telemetry",
        )
    return deleted


def _download_qdrant_snapshot(
    *,
    qdrant_url: str,
    api_key: str | None,
    collection: str,
    snapshot_name: str,
    dest_file: Path,
    timeout_s: int,
) -> int:
    base = qdrant_url.rstrip("/")
    url = (
        f"{base}/collections/{urllib.parse.quote(collection)}/snapshots/"
        f"{urllib.parse.quote(snapshot_name)}"
    )
    headers = {"api-key": api_key} if api_key else {}
    req = urllib.request.Request(url, method="GET", headers=headers)  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        _safe_mkdir(dest_file.parent)
        with dest_file.open("wb") as f:
            shutil.copyfileobj(resp, f)
    return dest_file.stat().st_size


def _qdrant_target_collections(cfg: DocMindSettings) -> list[str]:
    targets = [
        str(cfg.database.qdrant_collection),
        str(cfg.database.qdrant_image_collection),
    ]
    if cfg.semantic_cache.enabled and cfg.semantic_cache.provider == "qdrant":
        name = (cfg.semantic_cache.collection_name or "").strip() or "docmind_semcache"
        targets.append(name)
    return sorted({t for t in targets if t})


def _create_qdrant_snapshots(
    *,
    cfg: DocMindSettings,
    dest_dir: Path,
    warnings: list[str],
) -> tuple[list[QdrantSnapshotFile], int]:
    """Create and download Qdrant collection snapshots (best-effort)."""
    from src.utils.storage import create_sync_client

    qdrant_url = str(cfg.database.qdrant_url).strip()
    timeout_s = int(cfg.database.qdrant_timeout)
    bytes_written = 0
    api_key = (
        cfg.database.qdrant_api_key.get_secret_value()
        if cfg.database.qdrant_api_key is not None
        else None
    )
    if api_key is not None:
        api_key = api_key.strip() or None
    snapshots: list[QdrantSnapshotFile] = []

    with create_sync_client() as client:
        collections = {c.name for c in client.get_collections().collections}
        for collection in _qdrant_target_collections(cfg):
            if collection not in collections:
                warnings.append(f"qdrant: collection not found: {collection}")
                continue
            desc = client.create_snapshot(collection_name=collection)
            if not isinstance(desc, qmodels.SnapshotDescription):
                warnings.append(
                    f"qdrant: snapshot create returned no description: {collection}"
                )
                continue

            filename = str(desc.name)
            dest_file = dest_dir / "qdrant" / collection / filename
            bytes_written += _download_qdrant_snapshot(
                qdrant_url=qdrant_url,
                api_key=api_key,
                collection=collection,
                snapshot_name=str(desc.name),
                dest_file=dest_file,
                timeout_s=timeout_s,
            )
            snapshots.append(
                QdrantSnapshotFile(
                    collection=collection,
                    snapshot_name=str(desc.name),
                    filename=str(dest_file.relative_to(dest_dir)),
                    size_bytes=int(desc.size),
                    checksum=str(desc.checksum) if desc.checksum else None,
                )
            )
    return snapshots, bytes_written


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    _safe_mkdir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_log_jsonl(payload: dict[str, Any], *, key_id: str) -> None:
    """Emit a JSONL telemetry event without surfacing exceptions.

    Args:
        payload: JSON-serializable event payload.
        key_id: Fingerprint namespace for exception redaction.
    """
    try:
        log_jsonl(payload)
    except Exception as exc:  # pragma: no cover - best effort
        redaction = build_pii_log_entry(str(exc), key_id=key_id)
        logger.debug(
            "telemetry skipped (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )


def _include_file(
    *,
    source: Path,
    dest: Path,
    label: str,
    included: list[str],
    warnings: list[str],
    warn_missing: bool = False,
) -> int:
    if not source.exists():
        if warn_missing:
            warnings.append(f"{label}: missing: {source}")
        return 0
    included.append(label)
    return _copy_file(source, dest)


def _include_tree(
    *,
    source: Path,
    dest: Path,
    label: str,
    included: list[str],
    warnings: list[str],
    warn_missing: bool = False,
) -> int:
    if not source.exists():
        if warn_missing:
            warnings.append(f"{label}: missing: {source}")
        return 0
    included.append(label)
    return _copy_tree(source, dest)


def _populate_backup_workspace(
    *,
    cfg: DocMindSettings,
    tmp_dir: Path,
    include_uploads: bool,
    include_analytics: bool,
    include_logs: bool,
    include_env: bool,
    qdrant_snapshot: bool,
) -> tuple[int, list[str], list[str], list[QdrantSnapshotFile]]:
    """Copy configured artifacts into a backup workspace and write its manifest.

    Args:
        cfg: Application settings.
        tmp_dir: Temporary backup directory that will later be renamed.
        include_uploads: Include `data_dir/uploads/` when present.
        include_analytics: Include analytics DuckDB database when present.
        include_logs: Include `logs/` directory when present.
        include_env: Include `.env` file when present (contains secrets).
        qdrant_snapshot: Attempt to snapshot Qdrant collections (best-effort).

    Returns:
        tuple[int, list[str], list[str], list[QdrantSnapshotFile]]: A tuple of:
            (bytes_written, included_labels, warnings, qdrant_snapshot_files).
    """
    warnings: list[str] = []
    included: list[str] = []
    bytes_written = 0
    qdrant_files: list[QdrantSnapshotFile] = []

    # Cache DB (DuckDB KV store)
    cache_db = cfg.cache_dir / cfg.cache.filename
    bytes_written += _include_file(
        source=cache_db,
        dest=tmp_dir / "cache" / cache_db.name,
        label="cache_db",
        included=included,
        warnings=warnings,
        warn_missing=True,
    )

    # Snapshots + manifests
    storage_dir = cfg.data_dir / "storage"
    bytes_written += _include_tree(
        source=storage_dir,
        dest=tmp_dir / "data" / "storage",
        label="snapshots",
        included=included,
        warnings=warnings,
        warn_missing=True,
    )

    # Chat DB (small but user-relevant)
    chat_db = cfg.data_dir / "chat.db"
    bytes_written += _include_file(
        source=chat_db,
        dest=tmp_dir / "data" / chat_db.name,
        label="chat_db",
        included=included,
        warnings=warnings,
        warn_missing=False,
    )

    # Optional: uploads
    if include_uploads:
        uploads_dir = cfg.data_dir / "uploads"
        bytes_written += _include_tree(
            source=uploads_dir,
            dest=tmp_dir / "data" / "uploads",
            label="uploads",
            included=included,
            warnings=warnings,
            warn_missing=False,
        )

    # Optional: analytics
    if include_analytics:
        analytics_path = (
            cfg.analytics_db_path
            if cfg.analytics_db_path is not None
            else (cfg.data_dir / "analytics" / "analytics.duckdb")
        )
        analytics_path = Path(analytics_path)
        bytes_written += _include_file(
            source=analytics_path,
            dest=tmp_dir / "data" / "analytics" / analytics_path.name,
            label="analytics",
            included=included,
            warnings=warnings,
            warn_missing=False,
        )

    # Optional: logs
    if include_logs:
        logs_dir = Path("./logs")
        bytes_written += _include_tree(
            source=logs_dir,
            dest=tmp_dir / "logs",
            label="logs",
            included=included,
            warnings=warnings,
            warn_missing=False,
        )

    # Optional: .env (contains secrets)
    if include_env:
        env_path = Path("./.env")
        bytes_written += _include_file(
            source=env_path,
            dest=tmp_dir / ".env",
            label="env",
            included=included,
            warnings=warnings,
            warn_missing=False,
        )

    # Optional: Qdrant snapshots (best-effort)
    if qdrant_snapshot:
        try:
            qdrant_files, qdrant_bytes = _create_qdrant_snapshots(
                cfg=cfg, dest_dir=tmp_dir, warnings=warnings
            )
            bytes_written += qdrant_bytes
            if qdrant_files:
                included.append("qdrant_snapshots")
        except Exception as exc:  # pragma: no cover - fail open
            warnings.append(f"qdrant: snapshot failed: {exc.__class__.__name__}")

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "app_version": str(cfg.app_version),
        "included": included,
        "bytes_written": int(bytes_written),
        "qdrant": {
            "url": safe_url_for_log(str(cfg.database.qdrant_url)),
            "collections": [q.__dict__ for q in qdrant_files],
        },
        "warnings": warnings,
    }
    _write_manifest(tmp_dir / "manifest.json", manifest)

    return bytes_written, included, warnings, qdrant_files


def create_backup(
    *,
    dest_root: Path | None = None,
    include_uploads: bool = False,
    include_analytics: bool = False,
    include_logs: bool = False,
    include_env: bool = False,
    keep_last: int | None = None,
    qdrant_snapshot: bool = True,
    cfg: DocMindSettings | None = None,
) -> BackupResult:
    """Create a local backup directory and rotate older backups.

    Args:
        dest_root: Destination root directory. When None, defaults to
            `settings.data_dir/backups`.
        include_uploads: Include `data_dir/uploads/`.
        include_analytics: Include the analytics DuckDB database (if present).
        include_logs: Include `logs/` (best effort).
        include_env: Include `.env` (contains secrets).
        keep_last: Backups to retain; defaults to settings.backup_keep_last.
        qdrant_snapshot: Attempt Qdrant collection snapshots (best-effort).
        cfg: Optional settings override.

    Returns:
        BackupResult with paths, bytes written, and warnings.
    """
    cfg = cfg or settings
    if not cfg.backup_enabled:
        raise ValueError("Backups are disabled (set DOCMIND_BACKUP_ENABLED=true)")

    start = time.perf_counter()

    root = (dest_root or (cfg.data_dir / "backups")).expanduser()
    _safe_mkdir(root)

    keep_last_value = int(keep_last if keep_last is not None else cfg.backup_keep_last)
    if keep_last_value < 1:
        raise ValueError("keep_last must be >= 1")

    stamp = _utc_timestamp_compact()
    tmp_dir = root / f"tmp-backup_{stamp}_{uuid.uuid4().hex[:8]}"
    final_dir = root / f"backup_{stamp}"
    _safe_mkdir(tmp_dir)

    bytes_written, included, warnings, qdrant_files = _populate_backup_workspace(
        cfg=cfg,
        tmp_dir=tmp_dir,
        include_uploads=include_uploads,
        include_analytics=include_analytics,
        include_logs=include_logs,
        include_env=include_env,
        qdrant_snapshot=qdrant_snapshot,
    )

    # Finalize backup dir (atomic rename when possible).
    if final_dir.exists():
        raise ValueError(f"Backup directory already exists: {final_dir}")
    tmp_dir.rename(final_dir)

    deleted = prune_backups(root, keep_last=keep_last_value)
    duration_ms = (time.perf_counter() - start) * 1000.0

    _safe_log_jsonl(
        {
            "event": "backup_created",
            "path": str(final_dir),
            "included": included,
            "bytes_written": int(bytes_written),
            "duration_ms": round(duration_ms, 2),
            "pruned": len(deleted),
        },
        key_id="backup.created.telemetry",
    )

    return BackupResult(
        backup_dir=final_dir,
        included=included,
        bytes_written=int(bytes_written),
        qdrant_snapshots=qdrant_files,
        duration_ms=duration_ms,
        warnings=warnings,
    )


__all__ = [
    "BackupResult",
    "QdrantSnapshotFile",
    "create_backup",
    "prune_backups",
]
