"""Local backup creation and rotation (SPEC-037).

This module provides the implementation used by `scripts/backup.py` and the
optional Settings UI entry point. Backups are local-only and should never log
secrets (notably `.env` contents or API keys).
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
from loguru import logger
from qdrant_client.http import models as qmodels

from src.config.settings import DocMindSettings, settings
from src.config.settings_utils import (
    endpoint_url_allowed,
    parse_endpoint_allowlist_hosts,
)
from src.persistence.artifacts import ArtifactRef, ArtifactStore
from src.persistence.deployment_identity import (
    DEPLOYMENT_ID_FILENAME,
    read_deployment_id,
)
from src.persistence.hashing import compute_corpus_hash
from src.persistence.lockfile import SnapshotLock
from src.persistence.snapshot import (
    latest_snapshot_dir,
    load_manifest,
    recover_snapshots,
)
from src.persistence.upload_journal import recover_upload_quarantines
from src.utils.hashing import sha256_file
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
        checksum: Local SHA256 for integrity verification.
        point_count: Exact source collection point count while writers are stopped.
    """

    collection: str
    snapshot_name: str
    filename: str
    size_bytes: int
    checksum: str
    point_count: int


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
        maintenance_warnings: Cleanup debt that does not affect recoverability.
    """

    backup_dir: Path
    included: list[str]
    bytes_written: int
    qdrant_snapshots: list[QdrantSnapshotFile]
    duration_ms: float
    warnings: list[str]
    maintenance_warnings: list[str]


def _utc_timestamp_compact() -> str:
    """Return a compact UTC timestamp for backup directories.

    Returns:
        A compact UTC timestamp string in YYYYMMDD_HHMMSS format. This
        function does not raise exceptions.
    """
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: Path, *, exist_ok: bool = True) -> None:
    """Create a directory, blocking symlinks.

    Args:
        path: Directory to create.
        exist_ok: Whether an existing directory is accepted.

    Raises:
        ValueError: If the path or any parent is a symlink.
    """
    # Block symlink targets and symlink parents (existing dirs) to avoid writes
    # escaping the intended destination.
    for parent in (path, *path.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("Symlink destination blocked")
    path.mkdir(parents=True, exist_ok=exist_ok)


@contextmanager
def _backup_root_lock(backup_root: Path) -> Iterator[None]:
    """Serialize create/prune operations across threads and local processes."""
    lock_path = backup_root / ".backup.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    locked = False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        try:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _copy_file(src: Path, dst: Path) -> int:
    """Copy a file, blocking symlinks.

    Args:
        src: Source file to copy.
        dst: Destination file to create.

    Returns:
        int: Number of bytes written to dst.

    Raises:
        ValueError: If the source is a symlink.
    """
    if src.is_symlink():
        raise ValueError(f"Symlink source blocked: {src}")
    _safe_mkdir(dst.parent)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def _copy_tree(src_dir: Path, dst_dir: Path) -> int:
    """Copy a directory tree without following symlinks.

    Args:
        src_dir: Source directory to copy.
        dst_dir: Destination directory to create.

    Returns:
        Total bytes written (best-effort based on destination sizes).

    Raises:
        ValueError: If the source is a symlink.
    """
    if src_dir.is_symlink():
        raise ValueError(f"Symlink source blocked: {src_dir}")
    _safe_mkdir(dst_dir)
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


def _tree_inventory(root: Path) -> list[dict[str, str | int]]:
    """Return a deterministic content inventory for one copied tree."""
    inventory: list[dict[str, str | int]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if path.is_symlink():
            raise ValueError(f"Symlink source blocked: {path}")
        if not path.is_file():
            continue
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def _file_inventory(root: Path, path: Path) -> dict[str, str | int]:
    """Return the exact recovery identity for one copied standalone file."""
    if root.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError(f"Backup file is unsafe or missing: {path}")
    relative = path.relative_to(root).as_posix()
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verified_file_inventory(root: Path, payload: object) -> bool:
    """Verify one standalone backup file against its manifest identity."""
    if root.is_symlink() or not root.is_dir() or not isinstance(payload, dict):
        return False
    if set(payload) != {"path", "size_bytes", "sha256"}:
        return False
    relative_name = payload.get("path")
    size_bytes = payload.get("size_bytes")
    checksum = payload.get("sha256")
    if (
        not isinstance(relative_name, str)
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
        or not isinstance(checksum, str)
        or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
    ):
        return False
    relative = Path(relative_name)
    if (
        relative.is_absolute()
        or relative == Path(".")
        or ".." in relative.parts
        or relative.as_posix() != relative_name
    ):
        return False
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
        return (
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == size_bytes
            and hmac.compare_digest(sha256_file(path), checksum)
        )
    except (OSError, ValueError):
        return False


def _tree_corpus_hash(root: Path) -> str:
    """Hash one copied upload tree with the canonical activation algorithm."""
    paths = [path for path in root.rglob("*") if path.is_file()]
    return compute_corpus_hash(paths, base_dir=root)


def _verified_tree_inventory(root: Path, payload: object) -> bool:  # noqa: PLR0911
    """Verify a copied tree against its exact manifest inventory."""
    if root.is_symlink() or not root.is_dir() or not isinstance(payload, dict):
        return False
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        return False
    expected: dict[str, tuple[int, str]] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            return False
        relative_name = item.get("path")
        size_bytes = item.get("size_bytes")
        checksum = item.get("sha256")
        if (
            not isinstance(relative_name, str)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(checksum, str)
            or len(checksum) != 64
        ):
            return False
        relative = Path(relative_name)
        if (
            relative.is_absolute()
            or relative == Path(".")
            or ".." in relative.parts
            or relative.as_posix() in expected
        ):
            return False
        expected[relative.as_posix()] = (size_bytes, checksum.lower())

    actual: set[str] = set()
    try:
        for path in root.rglob("*"):
            if path.is_symlink():
                return False
            if not path.is_file():
                continue
            relative_name = path.relative_to(root).as_posix()
            actual.add(relative_name)
            expected_file = expected.get(relative_name)
            if expected_file is None:
                return False
            size_bytes, checksum = expected_file
            if path.stat().st_size != size_bytes or not hmac.compare_digest(
                sha256_file(path), checksum
            ):
                return False
    except OSError:
        return False
    return actual == set(expected)


def _fsync_file(path: Path) -> None:
    """Durably flush one completed backup file."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Durably flush directory entries before destructive retention."""
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Durably flush a complete workspace from leaves to root."""
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink backup payload blocked: {path}")
        if path.is_file():
            _fsync_file(path)
        elif path.is_dir():
            directories.append(path)
    for directory in sorted(
        directories, key=lambda item: len(item.parts), reverse=True
    ):
        _fsync_directory(directory)


def _verified_qdrant_capture(  # noqa: PLR0911
    backup_dir: Path,
    manifest: dict[str, Any],
) -> bool:
    """Verify exact activation collection snapshots still exist and match."""
    activation = manifest.get("activation")
    qdrant = manifest.get("qdrant")
    if not isinstance(activation, dict) or not isinstance(qdrant, dict):
        return False
    collections = activation.get("collections")
    snapshots = qdrant.get("collections")
    version = qdrant.get("version")
    if (
        not isinstance(collections, dict)
        or not isinstance(snapshots, list)
        or not isinstance(version, str)
        or not version.strip()
    ):
        return False
    required = {collections.get("text"), collections.get("image")}
    if None in required or not all(isinstance(item, str) and item for item in required):
        return False
    captured: set[str] = set()
    for item in snapshots:
        if not isinstance(item, dict):
            return False
        collection = item.get("collection")
        filename = item.get("filename")
        size_bytes = item.get("size_bytes")
        checksum = item.get("checksum")
        point_count = item.get("point_count")
        if (
            not isinstance(collection, str)
            or not isinstance(filename, str)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or not isinstance(point_count, int)
            or isinstance(point_count, bool)
            or point_count < 0
            or collection not in required
            or collection in captured
        ):
            return False
        relative = Path(filename)
        if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
            return False
        snapshot_file = backup_dir / relative
        if snapshot_file.is_symlink() or not snapshot_file.is_file():
            return False
        try:
            if snapshot_file.stat().st_size != size_bytes:
                return False
            if not hmac.compare_digest(sha256_file(snapshot_file), checksum.lower()):
                return False
        except OSError:
            return False
        captured.add(collection)
    return captured == required


def _retention_eligible_backup(  # noqa: PLR0911
    backup_dir: Path,
    manifest: dict[str, Any],
) -> bool:
    """Return whether a backup is still a complete recovery point."""
    included = manifest.get("included")
    required_labels = {
        "cache_db",
        "snapshots",
        "chat_db",
        "uploads",
        "deployment_identity",
        "qdrant_snapshots",
    }
    if (
        manifest.get("complete") is not True
        or not isinstance(manifest.get("app_version"), str)
        or not manifest["app_version"].strip()
        or manifest.get("warnings") != []
        or not isinstance(manifest.get("maintenance_warnings"), list)
        or not isinstance(included, list)
        or not required_labels <= set(included)
    ):
        return False
    databases = manifest.get("databases")
    if not isinstance(databases, dict) or set(databases) != {"cache_db", "chat_db"}:
        return False
    if not all(
        _verified_file_inventory(backup_dir, databases[label])
        for label in ("cache_db", "chat_db")
    ):
        return False
    activation = manifest.get("activation")
    if not isinstance(activation, dict):
        return False
    deployment_id = activation.get("deployment_id")
    try:
        if read_deployment_id(backup_dir / "data") != deployment_id:
            return False
    except (OSError, RuntimeError, ValueError):
        return False
    uploads = backup_dir / "data" / "uploads"
    if not _verified_tree_inventory(uploads, manifest.get("uploads")):
        return False
    artifacts_included = "artifacts" in included
    artifacts = backup_dir / "data" / "artifacts"
    if artifacts_included != artifacts.is_dir() or (
        artifacts_included
        and not _verified_tree_inventory(artifacts, manifest.get("artifacts"))
    ):
        return False
    snapshot_name = activation.get("snapshot")
    storage = backup_dir / "data" / "storage"
    active = latest_snapshot_dir(storage)
    if (
        not isinstance(snapshot_name, str)
        or active is None
        or active.name != snapshot_name
    ):
        return False
    active_manifest = load_manifest(active)
    if active_manifest is None or active_manifest.get("collections") != activation.get(
        "collections"
    ):
        return False
    expected_corpus_hash = active_manifest.get("corpus_hash")
    if not isinstance(expected_corpus_hash, str) or len(expected_corpus_hash) != 64:
        return False
    try:
        if not hmac.compare_digest(
            _tree_corpus_hash(uploads), expected_corpus_hash.lower()
        ):
            return False
    except OSError:
        return False
    return _verified_qdrant_capture(backup_dir, manifest)


def _backup_sort_key(path: Path, manifest: dict[str, Any]) -> tuple[datetime, str]:
    """Order backups by recorded creation time, including same-second runs."""
    raw_created = manifest.get("created_at")
    if isinstance(raw_created, str):
        try:
            created = datetime.fromisoformat(raw_created)
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            return created, path.name
        except ValueError:
            pass
    match = re.fullmatch(
        r"backup_(\d{8}_\d{6})(?:_[0-9a-f]{12})?",
        path.name,
    )
    if match is not None:
        created = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
        return created, path.name
    return datetime.min.replace(tzinfo=UTC), path.name


def _list_backup_dirs(backup_root: Path) -> list[Path]:
    """List backup directories in the root directory.

    Args:
        backup_root: Root directory containing backups.

    Returns:
        List of backup directories.
    """
    if not backup_root.exists():
        return []
    out: list[tuple[datetime, str, Path]] = []
    for child in backup_root.iterdir():
        if child.is_symlink() or not child.is_dir():
            continue
        if re.fullmatch(r"backup_\d{8}_\d{6}(?:_[0-9a-f]{12})?", child.name) is None:
            continue
        manifest_path = child / "manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(manifest, dict) and _retention_eligible_backup(child, manifest):
            created, name = _backup_sort_key(child, manifest)
            out.append((created, name, child))
    return [item[2] for item in sorted(out)]


def _prune_backups_unlocked(backup_root: Path, *, keep_last: int) -> list[Path]:
    """Prune older backup directories while the caller owns the root lock.

    Args:
        backup_root: Root directory containing backups.
        keep_last: Number of most recent backups to keep.

    Returns:
        A list of deleted backup directories (best-effort).

    Raises:
        ValueError: If keep_last is less than 1.
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
            deleted.append(path)
        except OSError as exc:
            redaction = build_pii_log_entry(str(exc), key_id="backup.prune.oserror")
            logger.warning(
                "Failed to delete backup directory (path={}, error_type={}, error={})",
                path,
                type(exc).__name__,
                redaction.redacted,
            )
            _safe_log_jsonl(
                {
                    "event": "backup_prune_failed",
                    "path": str(path),
                    "error_type": type(exc).__name__,
                    "error_fingerprint": redaction.fingerprint[:12],
                },
                key_id="backup.prune.failed.telemetry",
            )
    if deleted:
        _fsync_directory(backup_root)
        _safe_log_jsonl(
            {
                "event": "backup_pruned",
                "deleted_count": len(deleted),
                "keep_last": keep_last,
            },
            key_id="backup.prune.telemetry",
        )
    return deleted


def prune_backups(backup_root: Path, *, keep_last: int) -> list[Path]:
    """Prune older backup directories beyond keep_last under an OS lock."""
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1")
    _safe_mkdir(backup_root)
    with _backup_root_lock(backup_root):
        return _prune_backups_unlocked(backup_root, keep_last=keep_last)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject snapshot redirects so credentials never cross request origins."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def _open_qdrant_snapshot(
    request: urllib.request.Request,
    *,
    timeout_s: int,
) -> Any:
    """Open one snapshot request with redirects disabled."""
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout_s)  # nosec B310


def _download_qdrant_snapshot(
    *,
    qdrant_url: str,
    api_key: str | None,
    collection: str,
    snapshot_name: str,
    dest_file: Path,
    timeout_s: int,
    allow_remote_endpoints: bool,
    allowed_hosts: set[str],
) -> int:
    """Download a Qdrant snapshot.

    Args:
        qdrant_url: URL of the Qdrant server.
        api_key: API key for authentication.
        collection: Name of the collection to snapshot.
        snapshot_name: Name of the snapshot to download.
        dest_file: Destination file to create.
        timeout_s: Timeout in seconds.
        allow_remote_endpoints: Whether non-local endpoints are allowed
            without the endpoint allowlist.
        allowed_hosts: Canonical endpoint host allowlist for SSRF checks.

    Returns:
        The total size of the downloaded snapshot in bytes.

    Raises:
        urllib.error.URLError: If the network request fails or the server
            returns a non-success HTTP status code.
        OSError: If there is a failure creating the destination directory
            or writing the snapshot file to disk.
        ValueError: If a symlink blocks the destination path or the Qdrant
            URL is invalid.
    """
    base = qdrant_url.rstrip("/")
    parsed_base = urllib.parse.urlparse(base)
    if parsed_base.scheme not in {"http", "https"}:
        raise ValueError("Qdrant snapshot URL must use http or https")
    if not allow_remote_endpoints and not endpoint_url_allowed(
        base,
        allowed_hosts=allowed_hosts,
    ):
        raise ValueError("Qdrant snapshot URL is not allowed by endpoint policy")
    url = (
        f"{base}/collections/{urllib.parse.quote(collection)}/snapshots/"
        f"{urllib.parse.quote(snapshot_name)}"
    )
    headers = {"api-key": api_key} if api_key else {}
    # B310 is safe here: qdrant_url comes from typed DocMind settings, and the
    # scheme is validated above before the request is issued.
    req = urllib.request.Request(url, method="GET", headers=headers)  # noqa: S310 # nosec B310
    _safe_mkdir(dest_file.parent)
    partial_file = dest_file.with_name(f".{dest_file.name}.{uuid.uuid4().hex}.part")
    try:
        with (
            _open_qdrant_snapshot(req, timeout_s=timeout_s) as resp,
            partial_file.open("xb") as f,
        ):
            shutil.copyfileobj(resp, f)
        partial_file.replace(dest_file)
    except BaseException:
        partial_file.unlink(missing_ok=True)
        raise
    return dest_file.stat().st_size


def _validate_qdrant_snapshot(
    *,
    description: qmodels.SnapshotDescription,
    downloaded_bytes: int,
    dest_file: Path,
) -> None:
    """Reject and remove a snapshot whose size or advertised SHA256 is invalid."""
    expected_bytes = int(description.size)
    if downloaded_bytes != expected_bytes:
        dest_file.unlink(missing_ok=True)
        raise ValueError(
            "Qdrant snapshot size mismatch: "
            f"expected {expected_bytes}, downloaded {downloaded_bytes}"
        )
    expected_checksum = str(description.checksum or "").strip().lower()
    if not expected_checksum:
        return
    with dest_file.open("rb") as snapshot_file:
        actual_checksum = hashlib.file_digest(snapshot_file, "sha256").hexdigest()
    if hmac.compare_digest(actual_checksum, expected_checksum):
        return
    dest_file.unlink(missing_ok=True)
    raise ValueError("Qdrant snapshot SHA256 mismatch")


def _active_snapshot_state(
    storage_dir: Path,
) -> tuple[Path, dict[str, str], dict[str, Any]]:
    """Resolve verified CURRENT identities needed for a consistent backup."""
    active_dir = latest_snapshot_dir(storage_dir)
    if active_dir is None:
        raise RuntimeError("No verified CURRENT snapshot is available")
    manifest = load_manifest(active_dir)
    if manifest is None:
        raise RuntimeError("CURRENT manifest is unavailable")
    collections = manifest.get("collections")
    if not isinstance(collections, dict):
        raise RuntimeError("CURRENT manifest has no collection identities")
    text_collection = collections.get("text")
    image_collection = collections.get("image")
    if (
        not isinstance(text_collection, str)
        or not text_collection
        or not isinstance(image_collection, str)
        or not image_collection
        or text_collection == image_collection
    ):
        raise RuntimeError("CURRENT collection identities are invalid")
    corpus_hash = manifest.get("corpus_hash")
    collection_metadata = manifest.get("collection_metadata")
    if not isinstance(corpus_hash, str) or len(corpus_hash) != 64:
        raise RuntimeError("CURRENT manifest has no corpus identity")
    if not isinstance(collection_metadata, dict) or any(
        not isinstance(collection_metadata.get(owner), dict)
        for owner in ("text", "image")
    ):
        raise RuntimeError("CURRENT manifest has no immutable collection metadata")
    return (
        active_dir,
        {"text": text_collection, "image": image_collection},
        manifest,
    )


def _capture_active_snapshot(
    *,
    active_dir: Path,
    dest_dir: Path,
) -> tuple[int, str]:
    """Copy one resolved CURRENT snapshot while the writer lock is held."""
    destination = dest_dir / active_dir.name
    bytes_written = _copy_tree(active_dir, destination)
    current_path = dest_dir / "CURRENT"
    _safe_mkdir(current_path.parent)
    current_path.write_text(f"{active_dir.name}\n", encoding="utf-8")
    bytes_written += current_path.stat().st_size
    return bytes_written, active_dir.name


def _qdrant_target_collections(collections: dict[str, str]) -> list[str]:
    """Return the manifest-selected Qdrant collections to snapshot.

    Args:
        collections: Verified CURRENT manifest collection identities.

    Returns:
        Sorted list of unique Qdrant collection names to be snapshotted.
    """
    targets = [collections["text"], collections["image"]]
    return sorted(set(targets))


def _read_qdrant_server_version(client: Any, warnings: list[str]) -> str | None:
    """Read the snapshot source version without failing the whole backup."""
    try:
        return str(client.info().version).strip() or None
    except Exception as exc:
        redaction = build_pii_log_entry(str(exc), key_id="backup.qdrant_version")
        warnings.append(
            "qdrant: server version unavailable: "
            f"{exc.__class__.__name__}: {redaction.redacted}"
        )
        return None


def _delete_qdrant_snapshot(
    *,
    client: Any,
    collection: str,
    snapshot_name: str,
    maintenance_warnings: list[str],
) -> None:
    """Delete a temporary server snapshot and record unconfirmed cleanup."""
    try:
        deleted = client.delete_snapshot(
            collection_name=collection,
            snapshot_name=snapshot_name,
        )
        if deleted is not True:
            message = (
                "qdrant: server-side snapshot cleanup failed: "
                f"{collection}: deletion was not confirmed"
            )
            logger.warning("{}", message)
            maintenance_warnings.append(message)
    except Exception as exc:  # pragma: no cover
        redaction = build_pii_log_entry(str(exc), key_id="backup.qdrant_delete")
        logger.warning(
            "qdrant: failed to delete server-side snapshot: "
            "collection={}, error_type={}, error={}",
            collection,
            type(exc).__name__,
            redaction.redacted,
        )
        maintenance_warnings.append(
            "qdrant: server-side snapshot cleanup failed: "
            f"{collection}: {exc.__class__.__name__}: {redaction.redacted}"
        )


def _qdrant_single_node_snapshot_supported(
    client: Any,
    warnings: list[str],
) -> bool:
    """Fail closed when one endpoint cannot represent the whole deployment."""
    try:
        cluster_status = client.cluster_status()
    except Exception as exc:
        redaction = build_pii_log_entry(str(exc), key_id="backup.qdrant_topology")
        warnings.append(
            "qdrant: cluster topology unavailable; snapshots skipped: "
            f"{exc.__class__.__name__}: {redaction.redacted}"
        )
        return False
    status = str(getattr(cluster_status, "status", "")).strip().lower()
    if status == "disabled":
        return True
    peers = getattr(cluster_status, "peers", None)
    if status == "enabled" and isinstance(peers, dict) and len(peers) <= 1:
        return True
    warnings.append(
        "qdrant: distributed snapshots require one capture per node; "
        "single-endpoint backup skipped"
    )
    return False


def _qdrant_collection_matches_activation(
    *,
    client: Any,
    collection: str,
    owner: str,
    expected_metadata: dict[str, Any],
    deployment_id: str | None,
    warnings: list[str],
) -> bool:
    """Require one live collection to match CURRENT's immutable owner identity."""
    expected = expected_metadata.get(owner)
    if (
        not isinstance(expected, dict)
        or not isinstance(deployment_id, str)
        or expected.get("docmind_deployment_id") != deployment_id
        or expected.get("docmind_owner") != owner
    ):
        warnings.append(
            f"qdrant: CURRENT {owner} collection ownership metadata is invalid"
        )
        return False
    try:
        info = client.get_collection(collection_name=collection)
    except Exception as exc:
        redaction = build_pii_log_entry(
            str(exc), key_id="backup.qdrant_collection_metadata"
        )
        warnings.append(
            f"qdrant: {owner} collection metadata unavailable: "
            f"{exc.__class__.__name__}: {redaction.redacted}"
        )
        return False
    actual = getattr(getattr(info, "config", None), "metadata", None)
    if not isinstance(actual, dict) or actual != expected:
        warnings.append(
            f"qdrant: {owner} collection ownership metadata does not match CURRENT"
        )
        return False
    return True


def _create_qdrant_snapshots(  # noqa: PLR0915
    *,
    cfg: DocMindSettings,
    target_collections: dict[str, str],
    dest_dir: Path,
    warnings: list[str],
    maintenance_warnings: list[str],
    artifact_dir: Path | None = None,
    target_collection_metadata: dict[str, Any] | None = None,
    deployment_id: str | None = None,
) -> tuple[list[QdrantSnapshotFile], int, str | None]:
    """Create and download Qdrant collection snapshots (best-effort).

    Args:
        cfg: DocMind settings.
        target_collections: Collection identities captured from CURRENT.
        dest_dir: Destination directory for snapshots.
        warnings: List of warnings to append to.
        maintenance_warnings: Server cleanup debt that does not affect recovery.
        artifact_dir: Copied ArtifactStore root to validate against image payloads.
        target_collection_metadata: Immutable metadata recorded by CURRENT.
        deployment_id: Copied local deployment identity.

    Returns:
        Snapshot descriptions, total bytes written, and server version when
        available.
    """
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
    allowed_hosts = parse_endpoint_allowlist_hosts(cfg.security.endpoint_allowlist)

    with create_sync_client(cfg) as client:
        server_version = _read_qdrant_server_version(client, warnings)
        if not _qdrant_single_node_snapshot_supported(client, warnings):
            return [], 0, server_version
        collections = {c.name for c in client.get_collections().collections}
        owners = {collection: owner for owner, collection in target_collections.items()}
        for collection in _qdrant_target_collections(target_collections):
            if collection not in collections:
                warnings.append(f"qdrant: collection not found: {collection}")
                continue
            owner = owners.get(collection)
            if target_collection_metadata is not None:
                if owner is None:
                    warnings.append(
                        f"qdrant: collection has no activation owner: {collection}"
                    )
                    continue
                if not _qdrant_collection_matches_activation(
                    client=client,
                    collection=collection,
                    owner=owner,
                    expected_metadata=target_collection_metadata,
                    deployment_id=deployment_id,
                    warnings=warnings,
                ):
                    continue

            desc: qmodels.SnapshotDescription | None = None
            snapshot_name: str | None = None
            try:
                point_count = int(
                    client.count(collection_name=collection, exact=True).count
                )
                if (
                    artifact_dir is not None
                    and collection == target_collections["image"]
                ):
                    _validate_image_artifact_references(
                        client=client,
                        collection=collection,
                        expected_points=point_count,
                        artifact_dir=artifact_dir,
                        warnings=warnings,
                    )
                created = client.create_snapshot(collection_name=collection)
                if not isinstance(created, qmodels.SnapshotDescription):
                    snapshot_name = next(
                        (
                            str(value)
                            for value in (
                                getattr(created, "name", None),
                                getattr(created, "snapshot_name", None),
                                getattr(created, "snapshot_id", None),
                                getattr(created, "id", None),
                            )
                            if value
                        ),
                        None,
                    )
                    warnings.append(
                        f"qdrant: snapshot create returned no description: {collection}"
                    )
                    continue
                desc = created
                snapshot_name = str(desc.name)
                snapshot_path = Path(snapshot_name)
                if (
                    not snapshot_name
                    or snapshot_name in {".", ".."}
                    or snapshot_path.is_absolute()
                    or snapshot_path.name != snapshot_name
                    or "/" in snapshot_name
                    or "\\" in snapshot_name
                ):
                    warnings.append(
                        f"qdrant: invalid snapshot name: {collection}: {snapshot_name}"
                    )
                    continue

                dest_file = dest_dir / "qdrant" / collection / snapshot_name
                snapshot_bytes = _download_qdrant_snapshot(
                    qdrant_url=qdrant_url,
                    api_key=api_key,
                    collection=collection,
                    snapshot_name=snapshot_name,
                    dest_file=dest_file,
                    timeout_s=timeout_s,
                    allow_remote_endpoints=cfg.security.allow_remote_endpoints,
                    allowed_hosts=allowed_hosts,
                )
                _validate_qdrant_snapshot(
                    description=desc,
                    downloaded_bytes=snapshot_bytes,
                    dest_file=dest_file,
                )
                bytes_written += snapshot_bytes
                snapshots.append(
                    QdrantSnapshotFile(
                        collection=collection,
                        snapshot_name=snapshot_name,
                        filename=str(dest_file.relative_to(dest_dir)),
                        size_bytes=int(desc.size),
                        checksum=sha256_file(dest_file),
                        point_count=point_count,
                    )
                )
            except Exception as exc:
                redaction = build_pii_log_entry(
                    str(exc), key_id="backup.qdrant_snapshot"
                )
                warnings.append(
                    "qdrant: snapshot/download failed: "
                    f"{collection}: {exc.__class__.__name__}: "
                    f"{redaction.redacted}"
                )
            finally:
                if snapshot_name:
                    _delete_qdrant_snapshot(
                        client=client,
                        collection=collection,
                        snapshot_name=snapshot_name,
                        maintenance_warnings=maintenance_warnings,
                    )
    return snapshots, bytes_written, server_version


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write a manifest file to the specified path using UTF-8 encoding.

    Args:
        path: Destination file path.
        payload: JSON-serializable manifest data.

    Returns:
        None: No return value.

    Raises:
        TypeError: If the payload contains objects that are not JSON-serializable.
        ValueError: If JSON serialization fails for other reasons.
        OSError: If filesystem operations (mkdir or write) fail.
    """
    _safe_mkdir(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_log_jsonl(payload: dict[str, Any], *, key_id: str) -> None:
    """Emit a JSONL telemetry event without surfacing exceptions.

    Args:
        payload: JSON-serializable event payload.
        key_id: Fingerprint namespace for exception redaction.

    Returns:
        None.

    Raises:
        None. This function internally catches and logs all exceptions
        (including IOError/OSError, TypeError, and json.JSONDecodeError)
        to ensure telemetry failures do not crash the caller.
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


def _append_backup_warning(
    warnings: list[str],
    *,
    prefix: str,
    exc: Exception,
    key_id: str,
) -> None:
    """Append one redacted recoverability warning."""
    redaction = build_pii_log_entry(str(exc), key_id=key_id)
    warnings.append(f"{prefix}: {exc.__class__.__name__}: {redaction.redacted}")


def _unresolved_upload_transactions(data_dir: Path) -> list[str]:
    """Return journal roots that still contain ambiguous recovery state."""
    unresolved: list[str] = []
    for name in (".upload-transactions", ".quarantine"):
        root = data_dir / name
        if not root.exists() and not root.is_symlink():
            continue
        try:
            if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
                unresolved.append(name)
        except OSError:
            unresolved.append(name)
    return unresolved


def _include_file(
    *,
    source: Path,
    dest: Path,
    label: str,
    included: list[str],
    warnings: list[str],
    warn_missing: bool = False,
) -> int:
    """Include a file in the backup.

    Args:
        source: Source file to include.
        dest: Full destination file path (e.g., tmp_dir / ".env").
        label: Label for the file.
        included: List of included files.
        warnings: List of warnings to append to.
        warn_missing: Whether to warn if the file is missing.

    Returns:
        Size of the included file in bytes.

    Raises:
        ValueError: If the source is a symlink.
        OSError: If the source cannot be read or the destination cannot be
            written.
    """
    if not source.exists():
        if warn_missing:
            warnings.append(f"{label}: missing: {source}")
        return 0
    if not source.is_file():
        warnings.append(f"{label}: expected a file: {source}")
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
    """Include a directory tree in the backup.

    Args:
        source: Source directory to include.
        dest: Destination directory for the tree.
        label: Label for the tree.
        included: List of included files.
        warnings: List of warnings to append to.
        warn_missing: Whether to warn if the directory is missing.

    Returns:
        Size of the included directory in bytes.

    Raises:
        ValueError: If the source is a symlink.
        OSError: If the source tree cannot be read or written.
    """
    if not source.exists():
        if warn_missing:
            warnings.append(f"{label}: missing: {source}")
        return 0
    if not source.is_dir():
        warnings.append(f"{label}: expected a directory: {source}")
        return 0
    included.append(label)
    return _copy_tree(source, dest)


def _include_sqlite_database(
    *,
    source: Path,
    dest: Path,
    label: str,
    included: list[str],
    warnings: list[str],
    warn_missing: bool = False,
) -> int:
    """Create one transactionally consistent SQLite online backup."""
    if not source.exists():
        if warn_missing:
            warnings.append(f"{label}: missing: {source}")
        return 0
    if not source.is_file():
        warnings.append(f"{label}: expected a SQLite file: {source}")
        return 0
    if source.is_symlink():
        raise ValueError(f"Symlink source blocked: {source}")
    _safe_mkdir(dest.parent)
    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    source_conn = sqlite3.connect(source_uri, timeout=5.0, uri=True)
    try:
        with closing(sqlite3.connect(str(dest))) as dest_conn, dest_conn:
            source_conn.backup(dest_conn)
    finally:
        source_conn.close()
    included.append(label)
    return dest.stat().st_size


def _duckdb_path_literal(path: Path) -> str:
    """Return one safely quoted DuckDB SQL string literal for a local path."""
    return "'" + str(path).replace("'", "''") + "'"


def _include_duckdb_database(
    *,
    source: Path,
    dest: Path,
    label: str,
    included: list[str],
    warnings: list[str],
    warn_missing: bool = False,
) -> int:
    """Create a transactionally consistent DuckDB database copy.

    `COPY FROM DATABASE` reads through DuckDB's transaction layer, including
    committed WAL state, and recreates the complete schema in a fresh file.
    """
    if not source.exists():
        if warn_missing:
            warnings.append(f"{label}: missing: {source}")
        return 0
    if not source.is_file():
        warnings.append(f"{label}: expected a DuckDB file: {source}")
        return 0
    if source.is_symlink():
        raise ValueError(f"Symlink source blocked: {source}")
    if dest.exists():
        raise ValueError(f"DuckDB backup destination already exists: {dest}")
    _safe_mkdir(dest.parent)
    source_literal = _duckdb_path_literal(source)
    dest_literal = _duckdb_path_literal(dest)
    with duckdb.connect(":memory:") as copier:
        copier.execute(f"ATTACH {source_literal} AS source_db (READ_ONLY)")
        copier.execute(f"ATTACH {dest_literal} AS backup_db")
        copier.execute("COPY FROM DATABASE source_db TO backup_db")
        copier.execute("DETACH backup_db")
        copier.execute("DETACH source_db")
    included.append(label)
    return dest.stat().st_size


def _validate_image_artifact_references(  # noqa: PLR0912, PLR0915
    *,
    client: Any,
    collection: str,
    expected_points: int,
    artifact_dir: Path,
    warnings: list[str],
) -> None:
    """Require complete, canonical, content-addressed image artifacts."""
    payload_fields = [
        "image_artifact_id",
        "image_artifact_suffix",
        "thumbnail_artifact_id",
        "thumbnail_artifact_suffix",
    ]
    store: ArtifactStore | None = None
    refs: dict[ArtifactRef, Path] = {}
    malformed = 0
    scanned = 0
    offset: Any = None
    try:
        while True:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=payload_fields,
                with_vectors=False,
            )
            page = points or []
            scanned += len(page)
            for point in page:
                payload = getattr(point, "payload", None)
                if not isinstance(payload, dict):
                    malformed += 1
                    continue

                pairs = [
                    (
                        payload.get("image_artifact_id"),
                        payload.get("image_artifact_suffix"),
                    )
                ]
                thumbnail = (
                    payload.get("thumbnail_artifact_id"),
                    payload.get("thumbnail_artifact_suffix"),
                )
                if thumbnail != (None, None):
                    pairs.append(thumbnail)

                for artifact_id, suffix in pairs:
                    if not isinstance(artifact_id, str) or not isinstance(suffix, str):
                        malformed += 1
                        continue
                    ref = ArtifactRef(sha256=artifact_id, suffix=suffix)
                    try:
                        if store is None:
                            store = ArtifactStore(root=artifact_dir)
                        refs[ref] = store.resolve_path(ref)
                    except ValueError:
                        malformed += 1

            if next_offset is None:
                break
            if next_offset == offset:
                warnings.append("qdrant: image artifact scan pagination stalled")
                return
            offset = next_offset
    except Exception as exc:
        redaction = build_pii_log_entry(str(exc), key_id="backup.artifact_scroll")
        warnings.append(
            "qdrant: image artifact scan failed: "
            f"{exc.__class__.__name__}: {redaction.redacted}"
        )
        return

    if scanned != expected_points:
        warnings.append(
            f"qdrant: image artifact scan covered {scanned} of {expected_points} points"
        )
    if malformed:
        warnings.append(
            f"qdrant: image payload contains {malformed} malformed "
            "artifact reference(s)"
        )

    missing = 0
    mismatched = 0
    unreadable = 0
    unreadable_detail: tuple[str, str] | None = None
    for ref, path in refs.items():
        if not path.is_file():
            missing += 1
            continue
        try:
            digest = sha256_file(path)
        except OSError as exc:
            unreadable += 1
            if unreadable_detail is None:
                redaction = build_pii_log_entry(
                    str(exc), key_id="backup.artifact_digest"
                )
                unreadable_detail = (exc.__class__.__name__, redaction.redacted)
            continue
        if not hmac.compare_digest(digest, ref.sha256):
            mismatched += 1

    if missing:
        warnings.append(f"artifacts: {missing} missing copied artifact file(s)")
    if mismatched:
        warnings.append(f"artifacts: {mismatched} copied artifact digest mismatch(es)")
    if unreadable and unreadable_detail is not None:
        error_type, redacted = unreadable_detail
        warnings.append(
            f"artifacts: {unreadable} copied artifact file(s) unreadable: "
            f"{error_type}: {redacted}"
        )


def _populate_backup_workspace(  # noqa: PLR0912, PLR0915
    *,
    cfg: DocMindSettings,
    tmp_dir: Path,
    include_uploads: bool,
    include_analytics: bool,
    include_logs: bool,
    include_env: bool,
    qdrant_snapshot: bool,
) -> tuple[int, list[str], list[str], list[str], list[QdrantSnapshotFile]]:
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
        Bytes, included labels, recoverability warnings, maintenance warnings,
        and Qdrant snapshot files.
    """
    warnings: list[str] = []
    maintenance_warnings: list[str] = []
    included: list[str] = []
    bytes_written = 0
    qdrant_files: list[QdrantSnapshotFile] = []
    qdrant_version: str | None = None
    activation_snapshot: str | None = None
    target_collections: dict[str, str] | None = None
    target_collection_metadata: dict[str, Any] | None = None
    target_corpus_hash: str | None = None
    deployment_id: str | None = None
    upload_inventory: list[dict[str, str | int]] = []
    artifact_inventory: list[dict[str, str | int]] = []
    database_inventory: dict[str, dict[str, str | int]] = {}

    # Cache DB (DuckDB KV store)
    cache_db = cfg.cache.ingestion_db_path
    cache_dest = tmp_dir / "cache" / cache_db.name
    bytes_written += _include_duckdb_database(
        source=cache_db,
        dest=cache_dest,
        label="cache_db",
        included=included,
        warnings=warnings,
        warn_missing=True,
    )
    if "cache_db" in included:
        database_inventory["cache_db"] = _file_inventory(tmp_dir, cache_dest)

    # Content-addressed page images and thumbnails. A missing root is valid for
    # text-only corpora, but any existing custom/default root is part of restore.
    artifact_dir = (
        Path(cfg.artifacts.dir)
        if cfg.artifacts.dir is not None
        else (cfg.data_dir / "artifacts")
    )

    # One canonical lease spans recovery and every activation-owned artifact.
    lock_cfg = cfg.snapshots
    storage_dir = cfg.data_dir / "storage"
    try:
        snapshot_lock = SnapshotLock(
            storage_dir / ".lock",
            timeout=float(lock_cfg.lock_timeout_seconds),
            ttl_seconds=float(lock_cfg.lock_ttl_seconds),
        )
        with snapshot_lock:
            recover_snapshots(storage_dir)
            active_dir: Path | None = None
            try:
                active_dir, target_collections, activation_manifest = (
                    _active_snapshot_state(storage_dir)
                )
                raw_collection_metadata = activation_manifest.get("collection_metadata")
                raw_corpus_hash = activation_manifest.get("corpus_hash")
                if isinstance(raw_collection_metadata, dict):
                    target_collection_metadata = raw_collection_metadata
                if isinstance(raw_corpus_hash, str):
                    target_corpus_hash = raw_corpus_hash
            except Exception as exc:
                _append_backup_warning(
                    warnings,
                    prefix="snapshots: CURRENT capture failed",
                    exc=exc,
                    key_id="backup.active_snapshot",
                )
            try:
                recover_upload_quarantines(
                    data_dir=cfg.data_dir,
                    active_collections=target_collections,
                )
            except Exception as exc:
                _append_backup_warning(
                    warnings,
                    prefix="uploads: transaction recovery incomplete",
                    exc=exc,
                    key_id="backup.upload_recovery",
                )
            unresolved_uploads = _unresolved_upload_transactions(cfg.data_dir)
            if unresolved_uploads and not any(
                item.startswith("uploads: transaction recovery incomplete")
                for item in warnings
            ):
                warnings.append(
                    "uploads: transaction recovery incomplete: "
                    + ", ".join(unresolved_uploads)
                )
            if active_dir is not None:
                try:
                    captured_bytes, activation_snapshot = _capture_active_snapshot(
                        active_dir=active_dir,
                        dest_dir=tmp_dir / "data" / "storage",
                    )
                    bytes_written += captured_bytes
                    included.append("snapshots")
                except Exception as exc:
                    _append_backup_warning(
                        warnings,
                        prefix="snapshots: CURRENT capture failed",
                        exc=exc,
                        key_id="backup.active_snapshot",
                    )

            try:
                deployment_id = read_deployment_id(cfg.data_dir)
                bytes_written += _include_file(
                    source=cfg.data_dir / DEPLOYMENT_ID_FILENAME,
                    dest=tmp_dir / "data" / DEPLOYMENT_ID_FILENAME,
                    label="deployment_identity",
                    included=included,
                    warnings=warnings,
                    warn_missing=True,
                )
            except Exception as exc:
                _append_backup_warning(
                    warnings,
                    prefix="deployment identity capture failed",
                    exc=exc,
                    key_id="backup.deployment_identity",
                )

            try:
                bytes_written += _include_tree(
                    source=artifact_dir,
                    dest=tmp_dir / "data" / "artifacts",
                    label="artifacts",
                    included=included,
                    warnings=warnings,
                    warn_missing=False,
                )
                if "artifacts" in included:
                    artifact_inventory = _tree_inventory(tmp_dir / "data" / "artifacts")
            except Exception as exc:
                _append_backup_warning(
                    warnings,
                    prefix="artifacts: capture failed",
                    exc=exc,
                    key_id="backup.artifacts",
                )

            if include_uploads:
                try:
                    bytes_written += _include_tree(
                        source=cfg.data_dir / "uploads",
                        dest=tmp_dir / "data" / "uploads",
                        label="uploads",
                        included=included,
                        warnings=warnings,
                        warn_missing=True,
                    )
                    if "uploads" in included:
                        copied_uploads = tmp_dir / "data" / "uploads"
                        upload_inventory = _tree_inventory(copied_uploads)
                        copied_corpus_hash = _tree_corpus_hash(copied_uploads)
                        if target_corpus_hash is None or not hmac.compare_digest(
                            copied_corpus_hash, target_corpus_hash.lower()
                        ):
                            warnings.append(
                                "uploads: copied corpus does not match CURRENT identity"
                            )
                except Exception as exc:
                    _append_backup_warning(
                        warnings,
                        prefix="uploads: capture failed",
                        exc=exc,
                        key_id="backup.uploads",
                    )
            else:
                warnings.append("uploads: required capture disabled")

            if not qdrant_snapshot:
                warnings.append("qdrant: required capture disabled")
            elif target_collections is None:
                warnings.append("qdrant: activation collections unavailable")
            else:
                try:
                    qdrant_files, qdrant_bytes, qdrant_version = (
                        _create_qdrant_snapshots(
                            cfg=cfg,
                            target_collections=target_collections,
                            dest_dir=tmp_dir,
                            warnings=warnings,
                            maintenance_warnings=maintenance_warnings,
                            artifact_dir=tmp_dir / "data" / "artifacts",
                            target_collection_metadata=target_collection_metadata,
                            deployment_id=deployment_id,
                        )
                    )
                    bytes_written += qdrant_bytes
                    required = set(_qdrant_target_collections(target_collections))
                    captured = {item.collection for item in qdrant_files}
                    if captured != required:
                        missing = ", ".join(sorted(required - captured)) or "unknown"
                        warnings.append(
                            "qdrant: exact activation capture incomplete; missing: "
                            f"{missing}"
                        )
                    else:
                        included.append("qdrant_snapshots")
                except Exception as exc:
                    _append_backup_warning(
                        warnings,
                        prefix="qdrant: snapshot failed",
                        exc=exc,
                        key_id="backup.qdrant",
                    )
    except Exception as exc:
        _append_backup_warning(
            warnings,
            prefix="snapshots: writer lease unavailable",
            exc=exc,
            key_id="backup.snapshot_lock",
        )

    # Chat DB (small but user-relevant)
    chat_db = Path(cfg.chat.sqlite_path)
    chat_dest = tmp_dir / "data" / chat_db.name
    bytes_written += _include_sqlite_database(
        source=chat_db,
        dest=chat_dest,
        label="chat_db",
        included=included,
        warnings=warnings,
        warn_missing=True,
    )
    if "chat_db" in included:
        database_inventory["chat_db"] = _file_inventory(tmp_dir, chat_dest)

    # Optional: analytics
    if include_analytics:
        analytics_path = (
            cfg.analytics_db_path
            if cfg.analytics_db_path is not None
            else (cfg.data_dir / "analytics" / "analytics.duckdb")
        )
        analytics_path = Path(analytics_path)
        bytes_written += _include_duckdb_database(
            source=analytics_path,
            dest=tmp_dir / "data" / "analytics" / analytics_path.name,
            label="analytics",
            included=included,
            warnings=warnings,
            warn_missing=True,
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
            warn_missing=True,
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
            warn_missing=True,
        )

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "app_version": str(cfg.app_version),
        "complete": not warnings,
        "included": included,
        "bytes_written": int(bytes_written),
        "qdrant": {
            "url": safe_url_for_log(str(cfg.database.qdrant_url)),
            "version": qdrant_version,
            "collections": [asdict(snapshot) for snapshot in qdrant_files],
        },
        "uploads": {"files": upload_inventory},
        "artifacts": {"files": artifact_inventory},
        "databases": database_inventory,
        "activation": {
            "snapshot": activation_snapshot,
            "collections": target_collections or {},
            "deployment_id": deployment_id,
        },
        "warnings": warnings,
        "maintenance_warnings": maintenance_warnings,
    }
    _write_manifest(tmp_dir / "manifest.json", manifest)

    return bytes_written, included, warnings, maintenance_warnings, qdrant_files


def _validate_backup_destination(
    root: Path,
    *,
    cfg: DocMindSettings,
    include_uploads: bool,
    include_logs: bool,
) -> None:
    """Reject a destination nested inside any recursively copied source tree."""
    artifact_dir = (
        Path(cfg.artifacts.dir)
        if cfg.artifacts.dir is not None
        else (cfg.data_dir / "artifacts")
    )
    source_trees = [cfg.data_dir / "storage", artifact_dir]
    if include_uploads:
        source_trees.append(cfg.data_dir / "uploads")
    if include_logs:
        source_trees.append(Path("./logs"))
    resolved_root = root.resolve()
    for source in source_trees:
        resolved_source = source.expanduser().resolve()
        if resolved_root == resolved_source or resolved_source in resolved_root.parents:
            raise ValueError(
                "Backup destination cannot be inside a recursively copied source: "
                f"{source}"
            )


def _remove_unpromoted_workspace(path: Path) -> None:
    """Best-effort removal for a failed backup workspace."""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as exc:
        redaction = build_pii_log_entry(str(exc), key_id="backup.workspace_cleanup")
        logger.error(
            "Failed to remove incomplete backup workspace "
            "(path={}, error_type={}, error={})",
            path,
            type(exc).__name__,
            redaction.redacted,
        )


def _verify_workspace_before_promotion(
    tmp_dir: Path,
    warnings: list[str],
) -> None:
    """Downgrade a nominally complete workspace that fails recovery validation."""
    if warnings:
        return
    manifest_path = tmp_dir / "manifest.json"
    payload: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Backup workspace manifest must be a JSON object")
    if _retention_eligible_backup(tmp_dir, payload):
        return
    warnings.append("backup: pre-promotion recoverability verification failed")
    payload["complete"] = False
    payload["warnings"] = warnings
    _write_manifest(manifest_path, payload)


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
        include_uploads: Include authoritative uploads. Disabling this creates an
            incomplete diagnostic backup that is excluded from retention.
        include_analytics: Include the analytics DuckDB database (if present).
        include_logs: Include `logs/` (best effort).
        include_env: Include `.env` (contains secrets).
        keep_last: Backups to retain; defaults to settings.backup_keep_last.
        qdrant_snapshot: Capture exact activation collections. Disabling this
            creates an incomplete diagnostic backup.
        cfg: Optional settings override.

    Returns:
        BackupResult with paths, bytes written, and warnings.

    Raises:
        ValueError: If backups are disabled, keep_last < 1, backup directory
            already exists, or if a symlink blocks the backup root path.
    """
    cfg = cfg or settings
    if not cfg.backup_enabled:
        raise ValueError("Backups are disabled (set DOCMIND_BACKUP_ENABLED=true)")

    start = time.perf_counter()

    root = (dest_root or (cfg.data_dir / "backups")).expanduser()
    _validate_backup_destination(
        root,
        cfg=cfg,
        include_uploads=include_uploads,
        include_logs=include_logs,
    )
    _safe_mkdir(root)

    keep_last_value = int(keep_last if keep_last is not None else cfg.backup_keep_last)
    if keep_last_value < 1:
        raise ValueError("keep_last must be >= 1")

    with _backup_root_lock(root):
        run_id = f"{_utc_timestamp_compact()}_{uuid.uuid4().hex[:12]}"
        tmp_dir = root / f"tmp-backup_{run_id}"
        _safe_mkdir(tmp_dir, exist_ok=False)
        try:
            (
                bytes_written,
                included,
                warnings,
                maintenance_warnings,
                qdrant_files,
            ) = _populate_backup_workspace(
                cfg=cfg,
                tmp_dir=tmp_dir,
                include_uploads=include_uploads,
                include_analytics=include_analytics,
                include_logs=include_logs,
                include_env=include_env,
                qdrant_snapshot=qdrant_snapshot,
            )
            _verify_workspace_before_promotion(tmp_dir, warnings)
            final_prefix = "backup" if not warnings else "incomplete-backup"
            final_dir = root / f"{final_prefix}_{run_id}"
            if final_dir.exists():
                raise ValueError(f"Backup directory already exists: {final_dir}")
            _fsync_tree(tmp_dir)
            tmp_dir.rename(final_dir)
            _fsync_directory(root)
        except BaseException:
            _remove_unpromoted_workspace(tmp_dir)
            raise

        # Never let a degraded backup evict the last known-good recovery point.
        # Incomplete runs use a distinct name and are ignored by future pruning.
        deleted = (
            _prune_backups_unlocked(root, keep_last=keep_last_value)
            if not warnings
            else []
        )
    duration_ms = (time.perf_counter() - start) * 1000.0

    _safe_log_jsonl(
        {
            "event": "backup_created",
            "path": str(final_dir),
            "included": included,
            "bytes_written": int(bytes_written),
            "duration_ms": round(duration_ms, 2),
            "pruned": len(deleted),
            "complete": not warnings,
            "retention_skipped": bool(warnings),
            "maintenance_warning_count": len(maintenance_warnings),
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
        maintenance_warnings=maintenance_warnings,
    )


__all__ = [
    "BackupResult",
    "QdrantSnapshotFile",
    "create_backup",
    "prune_backups",
]
