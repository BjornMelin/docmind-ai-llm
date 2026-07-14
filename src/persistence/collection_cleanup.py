"""Offline cleanup for unreferenced DocMind Qdrant collections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.config.settings import DocMindSettings, settings
from src.persistence.deployment_identity import read_deployment_id
from src.persistence.lockfile import SnapshotLock
from src.persistence.snapshot import is_snapshot_version_name, load_manifest

_OWNERS = {"text", "image"}


class CollectionCleanupError(RuntimeError):
    """Raised when safe orphan discovery cannot be completed."""


@dataclass(frozen=True, slots=True)
class CollectionCleanupSummary:
    """Structured result from an offline collection cleanup pass."""

    mode: Literal["dry-run", "delete"]
    inspected_collections: tuple[str, ...]
    retained_collections: tuple[str, ...]
    orphan_candidates: tuple[str, ...]
    deleted_collections: tuple[str, ...]
    preserved_collections: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable cleanup summary.

        Returns:
            dict[str, object]: Cleanup evidence suitable for CLI output.
        """
        return {
            "status": "ok",
            "mode": self.mode,
            "inspected_collections": list(self.inspected_collections),
            "retained_collections": list(self.retained_collections),
            "orphan_candidates": list(self.orphan_candidates),
            "deleted_collections": list(self.deleted_collections),
            "preserved_collections": list(self.preserved_collections),
        }


def cleanup_orphan_collections(
    client: Any,
    *,
    delete: bool = False,
    cfg: DocMindSettings = settings,
) -> CollectionCleanupSummary:
    """Discover or delete deployment-owned collections unused by any snapshot.

    This function is intentionally operator-invoked only. The caller must quiesce
    every DocMind reader and writer before entering this cleanup boundary.

    Args:
        client: Synchronous Qdrant client.
        delete: Delete eligible collections instead of running a dry run.
        cfg: DocMind settings defining data, lock, and Qdrant ownership scope.

    Returns:
        CollectionCleanupSummary: Collections inspected, retained, and deleted.

    Raises:
        CollectionCleanupError: If retained snapshot state cannot be trusted.
        DeploymentIdentityError: If the local deployment identity is invalid.
        SnapshotLockError: If the canonical writer lock cannot be acquired.
    """
    storage_dir = cfg.data_dir / "storage"
    lock_config = cfg.snapshots
    lock = SnapshotLock(
        storage_dir / ".lock",
        timeout=float(lock_config.lock_timeout_seconds),
        ttl_seconds=float(lock_config.lock_ttl_seconds),
    )
    with lock:
        deployment_id = read_deployment_id(cfg.data_dir)
        retained = _load_retained_collections(storage_dir)
        inspected = _list_collection_names(client)
        candidates = _find_owned_orphans(
            client,
            inspected=inspected,
            retained=retained,
            deployment_id=deployment_id,
        )
        deleted: tuple[str, ...] = ()
        if delete:
            for collection_name in candidates:
                if (
                    client.delete_collection(collection_name=collection_name)
                    is not True
                ):
                    raise CollectionCleanupError(
                        f"Qdrant did not confirm deletion of {collection_name!r}"
                    )
            deleted = candidates

    return CollectionCleanupSummary(
        mode="delete" if delete else "dry-run",
        inspected_collections=inspected,
        retained_collections=retained,
        orphan_candidates=candidates,
        deleted_collections=deleted,
        preserved_collections=tuple(sorted(set(inspected) - set(deleted))),
    )


def _load_retained_collections(storage_dir: Path) -> tuple[str, ...]:
    """Load collection names from every verified canonical snapshot."""
    if not storage_dir.exists():
        return ()
    if storage_dir.is_symlink() or not storage_dir.is_dir():
        raise CollectionCleanupError("Snapshot storage is not a trusted directory")
    try:
        candidates = sorted(storage_dir.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise CollectionCleanupError("Snapshot storage cannot be read safely") from exc

    retained: set[str] = set()
    verified_directories: set[str] = set()
    for snapshot_dir in candidates:
        if not is_snapshot_version_name(snapshot_dir.name):
            continue
        if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
            raise CollectionCleanupError(
                f"Canonical snapshot {snapshot_dir.name!r} is not a trusted directory"
            )
        manifest_files = (
            snapshot_dir / "manifest.jsonl",
            snapshot_dir / "manifest.meta.json",
            snapshot_dir / "manifest.checksum",
        )
        if any(path.is_symlink() or not path.is_file() for path in manifest_files):
            raise CollectionCleanupError(
                f"Canonical snapshot {snapshot_dir.name!r} has an untrusted manifest"
            )
        manifest = load_manifest(snapshot_dir)
        if manifest is None:
            raise CollectionCleanupError(
                f"Canonical snapshot {snapshot_dir.name!r} is not complete and verified"
            )
        collections = manifest["collections"]
        retained.update((collections["text"], collections["image"]))
        verified_directories.add(snapshot_dir.name)

    _validate_current(storage_dir, verified_directories)
    return tuple(sorted(retained))


def _validate_current(storage_dir: Path, verified_directories: set[str]) -> None:
    current = storage_dir / "CURRENT"
    if not current.exists() and not current.is_symlink():
        return
    if current.is_symlink() or not current.is_file():
        raise CollectionCleanupError("CURRENT is not a trusted regular file")
    try:
        snapshot_name = current.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise CollectionCleanupError("CURRENT cannot be read safely") from exc
    if snapshot_name not in verified_directories:
        raise CollectionCleanupError("CURRENT does not reference a verified snapshot")


def _list_collection_names(client: Any) -> tuple[str, ...]:
    response = client.get_collections()
    names: set[str] = set()
    for collection in response.collections:
        name = getattr(collection, "name", None)
        if not isinstance(name, str) or not name:
            raise CollectionCleanupError("Qdrant returned an invalid collection name")
        names.add(name)
    return tuple(sorted(names))


def _find_owned_orphans(
    client: Any,
    *,
    inspected: tuple[str, ...],
    retained: tuple[str, ...],
    deployment_id: str,
) -> tuple[str, ...]:
    retained_names = set(retained)
    candidates: list[str] = []
    for collection_name in inspected:
        if collection_name in retained_names:
            continue
        info = client.get_collection(collection_name=collection_name)
        metadata = getattr(getattr(info, "config", None), "metadata", None)
        if not isinstance(metadata, dict):
            continue
        if (
            metadata.get("docmind_deployment_id") == deployment_id
            and metadata.get("docmind_owner") in _OWNERS
        ):
            candidates.append(collection_name)
    return tuple(candidates)


__all__ = [
    "CollectionCleanupError",
    "CollectionCleanupSummary",
    "cleanup_orphan_collections",
]
