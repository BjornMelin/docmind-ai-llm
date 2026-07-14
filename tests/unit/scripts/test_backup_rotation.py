"""Unit tests for backup rotation and prune_backups behavior."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from src.persistence import snapshot_writer
from src.persistence.backup_service import prune_backups
from src.persistence.hashing import compute_config_hash, compute_corpus_hash
from src.persistence.snapshot import load_manifest

pytestmark = pytest.mark.unit

_PHYSICAL_COLLECTIONS = {
    "text": "physical-text-v2",
    "image": "physical-image-v2",
}


def _complete_backup(path: Path) -> None:
    path.mkdir()
    storage = path / "data" / "storage"
    active = storage / "20260101T000000-deadbeef"
    active.mkdir(parents=True)
    deployment_id = "00000000-0000-4000-8000-000000000001"
    (path / "data" / ".deployment-id").write_text(
        f"{deployment_id}\n", encoding="ascii"
    )
    uploads = path / "data" / "uploads"
    uploads.mkdir()
    upload_file = uploads / "document.txt"
    upload_file.write_text("corpus", encoding="utf-8")
    artifacts = path / "data" / "artifacts"
    artifacts.mkdir()
    artifact_file = artifacts / "page.webp"
    artifact_file.write_bytes(b"image artifact")
    cache_db = path / "cache" / "docmind.duckdb"
    cache_db.parent.mkdir()
    cache_db.write_bytes(b"cache database")
    chat_db = path / "data" / "chat.db"
    chat_db.write_bytes(b"chat database")
    snapshot_writer.write_manifest(
        active,
        manifest_meta={
            "index_id": "backup-test-index",
            "graph_store_type": "none",
            "vector_store_type": "qdrant",
            "collections": dict(_PHYSICAL_COLLECTIONS),
            "collection_metadata": {
                owner: {
                    "docmind_deployment_id": deployment_id,
                    "docmind_owner": owner,
                }
                for owner in ("text", "image")
            },
            "corpus_hash": compute_corpus_hash([upload_file], base_dir=uploads),
            "config_hash": "f" * 64,
            "versions": {},
            "graph_exports": [],
            "activation_config": {},
            "activation_config_hash": compute_config_hash({}),
        },
    )
    snapshot_writer.mark_manifest_complete(active)
    (storage / "CURRENT").write_text(f"{active.name}\n", encoding="utf-8")
    qdrant_snapshots: list[dict[str, object]] = []
    for collection in sorted(_PHYSICAL_COLLECTIONS.values()):
        body = f"snapshot:{collection}".encode()
        snapshot_file = path / "qdrant" / collection / "snapshot.bin"
        snapshot_file.parent.mkdir(parents=True)
        snapshot_file.write_bytes(body)
        qdrant_snapshots.append(
            {
                "collection": collection,
                "snapshot_name": snapshot_file.name,
                "filename": str(snapshot_file.relative_to(path)),
                "size_bytes": len(body),
                "checksum": sha256(body).hexdigest(),
                "point_count": 1,
            }
        )
    manifest = load_manifest(active)
    assert manifest is not None
    assert manifest["schema_version"] == "2.0"
    assert manifest["collections"] == _PHYSICAL_COLLECTIONS
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "app_version": "1.0.0",
                "complete": True,
                "included": [
                    "cache_db",
                    "snapshots",
                    "chat_db",
                    "uploads",
                    "artifacts",
                    "deployment_identity",
                    "qdrant_snapshots",
                ],
                "activation": {
                    "snapshot": active.name,
                    "collections": dict(_PHYSICAL_COLLECTIONS),
                    "deployment_id": deployment_id,
                },
                "uploads": {
                    "files": [
                        {
                            "path": upload_file.relative_to(uploads).as_posix(),
                            "size_bytes": upload_file.stat().st_size,
                            "sha256": sha256(upload_file.read_bytes()).hexdigest(),
                        }
                    ]
                },
                "artifacts": {
                    "files": [
                        {
                            "path": artifact_file.relative_to(artifacts).as_posix(),
                            "size_bytes": artifact_file.stat().st_size,
                            "sha256": sha256(artifact_file.read_bytes()).hexdigest(),
                        }
                    ]
                },
                "databases": {
                    "cache_db": {
                        "path": cache_db.relative_to(path).as_posix(),
                        "size_bytes": cache_db.stat().st_size,
                        "sha256": sha256(cache_db.read_bytes()).hexdigest(),
                    },
                    "chat_db": {
                        "path": chat_db.relative_to(path).as_posix(),
                        "size_bytes": chat_db.stat().st_size,
                        "sha256": sha256(chat_db.read_bytes()).hexdigest(),
                    },
                },
                "qdrant": {
                    "version": "1.18.2",
                    "collections": qdrant_snapshots,
                },
                "warnings": [],
                "maintenance_warnings": [],
            }
        ),
        encoding="utf-8",
    )


def test_prune_backups_keeps_most_recent(tmp_path: Path) -> None:
    """Remove older backups while keeping the newest entries.

    Args:
        tmp_path: Temporary directory for test artifacts.

    Returns:
        None.
    """
    root = tmp_path / "backups"
    root.mkdir()

    # Names sort lexicographically; these simulate timestamps.
    old = root / "backup_20260101_000000"
    mid = root / "backup_20260101_000001"
    new = root / "backup_20260101_000002"
    _complete_backup(old)
    _complete_backup(mid)
    _complete_backup(new)
    (root / "not_a_backup").mkdir()

    deleted = prune_backups(root, keep_last=2)

    assert old in deleted
    assert not old.exists()
    assert mid.exists()
    assert new.exists()
    assert (root / "not_a_backup").exists()


def test_prune_backups_rejects_invalid_keep_last(tmp_path: Path) -> None:
    """Reject keep_last values below the minimum of 1.

    Args:
        tmp_path: Temporary directory for test artifacts.

    Raises:
        ValueError: if keep_last is less than 1.

    Returns:
        None.
    """
    root = tmp_path / "backups"
    root.mkdir()

    with pytest.raises(ValueError, match="keep_last must be >= 1"):
        prune_backups(root, keep_last=0)


def test_prune_ignores_incomplete_and_unverified_directories(tmp_path: Path) -> None:
    """Only finalized backups with a complete manifest enter retention."""
    root = tmp_path / "backups"
    root.mkdir()
    old = root / "backup_20260101_000000"
    new = root / "backup_20260101_000001"
    _complete_backup(old)
    _complete_backup(new)
    incomplete = root / "incomplete-backup_20260101_000002_deadbeefcafe"
    incomplete.mkdir()
    (incomplete / "manifest.json").write_text(
        json.dumps({"complete": False}),
        encoding="utf-8",
    )
    unverified = root / "backup_20260101_000003"
    unverified.mkdir()

    deleted = prune_backups(root, keep_last=1)

    assert deleted == [old]
    assert new.is_dir()
    assert incomplete.is_dir()
    assert unverified.is_dir()


def test_prune_rejects_symlink_root(tmp_path: Path) -> None:
    """A prune root cannot redirect deletion through a symlink."""
    target = tmp_path / "target"
    target.mkdir()
    backup = target / "backup_20260101_000000"
    _complete_backup(backup)
    root = tmp_path / "linked-backups"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="Symlink destination blocked"):
        prune_backups(root, keep_last=1)

    assert backup.is_dir()


def test_prune_preserves_good_backup_when_newer_qdrant_file_is_corrupt(
    tmp_path: Path,
) -> None:
    """Retention never trusts a manifest after its recovery data drifts."""
    root = tmp_path / "backups"
    root.mkdir()
    good = root / "backup_20260101_000000"
    corrupt = root / "backup_20260101_000001"
    _complete_backup(good)
    _complete_backup(corrupt)
    (corrupt / "qdrant" / _PHYSICAL_COLLECTIONS["text"] / "snapshot.bin").write_bytes(
        b"corrupt"
    )

    deleted = prune_backups(root, keep_last=1)

    assert deleted == []
    assert good.is_dir()
    assert corrupt.is_dir()


@pytest.mark.parametrize("point_count", [None, True, -1])
def test_prune_preserves_good_backup_when_qdrant_point_count_is_invalid(
    point_count: object,
    tmp_path: Path,
) -> None:
    """Retention requires an exact non-negative Qdrant restore count."""
    root = tmp_path / "backups"
    root.mkdir()
    good = root / "backup_20260101_000000"
    corrupt = root / "backup_20260101_000001"
    _complete_backup(good)
    _complete_backup(corrupt)
    manifest_path = corrupt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["qdrant"]["collections"][0]["point_count"] = point_count
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    deleted = prune_backups(root, keep_last=1)

    assert deleted == []
    assert good.is_dir()
    assert corrupt.is_dir()


@pytest.mark.parametrize(
    ("field", "value"),
    [("app_version", ""), ("qdrant.version", None)],
)
def test_prune_preserves_good_backup_when_restore_version_is_invalid(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    """Retention requires exact application and Qdrant restore versions."""
    root = tmp_path / "backups"
    root.mkdir()
    good = root / "backup_20260101_000000"
    corrupt = root / "backup_20260101_000001"
    _complete_backup(good)
    _complete_backup(corrupt)
    manifest_path = corrupt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "app_version":
        manifest[field] = value
    else:
        manifest["qdrant"]["version"] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    deleted = prune_backups(root, keep_last=1)

    assert deleted == []
    assert good.is_dir()
    assert corrupt.is_dir()


@pytest.mark.parametrize(
    ("tree", "filename"),
    [("uploads", "document.txt"), ("artifacts", "page.webp")],
)
@pytest.mark.parametrize("mutation", ["delete", "truncate"])
def test_prune_preserves_good_backup_when_newer_tree_inventory_drifts(
    tree: str,
    filename: str,
    mutation: str,
    tmp_path: Path,
) -> None:
    """Retention rejects missing or content-drifted authoritative files."""
    root = tmp_path / "backups"
    root.mkdir()
    good = root / "backup_20260101_000000"
    corrupt = root / "backup_20260101_000001"
    _complete_backup(good)
    _complete_backup(corrupt)
    target = corrupt / "data" / tree / filename
    if mutation == "delete":
        target.unlink()
    else:
        target.write_bytes(b"")

    deleted = prune_backups(root, keep_last=1)

    assert deleted == []
    assert good.is_dir()
    assert corrupt.is_dir()


@pytest.mark.parametrize(
    "relative_path",
    [Path("cache/docmind.duckdb"), Path("data/chat.db")],
)
@pytest.mark.parametrize("mutation", ["delete", "same_size_corruption"])
def test_prune_preserves_good_backup_when_required_database_drifts(
    relative_path: Path,
    mutation: str,
    tmp_path: Path,
) -> None:
    """Retention rejects a missing or checksum-drifted required database."""
    root = tmp_path / "backups"
    root.mkdir()
    good = root / "backup_20260101_000000"
    corrupt = root / "backup_20260101_000001"
    _complete_backup(good)
    _complete_backup(corrupt)
    target = corrupt / relative_path
    if mutation == "delete":
        target.unlink()
    else:
        target.write_bytes(b"x" * target.stat().st_size)

    deleted = prune_backups(root, keep_last=1)

    assert deleted == []
    assert good.is_dir()
    assert corrupt.is_dir()


def test_prune_rejects_self_consistent_uploads_that_disagree_with_current(
    tmp_path: Path,
) -> None:
    """Rewriting the backup inventory cannot conceal activation corpus drift."""
    root = tmp_path / "backups"
    root.mkdir()
    good = root / "backup_20260101_000000"
    corrupt = root / "backup_20260101_000001"
    _complete_backup(good)
    _complete_backup(corrupt)
    upload = corrupt / "data" / "uploads" / "document.txt"
    upload.write_text("different but inventoried", encoding="utf-8")
    manifest_path = corrupt / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["uploads"]["files"][0].update(
        {
            "size_bytes": upload.stat().st_size,
            "sha256": sha256(upload.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    deleted = prune_backups(root, keep_last=1)

    assert deleted == []
    assert good.is_dir()
    assert corrupt.is_dir()
