"""Tests for explicit offline Qdrant collection cleanup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from src.persistence import snapshot_writer
from src.persistence.collection_cleanup import (
    CollectionCleanupError,
    cleanup_orphan_collections,
)
from src.persistence.deployment_identity import (
    DEPLOYMENT_ID_FILENAME,
    DeploymentIdentityError,
    get_or_create_deployment_id,
)
from src.persistence.hashing import compute_config_hash
from src.persistence.lockfile import SnapshotLock, SnapshotLockTimeoutError
from tests.fixtures.test_settings import create_test_settings


class _QdrantClient:
    def __init__(
        self,
        metadata: dict[str, dict[str, object]],
        *,
        delete_result: bool = True,
    ) -> None:
        self.metadata = metadata
        self.delete_result = delete_result
        self.deleted: list[str] = []
        self.list_calls = 0

    def get_collections(self) -> SimpleNamespace:
        self.list_calls += 1
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.metadata]
        )

    def get_collection(self, *, collection_name: str) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(metadata=self.metadata[collection_name])
        )

    def delete_collection(self, *, collection_name: str) -> bool:
        self.deleted.append(collection_name)
        return self.delete_result


def _settings(tmp_path: Path):  # type: ignore[no-untyped-def]
    return create_test_settings(
        data_dir=tmp_path,
        cache={"dir": tmp_path / "cache"},
        log_file=tmp_path / "logs" / "test.log",
        snapshots={
            "lock_timeout_seconds": 0.5,
            "lock_ttl_seconds": 5.0,
            "retention_count": 5,
            "gc_grace_seconds": 0,
        },
    )


def _write_snapshot(
    storage_dir: Path,
    name: str,
    *,
    text: str,
    image: str,
) -> Path:
    directory = storage_dir / name
    directory.mkdir(parents=True)
    activation_config: dict[str, object] = {}
    snapshot_writer.write_manifest(
        directory,
        manifest_meta={
            "index_id": name,
            "graph_store_type": "none",
            "vector_store_type": "qdrant",
            "collections": {"text": text, "image": image},
            "corpus_hash": "0" * 64,
            "config_hash": "1" * 64,
            "versions": {},
            "graph_exports": [],
            "collection_metadata": {},
            "activation_config": activation_config,
            "activation_config_hash": compute_config_hash(activation_config),
        },
    )
    snapshot_writer.mark_manifest_complete(directory)
    return directory


def _owned(deployment_id: str, owner: str) -> dict[str, object]:
    return {
        "docmind_deployment_id": deployment_id,
        "docmind_owner": owner,
    }


def test_dry_run_preserves_every_collection(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    deployment_id = get_or_create_deployment_id(cfg.data_dir)
    retained = _write_snapshot(
        cfg.data_dir / "storage",
        "20260714T010203-a1b2c3d4",
        text="retained-text",
        image="retained-image",
    )
    (cfg.data_dir / "storage" / "CURRENT").write_text(retained.name, encoding="utf-8")
    client = _QdrantClient(
        {
            "retained-text": _owned(deployment_id, "text"),
            "retained-image": _owned(deployment_id, "image"),
            "old-prefix-before-rename-aabbccdd": _owned(deployment_id, "text"),
        }
    )

    summary = cleanup_orphan_collections(client, cfg=cfg)

    assert summary.mode == "dry-run"
    assert summary.orphan_candidates == ("old-prefix-before-rename-aabbccdd",)
    assert summary.deleted_collections == ()
    assert client.deleted == []
    assert summary.preserved_collections == summary.inspected_collections


def test_delete_only_local_owned_unreferenced_collections(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    deployment_id = get_or_create_deployment_id(cfg.data_dir)
    other_deployment = str(UUID(int=2))
    _write_snapshot(
        cfg.data_dir / "storage",
        "20260714T010203-a1b2c3d4",
        text="renamed-retained-text",
        image="renamed-retained-image",
    )
    _write_snapshot(
        cfg.data_dir / "storage",
        "20260713T010203-01020304",
        text="older-retained-text",
        image="older-retained-image",
    )
    client = _QdrantClient(
        {
            "renamed-retained-text": _owned(deployment_id, "text"),
            "renamed-retained-image": _owned(deployment_id, "image"),
            "older-retained-text": _owned(deployment_id, "text"),
            "older-retained-image": _owned(deployment_id, "image"),
            "legacy-base-11111111": _owned(deployment_id, "text"),
            "other-install-22222222": _owned(other_deployment, "text"),
            "foreign-owner": _owned(deployment_id, "cache"),
            "no-metadata": {},
        }
    )

    summary = cleanup_orphan_collections(client, cfg=cfg, delete=True)

    assert summary.orphan_candidates == ("legacy-base-11111111",)
    assert summary.deleted_collections == ("legacy-base-11111111",)
    assert client.deleted == ["legacy-base-11111111"]
    assert "other-install-22222222" in summary.preserved_collections
    assert "renamed-retained-text" in summary.preserved_collections
    assert "older-retained-text" in summary.preserved_collections


def test_bad_canonical_manifest_fails_before_qdrant_scan(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    get_or_create_deployment_id(cfg.data_dir)
    invalid = cfg.data_dir / "storage" / "20260714T010203-a1b2c3d4"
    invalid.mkdir(parents=True)
    (invalid / "manifest.jsonl").write_text("", encoding="utf-8")
    (invalid / "manifest.meta.json").write_text("{bad", encoding="utf-8")
    (invalid / "manifest.checksum").write_text("{}", encoding="utf-8")
    client = _QdrantClient({"orphan": {}})

    with pytest.raises(CollectionCleanupError, match="not complete and verified"):
        cleanup_orphan_collections(client, cfg=cfg, delete=True)

    assert client.list_calls == 0
    assert client.deleted == []


@pytest.mark.parametrize("identity", [None, "not-a-uuid\n"])
def test_missing_or_invalid_deployment_identity_fails_closed(
    tmp_path: Path,
    identity: str | None,
) -> None:
    cfg = _settings(tmp_path)
    if identity is not None:
        (cfg.data_dir / DEPLOYMENT_ID_FILENAME).write_text(identity, encoding="ascii")
    client = _QdrantClient({"orphan": {}})

    with pytest.raises(DeploymentIdentityError):
        cleanup_orphan_collections(client, cfg=cfg, delete=True)

    assert client.list_calls == 0
    assert client.deleted == []


def test_lock_contention_fails_before_qdrant_scan(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    get_or_create_deployment_id(cfg.data_dir)
    lock_path = cfg.data_dir / "storage" / ".lock"
    owner = SnapshotLock(lock_path, timeout=0.5, ttl_seconds=5.0)
    owner.acquire()
    client = _QdrantClient({"orphan": {}})
    try:
        with pytest.raises(SnapshotLockTimeoutError):
            cleanup_orphan_collections(client, cfg=cfg, delete=True)
    finally:
        owner.release()

    assert client.list_calls == 0
    assert client.deleted == []


def test_delete_requires_qdrant_confirmation(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    deployment_id = get_or_create_deployment_id(cfg.data_dir)
    client = _QdrantClient(
        {"orphan": _owned(deployment_id, "text")},
        delete_result=False,
    )

    with pytest.raises(CollectionCleanupError, match="did not confirm deletion"):
        cleanup_orphan_collections(client, cfg=cfg, delete=True)


def test_current_must_reference_a_verified_snapshot(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    get_or_create_deployment_id(cfg.data_dir)
    storage = cfg.data_dir / "storage"
    storage.mkdir()
    (storage / "CURRENT").write_text("missing-snapshot", encoding="utf-8")

    with pytest.raises(CollectionCleanupError, match="CURRENT"):
        cleanup_orphan_collections(_QdrantClient({}), cfg=cfg)
