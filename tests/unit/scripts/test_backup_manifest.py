"""Tests for backup manifest creation."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import threading
from contextlib import closing, nullcontext
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
from qdrant_client.http import models as qmodels

from src.persistence import backup_service as mod
from src.persistence.backup_service import create_backup
from src.persistence.deployment_identity import (
    DEPLOYMENT_ID_FILENAME,
    get_or_create_deployment_id,
)
from src.persistence.hashing import compute_corpus_hash
from src.persistence.lockfile import SnapshotLock, SnapshotLockTimeoutError
from src.persistence.snapshot import SnapshotManager
from src.persistence.snapshot_utils import collect_corpus_paths
from src.persistence.upload_journal import quarantine_upload
from tests.fixtures.test_settings import create_test_settings

_PHYSICAL_COLLECTIONS = {
    "text": "physical-text-v2",
    "image": "physical-image-v2",
}


def _seed_active_snapshot(cfg) -> Path:  # type: ignore[no-untyped-def]
    """Create one verified manifest-v2 snapshot and commit it through CURRENT."""
    deployment_id = get_or_create_deployment_id(cfg.data_dir)
    uploads = cfg.data_dir / "uploads"
    corpus_hash = compute_corpus_hash(
        collect_corpus_paths(uploads),
        base_dir=uploads,
    )
    manager = SnapshotManager(cfg.data_dir / "storage")
    workspace = manager.begin_snapshot()
    try:
        manager.write_manifest(
            workspace,
            index_id="backup-test-index",
            graph_store_type="none",
            vector_store_type="qdrant",
            text_collection=_PHYSICAL_COLLECTIONS["text"],
            image_collection=_PHYSICAL_COLLECTIONS["image"],
            corpus_hash=corpus_hash,
            config_hash="f" * 64,
            versions={"app": "test"},
            collection_metadata={
                "text": {
                    "docmind_deployment_id": deployment_id,
                    "docmind_owner": "text",
                },
                "image": {
                    "docmind_deployment_id": deployment_id,
                    "docmind_owner": "image",
                },
            },
        )
        return manager.finalize_snapshot(workspace).path
    except BaseException:
        manager.cleanup_tmp(workspace)
        raise


def _backup_chat_only(cfg, tmp_path: Path):  # type: ignore[no-untyped-def]
    return create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=False,
        include_analytics=False,
        include_logs=False,
        keep_last=3,
        qdrant_snapshot=False,
        cfg=cfg,
    )


def _seed_required_backup_sources(cfg) -> Path:  # type: ignore[no-untyped-def]
    uploads = cfg.data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "document.txt").write_text("authoritative corpus", encoding="utf-8")
    active_snapshot = _seed_active_snapshot(cfg)
    cache_db = cfg.cache.ingestion_db_path
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(cache_db)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS cache_probe (value INTEGER)")
    chat_db = Path(cfg.chat.sqlite_path)
    chat_db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(chat_db)) as conn, conn:
        conn.execute("CREATE TABLE IF NOT EXISTS chat_probe (value INTEGER)")
    return active_snapshot


def _write_verified_qdrant_capture(
    *, dest_dir: Path, target_collections: dict[str, str]
) -> tuple[list[mod.QdrantSnapshotFile], int, str]:
    """Create locally verified stand-ins for both activation collections."""
    snapshots: list[mod.QdrantSnapshotFile] = []
    total = 0
    for collection in sorted(set(target_collections.values())):
        body = f"snapshot:{collection}".encode()
        snapshot_path = dest_dir / "qdrant" / collection / "snapshot.bin"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(body)
        total += len(body)
        snapshots.append(
            mod.QdrantSnapshotFile(
                collection=collection,
                snapshot_name="snapshot.bin",
                filename=str(snapshot_path.relative_to(dest_dir)),
                size_bytes=len(body),
                checksum=hashlib.sha256(body).hexdigest(),
                point_count=1,
            )
        )
    return snapshots, total, "1.18.2"


def _install_verified_qdrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _capture(**kwargs: object):  # type: ignore[no-untyped-def]
        destination = kwargs["dest_dir"]
        collections = kwargs["target_collections"]
        assert isinstance(destination, Path)
        assert isinstance(collections, dict)
        return _write_verified_qdrant_capture(
            dest_dir=destination,
            target_collections=collections,
        )

    monkeypatch.setattr(mod, "_create_qdrant_snapshots", _capture)


class _ArtifactScrollClient:
    def __init__(
        self,
        pages: list[tuple[list[SimpleNamespace], object | None]],
        *,
        error: Exception | None = None,
    ) -> None:
        self._pages = list(pages)
        self._error = error
        self.calls: list[dict[str, object]] = []

    def scroll(self, **kwargs: object) -> tuple[list[SimpleNamespace], object | None]:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._pages.pop(0)


def _artifact_payload(
    artifact_root: Path,
    body: bytes,
    *,
    suffix: str = ".webp",
) -> dict[str, str]:
    digest = hashlib.sha256(body).hexdigest()
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / f"{digest}{suffix}").write_bytes(body)
    return {
        "image_artifact_id": digest,
        "image_artifact_suffix": suffix,
    }


@pytest.mark.unit
def test_create_backup_uses_live_ingestion_cache_not_stale_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Creates a backup and writes a manifest.

    Args:
        tmp_path: Temporary directory for test artifacts.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.chdir(tmp_path)

    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
        backup_keep_last=3,
    )

    # Required artifacts
    active_snapshot = _seed_active_snapshot(cfg)
    deployment_id = get_or_create_deployment_id(cfg.data_dir)
    uploads = cfg.data_dir / "uploads"
    uploads.mkdir(parents=True)
    cache_db = cfg.cache.ingestion_db_path
    assert cache_db == cfg.cache.dir / "ingestion" / cfg.cache.filename
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(cache_db)) as conn:
        conn.execute("CREATE TABLE live_cache (value VARCHAR NOT NULL)")
        conn.execute("INSERT INTO live_cache VALUES ('live ingestion cache')")
    stale_parent_cache = cfg.cache.dir / cfg.cache.filename
    stale_parent_cache.write_text("stale parent cache", encoding="utf-8")
    chat_db = Path(cfg.chat.sqlite_path)
    chat_db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(chat_db)) as conn, conn:
        conn.execute("CREATE TABLE chat_probe (value INTEGER NOT NULL)")
    _install_verified_qdrant(monkeypatch)

    result = create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=True,
        include_analytics=False,
        include_logs=False,
        keep_last=3,
        qdrant_snapshot=True,
        cfg=cfg,
    )

    manifest_path = result.backup_dir / "manifest.json"
    assert manifest_path.is_file()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["bytes_written"] == result.bytes_written
    assert payload["complete"] is True
    assert payload["included"] == result.included
    assert {item["collection"] for item in payload["qdrant"]["collections"]} == set(
        _PHYSICAL_COLLECTIONS.values()
    )
    assert payload["qdrant"]["version"] == "1.18.2"
    assert payload["activation"] == {
        "snapshot": active_snapshot.name,
        "collections": _PHYSICAL_COLLECTIONS,
        "deployment_id": deployment_id,
    }
    assert payload["maintenance_warnings"] == []
    assert payload["uploads"] == {"files": []}
    assert payload["artifacts"] == {"files": []}
    assert result.maintenance_warnings == []
    assert (result.backup_dir / "data" / DEPLOYMENT_ID_FILENAME).read_text(
        encoding="ascii"
    ).strip() == deployment_id
    backed_up_storage = result.backup_dir / "data" / "storage"
    assert (backed_up_storage / "CURRENT").read_text(encoding="utf-8").strip() == (
        active_snapshot.name
    )
    copied_meta = json.loads(
        (backed_up_storage / active_snapshot.name / "manifest.meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert copied_meta["schema_version"] == "2.0"
    assert copied_meta["complete"] is True
    assert copied_meta["collections"] == _PHYSICAL_COLLECTIONS

    assert "cache_db" in result.included
    assert "snapshots" in result.included
    backed_up_cache = result.backup_dir / "cache" / cfg.cache.filename
    with duckdb.connect(str(backed_up_cache), read_only=True) as conn:
        assert conn.execute("SELECT value FROM live_cache").fetchone() == (
            "live ingestion cache",
        )
    assert stale_parent_cache.read_text(encoding="utf-8") == "stale parent cache"


@pytest.mark.unit
def test_duckdb_backups_include_committed_wal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    analytics_db = tmp_path / "custom'quoted" / "analytics.duckdb"
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache'quoted"},
        analytics_db_path=analytics_db,
        backup_enabled=True,
    )
    cache_db = cfg.cache.ingestion_db_path
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    analytics_db.parent.mkdir(parents=True, exist_ok=True)

    cache_writer = duckdb.connect(str(cache_db))
    analytics_writer = duckdb.connect(str(analytics_db))
    try:
        for writer in (cache_writer, analytics_writer):
            writer.execute("CREATE TABLE wal_probe (value INTEGER NOT NULL)")
            writer.execute("CHECKPOINT")
            writer.execute("INSERT INTO wal_probe VALUES (7)")
        assert cache_db.with_name(f"{cache_db.name}.wal").exists()
        assert analytics_db.with_name(f"{analytics_db.name}.wal").exists()

        result = create_backup(
            dest_root=tmp_path / "backups",
            include_uploads=False,
            include_analytics=True,
            include_logs=False,
            keep_last=3,
            qdrant_snapshot=False,
            cfg=cfg,
        )

        backed_up_cache = result.backup_dir / "cache" / cache_db.name
        backed_up_analytics = (
            result.backup_dir / "data" / "analytics" / analytics_db.name
        )
        assert {"cache_db", "analytics"} <= set(result.included)
        for backed_up in (backed_up_cache, backed_up_analytics):
            with duckdb.connect(str(backed_up), read_only=True) as conn:
                assert conn.execute("SELECT value FROM wal_probe").fetchall() == [(7,)]
    finally:
        cache_writer.close()
        analytics_writer.close()


@pytest.mark.unit
def test_chat_backup_includes_committed_wal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    chat_db = Path(cfg.chat.sqlite_path)
    chat_db.parent.mkdir(parents=True, exist_ok=True)
    writer = sqlite3.connect(chat_db, isolation_level=None)
    try:
        writer.execute("PRAGMA journal_mode=WAL;")
        writer.execute("CREATE TABLE chat_probe (value TEXT NOT NULL);")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        writer.execute("INSERT INTO chat_probe VALUES ('committed-in-wal');")
        assert chat_db.with_name(f"{chat_db.name}-wal").exists()

        result = _backup_chat_only(cfg, tmp_path)
        backed_up = result.backup_dir / "data" / chat_db.name
        with closing(sqlite3.connect(backed_up)) as conn, conn:
            assert conn.execute("SELECT value FROM chat_probe;").fetchone() == (
                "committed-in-wal",
            )
    finally:
        writer.close()


@pytest.mark.unit
def test_chat_backup_uses_configured_custom_sqlite_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    custom_path = tmp_path / "data" / "custom" / "conversation's state.sqlite"
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        chat={"sqlite_path": custom_path},
        backup_enabled=True,
    )
    custom_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(custom_path)) as conn, conn:
        conn.execute("CREATE TABLE custom_probe (value INTEGER NOT NULL);")
        conn.execute("INSERT INTO custom_probe VALUES (7);")

    result = _backup_chat_only(cfg, tmp_path)
    backed_up = result.backup_dir / "data" / custom_path.name

    assert "chat_db" in result.included
    assert backed_up.is_file()
    with closing(sqlite3.connect(backed_up)) as conn, conn:
        assert conn.execute("SELECT value FROM custom_probe;").fetchone() == (7,)


@pytest.mark.unit
def test_create_backup_requires_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raises when backups are disabled.

    Args:
        tmp_path: Temporary directory for test artifacts.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        ValueError: When backups are disabled.
    """
    monkeypatch.chdir(tmp_path)

    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=False,
    )

    with pytest.raises(ValueError, match="Backups are disabled"):
        create_backup(dest_root=tmp_path / "backups", qdrant_snapshot=False, cfg=cfg)


@pytest.mark.unit
def test_incomplete_backup_never_prunes_previous_known_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing required artifact preserves the previous recovery point."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        chat={"sqlite_path": tmp_path / "missing-chat.sqlite"},
        backup_enabled=True,
    )
    cache_db = cfg.cache.ingestion_db_path
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(cache_db)) as conn:
        conn.execute("CREATE TABLE cache_probe (value INTEGER NOT NULL)")
    _seed_active_snapshot(cfg)
    backup_root = tmp_path / "backups"
    previous = backup_root / "backup_20200101_000000"
    previous.mkdir(parents=True)
    (previous / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "included": ["cache_db", "snapshots", "chat_db"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    result = create_backup(
        dest_root=backup_root,
        keep_last=1,
        qdrant_snapshot=False,
        cfg=cfg,
    )
    manifest = json.loads(
        (result.backup_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["complete"] is False
    assert any("chat_db: missing" in warning for warning in result.warnings)
    assert previous.is_dir()
    assert len([path for path in backup_root.iterdir() if path.is_dir()]) == 2


@pytest.mark.unit
def test_wrong_type_storage_is_incomplete_and_not_retained_as_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "storage").write_text("not a directory", encoding="utf-8")
    cache_db = cfg.cache.ingestion_db_path
    cache_db.parent.mkdir(parents=True)
    with duckdb.connect(str(cache_db)) as conn:
        conn.execute("CREATE TABLE cache_probe (value INTEGER)")
    chat_db = Path(cfg.chat.sqlite_path)
    chat_db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(chat_db)) as conn, conn:
        conn.execute("CREATE TABLE chat_probe (value INTEGER)")
    backup_root = tmp_path / "backups"
    previous = backup_root / "backup_20200101_000000"
    previous.mkdir(parents=True)
    (previous / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "included": ["cache_db", "snapshots", "chat_db"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    result = create_backup(
        dest_root=backup_root,
        keep_last=1,
        qdrant_snapshot=False,
        cfg=cfg,
    )

    assert result.backup_dir.name.startswith("incomplete-backup_")
    assert any("writer lease unavailable" in item for item in result.warnings)
    assert previous.is_dir()
    assert mod.prune_backups(backup_root, keep_last=1) == []
    assert previous.is_dir()


@pytest.mark.unit
def test_wrong_type_data_dir_is_an_incomplete_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "data"
    cfg = create_test_settings(
        data_dir=data_path,
        cache={"dir": tmp_path / "cache"},
        chat={"sqlite_path": tmp_path / "chat.sqlite"},
        backup_enabled=True,
    )
    data_path.rmdir()
    data_path.write_text("not a directory", encoding="utf-8")
    cache_db = cfg.cache.ingestion_db_path
    cache_db.parent.mkdir(parents=True)
    with duckdb.connect(str(cache_db)) as conn:
        conn.execute("CREATE TABLE cache_probe (value INTEGER)")
    with closing(sqlite3.connect(cfg.chat.sqlite_path)) as conn, conn:
        conn.execute("CREATE TABLE chat_probe (value INTEGER)")

    result = create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=True,
        cfg=cfg,
    )

    assert result.backup_dir.name.startswith("incomplete-backup_")
    assert any("writer lease unavailable" in item for item in result.warnings)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("include_uploads", "qdrant_snapshot", "expected_warning"),
    [
        (False, True, "uploads: required capture disabled"),
        (True, False, "qdrant: required capture disabled"),
    ],
)
def test_partial_capture_never_replaces_a_complete_recovery_point(
    include_uploads: bool,
    qdrant_snapshot: bool,
    expected_warning: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabling either required owner produces a diagnostic backup only."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    _install_verified_qdrant(monkeypatch)
    root = tmp_path / "backups"
    complete = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=1,
        cfg=cfg,
    )

    partial = create_backup(
        dest_root=root,
        include_uploads=include_uploads,
        keep_last=1,
        qdrant_snapshot=qdrant_snapshot,
        cfg=cfg,
    )

    assert complete.backup_dir.is_dir()
    assert partial.backup_dir.name.startswith("incomplete-backup_")
    assert any(expected_warning in item for item in partial.warnings)


@pytest.mark.unit
def test_missing_upload_tree_is_an_incomplete_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    (cfg.data_dir / "uploads" / "document.txt").unlink()
    (cfg.data_dir / "uploads").rmdir()
    _install_verified_qdrant(monkeypatch)

    result = create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=True,
        cfg=cfg,
    )

    assert result.backup_dir.name.startswith("incomplete-backup_")
    assert any("uploads: missing" in item for item in result.warnings)


@pytest.mark.unit
def test_backup_recovers_precommit_upload_quarantine_before_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash before CURRENT activation restores the authoritative source."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    source = cfg.data_dir / "uploads" / "document.txt"
    quarantine_upload(
        data_dir=cfg.data_dir,
        source_path=source,
        transaction_id="crashed-precommit",
        collections={"text": "future-text", "image": "future-image"},
    )
    assert not source.exists()
    _install_verified_qdrant(monkeypatch)

    result = create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=True,
        cfg=cfg,
    )

    assert result.warnings == []
    assert source.read_text(encoding="utf-8") == "authoritative corpus"
    assert (result.backup_dir / "data" / "uploads" / source.name).read_text(
        encoding="utf-8"
    ) == "authoritative corpus"
    assert not (cfg.data_dir / ".quarantine").exists()


@pytest.mark.unit
@pytest.mark.parametrize("failure_mode", ["malformed", "symlink", "conflict"])
def test_unresolved_upload_recovery_is_an_incomplete_diagnostic(
    failure_mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous upload recovery can never evict a known-good backup."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    _install_verified_qdrant(monkeypatch)
    root = tmp_path / "backups"
    complete = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=1,
        cfg=cfg,
    )

    source = cfg.data_dir / "uploads" / "document.txt"
    _, quarantined = quarantine_upload(
        data_dir=cfg.data_dir,
        source_path=source,
        transaction_id=f"unresolved-{failure_mode}",
        collections={"text": "future-text", "image": "future-image"},
    )
    journal = quarantined.parent.parent / "transaction.json"
    if failure_mode == "malformed":
        journal.write_text("{", encoding="utf-8")
    elif failure_mode == "symlink":
        external = tmp_path / "external-transaction.json"
        external.write_text(journal.read_text(encoding="utf-8"), encoding="utf-8")
        journal.unlink()
        journal.symlink_to(external)
    else:
        source.write_text("conflicting authoritative bytes", encoding="utf-8")

    result = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=1,
        cfg=cfg,
    )

    assert result.backup_dir.name.startswith("incomplete-backup_")
    assert any(
        "uploads: transaction recovery incomplete" in warning
        for warning in result.warnings
    )
    assert complete.backup_dir.is_dir()
    assert quarantined.parent.is_dir()


@pytest.mark.unit
def test_unresolved_upload_promotion_is_an_incomplete_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable promotion journal also blocks retention eligibility."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    _install_verified_qdrant(monkeypatch)
    root = tmp_path / "backups"
    complete = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=1,
        cfg=cfg,
    )
    transaction = cfg.data_dir / ".upload-transactions" / "unreadable"
    transaction.mkdir(parents=True)
    (transaction / "transaction.json").write_text("{", encoding="utf-8")

    result = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=1,
        cfg=cfg,
    )

    assert result.backup_dir.name.startswith("incomplete-backup_")
    assert any(
        "uploads: transaction recovery incomplete" in warning
        for warning in result.warnings
    )
    assert complete.backup_dir.is_dir()
    assert transaction.is_dir()


@pytest.mark.unit
def test_upload_drift_from_current_is_an_incomplete_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copied uploads must retain the corpus identity activated by CURRENT."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    _install_verified_qdrant(monkeypatch)
    root = tmp_path / "backups"
    complete = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=1,
        cfg=cfg,
    )
    (cfg.data_dir / "uploads" / "document.txt").write_text(
        "changed after activation",
        encoding="utf-8",
    )

    result = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=1,
        cfg=cfg,
    )

    assert result.backup_dir.name.startswith("incomplete-backup_")
    assert any("does not match CURRENT identity" in item for item in result.warnings)
    assert complete.backup_dir.is_dir()


@pytest.mark.unit
def test_snapshot_writer_is_blocked_through_upload_and_qdrant_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canonical writer lease spans every activation-owned artifact."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    qdrant_capture_started = threading.Event()
    allow_qdrant_capture = threading.Event()
    upload_copy_started = threading.Event()
    allow_upload_copy = threading.Event()

    def _qdrant(**kwargs: object):  # type: ignore[no-untyped-def]
        qdrant_capture_started.set()
        assert allow_qdrant_capture.wait(2.0)
        destination = kwargs["dest_dir"]
        collections = kwargs["target_collections"]
        assert isinstance(destination, Path)
        assert isinstance(collections, dict)
        return _write_verified_qdrant_capture(
            dest_dir=destination,
            target_collections=collections,
        )

    original_copy_tree = mod._copy_tree

    def _copy_tree(source: Path, destination: Path) -> int:
        if source == cfg.data_dir / "uploads":
            upload_copy_started.set()
            assert allow_upload_copy.wait(2.0)
        return original_copy_tree(source, destination)

    monkeypatch.setattr(mod, "_create_qdrant_snapshots", _qdrant)
    monkeypatch.setattr(mod, "_copy_tree", _copy_tree)
    outcome: dict[str, object] = {}

    def _run_backup() -> None:
        try:
            outcome["result"] = create_backup(
                dest_root=tmp_path / "backups",
                include_uploads=True,
                cfg=cfg,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            outcome["error"] = exc

    worker = threading.Thread(target=_run_backup)
    worker.start()
    try:
        assert upload_copy_started.wait(2.0)
        upload_contender = SnapshotLock(
            cfg.data_dir / "storage" / ".lock",
            timeout=0.05,
            ttl_seconds=30.0,
        )
        with pytest.raises(SnapshotLockTimeoutError):
            upload_contender.acquire()
        allow_upload_copy.set()
        assert qdrant_capture_started.wait(2.0)
        qdrant_contender = SnapshotLock(
            cfg.data_dir / "storage" / ".lock",
            timeout=0.05,
            ttl_seconds=30.0,
        )
        with pytest.raises(SnapshotLockTimeoutError):
            qdrant_contender.acquire()
    finally:
        allow_upload_copy.set()
        allow_qdrant_capture.set()
        worker.join(timeout=3.0)

    assert not worker.is_alive()
    assert "error" not in outcome
    assert isinstance(outcome.get("result"), mod.BackupResult)


@pytest.mark.unit
@pytest.mark.parametrize("failure_phase", ["setup", "acquire"])
def test_snapshot_lock_failure_creates_an_incomplete_backup(
    failure_phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    cache_db = cfg.cache.ingestion_db_path
    cache_db.parent.mkdir(parents=True)
    with duckdb.connect(str(cache_db)) as conn:
        conn.execute("CREATE TABLE cache_probe (value INTEGER)")
    chat_db = Path(cfg.chat.sqlite_path)
    chat_db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(chat_db)) as conn, conn:
        conn.execute("CREATE TABLE chat_probe (value INTEGER)")

    class _UnavailableLock:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            if failure_phase == "setup":
                raise OSError("lock path unavailable")

        def __enter__(self) -> None:
            raise SnapshotLockTimeoutError("writer busy")

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(mod, "SnapshotLock", _UnavailableLock)

    result = create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=True,
        cfg=cfg,
    )

    assert result.backup_dir.name.startswith("incomplete-backup_")
    assert any("writer lease unavailable" in item for item in result.warnings)


@pytest.mark.unit
def test_invalid_deployment_identity_is_not_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    (cfg.data_dir / DEPLOYMENT_ID_FILENAME).write_text("not-a-uuid\n", encoding="ascii")
    _install_verified_qdrant(monkeypatch)

    result = create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=True,
        cfg=cfg,
    )

    assert result.backup_dir.name.startswith("incomplete-backup_")
    assert "deployment_identity" not in result.included
    assert any("deployment identity" in item for item in result.warnings)
    assert not (result.backup_dir / "data" / DEPLOYMENT_ID_FILENAME).exists()


@pytest.mark.unit
@pytest.mark.parametrize("source_kind", ["storage", "artifacts", "uploads", "logs"])
def test_backup_destination_cannot_be_inside_recursive_source(
    source_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(data_dir=tmp_path / "data", backup_enabled=True)
    kwargs: dict[str, bool] = {"qdrant_snapshot": False}
    if source_kind == "storage":
        source = cfg.data_dir / "storage"
    elif source_kind == "artifacts":
        source = cfg.data_dir / "artifacts"
    elif source_kind == "uploads":
        source = cfg.data_dir / "uploads"
        kwargs["include_uploads"] = True
    else:
        source = tmp_path / "logs"
        kwargs["include_logs"] = True
    destination = source / "backups"

    with pytest.raises(ValueError, match="recursively copied source"):
        create_backup(dest_root=destination, cfg=cfg, **kwargs)

    assert not destination.exists()


@pytest.mark.unit
def test_same_second_backups_are_unique_and_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    _install_verified_qdrant(monkeypatch)
    monkeypatch.setattr(mod, "_utc_timestamp_compact", lambda: "20260713_120000")
    root = tmp_path / "backups"

    first = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=2,
        cfg=cfg,
    )
    second = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=2,
        cfg=cfg,
    )

    assert first.backup_dir != second.backup_dir
    assert first.backup_dir.is_dir()
    assert second.backup_dir.is_dir()
    assert not any(path.name.startswith("tmp-backup_") for path in root.iterdir())


@pytest.mark.unit
def test_failed_prepromotion_verification_downgrades_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No nominally complete workspace bypasses the retention validator."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    _install_verified_qdrant(monkeypatch)
    monkeypatch.setattr(mod, "_retention_eligible_backup", lambda *_args: False)

    result = create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=True,
        cfg=cfg,
    )
    manifest = json.loads(
        (result.backup_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert result.backup_dir.name.startswith("incomplete-backup_")
    assert result.warnings == [
        "backup: pre-promotion recoverability verification failed"
    ]
    assert manifest["complete"] is False
    assert manifest["warnings"] == result.warnings


@pytest.mark.unit
def test_backup_is_durable_before_retention_prunes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workspace data and its promoted root entry are fsynced before pruning."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    _install_verified_qdrant(monkeypatch)
    root = tmp_path / "backups"
    first = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=1,
        cfg=cfg,
    )

    original_fsync_tree = mod._fsync_tree
    original_fsync_directory = mod._fsync_directory
    original_prune = mod._prune_backups_unlocked
    state = {"workspace": False, "root": False, "pruned": False}

    def _fsync_tree(path: Path) -> None:
        original_fsync_tree(path)
        state["workspace"] = True

    def _fsync_directory(path: Path) -> None:
        original_fsync_directory(path)
        if path == root and state["workspace"]:
            state["root"] = True

    def _prune(path: Path, *, keep_last: int) -> list[Path]:
        assert state == {"workspace": True, "root": True, "pruned": False}
        state["pruned"] = True
        return original_prune(path, keep_last=keep_last)

    monkeypatch.setattr(mod, "_fsync_tree", _fsync_tree)
    monkeypatch.setattr(mod, "_fsync_directory", _fsync_directory)
    monkeypatch.setattr(mod, "_prune_backups_unlocked", _prune)

    second = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=1,
        cfg=cfg,
    )

    assert state == {"workspace": True, "root": True, "pruned": True}
    assert not first.backup_dir.exists()
    assert second.backup_dir.is_dir()


@pytest.mark.unit
def test_workspace_fsync_failure_preserves_prior_complete_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durability failure aborts promotion and leaves retention untouched."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    _install_verified_qdrant(monkeypatch)
    root = tmp_path / "backups"
    first = create_backup(
        dest_root=root,
        include_uploads=True,
        keep_last=1,
        cfg=cfg,
    )

    def _fail_fsync(_path: Path) -> None:
        raise OSError("durability unavailable")

    monkeypatch.setattr(mod, "_fsync_tree", _fail_fsync)

    with pytest.raises(OSError, match="durability unavailable"):
        create_backup(
            dest_root=root,
            include_uploads=True,
            keep_last=1,
            cfg=cfg,
        )

    assert first.backup_dir.is_dir()
    assert not any(path.name.startswith("tmp-backup_") for path in root.iterdir())
    assert [path for path in root.iterdir() if path.name.startswith("backup_")] == [
        first.backup_dir
    ]


@pytest.mark.unit
def test_failed_backup_removes_sensitive_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(data_dir=tmp_path / "data", backup_enabled=True)

    def _fail_population(**kwargs: object):  # type: ignore[no-untyped-def]
        workspace = kwargs["tmp_dir"]
        assert isinstance(workspace, Path)
        (workspace / ".env").write_text("SECRET=value", encoding="utf-8")
        raise RuntimeError("copy failed")

    monkeypatch.setattr(mod, "_populate_backup_workspace", _fail_population)
    root = tmp_path / "backups"

    with pytest.raises(RuntimeError, match="copy failed"):
        create_backup(
            dest_root=root,
            include_env=True,
            qdrant_snapshot=False,
            cfg=cfg,
        )

    assert not any(path.is_dir() for path in root.iterdir())


@pytest.mark.unit
def test_backup_includes_custom_artifact_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    artifact_root = tmp_path / "custom-artifacts"
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        artifacts={"dir": artifact_root},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    artifact_root.mkdir(parents=True)
    (artifact_root / "page.webp").write_bytes(b"image")

    result = create_backup(
        dest_root=tmp_path / "backups",
        qdrant_snapshot=False,
        cfg=cfg,
    )

    assert "artifacts" in result.included
    assert (result.backup_dir / "data/artifacts/page.webp").read_bytes() == b"image"
    manifest = json.loads(
        (result.backup_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifacts"] == {
        "files": [
            {
                "path": "page.webp",
                "size_bytes": len(b"image"),
                "sha256": hashlib.sha256(b"image").hexdigest(),
            }
        ]
    }


@pytest.mark.unit
def test_empty_image_validation_does_not_create_an_unlisted_artifact_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validating an empty image collection is side-effect-free."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)

    class _EmptyImageClient:
        def scroll(self, **_kwargs: object) -> tuple[list[object], None]:
            return [], None

    def _capture(**kwargs: object):  # type: ignore[no-untyped-def]
        destination = kwargs["dest_dir"]
        collections = kwargs["target_collections"]
        warnings = kwargs["warnings"]
        artifact_dir = kwargs["artifact_dir"]
        assert isinstance(destination, Path)
        assert isinstance(collections, dict)
        assert isinstance(warnings, list)
        assert isinstance(artifact_dir, Path)
        mod._validate_image_artifact_references(
            client=_EmptyImageClient(),
            collection=collections["image"],
            expected_points=0,
            artifact_dir=artifact_dir,
            warnings=warnings,
        )
        return _write_verified_qdrant_capture(
            dest_dir=destination,
            target_collections=collections,
        )

    monkeypatch.setattr(mod, "_create_qdrant_snapshots", _capture)

    result = create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=True,
        cfg=cfg,
    )

    assert result.warnings == []
    assert result.backup_dir.name.startswith("backup_")
    assert "artifacts" not in result.included
    assert not (result.backup_dir / "data" / "artifacts").exists()


@pytest.mark.unit
def test_qdrant_snapshot_metadata_serializes_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slotted snapshot metadata is represented explicitly in JSON."""
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    active_snapshot = _seed_required_backup_sources(cfg)

    def _snapshots(**kwargs: object):  # type: ignore[no-untyped-def]
        assert kwargs["target_collections"] == _PHYSICAL_COLLECTIONS
        destination = kwargs["dest_dir"]
        assert isinstance(destination, Path)
        return _write_verified_qdrant_capture(
            dest_dir=destination,
            target_collections=dict(_PHYSICAL_COLLECTIONS),
        )

    monkeypatch.setattr(mod, "_create_qdrant_snapshots", _snapshots)

    result = create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=True,
        cfg=cfg,
    )
    manifest = json.loads(
        (result.backup_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["activation"] == {
        "snapshot": active_snapshot.name,
        "collections": _PHYSICAL_COLLECTIONS,
        "deployment_id": get_or_create_deployment_id(cfg.data_dir),
    }
    assert {item["collection"] for item in manifest["qdrant"]["collections"]} == set(
        _PHYSICAL_COLLECTIONS.values()
    )
    assert manifest["complete"] is True


@pytest.mark.unit
def test_image_points_without_artifacts_mark_validation_incomplete(
    tmp_path: Path,
) -> None:
    """Image-vector references cannot outlive their required local payloads."""
    payload = {
        "image_artifact_id": hashlib.sha256(b"missing").hexdigest(),
        "image_artifact_suffix": ".webp",
    }
    client = _ArtifactScrollClient([([SimpleNamespace(payload=payload)], None)])
    warnings: list[str] = []

    mod._validate_image_artifact_references(
        client=client,
        collection="images",
        expected_points=1,
        artifact_dir=tmp_path / "artifacts",
        warnings=warnings,
    )

    assert any("missing copied artifact" in item for item in warnings)


@pytest.mark.unit
def test_image_artifact_scan_rejects_partial_collection_coverage(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    payload = _artifact_payload(artifact_root, b"first")
    client = _ArtifactScrollClient([([SimpleNamespace(payload=payload)], None)])
    warnings: list[str] = []

    mod._validate_image_artifact_references(
        client=client,
        collection="images",
        expected_points=2,
        artifact_dir=artifact_root,
        warnings=warnings,
    )

    assert any("artifact scan covered 1 of 2" in item for item in warnings)


@pytest.mark.unit
def test_image_artifact_scan_rejects_wrong_file_bytes(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    expected_digest = hashlib.sha256(b"expected").hexdigest()
    artifact_root.mkdir()
    (artifact_root / f"{expected_digest}.webp").write_bytes(b"wrong-bytes")
    client = _ArtifactScrollClient(
        [
            (
                [
                    SimpleNamespace(
                        payload={
                            "image_artifact_id": expected_digest,
                            "image_artifact_suffix": ".webp",
                        }
                    )
                ],
                None,
            )
        ]
    )
    warnings: list[str] = []

    mod._validate_image_artifact_references(
        client=client,
        collection="images",
        expected_points=1,
        artifact_dir=artifact_root,
        warnings=warnings,
    )

    assert any("digest mismatch" in item for item in warnings)


@pytest.mark.unit
def test_image_artifact_scan_sanitizes_scroll_failure(tmp_path: Path) -> None:
    client = _ArtifactScrollClient(
        [], error=OSError("/private/customer/secret-document.webp")
    )
    warnings: list[str] = []

    mod._validate_image_artifact_references(
        client=client,
        collection="images",
        expected_points=1,
        artifact_dir=tmp_path / "artifacts",
        warnings=warnings,
    )

    assert len(warnings) == 1
    assert "image artifact scan failed: OSError" in warnings[0]
    assert "[redacted:" in warnings[0]
    assert "secret-document" not in warnings[0]


@pytest.mark.unit
def test_image_artifact_scan_rejects_malformed_reference_payload(
    tmp_path: Path,
) -> None:
    client = _ArtifactScrollClient(
        [
            (
                [
                    SimpleNamespace(
                        payload={
                            "image_artifact_id": "not-a-sha256",
                            "image_artifact_suffix": ".webp",
                            "thumbnail_artifact_id": "a" * 64,
                        }
                    )
                ],
                None,
            )
        ]
    )
    warnings: list[str] = []

    mod._validate_image_artifact_references(
        client=client,
        collection="images",
        expected_points=1,
        artifact_dir=tmp_path / "artifacts",
        warnings=warnings,
    )

    assert any("malformed artifact reference" in item for item in warnings)


@pytest.mark.unit
def test_image_artifact_scan_accepts_all_valid_paginated_refs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    first = _artifact_payload(artifact_root, b"first")
    thumbnail_body = b"thumbnail"
    thumbnail_digest = hashlib.sha256(thumbnail_body).hexdigest()
    (artifact_root / f"{thumbnail_digest}.webp").write_bytes(thumbnail_body)
    first.update(
        {
            "thumbnail_artifact_id": thumbnail_digest,
            "thumbnail_artifact_suffix": ".webp",
        }
    )
    second = _artifact_payload(artifact_root, b"second")
    client = _ArtifactScrollClient(
        [
            ([SimpleNamespace(payload=first)], "next-page"),
            ([SimpleNamespace(payload=second)], None),
        ]
    )
    warnings: list[str] = []

    mod._validate_image_artifact_references(
        client=client,
        collection="images",
        expected_points=2,
        artifact_dir=artifact_root,
        warnings=warnings,
    )

    assert warnings == []
    assert [call["offset"] for call in client.calls] == [None, "next-page"]
    assert all(call["with_vectors"] is False for call in client.calls)
    assert all(
        call["with_payload"]
        == [
            "image_artifact_id",
            "image_artifact_suffix",
            "thumbnail_artifact_id",
            "thumbnail_artifact_suffix",
        ]
        for call in client.calls
    )


@pytest.mark.unit
def test_qdrant_snapshot_download_skips_allowlist_when_remote_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote snapshot downloads skip allowlist checks.

    Args:
        tmp_path: Temporary path fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.chdir(tmp_path)

    def _deny_allowlist(*args: object, **kwargs: object) -> bool:
        raise AssertionError("endpoint allowlist should not be consulted")

    monkeypatch.setattr(mod, "endpoint_url_allowed", _deny_allowlist)
    monkeypatch.setattr(
        mod,
        "_open_qdrant_snapshot",
        lambda *args, **kwargs: nullcontext(io.BytesIO(b"snapshot-bytes")),
    )

    dest_file = tmp_path / "qdrant" / "collection" / "snapshot.bin"
    written = mod._download_qdrant_snapshot(
        qdrant_url="https://qdrant.example.com",
        api_key=None,
        collection="collection",
        snapshot_name="snapshot.bin",
        dest_file=dest_file,
        timeout_s=5,
        allow_remote_endpoints=True,
        allowed_hosts=set(),
    )

    assert written == len(b"snapshot-bytes")
    assert dest_file.read_bytes() == b"snapshot-bytes"


@pytest.mark.unit
def test_qdrant_snapshot_redirects_are_rejected() -> None:
    """Snapshot credentials cannot be forwarded through an HTTP redirect."""
    request = mod.urllib.request.Request(
        "http://127.0.0.1:6333/collections/documents/snapshots/snapshot.bin",
        headers={"api-key": "secret"},
    )

    redirected = mod._NoRedirectHandler().redirect_request(
        request,
        io.BytesIO(),
        302,
        "Found",
        {},
        "https://attacker.example/snapshot.bin",
    )

    assert redirected is None


@pytest.mark.unit
def test_qdrant_snapshot_client_uses_backup_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The snapshot client and download policy share the supplied settings."""
    from src.utils import storage as storage_mod

    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        database={"qdrant_url": "http://127.0.0.1:7333"},
    )
    captured: list[object] = []

    class _Client:
        def info(self) -> SimpleNamespace:
            return SimpleNamespace(version="1.18.2")

        def cluster_status(self) -> SimpleNamespace:
            return SimpleNamespace(status="disabled")

        def get_collections(self) -> SimpleNamespace:
            return SimpleNamespace(collections=[])

    def _client_factory(client_cfg: object) -> nullcontext[_Client]:
        captured.append(client_cfg)
        return nullcontext(_Client())

    monkeypatch.setattr(storage_mod, "create_sync_client", _client_factory)

    snapshots, bytes_written, version = mod._create_qdrant_snapshots(
        cfg=cfg,
        target_collections=dict(_PHYSICAL_COLLECTIONS),
        dest_dir=tmp_path / "backup",
        warnings=[],
        maintenance_warnings=[],
    )

    assert captured == [cfg]
    assert snapshots == []
    assert bytes_written == 0
    assert version == "1.18.2"


@pytest.mark.unit
@pytest.mark.parametrize("mismatch", ["deployment", "owner"])
def test_qdrant_snapshot_rejects_collection_ownership_mismatch(
    mismatch: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-named foreign or owner-swapped collections cannot enter a backup."""
    from src.utils import storage as storage_mod

    cfg = create_test_settings(data_dir=tmp_path / "data")
    deployment_id = get_or_create_deployment_id(cfg.data_dir)
    expected_metadata = {
        owner: {
            "docmind_deployment_id": deployment_id,
            "docmind_owner": owner,
        }
        for owner in ("text", "image")
    }

    class _Client:
        def info(self) -> SimpleNamespace:
            return SimpleNamespace(version="1.18.2")

        def cluster_status(self) -> SimpleNamespace:
            return SimpleNamespace(status="disabled")

        def get_collections(self) -> SimpleNamespace:
            return SimpleNamespace(
                collections=[
                    SimpleNamespace(name=collection)
                    for collection in _PHYSICAL_COLLECTIONS.values()
                ]
            )

        def get_collection(self, *, collection_name: str) -> SimpleNamespace:
            owner = next(
                key
                for key, collection in _PHYSICAL_COLLECTIONS.items()
                if collection == collection_name
            )
            metadata = dict(expected_metadata[owner])
            if mismatch == "deployment":
                metadata["docmind_deployment_id"] = (
                    "00000000-0000-4000-8000-000000000099"
                )
            else:
                metadata["docmind_owner"] = "image" if owner == "text" else "text"
            return SimpleNamespace(config=SimpleNamespace(metadata=metadata))

        def create_snapshot(self, **_kwargs: object) -> None:
            raise AssertionError("foreign collection must not be snapshotted")

    monkeypatch.setattr(
        storage_mod,
        "create_sync_client",
        lambda _cfg: nullcontext(_Client()),
    )
    warnings: list[str] = []

    snapshots, bytes_written, version = mod._create_qdrant_snapshots(
        cfg=cfg,
        target_collections=dict(_PHYSICAL_COLLECTIONS),
        dest_dir=tmp_path / "backup",
        warnings=warnings,
        maintenance_warnings=[],
        target_collection_metadata=expected_metadata,
        deployment_id=deployment_id,
    )

    assert snapshots == []
    assert bytes_written == 0
    assert version == "1.18.2"
    assert len(warnings) == 2
    assert all("ownership metadata does not match CURRENT" in item for item in warnings)


@pytest.mark.unit
def test_qdrant_snapshot_rejects_truncated_success_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short HTTP-200 body cannot become an included backup artifact."""
    from src.utils import storage as storage_mod

    cfg = create_test_settings(data_dir=tmp_path / "data")
    deleted: list[str] = []
    collection = _PHYSICAL_COLLECTIONS["text"]

    class _Client:
        def info(self) -> SimpleNamespace:
            return SimpleNamespace(version="1.18.2")

        def cluster_status(self) -> SimpleNamespace:
            return SimpleNamespace(status="disabled")

        def get_collections(self) -> SimpleNamespace:
            return SimpleNamespace(collections=[SimpleNamespace(name=collection)])

        def count(self, *, collection_name: str, exact: bool) -> SimpleNamespace:
            assert collection_name == collection
            assert exact is True
            return SimpleNamespace(count=7)

        def create_snapshot(
            self, *, collection_name: str
        ) -> qmodels.SnapshotDescription:
            assert collection_name == collection
            return qmodels.SnapshotDescription(name="snapshot.bin", size=10)

        def delete_snapshot(self, *, collection_name: str, snapshot_name: str) -> bool:
            assert collection_name == collection
            deleted.append(snapshot_name)
            return True

    monkeypatch.setattr(
        storage_mod,
        "create_sync_client",
        lambda _cfg: nullcontext(_Client()),
    )

    def _short_download(**kwargs: object) -> int:
        dest_file = kwargs["dest_file"]
        assert isinstance(dest_file, Path)
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_bytes(b"short")
        return 5

    monkeypatch.setattr(mod, "_download_qdrant_snapshot", _short_download)
    warnings: list[str] = []

    snapshots, bytes_written, version = mod._create_qdrant_snapshots(
        cfg=cfg,
        target_collections=dict(_PHYSICAL_COLLECTIONS),
        dest_dir=tmp_path / "backup",
        warnings=warnings,
        maintenance_warnings=[],
    )

    assert snapshots == []
    assert bytes_written == 0
    assert version == "1.18.2"
    assert deleted == ["snapshot.bin"]
    assert not (tmp_path / "backup" / "qdrant" / collection / "snapshot.bin").exists()
    assert any("ValueError" in warning for warning in warnings)


@pytest.mark.unit
def test_qdrant_snapshot_download_removes_partial_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupted downloads leave neither a final file nor a partial artifact."""
    monkeypatch.setattr(
        mod,
        "_open_qdrant_snapshot",
        lambda *args, **kwargs: nullcontext(io.BytesIO(b"partial")),
    )

    def _fail_copy(source: object, destination: object) -> None:
        assert hasattr(destination, "write")
        destination.write(b"partial")  # type: ignore[attr-defined]
        raise OSError("connection interrupted")

    monkeypatch.setattr(mod.shutil, "copyfileobj", _fail_copy)
    dest_file = tmp_path / "qdrant" / "documents" / "snapshot.bin"

    with pytest.raises(OSError, match="connection interrupted"):
        mod._download_qdrant_snapshot(
            qdrant_url="http://127.0.0.1:6333",
            api_key=None,
            collection="documents",
            snapshot_name="snapshot.bin",
            dest_file=dest_file,
            timeout_s=5,
            allow_remote_endpoints=False,
            allowed_hosts=set(),
        )

    assert not dest_file.exists()
    assert list(dest_file.parent.iterdir()) == []


@pytest.mark.unit
def test_qdrant_snapshot_rejects_same_size_checksum_mismatch(tmp_path: Path) -> None:
    """An advertised Qdrant SHA256 protects against same-length corruption."""
    dest_file = tmp_path / "snapshot.bin"
    body = b"same-length-corruption"
    dest_file.write_bytes(body)
    description = qmodels.SnapshotDescription(
        name=dest_file.name,
        size=len(body),
        checksum=hashlib.sha256(b"different-content").hexdigest(),
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        mod._validate_qdrant_snapshot(
            description=description,
            downloaded_bytes=len(body),
            dest_file=dest_file,
        )

    assert not dest_file.exists()


@pytest.mark.unit
def test_qdrant_snapshot_records_verified_checksum_count_and_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verified data is kept while server cleanup debt remains maintenance-only."""
    from src.utils import storage as storage_mod

    body = b"verified-snapshot"
    checksum = hashlib.sha256(body).hexdigest()
    cfg = create_test_settings(data_dir=tmp_path / "data")
    collection = _PHYSICAL_COLLECTIONS["text"]

    class _Client:
        def info(self) -> SimpleNamespace:
            return SimpleNamespace(version="1.18.2")

        def cluster_status(self) -> SimpleNamespace:
            return SimpleNamespace(status="disabled")

        def get_collections(self) -> SimpleNamespace:
            return SimpleNamespace(collections=[SimpleNamespace(name=collection)])

        def count(self, *, collection_name: str, exact: bool) -> SimpleNamespace:
            assert collection_name == collection
            assert exact is True
            return SimpleNamespace(count=11)

        def create_snapshot(
            self, *, collection_name: str
        ) -> qmodels.SnapshotDescription:
            assert collection_name == collection
            return qmodels.SnapshotDescription(
                name="snapshot.bin",
                size=len(body),
                checksum=checksum,
            )

        def delete_snapshot(self, **_kwargs: object) -> None:
            raise OSError("server cleanup failed")

    monkeypatch.setattr(
        storage_mod,
        "create_sync_client",
        lambda _cfg: nullcontext(_Client()),
    )

    def _download(**kwargs: object) -> int:
        dest_file = kwargs["dest_file"]
        assert isinstance(dest_file, Path)
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_bytes(body)
        return len(body)

    monkeypatch.setattr(mod, "_download_qdrant_snapshot", _download)
    warnings: list[str] = []
    maintenance_warnings: list[str] = []

    snapshots, bytes_written, _version = mod._create_qdrant_snapshots(
        cfg=cfg,
        target_collections={"text": collection, "image": collection},
        dest_dir=tmp_path / "backup",
        warnings=warnings,
        maintenance_warnings=maintenance_warnings,
    )

    assert bytes_written == len(body)
    assert len(snapshots) == 1
    assert snapshots[0].checksum == checksum
    assert snapshots[0].point_count == 11
    assert warnings == []
    assert any("snapshot cleanup failed" in warning for warning in maintenance_warnings)


@pytest.mark.unit
def test_qdrant_snapshot_unconfirmed_cleanup_is_a_maintenance_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A false delete result must not silently leave a server-side snapshot."""
    from src.utils import storage as storage_mod

    body = b"verified-snapshot"
    cfg = create_test_settings(data_dir=tmp_path / "data")
    collection = _PHYSICAL_COLLECTIONS["text"]

    class _Client:
        def info(self) -> SimpleNamespace:
            return SimpleNamespace(version="1.18.2")

        def cluster_status(self) -> SimpleNamespace:
            return SimpleNamespace(status="disabled")

        def get_collections(self) -> SimpleNamespace:
            return SimpleNamespace(collections=[SimpleNamespace(name=collection)])

        def count(self, *, collection_name: str, exact: bool) -> SimpleNamespace:
            assert collection_name == collection
            assert exact is True
            return SimpleNamespace(count=1)

        def create_snapshot(
            self, *, collection_name: str
        ) -> qmodels.SnapshotDescription:
            assert collection_name == collection
            return qmodels.SnapshotDescription(name="snapshot.bin", size=len(body))

        def delete_snapshot(self, **_kwargs: object) -> bool:
            return False

    monkeypatch.setattr(
        storage_mod,
        "create_sync_client",
        lambda _cfg: nullcontext(_Client()),
    )

    def _download(**kwargs: object) -> int:
        dest_file = kwargs["dest_file"]
        assert isinstance(dest_file, Path)
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        dest_file.write_bytes(body)
        return len(body)

    monkeypatch.setattr(mod, "_download_qdrant_snapshot", _download)
    warnings: list[str] = []
    maintenance_warnings: list[str] = []

    snapshots, bytes_written, _version = mod._create_qdrant_snapshots(
        cfg=cfg,
        target_collections={"text": collection, "image": collection},
        dest_dir=tmp_path / "backup",
        warnings=warnings,
        maintenance_warnings=maintenance_warnings,
    )

    assert snapshots
    assert bytes_written == len(body)
    assert warnings == []
    assert any(
        "deletion was not confirmed" in warning for warning in maintenance_warnings
    )


@pytest.mark.unit
def test_cleanup_debt_is_preserved_without_downgrading_complete_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache={"dir": tmp_path / "cache"},
        backup_enabled=True,
    )
    _seed_required_backup_sources(cfg)
    cleanup_warning = "qdrant: server-side snapshot cleanup failed: stale snapshot"

    def _capture(**kwargs: object):  # type: ignore[no-untyped-def]
        maintenance = kwargs["maintenance_warnings"]
        destination = kwargs["dest_dir"]
        collections = kwargs["target_collections"]
        assert isinstance(maintenance, list)
        assert isinstance(destination, Path)
        assert isinstance(collections, dict)
        maintenance.append(cleanup_warning)
        return _write_verified_qdrant_capture(
            dest_dir=destination,
            target_collections=collections,
        )

    monkeypatch.setattr(mod, "_create_qdrant_snapshots", _capture)

    result = create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=True,
        cfg=cfg,
    )
    manifest = json.loads(
        (result.backup_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert result.backup_dir.name.startswith("backup_")
    assert result.warnings == []
    assert result.maintenance_warnings == [cleanup_warning]
    assert manifest["complete"] is True
    assert manifest["maintenance_warnings"] == [cleanup_warning]


@pytest.mark.unit
def test_qdrant_snapshot_rejects_distributed_single_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-node capture cannot masquerade as a distributed-cluster backup."""
    from src.utils import storage as storage_mod

    cfg = create_test_settings(data_dir=tmp_path / "data")

    class _Client:
        def info(self) -> SimpleNamespace:
            return SimpleNamespace(version="1.18.2")

        def cluster_status(self) -> SimpleNamespace:
            return SimpleNamespace(status="enabled", peers={"1": {}, "2": {}})

        def get_collections(self) -> SimpleNamespace:
            raise AssertionError("distributed collection capture must not start")

    monkeypatch.setattr(
        storage_mod,
        "create_sync_client",
        lambda _cfg: nullcontext(_Client()),
    )
    warnings: list[str] = []

    snapshots, bytes_written, version = mod._create_qdrant_snapshots(
        cfg=cfg,
        target_collections=dict(_PHYSICAL_COLLECTIONS),
        dest_dir=tmp_path / "backup",
        warnings=warnings,
        maintenance_warnings=[],
    )

    assert snapshots == []
    assert bytes_written == 0
    assert version == "1.18.2"
    assert any("distributed snapshots" in warning for warning in warnings)
