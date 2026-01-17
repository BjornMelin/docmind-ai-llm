from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.persistence.backup_service import create_backup
from tests.fixtures.test_settings import create_test_settings


def test_create_backup_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        backup_enabled=True,
        backup_keep_last=3,
    )

    # Required artifacts
    (cfg.data_dir / "storage" / "snapshot_1").mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "storage" / "snapshot_1" / "manifest.jsonl").write_text(
        '{"ok": true}\n',
        encoding="utf-8",
    )
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_db = cfg.cache_dir / cfg.cache.filename
    cache_db.write_text("duckdb", encoding="utf-8")

    result = create_backup(
        dest_root=tmp_path / "backups",
        include_uploads=False,
        include_analytics=False,
        include_logs=False,
        keep_last=3,
        qdrant_snapshot=False,
        cfg=cfg,
    )

    manifest_path = result.backup_dir / "manifest.json"
    assert manifest_path.is_file()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["bytes_written"] == result.bytes_written
    assert payload["included"] == result.included
    assert payload["qdrant"]["collections"] == []

    assert "cache_db" in result.included
    assert "snapshots" in result.included


def test_create_backup_requires_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    cfg = create_test_settings(
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        backup_enabled=False,
    )

    with pytest.raises(ValueError, match="Backups are disabled"):
        create_backup(dest_root=tmp_path / "backups", qdrant_snapshot=False, cfg=cfg)
