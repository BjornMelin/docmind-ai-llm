"""Tests for backup manifest creation."""

from __future__ import annotations

import io
import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from src.persistence import backup_service as mod
from src.persistence.backup_service import create_backup
from tests.fixtures.test_settings import create_test_settings


@pytest.mark.unit
def test_create_backup_writes_manifest(
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
        cache_dir=tmp_path / "cache",
        backup_enabled=False,
    )

    with pytest.raises(ValueError, match="Backups are disabled"):
        create_backup(dest_root=tmp_path / "backups", qdrant_snapshot=False, cfg=cfg)


@pytest.mark.unit
def test_qdrant_snapshot_download_skips_allowlist_when_remote_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote snapshot downloads should not consult the allowlist."""
    monkeypatch.chdir(tmp_path)

    def _deny_allowlist(*args: object, **kwargs: object) -> bool:
        raise AssertionError("endpoint allowlist should not be consulted")

    monkeypatch.setattr(mod, "endpoint_url_allowed", _deny_allowlist)
    monkeypatch.setattr(
        mod.urllib.request,
        "urlopen",
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
