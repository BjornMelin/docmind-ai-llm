"""Unit tests for content-addressed ArtifactStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.persistence.artifacts import ArtifactRef, ArtifactStore

pytestmark = pytest.mark.unit


def test_suffix_for_artifact_handles_empty_and_enc_suffix(tmp_path: Path) -> None:
    from src.persistence.artifacts import _suffix_for_artifact

    assert _suffix_for_artifact(tmp_path / "noext") == ""
    assert _suffix_for_artifact(tmp_path / "page.webp.enc") == ".webp.enc"


def test_put_file_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root=root)

    src = tmp_path / "x.webp"
    src.write_bytes(b"abc123")

    ref1 = store.put_file(src)
    assert len(ref1.sha256) == 64
    assert ref1.suffix == ".webp"
    assert store.exists(ref1)

    # Idempotent: storing same content yields same ref, no duplicate.
    before = sorted(p.name for p in root.glob("*"))
    ref2 = store.put_file(src)
    after = sorted(p.name for p in root.glob("*"))
    assert ref2 == ref1
    assert before == after


def test_resolve_path_validates_sha_and_suffix(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path / "artifacts")

    with pytest.raises(ValueError, match="Invalid artifact sha256"):
        store.resolve_path(ArtifactRef(sha256="not-hex", suffix=".webp"))

    with pytest.raises(ValueError, match="Invalid artifact suffix"):
        store.resolve_path(ArtifactRef(sha256="a" * 64, suffix="webp"))

    with pytest.raises(ValueError, match="Invalid artifact suffix"):
        store.resolve_path(ArtifactRef(sha256="a" * 64, suffix="../x"))

    with pytest.raises(ValueError, match="Invalid artifact suffix"):
        store.resolve_path(ArtifactRef(sha256="a" * 64, suffix=".this-is-way-too-long"))

    with pytest.raises(ValueError, match="Invalid artifact suffix"):
        store.resolve_path(ArtifactRef(sha256="a" * 64, suffix=".bad@x"))


def test_put_file_missing_raises(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path / "artifacts")
    with pytest.raises(FileNotFoundError):
        store.put_file(tmp_path / "missing.webp")


def test_put_file_handles_fileexists_race(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root=root)

    src = tmp_path / "x.webp"
    src.write_bytes(b"abc123")

    def _rename(self, target):  # type: ignore[no-untyped-def]
        Path(target).write_bytes(b"abc123")
        raise FileExistsError()

    monkeypatch.setattr(Path, "rename", _rename)

    ref = store.put_file(src)
    assert store.exists(ref)
    assert not any(p.suffix == ".tmp" for p in root.glob(".*.tmp"))


def test_delete_is_best_effort_when_missing(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path / "artifacts")
    ref = ArtifactRef(sha256="a" * 64, suffix=".webp")
    store.delete(ref)
