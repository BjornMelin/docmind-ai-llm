"""Parser model supply-chain integrity tests."""

from pathlib import Path

import pytest

from src.processing.parsing.backends.docling_backend import (
    _pinned_docling_layout_config,
)
from src.processing.parsing.model_artifacts import (
    DOCLING_LAYOUT_BUNDLE,
    ModelArtifact,
    ModelBundle,
    ModelIntegrityError,
    verify_model_bundle,
)

pytestmark = pytest.mark.unit

_TEST_BUNDLE = ModelBundle(
    name="test model",
    root="test-model",
    repo_id="example/test-model",
    revision="0123456789abcdef0123456789abcdef01234567",
    artifacts=(
        ModelArtifact(
            path="model.bin",
            size=13,
            sha256="a838b2faead8e6ae8f27cca6629fb1c660bfdedb6763a5c4e5f268f80e557496",
        ),
    ),
)


def test_model_bundle_verification_accepts_exact_manifest(tmp_path: Path) -> None:
    model = tmp_path / _TEST_BUNDLE.root / "model.bin"
    model.parent.mkdir()
    model.write_bytes(b"trusted-model")

    assert verify_model_bundle(tmp_path, _TEST_BUNDLE) == model.parent


def test_model_bundle_verification_rejects_sha256_mismatch(tmp_path: Path) -> None:
    model = tmp_path / _TEST_BUNDLE.root / "model.bin"
    model.parent.mkdir()
    model.write_bytes(b"tampered-mode")

    with pytest.raises(ModelIntegrityError, match="SHA-256 mismatch"):
        verify_model_bundle(tmp_path, _TEST_BUNDLE)


def test_model_bundle_verification_rejects_unexpected_file(tmp_path: Path) -> None:
    root = tmp_path / _TEST_BUNDLE.root
    root.mkdir()
    (root / "model.bin").write_bytes(b"trusted-model")
    (root / "injected.bin").write_bytes(b"unexpected")

    with pytest.raises(ModelIntegrityError, match=r"injected.bin: unexpected file"):
        verify_model_bundle(tmp_path, _TEST_BUNDLE)


def test_model_bundle_verification_rejects_symlinked_parent(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.bin").write_bytes(b"trusted-model")
    root = tmp_path / "nested-model"
    root.mkdir()
    (root / "nested").symlink_to(outside, target_is_directory=True)
    bundle = ModelBundle(
        name="nested test model",
        root=root.name,
        repo_id="example/nested-test-model",
        revision="0123456789abcdef0123456789abcdef01234567",
        artifacts=(
            ModelArtifact(
                path="nested/model.bin",
                size=13,
                sha256=_TEST_BUNDLE.artifacts[0].sha256,
            ),
        ),
    )

    with pytest.raises(ModelIntegrityError, match="symlinked path"):
        verify_model_bundle(tmp_path, bundle)


def test_docling_layout_revision_is_an_exact_commit() -> None:
    revision = DOCLING_LAYOUT_BUNDLE.revision
    native_config = _pinned_docling_layout_config()

    assert len(revision) == 40
    assert revision != "main"
    assert all(character in "0123456789abcdef" for character in revision)
    assert native_config.repo_id == DOCLING_LAYOUT_BUNDLE.repo_id
    assert native_config.revision == revision
