"""Unit tests for reindex page-image lifecycle semantics (final-release)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.models.processing import ExportArtifact, IngestionConfig
from src.processing import ingestion_pipeline as ingestion

pytestmark = pytest.mark.unit


def test_index_page_images_purges_existing_doc_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    # Patch settings used by ingestion pipeline so artifacts land under tmp_path.
    monkeypatch.setattr(
        ingestion,
        "app_settings",
        SimpleNamespace(
            data_dir=data_dir,
            cache_dir=cache_dir,
            processing=SimpleNamespace(
                thumbnail_max_side=64, encrypt_page_images=False
            ),
            artifacts=SimpleNamespace(
                dir=data_dir / "artifacts",
                max_total_mb=0,
                gc_min_age_seconds=0,
            ),
            database=SimpleNamespace(qdrant_image_collection="docmind_images"),
        ),
        raising=False,
    )

    img_path = tmp_path / "page-1.jpg"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(img_path, format="JPEG")

    exports = [
        ExportArtifact(
            name="pdf-page-1",
            path=img_path,
            content_type="image/jpeg",
            metadata={"doc_id": "doc-1", "page_no": 1, "source_filename": "x.pdf"},
        )
    ]
    cfg = IngestionConfig(cache_dir=cache_dir)

    fake_client = MagicMock()
    with (
        patch("qdrant_client.QdrantClient", return_value=fake_client),
        patch(
            "src.retrieval.image_index.delete_page_images_for_doc_id", return_value=7
        ) as mock_delete,
        patch("src.retrieval.image_index.index_page_images_siglip", return_value=1),
    ):
        meta = ingestion._index_page_images(  # type: ignore[attr-defined]
            exports,
            cfg,
            purge_doc_ids={"doc-1"},
        )

    mock_delete.assert_called_with(
        fake_client,
        "docmind_images",
        doc_id="doc-1",
    )
    assert int(meta.get("image_index.purged_points", 0)) == 7


def test_index_page_images_skips_purge_when_no_doc_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        ingestion,
        "app_settings",
        SimpleNamespace(
            data_dir=data_dir,
            cache_dir=cache_dir,
            processing=SimpleNamespace(
                thumbnail_max_side=64, encrypt_page_images=False
            ),
            artifacts=SimpleNamespace(
                dir=data_dir / "artifacts",
                max_total_mb=0,
                gc_min_age_seconds=0,
            ),
            database=SimpleNamespace(qdrant_image_collection="docmind_images"),
        ),
        raising=False,
    )

    img_path = tmp_path / "page-1.jpg"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(img_path, format="JPEG")

    exports = [
        ExportArtifact(
            name="pdf-page-1",
            path=img_path,
            content_type="image/jpeg",
            metadata={"doc_id": "doc-1", "page_no": 1, "source_filename": "x.pdf"},
        )
    ]
    cfg = IngestionConfig(cache_dir=cache_dir)

    fake_client = MagicMock()
    with (
        patch("qdrant_client.QdrantClient", return_value=fake_client),
        patch(
            "src.retrieval.image_index.delete_page_images_for_doc_id", return_value=7
        ) as mock_delete,
        patch("src.retrieval.image_index.index_page_images_siglip", return_value=1),
    ):
        meta = ingestion._index_page_images(  # type: ignore[attr-defined]
            exports,
            cfg,
            purge_doc_ids=None,
        )

    mock_delete.assert_not_called()
    assert int(meta.get("image_index.purged_points", 0)) == 0
