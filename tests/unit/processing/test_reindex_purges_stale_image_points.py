"""Unit tests for lossless page-image replacement semantics."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.models.processing import ExportArtifact, IngestionConfig
from src.processing import ingestion_pipeline as ingestion

pytestmark = pytest.mark.unit

_PAGE_IMAGE_NAMESPACE = uuid.UUID("d3b17330-1e80-4c4f-9f5d-9f2a1432f6cf")


@pytest.fixture
def image_reindex_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[ExportArtifact], IngestionConfig]:
    """Build one deterministic page-image replacement request."""
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
            artifacts=SimpleNamespace(dir=data_dir / "artifacts"),
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
    return exports, IngestionConfig(cache_dir=cache_dir)


def test_reindex_upserts_and_verifies_before_deleting_only_stale_points(
    image_reindex_case: tuple[list[ExportArtifact], IngestionConfig],
) -> None:
    exports, cfg = image_reindex_case
    expected_id = str(uuid.uuid5(_PAGE_IMAGE_NAMESPACE, "doc-1::page::1"))
    fake_client = MagicMock()
    with (
        patch("qdrant_client.QdrantClient", return_value=fake_client),
        patch(
            "src.retrieval.image_index.collect_page_image_point_ids_for_doc_id",
            side_effect=[{"stale", expected_id}, {"stale", expected_id}],
        ) as collect_ids,
        patch(
            "src.retrieval.image_index.index_page_images_siglip", return_value=1
        ) as index_images,
        patch(
            "src.retrieval.image_index.delete_page_image_points_by_id",
            return_value=1,
        ) as delete_stale,
    ):
        meta = ingestion._index_page_images(  # type: ignore[attr-defined]
            exports, cfg, purge_doc_ids={"doc-1"}
        )

    assert collect_ids.call_count == 2
    assert index_images.call_count == 1
    delete_stale.assert_called_once_with(
        fake_client,
        "docmind_images",
        point_ids={"stale"},
    )
    assert meta["image_index.indexed"] == 1
    assert meta["image_index.purged_points"] == 1


@pytest.mark.parametrize("failure_mode", ["embed", "partial", "verify"])
def test_reindex_never_deletes_old_points_until_replacement_is_complete(
    image_reindex_case: tuple[list[ExportArtifact], IngestionConfig],
    failure_mode: str,
) -> None:
    exports, cfg = image_reindex_case
    expected_id = str(uuid.uuid5(_PAGE_IMAGE_NAMESPACE, "doc-1::page::1"))
    collect_results: list[set[str]] = [{"old"}]
    index_result: int | Exception = 1
    if failure_mode == "embed":
        index_result = RuntimeError("embedding failed")
    elif failure_mode == "partial":
        index_result = 0
    elif failure_mode == "verify":
        collect_results.append(set())

    def _index(*_args: object, **_kwargs: object) -> int:
        if isinstance(index_result, Exception):
            raise index_result
        return index_result

    fake_client = MagicMock()
    with (
        patch("qdrant_client.QdrantClient", return_value=fake_client),
        patch(
            "src.retrieval.image_index.collect_page_image_point_ids_for_doc_id",
            side_effect=collect_results,
        ),
        patch("src.retrieval.image_index.index_page_images_siglip", side_effect=_index),
        patch(
            "src.retrieval.image_index.delete_page_image_points_by_id"
        ) as delete_stale,
    ):
        meta = ingestion._index_page_images(  # type: ignore[attr-defined]
            exports, cfg, purge_doc_ids={"doc-1"}
        )

    delete_stale.assert_not_called()
    assert meta["image_index.indexed"] == 0
    assert "image_index.error_type" in meta
    if failure_mode == "verify":
        assert expected_id not in collect_results[-1]


def test_index_page_images_skips_replacement_scan_without_doc_ids(
    image_reindex_case: tuple[list[ExportArtifact], IngestionConfig],
) -> None:
    exports, cfg = image_reindex_case
    fake_client = MagicMock()
    with (
        patch("qdrant_client.QdrantClient", return_value=fake_client),
        patch(
            "src.retrieval.image_index.collect_page_image_point_ids_for_doc_id"
        ) as collect_ids,
        patch("src.retrieval.image_index.index_page_images_siglip", return_value=1),
        patch(
            "src.retrieval.image_index.delete_page_image_points_by_id"
        ) as delete_stale,
    ):
        meta = ingestion._index_page_images(  # type: ignore[attr-defined]
            exports, cfg, purge_doc_ids=None
        )

    collect_ids.assert_not_called()
    delete_stale.assert_not_called()
    assert meta["image_index.purged_points"] == 0
