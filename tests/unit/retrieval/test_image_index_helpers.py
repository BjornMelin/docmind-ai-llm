"""Unit tests for image indexing helpers (SigLIP collection + upserts)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from loguru import logger

from src.persistence.artifacts import ArtifactRef
from src.retrieval.image_index import (
    PageImageRecord,
    collect_artifact_refs_for_doc_id,
    count_artifact_references_in_image_collection,
    ensure_siglip_image_collection,
    index_page_images_siglip,
)

pytestmark = pytest.mark.unit


@dataclass
class _Point:
    id: object
    payload: dict[str, object] | None = None


class _Client:
    def __init__(self) -> None:
        self.exists = False
        self.create_calls: list[dict[str, Any]] = []
        self.retrieve_calls = 0
        self.set_payload_calls: list[tuple[list[object], dict[str, object]]] = []
        self.upsert_calls: list[list[Any]] = []
        self.count_calls = 0
        self.scroll_calls = 0
        self._retrieve_points: list[_Point] = []
        self._scroll_pages: list[tuple[list[_Point], object | None]] = []
        self._count_value = 0

    def collection_exists(self, _name: str) -> bool:
        return bool(self.exists)

    def create_collection(self, **kwargs: Any) -> None:
        self.create_calls.append(kwargs)

    def retrieve(self, **_kwargs: Any) -> list[_Point]:
        self.retrieve_calls += 1
        return list(self._retrieve_points)

    def set_payload(
        self, *, points: list[object], payload: dict[str, object], **_k: Any
    ) -> None:
        self.set_payload_calls.append((points, payload))

    def upsert(self, *, points: list[Any], **_k: Any) -> None:
        self.upsert_calls.append(points)

    def count(self, **_kwargs: Any) -> object:
        self.count_calls += 1
        return SimpleNamespace(count=self._count_value)

    def scroll(self, **_kwargs: Any) -> tuple[list[_Point], object | None]:
        self.scroll_calls += 1
        if not self._scroll_pages:
            return ([], None)
        return self._scroll_pages.pop(0)


def test_ensure_siglip_image_collection_creates_when_missing() -> None:
    client = _Client()
    client.exists = False
    ensure_siglip_image_collection(client, "img")  # type: ignore[arg-type]
    assert len(client.create_calls) == 1
    assert client.create_calls[0]["collection_name"] == "img"


def test_index_page_images_short_circuits_on_phash_match(
    monkeypatch, tmp_path: Path
) -> None:
    client = _Client()
    client.exists = True

    rec = PageImageRecord(
        doc_id="d1",
        page_no=1,
        image=ArtifactRef(sha256="i", suffix=".webp"),
        image_path=tmp_path / "p1.webp",
        phash="same",
    )
    pid = rec.point_id()
    client._retrieve_points = [_Point(id=pid, payload={"phash": "same"})]

    embed_calls: list[int] = []

    class _Embed:
        def get_image_embeddings(self, _imgs: list[object], batch_size: int = 1):  # type: ignore[no-untyped-def]
            embed_calls.append(batch_size)
            return np.zeros((batch_size, 768), dtype=np.float32)

    monkeypatch.setattr(
        "src.retrieval.image_index._load_rgb_image", lambda _p: object()
    )
    indexed = index_page_images_siglip(
        client,  # type: ignore[arg-type]
        "img",
        [rec],
        embedder=_Embed(),
        batch_size=4,
    )
    assert indexed == 0
    assert embed_calls == []
    assert client.set_payload_calls
    assert client.upsert_calls == []


def test_index_page_images_upserts_embeddings_when_needed(
    monkeypatch, tmp_path: Path
) -> None:
    client = _Client()
    client.exists = True

    rec = PageImageRecord(
        doc_id="d1",
        page_no=1,
        image=ArtifactRef(sha256="i", suffix=".webp"),
        image_path=tmp_path / "p1.webp",
        phash="new",
    )
    client._retrieve_points = []

    class _Embed:
        def get_image_embeddings(self, imgs: list[object], batch_size: int = 1):  # type: ignore[no-untyped-def]
            assert len(imgs) == batch_size
            return np.ones((batch_size, 768), dtype=np.float32)

    monkeypatch.setattr(
        "src.retrieval.image_index._load_rgb_image", lambda _p: object()
    )
    indexed = index_page_images_siglip(
        client,  # type: ignore[arg-type]
        "img",
        [rec],
        embedder=_Embed(),
        batch_size=8,
    )
    assert indexed == 1
    assert client.upsert_calls
    assert len(client.upsert_calls[0]) == 1


def test_index_page_images_handles_embedding_count_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    client = _Client()
    client.exists = True

    rec1 = PageImageRecord(
        doc_id="d1",
        page_no=1,
        image=ArtifactRef(sha256="i1", suffix=".webp"),
        image_path=tmp_path / "p1.webp",
        phash="new",
    )
    rec2 = PageImageRecord(
        doc_id="d1",
        page_no=2,
        image=ArtifactRef(sha256="i2", suffix=".webp"),
        image_path=tmp_path / "p2.webp",
        phash="new",
    )
    client._retrieve_points = []

    class _Embed:
        def get_image_embeddings(self, imgs: list[object], batch_size: int = 1):  # type: ignore[no-untyped-def]
            assert len(imgs) == batch_size
            return np.ones((batch_size - 1, 768), dtype=np.float32)

    monkeypatch.setattr(
        "src.retrieval.image_index._load_rgb_image", lambda _p: object()
    )

    indexed = index_page_images_siglip(
        client,  # type: ignore[arg-type]
        "img",
        [rec1, rec2],
        embedder=_Embed(),
        batch_size=2,
    )

    assert indexed == 1
    assert client.upsert_calls
    assert len(client.upsert_calls[0]) == 1

    messages: list[str] = []

    def _sink(msg):
        messages.append(str(msg))

    sink_id = logger.add(_sink, level="WARNING")
    try:
        # Trigger another mismatch to capture the warning through loguru.
        index_page_images_siglip(
            client,  # type: ignore[arg-type]
            "img",
            [rec1, rec2],
            embedder=_Embed(),
            batch_size=2,
        )
    finally:
        logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Embedding count mismatch" in joined
    assert "len(batch)=2" in joined
    assert "len(vecs)=1" in joined
    assert "p2.webp" in joined


def test_collect_artifact_refs_for_doc_id_extracts_image_and_thumbnail() -> None:
    client = _Client()
    client._scroll_pages = [
        (
            [
                _Point(
                    id="x",
                    payload={
                        "image_artifact_id": "a",
                        "image_artifact_suffix": ".webp",
                        "thumbnail_artifact_id": "b",
                        "thumbnail_artifact_suffix": ".webp",
                    },
                )
            ],
            None,
        )
    ]
    refs = collect_artifact_refs_for_doc_id(client, "img", doc_id="d1")  # type: ignore[arg-type]
    assert {r.sha256 for r in refs} == {"a", "b"}


def test_count_artifact_references_in_image_collection_returns_count() -> None:
    client = _Client()
    client._count_value = 5
    n = count_artifact_references_in_image_collection(client, "img", artifact_id="x")  # type: ignore[arg-type]
    assert n == 5
