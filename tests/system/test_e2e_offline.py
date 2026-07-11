"""Offline end-to-end smoke checks when Qdrant is available locally."""

import hashlib
import os
import socket
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from src.config.embedding_defaults import BGE_M3_EMBEDDING_DIMENSION
from src.config.settings import DocMindSettings
from src.processing.ingestion_api import load_documents
from src.processing.parsing.service import parse_document_sync
from src.utils.storage import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_hybrid_collection,
)

_QDRANT_SYSTEM_URL = os.getenv("DOCMIND_QDRANT_SYSTEM_URL")


@pytest.mark.system
@pytest.mark.requires_network
@pytest.mark.asyncio
@pytest.mark.skipif(
    not _QDRANT_SYSTEM_URL,
    reason="set DOCMIND_QDRANT_SYSTEM_URL to run the Qdrant system smoke",
)
async def test_text_ingest_index_query_qdrant_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Parse text, create the production schema, index it, and query Qdrant."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    source = tmp_path / "qdrant-roundtrip.md"
    source.write_text("DocMind Qdrant roundtrip sentinel", encoding="utf-8")
    documents = await load_documents([source])
    assert len(documents) == 1

    document = documents[0]
    content = document.get_content()
    dense = [
        byte / 255.0
        for byte in hashlib.shake_256(content.encode()).digest(
            BGE_M3_EMBEDDING_DIMENSION
        )
    ]
    collection = f"docmind-system-{uuid.uuid4().hex}"
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, document.doc_id))
    client = QdrantClient(url=_QDRANT_SYSTEM_URL, timeout=10, prefer_grpc=False)
    try:
        compatibility = ensure_hybrid_collection(
            client,
            collection,
            dense_dim=BGE_M3_EMBEDDING_DIMENSION,
        )
        assert compatibility.compatible is True
        assert compatibility.action == "created"
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: dense,
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=[0, 1],
                            values=[1.0, 0.5],
                        ),
                    },
                    payload={
                        "doc_id": document.doc_id,
                        "source": source.name,
                        "text": content,
                    },
                )
            ],
            wait=True,
        )

        result = client.query_points(
            collection_name=collection,
            query=dense,
            using=DENSE_VECTOR_NAME,
            with_payload=True,
            limit=1,
        )

        assert [point.id for point in result.points] == [point_id]
        assert result.points[0].payload == {
            "doc_id": document.doc_id,
            "source": source.name,
            "text": content,
        }
    finally:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.close()


@pytest.mark.system
def test_local_text_parse_uses_no_network(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network egress attempted")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    text_file = tmp_path / "sample.md"
    text_file.write_text("offline parser smoke", encoding="utf-8")
    settings = DocMindSettings(_env_file=None)  # type: ignore[arg-type]

    result = parse_document_sync(text_file, settings=settings)

    assert result.pages[0].text_markdown == "offline parser smoke"
