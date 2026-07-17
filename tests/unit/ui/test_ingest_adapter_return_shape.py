"""Unit tests for immutable-generation ingestion adapter contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from llama_index.core.schema import TextNode

from src.models.processing import (
    CANONICAL_DOCUMENT_ID_KEY,
    IngestionInput,
    IngestionResult,
    ManifestSummary,
)
from src.persistence.hashing import compute_corpus_hash
from src.retrieval import vector_contract
from src.ui import ingest_adapter as adapter


def _noop(*_args: Any, **_kwargs: Any) -> None:
    """No-op stub for monkeypatching."""


def _result_for(document_ids: list[str]) -> IngestionResult:
    nodes = [
        TextNode(
            text=f"content for {document_id}",
            metadata={CANONICAL_DOCUMENT_ID_KEY: document_id},
        )
        for document_id in document_ids
    ]
    documents = [
        SimpleNamespace(metadata={CANONICAL_DOCUMENT_ID_KEY: document_id})
        for document_id in document_ids
    ]
    return IngestionResult(
        nodes=nodes,
        documents=documents,
        manifest=ManifestSummary(
            corpus_hash="complete-corpus",
            config_hash="complete-config",
            payload_count=len(nodes),
        ),
        exports=[],
        metadata={"image_index.indexed": 0},
        duration_ms=1.0,
    )


def _patch_lightweight_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Any,
) -> None:
    monkeypatch.setattr(adapter.settings, "data_dir", tmp_path)
    monkeypatch.setattr(adapter, "setup_llamaindex", _noop)
    monkeypatch.setattr(adapter, "configure_observability", _noop)
    monkeypatch.setattr(adapter, "get_settings_embed_model", lambda: object())
    monkeypatch.setattr(adapter, "embedding_allowed_for_ingestion", lambda _: True)
    monkeypatch.setattr(adapter, "_ensure_staged_image_collection", _noop)
    monkeypatch.setattr(adapter, "_verify_image_collection_count", _noop)


def test_ingest_inputs_rejects_duplicate_ids_before_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        adapter,
        "setup_llamaindex",
        lambda **_kwargs: calls.append("embedding_setup"),
    )
    inputs = [
        IngestionInput(document_id="doc-duplicate", payload_text="alpha"),
        IngestionInput(document_id="doc-duplicate", payload_text="beta"),
    ]

    with pytest.raises(ValueError, match=r"Duplicate document_id.*doc-duplicate"):
        adapter.ingest_inputs(
            inputs,
            text_collection_name="text__build",
            image_collection_name="image__build",
        )

    assert calls == []


@pytest.mark.parametrize("invalid_kind", ["missing", "symlink"])
def test_skipped_source_fails_closed_before_collection_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    invalid_kind: str,
) -> None:
    _patch_lightweight_runtime(monkeypatch, tmp_path=tmp_path)
    valid = IngestionInput(document_id="doc-valid", payload_text="valid content")
    invalid_path = tmp_path / "missing.txt"
    if invalid_kind == "symlink":
        target = tmp_path / "target.txt"
        target.write_text("target", encoding="utf-8")
        try:
            invalid_path.symlink_to(target)
        except OSError:
            pytest.skip("symlinks unavailable")
    invalid = IngestionInput(document_id="doc-invalid", source_path=invalid_path)
    monkeypatch.setattr(
        adapter,
        "ingest_documents_sync",
        lambda *_args, **_kwargs: _result_for(["doc-valid"]),
    )

    with pytest.raises(ValueError, match="Ingestion source"):
        adapter.ingest_inputs(
            [invalid, valid],
            text_collection_name="text__build",
            image_collection_name="image__build",
        )


def test_ingest_rebuilds_complete_corpus_into_supplied_collections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    first = uploads / "first.txt"
    second = uploads / "second.txt"
    first.write_text("first corpus document", encoding="utf-8")
    second.write_text("second corpus document", encoding="utf-8")
    first_id = adapter.document_id_from_sha256(adapter.sha256_file(first))
    second_digest = adapter.sha256_file(second)
    second_id = adapter.document_id_from_sha256(second_digest)
    current = IngestionInput(
        document_id=second_id,
        source_path=second,
        metadata={"sha256": second_digest},
    )
    _patch_lightweight_runtime(monkeypatch, tmp_path=tmp_path)
    calls: list[list[str]] = []

    def _ingest(_cfg: Any, inputs: list[IngestionInput], **_kwargs: Any) -> Any:
        ids = [str(item.document_id) for item in inputs]
        calls.append(ids)
        return _result_for(ids)

    indexed: list[tuple[set[str], str]] = []
    resource = SimpleNamespace(index="vector-index", close=lambda: None)
    monkeypatch.setattr(adapter, "ingest_documents_sync", _ingest)
    monkeypatch.setattr(
        adapter,
        "_build_vector_index",
        lambda _nodes, *, document_ids, collection_name: (
            indexed.append((set(document_ids), collection_name)) or resource
        ),
    )
    graph_ids: list[set[str]] = []
    monkeypatch.setattr(
        adapter,
        "PropertyGraphIndex",
        SimpleNamespace(
            from_documents=lambda documents, show_progress=False: (
                graph_ids.append(
                    {
                        str(document.metadata[CANONICAL_DOCUMENT_ID_KEY])
                        for document in documents
                    }
                )
                or SimpleNamespace(property_graph_store=object())
            )
        ),
    )

    result = adapter.ingest_inputs(
        [current],
        text_collection_name="text__build",
        image_collection_name="image__build",
        enable_graphrag=True,
    )

    assert result["vector_index"] == "vector-index"
    assert calls == [[second_id, first_id]]
    assert indexed == [({first_id, second_id}, "text__build")]
    assert graph_ids == [{first_id, second_id}]
    assert result["collections"] == {
        "text": "text__build",
        "image": "image__build",
    }
    assert result["activation_corpus_hash"] == compute_corpus_hash(
        [first, second], base_dir=uploads
    )


@pytest.mark.parametrize(
    ("embedding", "allowed", "message"),
    [
        (None, True, "unavailable"),
        (object(), False, "blocked by endpoint policy"),
    ],
)
def test_ingest_fails_closed_without_canonical_embedding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    embedding: object | None,
    allowed: bool,
    message: str,
) -> None:
    monkeypatch.setattr(adapter.settings, "data_dir", tmp_path)
    monkeypatch.setattr(adapter, "setup_llamaindex", _noop)
    monkeypatch.setattr(adapter, "get_settings_embed_model", lambda: embedding)
    monkeypatch.setattr(adapter, "embedding_allowed_for_ingestion", lambda _: allowed)

    with pytest.raises(RuntimeError, match=message):
        adapter.ingest_inputs(
            [IngestionInput(document_id="doc", payload_text="payload")],
            text_collection_name="text__build",
            image_collection_name="image__build",
        )


def test_build_vector_index_uses_fresh_physical_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = TextNode(
        text="hello",
        metadata={CANONICAL_DOCUMENT_ID_KEY: "doc-1", "page_id": "page-1"},
    )
    store = SimpleNamespace(collection_name="text__build", client=object())
    created: list[str] = []
    verified: list[tuple[set[str], bool]] = []
    monkeypatch.setattr(
        adapter,
        "create_vector_store",
        lambda collection_name: created.append(collection_name) or store,
    )
    monkeypatch.setattr(
        adapter,
        "StorageContext",
        SimpleNamespace(from_defaults=lambda **_kwargs: SimpleNamespace()),
    )
    monkeypatch.setattr(
        adapter,
        "VectorStoreIndex",
        lambda _nodes, **_kwargs: "vector-index",
    )
    monkeypatch.setattr(vector_contract, "sparse_retrieval_enabled", lambda: True)
    monkeypatch.setattr(
        adapter,
        "_verify_text_collection",
        lambda _store, *, nodes, document_ids, sparse_enabled: verified.append(
            (set(document_ids), sparse_enabled)
        ),
    )

    resource = adapter._build_vector_index(
        [node],
        document_ids={"doc-1"},
        collection_name="text__build",
    )

    assert resource.index == "vector-index"
    assert created == ["text__build"]
    assert verified == [({"doc-1"}, True)]


def test_build_vector_index_allows_verified_empty_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SimpleNamespace(collection_name="text__empty", client=object())
    index_api = SimpleNamespace(
        from_vector_store=lambda vector_store: (
            "empty-vector-index" if vector_store is store else None
        )
    )
    monkeypatch.setattr(adapter, "create_vector_store", lambda _name: store)
    monkeypatch.setattr(
        adapter,
        "StorageContext",
        SimpleNamespace(from_defaults=lambda **_kwargs: SimpleNamespace()),
    )
    monkeypatch.setattr(adapter, "VectorStoreIndex", index_api)
    monkeypatch.setattr(vector_contract, "sparse_retrieval_enabled", lambda: False)
    monkeypatch.setattr(adapter, "_verify_text_collection", _noop)

    resource = adapter._build_vector_index(
        [],
        document_ids=set(),
        collection_name="text__empty",
    )

    assert resource.index == "empty-vector-index"


def test_build_vector_index_rejects_inconsistent_empty_state() -> None:
    with pytest.raises(ValueError, match="emptiness must agree"):
        adapter._build_vector_index(
            [],
            document_ids={"doc-1"},
            collection_name="text__build",
        )
