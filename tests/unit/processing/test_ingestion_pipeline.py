"""Unit tests for the LlamaIndex ingestion pipeline wrapper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding

from src.models.processing import IngestionConfig, IngestionInput, ParsingOverrides
from src.processing.ingestion_api import load_documents_from_inputs
from src.processing.ingestion_pipeline import (
    _collect_parsing_provenance,
    _ingestion_corpus_hash,
    _manifest_parsing_config,
    _resolve_embedding,
    build_ingestion_pipeline,
    embedding_allowed_for_ingestion,
    ingest_documents,
    ingest_documents_sync,
)


def test_manifest_parsing_config_tracks_per_input_overrides() -> None:
    baseline = [IngestionInput(document_id="doc-1", payload_text="same payload")]
    forced_ocr = [
        IngestionInput(
            document_id="doc-1",
            payload_text="same payload",
            parsing_overrides=ParsingOverrides(force_ocr=True),
        )
    ]

    assert _manifest_parsing_config(baseline) != _manifest_parsing_config(forced_ocr)
    assert _manifest_parsing_config(forced_ocr)["input_overrides"] == [
        {"document_id": "doc-1", "overrides": {"force_ocr": True}}
    ]


def test_ingestion_corpus_hash_tracks_ordered_payload_content() -> None:
    baseline = [
        IngestionInput(document_id="doc-b", payload_text="beta"),
        IngestionInput(document_id="doc-a", payload_text="alpha"),
    ]
    reordered = list(reversed(baseline))
    changed = [
        IngestionInput(document_id="doc-a", payload_text="beta"),
        IngestionInput(document_id="doc-b", payload_text="alpha"),
    ]

    assert _ingestion_corpus_hash(baseline) == _ingestion_corpus_hash(reordered)
    assert _ingestion_corpus_hash(baseline) != _ingestion_corpus_hash(changed)


@pytest.mark.asyncio
async def test_ingest_documents_rejects_duplicate_ids_before_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.processing import ingestion_pipeline as module

    def _unexpected_io(*_args: object, **_kwargs: object) -> None:
        pytest.fail("embedding resolution ran before duplicate ID validation")

    monkeypatch.setattr(module, "_resolve_embedding", _unexpected_io)
    inputs = [
        IngestionInput(document_id="doc-duplicate", payload_text="alpha"),
        IngestionInput(document_id="doc-duplicate", payload_text="beta"),
    ]

    with pytest.raises(ValueError, match=r"Duplicate document_id.*doc-duplicate"):
        await module.ingest_documents(IngestionConfig(), inputs)


@pytest.mark.asyncio
async def test_ingest_documents_rejects_zero_node_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.processing import ingestion_pipeline as module

    class _EmptyPipeline:
        async def arun(self, *, documents: list[Document]) -> list[object]:
            assert documents
            return []

    async def _load(
        _cfg: IngestionConfig,
        _inputs: list[IngestionInput],
    ) -> tuple[list[Document], list[object]]:
        return [Document(text="parsed", doc_id="doc-1")], []

    monkeypatch.setattr(module, "_resolve_embedding", lambda _embedding: None)
    monkeypatch.setattr(
        module,
        "build_ingestion_pipeline",
        lambda *_args, **_kwargs: (
            _EmptyPipeline(),
            tmp_path / "cache.duckdb",
        ),
    )
    monkeypatch.setattr(module, "_load_documents", _load)
    monkeypatch.setattr(
        module,
        "_index_page_images",
        lambda *_args, **_kwargs: pytest.fail(
            "image indexing ran after an empty replacement"
        ),
    )

    inputs = [IngestionInput(document_id="doc-1", payload_text="parsed")]
    with pytest.raises(RuntimeError, match="produced no nodes"):
        await module.ingest_documents(
            IngestionConfig(cache_dir=tmp_path),
            inputs,
        )


class DummyEmbedding(BaseEmbedding):
    """Deterministic embedding for tests."""

    def _get_text_embedding(self, text: str):  # type: ignore[override]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        return [float(value)]

    async def _aget_text_embedding(self, text: str):  # type: ignore[override]
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str):  # type: ignore[override]
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str):  # type: ignore[override]
        return self._get_text_embedding(query)

    def _get_text_embeddings(self, texts):  # type: ignore[override]
        return [self._get_text_embedding(t) for t in texts]

    async def _aget_text_embeddings(self, texts):  # type: ignore[override]
        return [self._get_text_embedding(t) for t in texts]


class EndpointEmbedding(DummyEmbedding):
    """Embedding stub exposing an OpenAI-compatible endpoint URL."""

    def __init__(self, endpoint: str) -> None:
        """Initialize the embedding stub with an endpoint URL."""
        super().__init__()
        object.__setattr__(self, "api_base", endpoint)


@pytest.mark.asyncio
async def test_exact_retry_replays_cached_nodes_without_zero_node_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unchanged retry returns the complete cached replacement node set."""
    from src.processing import ingestion_pipeline as module

    monkeypatch.setattr(module.app_settings.spacy, "enabled", False)
    cfg = IngestionConfig(
        chunk_size=128,
        chunk_overlap=0,
        cache_dir=tmp_path / "cache",
        enable_image_indexing=False,
    )
    inputs = [
        IngestionInput(
            document_id="doc-unchanged",
            payload_text="unchanged retry content " * 40,
        )
    ]
    embedding_calls: list[str] = []

    class _CountingEmbedding(DummyEmbedding):
        def _get_text_embedding(self, text: str):  # type: ignore[override]
            embedding_calls.append(text)
            return super()._get_text_embedding(text)

    embedding = _CountingEmbedding()

    first = await ingest_documents(cfg, inputs, embedding=embedding)
    first_call_count = len(embedding_calls)
    retried = await ingest_documents(cfg, inputs, embedding=embedding)

    assert first.nodes
    assert first_call_count > 0
    assert len(embedding_calls) == first_call_count
    assert len(retried.nodes) == len(first.nodes)
    assert retried.manifest.payload_count == len(retried.nodes)
    assert {node.metadata["document_id"] for node in retried.nodes} == {"doc-unchanged"}


@pytest.mark.asyncio
async def test_mixed_retry_builds_complete_immutable_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A mixed cached/new batch fully verifies one new physical collection."""
    from src.processing import ingestion_pipeline as module
    from src.ui import ingest_adapter as adapter

    monkeypatch.setattr(module.app_settings.spacy, "enabled", False)
    cfg = IngestionConfig(
        chunk_size=128,
        chunk_overlap=0,
        cache_dir=tmp_path / "cache",
        enable_image_indexing=False,
    )
    unchanged = IngestionInput(
        document_id="doc-unchanged",
        payload_text="unchanged mixed-batch content " * 40,
    )
    new = IngestionInput(
        document_id="doc-new",
        payload_text="new mixed-batch content " * 40,
    )
    embedding = DummyEmbedding()

    first = await ingest_documents(cfg, [unchanged], embedding=embedding)
    assert first.nodes

    mixed = await ingest_documents(cfg, [unchanged, new], embedding=embedding)
    assert {node.metadata["document_id"] for node in mixed.nodes} == {
        "doc-unchanged",
        "doc-new",
    }

    class _Client:
        def __init__(self) -> None:
            self.closed = False
            self.count_calls: list[tuple[str, bool]] = []
            self.retrieve_calls: list[list[str]] = []

        def count(self, *, collection_name: str, exact: bool) -> SimpleNamespace:
            self.count_calls.append((collection_name, exact))
            return SimpleNamespace(count=len(mixed.nodes))

        def retrieve(
            self,
            *,
            collection_name: str,
            ids: list[str],
            with_payload: list[str],
            with_vectors: bool,
        ) -> list[SimpleNamespace]:
            assert collection_name == "physical-text-build-2"
            assert with_payload == [adapter.CANONICAL_DOCUMENT_ID_KEY]
            assert with_vectors is True
            self.retrieve_calls.append(ids)
            nodes_by_id = {str(node.node_id): node for node in mixed.nodes}
            return [
                SimpleNamespace(
                    id=node_id,
                    payload={
                        adapter.CANONICAL_DOCUMENT_ID_KEY: nodes_by_id[
                            node_id
                        ].metadata[adapter.CANONICAL_DOCUMENT_ID_KEY]
                    },
                    vector={adapter.DENSE_VECTOR_NAME: [1.0]},
                )
                for node_id in ids
            ]

        def close(self) -> None:
            self.closed = True

    class _Store:
        def __init__(self) -> None:
            self.collection_name = "physical-text-build-2"
            self.client = _Client()

    store = _Store()
    collection_calls: list[str] = []

    def _create_vector_store(collection_name: str):  # type: ignore[no-untyped-def]
        collection_calls.append(collection_name)
        return store

    monkeypatch.setattr(adapter, "create_vector_store", _create_vector_store)
    monkeypatch.setattr(adapter, "sparse_retrieval_enabled", lambda: False)
    monkeypatch.setattr(
        adapter,
        "StorageContext",
        SimpleNamespace(from_defaults=lambda **_kwargs: SimpleNamespace()),
    )
    monkeypatch.setattr(
        adapter,
        "VectorStoreIndex",
        lambda *_args, **_kwargs: "vector-index",
    )

    result = adapter._build_vector_index(
        mixed.nodes,
        document_ids={"doc-unchanged", "doc-new"},
        collection_name="physical-text-build-2",
    )

    assert isinstance(result, adapter.VectorIndexResource)
    assert result.index == "vector-index"
    assert collection_calls == ["physical-text-build-2"]
    assert store.client.count_calls == [("physical-text-build-2", True)]
    assert set().union(*map(set, store.client.retrieve_calls)) == {
        str(node.node_id) for node in mixed.nodes
    }
    result.close()
    assert store.client.closed is True


@pytest.mark.asyncio
async def test_ingest_documents_with_bytes_payload(tmp_path: Path) -> None:
    cfg = IngestionConfig(
        chunk_size=64,
        chunk_overlap=16,
        cache_dir=tmp_path / "cache",
    )
    inputs = [
        IngestionInput(document_id="doc-1", payload_text="DocMind ingestion test.")
    ]

    result = await ingest_documents(cfg, inputs, embedding=DummyEmbedding())

    assert result.duration_ms >= 0
    assert result.manifest.payload_count == len(result.nodes)
    assert result.metadata["document_count"] == 1


@pytest.mark.asyncio
async def test_ingest_documents_with_path(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("Library first ingestion pipeline")

    cfg = IngestionConfig(
        chunk_size=128,
        chunk_overlap=32,
        cache_dir=tmp_path / "cache",
    )
    inputs = [
        IngestionInput(
            document_id="doc-file",
            source_path=source,
            metadata={"source_tag": "unit"},
        )
    ]

    result = await ingest_documents(cfg, inputs, embedding=DummyEmbedding())

    assert result.manifest.corpus_hash
    assert result.metadata["cache_db"] == "docmind.duckdb"
    assert result.documents
    for doc in result.documents:
        meta = getattr(doc, "metadata", {}) or {}
        assert "source_path" not in meta


def test_ingest_documents_sync_wrapper(tmp_path: Path) -> None:
    cfg = IngestionConfig(cache_dir=tmp_path / "cache")
    inputs = [IngestionInput(document_id="doc", payload_text="inline text")]

    result = ingest_documents_sync(cfg, inputs, embedding=DummyEmbedding())
    assert not result.exports


def test_collect_parsing_provenance_keeps_searchable_artifacts() -> None:
    doc = Document(
        text="parsed",
        doc_id="doc-1",
        metadata={
            "document_id": "doc-1",
            "parsing": {
                "framework": "docling",
                "profile": "cpu_safe",
                "config_hash": "a" * 64,
                "health": {"rapidocr": {"dependencies_ready": True}},
                "ocr_applied_pages": [1],
                "searchable_pdf_artifacts": [
                    {"kind": "searchable_pdf", "artifact_id": "abc", "suffix": ".pdf"}
                ],
            },
        },
    )

    provenance = _collect_parsing_provenance([doc])
    by_doc = provenance["parsing.provenance"]["doc-1"]

    assert by_doc["config_hash"] == "a" * 64
    assert by_doc["health"]["rapidocr"]["dependencies_ready"] is True
    assert by_doc["searchable_pdf_artifacts"][0]["artifact_id"] == "abc"
    assert provenance["parsing.profile"] == "cpu_safe"


@pytest.mark.asyncio
async def test_ingest_documents_sync_guard() -> None:
    cfg = IngestionConfig()
    inputs = [IngestionInput(document_id="doc", payload_text="inline text")]

    with pytest.raises(RuntimeError) as exc_info:
        ingest_documents_sync(cfg, inputs, embedding=DummyEmbedding())
    assert "await ingest_documents" in str(exc_info.value)


def test_build_ingestion_pipeline_uses_cache_without_docstore(tmp_path: Path) -> None:
    """The transformation cache does not pre-filter replacement inputs."""
    cfg = IngestionConfig(
        chunk_size=64,
        chunk_overlap=16,
        cache_dir=tmp_path / "cache",
    )

    pipeline, cache_path = build_ingestion_pipeline(cfg, embedding=DummyEmbedding())

    assert cfg.cache_dir is not None
    assert cache_path == cfg.cache_dir / "docmind.duckdb"
    assert pipeline.transformations  # TokenTextSplitter + optional components
    assert pipeline.docstore is None


def test_build_ingestion_pipeline_without_embedding(tmp_path: Path) -> None:
    """Pipeline construction succeeds when no embedding is configured."""
    cfg = IngestionConfig(
        chunk_size=64,
        chunk_overlap=16,
        cache_dir=tmp_path / "cache",
    )

    pipeline, _cache_path = build_ingestion_pipeline(cfg, embedding=None)

    # TokenTextSplitter is always present even without embeddings.
    assert pipeline.transformations
    assert all(component is not None for component in pipeline.transformations)
    assert not any(
        isinstance(component, DummyEmbedding) for component in pipeline.transformations
    )


def test_resolve_embedding_configures_llamaindex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_embedding calls setup_llamaindex when Settings lacks a model."""
    from src.processing import ingestion_pipeline as module

    dummy = DummyEmbedding()
    call_state = {"get": 0, "force": False}

    def fake_get_settings_embed_model() -> DummyEmbedding | None:  # pragma: no cover
        call_state["get"] += 1
        return dummy if call_state["get"] > 1 else None

    def fake_setup_llamaindex(*, force_embed: bool = False) -> None:  # pragma: no cover
        call_state["force"] = force_embed

    monkeypatch.setattr(
        module,
        "get_settings_embed_model",
        fake_get_settings_embed_model,
    )
    monkeypatch.setattr(module, "setup_llamaindex", fake_setup_llamaindex)

    resolved = _resolve_embedding(None)

    assert call_state == {"get": 2, "force": True}
    assert resolved is dummy


def test_embedding_allowed_for_ingestion_accepts_loopback() -> None:
    """Loopback embedding endpoints remain allowed in local-first mode."""
    embedding = EndpointEmbedding("http://127.0.0.1:18081/v1")

    assert embedding_allowed_for_ingestion(embedding) is True


def test_embedding_allowed_for_ingestion_blocks_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote embedding endpoints are blocked unless explicitly allowed."""
    from src.processing import ingestion_pipeline as module

    security = module.app_settings.security.model_copy(
        update={"allow_remote_endpoints": False, "endpoint_allowlist": []}
    )
    monkeypatch.setattr(module.app_settings, "security", security)
    embedding = EndpointEmbedding("https://api.openai.com/v1")

    assert embedding_allowed_for_ingestion(embedding) is False


def test_resolve_embedding_rejects_remote_global_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote global embedding objects do not leak into local ingestion."""
    from src.processing import ingestion_pipeline as module

    security = module.app_settings.security.model_copy(
        update={"allow_remote_endpoints": False, "endpoint_allowlist": []}
    )
    monkeypatch.setattr(module.app_settings, "security", security)
    monkeypatch.setattr(
        module,
        "get_settings_embed_model",
        lambda: EndpointEmbedding("https://api.openai.com/v1"),
    )
    monkeypatch.setattr(module, "setup_llamaindex", lambda **_: None)

    assert _resolve_embedding(None) is None


@pytest.mark.asyncio
async def test_ingest_documents_without_embedding_warns_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ingest_documents proceeds when embeddings remain unavailable."""
    from src.processing import ingestion_pipeline as module

    call_state: dict[str, Any] = {}

    async def _fake_arun(documents):  # type: ignore[no-untyped-def]
        return [
            {"doc_id": doc.doc_id, "text": getattr(doc, "text", "")}
            for doc in documents
        ]

    class _Pipeline:  # pragma: no cover - simple stub
        def __init__(self) -> None:
            self.transformations: list[Any] = []

        async def arun(self, documents):  # type: ignore[no-untyped-def]
            return await _fake_arun(documents)

    def _fake_build(cfg, embedding, *, nlp_service=None):  # type: ignore[no-untyped-def]
        call_state["embedding"] = embedding
        pipeline = _Pipeline()
        return pipeline, tmp_path / "cache.duckdb"

    monkeypatch.setattr(module, "build_ingestion_pipeline", _fake_build)
    monkeypatch.setattr(module, "get_settings_embed_model", lambda: None)
    monkeypatch.setattr(module, "setup_llamaindex", lambda **_: None)
    warnings: list[str] = []

    def _warn(msg: str, *args: Any, **kwargs: Any) -> None:
        warnings.append(str(msg))

    monkeypatch.setattr(module.logger, "warning", _warn, raising=False)

    cfg = IngestionConfig(cache_dir=tmp_path / "cache")
    inputs = [IngestionInput(document_id="doc", payload_text="payload")]

    result = await module.ingest_documents(cfg, inputs, embedding=None)

    assert call_state["embedding"] is None
    assert result.nodes == [{"doc_id": "doc", "text": "payload"}]
    assert any("No embedding model configured" in msg for msg in warnings)


@pytest.mark.asyncio
async def test_document_from_input_propagates_parser_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Path inputs have one canonical parser failure boundary."""
    sample = tmp_path / "sample.txt"
    sample.write_text("Parser-owned content", encoding="utf-8")

    from src.processing import ingestion_api as api
    from src.processing.parsing.errors import DocumentParseError

    async def _explode(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise TypeError("unexpected signature")

    monkeypatch.setattr(api, "_parse_path", _explode)

    item = IngestionInput(document_id="doc-1", source_path=sample)

    with pytest.raises(DocumentParseError) as raised:
        await load_documents_from_inputs([item])

    assert raised.value.stage == "ingestion_facade"
    assert raised.value.reason == "parser_service_failed"


@pytest.mark.asyncio
async def test_pdf_parser_failure_stops_before_page_exports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A PDF parse failure propagates before image artifacts are generated."""
    from src.processing import ingestion_api as api
    from src.processing import ingestion_pipeline as module
    from src.processing.parsing.errors import DocumentParseError

    async def _explode(*_args: object, **_kwargs: object) -> None:
        raise DocumentParseError(
            pdf,
            stage="docling_conversion",
            reason="conversion_failed",
        )

    export_called = False

    def _exports(*_args: object, **_kwargs: object) -> list[Any]:
        nonlocal export_called
        export_called = True
        return []

    pdf = tmp_path / "malformed.pdf"
    pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n")
    monkeypatch.setattr(api, "_parse_path", _explode)
    monkeypatch.setattr(module, "_page_image_exports", _exports)
    cfg = IngestionConfig(cache_dir=tmp_path / "cache")
    inputs = [IngestionInput(document_id="doc-1", source_path=pdf)]

    with pytest.raises(DocumentParseError):
        await module._load_documents(cfg, inputs)

    assert export_called is False


def test_page_image_exports_builds_metadata(monkeypatch, tmp_path: Path) -> None:
    from src.processing import ingestion_pipeline as module

    entries = [
        {
            "page": 1,
            "image_path": str(tmp_path / "sample.webp"),
            "phash": "abc",
        },
        {
            "page": 2,
            "image_path": str(tmp_path / "sample.jpg.enc"),
            "phash": "def",
        },
    ]
    monkeypatch.setattr(module, "save_pdf_page_images", lambda *args, **kwargs: entries)

    pdf = tmp_path / "doc.pdf"
    pdf.write_text("pdf", encoding="utf-8")
    cfg = IngestionConfig(cache_dir=tmp_path)

    exports = module._page_image_exports(pdf, cfg, encrypt_override=False)

    assert exports[0].content_type == "image/webp"
    assert exports[1].content_type == "image/jpeg"


@pytest.mark.asyncio
async def test_load_documents_uses_parser_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src.processing import ingestion_api as api
    from src.processing import ingestion_pipeline as module
    from src.processing.parsing.canonical_types import (
        DocumentParseResult,
        PageParseResult,
        ParserFramework,
        ParserProfile,
    )

    async def _parse(
        path: Path,
        *,
        document_id: str,
        parsing_overrides: dict[str, Any] | None = None,
    ) -> DocumentParseResult:
        assert parsing_overrides == {"force_ocr": True}
        return DocumentParseResult(
            document_id=document_id,
            source_filename=path.name,
            source_hash="abc",
            profile=ParserProfile.CPU_SAFE,
            parser_framework=ParserFramework.DOCLING,
            page_count=1,
            pages=[
                PageParseResult(
                    page_id=f"{document_id}::page::1",
                    page_index=0,
                    text_markdown="doc",
                    routing_reason="test",
                )
            ],
        )

    monkeypatch.setattr(api, "_parse_path", _parse)
    monkeypatch.setattr(
        module, "_page_image_exports", lambda *args, **kwargs: ["export"]
    )

    sample = tmp_path / "sample.pdf"
    sample.write_text("content", encoding="utf-8")
    cfg = IngestionConfig(cache_dir=tmp_path)
    inputs = [
        IngestionInput(
            document_id="doc-1",
            source_path=sample,
            encrypt_images=True,
            parsing_overrides=ParsingOverrides(force_ocr=True),
        )
    ]

    docs, exports = await module._load_documents(cfg, inputs)
    assert docs[0].metadata["document_id"] == "doc-1"
    assert exports == ["export"]
