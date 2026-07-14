"""Unit tests for verified graph snapshots and live-Qdrant activation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from llama_index.core import PropertyGraphIndex, Settings, StorageContext
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    EntityNode,
    Relation,
)
from llama_index.core.indices.property_graph.sub_retrievers.vector import (
    VectorContextRetriever,
)
from llama_index.core.indices.property_graph.transformations import (
    ImplicitPathExtractor,
)
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore

from src.persistence.snapshot import (
    SnapshotManager,
    load_property_graph_index,
    load_vector_index,
)


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point data_dir to tmp so SnapshotManager writes under it
    from src.config.settings import settings as _settings

    monkeypatch.setattr(_settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(
        _settings.chat, "sqlite_path", tmp_path / "chat.db", raising=False
    )


def test_snapshot_roundtrip_with_stubs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.settings import settings

    monkeypatch.setattr(settings.retrieval, "enable_server_hybrid", False)
    monkeypatch.setattr(settings.retrieval, "enable_keyword_tool", True)
    # Persist a snapshot with our stub indices
    mgr = SnapshotManager(tmp_path / "storage")
    tmp = mgr.begin_snapshot()
    try:
        mgr.persist_graph_storage_context(
            StorageContext.from_defaults(
                property_graph_store=SimplePropertyGraphStore()
            ),
            tmp,
        )
        # Minimal manifest
        mgr.write_manifest(
            tmp,
            index_id="x",
            graph_store_type="property_graph",
            vector_store_type="qdrant",
            text_collection="physical-text-v2",
            image_collection="physical-image-v2",
            corpus_hash="c" * 64,
            config_hash="f" * 64,
            versions={"app": "test"},
        )
        final = mgr.finalize_snapshot(tmp)
    except Exception:
        mgr.cleanup_tmp(tmp)
        raise

    # Create stub modules for llama_index loaders and force-register them in
    # sys.modules so subsequent imports within snapshot helpers resolve to these
    # stubs regardless of prior imports elsewhere in the suite.
    core_mod = ModuleType("llama_index.core")
    monkeypatch.setitem(sys.modules, "llama_index.core", core_mod)

    vector_store = SimpleNamespace(client=SimpleNamespace(close=lambda: None))

    class _VectorStoreIndex:
        @staticmethod
        def from_vector_store(store: object) -> SimpleNamespace:
            assert store is vector_store
            return SimpleNamespace(vector_store=store)

    class _PropertyGraphIndex:
        @staticmethod
        def from_existing(  # type: ignore[no-untyped-def]
            *, property_graph_store, vector_store, storage_context
        ):
            assert vector_store is storage_context.vector_store
            return SimpleNamespace(property_graph_store=property_graph_store)

    class _StorageContext:
        @staticmethod
        def from_defaults(*, persist_dir: str):  # type: ignore[no-untyped-def]
            graph_dir = Path(persist_dir)
            assert (graph_dir / "property_graph_store.json").exists()
            return SimpleNamespace(
                property_graph_store=SimpleNamespace(persist_dir=persist_dir),
                vector_store=SimpleNamespace(persist_dir=persist_dir),
            )

    monkeypatch.setattr(core_mod, "VectorStoreIndex", _VectorStoreIndex, raising=False)
    monkeypatch.setattr(
        core_mod, "PropertyGraphIndex", _PropertyGraphIndex, raising=False
    )
    monkeypatch.setattr(core_mod, "StorageContext", _StorageContext, raising=False)
    connected_collections: list[str] = []
    connected_options: list[dict[str, object]] = []

    def _connect(collection: str, **kwargs: object) -> object:
        connected_collections.append(collection)
        connected_options.append(kwargs)
        return vector_store

    monkeypatch.setattr("src.utils.storage.connect_vector_store", _connect)

    # Load via helpers and assert non-null
    vec = load_vector_index(final)
    pg = load_property_graph_index(final)
    assert vec is not None
    assert pg is not None
    # Spot-check attributes from stubs
    assert vec.vector_store is vector_store
    assert connected_collections == ["physical-text-v2"]
    assert connected_options[0]["enable_hybrid"] is True
    assert Path(pg.property_graph_store.persist_dir).name == "graph"
    assert not (final / "vector").exists()


def test_property_graph_vector_retrieval_survives_snapshot_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native graph context retains KG embeddings across activation reload."""
    alpha = EntityNode(name="Alpha")
    beta = EntityNode(name="Beta")
    relation = Relation(
        label="relates",
        source_id=alpha.id,
        target_id=beta.id,
    )
    embed_model = MockEmbedding(embed_dim=4)
    source_node = TextNode(
        text="Alpha relates to Beta.",
        metadata={
            KG_NODES_KEY: [alpha, beta],
            KG_RELATIONS_KEY: [relation],
        },
    )
    graph_index = PropertyGraphIndex(
        nodes=[source_node],
        kg_extractors=[ImplicitPathExtractor()],
        property_graph_store=SimplePropertyGraphStore(),
        vector_store=SimpleVectorStore(),
        use_async=False,
        embed_model=embed_model,
        show_progress=False,
    )
    assert graph_index.vector_store is not None

    manager = SnapshotManager(tmp_path / "storage")
    workspace = manager.begin_snapshot()
    manager.persist_graph_storage_context(graph_index.storage_context, workspace)
    manager.write_manifest(
        workspace,
        index_id="graph-roundtrip",
        graph_store_type="property_graph",
        vector_store_type="qdrant",
        text_collection="physical-text-v2",
        image_collection="physical-image-v2",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
        versions={"app": "test"},
    )
    final = manager.finalize_snapshot(workspace)

    monkeypatch.setattr(Settings, "_embed_model", embed_model)
    loaded = load_property_graph_index(final)

    assert loaded is not None
    retriever = VectorContextRetriever(
        graph_store=loaded.property_graph_store,
        vector_store=loaded.vector_store,
        embed_model=embed_model,
        include_text=False,
        similarity_top_k=2,
    )
    results = retriever.retrieve("Alpha")
    assert len(results) == 1
    assert "Alpha -> relates -> Beta" in results[0].node.get_content()


def test_vector_activation_failure_closes_both_qdrant_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed index wrapper construction releases both live-Qdrant clients."""
    mgr = SnapshotManager(tmp_path / "storage")
    tmp = mgr.begin_snapshot()
    mgr.write_manifest(
        tmp,
        index_id="x",
        graph_store_type="none",
        vector_store_type="qdrant",
        text_collection="physical-text-v2",
        image_collection="physical-image-v2",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
        versions={"app": "test"},
    )
    final = mgr.finalize_snapshot(tmp)

    core_mod = ModuleType("llama_index.core")
    monkeypatch.setitem(sys.modules, "llama_index.core", core_mod)

    class _VectorStoreIndex:
        @staticmethod
        def from_vector_store(_store: object) -> None:
            raise RuntimeError("index wrapper failed")

    class _AsyncClient:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    client = SimpleNamespace(closed=False)

    def _close() -> None:
        client.closed = True

    client.close = _close
    async_client = _AsyncClient()
    vector_store = SimpleNamespace(client=client, _aclient=async_client)
    monkeypatch.setattr(core_mod, "VectorStoreIndex", _VectorStoreIndex, raising=False)
    monkeypatch.setattr(
        "src.utils.storage.connect_vector_store",
        lambda *_args, **_kwargs: vector_store,
    )

    assert load_vector_index(final) is None
    assert client.closed is True
    assert async_client.closed is True
