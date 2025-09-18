from __future__ import annotations

import pytest

"""Top-level pytest configuration for shared fixtures.

Loads the shared fixtures module so fixtures like `supervisor_stream_shim` are
available across test tiers without explicit imports in each test file.
"""

pytest_plugins = [
    "tests.shared_fixtures",
]


@pytest.fixture(autouse=True)
def _mock_qdrant_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide an in-memory Qdrant client stub to avoid network access."""
    from types import SimpleNamespace

    from src.config import settings as app_settings

    class _FakeQdrantClient:
        def __init__(self, *args, **kwargs) -> None:
            self._collections: dict[str, dict[str, object]] = {}

        def collection_exists(self, name: str) -> bool:
            return name in self._collections

        def get_collection(self, name: str) -> SimpleNamespace:
            cfg = self._collections.get(name)
            if cfg is None:
                cfg = {
                    "vectors_config": {
                        "text-dense": SimpleNamespace(
                            size=app_settings.embedding.dimension
                        )
                    },
                    "sparse_vectors_config": {"text-sparse": SimpleNamespace()},
                }
            params = SimpleNamespace(
                vectors_config=cfg.get("vectors_config"),
                sparse_vectors_config=cfg.get("sparse_vectors_config"),
            )
            return SimpleNamespace(config=SimpleNamespace(params=params))

        def create_collection(self, collection_name: str, **kwargs) -> None:
            self._collections[collection_name] = {
                "vectors_config": kwargs.get("vectors_config", {}),
                "sparse_vectors_config": kwargs.get("sparse_vectors_config", {}),
            }

        def update_collection(self, *args, **kwargs) -> None:  # pragma: no cover
            return None

        def recreate_collection(self, *args, **kwargs) -> None:  # pragma: no cover
            return None

        def query_points(self, **kwargs):  # pragma: no cover - tests patch dynamically
            return SimpleNamespace(points=[])

        def close(self) -> None:  # pragma: no cover - simple stub
            return None

    monkeypatch.setattr(
        "src.retrieval.hybrid.QdrantClient",
        lambda *args, **kwargs: _FakeQdrantClient(*args, **kwargs),
    )
    monkeypatch.setattr(
        "tools.eval.run_beir.QdrantClient",
        lambda *args, **kwargs: _FakeQdrantClient(*args, **kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        "src.utils.storage.QdrantClient",
        _FakeQdrantClient,
        raising=False,
    )


@pytest.fixture
def integration_settings():
    """Provide standardized integration settings with lightweight embedding.

    Ensures `embedding_dimension == 384` and model name contains "MiniLM" as
    asserted by infrastructure validation tests.
    """
    # Lazy import to avoid early import-time path issues during PyTest startup
    from tests.fixtures.test_settings import create_integration_settings

    s = create_integration_settings()
    # Ensure expected lightweight model naming for assertions
    s.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return s


@pytest.fixture
def system_settings():
    """Provide system-tier settings approximating production configuration."""
    from tests.fixtures.test_settings import create_system_settings

    return create_system_settings()


@pytest.fixture
def lightweight_embedding_model():
    """Provide a lightweight embedding model stub for integration tests.

    Mimics all-MiniLM-L6-v2 behavior by exposing an `encode` method that returns
    (N, 384) float32 embeddings.
    """
    import numpy as np

    class _MiniLM:
        def encode(self, items: list[str]):
            return np.zeros((len(items), 384), dtype=np.float32)

    return _MiniLM()



@pytest.fixture(autouse=True)
def _stub_llm_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent external LLM initialization during tests."""
    from llama_index.core.llms.mock import MockLLM
    from llama_index.core import Settings
    from src.config import integrations as integrations_module

    monkeypatch.setattr(integrations_module, "build_llm", lambda *_args, **_kwargs: MockLLM())
    monkeypatch.setattr(Settings, "_llm", MockLLM(), raising=False)






@pytest.fixture(autouse=True)
def _stub_router_postprocessor_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub postprocessor builders to work with lightweight test doubles."""
    from types import SimpleNamespace
    from src.retrieval import postprocessor_utils as pu

    def _vector(index, post, **kwargs):
        try:
            qe = index.as_query_engine(node_postprocessors=post, **kwargs)
        except TypeError:
            qe = index.as_query_engine(**kwargs)
        if hasattr(index, "kwargs"):
            index.kwargs = {"node_postprocessors": post, **kwargs}
        if hasattr(qe, "kwargs"):
            qe.kwargs = {"node_postprocessors": post, **kwargs}
        return qe

    def _graph(pg_index, post, **kwargs):
        try:
            retriever = pg_index.as_retriever(
                include_text=kwargs.pop("include_text", True),
                path_depth=kwargs.get("path_depth", 1),
            )
        except TypeError:
            retriever = pg_index.as_retriever()
        try:
            qe = pg_index.as_query_engine(node_postprocessors=post)
        except TypeError:
            qe = pg_index.as_query_engine()
        if hasattr(pg_index, "kwargs"):
            pg_index.kwargs = {"node_postprocessors": post}
        if hasattr(qe, "kwargs"):
            qe.kwargs = {"node_postprocessors": post}
        if hasattr(retriever, "kwargs"):
            retriever.kwargs = {"node_postprocessors": post}
        return SimpleNamespace(query_engine=qe, retriever=retriever)

    def _retriever(retriever, post, llm=None, engine_cls=None, **kwargs):
        engine = engine_cls or SimpleNamespace
        try:
            out = engine.from_args(
                retriever=retriever,
                llm=llm,
                node_postprocessors=post,
                **kwargs,
            )
        except AttributeError:
            out = SimpleNamespace(retriever=retriever, llm=llm, kwargs=kwargs)
        if hasattr(retriever, "kwargs"):
            retriever.kwargs = {"node_postprocessors": post, **kwargs}
        if hasattr(out, "kwargs"):
            out.kwargs = {"node_postprocessors": post, **kwargs}
        return out

    monkeypatch.setattr(pu, "build_vector_query_engine", _vector, raising=False)
    monkeypatch.setattr(pu, "build_pg_query_engine", _graph, raising=False)
    monkeypatch.setattr(pu, "build_retriever_query_engine", _retriever, raising=False)


@pytest.fixture(autouse=True)
def _router_factory_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide lightweight stubs for router factory dependencies."""
    from types import SimpleNamespace
    from src.retrieval import router_factory as rf
    from src.retrieval import postprocessor_utils as pu
    from src.retrieval import hybrid as hybrid_module
    from src.retrieval import graph_config as gc

    class _ToolMetadata:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

    class _QueryEngineTool:
        def __init__(self, query_engine, metadata):
            self.query_engine = query_engine
            self.metadata = metadata

    class _RouterQueryEngine:
        def __init__(self, selector=None, query_engine_tools=None, verbose=False, llm=None, **kwargs):
            self.selector = selector
            self.query_engine_tools = list(query_engine_tools or [])
            self.verbose = verbose
            self.llm = llm
            self.kwargs = kwargs

        @classmethod
        def from_args(cls, **kwargs):
            return cls(**kwargs)

    class _RetrieverQueryEngine:
        @classmethod
        def from_args(cls, retriever, llm=None, **kwargs):
            obj = SimpleNamespace(retriever=retriever, llm=llm, kwargs=kwargs)
            return obj

    def _vector(index, post, **kwargs):
        try:
            qe = index.as_query_engine(node_postprocessors=post, **kwargs)
        except TypeError:
            qe = index.as_query_engine(**kwargs)
        if hasattr(index, "kwargs"):
            index.kwargs = {"node_postprocessors": post, **kwargs}
        if hasattr(qe, "kwargs"):
            qe.kwargs = {"node_postprocessors": post, **kwargs}
        return qe

    def _graph(pg_index, post, **kwargs):
        include_text = kwargs.get("include_text", True)
        path_depth = kwargs.get("path_depth", 1)
        try:
            retriever = pg_index.as_retriever(include_text=include_text, path_depth=path_depth)
        except TypeError:
            retriever = pg_index.as_retriever()
        try:
            qe = pg_index.as_query_engine(node_postprocessors=post)
        except TypeError:
            qe = pg_index.as_query_engine()
        if hasattr(qe, "kwargs"):
            qe.kwargs = {"node_postprocessors": post}
        if hasattr(retriever, "kwargs"):
            retriever.kwargs = {"node_postprocessors": post}
        return SimpleNamespace(query_engine=qe, retriever=retriever)

    def _retriever(retriever, post, llm=None, engine_cls=None, **kwargs):
        engine = engine_cls or _RetrieverQueryEngine
        return engine.from_args(retriever=retriever, llm=llm, node_postprocessors=post, **kwargs)

    class _StubHybrid:
        def __init__(self, *_args, **_kwargs) -> None:
            self._closed = False

        def retrieve(self, _query):
            return []

        def close(self) -> None:
            self._closed = True

    monkeypatch.setattr(rf, "QueryEngineTool", _QueryEngineTool, raising=False)
    monkeypatch.setattr(rf, "ToolMetadata", _ToolMetadata, raising=False)
    monkeypatch.setattr(rf, "RouterQueryEngine", _RouterQueryEngine, raising=False)
    monkeypatch.setattr(rf, "RetrieverQueryEngine", _RetrieverQueryEngine, raising=False)
    monkeypatch.setattr(pu, "build_vector_query_engine", _vector, raising=False)
    monkeypatch.setattr(pu, "build_pg_query_engine", _graph, raising=False)
    monkeypatch.setattr(pu, "build_retriever_query_engine", _retriever, raising=False)
    monkeypatch.setattr(gc, "build_graph_query_engine", _graph, raising=False)
    monkeypatch.setattr(hybrid_module, "ServerHybridRetriever", _StubHybrid, raising=False)
