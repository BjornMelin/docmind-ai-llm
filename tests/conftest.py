from __future__ import annotations

import importlib.util
import os

import pytest

"""Top-level pytest configuration for shared fixtures.

Loads the shared fixtures module so fixtures like `supervisor_stream_shim` are
available across test tiers without explicit imports in each test file.
"""

_REQUIRE_REAL_LLAMA = os.getenv("REQUIRE_REAL_LLAMA", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


pytest_plugins = [
    "tests.shared_fixtures",
]


def _llama_available() -> bool:
    """Return True when the real llama_index dependency is importable."""
    return importlib.util.find_spec("llama_index.core") is not None


def pytest_configure(config) -> None:
    """Register custom markers for this test suite."""
    config.addinivalue_line(
        "markers",
        (
            "requires_llama: mark tests that must run with the real "
            "llama_index.core dependency."
        ),
    )


def pytest_runtest_setup(item) -> None:
    """Handle requires_llama marks with skip/fail behavior."""
    if "requires_llama" not in item.keywords or _llama_available():
        return
    if _REQUIRE_REAL_LLAMA:
        pytest.fail(
            "llama_index.core is required for this test but is not installed",
            pytrace=False,
        )
    pytest.skip("llama_index.core not installed")


@pytest.fixture(autouse=True)
def _reset_otel_providers() -> None:
    """Ensure OpenTelemetry providers reset between tests."""
    from src.telemetry import opentelemetry as otel_module

    otel_module.shutdown_metrics()
    otel_module.shutdown_tracing()


@pytest.fixture(autouse=True)
def _reset_telemetry_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset telemetry globals and environment overrides between tests."""
    from pathlib import Path

    from src.utils import telemetry as telem

    monkeypatch.setattr(
        telem, "_TELEM_PATH", Path("./logs/telemetry.jsonl"), raising=False
    )
    telem.set_request_id(None)
    monkeypatch.delenv("DOCMIND_TELEMETRY_DISABLED", raising=False)
    monkeypatch.delenv("DOCMIND_TELEMETRY_SAMPLE", raising=False)
    monkeypatch.delenv("DOCMIND_TELEMETRY_ROTATE_BYTES", raising=False)


@pytest.fixture(autouse=True)
def _stub_embed_model(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Provide a lightweight embedding model for hybrid retrieval tests."""
    module_name = getattr(request.module, "__name__", "")
    if module_name.startswith("tests.unit.config"):
        return
    from src.config import integrations as integrations_module

    class _Embed:
        def get_query_embedding(self, _text: str) -> list[float]:
            return [0.0, 0.1, 0.2]

    monkeypatch.setattr(
        integrations_module,
        "get_settings_embed_model",
        lambda: _Embed(),
        raising=False,
    )
    monkeypatch.setattr(
        "src.retrieval.hybrid.get_settings_embed_model",
        lambda: _Embed(),
        raising=False,
    )


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
    from llama_index.core import Settings
    from llama_index.core.llms.mock import MockLLM

    from src.config import integrations as integrations_module

    monkeypatch.setattr(
        integrations_module, "build_llm", lambda *_args, **_kwargs: MockLLM()
    )
    monkeypatch.setattr(Settings, "_llm", MockLLM(), raising=False)


@pytest.fixture(autouse=True)
def _stub_router_postprocessor_builders(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Stub postprocessor builders to work with lightweight test doubles."""
    module_name = getattr(request.module, "__name__", "")
    if module_name.endswith("test_postprocessor_utils_builders"):
        return
    from types import SimpleNamespace

    from src.retrieval import graph_config as gc
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

    def _graph(pg_index, post=None, **kwargs):
        post = kwargs.pop("node_postprocessors", post)
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
        effective_kwargs = {"node_postprocessors": post}
        if hasattr(qe, "kwargs"):
            qe.kwargs = effective_kwargs
        if hasattr(retriever, "kwargs"):
            retriever.kwargs = effective_kwargs
        out = SimpleNamespace(
            query_engine=qe, retriever=retriever, kwargs=effective_kwargs
        )
        return out

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
        effective_kwargs = {"node_postprocessors": post, **kwargs}
        if hasattr(retriever, "kwargs"):
            retriever.kwargs = effective_kwargs
        if hasattr(out, "kwargs"):
            out.kwargs = effective_kwargs
        return out

    class _GraphRetrieverEngine:
        @staticmethod
        def from_args(**kwargs):
            return SimpleNamespace(**kwargs)

    monkeypatch.setattr(pu, "build_vector_query_engine", _vector, raising=False)
    monkeypatch.setattr(pu, "build_pg_query_engine", _graph, raising=False)
    monkeypatch.setattr(pu, "build_retriever_query_engine", _retriever, raising=False)
    monkeypatch.setattr(
        gc,
        "RetrieverQueryEngine",
        _GraphRetrieverEngine,
        raising=False,
    )
    monkeypatch.setattr(gc, "_require_llamaindex", lambda: None, raising=False)


@pytest.fixture(autouse=True)
def _router_factory_stubs(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
):
    """Provide lightweight stubs for router factory dependencies."""
    from types import SimpleNamespace

    from src.retrieval import graph_config as gc
    from src.retrieval import postprocessor_utils as pu
    from src.retrieval import router_factory as rf
    from src.retrieval.llama_index_adapter import set_llama_index_adapter

    if "requires_llama" in request.keywords:
        set_llama_index_adapter(None)
        try:
            yield
        finally:
            set_llama_index_adapter(None)
        return

    module_name = getattr(request.module, "__name__", "")
    if module_name.endswith("test_postprocessor_utils_builders"):
        set_llama_index_adapter(None)
        try:
            yield
        finally:
            set_llama_index_adapter(None)
        return

    class _ToolMetadata:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

    class _QueryEngineTool:
        def __init__(self, query_engine, metadata) -> None:
            self.query_engine = query_engine
            self.metadata = metadata

    class _RouterQueryEngine:
        def __init__(
            self,
            selector=None,
            query_engine_tools=None,
            verbose=False,
            llm=None,
            **kwargs,
        ) -> None:
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
            return SimpleNamespace(retriever=retriever, llm=llm, kwargs=kwargs)

    class _LLMSingleSelector:
        def __init__(self, llm=None) -> None:
            self.llm = llm
            self.kind = "llm_selector"

        @classmethod
        def from_defaults(cls, llm=None):
            return cls(llm=llm)

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

    def _graph(pg_index, node_postprocessors=None, **kwargs):
        include_text = kwargs.get("include_text", True)
        path_depth = kwargs.get("path_depth", 1)
        try:
            retriever = pg_index.as_retriever(
                include_text=include_text, path_depth=path_depth
            )
        except TypeError:
            retriever = pg_index.as_retriever()
        except AttributeError:
            retriever = SimpleNamespace(kwargs={})
        try:
            qe = pg_index.as_query_engine(node_postprocessors=node_postprocessors)
        except TypeError:
            qe = pg_index.as_query_engine()
        except AttributeError:
            qe = SimpleNamespace(kwargs={})
        effective_kwargs = {"node_postprocessors": node_postprocessors}
        if hasattr(qe, "kwargs"):
            qe.kwargs = effective_kwargs
        if hasattr(retriever, "kwargs"):
            retriever.kwargs = effective_kwargs
        return SimpleNamespace(
            query_engine=qe, retriever=retriever, kwargs=effective_kwargs
        )

    def _retriever(retriever, post, llm=None, engine_cls=None, **kwargs):
        engine = engine_cls or _RetrieverQueryEngine
        return engine.from_args(
            retriever=retriever, llm=llm, node_postprocessors=post, **kwargs
        )

    adapter = SimpleNamespace(
        RouterQueryEngine=_RouterQueryEngine,
        RetrieverQueryEngine=_RetrieverQueryEngine,
        QueryEngineTool=_QueryEngineTool,
        ToolMetadata=_ToolMetadata,
        LLMSingleSelector=_LLMSingleSelector,
        get_pydantic_selector=lambda llm: None,
        __is_stub__=True,
        supports_graphrag=False,
        graphrag_disabled_reason=rf.GRAPH_DEPENDENCY_HINT,
    )

    set_llama_index_adapter(adapter)

    monkeypatch.setattr(rf, "QueryEngineTool", _QueryEngineTool, raising=False)
    monkeypatch.setattr(rf, "ToolMetadata", _ToolMetadata, raising=False)
    monkeypatch.setattr(rf, "RouterQueryEngine", _RouterQueryEngine, raising=False)
    monkeypatch.setattr(
        rf, "RetrieverQueryEngine", _RetrieverQueryEngine, raising=False
    )
    monkeypatch.setattr(pu, "build_vector_query_engine", _vector, raising=False)
    monkeypatch.setattr(pu, "build_pg_query_engine", _graph, raising=False)
    monkeypatch.setattr(pu, "build_retriever_query_engine", _retriever, raising=False)
    monkeypatch.setattr(rf, "build_vector_query_engine", _vector, raising=False)
    monkeypatch.setattr(rf, "build_pg_query_engine", _graph, raising=False)
    monkeypatch.setattr(rf, "build_retriever_query_engine", _retriever, raising=False)
    monkeypatch.setattr(gc, "build_graph_query_engine", _graph, raising=False)
    monkeypatch.setattr(rf, "build_graph_query_engine", _graph, raising=False)
    monkeypatch.setattr(
        gc,
        "RetrieverQueryEngine",
        _RetrieverQueryEngine,
        raising=False,
    )
    monkeypatch.setattr(gc, "_require_llamaindex", lambda: None, raising=False)
    try:
        yield
    finally:
        set_llama_index_adapter(None)

    class _StubToolMetadata:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

    class _StubQueryEngineTool:
        def __init__(self, query_engine, metadata) -> None:
            self.query_engine = query_engine
            self.metadata = metadata

    class _StubRouterQueryEngine:
        def __init__(
            self,
            *,
            selector=None,
            query_engine_tools=None,
            verbose=False,
            llm=None,
            **kwargs,
        ) -> None:
            self.selector = selector
            self.query_engine_tools = list(query_engine_tools or [])
            self.verbose = verbose
            self.llm = llm
            self.kwargs = kwargs

        @classmethod
        def from_args(cls, **kwargs):
            return cls(**kwargs)

    class _StubRetrieverQueryEngine:
        @classmethod
        def from_args(cls, **kwargs):
            return SimpleNamespace(**kwargs)

    class _StubLLMSingleSelector:
        def __init__(self, llm=None) -> None:
            self.llm = llm

        @classmethod
        def from_defaults(cls, llm=None):
            return cls(llm=llm)

    def _get_pydantic_selector(_llm):
        return None

    adapter = SimpleNamespace(
        RouterQueryEngine=_StubRouterQueryEngine,
        RetrieverQueryEngine=_StubRetrieverQueryEngine,
        QueryEngineTool=_StubQueryEngineTool,
        ToolMetadata=_StubToolMetadata,
        LLMSingleSelector=_StubLLMSingleSelector,
        get_pydantic_selector=_get_pydantic_selector,
        __is_stub__=True,
        supports_graphrag=False,
        graphrag_disabled_reason=rf.GRAPH_DEPENDENCY_HINT,
    )

    def _resolved(adapter_arg):
        return adapter if adapter_arg is None else adapter_arg

    monkeypatch.setattr(rf, "_resolve_adapter", _resolved, raising=False)
