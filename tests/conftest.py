"""Top-level pytest configuration and shared fixtures."""

from __future__ import annotations

import importlib
import importlib.util
import os
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest_plugins = [
    "tests.shared_fixtures",
]

_REQUIRE_REAL_LLAMA = os.getenv("REQUIRE_REAL_LLAMA", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _llama_available() -> bool:
    return importlib.util.find_spec("llama_index.core") is not None


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    """Register custom markers for this test suite."""
    config.addinivalue_line(
        "markers",
        "requires_llama: mark tests that must run with real llama_index installed.",
    )


def pytest_runtest_setup(item) -> None:  # type: ignore[no-untyped-def]
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
    """Provide a lightweight embedding model for tests that touch embeddings."""
    module_name = getattr(request.module, "__name__", "")
    if module_name.startswith("tests.unit.config"):
        return

    from src.config import integrations as integrations_module

    class _Embed:
        def get_query_embedding(self, _text: str) -> list[float]:
            """Return a fixed embedding vector for testing."""
            return [0.0, 0.1, 0.2]

    def _get_embed_model() -> _Embed:
        return _Embed()

    monkeypatch.setattr(
        integrations_module,
        "get_settings_embed_model",
        _get_embed_model,
        raising=False,
    )
    monkeypatch.setattr(
        "src.retrieval.hybrid.get_settings_embed_model",
        _get_embed_model,
        raising=False,
    )


@pytest.fixture(autouse=True)
def _mock_qdrant_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide an in-memory Qdrant client stub to avoid network access."""
    from src.config import settings as app_settings

    class _FakeQdrantClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._collections: dict[str, dict[str, object]] = {}

        def collection_exists(self, name: str) -> bool:
            """Check if a collection exists."""
            return name in self._collections

        def get_collection(self, name: str) -> SimpleNamespace:
            """Get collection configuration and metadata."""
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

        def create_collection(self, collection_name: str, **kwargs: object) -> None:
            """Create a new collection with the given configuration."""
            self._collections[collection_name] = {
                "vectors_config": kwargs.get("vectors_config", {}),
                "sparse_vectors_config": kwargs.get("sparse_vectors_config", {}),
            }

        def update_collection(
            self, *args: object, **kwargs: object
        ) -> None:  # pragma: no cover
            """Update collection configuration."""
            return None

        def recreate_collection(
            self, *args: object, **kwargs: object
        ) -> None:  # pragma: no cover
            """Recreate an existing collection."""
            return None

        def query_points(self, **kwargs: object) -> SimpleNamespace:  # pragma: no cover
            """Query points from the collection."""
            return SimpleNamespace(points=[])

        def close(self) -> None:  # pragma: no cover
            """Close the client connection."""
            return None

    def _create_fake_qdrant(*args: object, **kwargs: object) -> _FakeQdrantClient:
        return _FakeQdrantClient(*args, **kwargs)

    monkeypatch.setattr(
        "src.retrieval.hybrid.QdrantClient",
        _create_fake_qdrant,
        raising=False,
    )
    with suppress(ImportError):
        from tools.eval import run_beir as beir_module

        monkeypatch.setattr(
            beir_module,
            "QdrantClient",
            _create_fake_qdrant,
            raising=False,
        )
    monkeypatch.setattr(
        "src.utils.storage.QdrantClient", _FakeQdrantClient, raising=False
    )


@pytest.fixture
def integration_settings():
    """Provide standardized integration settings with lightweight embedding."""
    from tests.fixtures.test_settings import create_integration_settings

    s = create_integration_settings()
    s.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return s


@pytest.fixture
def system_settings():
    """Provide system-tier settings approximating production configuration."""
    from tests.fixtures.test_settings import create_system_settings

    return create_system_settings()


@pytest.fixture
def lightweight_embedding_model():
    """Provide a lightweight embedding model stub for integration tests."""
    import numpy as np

    class _MiniLM:
        def encode(self, items: list[str]):
            """Encode text items to fixed embeddings for testing."""
            return np.zeros((len(items), 384), dtype=np.float32)

    return _MiniLM()


@pytest.fixture(autouse=True)
def _stub_llm_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent external LLM initialization during tests."""
    if not _llama_available():
        return
    from llama_index.core import Settings
    from llama_index.core.llms.mock import MockLLM

    from src.config import integrations as integrations_module

    def _build_mock_llm(*_a: object, **_k: object) -> MockLLM:
        return MockLLM()

    monkeypatch.setattr(integrations_module, "build_llm", _build_mock_llm)
    monkeypatch.setattr(Settings, "_llm", MockLLM(), raising=False)


@pytest.fixture(autouse=True)
def _router_factory_stubs(request: pytest.FixtureRequest) -> None:
    """Inject a stub llama-index adapter unless the test requests the real one."""
    from src.retrieval.llama_index_adapter import set_llama_index_adapter

    if "requires_llama" in request.keywords:
        set_llama_index_adapter(None)
        try:
            yield
        finally:
            set_llama_index_adapter(None)
        return

    class _ToolMetadata:
        """Stub for tool metadata."""

        def __init__(self, name: str, description: str) -> None:
            """Initialize tool metadata."""
            self.name = name
            self.description = description

    class _QueryEngineTool:
        """Stub for query engine tool."""

        def __init__(self, query_engine: object, metadata: _ToolMetadata) -> None:
            """Initialize query engine tool."""
            self.query_engine = query_engine
            self.metadata = metadata

    class _RouterQueryEngine:
        """Stub for router query engine."""

        def __init__(
            self,
            selector: object | None = None,
            query_engine_tools: list[object] | None = None,
            verbose: bool = False,
            llm: object | None = None,
            **kwargs: object,
        ) -> None:
            """Initialize router query engine."""
            self.selector = selector
            self.query_engine_tools = list(query_engine_tools or [])
            self.verbose = verbose
            self.llm = llm
            self.kwargs = kwargs

        @classmethod
        def from_args(cls, **kwargs: object) -> _RouterQueryEngine:
            """Create instance from keyword arguments."""
            return cls(**kwargs)

    class _RetrieverQueryEngine:
        """Stub for retriever query engine."""

        @classmethod
        def from_args(cls, **kwargs: object) -> SimpleNamespace:
            """Create instance from keyword arguments."""
            return SimpleNamespace(**kwargs)

    class _LLMSingleSelector:
        """Stub for LLM single selector."""

        def __init__(self, llm: object | None = None) -> None:
            """Initialize LLM single selector."""
            self.llm = llm
            self.kind = "llm_selector"

        @classmethod
        def from_defaults(cls, llm: object | None = None) -> _LLMSingleSelector:
            """Create instance with default settings."""
            return cls(llm=llm)

    def _get_pydantic_selector(_llm: object) -> None:
        return None

    adapter = SimpleNamespace(
        RouterQueryEngine=_RouterQueryEngine,
        RetrieverQueryEngine=_RetrieverQueryEngine,
        QueryEngineTool=_QueryEngineTool,
        ToolMetadata=_ToolMetadata,
        LLMSingleSelector=_LLMSingleSelector,
        get_pydantic_selector=_get_pydantic_selector,
        __is_stub__=True,
    )
    set_llama_index_adapter(adapter)
    try:
        yield
    finally:
        set_llama_index_adapter(None)
