"""Global pytest fixtures for deterministic, offline tests.

Configures LlamaIndex mocks, provides ChatMessage factory, LangGraph in-memory
checkpointer, and a deterministic LangChain LLM for agent tests.
"""

import sys
from pathlib import Path

import pytest

# LangChain deterministic LLM (for agent/graph tests)
from langchain_core.language_models.fake import FakeListLLM

# LangGraph in-memory checkpointer
from langgraph.checkpoint.memory import InMemorySaver

# LlamaIndex deterministic mocks
from llama_index.core import Document, MockEmbedding, Settings, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MockLLM

# Ensure project root is importable as a package root (for `import src`)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load shared fixtures (e.g., supervisor_stream_shim) across the test suite
pytest_plugins = ("tests.shared_fixtures",)


@pytest.fixture(scope="session", autouse=True)
def mock_llamaindex_settings():
    """Configure deterministic, offline LlamaIndex defaults for tests.

    - MockLLM prevents network calls and produces stable outputs
    - MockEmbedding provides deterministic embeddings
    """
    Settings.llm = MockLLM(max_tokens=256)
    # Use typical embedding dimension; adjust per test as needed
    Settings.embed_model = MockEmbedding(embed_dim=1024)
    return None


@pytest.fixture
def chat_message_factory():
    """Factory to create LlamaIndex ChatMessage objects."""

    def _make(role: str, content: str) -> ChatMessage:
        return ChatMessage(role=role, content=content)

    return _make


@pytest.fixture
def inmemory_checkpointer():
    """Provide LangGraph in-memory saver for thread-level persistence tests."""
    return InMemorySaver()


@pytest.fixture
def fake_lc_llm():
    """Provide a deterministic LangChain LLM for agent/graph tests."""
    return FakeListLLM(responses=["ok", "next"])  # override responses in test if needed


@pytest.fixture
def make_index_from_texts():
    """Factory to build an in-memory VectorStoreIndex from (text, metadata) pairs.

    Usage:
        index = make_index_from_texts([
            ("doc one text", {"id": 1}),
            ("doc two text", {"id": 2}),
        ])
    """

    def _build(pairs: list[tuple[str, dict]] | list[str]) -> VectorStoreIndex:  # type: ignore[override]
        docs: list[Document] = []
        if pairs and isinstance(pairs[0], tuple):  # type: ignore[index]
            for text, meta in pairs:  # type: ignore[assignment]
                docs.append(Document(text=text, metadata=meta))
        else:
            for text in pairs:  # type: ignore[assignment]
                docs.append(Document(text=text))
        return VectorStoreIndex.from_documents(docs)

    return _build


@pytest.fixture
def system_settings():
    """Provide system-tier settings fixture for system tests.

    Uses production-like defaults via tests fixtures. Keeps tests offline.
    """
    try:
        from tests.fixtures.test_settings import create_system_settings

        return create_system_settings()
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture
def integration_settings():
    """Provide integration-tier settings fixture for integration tests."""
    try:
        from tests.fixtures.test_settings import create_integration_settings

        s = create_integration_settings(
            embedding={"model_name": "sentence-transformers/all-MiniLM-L6-v2"}
        )
        # Provide convenience alias expected by validation tests
        from contextlib import suppress

        with suppress(Exception):
            s.embedding_dimension = s.embedding.dimension
        return s
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture
def lightweight_embedding_model():
    """Provide a lightweight embedding model stub for integration tests.

    Returns a CPU-only dummy model with an `encode` method.
    """
    class _LightweightModel:
        def encode(self, texts):  # type: ignore[override]
            return [[0.0] * 8 for _ in texts]

    return _LightweightModel()
