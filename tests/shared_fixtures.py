"""Shared test fixtures and utilities for DocMind AI test suite.

This module provides reusable mock factories, test data generators, and common
utilities used across all test modules. Follows the DRY principle and modern
pytest patterns.

Key Components:
- Mock factories for embeddings, LLMs, and retrievers
- Test data generators with realistic content
- Performance measurement utilities
- Async test utilities with proper pytest-asyncio support
"""

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import pytest_asyncio
from llama_index.core import Document
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.mock import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode

# ============================================================================
# LLAMAINDEX MOCK FACTORIES (LIBRARY-FIRST APPROACH)
# ============================================================================


class MockEmbeddingFactory:
    """Factory for creating mock embedding models and embeddings.

    Provides consistent mock embedding implementations across test suite.
    Uses library-first approach with LlamaIndex MockEmbedding when possible.
    """

    @staticmethod
    def create_mock_embedding_model():
        """Create a mock embedding model for testing."""
        mock = Mock()
        mock.get_text_embedding.return_value = np.random.rand(1024).tolist()
        mock.get_query_embedding.return_value = np.random.rand(1024).tolist()
        mock.embed_dim = 1024
        return mock

    @staticmethod
    def create_async_embedding_model():
        """Create an async mock embedding model."""
        mock = AsyncMock()
        mock.aget_text_embedding.return_value = np.random.rand(1024).tolist()
        mock.aget_query_embedding.return_value = np.random.rand(1024).tolist()
        mock.embed_dim = 1024
        return mock

    @staticmethod
    def create_dense_embeddings(
        dimension: int = 1024, num_docs: int = 3
    ) -> list[list[float]]:
        """Create mock dense embeddings with specified dimensions."""
        return [np.random.rand(dimension).tolist() for _ in range(num_docs)]

    @staticmethod
    def create_sparse_embeddings(num_docs: int = 3) -> list[dict[int, float]]:
        """Create mock sparse embeddings with token indices."""
        embeddings = []
        for _i in range(num_docs):
            indices = np.random.choice(1000, size=10, replace=False)
            values = np.random.rand(10)
            embeddings.append(
                {
                    int(idx): float(val)
                    for idx, val in zip(indices, values, strict=False)
                }
            )
        return embeddings


class LlamaIndexMockFactory:
    """Factory for creating LlamaIndex-native mock objects.

    ELIMINATES custom mock implementations in favor of library patterns.
    Uses MockEmbedding and MockLLM from LlamaIndex core for consistency.
    """

    @staticmethod
    def create_mock_embedding(embed_dim: int = 1024) -> MockEmbedding:
        """Create LlamaIndex-native MockEmbedding.

        Uses built-in MockEmbedding instead of manual Mock objects.
        Provides consistent dimensions and deterministic behavior.
        """
        return MockEmbedding(embed_dim=embed_dim)

    @staticmethod
    def create_mock_llm(max_tokens: int = 512) -> MockLLM:
        """Create LlamaIndex-native MockLLM.

        Uses built-in MockLLM instead of manual Mock objects.
        Provides realistic completion behavior for testing.
        """
        return MockLLM(max_tokens=max_tokens)


class MockLLMFactory:
    """Factory for creating consistent LLM mocks."""

    @staticmethod
    def create_simple_mock_llm(response: str = "Mock LLM response") -> Mock:
        """Create a simple mock LLM for basic testing."""
        mock = Mock()
        mock.complete.return_value = Mock(text=response)
        mock.invoke.return_value = response
        mock.predict.return_value = response
        return mock

    @staticmethod
    def create_llamaindex_mock_llm(
        response: str = "Mock LlamaIndex LLM response",
    ) -> "MockLlamaIndexLLM":
        """Create a proper LlamaIndex LLM mock."""
        return MockLlamaIndexLLM(response_text=response)


class MockLlamaIndexLLM(LLM):
    """Proper LlamaIndex LLM mock that inherits from LLM base class."""

    response_text: str = "Mock LLM response"

    def __init__(self, response_text: str = "Mock LLM response", **kwargs):
        """Initialize MockLLM with configurable response text."""
        super().__init__(response_text=response_text, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata for mock model."""
        return LLMMetadata(
            context_window=128000, num_output=2048, model_name="mock-llm"
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Complete a prompt with mock response."""
        return CompletionResponse(text=self.response_text)

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Asynchronously complete a prompt with mock response."""
        return self.complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, **kwargs):
        """Stream complete a prompt with mock response."""
        yield CompletionResponse(text=self.response_text)

    async def astream_complete(self, prompt: str, **kwargs):
        """Asynchronously stream complete a prompt with mock response."""
        for response in self.stream_complete(prompt, **kwargs):
            yield response

    def chat(self, messages, **kwargs):
        """Chat with messages using mock response."""
        from llama_index.core.base.llms.types import ChatMessage, ChatResponse

        return ChatResponse(
            message=ChatMessage(role="assistant", content=self.response_text)
        )

    async def achat(self, messages, **kwargs):
        """Asynchronously chat with messages using mock response."""
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages, **kwargs):
        """Stream chat with messages using mock response."""
        from llama_index.core.base.llms.types import ChatMessage, ChatResponse

        yield ChatResponse(
            message=ChatMessage(role="assistant", content=self.response_text)
        )

    async def astream_chat(self, messages, **kwargs):
        """Asynchronously stream chat with messages using mock response."""
        for response in self.stream_chat(messages, **kwargs):
            yield response


class MockRetrieverFactory:
    """Factory for creating mock retrievers and search results."""

    @staticmethod
    def create_search_results(
        num_results: int = 3, base_score: float = 0.9
    ) -> list[NodeWithScore]:
        """Create mock search results with nodes."""
        results = []
        for i in range(num_results):
            node = TextNode(
                text=f"Mock search result {i} with relevant content for testing.",
                id_=f"node_{i}",
            )
            score = base_score - (i * 0.05)  # Decreasing scores
            results.append(NodeWithScore(node=node, score=score))
        return results

    @staticmethod
    def create_mock_retriever() -> Mock:
        """Create a mock retriever with realistic results."""
        mock = Mock()
        mock.retrieve.return_value = MockRetrieverFactory.create_search_results()
        return mock

    @staticmethod
    def create_mock_reranker() -> Mock:
        """Create a mock reranker for search result reranking."""
        mock = Mock()
        mock.rerank.return_value = [
            {"index": 0, "score": 0.95},
            {"index": 2, "score": 0.85},
            {"index": 1, "score": 0.75},
        ]
        return mock


class MockVectorStoreFactory:
    """Factory for creating mock vector stores and clients."""

    @staticmethod
    def create_qdrant_client() -> Mock:
        """Create a comprehensive mock Qdrant client."""
        mock = Mock()

        # Mock search results
        search_results = [
            Mock(
                id=f"doc_{i}",
                score=0.9 - (i * 0.05),
                payload={
                    "text": f"Mock document {i} content",
                    "metadata": {"source": f"doc{i}.pdf", "page": i + 1},
                },
            )
            for i in range(3)
        ]

        # Configure sync methods
        mock.search.return_value = search_results
        mock.count.return_value = Mock(count=100)
        mock.create_collection.return_value = True
        mock.collection_exists.return_value = True

        # Configure async methods
        mock.asearch = AsyncMock(return_value=search_results)
        mock.aupsert = AsyncMock(return_value=True)
        mock.acreate_collection = AsyncMock(return_value=True)

        return mock


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================


class TestDataFactory:
    """Factory for generating realistic test data."""

    @staticmethod
    def create_test_documents(num_docs: int = 5) -> list[Document]:
        """Create realistic test documents for retrieval testing."""
        topics = [
            "machine learning algorithms",
            "neural network architectures",
            "natural language processing",
            "computer vision techniques",
            "deep learning frameworks",
        ]

        documents = []
        for i, topic in enumerate(topics[:num_docs]):
            doc = Document(
                text=f"This document discusses {topic} and related concepts. "
                f"It covers various aspects of {topic} implementation and "
                f"provides examples of {topic} usage in practice.",
                metadata={
                    "source": f"{topic.replace(' ', '_')}.pdf",
                    "page": i + 1,
                    "topic": topic,
                    "doc_id": f"doc_{i}",
                },
            )
            documents.append(doc)

        return documents

    @staticmethod
    def create_query_scenarios() -> list[dict[str, Any]]:
        """Create test query scenarios with expected outcomes."""
        return [
            {
                "query": "What is machine learning?",
                "expected_strategy": "semantic_search",
                "expected_complexity": "simple",
                "expected_docs": 1,
                "keywords": ["machine", "learning", "algorithm"],
            },
            {
                "query": "Compare neural networks and traditional algorithms",
                "expected_strategy": "hybrid_search",
                "expected_complexity": "moderate",
                "expected_docs": 2,
                "keywords": ["neural", "network", "traditional", "comparison"],
            },
            {
                "query": "Explain the architectural differences between CNNs and RNNs, "
                "including their applications and performance characteristics",
                "expected_strategy": "multi_query_search",
                "expected_complexity": "complex",
                "expected_docs": 3,
                "keywords": ["CNN", "RNN", "architecture", "performance"],
            },
        ]


# ============================================================================
# PERFORMANCE UTILITIES
# ============================================================================


class PerformanceTimer:
    """Utility for measuring test performance and timing."""

    def __init__(self):
        """Initialize performance timer with empty tracking dictionaries."""
        self.timings: dict[str, list[float]] = {}
        self.start_times: dict[str, float] = {}

    def start(self, name: str) -> None:
        """Start timing an operation."""
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing and record duration."""
        if name not in self.start_times:
            return 0.0

        duration = time.perf_counter() - self.start_times[name]
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)

        del self.start_times[name]
        return duration

    def get_stats(self, name: str) -> dict[str, float]:
        """Get timing statistics for an operation."""
        if name not in self.timings or not self.timings[name]:
            return {}

        times = self.timings[name]
        return {
            "mean_ms": np.mean(times) * 1000,
            "p50_ms": np.percentile(times, 50) * 1000,
            "p95_ms": np.percentile(times, 95) * 1000,
            "p99_ms": np.percentile(times, 99) * 1000,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000,
            "count": len(times),
        }


# ============================================================================
# ASYNC TEST UTILITIES
# ============================================================================


class AsyncTestUtils:
    """Utilities for async testing."""

    @staticmethod
    async def run_with_timeout(coro, timeout_seconds: float = 5.0):
        """Run async operation with timeout."""
        return await asyncio.wait_for(coro, timeout=timeout_seconds)

    @staticmethod
    def create_mock_async_context_manager():
        """Create a mock async context manager."""
        mock = AsyncMock()
        mock.__aenter__ = AsyncMock(return_value=mock)
        mock.__aexit__ = AsyncMock(return_value=None)
        return mock


# ============================================================================
# SHARED FIXTURES
# ============================================================================


@pytest.fixture
def mock_embedding_model() -> Mock:
    """Create a mock embedding model for testing."""
    return MockEmbeddingFactory.create_mock_embedding_model()


@pytest_asyncio.fixture(loop_scope="function")
async def mock_async_embedding_model() -> AsyncMock:
    """Create an async mock embedding model with proper loop scope."""
    return MockEmbeddingFactory.create_async_embedding_model()


@pytest.fixture
def mock_llm() -> Mock:
    """Create a simple mock LLM."""
    return MockLLMFactory.create_simple_mock_llm()


@pytest.fixture
def mock_llamaindex_llm() -> MockLlamaIndexLLM:
    """Create a proper LlamaIndex LLM mock."""
    return MockLLMFactory.create_llamaindex_mock_llm()


@pytest.fixture
def mock_retriever() -> Mock:
    """Create a mock retriever."""
    return MockRetrieverFactory.create_mock_retriever()


@pytest.fixture
def mock_reranker() -> Mock:
    """Create a mock reranker."""
    return MockRetrieverFactory.create_mock_reranker()


@pytest.fixture
def mock_qdrant_client() -> Mock:
    """Create a mock Qdrant client."""
    return MockVectorStoreFactory.create_qdrant_client()


@pytest.fixture
def realistic_test_documents() -> list[Document]:
    """Create realistic test documents."""
    return TestDataFactory.create_test_documents()


@pytest.fixture
def query_scenarios() -> list[dict[str, Any]]:
    """Create query test scenarios."""
    return TestDataFactory.create_query_scenarios()


@pytest.fixture
def performance_timer() -> PerformanceTimer:
    """Create a performance timer for benchmarking."""
    return PerformanceTimer()


@pytest_asyncio.fixture(loop_scope="function")
async def async_test_utils() -> AsyncTestUtils:
    """Provide async test utilities with proper loop scope."""
    return AsyncTestUtils()


@pytest.fixture
def supervisor_stream_shim() -> Mock:
    """Provide a deterministic supervisor shim with compile().stream().

    Returns an object mimicking the minimal interface used by
    MultiAgentCoordinator:
    - graph.compile(checkpointer=...) -> compiled_graph
    - compiled_graph.stream(initial_state, config=..., stream_mode="values")
      yields a final state dict with a response-like message.
    """

    class _Compiled:
        def stream(self, initial_state, config=None, stream_mode: str | None = None):
            del config, stream_mode
            # Copy initial state and append a deterministic assistant message
            messages = list(initial_state.get("messages", []))
            messages.append(SimpleNamespace(content="Shim: processed successfully"))
            final = dict(initial_state)
            final["messages"] = messages
            final["agent_timings"] = {"router_agent": 0.01}
            yield final

    class _Graph:
        def compile(self, checkpointer=None):  # noqa: D401, ARG002
            return _Compiled()

    return _Graph()


# ============================================================================
# SESSION-SCOPED FIXTURES FOR PERFORMANCE
# ============================================================================


@pytest.fixture(scope="session")
def session_test_documents() -> list[Document]:
    """Session-scoped test documents to avoid recreation.

    Expensive document creation happens once per test session.
    Use this for tests that need consistent document references.
    """
    return TestDataFactory.create_test_documents(num_docs=10)


@pytest.fixture(scope="session")
def session_query_scenarios() -> list[dict[str, Any]]:
    """Session-scoped query scenarios for consistent test data.

    Prevents recreation of test scenarios across multiple tests.
    """
    return TestDataFactory.create_query_scenarios()


@pytest.fixture(scope="session")
def shared_mock_qdrant_client() -> Mock:
    """Session-scoped mock Qdrant client for expensive operations.

    Use this for tests that need consistent vector store behavior
    across multiple test functions without recreation overhead.
    """
    return MockVectorStoreFactory.create_qdrant_client()


# ============================================================================
# TEST-ONLY HELPER: ensure directories for a DocMindSettings instance
# ============================================================================


def ensure_settings_dirs(s) -> None:  # pragma: no cover
    """Create filesystem directories for a DocMindSettings instance (test-only).

    Rationale:
    - Keep src/config/settings.py pure (no side effects).
    - Tests that assert on real file operations can call this explicitly.
    """
    try:
        # Paths are Path objects on DocMindSettings; tolerate strings if present.
        from pathlib import Path

        data_dir = Path(s.data_dir)
        cache_dir = Path(s.cache_dir)
        log_parent = Path(s.log_file).parent
        db_parent = Path(s.sqlite_db_path).parent

        data_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        log_parent.mkdir(parents=True, exist_ok=True)
        db_parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: S110
        # Tests should fail on their own assertions; avoid raising
        # here to keep helper minimal.
        pass
