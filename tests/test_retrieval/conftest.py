"""Fixtures for FEAT-002 Retrieval & Search System tests.

This module provides specialized fixtures for testing:
- BGE-M3 unified embeddings
- RouterQueryEngine adaptive strategies
- CrossEncoder reranking with BGE-reranker-v2-m3
- Performance validation on RTX 4090 Laptop
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore, TextNode


# Mock imports to avoid dependency issues in tests
@pytest.fixture
def mock_bgem3_flag_model():
    """Mock BGE-M3 FlagEmbedding model."""
    mock_model = MagicMock()

    # Mock unified embeddings output
    mock_model.encode.return_value = {
        "dense_vecs": np.random.rand(2, 1024).astype(np.float32),
        "lexical_weights": [
            {1: 0.8, 5: 0.6, 10: 0.4, 23: 0.9},  # Token indices to weights
            {2: 0.7, 7: 0.5, 15: 0.3, 31: 0.8},
        ],
        "colbert_vecs": [
            np.random.rand(10, 1024).astype(np.float32),  # Multi-vector embeddings
            np.random.rand(12, 1024).astype(np.float32),
        ],
    }

    return mock_model


@pytest.fixture
def mock_cross_encoder():
    """Mock sentence-transformers CrossEncoder."""
    mock_encoder = MagicMock()

    # Mock relevance scores for query-document pairs
    mock_encoder.predict.return_value = np.array([0.95, 0.85, 0.75, 0.65, 0.55])

    # Mock model properties
    mock_encoder.model = MagicMock()
    mock_encoder.model.half.return_value = None

    return mock_encoder


@pytest.fixture
def sample_bgem3_embeddings():
    """Sample BGE-M3 unified embeddings for testing."""
    return {
        "dense": np.random.rand(3, 1024).astype(np.float32),
        "sparse": [
            {1: 0.8, 5: 0.6, 10: 0.4, 23: 0.9},
            {2: 0.7, 7: 0.5, 15: 0.3, 31: 0.8},
            {3: 0.9, 9: 0.7, 18: 0.5, 42: 0.6},
        ],
        "colbert": [
            np.random.rand(10, 1024).astype(np.float32),
            np.random.rand(12, 1024).astype(np.float32),
            np.random.rand(8, 1024).astype(np.float32),
        ],
    }


@pytest.fixture
def sample_test_documents():
    """Documents specifically for retrieval testing scenarios."""
    return [
        Document(
            text=(
                "Quantum computing applications in machine learning enable "
                "exponential speedups for certain optimization problems and "
                "pattern recognition tasks."
            ),
            metadata={
                "source": "quantum_ml.pdf",
                "page": 1,
                "topic": "quantum_computing",
            },
        ),
        Document(
            text=(
                "BGE-M3 unified embeddings combine dense semantic vectors "
                "with sparse lexical representations in a single "
                "1024-dimensional model architecture."
            ),
            metadata={"source": "embeddings.pdf", "page": 2, "topic": "embeddings"},
        ),
        Document(
            text=(
                "CrossEncoder reranking using BGE-reranker-v2-m3 provides "
                "superior relevance scoring compared to bi-encoder approaches."
            ),
            metadata={"source": "reranking.pdf", "page": 1, "topic": "reranking"},
        ),
        Document(
            text=(
                "Hybrid retrieval systems leverage both dense semantic "
                "similarity and sparse keyword matching for comprehensive "
                "document search."
            ),
            metadata={
                "source": "hybrid_search.pdf",
                "page": 3,
                "topic": "hybrid_search",
            },
        ),
        Document(
            text=(
                "Multi-query search decomposes complex questions into "
                "sub-queries enabling better coverage of information needs."
            ),
            metadata={"source": "multi_query.pdf", "page": 1, "topic": "multi_query"},
        ),
    ]


@pytest.fixture
def sample_query_scenarios():
    """Test queries mapping to Gherkin scenarios."""
    return [
        {
            "query": "explain quantum computing applications",
            "expected_strategy": "multi_query_search",
            "expected_complexity": "analytical/complex",
            "expected_docs": 3,
        },
        {
            "query": "BGE-M3 embeddings",
            "expected_strategy": "semantic_search",
            "expected_complexity": "semantic/simple",
            "expected_docs": 1,
        },
        {
            "query": "how does reranking improve search results",
            "expected_strategy": "hybrid_search",
            "expected_complexity": "hybrid/moderate",
            "expected_docs": 2,
        },
    ]


@pytest.fixture
def mock_vector_index():
    """Mock VectorStoreIndex for testing."""
    mock_index = MagicMock()

    # Mock query engine
    mock_query_engine = MagicMock()
    mock_query_engine.query.return_value = MagicMock(
        response="Mock response",
        source_nodes=[
            NodeWithScore(
                node=TextNode(text="Mock document 1", id_="node_1"), score=0.9
            ),
            NodeWithScore(
                node=TextNode(text="Mock document 2", id_="node_2"), score=0.8
            ),
        ],
    )

    mock_index.as_query_engine.return_value = mock_query_engine
    return mock_index


@pytest.fixture
def mock_hybrid_retriever():
    """Mock hybrid retriever for dense + sparse search."""
    mock_retriever = MagicMock()

    # Mock retrieval results
    mock_retriever.retrieve.return_value = [
        NodeWithScore(
            node=TextNode(text="Hybrid search result 1", id_="hybrid_1"), score=0.95
        ),
        NodeWithScore(
            node=TextNode(text="Hybrid search result 2", id_="hybrid_2"), score=0.85
        ),
        NodeWithScore(
            node=TextNode(text="Hybrid search result 3", id_="hybrid_3"), score=0.75
        ),
    ]

    return mock_retriever


@pytest.fixture
def mock_llm_for_routing():
    """Mock LLM specifically for RouterQueryEngine selection."""
    mock_llm = MagicMock()

    # Mock strategy selection responses
    mock_llm.complete.return_value = MagicMock(
        text="hybrid_search",  # Default to hybrid strategy
        additional_kwargs={"tool_choice": "hybrid_search"},
    )

    mock_llm.predict.return_value = "hybrid_search"

    return mock_llm


@pytest.fixture
def performance_test_nodes():
    """Generate nodes for performance testing (20 documents for reranking)."""
    nodes = []
    for i in range(20):
        node = NodeWithScore(
            node=TextNode(
                text=(
                    f"Performance test document {i} containing relevant "
                    f"information for benchmarking reranking latency and "
                    f"throughput."
                ),
                id_=f"perf_node_{i}",
            ),
            score=0.8 - (i * 0.02),  # Decreasing scores
        )
        nodes.append(node)

    return nodes


@pytest.fixture
def rtx_4090_performance_targets():
    """Performance targets for RTX 4090 Laptop validation."""
    return {
        "bgem3_embedding_latency_ms": 50,
        "reranking_latency_ms": 100,
        "query_p95_latency_s": 2.0,
        "vram_usage_gb": 14.0,
        "min_retrieval_accuracy": 0.8,
        "strategy_selection_latency_ms": 50,
    }


@pytest.fixture
def async_bgem3_embedding():
    """Mock async BGE-M3 embedding model."""
    mock_embedding = AsyncMock()

    # Mock async embedding generation
    mock_embedding._aget_query_embedding.return_value = np.random.rand(1024).tolist()  # pylint: disable=protected-access
    mock_embedding.get_unified_embeddings.return_value = {
        "dense": np.random.rand(1, 1024).astype(np.float32),
        "sparse": [{1: 0.8, 5: 0.6, 10: 0.4}],
    }

    return mock_embedding


@pytest.fixture
def cuda_available():
    """Mock CUDA availability for GPU tests."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
    ):
        yield True


@pytest.fixture
def benchmark_timer():
    """Timer utility for performance benchmarking."""

    class BenchmarkTimer:
        """Timer for benchmarking test performance."""

        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.latencies = []

        def start(self):
            """Start timing."""
            self.start_time = time.perf_counter()

        def stop(self):
            """Stop timing and record latency."""
            self.end_time = time.perf_counter()
            if self.start_time:
                latency_ms = (self.end_time - self.start_time) * 1000
                self.latencies.append(latency_ms)
                return latency_ms
            return None

        def get_stats(self):
            """Get timing statistics."""
            if not self.latencies:
                return {}
            return {
                "mean_ms": np.mean(self.latencies),
                "p50_ms": np.percentile(self.latencies, 50),
                "p95_ms": np.percentile(self.latencies, 95),
                "p99_ms": np.percentile(self.latencies, 99),
                "min_ms": np.min(self.latencies),
                "max_ms": np.max(self.latencies),
                "count": len(self.latencies),
            }

    return BenchmarkTimer()


@pytest.fixture
def mock_memory_monitor():
    """Mock GPU memory monitoring."""

    class MockMemoryMonitor:
        """Mock memory monitor for testing."""

        def __init__(self):
            self.current_usage_gb = 8.5  # Simulate current usage
            self.peak_usage_gb = 12.3

        def get_memory_usage(self):
            """Get current memory usage."""
            return {
                "used_gb": self.current_usage_gb,
                "total_gb": 16.0,
                "free_gb": 16.0 - self.current_usage_gb,
            }

        def track_peak_usage(self):
            """Track peak memory usage."""
            return self.peak_usage_gb

    return MockMemoryMonitor()


# Test configuration helpers
@pytest.fixture
def test_config_bgem3():
    """BGE-M3 test configuration."""
    return {
        "model_name": "BAAI/bge-m3",
        "max_length": 8192,
        "use_fp16": True,
        "batch_size": 12,
        "device": "cuda",
        "normalize_embeddings": True,
    }


@pytest.fixture
def test_config_reranker():
    """CrossEncoder reranker test configuration."""
    return {
        "model_name": "BAAI/bge-reranker-v2-m3",
        "top_n": 5,
        "device": "cuda",
        "use_fp16": True,
        "batch_size": 16,
        "max_length": 512,
        "normalize_scores": True,
    }
