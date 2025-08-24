"""Fixtures for retrieval and search system tests.

Provides specialized fixtures for testing retrieval components:
- BGE-M3 unified embeddings (dense + sparse)
- CrossEncoder reranking with BGE-reranker-v2-m3
- RouterQueryEngine adaptive strategies
- vLLM/Qwen model integration
- PropertyGraphIndex knowledge graphs
- Performance validation utilities

Uses shared fixtures from shared_fixtures.py where possible to maintain DRY principles.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from llama_index.core.schema import NodeWithScore, TextNode

# Import shared utilities to avoid duplication
from tests.shared_fixtures import (
    MockEmbeddingFactory,
    MockRetrieverFactory,
    PerformanceTimer,
    TestDataFactory,
)

# ============================================================================
# SPECIALIZED RETRIEVAL MOCKS
# ============================================================================


@pytest.fixture
def mock_bgem3_model():
    """Mock BGE-M3 unified embedding model with realistic outputs."""
    mock_model = MagicMock()
    mock_model.encode.return_value = {
        "dense_vecs": np.random.rand(2, 1024).astype(np.float32),
        "lexical_weights": MockEmbeddingFactory.create_sparse_embeddings(2),
        "colbert_vecs": [
            np.random.rand(10, 1024).astype(np.float32),
            np.random.rand(12, 1024).astype(np.float32),
        ],
    }
    return mock_model


@pytest.fixture
def mock_clip_embedding():
    """Mock CLIP multimodal embedding model."""
    mock_clip = MagicMock()
    mock_clip.get_text_embedding.return_value = np.random.rand(512).tolist()
    mock_clip.get_query_embedding.return_value = np.random.rand(512).tolist()
    mock_clip.embed_dim = 512
    return mock_clip


@pytest.fixture
def mock_cross_encoder():
    """Mock CrossEncoder for reranking - use shared mock factory."""
    return MockRetrieverFactory.create_mock_reranker()


@pytest.fixture
def mock_vllm_components():
    """Mock vLLM configuration and manager for testing."""
    config = MagicMock()
    config.model_name = "Qwen/Qwen3-4B-Instruct-2507-FP8"
    config.max_model_len = 131072
    config.kv_cache_dtype = "fp8_e5m2"
    config.attention_backend = "FLASHINFER"

    manager = MagicMock()
    manager.config = config
    manager.performance_metrics = {
        "vram_usage_gb": 12.5,
        "decode_throughput_tokens_per_sec": 150.0,
        "meets_targets": True,
    }

    return {"config": config, "manager": manager}


@pytest.fixture
def mock_property_graph():
    """Mock PropertyGraphIndex for knowledge graph testing."""
    mock_graph = MagicMock()

    # Mock graph traversal
    mock_node = TextNode(text="Graph traversal result", id_="graph_node_1")
    mock_graph.traverse_graph = AsyncMock(
        return_value=[NodeWithScore(node=mock_node, score=0.9)]
    )

    # Mock entity extraction
    mock_graph.extract_entities = AsyncMock(
        return_value=[
            {"text": "LlamaIndex", "type": "FRAMEWORK", "confidence": 0.95},
            {"text": "BGE-M3", "type": "MODEL", "confidence": 0.92},
        ]
    )

    # Mock retriever interface
    mock_retriever = MockRetrieverFactory.create_mock_retriever()
    mock_graph.as_retriever.return_value = mock_retriever
    mock_graph.as_query_engine.return_value = MagicMock()

    return mock_graph


@pytest.fixture
def mock_dspy_components():
    """Mock DSPy components for optimization testing."""
    mock = MagicMock()
    mock.settings.configure = MagicMock()

    # Mock prediction with answer attribute
    mock_prediction = MagicMock()
    mock_prediction.answer = "Mock DSPy optimized response"
    mock_prediction.strategy = "factual"

    # Mock optimizer
    mock_optimizer = MagicMock()
    mock_optimizer.compile.return_value = mock_prediction
    mock.teleprompt.MIPROv2.return_value = mock_optimizer

    return mock


@pytest.fixture
def mock_multimodal_utilities():
    """Mock multimodal utilities for CLIP and image processing."""
    return {
        "cross_modal_search": AsyncMock(
            return_value=[
                {"score": 0.95, "image_path": "/path/to/image1.jpg"},
                {"score": 0.87, "image_path": "/path/to/image2.jpg"},
            ]
        ),
        "generate_image_embeddings": AsyncMock(
            return_value=np.random.rand(512).astype(np.float32)
        ),
        "validate_vram_usage": MagicMock(return_value=1.2),
    }


# ============================================================================
# TEST DATA AND UTILITIES
# ============================================================================


@pytest.fixture
def sample_bgem3_embeddings():
    """Sample BGE-M3 unified embeddings for testing."""
    return {
        "dense": np.random.rand(3, 1024).astype(np.float32),
        "sparse": MockEmbeddingFactory.create_sparse_embeddings(3),
        "colbert": [
            np.random.rand(10, 1024).astype(np.float32),
            np.random.rand(12, 1024).astype(np.float32),
            np.random.rand(8, 1024).astype(np.float32),
        ],
    }


@pytest.fixture
def sample_retrieval_documents():
    """Documents specifically for retrieval testing scenarios - use shared factory."""
    return TestDataFactory.create_test_documents(num_docs=5)


@pytest.fixture
def sample_query_scenarios():
    """Test queries for retrieval scenarios - use shared factory."""
    return TestDataFactory.create_query_scenarios()


@pytest.fixture
def mock_vector_index():
    """Mock VectorStoreIndex for testing - simplified version."""
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_query_engine.query.return_value = MagicMock(
        response="Mock response",
        source_nodes=MockRetrieverFactory.create_search_results(2),
    )
    mock_index.as_query_engine.return_value = mock_query_engine
    return mock_index


@pytest.fixture
def mock_hybrid_retriever():
    """Mock hybrid retriever for dense + sparse search - use shared factory."""
    return MockRetrieverFactory.create_mock_retriever()


# Note: MockLLM removed - use MockLLMFactory from shared_fixtures.py


@pytest.fixture
def mock_llm_for_routing():
    """Mock LLM for RouterQueryEngine selection - use shared factory."""
    from tests.shared_fixtures import MockLLMFactory

    return MockLLMFactory.create_llamaindex_mock_llm("hybrid_search")


@pytest.fixture
def mock_retrieval_llm():
    """Mock LLM for retrieval tests - use shared factory."""
    from tests.shared_fixtures import MockLLMFactory

    return MockLLMFactory.create_llamaindex_mock_llm(
        "Entities: LlamaIndex (FRAMEWORK), BGE-M3 (MODEL), RTX 4090 (HARDWARE)"
    )


@pytest.fixture
def performance_test_nodes():
    """Generate nodes for performance testing (20 documents for reranking)."""
    return MockRetrieverFactory.create_search_results(20, base_score=0.8)


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
    """Mock async BGE-M3 embedding model - use shared factory."""
    return MockEmbeddingFactory.create_async_embedding_model()


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
    """Timer utility for performance benchmarking - use shared PerformanceTimer."""
    return PerformanceTimer()


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


# Note: mock_vllm_config and mock_vllm_manager fixtures removed.
# Use mock_vllm_components fixture instead for vLLM testing.
