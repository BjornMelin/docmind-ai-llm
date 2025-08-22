"""Comprehensive fixtures for FEAT-002 Retrieval & Search System tests.

This module provides specialized fixtures and mocks for testing:
- BGE-M3 unified embeddings with FlagEmbedding mocks
- CLIP multimodal embeddings with model mocks
- RouterQueryEngine adaptive strategies
- CrossEncoder reranking with BGE-reranker-v2-m3
- vLLM/Qwen model integration mocks
- PropertyGraphIndex mocks
- DSPy optimization mocks
- Performance validation on RTX 4090 Laptop

All mocks are designed to prevent real model loading while providing realistic outputs.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore, TextNode

# ============================================================================
# COMPREHENSIVE MODEL MOCKS - Prevent Real Model Loading
# ============================================================================


# Mock FlagEmbedding at module level to prevent import/loading issues
@pytest.fixture(autouse=True)
def mock_flag_embedding_imports():
    """Auto-use fixture to mock FlagEmbedding imports before any tests run."""
    with patch.dict(
        "sys.modules",
        {
            "FlagEmbedding": MagicMock(),
            "FlagEmbedding.BGEM3FlagModel": MagicMock(),
        },
    ):
        # Create comprehensive BGEM3FlagModel mock
        mock_bgem3_model = MagicMock()
        mock_bgem3_model.encode.return_value = {
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

        with patch("FlagEmbedding.BGEM3FlagModel", return_value=mock_bgem3_model):
            yield mock_bgem3_model


# Mock CLIP models to prevent loading
@pytest.fixture(autouse=True)
def mock_clip_models():
    """Auto-use fixture to mock CLIP models before any tests run."""
    # Mock CLIP embedding from LlamaIndex
    mock_clip = MagicMock()
    mock_clip.get_text_embedding.return_value = np.random.rand(512).tolist()
    mock_clip.get_query_embedding.return_value = np.random.rand(512).tolist()
    mock_clip.embed_dim = 512

    with patch(
        "llama_index.embeddings.clip.base.ClipEmbedding", return_value=mock_clip
    ):
        yield mock_clip


# Mock sentence-transformers CrossEncoder to prevent loading
@pytest.fixture(autouse=True)
def mock_sentence_transformers():
    """Auto-use fixture to mock sentence-transformers before any tests run."""
    # Mock CrossEncoder class
    mock_cross_encoder = MagicMock()
    mock_cross_encoder.predict.return_value = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    mock_cross_encoder.model = MagicMock()
    mock_cross_encoder.model.half.return_value = None
    mock_cross_encoder.num_labels = 1
    mock_cross_encoder.tokenizer = MagicMock()

    # Mock the module at import level
    mock_sentence_transformers_module = MagicMock()
    mock_sentence_transformers_module.CrossEncoder = MagicMock(
        return_value=mock_cross_encoder
    )

    with (
        patch.dict(
            "sys.modules",
            {
                "sentence_transformers": mock_sentence_transformers_module,
                "sentence_transformers.cross_encoder": MagicMock(),
                "sentence_transformers.cross_encoder.CrossEncoder": MagicMock(
                    return_value=mock_cross_encoder
                ),
            },
        ),
        patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder),
    ):
        yield mock_cross_encoder


# Mock vLLM components to prevent GPU model loading
@pytest.fixture
def mock_vllm_components():
    """Auto-use fixture to mock vLLM components before any tests run."""
    mock_vllm_config = MagicMock()
    mock_vllm_config.model = "Qwen/Qwen3-4B-Instruct-2507-FP8"
    mock_vllm_config.max_model_len = 131072
    mock_vllm_config.kv_cache_dtype = "fp8_e5m2"
    mock_vllm_config.quantization = "fp8"
    mock_vllm_config.gpu_memory_utilization = 0.85

    # Mock VLLM manager
    mock_vllm_manager = AsyncMock()
    mock_vllm_manager.initialize_engine = AsyncMock()
    mock_vllm_manager.generate = AsyncMock(return_value="Mocked LLM response")
    mock_vllm_manager.get_generation_metrics.return_value = {
        "context_tokens": 100000,
        "decode_throughput": 150.0,
        "prefill_throughput": 1200.0,
    }

    # Mock functions that would try to load real models
    patches = {
        "src.core.infrastructure.vllm_config.VLLMConfig": MagicMock(
            return_value=mock_vllm_config
        ),
        "src.core.infrastructure.vllm_config.VLLMManager": MagicMock(
            return_value=mock_vllm_manager
        ),
        "src.core.infrastructure.vllm_config.create_vllm_manager": MagicMock(
            return_value=mock_vllm_manager
        ),
        "src.core.infrastructure.vllm_config.validate_fp8_requirements": MagicMock(
            return_value={
                "cuda_available": True,
                "fp8_support": True,
                "sufficient_vram": True,
                "flashinfer_backend": True,
            }
        ),
    }

    with (
        patch.multiple("builtins", **{"__import__": MagicMock()}),
        patch.dict("sys.modules", {"vllm": MagicMock()}),
    ):
        for target, mock_obj in patches.items():
            with patch(target, mock_obj):
                yield {
                    "config": mock_vllm_config,
                    "manager": mock_vllm_manager,
                    "patches": patches,
                }
                break  # Exit after first iteration


# Mock PropertyGraph components
@pytest.fixture
def mock_property_graph_components():
    """Auto-use fixture to mock PropertyGraph components before any tests run."""
    # Mock PropertyGraphIndex
    mock_property_graph = AsyncMock()
    mock_property_graph.build_graph = AsyncMock()
    mock_property_graph.extract_entities = AsyncMock(
        return_value=[
            {"text": "LlamaIndex", "type": "FRAMEWORK", "confidence": 0.95},
            {"text": "BGE-M3", "type": "MODEL", "confidence": 0.90},
            {"text": "RTX 4090", "type": "HARDWARE", "confidence": 0.88},
        ]
    )
    mock_property_graph.extract_relationships = AsyncMock(
        return_value=[
            {
                "source": "DocMind AI",
                "target": "LlamaIndex",
                "type": "USES",
                "confidence": 0.92,
            },
            {
                "source": "LlamaIndex",
                "target": "RTX 4090",
                "type": "OPTIMIZED_FOR",
                "confidence": 0.85,
            },
        ]
    )
    mock_property_graph.traverse_graph = AsyncMock(
        return_value=[["LlamaIndex", "USES", "BGE-M3"]]
    )
    mock_property_graph.as_retriever.return_value = MagicMock()
    mock_property_graph.as_query_engine.return_value = MagicMock()

    # Mock PropertyGraphConfig
    mock_config = MagicMock()
    mock_config.entities = [
        "FRAMEWORK",
        "LIBRARY",
        "MODEL",
        "HARDWARE",
        "PERSON",
        "ORG",
    ]
    mock_config.relations = [
        "USES",
        "OPTIMIZED_FOR",
        "PART_OF",
        "CREATED_BY",
        "SUPPORTS",
    ]
    mock_config.path_depth = 2
    mock_config.strict_schema = True

    # Apply patches defensively - only patch what exists
    try:
        with patch(
            "src.retrieval.graph.property_graph_config.PropertyGraphConfig",
            MagicMock(return_value=mock_config),
        ):
            pass
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass

    try:
        with patch(
            "src.retrieval.graph.property_graph_config.create_property_graph_index",
            AsyncMock(return_value=mock_property_graph),
        ):
            pass
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass

    return {"property_graph": mock_property_graph, "config": mock_config}


# Mock DSPy components to prevent model loading
@pytest.fixture
def mock_dspy_components():
    """Mock DSPy components - use when DSPy tests are run."""
    dspy_mock = MagicMock()

    # Mock DSPy settings
    dspy_mock.settings = MagicMock()
    dspy_mock.settings.configure = MagicMock()

    # Mock DSPy LM
    dspy_mock.LM = MagicMock()

    # Mock DSPy Example
    example_mock = MagicMock()
    example_mock.with_inputs = MagicMock(return_value=example_mock)
    dspy_mock.Example = MagicMock(return_value=example_mock)

    # Mock DSPy evaluate functions
    dspy_mock.evaluate = MagicMock()
    dspy_mock.evaluate.answer_exact_match = MagicMock()

    # Mock DSPy optimization classes
    mock_optimizer = AsyncMock()
    mock_optimized_model = AsyncMock()
    mock_optimized_model.forward = AsyncMock(
        return_value=MagicMock(answer="Optimized response")
    )
    mock_optimizer.compile = AsyncMock(return_value=mock_optimized_model)

    with patch.dict("sys.modules", {"dspy": dspy_mock}):
        yield dspy_mock


# Mock multimodal utilities
@pytest.fixture
def mock_multimodal_utilities():
    """Mock multimodal utilities - use when multimodal tests are run."""
    patches = {
        "src.utils.multimodal.cross_modal_search": AsyncMock(
            return_value=[
                {"score": 0.95, "image_path": "/path/to/image1.jpg"},
                {"score": 0.87, "image_path": "/path/to/image2.jpg"},
            ]
        ),
        "src.utils.multimodal.generate_image_embeddings": AsyncMock(
            return_value=np.random.rand(512).astype(np.float32)
        ),
        "src.utils.multimodal.validate_vram_usage": MagicMock(return_value=1.2),
        "src.utils.multimodal.validate_end_to_end_pipeline": AsyncMock(
            return_value={
                "visual_similarity": 0.92,
                "entity_relationships": ["LlamaIndex", "USES", "BGE-M3"],
                "final_response": "Mock pipeline response",
            }
        ),
        "src.retrieval.embeddings.clip_config.ClipConfig": MagicMock(),
        "src.retrieval.embeddings.clip_config.create_clip_embedding": MagicMock(),
        "src.retrieval.integration.create_multimodal_index": AsyncMock(),
    }

    return patches


# ============================================================================
# ENHANCED LEGACY FIXTURES WITH REALISTIC DATA
# ============================================================================


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
