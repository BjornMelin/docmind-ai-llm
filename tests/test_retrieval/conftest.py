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
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.llms.llm import LLM
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


class MockVLLMConfig:
    """Mock VLLMConfig class without spec issues."""

    def __init__(self, **kwargs):
        """Initialize mock VLLMConfig with FP8 optimization settings.

        Args:
            **kwargs: Configuration parameters for vLLM settings including
                model_name, max_model_len, kv_cache_dtype, quantization,
                gpu_memory_utilization, attention_backend, and performance targets.
        """
        self.model_name = kwargs.get("model_name", "Qwen/Qwen3-4B-Instruct-2507-FP8")
        self.max_model_len = kwargs.get("max_model_len", 131072)
        self.kv_cache_dtype = kwargs.get("kv_cache_dtype", "fp8_e5m2")
        self.quantization = kwargs.get("quantization", "fp8")
        self.gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.85)
        self.attention_backend = kwargs.get("attention_backend", "FLASHINFER")
        self.target_decode_tokens_per_sec = kwargs.get(
            "target_decode_tokens_per_sec", 120.0
        )
        self.target_prefill_tokens_per_sec = kwargs.get(
            "target_prefill_tokens_per_sec", 900.0
        )
        self.max_vram_gb = kwargs.get("max_vram_gb", 14.0)
        self.fp8_memory_reduction_target = kwargs.get(
            "fp8_memory_reduction_target", 0.5
        )
        self.enable_performance_validation = kwargs.get(
            "enable_performance_validation", True
        )


class MockVLLMManager:
    """Mock VLLMManager class without spec issues."""

    def __init__(self, config=None):
        """Initialize mock VLLMManager with performance metrics simulation.

        Args:
            config: Optional MockVLLMConfig instance. If None, creates default config.
        """
        self.config = config or MockVLLMConfig()
        self.llm_instance = None
        self.performance_metrics = {
            "vram_usage_gb": 12.5,
            "decode_throughput_tokens_per_sec": 150.0,
            "prefill_throughput_tokens_per_sec": 1200.0,
            "fp8_memory_reduction": 0.55,
            "meets_decode_target": True,
            "meets_prefill_target": True,
            "meets_vram_target": True,
            "meets_memory_reduction_target": True,
        }

    def create_vllm_instance(self):
        """Create mock vLLM LLM instance for testing.

        Returns:
            MockLLM: Mock LLM instance configured for vLLM testing.
        """
        self.llm_instance = MockLLM(response_text="Mock vLLM response")
        return self.llm_instance

    def validate_fp8_performance(self):
        """Validate FP8 quantization performance metrics.

        Returns:
            dict: Performance metrics including VRAM usage, decode/prefill throughput,
                FP8 memory reduction, and target achievement status.
        """
        return self.performance_metrics

    def integrate_with_llamaindex(self):
        """Mock integration with LlamaIndex framework.

        Simulates the integration process between vLLM and LlamaIndex
        for testing purposes.
        """
        pass

    def test_128k_context_support(self):
        """Test 128K context window support with various context sizes.

        Returns:
            dict: Test results containing max_context_supported, supports_128k flag,
                and detailed results for different context sizes with success status,
                latency, and VRAM usage measurements.
        """
        return {
            "max_context_supported": 131072,
            "supports_128k": True,
            "results": [
                {
                    "context_size": 32768,
                    "success": True,
                    "latency": 1.5,
                    "vram_usage": 10.2,
                },
                {
                    "context_size": 65536,
                    "success": True,
                    "latency": 2.1,
                    "vram_usage": 11.8,
                },
                {
                    "context_size": 131072,
                    "success": True,
                    "latency": 3.2,
                    "vram_usage": 13.5,
                },
            ],
        }


# Mock vLLM components to prevent GPU model loading
@pytest.fixture
def mock_vllm_components():
    """Auto-use fixture to mock vLLM components before any tests run."""
    mock_vllm_config = MockVLLMConfig()
    mock_vllm_manager = MockVLLMManager(config=mock_vllm_config)

    return {
        "config": mock_vllm_config,
        "manager": mock_vllm_manager,
    }


class MockPropertyGraphIndex:
    """Mock PropertyGraphIndex with proper method signatures."""

    def __init__(self, documents=None, kg_extractors=None, **kwargs):
        """Initialize mock PropertyGraphIndex for knowledge graph testing.

        Args:
            documents: Optional list of documents to index. Defaults to empty list.
            kg_extractors: Optional knowledge graph extractors. Defaults to empty list.
            **kwargs: Additional configuration parameters including schema and
                path_depth.
        """
        self.documents = documents or []
        self.kg_extractors = kg_extractors or []
        self.schema = kwargs.get("schema", {})
        self.path_depth = kwargs.get("path_depth", 2)

    async def extract_entities(self, document):
        """Extract entities from a document using mock knowledge graph extraction.

        Args:
            document: Document to extract entities from.

        Returns:
            list: List of extracted entities with text, type, and confidence scores.
                Mock entities include LlamaIndex (FRAMEWORK), BGE-M3 (MODEL),
                and RTX 4090 (HARDWARE).
        """
        return [
            {"text": "LlamaIndex", "type": "FRAMEWORK", "confidence": 0.95},
            {"text": "BGE-M3", "type": "MODEL", "confidence": 0.92},
            {"text": "RTX 4090", "type": "HARDWARE", "confidence": 0.98},
        ]

    async def extract_relationships(self, document):
        """Extract relationships between entities from a document.

        Args:
            document: Document to extract relationships from.

        Returns:
            list: List of relationships with source, target, type, and confidence.
                Mock relationships include LlamaIndex USES BGE-M3 and
                LlamaIndex OPTIMIZED_FOR RTX 4090.
        """
        return [
            {
                "source": "LlamaIndex",
                "target": "BGE-M3",
                "type": "USES",
                "confidence": 0.88,
            },
            {
                "source": "LlamaIndex",
                "target": "RTX 4090",
                "type": "OPTIMIZED_FOR",
                "confidence": 0.85,
            },
        ]

    async def traverse_graph(self, query, max_depth=2, timeout=3.0):
        """Traverse the knowledge graph to find relevant nodes.

        Args:
            query: Query to search for in the graph.
            max_depth: Maximum traversal depth. Defaults to 2.
            timeout: Query timeout in seconds. Defaults to 3.0.

        Returns:
            list[NodeWithScore]: List of graph nodes with relevance scores.
        """
        # Mock graph traversal results
        mock_node = TextNode(text="Graph traversal result", id_="graph_node_1")
        return [NodeWithScore(node=mock_node, score=0.9)]

    def as_retriever(self, **kwargs):
        """Create a mock retriever interface for the property graph.

        Args:
            **kwargs: Optional retriever configuration parameters.

        Returns:
            MagicMock: Mock retriever that returns NodeWithScore objects
                with graph retrieval results.
        """
        mock_retriever = MagicMock()
        mock_node = TextNode(text="Graph retrieval result", id_="graph_retriever_1")
        mock_retriever.retrieve.return_value = [
            NodeWithScore(node=mock_node, score=0.85)
        ]
        return mock_retriever

    def as_query_engine(self, **kwargs):
        """Create a mock query engine from the property graph.

        Args:
            **kwargs: Additional configuration options for the query engine.

        Returns:
            MagicMock: Mock query engine for testing purposes.
        """
        return MagicMock()


# Mock PropertyGraph components
@pytest.fixture
def mock_property_graph_components():
    """Auto-use fixture to mock PropertyGraph components before any tests run."""
    return {
        "property_graph": MockPropertyGraphIndex(),
        "config": MockVLLMConfig(),
    }


class MockDSPyPrediction:
    """Mock DSPy prediction with proper attributes."""

    def __init__(self, answer="Mock DSPy answer", **kwargs):
        """Initialize MockDSPyPrediction with answer and attributes.

        Args:
            answer: The predicted answer text.
            **kwargs: Additional attributes to set on the prediction object.
        """
        self.answer = answer
        self.strategy = kwargs.get("strategy", "factual")
        self.search_query = kwargs.get("search_query", "mock query")
        # Add any other attributes that tests might expect
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockDSPyExample:
    """Mock DSPy Example with proper interface."""

    def __init__(self, question="", answer="", **kwargs):
        """Initialize MockDSPyExample with question and answer.

        Args:
            question: The example question.
            answer: The example answer.
            **kwargs: Additional attributes to set on the example object.
        """
        self.question = question
        self.answer = answer
        for key, value in kwargs.items():
            setattr(self, key, value)

    def with_inputs(self, *inputs):
        """Set inputs for the DSPy example (mock implementation).

        Args:
            *inputs: Variable number of input arguments.

        Returns:
            MockDSPyExample: Returns self for method chaining.
        """
        return self


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
    dspy_mock.Example = MockDSPyExample

    # Mock DSPy prediction
    dspy_mock.Prediction = MockDSPyPrediction

    # Mock DSPy evaluate functions
    dspy_mock.evaluate = MagicMock()
    dspy_mock.evaluate.answer_exact_match = MagicMock()

    # Mock DSPy optimization classes with proper interfaces
    mock_optimizer = MagicMock()
    mock_optimized_model = MagicMock()

    def mock_forward(question=None, **kwargs):
        return MockDSPyPrediction(
            answer="Optimized DSPy response",
            strategy="factual",
            search_query=question or "default query",
        )

    mock_optimized_model.side_effect = mock_forward
    mock_optimizer.compile.return_value = mock_optimized_model

    # Mock DSPy teleprompt modules
    dspy_mock.teleprompt = MagicMock()
    dspy_mock.teleprompt.MIPROv2 = MagicMock(return_value=mock_optimizer)
    dspy_mock.teleprompt.BootstrapFewShot = MagicMock(return_value=mock_optimizer)
    dspy_mock.teleprompt.BootstrapFewShotWithRandomSearch = MagicMock(
        return_value=mock_optimizer
    )

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


class MockLLM(LLM):
    """Proper LlamaIndex LLM mock that inherits from LLM base class."""

    response_text: str = "Mock LLM response"

    def __init__(self, response_text: str = "Mock LLM response", **kwargs):
        """Initialize MockLLM with configurable response text.

        Args:
            response_text: The text to return in all LLM responses.
            **kwargs: Additional arguments passed to LLM base class.
        """
        super().__init__(**kwargs)
        self.response_text = response_text

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata for mock model.

        Returns:
            LLMMetadata: Mock metadata with 128K context window.
        """
        return LLMMetadata(
            context_window=128000, num_output=2048, model_name="mock-llm"
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Complete a prompt with mock response.

        Args:
            prompt: Input prompt to complete.
            **kwargs: Additional completion parameters.

        Returns:
            CompletionResponse: Mock completion with configured response text.
        """
        return CompletionResponse(
            text=self.response_text, additional_kwargs={"tool_choice": "hybrid_search"}
        )

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        """Asynchronously complete a prompt with mock response.

        Args:
            prompt: Input prompt to complete.
            **kwargs: Additional completion parameters.

        Returns:
            CompletionResponse: Mock completion with configured response text.
        """
        return self.complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, **kwargs):
        """Stream complete a prompt with mock response.

        Args:
            prompt: Input prompt to complete.
            **kwargs: Additional completion parameters.

        Yields:
            CompletionResponse: Mock streaming completion responses.
        """
        # Return a mock stream
        yield CompletionResponse(text=self.response_text)

    def chat(self, messages, **kwargs):
        """Chat with messages using mock response.

        Args:
            messages: Chat messages to respond to.
            **kwargs: Additional chat parameters.

        Returns:
            ChatResponse: Mock chat response with configured text.
        """
        from llama_index.core.base.llms.types import ChatMessage, ChatResponse

        return ChatResponse(
            message=ChatMessage(role="assistant", content=self.response_text)
        )

    async def achat(self, messages, **kwargs):
        """Asynchronously chat with messages using mock response.

        Args:
            messages: Chat messages to respond to.
            **kwargs: Additional chat parameters.

        Returns:
            ChatResponse: Mock chat response with configured text.
        """
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages, **kwargs):
        """Stream chat with messages using mock response.

        Args:
            messages: Chat messages to respond to.
            **kwargs: Additional chat parameters.

        Yields:
            ChatResponse: Mock streaming chat responses.
        """
        from llama_index.core.base.llms.types import ChatMessage, ChatResponse

        yield ChatResponse(
            message=ChatMessage(role="assistant", content=self.response_text)
        )

    async def astream_chat(self, messages, **kwargs):
        """Asynchronously stream chat with messages using mock response.

        Args:
            messages: Chat messages to respond to.
            **kwargs: Additional chat parameters.

        Yields:
            ChatResponse: Mock streaming chat responses.
        """
        for response in self.stream_chat(messages, **kwargs):
            yield response

    async def astream_complete(self, prompt: str, **kwargs):
        """Asynchronously stream complete a prompt with mock response.

        Args:
            prompt: Input prompt to complete.
            **kwargs: Additional completion parameters.

        Yields:
            CompletionResponse: Mock streaming completion responses.
        """
        for response in self.stream_complete(prompt, **kwargs):
            yield response


@pytest.fixture
def mock_llm_for_routing():
    """Mock LLM specifically for RouterQueryEngine selection."""
    return MockLLM(response_text="hybrid_search")


@pytest.fixture
def mock_llm():
    """Proper mock LLM for property graph and other tests."""
    return MockLLM(
        response_text=(
            "Entities: LlamaIndex (FRAMEWORK), BGE-M3 (MODEL), RTX 4090 (HARDWARE)"
        )
    )


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


@pytest.fixture
def mock_vllm_config():
    """Mock VLLMConfig for FP8 tests."""
    return MockVLLMConfig()


@pytest.fixture
def mock_vllm_manager():
    """Mock VLLMManager for FP8 tests."""
    return MockVLLMManager()
