"""Shared pytest fixtures and configuration for DocMind AI test suite.

This module provides common fixtures, configuration, and utilities used across
all test modules. It follows 2025 pytest best practices for AI/ML systems with
proper LlamaIndex MockEmbedding/MockLLM usage and tiered testing strategy.

Testing Strategy:
- Unit Tests: Fast (<5s), CPU-only, use MockEmbedding/MagicMock LLM
- Integration Tests: Moderate speed, lightweight models (all-MiniLM-L6-v2 80MB)
- GPU Smoke Tests: Optional manual validation outside CI
"""

import asyncio
import contextlib
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from llama_index.core import Document
from llama_index.core.graph_stores import SimplePropertyGraphStore

# MOCK REDUCTION: Use LlamaIndex MockEmbedding instead of MagicMock

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent))
# Import new centralized settings for new fixtures
from src.config.settings import DocMindSettings as AppSettings

# Configure pytest-asyncio for proper async handling
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for the entire test session."""
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for tests to reduce noise."""
    # Loguru is already configured in __init__.py
    # These libraries will still log to their own loggers, but we can suppress them
    import logging

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def mock_settings() -> AppSettings:
    """Configure test settings for unit tests with the recovered architecture.

    Uses TestDocMindSettings which provides CPU-only, fast, test-optimized defaults
    with proper nested configuration structure.
    """
    # Import here to avoid import issues
    from tests.fixtures.test_settings import TestDocMindSettings

    return TestDocMindSettings(
        debug=True,
        log_level="DEBUG",
        enable_gpu_acceleration=False,  # CPU-only for unit tests
        enable_dspy_optimization=False,  # Disabled for speed
        enable_performance_logging=False,  # Minimal logging
    )


@pytest.fixture(scope="session")
def integration_settings() -> AppSettings:
    """Test settings for integration tests with the recovered architecture.

    Uses IntegrationTestSettings which provides moderate performance settings
    for realistic component integration testing.
    """
    # Import here to avoid import issues
    from tests.fixtures.test_settings import IntegrationTestSettings

    return IntegrationTestSettings(
        debug=False,
        log_level="INFO",
        enable_gpu_acceleration=True,  # Realistic for integration tests
        enable_performance_logging=True,  # Monitor performance
    )


@pytest.fixture(scope="session")
def system_settings() -> AppSettings:
    """Full system test settings with production configuration.

    Uses SystemTestSettings which provides full production defaults
    for comprehensive end-to-end validation.
    """
    # Import here to avoid import issues
    from tests.fixtures.test_settings import SystemTestSettings

    return SystemTestSettings(
        # Uses all production defaults from recovered architecture
        # This validates the actual production configuration
    )


# New centralized settings fixtures using the recovered architecture
@pytest.fixture(scope="session")
def test_settings_factory():
    """Factory for creating test settings with custom overrides."""
    from tests.fixtures.test_settings import create_test_settings

    return create_test_settings


@pytest.fixture(scope="session")
def integration_settings_factory():
    """Factory for creating integration test settings with custom overrides."""
    from tests.fixtures.test_settings import create_integration_settings

    return create_integration_settings


@pytest.fixture(scope="session")
def system_settings_factory():
    """Factory for creating system test settings with custom overrides."""
    from tests.fixtures.test_settings import create_system_settings

    return create_system_settings


@pytest.fixture(scope="session")
def centralized_mock_settings(tmp_path_factory) -> AppSettings:
    """Mock centralized settings for unit tests.

    Uses temporary directories and conservative values for fast, deterministic testing.
    Designed for the new centralized settings system in src/config/settings.py.
    """
    # Create temporary directories for testing
    temp_dir = tmp_path_factory.mktemp("settings_test")

    return AppSettings(
        debug=True,  # Debug mode for testing
        log_level="DEBUG",
        # Use temporary directories
        data_dir=str(temp_dir / "data"),
        cache_dir=str(temp_dir / "cache"),
        log_file=str(temp_dir / "logs" / "test.log"),
        sqlite_db_path=str(temp_dir / "test.db"),
        # Conservative performance settings for testing
        max_memory_gb=2.0,
        max_vram_gb=4.0,
        enable_gpu_acceleration=False,  # Disabled for unit tests
        # Fast timeouts for testing
        agent_decision_timeout=100,  # Fast timeout
        request_timeout_seconds=5.0,
        streaming_delay_seconds=0.001,  # Minimal delay
        # Minimal context for speed
        context_window_size=8192,
        context_buffer_size=8192,
        default_token_limit=8192,
        # Test-optimized batch sizes
        bge_m3_batch_size_gpu=2,
        bge_m3_batch_size_cpu=1,
        default_batch_size=5,
        # Disable expensive operations for unit tests
        use_reranking=False,
        use_sparse_embeddings=False,
        enable_dspy_optimization=False,
        enable_performance_logging=False,
    )


@pytest.fixture(scope="session")
def centralized_integration_settings(tmp_path_factory) -> AppSettings:
    """Integration test settings for the centralized settings system.

    Uses lightweight models and reasonable performance settings for integration tests.
    Balances test speed with realistic configuration testing.
    """
    temp_dir = tmp_path_factory.mktemp("integration_test")

    return AppSettings(
        debug=False,
        log_level="INFO",
        # Test directories
        data_dir=str(temp_dir / "data"),
        cache_dir=str(temp_dir / "cache"),
        # Integration-appropriate settings
        model_name="llama3.2:1b",  # Lightweight model
        llm_backend="ollama",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # 80MB model
        embedding_dimension=384,  # all-MiniLM-L6-v2 dimensions
        # Moderate performance settings
        max_memory_gb=4.0,
        max_vram_gb=8.0,
        enable_gpu_acceleration=True,
        # Reasonable timeouts
        agent_decision_timeout=200,
        max_agent_retries=1,  # Fewer retries for speed
        # Enable key features for integration testing
        enable_multi_agent=True,
        use_reranking=True,
        use_sparse_embeddings=True,
        retrieval_strategy="hybrid",
        # Moderate batch sizes
        bge_m3_batch_size_gpu=6,
        bge_m3_batch_size_cpu=2,
        # Enable caching for integration tests
        enable_document_caching=True,
        enable_performance_logging=True,
    )


@pytest.fixture(scope="session")
def centralized_system_settings() -> AppSettings:
    """Full system test settings for the centralized settings system.

    Production-like configuration for comprehensive system testing.
    Uses default values from the centralized settings system.
    """
    return AppSettings()  # Use all defaults - production configuration


@pytest.fixture
def temp_settings_dirs(tmp_path):
    """Create temporary directories for settings testing.

    Provides clean temporary directories for each test that needs
    to test directory creation and file operations.
    """
    return {
        "data_dir": tmp_path / "test_data",
        "cache_dir": tmp_path / "test_cache",
        "log_dir": tmp_path / "test_logs",
        "db_dir": tmp_path / "test_db",
    }


@pytest.fixture
def centralized_settings_with_temp_dirs(tmp_path):
    """Create settings instance with temporary directories.

    Useful for tests that need to verify directory creation
    and file system integration without affecting real directories.
    """
    return AppSettings(
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        log_file=str(tmp_path / "logs" / "test.log"),
        sqlite_db_path=str(tmp_path / "db" / "test.db"),
    )


@pytest.fixture
def settings_environment_override():
    """Context manager for testing environment variable overrides.

    Usage:
        with settings_environment_override({'DOCMIND_DEBUG': 'true'}):
            settings = AppSettings()
            assert settings.debug is True
    """
    import contextlib
    import os
    from unittest.mock import patch

    @contextlib.contextmanager
    def _override(env_vars):
        with patch.dict(os.environ, env_vars):
            yield

    return _override


@pytest.fixture
def benchmark_settings() -> AppSettings:
    """Settings optimized for performance benchmarking.

    Realistic production settings for accurate performance measurement.
    """
    return AppSettings(
        debug=False,
        log_level="ERROR",  # Minimal logging
        enable_performance_logging=True,
        # Performance-optimized settings
        enable_gpu_acceleration=True,
        vllm_gpu_memory_utilization=0.90,
        enable_kv_cache_optimization=True,
        # Production batch sizes
        bge_m3_batch_size_gpu=12,
        default_batch_size=20,
        # All features enabled for realistic benchmarking
        enable_multi_agent=True,
        use_reranking=True,
        use_sparse_embeddings=True,
        enable_dspy_optimization=True,
        retrieval_strategy="hybrid",
    )


@pytest.fixture
def test_documents() -> list[Document]:
    """Small, consistent test document set for unit and integration tests.

    Provides 5 diverse documents covering DocMind AI functionality.
    Optimized for test speed and deterministic results.
    """
    return [
        Document(
            text="DocMind AI uses SPLADE++ sparse embeddings for efficient retrieval. "
            "This approach enables fast neural lexical matching across documents.",
            metadata={
                "source": "retrieval_guide.pdf",
                "page": 1,
                "chunk_id": "chunk_1",
                "category": "retrieval",
                "word_count": 15,
            },
        ),
        Document(
            text="BGE-Large dense embeddings provide rich semantic understanding. "
            "These 1024-dimensional vectors capture contextual relationships.",
            metadata={
                "source": "embedding_theory.pdf",
                "page": 1,
                "chunk_id": "chunk_2",
                "category": "embeddings",
                "word_count": 12,
            },
        ),
        Document(
            text="ColBERT reranking improves search result relevance significantly. "
            "Late interaction modeling enables precise relevance scoring.",
            metadata={
                "source": "reranking_methods.pdf",
                "page": 2,
                "chunk_id": "chunk_3",
                "category": "reranking",
                "word_count": 13,
            },
        ),
        Document(
            text="Hybrid search combines dense and sparse retrieval methods. "
            "RRF fusion weights multiple retrieval signals optimally.",
            metadata={
                "source": "hybrid_search.pdf",
                "page": 1,
                "chunk_id": "chunk_4",
                "category": "search",
                "word_count": 14,
            },
        ),
        Document(
            text="Multi-agent coordination enables complex query decomposition. "
            "LangGraph supervisor manages agent communication effectively.",
            metadata={
                "source": "agent_architecture.pdf",
                "page": 3,
                "chunk_id": "chunk_5",
                "category": "agents",
                "word_count": 12,
            },
        ),
    ]


@pytest.fixture
def large_document_set() -> list[Document]:
    """Generate larger document set for performance testing."""
    documents = []
    topics = [
        "machine learning algorithms",
        "neural network architectures",
        "natural language processing",
        "computer vision techniques",
        "deep learning frameworks",
        "data preprocessing methods",
        "model evaluation metrics",
        "optimization algorithms",
        "transfer learning approaches",
        "reinforcement learning concepts",
    ]

    for i, topic in enumerate(topics):
        for j in range(10):  # 100 documents total
            doc_id = i * 10 + j
            documents.append(
                Document(
                    text=f"Document {doc_id} discusses {topic} and related concepts. "
                    f"This content covers various aspects of {topic} implementation. "
                    f"Advanced techniques in {topic} are explained with examples.",
                    metadata={
                        "source": f"doc_{doc_id}.pdf",
                        "page": j + 1,
                        "chunk_id": f"chunk_{doc_id}",
                        "topic": topic,
                    },
                )
            )
    return documents


@pytest.fixture
def temp_pdf_file(tmp_path: Path) -> Path:
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test_document.pdf"
    # Create minimal PDF content
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
180
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture(scope="session")
def lightweight_embedding_model():
    """Lightweight embedding model for integration tests.

    Uses all-MiniLM-L6-v2 (80MB) instead of BGE-M3 (1GB).
    Only loads if integration tests are running to avoid unnecessary overhead.
    Session-scoped for performance - expensive model loading happens once.
    """
    # Only import and load if we're running integration tests
    if "integration" in os.environ.get("PYTEST_CURRENT_TEST", "") or any(
        "integration" in arg for arg in sys.argv
    ):
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            pytest.skip("sentence-transformers not available for integration tests")
    else:
        # Return None for unit tests - they should use MockEmbedding
        return None


@pytest.fixture
def mock_sparse_embedding_model() -> MagicMock:
    """Create a mock sparse embedding model for testing."""
    mock_model = MagicMock()
    # Mock sparse embeddings as dict with token indices and values
    mock_model.encode.return_value = [
        {"indices": [1, 5, 10, 23], "values": [0.8, 0.6, 0.4, 0.9]},
        {"indices": [2, 7, 15, 31], "values": [0.7, 0.5, 0.3, 0.8]},
        {"indices": [3, 9, 18, 42], "values": [0.9, 0.7, 0.5, 0.6]},
    ]
    return mock_model


@pytest.fixture
def mock_reranker() -> MagicMock:
    """Create a mock reranker for testing."""
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"index": 0, "score": 0.95},
        {"index": 2, "score": 0.85},
        {"index": 1, "score": 0.75},
    ]
    return mock_reranker


@pytest.fixture
def mock_ai_stack():
    """MOCK REDUCTION: Comprehensive AI stack using LlamaIndex MockEmbedding.

    Replaces multiple MagicMock instances with proper LlamaIndex mock components.
    This single fixture eliminates dozens of individual mocks across the test suite.

    Returns:
        dict: AI stack components with proper mock implementations
    """
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    # Use LlamaIndex MockEmbedding instead of MagicMock
    embed_model = MockEmbedding(embed_dim=1024)

    # Keep minimal MagicMock only for external boundaries
    mock_llm = MagicMock()
    mock_llm.complete.return_value.text = "Mock LLM response"
    mock_llm.stream_complete.return_value = iter(
        [MagicMock(text="Stream "), MagicMock(text="response")]
    )

    return {
        "embed_model": embed_model,
        "llm": mock_llm,
        "embed_dim": 1024,
        "sparse_dim": 30522,  # SPLADE++ vocabulary size
    }


@pytest_asyncio.fixture(loop_scope="function")
async def mock_async_embedding_model() -> AsyncMock:
    """Create an async mock embedding model for testing.

    Returns:
        AsyncMock: Async mock embedding model with proper aembed methods.
    """
    mock_model = AsyncMock()
    mock_model.aembed_documents.return_value = [
        [0.1, 0.2, 0.3] * 341,  # 1024-dim embedding
        [0.4, 0.5, 0.6] * 341,
        [0.7, 0.8, 0.9] * 341,
    ]
    mock_model.aembed_query.return_value = [0.5, 0.5, 0.5] * 341
    return mock_model


@pytest.fixture
def in_memory_graph_store():
    """In-memory graph store for testing property graph functionality.

    Provides SimplePropertyGraphStore for testing graph RAG features
    without external dependencies.
    """
    return SimplePropertyGraphStore()


@pytest_asyncio.fixture(loop_scope="function")
async def mock_async_llm() -> AsyncMock:
    """Create an async mock LLM for testing.

    Returns:
        AsyncMock: Async mock LLM with proper async methods and streaming support.
    """
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = "Mock async LLM response"
    mock_llm.astream.return_value = iter(["Mock ", "async ", "stream"])

    # Add additional async methods commonly used
    mock_llm.acomplete.return_value = AsyncMock(text="Mock async completion")
    mock_llm.astream_complete.return_value = iter(
        [AsyncMock(text="Token1"), AsyncMock(text="Token2")]
    )

    return mock_llm


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Comprehensive mock Qdrant client with proper async methods.

    Provides realistic responses for both sync and async operations.
    Includes proper collection management and search functionality.
    """
    mock_client = MagicMock()

    # Mock search results with realistic structure
    mock_search_results = [
        MagicMock(
            id="doc_1",
            score=0.92,
            payload={
                "text": "DocMind AI uses advanced retrieval techniques",
                "metadata": {"source": "doc1.pdf", "page": 1},
            },
        ),
        MagicMock(
            id="doc_2",
            score=0.87,
            payload={
                "text": "BGE embeddings provide semantic understanding",
                "metadata": {"source": "doc2.pdf", "page": 2},
            },
        ),
        MagicMock(
            id="doc_3",
            score=0.81,
            payload={
                "text": "Hybrid search combines multiple methods",
                "metadata": {"source": "doc3.pdf", "page": 1},
            },
        ),
    ]

    # Configure sync methods
    mock_client.search.return_value = mock_search_results
    mock_client.count.return_value = MagicMock(count=150)
    mock_client.create_collection.return_value = True
    mock_client.delete_collection.return_value = True
    mock_client.collection_exists.return_value = True

    # Configure async methods
    mock_client.asearch = AsyncMock(return_value=mock_search_results)
    mock_client.aupsert = AsyncMock(return_value=True)
    mock_client.acreate_collection = AsyncMock(return_value=True)
    mock_client.adelete_collection = AsyncMock(return_value=True)

    return mock_client


@pytest.fixture
def temp_vector_store(tmp_path: Path) -> Path:
    """Create a temporary directory for vector store testing."""
    vector_store_path = tmp_path / "test_vector_store"
    vector_store_path.mkdir(exist_ok=True)
    return vector_store_path
    # Cleanup is automatic with tmp_path


@pytest.fixture
def sample_query_responses() -> list[dict]:
    """Sample query-response pairs for testing."""
    return [
        {
            "query": "What is SPLADE++ embedding?",
            "expected_keywords": ["sparse", "embedding", "efficient"],
            "context_should_contain": ["SPLADE++", "retrieval"],
        },
        {
            "query": "How does ColBERT reranking work?",
            "expected_keywords": ["reranking", "relevance", "interaction"],
            "context_should_contain": ["ColBERT", "search result"],
        },
        {
            "query": "What are the benefits of hybrid search?",
            "expected_keywords": ["hybrid", "dense", "sparse"],
            "context_should_contain": ["combines", "methods"],
        },
    ]


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """Clean up test artifacts after session.

    Ensures test isolation by cleaning up temporary files,
    cached models, and other test artifacts.
    Session-scoped with autouse to ensure cleanup always happens.
    """
    yield  # Tests run here

    # Cleanup after all tests complete
    import shutil
    import tempfile

    # Clean up any test cache directories
    temp_dirs = [
        Path(tempfile.gettempdir()) / "docmind_test_cache",
        Path(tempfile.gettempdir()) / "sentence_transformers_cache",
        Path(tempfile.gettempdir()) / "pytest_cache",
    ]

    for temp_dir in temp_dirs:
        if temp_dir.exists():
            with contextlib.suppress(PermissionError):
                shutil.rmtree(temp_dir)


# Enhanced pytest configuration with tiered testing strategy
def pytest_configure(config):
    """Configure custom pytest markers for tiered testing strategy.

    Implements ML testing best practices with clear test categories:
    - Unit: Fast, mocked, deterministic
    - Integration: Moderate speed, lightweight models
    - System: Full models, GPU, end-to-end
    """
    # Core test categories (tiered strategy)
    config.addinivalue_line(
        "markers", "unit: Fast unit tests (<5s) using MockEmbedding/MockLLM - CPU only"
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration tests with lightweight models (all-MiniLM-L6-v2)",
    )
    config.addinivalue_line(
        "markers", "system: Full system tests with production models and GPU"
    )

    # Performance and resource markers
    config.addinivalue_line(
        "markers", "performance: Performance benchmarks and memory usage tests"
    )
    config.addinivalue_line(
        "markers", "slow: Long-running tests (deselect with '-m \"not slow\"')"
    )

    # Hardware requirement markers
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU acceleration")
    config.addinivalue_line(
        "markers", "requires_network: Tests requiring network access"
    )
    config.addinivalue_line("markers", "requires_ollama: Tests requiring Ollama server")

    # Feature-specific markers
    config.addinivalue_line("markers", "agents: Multi-agent coordination system tests")
    config.addinivalue_line(
        "markers", "retrieval: Retrieval and search system tests (FEAT-002)"
    )
    config.addinivalue_line(
        "markers", "embeddings: Embedding model and vectorstore tests"
    )
    config.addinivalue_line(
        "markers", "multimodal: CLIP and multimodal functionality tests"
    )

    config.addinivalue_line("markers", "spec: Specification-based tests")


@pytest.fixture(scope="session")
def benchmark_config():
    """Configuration for pytest-benchmark tests.

    Session-scoped since benchmark configuration is constant across tests.
    """
    return {
        "min_rounds": 3,
        "max_time": 1.0,
        "min_time": 0.01,
        "warmup": True,
        "disable_gc": True,
        "timer": time.perf_counter,
    }
