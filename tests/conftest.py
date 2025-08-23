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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from llama_index.core import Document
from llama_index.core.graph_stores import SimplePropertyGraphStore

# MockLLM not available in this version, will use MagicMock

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import AppSettings

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
    """Configure mock LlamaIndex components for unit tests.

    Sets up MagicMock LLM and MockEmbedding with proper dimensions to match BGE-M3.
    This ensures fast, deterministic unit tests without external dependencies.
    """
    # Note: We don't set global Settings here as it expects real LLM instances
    # Individual tests can use MockEmbedding and mock LLMs as needed

    return AppSettings(
        default_model="mock-llm",
        dense_embedding_model="mock-embedding",
        sparse_embedding_model="mock-sparse",
        dense_embedding_dimension=1024,
        ollama_base_url="http://mock:11434",
        rerank_enabled=False,  # Disable for unit tests
        enable_sparse_embeddings=False,  # Disable for unit tests
    )


@pytest.fixture(scope="session")
def integration_settings() -> AppSettings:
    """Test settings for integration tests using lightweight models.

    Uses all-MiniLM-L6-v2 (80MB) instead of BGE-M3 (1GB) for faster integration tests.
    Still validates component integration without full model overhead.
    """
    return AppSettings(
        default_model="llama3.2:1b",  # Smallest Ollama model for integration
        dense_embedding_model=(
            "sentence-transformers/all-MiniLM-L6-v2"  # 80MB lightweight
        ),
        sparse_embedding_model="mock-sparse",  # Keep sparse as mock for integration
        dense_embedding_dimension=384,  # all-MiniLM-L6-v2 dimensions
        ollama_base_url="http://localhost:11434",
        rerank_enabled=False,  # Disable expensive operations
        enable_sparse_embeddings=True,  # Test hybrid search logic
    )


@pytest.fixture(scope="session")
def system_settings() -> AppSettings:
    """Full system test settings with real models and GPU.

    Uses production models for full end-to-end validation.
    Only used in system tests marked with @pytest.mark.system.
    """
    return AppSettings(
        default_model="llama3.2:3b",
        dense_embedding_model="BAAI/bge-large-en-v1.5",
        sparse_embedding_model="prithvida/Splade_PP_en_v1",
        dense_embedding_dimension=1024,
        ollama_base_url="http://localhost:11434",
        rerank_enabled=True,
        enable_sparse_embeddings=True,
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


@pytest_asyncio.fixture(loop_scope="function")
async def mock_async_embedding_model() -> AsyncMock:
    """Create an async mock embedding model for testing."""
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
    """Create an async mock LLM for testing."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = "Mock async LLM response"
    mock_llm.astream.return_value = iter(["Mock ", "async ", "stream"])
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


@pytest.fixture(scope="session")
def cleanup_test_artifacts():
    """Clean up test artifacts after session.

    Ensures test isolation by cleaning up temporary files,
    cached models, and other test artifacts.
    """
    yield  # Tests run here

    # Cleanup after all tests complete
    import shutil
    import tempfile

    # Clean up any test cache directories
    temp_dirs = [
        Path(tempfile.gettempdir()) / "docmind_test_cache",
        Path(tempfile.gettempdir()) / "sentence_transformers_cache",
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

    # Legacy markers (for backward compatibility)
    config.addinivalue_line(
        "markers", "feat_002: Legacy marker for FEAT-002 (use 'retrieval' instead)"
    )
    config.addinivalue_line("markers", "spec: Specification-based tests")


@pytest.fixture
def benchmark_config():
    """Configuration for pytest-benchmark tests."""
    return {
        "min_rounds": 3,
        "max_time": 1.0,
        "min_time": 0.01,
        "warmup": True,
        "disable_gc": True,
    }
