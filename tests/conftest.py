"""Shared pytest fixtures and configuration for DocMind AI test suite.

This module provides common fixtures, configuration, and utilities used across
all test modules. It follows 2025 pytest best practices for AI/ML systems.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from llama_index.core import Document

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
def test_settings() -> AppSettings:
    """Provide test settings with safe defaults."""
    return AppSettings(
        backend="ollama",
        default_model="llama3.2:3b",
        dense_embedding_model="BAAI/bge-large-en-v1.5",
        sparse_embedding_model="prithvida/Splade_PP_en_v1",
        dense_embedding_dimension=1024,
        context_size=4096,
        ollama_base_url="http://localhost:11434",
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    """Generate sample documents for testing."""
    return [
        Document(
            text="DocMind AI uses SPLADE++ sparse embeddings for efficient retrieval.",
            metadata={"source": "doc1.pdf", "page": 1, "chunk_id": "chunk_1"},
        ),
        Document(
            text="BGE-Large dense embeddings provide rich semantic understanding.",
            metadata={"source": "doc2.pdf", "page": 1, "chunk_id": "chunk_2"},
        ),
        Document(
            text="ColBERT reranking improves search result relevance significantly.",
            metadata={"source": "doc3.pdf", "page": 2, "chunk_id": "chunk_3"},
        ),
        Document(
            text="Hybrid search combines dense and sparse retrieval methods.",
            metadata={"source": "doc4.pdf", "page": 1, "chunk_id": "chunk_4"},
        ),
        Document(
            text="RRF fusion algorithm weights dense and sparse results optimally.",
            metadata={"source": "doc5.pdf", "page": 3, "chunk_id": "chunk_5"},
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


@pytest.fixture
def mock_embedding_model() -> MagicMock:
    """Create a mock embedding model for testing."""
    mock_model = MagicMock()
    mock_model.embed_documents.return_value = [
        [0.1, 0.2, 0.3] * 341,  # 1024-dim embedding
        [0.4, 0.5, 0.6] * 341,
        [0.7, 0.8, 0.9] * 341,
    ]
    mock_model.embed_query.return_value = [0.5, 0.5, 0.5] * 341
    return mock_model


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
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Mock LLM response"
    mock_llm.stream.return_value = iter(["Mock ", "stream ", "response"])
    return mock_llm


@pytest_asyncio.fixture(loop_scope="function")
async def mock_async_llm() -> AsyncMock:
    """Create an async mock LLM for testing."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = "Mock async LLM response"
    mock_llm.astream.return_value = iter(["Mock ", "async ", "stream"])
    return mock_llm


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Create a mock Qdrant client for testing."""
    mock_client = MagicMock()
    mock_client.search.return_value = [
        MagicMock(id=1, score=0.9, payload={"text": "Document 1"}),
        MagicMock(id=2, score=0.8, payload={"text": "Document 2"}),
        MagicMock(id=3, score=0.7, payload={"text": "Document 3"}),
    ]
    mock_client.count.return_value = MagicMock(count=100)
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


# Performance testing markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: marks tests as fast unit tests")
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line(
        "markers", "requires_network: marks tests that require network access"
    )
    config.addinivalue_line(
        "markers", "feat_002: marks tests for FEAT-002 Retrieval & Search System"
    )


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
