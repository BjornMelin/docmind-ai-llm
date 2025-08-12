"""Comprehensive pytest fixtures for DocMind AI test suite.

This module provides reusable fixtures for all test scenarios across document ingestion,
orchestration agents, and embedding/vectorstore clusters. Fixtures follow 2025 pytest
best practices with proper async handling, resource management, and deterministic behavior.

Usage:
    # In test files
    import pytest
    from library_research.fixtures import *  # Import all fixtures

    def test_example(mock_embedding_model, sample_documents):
        # Use fixtures in tests
        pass
"""

import asyncio
import base64
import io
import os
import random
import sys
import tempfile
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from llama_index.core import Document
from PIL import Image

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test data seeds for reproducibility
RANDOM_SEED = 42
TEST_UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-123456789abc")

# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "random_seed": RANDOM_SEED,
        "timeout": 30.0,
        "max_retries": 3,
        "batch_size": 16,
        "embedding_dimension": 1024,
        "test_collection_prefix": "test_",
    }


@pytest.fixture
def mock_settings():
    """Create mock settings for different test scenarios."""
    from src.models.core import AppSettings

    return AppSettings(
        backend="ollama",
        default_model="llama3.2:3b",
        dense_embedding_model="BAAI/bge-small-en-v1.5",
        dense_embedding_dimension=384,  # BGE-small dimension
        context_size=4096,
        chunk_size=512,
        chunk_overlap=128,
        retrieval_top_k=10,
        gpu_acceleration=False,  # Default to CPU for tests
        enable_quantization=False,  # Default disabled for tests
        memory_backend="memory",  # Use in-memory for tests
        enable_native_bm25=True,
        rrf_k_param=60,
    )


@pytest.fixture
def gpu_settings(mock_settings):
    """Settings with GPU acceleration enabled."""
    mock_settings.gpu_acceleration = True
    mock_settings.multi_gpu = False
    mock_settings.gpu_device_ids = [0]
    return mock_settings


@pytest.fixture
def multi_gpu_settings(mock_settings):
    """Settings with multi-GPU acceleration enabled."""
    mock_settings.gpu_acceleration = True
    mock_settings.multi_gpu = True
    mock_settings.gpu_device_ids = [0, 1]
    return mock_settings


@pytest.fixture
def quantization_settings(mock_settings):
    """Settings with quantization enabled."""
    mock_settings.enable_quantization = True
    mock_settings.quantization_type = "asymmetric"
    mock_settings.quantization_bits = 8
    return mock_settings


@pytest.fixture
def production_settings(mock_settings):
    """Production-like settings for integration tests."""
    mock_settings.memory_backend = "postgres"
    mock_settings.database_url = "postgresql://test:test@localhost:5432/test"
    mock_settings.enable_quantization = True
    mock_settings.gpu_acceleration = True
    mock_settings.enable_native_bm25 = True
    return mock_settings


# =============================================================================
# DOCUMENT FIXTURES
# =============================================================================


@pytest.fixture
def deterministic_random():
    """Provide deterministic random state for tests."""
    random.seed(RANDOM_SEED)
    return random.Random(RANDOM_SEED)


@pytest.fixture
def sample_documents(deterministic_random) -> list[Document]:
    """Generate sample documents with deterministic content."""
    documents = [
        Document(
            text="DocMind AI uses FastEmbed for efficient local embedding generation.",
            metadata={
                "source": "doc1.pdf",
                "page": 1,
                "chunk_id": "chunk_1",
                "topic": "embeddings",
            },
        ),
        Document(
            text="Qdrant vector database provides native BM25 sparse embeddings support.",
            metadata={
                "source": "doc2.pdf",
                "page": 1,
                "chunk_id": "chunk_2",
                "topic": "vectorstore",
            },
        ),
        Document(
            text="LangGraph enables sophisticated multi-agent orchestration patterns.",
            metadata={
                "source": "doc3.pdf",
                "page": 2,
                "chunk_id": "chunk_3",
                "topic": "agents",
            },
        ),
        Document(
            text="Binary quantization reduces memory usage by up to 70% with minimal accuracy loss.",
            metadata={
                "source": "doc4.pdf",
                "page": 1,
                "chunk_id": "chunk_4",
                "topic": "optimization",
            },
        ),
        Document(
            text="Hybrid search combines dense semantic and sparse keyword retrieval methods.",
            metadata={
                "source": "doc5.pdf",
                "page": 3,
                "chunk_id": "chunk_5",
                "topic": "search",
            },
        ),
    ]

    # Add some randomized documents for variety
    topics = [
        "machine learning",
        "artificial intelligence",
        "data science",
        "neural networks",
    ]
    for i in range(5, 15):
        topic = deterministic_random.choice(topics)
        documents.append(
            Document(
                text=f"Document {i} discusses {topic} concepts and applications. "
                f"Advanced techniques in {topic} are covered with practical examples.",
                metadata={
                    "source": f"doc_{i}.pdf",
                    "page": i % 3 + 1,
                    "chunk_id": f"chunk_{i}",
                    "topic": topic,
                },
            )
        )

    return documents


@pytest.fixture
def large_document_set(deterministic_random) -> list[Document]:
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
        for j in range(20):  # 200 documents total
            doc_id = i * 20 + j
            # Generate varied content lengths
            content_length = deterministic_random.randint(100, 1000)
            base_content = f"Document {doc_id} discusses {topic} and related concepts. "
            full_content = base_content * (content_length // len(base_content))

            documents.append(
                Document(
                    text=full_content,
                    metadata={
                        "source": f"doc_{doc_id}.pdf",
                        "page": j + 1,
                        "chunk_id": f"chunk_{doc_id}",
                        "topic": topic,
                        "content_length": len(full_content),
                    },
                )
            )
    return documents


@pytest.fixture
def multimodal_documents(sample_image_base64) -> list[Document]:
    """Generate documents with multimodal content."""
    return [
        Document(
            text="This document contains an embedded image showing neural network architecture.",
            metadata={
                "source": "multimodal_doc1.pdf",
                "page": 1,
                "chunk_id": "mm_chunk_1",
                "has_images": True,
                "image_base64": sample_image_base64,
                "image_count": 1,
            },
        ),
        Document(
            text="Visualization of data flow in machine learning pipelines with diagrams.",
            metadata={
                "source": "multimodal_doc2.pdf",
                "page": 2,
                "chunk_id": "mm_chunk_2",
                "has_images": True,
                "image_base64": sample_image_base64,
                "image_count": 2,
            },
        ),
    ]


# =============================================================================
# EMBEDDING FIXTURES
# =============================================================================


@pytest.fixture
def mock_embedding_model() -> MagicMock:
    """Create a mock dense embedding model."""
    mock_model = MagicMock()

    # Simulate BGE-small embeddings (384 dimensions)
    def mock_embed_documents(texts):
        return [[0.1, 0.2, 0.3] * 128 for _ in texts]  # 384-dim embeddings

    def mock_embed_query(text):
        return [0.5, 0.5, 0.5] * 128  # 384-dim embedding

    mock_model.embed_documents.side_effect = mock_embed_documents
    mock_model.embed_query.side_effect = mock_embed_query
    mock_model.get_text_embedding.side_effect = mock_embed_query
    mock_model.model_name = "BAAI/bge-small-en-v1.5"
    mock_model.embedding_dimension = 384

    return mock_model


@pytest_asyncio.fixture
async def mock_async_embedding_model() -> AsyncMock:
    """Create an async mock embedding model."""
    mock_model = AsyncMock()

    async def mock_aembed_documents(texts):
        # Simulate async processing delay
        await asyncio.sleep(0.01)
        return [[0.1, 0.2, 0.3] * 128 for _ in texts]  # 384-dim embeddings

    async def mock_aembed_query(text):
        await asyncio.sleep(0.01)
        return [0.5, 0.5, 0.5] * 128  # 384-dim embedding

    mock_model.aembed_documents.side_effect = mock_aembed_documents
    mock_model.aembed_query.side_effect = mock_aembed_query
    mock_model.model_name = "BAAI/bge-small-en-v1.5"
    mock_model.embedding_dimension = 384

    return mock_model


@pytest.fixture
def mock_sparse_embedding_model() -> MagicMock:
    """Create a mock sparse embedding model."""
    mock_model = MagicMock()

    def mock_encode(texts):
        # Simulate sparse embeddings with token indices and values
        results = []
        for i, text in enumerate(texts):
            # Deterministic sparse embedding based on text hash
            text_hash = hash(text) % 1000
            indices = [text_hash % 100, (text_hash + 1) % 100, (text_hash + 2) % 100]
            values = [0.8, 0.6, 0.4]
            results.append({"indices": indices, "values": values})
        return results

    mock_model.encode.side_effect = mock_encode
    mock_model.model_name = "Qdrant/bm25"
    return mock_model


@pytest.fixture
def mock_fastembed_model() -> MagicMock:
    """Create a mock FastEmbed model."""
    mock_model = MagicMock()

    # FastEmbed-specific methods
    mock_model.embed.return_value = [[0.1, 0.2, 0.3] * 128]  # 384-dim
    mock_model.passage_embed.return_value = [[0.1, 0.2, 0.3] * 128]
    mock_model.query_embed.return_value = [[0.5, 0.5, 0.5] * 128]

    # Model info
    mock_model.model_name = "BAAI/bge-small-en-v1.5"
    mock_model.dim = 384
    mock_model.max_length = 512
    mock_model.device = "cpu"

    return mock_model


@pytest.fixture
def mock_multi_gpu_embedding_model() -> MagicMock:
    """Create a mock multi-GPU embedding model."""
    mock_model = MagicMock()

    # Multi-GPU specific attributes
    mock_model.device_ids = [0, 1]
    mock_model.multi_gpu = True
    mock_model.batch_size_per_gpu = 512

    def mock_embed_batch_gpu(texts, device_ids=None):
        # Simulate faster processing with multiple GPUs
        time.sleep(0.001)  # Faster than single GPU
        return [[0.1, 0.2, 0.3] * 128 for _ in texts]

    mock_model.embed_documents.side_effect = mock_embed_batch_gpu
    mock_model.get_gpu_stats.return_value = {
        "cuda_available": True,
        "device_count": 2,
        "devices": [
            {"device_id": 0, "name": "GPU_0", "memory_allocated": 1024},
            {"device_id": 1, "name": "GPU_1", "memory_allocated": 1024},
        ],
    }

    return mock_model


# =============================================================================
# VECTOR STORE FIXTURES
# =============================================================================


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Create a mock Qdrant client."""
    mock_client = MagicMock()

    # Mock search results
    mock_search_results = [
        MagicMock(
            id=1, score=0.95, payload={"text": "Relevant document 1", "metadata": {}}
        ),
        MagicMock(
            id=2, score=0.88, payload={"text": "Relevant document 2", "metadata": {}}
        ),
        MagicMock(
            id=3, score=0.82, payload={"text": "Relevant document 3", "metadata": {}}
        ),
    ]

    mock_client.search.return_value = mock_search_results
    mock_client.count.return_value = MagicMock(count=100)
    mock_client.get_collection.return_value = MagicMock(
        config=MagicMock(
            params=MagicMock(
                vectors=MagicMock(size=384, distance="Cosine"),
                sparse_vectors={"text-sparse": MagicMock(modifier="IDF")},
            )
        )
    )

    # Mock collection operations
    mock_client.create_collection.return_value = True
    mock_client.delete_collection.return_value = True
    mock_client.upsert.return_value = MagicMock(status="completed")

    return mock_client


@pytest_asyncio.fixture
async def mock_async_qdrant_client() -> AsyncMock:
    """Create an async mock Qdrant client."""
    mock_client = AsyncMock()

    # Mock async search results
    mock_search_results = [
        MagicMock(id=1, score=0.95, payload={"text": "Async relevant document 1"}),
        MagicMock(id=2, score=0.88, payload={"text": "Async relevant document 2"}),
        MagicMock(id=3, score=0.82, payload={"text": "Async relevant document 3"}),
    ]

    mock_client.search.return_value = mock_search_results
    mock_client.count.return_value = MagicMock(count=100)

    # Mock async collection operations
    async def mock_create_collection(*args, **kwargs):
        await asyncio.sleep(0.01)  # Simulate async delay
        return True

    async def mock_upsert(*args, **kwargs):
        await asyncio.sleep(0.01)
        return MagicMock(status="completed")

    mock_client.create_collection.side_effect = mock_create_collection
    mock_client.upsert.side_effect = mock_upsert

    return mock_client


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock vector store."""
    mock_store = MagicMock()

    # Vector store operations
    mock_store.add_documents.return_value = ["id_1", "id_2", "id_3"]
    mock_store.similarity_search.return_value = [
        MagicMock(page_content="Document 1", metadata={"score": 0.95}),
        MagicMock(page_content="Document 2", metadata={"score": 0.88}),
    ]

    # Hybrid search capabilities
    mock_store.enable_hybrid = True
    mock_store.fastembed_sparse_model = "Qdrant/bm25"
    mock_store.batch_size = 16

    return mock_store


@pytest.fixture
def quantized_vector_store(mock_vector_store) -> MagicMock:
    """Create a mock quantized vector store."""
    mock_store = mock_vector_store

    # Quantization-specific attributes
    mock_store.quantization_enabled = True
    mock_store.quantization_config = {
        "type": "asymmetric",
        "bits_storage": 8,
        "bits_query": 8,
        "compression_ratio": 0.3,  # 70% compression
    }

    # Mock memory usage reduction
    mock_store.get_memory_usage.return_value = {
        "total_memory_mb": 100,  # Reduced from ~333MB baseline
        "compression_ratio": 0.3,
        "quantization_enabled": True,
    }

    return mock_store


# =============================================================================
# AGENT FIXTURES
# =============================================================================


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM."""
    mock_llm = MagicMock()

    # Synchronous methods
    mock_llm.invoke.return_value = "Mock LLM response for the given query."
    mock_llm.stream.return_value = iter(["Mock ", "streaming ", "response."])
    mock_llm.predict.return_value = "Mock prediction result."

    # Model info
    mock_llm.model_name = "llama3.2:3b"
    mock_llm.temperature = 0.1
    mock_llm.max_tokens = 2048

    return mock_llm


@pytest_asyncio.fixture
async def mock_async_llm() -> AsyncMock:
    """Create an async mock LLM."""
    mock_llm = AsyncMock()

    # Asynchronous methods
    async def mock_ainvoke(prompt):
        await asyncio.sleep(0.01)  # Simulate processing delay
        return f"Async mock response to: {prompt[:50]}..."

    async def mock_astream(prompt):
        await asyncio.sleep(0.01)
        for chunk in ["Async ", "streaming ", "response."]:
            yield chunk

    mock_llm.ainvoke.side_effect = mock_ainvoke
    mock_llm.astream.side_effect = mock_astream
    mock_llm.model_name = "llama3.2:3b"

    return mock_llm


@pytest.fixture
def mock_agent_system() -> MagicMock:
    """Create a mock agent system."""
    mock_system = MagicMock()

    # Agent system operations
    mock_system.process_query.return_value = {
        "response": "Mock agent response",
        "sources": ["doc1.pdf", "doc2.pdf"],
        "agent_used": "document_specialist",
    }

    # Agent configuration
    mock_system.agents = {
        "document_specialist": MagicMock(),
        "knowledge_specialist": MagicMock(),
        "multimodal_specialist": MagicMock(),
    }

    # Memory and state
    mock_system.get_conversation_history.return_value = [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is artificial intelligence..."},
    ]

    return mock_system


@pytest_asyncio.fixture
async def mock_async_agent_system() -> AsyncMock:
    """Create an async mock agent system."""
    mock_system = AsyncMock()

    # Async agent operations
    async def mock_aprocess_query(query, thread_id=None):
        await asyncio.sleep(0.02)  # Simulate agent processing
        return {
            "response": f"Async mock response to: {query}",
            "sources": ["doc1.pdf", "doc2.pdf"],
            "agent_used": "document_specialist",
            "thread_id": thread_id,
        }

    async def mock_astream_query(query, thread_id=None):
        await asyncio.sleep(0.01)
        chunks = ["Processing ", "your ", "query...", " Complete!"]
        for chunk in chunks:
            yield {"content": chunk, "type": "text"}

    mock_system.aprocess_query.side_effect = mock_aprocess_query
    mock_system.astream_query.side_effect = mock_astream_query

    return mock_system


@pytest.fixture
def mock_langgraph_supervisor() -> MagicMock:
    """Create a mock LangGraph supervisor."""
    mock_supervisor = MagicMock()

    # LangGraph-specific methods
    mock_supervisor.compile.return_value = mock_supervisor
    mock_supervisor.stream.return_value = iter(
        [
            {"agent": "document_specialist", "output": "Processing document..."},
            {"agent": "supervisor", "output": "Task completed successfully."},
        ]
    )

    # State management
    mock_supervisor.get_state.return_value = {
        "messages": [],
        "current_agent": "supervisor",
        "task_status": "completed",
    }

    # Configuration
    mock_supervisor.checkpointer = MagicMock()
    mock_supervisor.agents = ["document_specialist", "knowledge_specialist"]

    return mock_supervisor


@pytest.fixture
def mock_memory_backend() -> MagicMock:
    """Create a mock memory backend."""
    mock_backend = MagicMock()

    # Memory operations
    mock_backend.save_checkpoint.return_value = "checkpoint_id_123"
    mock_backend.load_checkpoint.return_value = {
        "state": {"messages": [], "current_step": 0},
        "metadata": {"timestamp": "2025-08-12T10:00:00Z"},
    }

    # Backend info
    mock_backend.backend_type = "memory"
    mock_backend.connection_status = "connected"

    return mock_backend


@pytest_asyncio.fixture
async def mock_postgres_backend() -> AsyncMock:
    """Create a mock PostgreSQL memory backend."""
    mock_backend = AsyncMock()

    # Async memory operations
    async def mock_save_checkpoint(state, metadata=None):
        await asyncio.sleep(0.005)  # Simulate DB write
        return f"pg_checkpoint_{uuid.uuid4().hex[:8]}"

    async def mock_load_checkpoint(checkpoint_id):
        await asyncio.sleep(0.005)  # Simulate DB read
        return {
            "state": {"messages": [], "current_step": 0},
            "metadata": {"checkpoint_id": checkpoint_id},
        }

    mock_backend.save_checkpoint.side_effect = mock_save_checkpoint
    mock_backend.load_checkpoint.side_effect = mock_load_checkpoint
    mock_backend.backend_type = "postgres"
    mock_backend.connection_status = "connected"

    return mock_backend


# =============================================================================
# FILE AND MEDIA FIXTURES
# =============================================================================


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_pdf_file(temp_directory) -> Path:
    """Create a sample PDF file for testing."""
    pdf_path = temp_directory / "test_document.pdf"

    # Minimal PDF content
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
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""

    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture
def sample_image_base64() -> str:
    """Create a sample base64-encoded image."""
    # Create a small test image
    img = Image.new("RGB", (10, 10), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def sample_image_file(temp_directory, sample_image_base64) -> Path:
    """Create a sample image file."""
    image_path = temp_directory / "test_image.png"
    image_data = base64.b64decode(sample_image_base64)
    image_path.write_bytes(image_data)
    return image_path


@pytest.fixture
def sample_text_file(temp_directory) -> Path:
    """Create a sample text file."""
    text_path = temp_directory / "test_document.txt"
    text_content = """This is a sample text document for testing.

It contains multiple paragraphs and various content types.
The document discusses artificial intelligence, machine learning,
and natural language processing concepts.

This content is used for testing document loading and processing
functionalities in the DocMind AI system."""

    text_path.write_text(text_content)
    return text_path


# =============================================================================
# DATABASE AND CONTAINER FIXTURES
# =============================================================================


@pytest.fixture
def test_collection_name() -> str:
    """Generate a unique test collection name."""
    return f"test_collection_{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def qdrant_test_container():
    """Start a Qdrant container for integration tests."""
    try:
        from testcontainers.general import DockerContainer

        with DockerContainer("qdrant/qdrant:v1.15.1") as container:
            container.with_exposed_ports(6333)
            container.start()

            # Wait for container to be ready
            await asyncio.sleep(2)

            connection_url = f"http://localhost:{container.get_exposed_port(6333)}"
            yield connection_url

    except ImportError:
        # Fallback when testcontainers not available
        pytest.skip("testcontainers not available")


@pytest_asyncio.fixture
async def postgres_test_container():
    """Start a PostgreSQL container for integration tests."""
    try:
        from testcontainers.postgres import PostgresContainer

        with PostgresContainer("postgres:15") as postgres:
            postgres.with_env("POSTGRES_DB", "test")
            postgres.with_env("POSTGRES_USER", "test")
            postgres.with_env("POSTGRES_PASSWORD", "test")
            postgres.start()

            # Wait for database to be ready
            await asyncio.sleep(3)

            connection_url = postgres.get_connection_url()
            yield connection_url

    except ImportError:
        pytest.skip("testcontainers not available")


@pytest_asyncio.fixture
async def redis_test_container():
    """Start a Redis container for integration tests."""
    try:
        from testcontainers.redis import RedisContainer

        with RedisContainer("redis:7") as redis:
            redis.start()

            await asyncio.sleep(1)

            connection_url = redis.get_connection_url()
            yield connection_url

    except ImportError:
        pytest.skip("testcontainers not available")


# =============================================================================
# PERFORMANCE AND BENCHMARKING FIXTURES
# =============================================================================


@pytest.fixture
def benchmark_config():
    """Configuration for pytest-benchmark tests."""
    return {
        "min_rounds": 3,
        "max_time": 1.0,
        "min_time": 0.01,
        "warmup": True,
        "disable_gc": True,
        "sort": "mean",
    }


@pytest.fixture
def performance_metrics():
    """Track performance metrics during tests."""
    metrics = {
        "start_time": time.perf_counter(),
        "memory_usage": [],
        "gpu_utilization": [],
        "throughput": [],
    }

    yield metrics

    # Calculate final metrics
    metrics["total_time"] = time.perf_counter() - metrics["start_time"]
    if metrics["throughput"]:
        metrics["avg_throughput"] = sum(metrics["throughput"]) / len(
            metrics["throughput"]
        )


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    try:
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        yield process

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Alert if memory increase is significant (>100MB)
        if memory_increase > 100 * 1024 * 1024:
            pytest.warns(
                UserWarning, f"Memory increase: {memory_increase / 1024 / 1024:.2f}MB"
            )

    except ImportError:
        yield None


# =============================================================================
# INTEGRATION AND E2E FIXTURES
# =============================================================================


@pytest_asyncio.fixture
async def full_system_integration(
    mock_embedding_model,
    mock_qdrant_client,
    mock_llm,
    mock_settings,
) -> AsyncMock:
    """Create a full system integration for E2E tests."""
    mock_system = AsyncMock()

    # Wire up all components
    mock_system.embedding_model = mock_embedding_model
    mock_system.vector_store_client = mock_qdrant_client
    mock_system.llm = mock_llm
    mock_system.settings = mock_settings

    # Full pipeline methods
    async def mock_full_pipeline(query, documents=None):
        await asyncio.sleep(0.05)  # Simulate full processing

        return {
            "query": query,
            "response": f"Full system response to: {query}",
            "sources": [
                doc.metadata.get("source", "unknown") for doc in (documents or [])
            ],
            "embeddings_generated": len(documents) if documents else 0,
            "search_results": 3,
            "processing_time_ms": 50,
        }

    mock_system.process_full_pipeline.side_effect = mock_full_pipeline

    return mock_system


@pytest.fixture
def integration_test_data():
    """Provide integration test scenarios and expected results."""
    return [
        {
            "scenario": "document_ingestion_to_search",
            "input_documents": [
                "Machine learning algorithms improve over time with more data.",
                "Neural networks are inspired by biological neural systems.",
            ],
            "test_query": "How do machine learning algorithms work?",
            "expected_keywords": ["machine learning", "algorithms", "neural"],
            "min_relevance_score": 0.7,
        },
        {
            "scenario": "multimodal_processing",
            "input_documents": ["Image showing deep learning architecture diagram."],
            "test_query": "Show me neural network architectures.",
            "expected_keywords": ["neural", "architecture", "deep learning"],
            "has_images": True,
            "min_relevance_score": 0.6,
        },
        {
            "scenario": "agent_coordination",
            "input_documents": [
                "Agent coordination patterns in multi-agent systems.",
                "Knowledge graphs represent entity relationships.",
            ],
            "test_query": "How do agents coordinate in multi-agent systems?",
            "expected_agent": "knowledge_specialist",
            "min_relevance_score": 0.8,
        },
    ]


# =============================================================================
# PROPERTY-BASED TESTING FIXTURES
# =============================================================================


@pytest.fixture
def hypothesis_settings():
    """Configure Hypothesis for property-based testing."""
    from hypothesis import Verbosity, settings

    return settings(
        max_examples=50,  # Reasonable number for CI
        verbosity=Verbosity.verbose,
        deadline=5000,  # 5 second deadline
        suppress_health_check=[],
    )


# =============================================================================
# CLEANUP AND TEARDOWN FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_test_environment():
    """Automatically clean up test environment after each test."""
    # Setup
    initial_state = {
        "env_vars": dict(os.environ),
        "random_state": random.getstate(),
    }

    yield  # Run the test

    # Teardown
    # Restore environment variables
    os.environ.clear()
    os.environ.update(initial_state["env_vars"])

    # Reset random state
    random.setstate(initial_state["random_state"])


@pytest_asyncio.fixture(autouse=True)
async def async_cleanup():
    """Clean up async resources after each test."""
    yield  # Run the test

    # Clean up any remaining async tasks
    tasks = [task for task in asyncio.all_tasks() if not task.done()]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def deterministic_uuid():
    """Generate deterministic UUIDs for tests."""

    def _generate_uuid(name: str) -> str:
        return str(uuid.uuid5(TEST_UUID_NAMESPACE, name))

    return _generate_uuid


@pytest.fixture
def mock_time():
    """Mock time functions for deterministic testing."""
    with pytest.mock.patch("time.time", return_value=1692700800.0):  # Fixed timestamp
        with pytest.mock.patch("time.perf_counter", return_value=100.0):
            yield


@pytest.fixture
def capture_logs():
    """Capture logs during test execution."""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)

    # Add handler to loguru (would need loguru interception setup)
    # For now, capture standard logging
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    yield log_capture

    root_logger.removeHandler(handler)
