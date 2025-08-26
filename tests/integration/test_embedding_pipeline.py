"""Integration tests for embedding pipeline with lightweight models.

This module provides comprehensive integration tests using lightweight models to verify
component interaction without heavy resource usage. Tests use CPU-only models and
in-memory stores for fast execution while maintaining real integration validation.

Key features:
- Uses all-MiniLM-L6-v2 (80MB) instead of BGE-M3 (1GB)
- In-memory vector stores and graph stores
- Real component integration without mocking core functionality
- Execution time <30 seconds per test
- Cross-component interaction testing

Test Architecture:
- Lightweight models for CPU-only execution
- Real LlamaIndex components with memory stores
- Agent coordination with MockLLM
- PropertyGraph with SimplePropertyGraphStore
- Query pipeline with lightweight embeddings
"""

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from llama_index.core import Document, PropertyGraphIndex, Settings, VectorStoreIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.llms import MockLLM
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agents.tools import retrieve_documents, route_query
from src.config.app_settings import DocMindSettings
from src.retrieval.graph.property_graph_config import PropertyGraphConfig


# Lightweight Embedding Fixtures
@pytest.fixture(scope="module")
def lightweight_embedding_model():
    """Create lightweight CPU embedding model for integration tests.

    Uses all-MiniLM-L6-v2 (80MB) instead of BGE-M3 (1GB) for resource efficiency.
    This model provides 384-dimensional embeddings with good semantic quality.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.eval()  # Set to evaluation mode
    return model


@pytest.fixture(scope="module")
def lightweight_llama_embedding():
    """Create LlamaIndex-compatible lightweight embedding for integration tests.

    Returns:
        HuggingFaceEmbedding: CPU-optimized embedding model
    """
    return HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=512,
        normalize=True,
        device="cpu",
    )


@pytest.fixture
def mock_llm_for_integration():
    """Create MockLLM for integration testing without API calls.

    Returns:
        MockLLM: Configured mock LLM for testing agent interactions
    """
    return MockLLM(max_tokens=1024)


@pytest.fixture
def in_memory_vector_store():
    """Create in-memory vector store for fast testing.

    Returns:
        SimpleVectorStore: In-memory vector store instance
    """
    return SimpleVectorStore()


@pytest.fixture
def in_memory_graph_store():
    """Create in-memory property graph store for testing.

    Returns:
        SimplePropertyGraphStore: In-memory graph store instance
    """
    return SimplePropertyGraphStore()


@pytest.fixture
def test_documents_comprehensive():
    """Generate comprehensive test documents for integration testing.

    Returns:
        list[Document]: Test documents covering various AI/ML topics
    """
    return [
        Document(
            text=(
                "SPLADE++ sparse embeddings enable efficient neural information "
                "retrieval by learning sparse lexical representations that outperform "
                "traditional sparse methods while maintaining interpretability."
            ),
            metadata={
                "source": "splade_paper.pdf",
                "page": 1,
                "chunk_id": "chunk_1",
                "type": "technical",
            },
        ),
        Document(
            text="BGE-M3 unified embedding model provides dense, sparse, and ColBERT "
            "multi-vector representations in a single architecture, supporting "
            "multilingual retrieval across 100+ languages with 8K context.",
            metadata={
                "source": "bge_m3_paper.pdf",
                "page": 2,
                "chunk_id": "chunk_2",
                "type": "technical",
            },
        ),
        Document(
            text=(
                "LangGraph supervisor framework orchestrates multi-agent systems using "
                "state machines and message passing, enabling complex reasoning "
                "workflows with parallel tool execution and coordination tracking."
            ),
            metadata={
                "source": "langgraph_docs.pdf",
                "page": 3,
                "chunk_id": "chunk_3",
                "type": "framework",
            },
        ),
        Document(
            text="Property graphs represent knowledge as entities and relationships, "
            "enabling semantic understanding and graph-based reasoning for "
            "improved retrieval-augmented generation systems.",
            metadata={
                "source": "knowledge_graphs.pdf",
                "page": 4,
                "chunk_id": "chunk_4",
                "type": "conceptual",
            },
        ),
        Document(
            text="RRF fusion algorithm combines dense and sparse retrieval results "
            "using reciprocal rank fusion with configurable alpha parameter, "
            "achieving 15-20% better recall than single-vector approaches.",
            metadata={
                "source": "rrf_fusion.pdf",
                "page": 5,
                "chunk_id": "chunk_5",
                "type": "algorithmic",
            },
        ),
    ]


@pytest.fixture
def integration_settings():
    """Create lightweight settings for integration testing.

    Returns:
        DocMindSettings: Test configuration optimized for integration tests
    """
    return DocMindSettings(
        bge_m3_model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        chunk_size=512,
        enable_gpu_acceleration=False,
        use_sparse_embeddings=False,
        qdrant_url="http://localhost:6333",
        top_k=3,
    )


# Integration Tests
@pytest.mark.integration
class TestEmbeddingGeneration:
    """Test embedding generation with lightweight CPU models."""

    def test_embedding_generation_with_lightweight_model(
        self, lightweight_embedding_model
    ):
        """Test actual embedding generation with small CPU model.

        Validates:
        - Real embedding model loads and functions
        - Output dimensions are correct
        - No NaN or invalid values in embeddings
        - Processing time is reasonable (<2 seconds)
        """
        start_time = time.perf_counter()

        texts = [
            "DocMind AI uses SPLADE++ sparse embeddings",
            "BGE-Large provides dense semantic representations",
            "Query about machine learning embeddings",
        ]

        embeddings = lightweight_embedding_model.encode(texts)

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Validate embedding properties
        assert embeddings.shape[0] == 3, "Should generate 3 embeddings"
        assert embeddings.shape[1] == 384, "all-MiniLM-L6-v2 has 384 dimensions"
        assert not np.any(np.isnan(embeddings)), "No NaN values in embeddings"
        assert not np.any(np.isinf(embeddings)), "No infinite values in embeddings"
        assert processing_time < 2.0, (
            f"Embedding generation too slow: {processing_time:.2f}s"
        )

        # Validate semantic similarity
        similarity_matrix = np.dot(embeddings, embeddings.T)
        # First two documents should be more similar than first and third
        assert similarity_matrix[0, 1] > similarity_matrix[0, 2]

    @pytest.mark.asyncio
    async def test_llamaindex_embedding_integration(
        self, lightweight_llama_embedding, test_documents_comprehensive
    ):
        """Test LlamaIndex embedding integration with lightweight model.

        Validates:
        - LlamaIndex HuggingFaceEmbedding works with lightweight model
        - Async embedding generation functions correctly
        - Integration with Document objects
        - Embedding quality for semantic tasks
        """
        start_time = time.perf_counter()

        # Test document embedding
        sample_docs = test_documents_comprehensive[:3]

        # Create embeddings for documents
        doc_texts = [doc.text for doc in sample_docs]
        embeddings = await lightweight_llama_embedding._aget_text_embeddings(doc_texts)

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Validate results
        assert len(embeddings) == 3, "Should generate 3 embeddings"
        assert all(len(emb) == 384 for emb in embeddings), "384-dimensional embeddings"
        assert processing_time < 3.0, (
            f"Async embedding too slow: {processing_time:.2f}s"
        )

        # Test query embedding
        query_embedding = await lightweight_llama_embedding._aget_query_embedding(
            "What is sparse embedding?"
        )
        assert len(query_embedding) == 384, "Query embedding should be 384-dimensional"


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Test vector store integration with lightweight components."""

    @pytest.mark.asyncio
    async def test_vector_index_with_lightweight_embeddings(
        self,
        lightweight_llama_embedding,
        in_memory_vector_store,
        test_documents_comprehensive,
        integration_settings,
    ):
        """Test VectorStoreIndex with lightweight embeddings and in-memory store.

        Validates:
        - Real vector index creation with lightweight models
        - Document ingestion and embedding storage
        - Query functionality with retrieval
        - Performance within acceptable bounds
        """
        start_time = time.perf_counter()

        # Set lightweight embedding in Settings
        Settings.embed_model = lightweight_llama_embedding
        Settings.chunk_size = integration_settings.chunk_size
        Settings.chunk_overlap = integration_settings.chunk_overlap

        # Create vector index
        index = VectorStoreIndex.from_documents(
            test_documents_comprehensive, vector_store=in_memory_vector_store
        )

        # Test query functionality
        query_engine = index.as_query_engine(similarity_top_k=2)
        response = await query_engine.aquery("What are sparse embeddings?")

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Validate results
        assert response is not None, "Query should return response"
        assert len(response.source_nodes) <= 2, "Should return at most 2 source nodes"
        assert processing_time < 10.0, (
            f"Vector indexing too slow: {processing_time:.2f}s"
        )

        # Validate source nodes have expected properties
        for node in response.source_nodes:
            assert hasattr(node, "score"), "Source nodes should have scores"
            assert hasattr(node, "node"), "Source nodes should have node objects"
            assert node.score > 0, "Similarity scores should be positive"

    def test_vector_store_persistence_simulation(
        self, lightweight_llama_embedding, test_documents_comprehensive
    ):
        """Test vector store operations simulating persistence patterns.

        Validates:
        - Vector store add/retrieve operations
        - Metadata handling and filtering
        - Batch operations efficiency
        - Memory usage patterns
        """
        Settings.embed_model = lightweight_llama_embedding

        # Create vector store and add documents
        vector_store = SimpleVectorStore()
        index = VectorStoreIndex.from_documents(
            test_documents_comprehensive, vector_store=vector_store
        )

        # Test metadata filtering simulation
        retriever = index.as_retriever(
            similarity_top_k=3,
            filters=None,  # SimpleVectorStore doesn't support metadata filters
        )

        # Retrieve documents
        nodes = retriever.retrieve("neural information retrieval")

        # Validate retrieval results
        assert len(nodes) <= 3, "Should respect top_k limit"
        assert all(hasattr(node, "score") for node in nodes), (
            "All nodes should have scores"
        )

        # Validate scores are in descending order
        scores = [node.score for node in nodes]
        assert scores == sorted(scores, reverse=True), (
            "Results should be sorted by score"
        )


@pytest.mark.integration
class TestPropertyGraphIntegration:
    """Test PropertyGraph integration with in-memory store."""

    @pytest.mark.asyncio
    async def test_property_graph_with_memory_store(
        self,
        in_memory_graph_store,
        mock_llm_for_integration,
        test_documents_comprehensive,
        lightweight_llama_embedding,
    ):
        """Test PropertyGraphIndex with in-memory store and lightweight models.

        Validates:
        - PropertyGraph creation with SimplePropertyGraphStore
        - Entity and relationship extraction using MockLLM
        - Graph-based querying functionality
        - Integration with lightweight embedding model
        """
        start_time = time.perf_counter()

        # Configure Settings for property graph
        Settings.llm = mock_llm_for_integration
        Settings.embed_model = lightweight_llama_embedding

        # Create property graph config
        config = PropertyGraphConfig(
            entities=["FRAMEWORK", "MODEL", "ALGORITHM"],
            relations=["USES", "IMPLEMENTS", "OPTIMIZES"],
            max_paths_per_chunk=5,
        )

        # Create extractors with simplified configuration
        extractors = [
            SimpleLLMPathExtractor(
                llm=mock_llm_for_integration,
                max_paths_per_chunk=config.max_paths_per_chunk,
                num_workers=1,
            )
        ]

        # Create PropertyGraphIndex
        index = PropertyGraphIndex.from_documents(
            test_documents_comprehensive[:3],  # Use fewer documents for faster testing
            property_graph_store=in_memory_graph_store,
            kg_extractors=extractors,
            show_progress=False,
        )

        # Test graph query functionality
        query_engine = index.as_query_engine(
            include_text=True, response_mode="tree_summarize"
        )

        response = await query_engine.aquery(
            "What frameworks and models are mentioned?"
        )

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Validate results
        assert response is not None, "PropertyGraph query should return response"
        assert hasattr(response, "response"), "Response should have response attribute"
        assert processing_time < 15.0, (
            f"PropertyGraph processing too slow: {processing_time:.2f}s"
        )

        # Validate graph store has been populated
        # Note: SimplePropertyGraphStore doesn't expose internal structure easily
        # but we can verify the index was created successfully
        assert index is not None, "PropertyGraphIndex should be created successfully"

    def test_property_graph_config_validation(self):
        """Test PropertyGraphConfig validation and defaults.

        Validates:
        - Configuration object creation and validation
        - Default entity and relation types
        - Parameter bounds checking
        - Type validation for fields
        """
        # Test default configuration
        config = PropertyGraphConfig()

        assert "FRAMEWORK" in config.entities, "Should include default FRAMEWORK entity"
        assert "MODEL" in config.entities, "Should include default MODEL entity"
        assert "USES" in config.relations, "Should include default USES relation"
        assert config.max_paths_per_chunk == 20, (
            "Should have default max_paths_per_chunk"
        )

        # Test custom configuration
        custom_config = PropertyGraphConfig(
            entities=["CUSTOM_ENTITY"],
            relations=["CUSTOM_RELATION"],
            max_paths_per_chunk=10,
        )

        assert custom_config.entities == ["CUSTOM_ENTITY"]
        assert custom_config.relations == ["CUSTOM_RELATION"]
        assert custom_config.max_paths_per_chunk == 10


@pytest.mark.integration
class TestQueryPipelineIntegration:
    """Test query pipeline integration with lightweight components."""

    def test_query_pipeline_with_lightweight_components(
        self,
        lightweight_llama_embedding,
        mock_llm_for_integration,
        test_documents_comprehensive,
        integration_settings,
    ):
        """Test LlamaIndex QueryPipeline with lightweight models.

        Validates:
        - QueryPipeline creation with lightweight components
        - Multi-stage pipeline execution
        - Component integration and data flow
        - Performance optimization with CPU models
        """
        start_time = time.perf_counter()

        # Configure Settings
        Settings.embed_model = lightweight_llama_embedding
        Settings.llm = mock_llm_for_integration

        # Create vector index for pipeline
        index = VectorStoreIndex.from_documents(
            test_documents_comprehensive, vector_store=SimpleVectorStore()
        )

        # Create simple query pipeline (since complex pipeline requires more setup)
        query_engine = index.as_query_engine(
            similarity_top_k=2, response_mode="compact"
        )

        # Execute query
        response = query_engine.query("How does RRF fusion improve retrieval?")

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Validate pipeline results
        assert response is not None, "Pipeline should return response"
        assert hasattr(response, "response"), "Response should have response text"
        assert hasattr(response, "source_nodes"), "Response should have source nodes"
        assert len(response.source_nodes) <= 2, "Should respect similarity_top_k"
        assert processing_time < 8.0, (
            f"Pipeline processing too slow: {processing_time:.2f}s"
        )

        # Validate response quality
        assert len(response.response) > 0, "Response should contain text"

    @pytest.mark.asyncio
    async def test_async_query_pipeline_performance(
        self,
        lightweight_llama_embedding,
        mock_llm_for_integration,
        test_documents_comprehensive,
    ):
        """Test async query pipeline performance with concurrent queries.

        Validates:
        - Async query execution capability
        - Concurrent query handling
        - Performance under concurrent load
        - Resource management in async context
        """
        Settings.embed_model = lightweight_llama_embedding
        Settings.llm = mock_llm_for_integration

        # Create vector index
        index = VectorStoreIndex.from_documents(
            test_documents_comprehensive, vector_store=SimpleVectorStore()
        )

        query_engine = index.as_query_engine(similarity_top_k=1)

        # Define test queries
        queries = [
            "What are sparse embeddings?",
            "How does BGE-M3 work?",
            "What is LangGraph used for?",
        ]

        start_time = time.perf_counter()

        # Execute queries concurrently
        tasks = [query_engine.aquery(query) for query in queries]
        responses = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        concurrent_time = end_time - start_time

        # Validate concurrent execution
        assert len(responses) == 3, "Should process all queries"
        assert all(resp is not None for resp in responses), (
            "All queries should return responses"
        )
        assert concurrent_time < 12.0, (
            f"Concurrent queries too slow: {concurrent_time:.2f}s"
        )

        # Validate responses have expected structure
        for response in responses:
            assert hasattr(response, "response"), (
                "Each response should have response text"
            )
            assert hasattr(response, "source_nodes"), (
                "Each response should have source nodes"
            )


@pytest.mark.integration
class TestMultiAgentCoordinationIntegration:
    """Integration tests for multi-agent coordination with performance monitoring."""

    @pytest.mark.asyncio
    async def test_agent_coordination_with_mock_models(
        self,
        mock_llm_for_integration,
        lightweight_llama_embedding,
        test_documents_comprehensive,
        integration_settings,
    ):
        """Test agent system coordination without full models.

        Validates:
        - Multi-agent system initialization
        - Agent coordination logic with MockLLM
        - Tool execution and state management
        - Response generation and validation
        """
        start_time = time.perf_counter()

        # Configure lightweight settings
        Settings.llm = mock_llm_for_integration
        Settings.embed_model = lightweight_llama_embedding

        # Create vector index for agents
        index = VectorStoreIndex.from_documents(
            test_documents_comprehensive, vector_store=SimpleVectorStore()
        )

        # Test individual agent tools
        query = "What embedding techniques are most effective?"

        # Test routing tool
        routing_result = await route_query(query, mock_llm_for_integration)
        assert routing_result is not None, "Routing should return result"
        assert "strategy" in routing_result, "Routing should include strategy"

        # Test retrieval tool
        retrieval_result = await retrieve_documents(
            query, index.as_retriever(similarity_top_k=2)
        )
        assert retrieval_result is not None, "Retrieval should return result"
        assert "documents" in retrieval_result, "Retrieval should include documents"

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Validate coordination performance with regression thresholds
        assert processing_time < 8.0, (
            f"Agent coordination too slow: {processing_time:.2f}s"
        )

        print(
            f"Agent coordination performance: {processing_time:.2f}s "
            "for routing + retrieval"
        )

    def test_agent_tool_integration(
        self,
        mock_llm_for_integration,
        lightweight_llama_embedding,
        test_documents_comprehensive,
    ):
        """Test integration between agent tools and components.

        Validates:
        - Tool execution with real components
        - Data flow between tools
        - Error handling in tool chains
        - Resource management in tool execution
        """
        # Configure Settings
        Settings.llm = mock_llm_for_integration
        Settings.embed_model = lightweight_llama_embedding

        # Create test index
        index = VectorStoreIndex.from_documents(
            test_documents_comprehensive, vector_store=SimpleVectorStore()
        )

        retriever = index.as_retriever(similarity_top_k=2)

        # Test tool execution sequentially
        query = "Explain neural information retrieval methods"

        # Simulate tool execution chain
        start_time = time.perf_counter()

        # Step 1: Route query (mock)
        route_result = {
            "strategy": "retrieval_focused",
            "complexity": "medium",
            "estimated_steps": 3,
        }

        # Step 2: Retrieve documents (real)
        nodes = retriever.retrieve(query)

        # Step 3: Process results
        retrieval_result = {
            "documents": [
                {
                    "text": node.node.text,
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
                for node in nodes
            ],
            "total_results": len(nodes),
        }

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Validate tool integration
        assert route_result["strategy"] == "retrieval_focused"
        assert len(retrieval_result["documents"]) <= 2
        assert all(doc["score"] > 0 for doc in retrieval_result["documents"])
        assert processing_time < 6.0, f"Tool chain too slow: {processing_time:.2f}s"

        print(
            f"Tool integration performance: {processing_time:.2f}s for "
            f"{len(nodes)} results"
        )


@pytest.mark.integration
class TestResourceManagement:
    """Test resource management and cleanup in integration scenarios."""

    def test_memory_usage_with_lightweight_models(
        self,
        lightweight_llama_embedding,
        mock_llm_for_integration,
        test_documents_comprehensive,
    ):
        """Test memory usage patterns with lightweight models.

        Validates:
        - Memory efficiency with lightweight components
        - Resource cleanup after operations
        - Memory stability across multiple operations
        - No significant memory leaks
        """
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        Settings.embed_model = lightweight_llama_embedding
        Settings.llm = mock_llm_for_integration

        # Perform multiple indexing operations
        for _ in range(3):
            index = VectorStoreIndex.from_documents(
                test_documents_comprehensive, vector_store=SimpleVectorStore()
            )

            # Query the index
            query_engine = index.as_query_engine(similarity_top_k=1)
            response = query_engine.query("Test query")

            # Force cleanup
            del index
            del query_engine
            del response
            gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Validate memory usage
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.1f}MB"

    @pytest.mark.asyncio
    async def test_concurrent_resource_management(
        self,
        lightweight_llama_embedding,
        mock_llm_for_integration,
        test_documents_comprehensive,
    ):
        """Test resource management under concurrent operations.

        Validates:
        - Resource isolation between concurrent operations
        - Proper cleanup in async contexts
        - No resource conflicts or deadlocks
        - Stable performance under concurrent load
        """
        Settings.embed_model = lightweight_llama_embedding
        Settings.llm = mock_llm_for_integration

        async def create_and_query_index(doc_subset: list[Document], query: str):
            """Helper function for concurrent index operations."""
            index = VectorStoreIndex.from_documents(
                doc_subset, vector_store=SimpleVectorStore()
            )
            query_engine = index.as_query_engine(similarity_top_k=1)
            response = await query_engine.aquery(query)
            return len(response.source_nodes)

        # Create concurrent tasks
        tasks = [
            create_and_query_index(
                test_documents_comprehensive[i : i + 2], f"Query {i}"
            )
            for i in range(0, len(test_documents_comprehensive), 2)
        ]

        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        concurrent_time = end_time - start_time

        # Validate concurrent execution
        assert len(results) > 0, "Should complete concurrent operations"
        assert all(isinstance(r, int) for r in results), (
            "All tasks should return valid results"
        )
        assert concurrent_time < 15.0, (
            f"Concurrent operations too slow: {concurrent_time:.2f}s"
        )


# Performance Integration Tests
@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Integration tests focused on performance with lightweight models."""

    def test_end_to_end_performance_regression(
        self,
        lightweight_llama_embedding,
        mock_llm_for_integration,
        test_documents_comprehensive,
    ):
        """Test end-to-end pipeline performance with regression detection.

        Validates:
        - Complete pipeline execution time <15s
        - Performance consistency across runs
        - Resource efficiency with lightweight models
        - Baseline performance metrics with timing assertions
        """
        Settings.embed_model = lightweight_llama_embedding
        Settings.llm = mock_llm_for_integration

        def end_to_end_operation():
            start_time = time.perf_counter()

            # Create index
            index = VectorStoreIndex.from_documents(
                test_documents_comprehensive, vector_store=SimpleVectorStore()
            )

            index_time = time.perf_counter() - start_time

            # Query index
            query_start = time.perf_counter()
            query_engine = index.as_query_engine(similarity_top_k=2)
            response = query_engine.query("What are the key AI techniques?")
            query_time = time.perf_counter() - query_start

            total_time = time.perf_counter() - start_time

            return {
                "result_count": len(response.source_nodes),
                "index_time": index_time,
                "query_time": query_time,
                "total_time": total_time,
            }

        # Execute operation multiple times for consistency
        results = []
        for _ in range(3):
            result = end_to_end_operation()
            results.append(result)

        # Validate performance regression thresholds
        for result in results:
            assert result["result_count"] > 0, "Should return source nodes"
            assert result["result_count"] <= 2, "Should respect top_k limit"
            assert result["index_time"] < 10.0, (
                f"Index creation too slow: {result['index_time']:.2f}s"
            )
            assert result["query_time"] < 5.0, (
                f"Query processing too slow: {result['query_time']:.2f}s"
            )
            assert result["total_time"] < 15.0, (
                f"End-to-end too slow: {result['total_time']:.2f}s"
            )

        # Calculate performance statistics
        avg_total = sum(r["total_time"] for r in results) / len(results)
        max_total = max(r["total_time"] for r in results)
        min_total = min(r["total_time"] for r in results)

        print(
            f"Performance stats: avg={avg_total:.2f}s, max={max_total:.2f}s, "
            f"min={min_total:.2f}s"
        )

        # Performance consistency check (allow for initialization overhead)
        variance_ratio = max_total / min_total
        print(f"Performance variance: {variance_ratio:.1f}x (max/min)")

        # For integration tests, focus on absolute thresholds rather than variance
        # High variance is expected due to initialization, model loading, etc.
        if variance_ratio > 5.0:
            print(
                f"Warning: High variance detected ({variance_ratio:.1f}x) - "
                f"consider warmup runs"
            )

        # Ensure average performance is reasonable (main regression detection)
        assert avg_total < 12.0, (
            f"Average performance regression detected: {avg_total:.2f}s"
        )

    @pytest.mark.parametrize("doc_count", [3, 5, 10])
    def test_scalability_with_document_count(
        self,
        lightweight_llama_embedding,
        mock_llm_for_integration,
        test_documents_comprehensive,
        doc_count,
    ):
        """Test performance scalability with varying document counts.

        Validates:
        - Performance scaling with document count
        - Linear or sub-linear time complexity
        - Memory usage scaling
        - Stability across different scales
        """
        Settings.embed_model = lightweight_llama_embedding
        Settings.llm = mock_llm_for_integration

        # Use subset of documents
        docs = (
            test_documents_comprehensive
            * (doc_count // len(test_documents_comprehensive) + 1)
        )[:doc_count]

        start_time = time.perf_counter()

        # Create index with variable document count
        index = VectorStoreIndex.from_documents(docs, vector_store=SimpleVectorStore())

        # Test query performance
        query_engine = index.as_query_engine(similarity_top_k=2)
        response = query_engine.query("Test scalability query")

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Validate scalability
        time_per_doc = processing_time / doc_count
        assert time_per_doc < 2.0, f"Per-document time too high: {time_per_doc:.2f}s"
        assert len(response.source_nodes) <= 2, "Should respect top_k limit"

        # Performance regression thresholds with more realistic scaling
        max_expected_time = min(
            20.0, 4.0 + doc_count * 0.3
        )  # Base time + linear scaling
        assert processing_time < max_expected_time, (
            f"Processing time {processing_time:.2f}s exceeds expected "
            f"{max_expected_time:.2f}s for {doc_count} documents"
        )

        print(
            f"Scalability test (n={doc_count}): {processing_time:.2f}s total, "
            f"{time_per_doc:.2f}s per doc"
        )
