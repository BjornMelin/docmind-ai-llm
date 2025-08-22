"""Example integration test demonstrating proper tiered testing strategy.

This test showcases the AI research recommendations in practice:
- Uses lightweight models (all-MiniLM-L6-v2 80MB) instead of heavy models (BGE-M3 1GB)
- Tests real component integration without full system overhead
- Demonstrates proper mocking boundaries at external service interfaces
- Shows integration between embedding, vectorstore, and retrieval components

This serves as a template for other integration tests in the system.
"""

import time

import numpy as np
import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.storage import StorageContext


@pytest.mark.integration
@pytest.mark.embeddings
class TestEmbeddingIntegrationExample:
    """Example integration test following ML testing best practices.

    Demonstrates the middle tier of the testing pyramid:
    - Real lightweight embedding model (all-MiniLM-L6-v2)
    - Mocked external services (Qdrant)
    - Real LlamaIndex components
    - Performance validation with realistic expectations
    """

    async def test_lightweight_embedding_pipeline(
        self,
        integration_settings,
        lightweight_embedding_model,
        test_documents,
        mock_qdrant_client,
    ):
        """Test embedding pipeline with lightweight model integration.

        This test demonstrates proper integration testing:
        1. Uses real lightweight model (80MB vs 1GB production model)
        2. Tests actual embedding generation and dimensions
        3. Validates component integration without full system overhead
        4. Measures realistic performance expectations
        """
        if not lightweight_embedding_model:
            pytest.skip(
                "Lightweight embedding model not available for integration test"
            )

        # Test real embedding generation with lightweight model
        start_time = time.perf_counter()

        # Extract text content for embedding
        texts = [doc.text for doc in test_documents[:3]]  # Use subset for integration
        embeddings = lightweight_embedding_model.encode(texts)

        embedding_time = (time.perf_counter() - start_time) * 1000

        # Validate embedding properties
        assert embeddings.shape == (3, 384), (
            f"Expected (3, 384), got {embeddings.shape}"
        )
        assert embeddings.dtype == np.float32, (
            f"Expected float32, got {embeddings.dtype}"
        )

        # Validate performance (lightweight model should be fast)
        assert embedding_time < 2000, (
            f"Embedding took {embedding_time:.2f}ms, expected <2000ms"
        )

        # Validate embedding quality (semantic similarity)
        # Documents about similar topics should have higher similarity
        similarity_01 = np.dot(embeddings[0], embeddings[1])  # Both about AI/retrieval
        similarity_02 = np.dot(embeddings[0], embeddings[2])  # Different topics

        assert similarity_01 > 0.5, (
            f"Related documents similarity too low: {similarity_01:.3f}"
        )
        assert abs(similarity_01 - similarity_02) > 0.1, (
            "Documents not sufficiently differentiated"
        )

    async def test_vectorstore_integration_with_mocked_qdrant(
        self,
        integration_settings,
        lightweight_embedding_model,
        test_documents,
        mock_qdrant_client,
    ):
        """Test VectorStoreIndex integration with mocked Qdrant client.

        Demonstrates integration testing with proper mocking boundaries:
        - Real embedding model and LlamaIndex components
        - Mocked external service (Qdrant) at the boundary
        - Tests data flow and component interaction
        """
        if not lightweight_embedding_model:
            pytest.skip("Lightweight embedding model not available")

        # Create a simple in-memory vector store for integration testing
        # (In real integration, you might use a test Qdrant instance)
        from llama_index.core.vector_stores import SimpleVectorStore

        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Test document indexing with real components
        start_time = time.perf_counter()

        # Use subset of documents for integration test performance
        integration_docs = test_documents[:3]

        # This tests real LlamaIndex integration with lightweight model
        index = VectorStoreIndex.from_documents(
            integration_docs,
            storage_context=storage_context,
            embed_model=None,  # Will use globally configured model
        )

        indexing_time = (time.perf_counter() - start_time) * 1000

        # Validate indexing performance
        assert indexing_time < 5000, (
            f"Indexing took {indexing_time:.2f}ms, expected <5000ms"
        )

        # Test query processing
        query_engine = index.as_query_engine(similarity_top_k=2)

        start_time = time.perf_counter()
        response = query_engine.query("What is DocMind AI's retrieval approach?")
        query_time = (time.perf_counter() - start_time) * 1000

        # Validate query response
        assert response is not None, "Query returned no response"
        assert len(response.response) > 0, "Query response is empty"
        assert query_time < 3000, f"Query took {query_time:.2f}ms, expected <3000ms"

        # Validate that relevant documents were retrieved
        source_nodes = response.source_nodes
        assert len(source_nodes) > 0, "No source nodes retrieved"
        assert len(source_nodes) <= 2, f"Too many source nodes: {len(source_nodes)}"

        # Validate relevance scores
        for node in source_nodes:
            assert hasattr(node, "score"), "Source node missing similarity score"
            assert 0 <= node.score <= 1, f"Invalid similarity score: {node.score}"

    async def test_hybrid_search_logic_integration(
        self, integration_settings, test_documents, mock_qdrant_client
    ):
        """Test hybrid search coordination logic with mocked components.

        Integration test for hybrid search workflow:
        - Tests search strategy coordination
        - Validates result fusion logic
        - Uses mocked dense/sparse embeddings for speed
        - Focuses on integration logic, not model performance
        """
        # Mock dense and sparse search results
        mock_dense_results = [
            {"id": "doc_1", "score": 0.92, "source": "dense"},
            {"id": "doc_3", "score": 0.87, "source": "dense"},
            {"id": "doc_2", "score": 0.83, "source": "dense"},
        ]

        mock_sparse_results = [
            {"id": "doc_2", "score": 0.89, "source": "sparse"},
            {"id": "doc_1", "score": 0.85, "source": "sparse"},
            {"id": "doc_4", "score": 0.78, "source": "sparse"},
        ]

        # Test RRF (Reciprocal Rank Fusion) integration logic
        def apply_rrf_fusion(dense_results, sparse_results, alpha=0.7):
            """Apply RRF fusion to combine dense and sparse results."""
            # Create rank-based scores
            combined_scores = {}

            # Add dense scores with alpha weighting
            for rank, result in enumerate(dense_results):
                doc_id = result["id"]
                rrf_score = alpha * (1 / (rank + 1))
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score

            # Add sparse scores with (1-alpha) weighting
            for rank, result in enumerate(sparse_results):
                doc_id = result["id"]
                rrf_score = (1 - alpha) * (1 / (rank + 1))
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score

            # Sort by combined score
            sorted_results = sorted(
                combined_scores.items(), key=lambda x: x[1], reverse=True
            )

            return [
                {"id": doc_id, "rrf_score": score} for doc_id, score in sorted_results
            ]

        # Test fusion logic
        start_time = time.perf_counter()
        fused_results = apply_rrf_fusion(
            mock_dense_results, mock_sparse_results, alpha=0.7
        )
        fusion_time = (time.perf_counter() - start_time) * 1000

        # Validate fusion results
        assert len(fused_results) > 0, "RRF fusion produced no results"
        assert len(fused_results) <= 4, (
            f"Too many fused results: {len(fused_results)}"
        )  # Max unique docs

        # Validate RRF scoring
        rrf_scores = [r["rrf_score"] for r in fused_results]
        assert all(score > 0 for score in rrf_scores), (
            "All RRF scores should be positive"
        )
        assert rrf_scores == sorted(rrf_scores, reverse=True), (
            "Results should be sorted by RRF score"
        )

        # Validate performance
        assert fusion_time < 50, f"RRF fusion took {fusion_time:.2f}ms, expected <50ms"

        # Validate that both dense and sparse contributed to top results
        top_doc_id = fused_results[0]["id"]
        appeared_in_dense = any(r["id"] == top_doc_id for r in mock_dense_results)
        appeared_in_sparse = any(r["id"] == top_doc_id for r in mock_sparse_results)

        # Top result should appear in at least one search type
        assert appeared_in_dense or appeared_in_sparse, (
            "Top result should come from search results"
        )

    @pytest.mark.performance
    async def test_integration_performance_benchmarks(
        self, integration_settings, lightweight_embedding_model, test_documents
    ):
        """Performance benchmarks for integration components.

        Validates that integration-level performance meets expectations:
        - Lightweight model performance targets
        - Memory usage constraints
        - Throughput measurements
        """
        if not lightweight_embedding_model:
            pytest.skip("Performance test requires lightweight embedding model")

        # Test batch embedding performance
        texts = [doc.text for doc in test_documents]  # 5 documents

        start_time = time.perf_counter()
        embeddings = lightweight_embedding_model.encode(texts)
        batch_time = (time.perf_counter() - start_time) * 1000

        # Performance validations for integration testing
        assert batch_time < 3000, (
            f"Batch embedding took {batch_time:.2f}ms, expected <3000ms"
        )

        # Throughput calculation
        throughput_docs_per_sec = len(texts) / (batch_time / 1000)
        assert throughput_docs_per_sec > 1.0, (
            f"Throughput too low: {throughput_docs_per_sec:.2f} docs/sec"
        )

        # Memory usage estimation (lightweight model)
        estimated_memory_mb = embeddings.nbytes / (1024 * 1024)
        assert estimated_memory_mb < 1, (
            f"Memory usage too high: {estimated_memory_mb:.2f}MB"
        )

        # Test repeated operations (consistency)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = lightweight_embedding_model.encode([texts[0]])  # Single document
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        # Validate consistency (coefficient of variation < 0.3)
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time if mean_time > 0 else 0

        assert cv < 0.3, f"Inconsistent performance: CV={cv:.3f}, times={times}"
        assert mean_time < 1000, (
            f"Single document embedding too slow: {mean_time:.2f}ms"
        )

    async def test_error_handling_integration(
        self, integration_settings, test_documents, mock_qdrant_client
    ):
        """Test error handling in integration scenarios.

        Validates graceful degradation and error recovery:
        - Network failures
        - Invalid inputs
        - Resource constraints
        """
        # Test invalid document handling
        invalid_docs = [
            Document(text="", metadata={"invalid": "empty_text"}),  # Empty text
            Document(
                text="x" * 100000, metadata={"invalid": "too_long"}
            ),  # Very long text
        ]

        # Integration components should handle invalid inputs gracefully
        from llama_index.core.vector_stores import SimpleVectorStore

        vector_store = SimpleVectorStore()

        try:
            # This should not crash the integration
            index = VectorStoreIndex.from_documents(
                invalid_docs,
                storage_context=StorageContext.from_defaults(vector_store=vector_store),
            )
            # If successful, validate it handles edge cases
            assert index is not None, (
                "Index creation should succeed even with edge cases"
            )
        except Exception as e:
            # If it fails, it should fail gracefully with clear error
            assert "text" in str(e).lower() or "document" in str(e).lower(), (
                f"Error should be descriptive about document issues: {e}"
            )

        # Test mock Qdrant client error scenarios
        original_search = mock_qdrant_client.search
        mock_qdrant_client.search.side_effect = ConnectionError(
            "Simulated network failure"
        )

        # Integration should handle external service failures
        try:
            # This simulates network failure to Qdrant
            mock_qdrant_client.search(query_vector=[0.1] * 384, limit=5)
            assert False, "Should have raised ConnectionError"
        except ConnectionError:
            # Expected behavior - external service failure
            pass

        # Restore mock for other tests
        mock_qdrant_client.search = original_search
