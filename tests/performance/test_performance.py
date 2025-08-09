"""Performance benchmarks and optimization tests for DocMind AI system.

This module provides comprehensive performance testing using pytest-benchmark
for critical system components including embeddings, search, reranking, and
complete workflows following 2025 best practices.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_factory import analyze_query_complexity, get_agent_system
from utils.model_manager import ModelManager


class TestEmbeddingPerformance:
    """Performance tests for embedding generation and caching."""

    @pytest.mark.performance
    def test_dense_embedding_performance(self, benchmark, mock_embedding_model):
        """Benchmark dense embedding generation performance.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_embedding_model: Mock dense embedding model.
        """
        with patch(
            "utils.FastEmbedModelManager.get_model", return_value=mock_embedding_model
        ):
            texts = [
                f"Document {i} with content about various topics" for i in range(50)
            ]

            def embed_documents():
                manager = ModelManager()
                model = manager.get_text_embedding_model("BAAI/bge-large-en-v1.5")
                return model.embed_documents(texts)

            result = benchmark(embed_documents)
            assert len(result) == 50

    @pytest.mark.performance
    def test_sparse_embedding_performance(self, benchmark, mock_sparse_embedding_model):
        """Benchmark sparse embedding generation performance.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_sparse_embedding_model: Mock sparse embedding model.
        """
        with patch(
            "utils.FastEmbedModelManager.get_model",
            return_value=mock_sparse_embedding_model,
        ):
            texts = [f"Document {i} analyzing various AI concepts" for i in range(50)]

            def encode_documents():
                manager = ModelManager()
                model = manager.get_text_embedding_model("prithvida/Splade_PP_en_v1")
                return model.encode(texts)

            result = benchmark(encode_documents)
            assert len(result) == 50

    @pytest.mark.performance
    def test_model_manager_caching_performance(self, benchmark):
        """Benchmark FastEmbedModelManager caching efficiency.

        Args:
            benchmark: Pytest benchmark fixture.
        """
        with patch("utils.FastEmbedModelManager._load_model") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            def access_cached_model():
                manager = ModelManager()
                # Access same model multiple times - should be cached
                model1 = manager.get_multimodal_embedding_model()
                model2 = manager.get_multimodal_embedding_model()
                model3 = manager.get_multimodal_embedding_model()
                return model1, model2, model3

            result = benchmark(access_cached_model)
            # Should only load once due to caching
            assert mock_load.call_count == 1
            assert all(model is result[0] for model in result)

    @pytest.mark.performance
    def test_concurrent_embedding_performance(self, benchmark, mock_embedding_model):
        """Benchmark concurrent embedding generation.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_embedding_model: Mock embedding model.
        """
        import concurrent.futures

        with patch(
            "utils.FastEmbedModelManager.get_model", return_value=mock_embedding_model
        ):

            def concurrent_embedding():
                texts = [f"Text batch {i}" for i in range(20)]

                def embed_single_text(text):
                    manager = ModelManager()
                    model = manager.get_multimodal_embedding_model()
                    return model.embed_query(text)

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(embed_single_text, text) for text in texts
                    ]
                    results = [future.result() for future in futures]
                return results

            result = benchmark(concurrent_embedding)
            assert len(result) == 20


class TestSearchPerformance:
    """Performance tests for vector search operations."""

    @pytest.mark.performance
    def test_vector_search_performance(self, benchmark, mock_qdrant_client):
        """Benchmark vector search performance.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_qdrant_client: Mock Qdrant client.
        """
        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            # Setup search results
            mock_qdrant_client.search.return_value = [
                MagicMock(id=i, score=0.9 - i * 0.01, payload={"text": f"Result {i}"})
                for i in range(20)
            ]

            query_vector = [0.1] * 1024

            def search_operation():
                return mock_qdrant_client.search(
                    collection_name="test", query_vector=query_vector, limit=20
                )

            result = benchmark(search_operation)
            assert len(result) == 20

    @pytest.mark.performance
    def test_hybrid_search_performance(
        self,
        benchmark,
        mock_qdrant_client,
        mock_embedding_model,
        mock_sparse_embedding_model,
    ):
        """Benchmark hybrid search performance.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_qdrant_client: Mock Qdrant client.
            mock_embedding_model: Mock dense embedding model.
            mock_sparse_embedding_model: Mock sparse embedding model.
        """
        with (
            patch("utils.QdrantClient", return_value=mock_qdrant_client),
            patch("utils.FastEmbedModelManager.get_model") as mock_get_model,
        ):

            def model_side_effect(model_name):
                if "splade" in model_name.lower():
                    return mock_sparse_embedding_model
                else:
                    return mock_embedding_model

            mock_get_model.side_effect = model_side_effect

            # Setup search results for both searches
            mock_qdrant_client.search.side_effect = [
                # Dense results
                [
                    MagicMock(id=i, score=0.9, payload={"text": f"Dense {i}"})
                    for i in range(10)
                ],
                # Sparse results
                [
                    MagicMock(id=i + 5, score=0.8, payload={"text": f"Sparse {i}"})
                    for i in range(10)
                ],
            ]

            def hybrid_search():
                query = "Test hybrid search query"
                manager = ModelManager()

                # Generate embeddings
                dense_model = manager.get_text_embedding_model("BAAI/bge-large-en-v1.5")
                sparse_model = manager.get_text_embedding_model(
                    "prithvida/Splade_PP_en_v1"
                )

                dense_embedding = dense_model.embed_query(query)
                sparse_embedding = sparse_model.encode([query])[0]

                # Perform searches
                dense_results = mock_qdrant_client.search(
                    collection_name="dense", query_vector=dense_embedding, limit=10
                )
                sparse_results = mock_qdrant_client.search(
                    collection_name="sparse", query_vector=sparse_embedding, limit=10
                )

                return dense_results, sparse_results

            result = benchmark(hybrid_search)
            assert len(result[0]) == 10  # Dense results
            assert len(result[1]) == 10  # Sparse results

    @pytest.mark.performance
    def test_batch_search_performance(self, benchmark, mock_qdrant_client):
        """Benchmark batch search operations.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_qdrant_client: Mock Qdrant client.
        """
        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            # Setup consistent search results
            mock_qdrant_client.search.return_value = [
                MagicMock(id=i, score=0.9, payload={"text": f"Result {i}"})
                for i in range(5)
            ]

            query_vectors = [[0.1] * 1024 for _ in range(20)]

            def batch_search():
                results = []
                for vector in query_vectors:
                    result = mock_qdrant_client.search(
                        collection_name="test", query_vector=vector, limit=5
                    )
                    results.extend(result)
                return results

            result = benchmark(batch_search)
            assert len(result) == 100  # 20 queries * 5 results each


class TestRerankingPerformance:
    """Performance tests for reranking operations."""

    @pytest.mark.performance
    def test_reranking_performance(self, benchmark, mock_reranker):
        """Benchmark ColBERT reranking performance.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_reranker: Mock reranker fixture.
        """
        # Setup reranking results
        mock_reranker.rerank.return_value = [
            {"index": i, "score": 0.9 - i * 0.01} for i in range(20)
        ]

        query = "Performance test query for reranking"
        documents = [f"Document {i} with detailed content" for i in range(50)]

        def rerank_operation():
            return mock_reranker.rerank(query=query, documents=documents, top_k=20)

        result = benchmark(rerank_operation)
        assert len(result) == 20

    @pytest.mark.performance
    def test_reranking_batch_performance(self, benchmark, mock_reranker):
        """Benchmark batch reranking performance.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_reranker: Mock reranker fixture.
        """
        mock_reranker.rerank.return_value = [
            {"index": 0, "score": 0.9},
            {"index": 1, "score": 0.8},
        ]

        queries = [f"Query {i} about various topics" for i in range(10)]
        documents = [f"Document {i} with content" for i in range(20)]

        def batch_rerank():
            results = []
            for query in queries:
                result = mock_reranker.rerank(query=query, documents=documents, top_k=2)
                results.extend(result)
            return results

        result = benchmark(batch_rerank)
        assert len(result) == 20  # 10 queries * 2 results each

    @pytest.mark.performance
    def test_reranking_scalability(self, benchmark, mock_reranker):
        """Benchmark reranking scalability with large document sets.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_reranker: Mock reranker fixture.
        """
        # Test with large document set
        mock_reranker.rerank.return_value = [
            {"index": i, "score": 0.9 - i * 0.001} for i in range(50)
        ]

        query = "Scalability test query"
        large_document_set = [
            f"Document {i} with extensive content" for i in range(500)
        ]

        def large_scale_rerank():
            return mock_reranker.rerank(
                query=query, documents=large_document_set, top_k=50
            )

        result = benchmark(large_scale_rerank)
        assert len(result) == 50


class TestAgentPerformance:
    """Performance tests for agent system operations."""

    @pytest.mark.performance
    def test_query_complexity_analysis_performance(self, benchmark):
        """Benchmark query complexity analysis performance.

        Args:
            benchmark: Pytest benchmark fixture.
        """
        complex_query = (
            "How do hybrid search systems integrate SPLADE++ sparse embeddings "
            "with BGE-Large dense embeddings using Reciprocal Rank Fusion for "
            "optimized document retrieval in large-scale AI applications?"
        )

        with patch("agent_factory.analyze_query_complexity") as mock_analyze:
            mock_analyze.return_value = {
                "complexity": "complex",
                "reasoning": "Multi-step analysis required",
                "features": {"word_count": 25, "technical_terms": 8},
            }

            def analyze_operation():
                return analyze_query_complexity(complex_query)

            result = benchmark(analyze_operation)
            assert result["complexity"] == "complex"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_agent_async_performance(self, benchmark, test_settings):
        """Benchmark asynchronous agent processing performance.

        Args:
            benchmark: Pytest benchmark fixture.
            test_settings: Test settings fixture.
        """
        with patch("agent_factory.get_agent_system") as mock_get_agent:
            mock_agent = AsyncMock()
            mock_agent.arun.return_value = "Fast agent response"
            mock_get_agent.return_value = mock_agent

            async def agent_operation():
                agent_system = get_agent_system(settings=test_settings)
                return await agent_system.arun("Performance test query")

            # Use async benchmark
            def sync_wrapper():
                return asyncio.run(agent_operation())

            result = benchmark(sync_wrapper)
            assert result == "Fast agent response"

    @pytest.mark.performance
    def test_agent_tool_invocation_performance(self, benchmark, test_settings):
        """Benchmark agent tool invocation performance.

        Args:
            benchmark: Pytest benchmark fixture.
            test_settings: Test settings fixture.
        """
        with (
            patch("agent_factory.get_agent_system") as mock_get_agent,
            patch("utils.create_tools_from_index") as mock_create_tools,
        ):
            # Mock fast tools
            mock_tools = [
                MagicMock(name=f"tool_{i}", func=MagicMock(return_value=f"Result {i}"))
                for i in range(5)
            ]
            mock_create_tools.return_value = mock_tools

            mock_agent = MagicMock()
            mock_agent.tools = mock_tools
            mock_get_agent.return_value = mock_agent

            def tool_invocation():
                agent_system = get_agent_system(settings=test_settings)
                results = []
                for tool in agent_system.tools:
                    result = tool.func("test input")
                    results.append(result)
                return results

            result = benchmark(tool_invocation)
            assert len(result) == 5


class TestMemoryPerformance:
    """Performance tests for memory usage and efficiency."""

    @pytest.mark.performance
    def test_memory_efficient_embedding_processing(
        self, large_document_set, mock_embedding_model
    ):
        """Test memory efficiency with large document processing.

        Args:
            large_document_set: Large document set fixture.
            mock_embedding_model: Mock embedding model.
        """
        import tracemalloc

        with patch(
            "utils.FastEmbedModelManager.get_model", return_value=mock_embedding_model
        ):
            # Setup mock for large dataset
            mock_embedding_model.embed_documents.return_value = [
                [0.1] * 1024 for _ in range(len(large_document_set))
            ]

            # Start memory tracing
            tracemalloc.start()

            manager = ModelManager()
            model = manager.get_multimodal_embedding_model()

            # Process in batches to test memory efficiency
            batch_size = 20
            texts = [doc.text for doc in large_document_set]

            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                embeddings = model.embed_documents(batch)
                all_embeddings.extend(embeddings)

            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            assert len(all_embeddings) == len(large_document_set)
            # Memory usage should be reasonable (test is implementation dependent)
            assert peak < 100 * 1024 * 1024  # Less than 100MB peak

    @pytest.mark.performance
    def test_search_result_memory_efficiency(self, mock_qdrant_client):
        """Test memory efficiency of search result processing.

        Args:
            mock_qdrant_client: Mock Qdrant client.
        """
        import sys

        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            # Create large search results
            large_results = [
                MagicMock(
                    id=i,
                    score=0.9 - i * 0.001,
                    payload={
                        "text": f"Document {i} with content" * 10
                    },  # Larger payloads
                )
                for i in range(1000)
            ]
            mock_qdrant_client.search.return_value = large_results

            # Measure memory usage
            initial_size = sys.getsizeof(large_results)

            # Process results
            results = mock_qdrant_client.search(
                collection_name="test", query_vector=[0.1] * 1024, limit=1000
            )

            # Extract relevant data (simulate processing)
            processed_results = [
                {
                    "id": r.id,
                    "score": r.score,
                    "text": r.payload["text"][:100],
                }  # Truncate
                for r in results
            ]

            processed_size = sys.getsizeof(processed_results)

            assert len(processed_results) == 1000
            # Processed results should be more memory efficient
            assert processed_size < initial_size * 2  # Allow some overhead


class TestConcurrencyPerformance:
    """Performance tests for concurrent operations."""

    @pytest.mark.performance
    def test_concurrent_search_performance(self, benchmark, mock_qdrant_client):
        """Benchmark concurrent search operations.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_qdrant_client: Mock Qdrant client.
        """
        import concurrent.futures

        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            mock_qdrant_client.search.return_value = [
                MagicMock(id=i, score=0.9, payload={"text": f"Result {i}"})
                for i in range(3)
            ]

            query_vectors = [[0.1] * 1024 for _ in range(20)]

            def concurrent_search():
                def single_search(vector):
                    return mock_qdrant_client.search(
                        collection_name="test", query_vector=vector, limit=3
                    )

                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(single_search, vector)
                        for vector in query_vectors
                    ]
                    results = []
                    for future in futures:
                        results.extend(future.result())
                return results

            result = benchmark(concurrent_search)
            assert len(result) == 60  # 20 queries * 3 results each

    @pytest.mark.performance
    def test_concurrent_reranking_performance(self, benchmark, mock_reranker):
        """Benchmark concurrent reranking operations.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_reranker: Mock reranker fixture.
        """
        import concurrent.futures

        mock_reranker.rerank.return_value = [
            {"index": 0, "score": 0.9},
            {"index": 1, "score": 0.8},
        ]

        queries = [f"Query {i}" for i in range(10)]
        documents = [f"Document {i}" for i in range(20)]

        def concurrent_rerank():
            def single_rerank(query):
                return mock_reranker.rerank(query=query, documents=documents, top_k=2)

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(single_rerank, query) for query in queries]
                results = []
                for future in futures:
                    results.extend(future.result())
                return results

            result = benchmark(concurrent_rerank)
            assert len(result) == 20  # 10 queries * 2 results each


class TestLatencyMonitoring:
    """Tests for latency monitoring and performance thresholds."""

    def test_embedding_latency_thresholds(self, mock_embedding_model):
        """Test that embedding generation meets latency requirements.

        Args:
            mock_embedding_model: Mock embedding model.
        """
        with patch(
            "utils.FastEmbedModelManager.get_model", return_value=mock_embedding_model
        ):
            texts = ["Test document for latency measurement"]

            start_time = time.perf_counter()
            manager = ModelManager()
            model = manager.get_multimodal_embedding_model()
            embeddings = model.embed_documents(texts)
            end_time = time.perf_counter()

            latency = end_time - start_time

            assert len(embeddings) == 1
            # Mock should be very fast
            assert latency < 1.0  # Less than 1 second

    def test_search_latency_thresholds(self, mock_qdrant_client):
        """Test that search operations meet latency requirements.

        Args:
            mock_qdrant_client: Mock Qdrant client.
        """
        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            mock_qdrant_client.search.return_value = [
                MagicMock(id=1, score=0.9, payload={"text": "Result"})
            ]

            start_time = time.perf_counter()
            results = mock_qdrant_client.search(
                collection_name="test", query_vector=[0.1] * 1024, limit=10
            )
            end_time = time.perf_counter()

            latency = end_time - start_time

            assert len(results) == 1
            # Mock should be very fast
            assert latency < 0.1  # Less than 100ms

    def test_reranking_latency_thresholds(self, mock_reranker):
        """Test that reranking operations meet latency requirements.

        Args:
            mock_reranker: Mock reranker fixture.
        """
        mock_reranker.rerank.return_value = [{"index": 0, "score": 0.9}]

        query = "Latency test query"
        documents = ["Test document"]

        start_time = time.perf_counter()
        results = mock_reranker.rerank(query=query, documents=documents, top_k=1)
        end_time = time.perf_counter()

        latency = end_time - start_time

        assert len(results) == 1
        # Mock should be very fast
        assert latency < 0.1  # Less than 100ms

    @pytest.mark.slow
    def test_end_to_end_performance_requirements(
        self,
        mock_embedding_model,
        mock_sparse_embedding_model,
        mock_qdrant_client,
        mock_reranker,
    ):
        """Test end-to-end performance meets requirements.

        Args:
            mock_embedding_model: Mock dense embedding model.
            mock_sparse_embedding_model: Mock sparse embedding model.
            mock_qdrant_client: Mock Qdrant client.
            mock_reranker: Mock reranker fixture.
        """
        with (
            patch("utils.FastEmbedModelManager.get_model") as mock_get_model,
            patch("utils.QdrantClient", return_value=mock_qdrant_client),
        ):

            def model_side_effect(model_name):
                if "splade" in model_name.lower():
                    return mock_sparse_embedding_model
                else:
                    return mock_embedding_model

            mock_get_model.side_effect = model_side_effect

            # Setup mocks
            mock_qdrant_client.search.side_effect = [
                [MagicMock(id=1, score=0.9, payload={"text": "Dense result"})],
                [MagicMock(id=2, score=0.8, payload={"text": "Sparse result"})],
            ]
            mock_reranker.rerank.return_value = [{"index": 0, "score": 0.95}]

            # Measure end-to-end latency
            start_time = time.perf_counter()

            # Complete workflow
            query = "End-to-end performance test"

            # 1. Generate embeddings
            manager = ModelManager()
            dense_model = manager.get_text_embedding_model("BAAI/bge-large-en-v1.5")
            sparse_model = manager.get_text_embedding_model("prithvida/Splade_PP_en_v1")

            dense_embedding = dense_model.embed_query(query)
            sparse_embedding = sparse_model.encode([query])[0]

            # 2. Perform searches
            dense_results = mock_qdrant_client.search(
                collection_name="dense", query_vector=dense_embedding, limit=5
            )
            sparse_results = mock_qdrant_client.search(
                collection_name="sparse", query_vector=sparse_embedding, limit=5
            )

            # 3. Rerank results
            all_documents = [r.payload["text"] for r in dense_results + sparse_results]
            reranked = mock_reranker.rerank(
                query=query, documents=all_documents, top_k=3
            )

            end_time = time.perf_counter()
            total_latency = end_time - start_time

            # Verify results
            assert len(dense_results) == 1
            assert len(sparse_results) == 1
            assert len(reranked) == 1

            # Performance requirement (mocked operations should be very fast)
            assert total_latency < 2.0  # Less than 2 seconds for complete workflow
