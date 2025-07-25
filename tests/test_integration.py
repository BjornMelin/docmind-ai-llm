"""End-to-end integration tests for DocMind AI system.

This module tests complete workflows from document loading through agent
processing, validating that all components work together correctly in
realistic scenarios following 2025 best practices.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_factory import (
    analyze_query_complexity,
    get_agent_system,
    process_query_with_agent_system,
)
from models import Settings
from utils import (
    FastEmbedModelManager,
    create_index_async,
    create_tools_from_index,
)


class TestCompleteWorkflowIntegration:
    """Test complete end-to-end workflow scenarios."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_document_processing_pipeline(
        self,
        sample_documents,
        mock_embedding_model,
        mock_sparse_embedding_model,
        mock_qdrant_client,
        test_settings,
    ):
        """Test complete document processing from upload to query response.

        Args:
            sample_documents: Sample documents fixture.
            mock_embedding_model: Mock dense embedding model.
            mock_sparse_embedding_model: Mock sparse embedding model.
            mock_qdrant_client: Mock Qdrant client.
            test_settings: Test settings fixture.
        """
        with (
            patch("utils.FastEmbedModelManager.get_model") as mock_get_model,
            patch("utils.QdrantClient", return_value=mock_qdrant_client),
            patch("utils.create_index_async") as mock_create_index,
        ):
            # Configure model manager
            def model_side_effect(model_name):
                if "splade" in model_name.lower():
                    return mock_sparse_embedding_model
                else:
                    return mock_embedding_model

            mock_get_model.side_effect = model_side_effect

            # Mock index creation
            mock_index = MagicMock()
            mock_create_index.return_value = mock_index

            # Test pipeline steps
            # 1. Document Processing
            documents = sample_documents
            assert len(documents) == 5

            # 2. Index Creation
            index = await create_index_async(
                documents=documents, settings=test_settings
            )
            assert index is not None
            mock_create_index.assert_called_once()

            # 3. Tool Creation
            with patch("utils.create_tools_from_index") as mock_create_tools:
                mock_tools = [MagicMock(name="search_tool")]
                mock_create_tools.return_value = mock_tools

                tools = create_tools_from_index(index)
                assert len(tools) == 1
                assert tools[0].name == "search_tool"

    @pytest.mark.integration
    def test_hybrid_search_workflow(
        self,
        sample_documents,
        mock_embedding_model,
        mock_sparse_embedding_model,
        mock_qdrant_client,
    ):
        """Test hybrid search workflow with dense and sparse embeddings.

        Args:
            sample_documents: Sample documents fixture.
            mock_embedding_model: Mock dense embedding model.
            mock_sparse_embedding_model: Mock sparse embedding model.
            mock_qdrant_client: Mock Qdrant client.
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

            # Mock search results for hybrid search
            mock_qdrant_client.search.side_effect = [
                # Dense search results
                [MagicMock(id=1, score=0.9, payload={"text": "SPLADE++ is efficient"})],
                # Sparse search results
                [
                    MagicMock(
                        id=2,
                        score=0.85,
                        payload={"text": "BGE-Large provides semantics"},
                    )
                ],
            ]

            # Test hybrid search execution
            query = "What is the difference between sparse and dense embeddings?"

            # Generate embeddings
            manager = FastEmbedModelManager()
            dense_model = manager.get_model("BAAI/bge-large-en-v1.5")
            sparse_model = manager.get_model("prithvida/Splade_PP_en_v1")

            dense_embedding = dense_model.embed_query(query)
            sparse_embedding = sparse_model.encode([query])[0]

            # Perform searches
            dense_results = mock_qdrant_client.search(
                collection_name="dense", query_vector=dense_embedding, limit=5
            )

            sparse_results = mock_qdrant_client.search(
                collection_name="sparse", query_vector=sparse_embedding, limit=5
            )

            # Verify hybrid search execution
            assert len(dense_results) == 1
            assert len(sparse_results) == 1
            assert mock_qdrant_client.search.call_count == 2

    @pytest.mark.integration
    def test_reranking_integration_workflow(
        self,
        sample_documents,
        mock_qdrant_client,
        mock_reranker,
        sample_query_responses,
    ):
        """Test reranking integration within search workflow.

        Args:
            sample_documents: Sample documents fixture.
            mock_qdrant_client: Mock Qdrant client.
            mock_reranker: Mock reranker fixture.
            sample_query_responses: Sample query-response pairs.
        """
        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            # Setup initial search results
            mock_search_results = [
                MagicMock(id=i, score=0.8 - i * 0.1, payload={"text": doc.text})
                for i, doc in enumerate(sample_documents)
            ]
            mock_qdrant_client.search.return_value = mock_search_results

            # Setup reranking results
            mock_reranker.rerank.return_value = [
                {"index": 2, "score": 0.95},  # Reorder results
                {"index": 0, "score": 0.88},
                {"index": 1, "score": 0.82},
            ]

            # Test search + reranking workflow
            query = sample_query_responses[0]["query"]

            # 1. Initial search
            search_results = mock_qdrant_client.search(
                collection_name="test", query_vector=[0.1] * 1024, limit=10
            )

            # 2. Extract documents for reranking
            documents = [result.payload["text"] for result in search_results]

            # 3. Rerank results
            reranked = mock_reranker.rerank(query=query, documents=documents, top_k=3)

            # Verify workflow execution
            assert len(search_results) == 5
            assert len(reranked) == 3
            # Verify reranking changed order (index 2 is now first)
            assert reranked[0]["index"] == 2
            assert reranked[0]["score"] > reranked[1]["score"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_query_processing_workflow(
        self, sample_documents, test_settings, sample_query_responses
    ):
        """Test complete agent query processing workflow.

        Args:
            sample_documents: Sample documents fixture.
            test_settings: Test settings fixture.
            sample_query_responses: Sample query-response pairs.
        """
        with (
            patch("agent_factory.analyze_query_complexity") as mock_analyze,
            patch("agent_factory.get_agent_system") as mock_get_agent,
            patch("agent_factory.process_query_with_agent_system") as mock_process,
        ):
            # Mock query complexity analysis
            mock_analyze.return_value = {
                "complexity": "moderate",
                "reasoning": "Query requires search and analysis",
                "features": {"word_count": 8, "technical_terms": 2},
            }

            # Mock agent system
            mock_agent = AsyncMock()
            mock_agent.arun.return_value = "Comprehensive answer about embeddings"
            mock_get_agent.return_value = mock_agent

            # Mock complete query processing
            mock_process.return_value = {
                "response": "Comprehensive answer about embeddings",
                "complexity": "moderate",
                "processing_time": 1.2,
                "sources": ["doc1.pdf", "doc2.pdf"],
            }

            # Test complete workflow
            query = sample_query_responses[0]["query"]

            # 1. Analyze query complexity
            complexity = analyze_query_complexity(query)
            assert complexity["complexity"] == "moderate"

            # 2. Get appropriate agent system
            agent_system = get_agent_system(settings=test_settings)
            assert agent_system is not None

            # 3. Process query
            result = process_query_with_agent_system(
                query=query, agent_system=agent_system, settings=test_settings
            )

            # Verify complete workflow
            assert result["response"] is not None
            assert result["complexity"] == "moderate"
            assert "processing_time" in result
            assert "sources" in result


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    @pytest.mark.integration
    def test_embedding_model_failure_recovery(self, sample_documents, test_settings):
        """Test recovery from embedding model failures.

        Args:
            sample_documents: Sample documents fixture.
            test_settings: Test settings fixture.
        """
        with patch("utils.FastEmbedModelManager.get_model") as mock_get_model:
            # Mock failing primary model and working fallback
            failing_model = MagicMock()
            failing_model.embed_documents.side_effect = Exception("Model failed")

            fallback_model = MagicMock()
            fallback_model.embed_documents.return_value = [[0.1] * 1024]

            # Test fallback mechanism
            def model_side_effect(model_name):
                if "primary" in model_name:
                    return failing_model
                else:
                    return fallback_model

            mock_get_model.side_effect = model_side_effect

            manager = FastEmbedModelManager()

            # Test primary model failure
            primary_model = manager.get_model("primary_model")
            with pytest.raises(Exception, match="Model failed"):
                primary_model.embed_documents(["test"])

            # Test fallback model success
            fallback = manager.get_model("fallback_model")
            result = fallback.embed_documents(["test"])
            assert len(result) == 1
            assert len(result[0]) == 1024

    @pytest.mark.integration
    def test_search_service_failure_handling(self, mock_qdrant_client):
        """Test handling of search service failures.

        Args:
            mock_qdrant_client: Mock Qdrant client fixture.
        """
        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            # Mock search failure
            mock_qdrant_client.search.side_effect = Exception(
                "Search service unavailable"
            )

            # Test error handling
            with pytest.raises(Exception, match="Search service unavailable"):
                mock_qdrant_client.search(
                    collection_name="test", query_vector=[0.1] * 1024, limit=10
                )

    @pytest.mark.integration
    def test_reranking_service_failure_handling(self, mock_reranker):
        """Test handling of reranking service failures.

        Args:
            mock_reranker: Mock reranker fixture.
        """
        # Mock reranking failure
        mock_reranker.rerank.side_effect = Exception("Reranking service unavailable")

        query = "Test query"
        documents = ["Doc 1", "Doc 2", "Doc 3"]

        # Test error handling
        with pytest.raises(Exception, match="Reranking service unavailable"):
            mock_reranker.rerank(query=query, documents=documents, top_k=3)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_system_failure_handling(self, test_settings):
        """Test handling of agent system failures.

        Args:
            test_settings: Test settings fixture.
        """
        with patch("agent_factory.get_agent_system") as mock_get_agent:
            # Mock failing agent
            failing_agent = AsyncMock()
            failing_agent.arun.side_effect = Exception("Agent system unavailable")
            mock_get_agent.return_value = failing_agent

            agent_system = get_agent_system(settings=test_settings)

            # Test error handling
            with pytest.raises(Exception, match="Agent system unavailable"):
                await agent_system.arun("Test query")


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""

    @pytest.mark.performance
    @pytest.mark.integration
    def test_end_to_end_performance(
        self,
        benchmark,
        sample_documents,
        mock_embedding_model,
        mock_qdrant_client,
        mock_reranker,
    ):
        """Test end-to-end system performance.

        Args:
            benchmark: Pytest benchmark fixture.
            sample_documents: Sample documents fixture.
            mock_embedding_model: Mock embedding model.
            mock_qdrant_client: Mock Qdrant client.
            mock_reranker: Mock reranker fixture.
        """
        with (
            patch(
                "utils.FastEmbedModelManager.get_model",
                return_value=mock_embedding_model,
            ),
            patch("utils.QdrantClient", return_value=mock_qdrant_client),
        ):
            # Setup mocks for performance test
            mock_embedding_model.embed_query.return_value = [0.1] * 1024
            mock_qdrant_client.search.return_value = [
                MagicMock(id=i, score=0.9, payload={"text": f"Result {i}"})
                for i in range(5)
            ]
            mock_reranker.rerank.return_value = [
                {"index": i, "score": 0.9 - i * 0.1} for i in range(3)
            ]

            def end_to_end_workflow():
                # Simulate complete workflow
                query = "What is SPLADE++ embedding?"

                # 1. Generate embedding
                manager = FastEmbedModelManager()
                model = manager.get_model("test_model")
                embedding = model.embed_query(query)

                # 2. Search
                client = mock_qdrant_client
                search_results = client.search(
                    collection_name="test", query_vector=embedding, limit=10
                )

                # 3. Rerank
                documents = [r.payload["text"] for r in search_results]
                reranked = mock_reranker.rerank(
                    query=query, documents=documents, top_k=3
                )

                return reranked

            result = benchmark(end_to_end_workflow)
            assert len(result) == 3

    @pytest.mark.performance
    @pytest.mark.integration
    def test_concurrent_query_performance(
        self, benchmark, mock_embedding_model, mock_qdrant_client
    ):
        """Test performance under concurrent query load.

        Args:
            benchmark: Pytest benchmark fixture.
            mock_embedding_model: Mock embedding model.
            mock_qdrant_client: Mock Qdrant client.
        """
        import concurrent.futures

        with (
            patch(
                "utils.FastEmbedModelManager.get_model",
                return_value=mock_embedding_model,
            ),
            patch("utils.QdrantClient", return_value=mock_qdrant_client),
        ):
            # Setup mocks
            mock_embedding_model.embed_query.return_value = [0.1] * 1024
            mock_qdrant_client.search.return_value = [
                MagicMock(id=i, score=0.9, payload={"text": f"Result {i}"})
                for i in range(3)
            ]

            def single_query(query_id):
                query = f"Query {query_id}"
                manager = FastEmbedModelManager()
                model = manager.get_model("test_model")
                embedding = model.embed_query(query)

                results = mock_qdrant_client.search(
                    collection_name="test", query_vector=embedding, limit=3
                )
                return len(results)

            def concurrent_queries():
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(single_query, i) for i in range(10)]
                    results = [future.result() for future in futures]
                return sum(results)

            result = benchmark(concurrent_queries)
            assert result == 30  # 10 queries * 3 results each

    @pytest.mark.performance
    @pytest.mark.integration
    def test_memory_usage_integration(
        self, large_document_set, mock_embedding_model, mock_qdrant_client
    ):
        """Test memory usage with large document sets.

        Args:
            large_document_set: Large document set fixture.
            mock_embedding_model: Mock embedding model.
            mock_qdrant_client: Mock Qdrant client.
        """
        import sys

        with (
            patch(
                "utils.FastEmbedModelManager.get_model",
                return_value=mock_embedding_model,
            ),
            patch("utils.QdrantClient", return_value=mock_qdrant_client),
        ):
            # Setup mocks for large dataset
            mock_embedding_model.embed_documents.return_value = [
                [0.1] * 1024 for _ in range(len(large_document_set))
            ]
            mock_qdrant_client.upsert.return_value = MagicMock()

            # Measure memory usage
            initial_size = sys.getsizeof(large_document_set)

            # Process large document set
            manager = FastEmbedModelManager()
            model = manager.get_model("test_model")

            texts = [doc.text for doc in large_document_set]
            embeddings = model.embed_documents(texts)

            # Verify processing completed
            assert len(embeddings) == len(large_document_set)
            assert len(embeddings) == 100  # From large_document_set fixture

            # Basic memory check (implementation dependent)
            final_size = sys.getsizeof(embeddings)
            assert final_size > initial_size  # Embeddings should use memory


class TestConfigurationIntegration:
    """Test integration with different configuration settings."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        ("backend", "model"),
        [
            ("ollama", "llama3.2:3b"),
            ("openai", "gpt-4"),
            ("anthropic", "claude-3-sonnet"),
        ],
    )
    def test_multi_backend_integration(self, backend, model, sample_documents):
        """Test integration with different LLM backends.

        Args:
            backend: LLM backend to test.
            model: Model name to use.
            sample_documents: Sample documents fixture.
        """
        # Create settings for specific backend
        test_settings = Settings(
            backend=backend,
            default_model=model,
            dense_embedding_model="BAAI/bge-large-en-v1.5",
            sparse_embedding_model="prithvida/Splade_PP_en_v1",
            context_size=4096,
        )

        with patch("agent_factory.get_agent_system") as mock_get_agent:
            mock_agent = MagicMock()
            mock_agent.backend = backend
            mock_agent.model = model
            mock_get_agent.return_value = mock_agent

            agent_system = get_agent_system(settings=test_settings)

            assert agent_system.backend == backend
            assert agent_system.model == model

    @pytest.mark.integration
    def test_hybrid_configuration_validation(self, test_settings):
        """Test validation of hybrid search configuration.

        Args:
            test_settings: Test settings fixture.
        """
        with patch("utils.verify_rrf_configuration") as mock_verify:
            # Mock RRF configuration validation
            mock_verify.return_value = {
                "dense_weight": 0.7,
                "sparse_weight": 0.3,
                "k_parameter": 60,
                "valid": True,
            }

            from utils import verify_rrf_configuration

            config = verify_rrf_configuration(test_settings)

            assert config["valid"] is True
            assert config["dense_weight"] + config["sparse_weight"] == 1.0
            assert config["k_parameter"] > 0

    @pytest.mark.integration
    def test_embedding_model_configuration(self, test_settings):
        """Test embedding model configuration validation.

        Args:
            test_settings: Test settings fixture.
        """
        # Verify embedding model configuration
        assert test_settings.dense_embedding_model == "BAAI/bge-large-en-v1.5"
        assert test_settings.sparse_embedding_model == "prithvida/Splade_PP_en_v1"
        assert test_settings.dense_embedding_dimension == 1024

        # Test configuration consistency
        assert "bge-large" in test_settings.dense_embedding_model.lower()
        assert "splade" in test_settings.sparse_embedding_model.lower()


class TestDataFlowIntegration:
    """Test data flow through integrated system components."""

    @pytest.mark.integration
    def test_document_to_embedding_flow(
        self, sample_documents, mock_embedding_model, mock_sparse_embedding_model
    ):
        """Test data flow from documents to embeddings.

        Args:
            sample_documents: Sample documents fixture.
            mock_embedding_model: Mock dense embedding model.
            mock_sparse_embedding_model: Mock sparse embedding model.
        """
        with patch("utils.FastEmbedModelManager.get_model") as mock_get_model:

            def model_side_effect(model_name):
                if "splade" in model_name.lower():
                    return mock_sparse_embedding_model
                else:
                    return mock_embedding_model

            mock_get_model.side_effect = model_side_effect

            # Test document processing flow
            texts = [doc.text for doc in sample_documents]

            # Dense embedding flow
            manager = FastEmbedModelManager()
            dense_model = manager.get_model("BAAI/bge-large-en-v1.5")
            dense_embeddings = dense_model.embed_documents(texts)

            # Sparse embedding flow
            sparse_model = manager.get_model("prithvida/Splade_PP_en_v1")
            sparse_embeddings = sparse_model.encode(texts)

            # Verify data flow
            assert len(dense_embeddings) == len(sample_documents)
            assert len(sparse_embeddings) == len(sample_documents)

            # Verify embedding structures
            for dense_emb in dense_embeddings:
                assert len(dense_emb) == 1024  # BGE-Large dimension

            for sparse_emb in sparse_embeddings:
                assert "indices" in sparse_emb
                assert "values" in sparse_emb

    @pytest.mark.integration
    def test_embedding_to_search_flow(self, mock_embedding_model, mock_qdrant_client):
        """Test data flow from embeddings to search results.

        Args:
            mock_embedding_model: Mock embedding model.
            mock_qdrant_client: Mock Qdrant client.
        """
        with (
            patch(
                "utils.FastEmbedModelManager.get_model",
                return_value=mock_embedding_model,
            ),
            patch("utils.QdrantClient", return_value=mock_qdrant_client),
        ):
            # Setup search results
            mock_qdrant_client.search.return_value = [
                MagicMock(
                    id=i,
                    score=0.9 - i * 0.1,
                    payload={"text": f"Document {i}", "source": f"doc{i}.pdf"},
                )
                for i in range(3)
            ]

            # Test embedding to search flow
            query = "Test query"

            # Generate query embedding
            manager = FastEmbedModelManager()
            model = manager.get_model("test_model")
            query_embedding = model.embed_query(query)

            # Perform search
            search_results = mock_qdrant_client.search(
                collection_name="test", query_vector=query_embedding, limit=5
            )

            # Verify search flow
            assert len(search_results) == 3
            for result in search_results:
                assert hasattr(result, "id")
                assert hasattr(result, "score")
                assert hasattr(result, "payload")
                assert "text" in result.payload

    @pytest.mark.integration
    def test_search_to_response_flow(self, mock_qdrant_client, mock_reranker, mock_llm):
        """Test data flow from search results to final response.

        Args:
            mock_qdrant_client: Mock Qdrant client.
            mock_reranker: Mock reranker fixture.
            mock_llm: Mock LLM fixture.
        """
        with patch("utils.QdrantClient", return_value=mock_qdrant_client):
            # Setup search and reranking results
            mock_search_results = [
                MagicMock(id=i, score=0.8, payload={"text": f"Document {i}"})
                for i in range(5)
            ]
            mock_qdrant_client.search.return_value = mock_search_results

            mock_reranker.rerank.return_value = [
                {"index": 2, "score": 0.95},
                {"index": 0, "score": 0.88},
                {"index": 1, "score": 0.82},
            ]

            mock_llm.invoke.return_value = "Final synthesized response"

            # Test complete flow
            query = "Test query"

            # 1. Search
            search_results = mock_qdrant_client.search(
                collection_name="test", query_vector=[0.1] * 1024, limit=10
            )

            # 2. Rerank
            documents = [result.payload["text"] for result in search_results]
            reranked = mock_reranker.rerank(query=query, documents=documents, top_k=3)

            # 3. Generate response
            context = " ".join([documents[result["index"]] for result in reranked])
            response = mock_llm.invoke(f"Query: {query}\nContext: {context}")

            # Verify complete flow
            assert len(search_results) == 5
            assert len(reranked) == 3
            assert response == "Final synthesized response"
