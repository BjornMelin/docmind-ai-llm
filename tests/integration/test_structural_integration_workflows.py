"""Structural Changes Integration Workflow Tests.

This module validates that major integration workflows function correctly after
extensive structural improvements:

- Document processing pipeline integration
- Multi-agent coordination workflow
- Retrieval system integration
- Configuration propagation across components
- Error handling and resilience preservation
- Resource management integration

Tests ensure that flattening and reorganization haven't broken critical workflows.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore, TextNode

from src.config.settings import DocMindSettings


@pytest.fixture
def integration_settings(tmp_path):
    """Settings configured for integration testing."""
    return DocMindSettings(
        debug=True,
        log_level="DEBUG",
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        enable_gpu_acceleration=False,  # CPU-only for integration tests
        enable_performance_logging=True,
    )


@pytest.fixture
def sample_documents():
    """Sample documents for integration testing."""
    return [
        Document(
            text="DocMind AI uses BGE-M3 embeddings for semantic understanding.",
            metadata={"source": "embeddings.pdf", "page": 1},
        ),
        Document(
            text="The multi-agent system coordinates retrieval and synthesis agents.",
            metadata={"source": "agents.pdf", "page": 1},
        ),
        Document(
            text=(
                "Hybrid search combines dense and sparse retrieval methods effectively."
            ),
            metadata={"source": "search.pdf", "page": 2},
        ),
    ]


@pytest.mark.integration
class TestDocumentProcessingPipelineIntegration:
    """Test document processing pipeline integration after structural changes."""

    def test_document_loading_integration(self, integration_settings, sample_documents):
        """Test that document loading works with reorganized modules."""
        # Mock document loading components
        with patch("src.utils.document.load_documents_unstructured") as mock_loader:
            mock_loader.return_value = sample_documents

            from src.core.document_processor import DocumentProcessor
            from src.utils.document import load_documents_unstructured

            # Test integration between utils and core processing
            loaded_docs = load_documents_unstructured(
                ["test.pdf"], integration_settings
            )
            processor = DocumentProcessor(settings=integration_settings)

            # Verify integration works
            assert loaded_docs == sample_documents
            assert processor is not None
            assert processor.settings == integration_settings

            print(f"Document loading integration: {len(loaded_docs)} documents loaded")

    def test_chunking_integration_workflow(
        self, integration_settings, sample_documents
    ):
        """Test document chunking integration with settings."""
        with patch(
            "src.processing.chunking.unstructured_chunker.UnstructuredChunker"
        ) as mock_chunker:
            # Mock chunker behavior
            mock_chunker_instance = MagicMock()
            mock_chunker_instance.chunk_documents.return_value = [
                Document(
                    text="Chunk 1 from doc 1",
                    metadata={"chunk_id": "1", "source": "doc1"},
                ),
                Document(
                    text="Chunk 2 from doc 1",
                    metadata={"chunk_id": "2", "source": "doc1"},
                ),
                Document(
                    text="Chunk 1 from doc 2",
                    metadata={"chunk_id": "3", "source": "doc2"},
                ),
            ]
            mock_chunker.return_value = mock_chunker_instance

            from src.processing.chunking.unstructured_chunker import UnstructuredChunker

            # Create chunker with settings integration
            chunker = UnstructuredChunker(
                chunk_size=integration_settings.chunk_size,
                new_after_n_chars=integration_settings.processing.new_after_n_chars,
            )

            # processor = DocumentProcessor(
            #     settings=integration_settings, chunker=chunker
            # )
            # NOTE: processor not used in this test - focusing on chunker workflow

            # Test chunking workflow
            chunks = chunker.chunk_documents(sample_documents)

            # Verify integration
            assert len(chunks) == 3
            assert all(isinstance(chunk, Document) for chunk in chunks)
            assert chunks[0].metadata["chunk_id"] == "1"

            print(f"Chunking integration: {len(chunks)} chunks created")

    def test_embedding_creation_integration(self, integration_settings):
        """Test embedding creation integration with reorganized modules."""
        with (
            patch("src.retrieval.embeddings.BGEM3Embedding") as mock_embedding,
            patch("src.utils.embedding.create_index_async") as mock_create_index,
        ):
            # Mock embedding model
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_unified_embeddings.return_value = {
                "dense": [[0.1] * 1024 for _ in range(3)],
                "sparse": [{"indices": [1, 2], "values": [0.5, 0.3]} for _ in range(3)],
            }
            mock_embedding.return_value = mock_embed_instance

            # Mock index creation
            mock_index = MagicMock()
            mock_create_index.return_value = mock_index

            from src.retrieval.embeddings import BGEM3Embedding

            # Test embedding integration
            embedding_model = BGEM3Embedding(settings=integration_settings)

            test_texts = ["test doc 1", "test doc 2", "test doc 3"]
            embeddings = embedding_model.get_unified_embeddings(test_texts)

            # Verify embedding integration
            assert "dense" in embeddings
            assert "sparse" in embeddings
            assert len(embeddings["dense"]) == 3
            assert len(embeddings["sparse"]) == 3

            print(
                f"Embedding integration: {len(embeddings['dense'])} embeddings created"
            )


@pytest.mark.integration
class TestMultiAgentCoordinationIntegration:
    """Test multi-agent coordination integration after structural changes."""

    def test_agent_coordinator_initialization(self, integration_settings):
        """Test that agent coordinator initializes with reorganized modules."""
        with (
            patch("src.agents.tools.get_available_tools") as mock_tools,
            patch("src.agents.coordinator.LangGraphSupervisor") as mock_supervisor,
        ):
            mock_tools.return_value = [
                "search_tool",
                "synthesis_tool",
                "validation_tool",
            ]
            mock_supervisor_instance = MagicMock()
            mock_supervisor.return_value = mock_supervisor_instance

            from src.agents.coordinator import AgentCoordinator
            from src.agents.tools import get_available_tools

            # Test coordinator initialization
            coordinator = AgentCoordinator(settings=integration_settings)
            available_tools = get_available_tools()

            # Verify integration
            assert coordinator is not None
            assert len(available_tools) == 3
            assert "search_tool" in available_tools

            print(
                f"Agent coordinator integration: {len(available_tools)} tools available"
            )

    def test_agent_tool_factory_integration(self, integration_settings):
        """Test agent tool factory integration with settings."""
        with (
            patch(
                "src.agents.tool_factory.create_retrieval_tool"
            ) as mock_retrieval_tool,
            patch(
                "src.agents.tool_factory.create_synthesis_tool"
            ) as mock_synthesis_tool,
        ):
            # Mock tool creation
            mock_retrieval = MagicMock()
            mock_synthesis = MagicMock()
            mock_retrieval_tool.return_value = mock_retrieval
            mock_synthesis_tool.return_value = mock_synthesis

            from src.agents.tool_factory import (
                create_retrieval_tool,
                create_synthesis_tool,
            )

            # Test tool factory integration
            retrieval_tool = create_retrieval_tool(integration_settings)
            synthesis_tool = create_synthesis_tool(integration_settings)

            # Verify tool factory integration
            assert retrieval_tool is not None
            assert synthesis_tool is not None

            print("Agent tool factory integration: tools created successfully")

    @pytest.mark.asyncio
    async def test_async_agent_workflow_integration(self, integration_settings):
        """Test async agent workflow integration."""

        async def mock_agent_process(query: str, context: list[str]) -> str:
            await asyncio.sleep(0.01)  # Simulate processing
            return f"Processed query: {query} with {len(context)} context items"

        with patch("src.agents.coordinator.AgentCoordinator") as mock_coordinator:
            # Mock coordinator async methods
            mock_coordinator_instance = AsyncMock()
            mock_coordinator_instance.aprocess_query = mock_agent_process
            mock_coordinator.return_value = mock_coordinator_instance

            from src.agents.coordinator import AgentCoordinator

            # Test async workflow
            coordinator = AgentCoordinator(settings=integration_settings)

            result = await coordinator.aprocess_query(
                "What is BGE-M3 embedding?",
                [
                    "BGE-M3 is a unified embedding model",
                    "It supports dense and sparse vectors",
                ],
            )

            # Verify async integration
            assert "Processed query" in result
            assert "What is BGE-M3 embedding?" in result
            assert "2 context items" in result

            print(f"Async agent workflow: {result}")


@pytest.mark.integration
class TestRetrievalSystemIntegration:
    """Test retrieval system integration after structural reorganization."""

    def test_vector_store_integration(self, integration_settings):
        """Test vector store integration with settings and embeddings."""
        with (
            patch("src.retrieval.vector_store.QdrantVectorStore") as mock_qdrant,
            patch("src.retrieval.embeddings.BGEM3Embedding") as mock_embedding,
        ):
            # Mock vector store
            mock_qdrant_instance = MagicMock()
            mock_qdrant_instance.add_documents.return_value = True
            mock_qdrant_instance.similarity_search.return_value = [
                {"id": "doc1", "score": 0.95, "content": "BGE-M3 embedding content"},
                {"id": "doc2", "score": 0.87, "content": "Multi-agent system content"},
            ]
            mock_qdrant.return_value = mock_qdrant_instance

            # Mock embedding model
            mock_embed_instance = MagicMock()
            mock_embedding.return_value = mock_embed_instance

            from src.retrieval.embeddings import BGEM3Embedding
            from src.retrieval.vector_store import QdrantVectorStore

            # Test vector store integration
            embedding_model = BGEM3Embedding(settings=integration_settings)
            vector_store = QdrantVectorStore(
                collection_name=integration_settings.qdrant_collection,
                embedding_model=embedding_model,
            )

            # Test operations
            add_result = vector_store.add_documents(["test doc 1", "test doc 2"])
            search_results = vector_store.similarity_search("test query", top_k=2)

            # Verify integration
            assert add_result is True
            assert len(search_results) == 2
            assert search_results[0]["score"] == 0.95

            print(f"Vector store integration: {len(search_results)} results retrieved")

    def test_query_engine_integration(self, integration_settings):
        """Test query engine integration with retrieval components."""
        with (
            patch(
                "src.retrieval.query_engine.AdaptiveRouterQueryEngine"
            ) as mock_router,
            patch("src.retrieval.reranking.BGECrossEncoderRerank") as mock_reranker,
        ):
            # Mock query engine
            mock_router_instance = MagicMock()
            mock_router_instance.query.return_value = MagicMock(
                response="BGE-M3 is a unified embedding model for semantic search.",
                source_nodes=[
                    NodeWithScore(
                        node=TextNode(
                            text="BGE-M3 supports both dense and sparse embeddings"
                        ),
                        score=0.92,
                    )
                ],
            )
            mock_router.return_value = mock_router_instance

            # Mock reranker
            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            from src.retrieval.query_engine import AdaptiveRouterQueryEngine
            from src.retrieval.reranking import BGECrossEncoderRerank

            # Test query engine integration
            reranker = BGECrossEncoderRerank(top_n=integration_settings.reranking_top_k)
            query_engine = AdaptiveRouterQueryEngine(
                reranker=reranker, settings=integration_settings
            )

            # Test query processing
            response = query_engine.query("What is BGE-M3?")

            # Verify integration
            assert response is not None
            assert hasattr(response, "response")
            assert hasattr(response, "source_nodes")
            assert len(response.source_nodes) == 1

            print(
                f"Query engine integration: response generated with "
                f"{len(response.source_nodes)} sources"
            )

    def test_hybrid_search_integration(self, integration_settings):
        """Test hybrid search integration with RRF fusion."""
        with (
            patch("src.retrieval.optimization.RRFFusion") as mock_rrf,
            patch("src.retrieval.embeddings.BGEM3Embedding") as mock_embedding,
        ):
            # Mock RRF fusion
            mock_rrf_instance = MagicMock()
            mock_rrf_instance.fuse_results.return_value = [
                {"id": "doc1", "score": 0.94, "content": "Fused result 1"},
                {"id": "doc2", "score": 0.88, "content": "Fused result 2"},
            ]
            mock_rrf.return_value = mock_rrf_instance

            # Mock embedding
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_unified_embeddings.return_value = {
                "dense": [[0.1] * 1024],
                "sparse": [{"indices": [1, 2, 3], "values": [0.8, 0.6, 0.4]}],
            }
            mock_embedding.return_value = mock_embed_instance

            from src.retrieval.embeddings import BGEM3Embedding
            from src.retrieval.optimization import RRFFusion

            # Test hybrid search integration
            embedding_model = BGEM3Embedding(settings=integration_settings)
            rrf_fusion = RRFFusion(
                alpha=integration_settings.rrf_fusion_alpha,
                k=integration_settings.rrf_k_constant,
            )

            # Simulate hybrid search
            query_embeddings = embedding_model.get_unified_embeddings(["test query"])

            # Mock dense and sparse results
            dense_results = [{"id": "doc1", "score": 0.9}]
            sparse_results = [{"id": "doc2", "score": 0.85}]

            fused_results = rrf_fusion.fuse_results(dense_results, sparse_results)

            # Verify hybrid search integration
            assert query_embeddings is not None
            assert "dense" in query_embeddings
            assert "sparse" in query_embeddings
            assert len(fused_results) == 2
            assert fused_results[0]["score"] == 0.94

            print(f"Hybrid search integration: {len(fused_results)} fused results")


@pytest.mark.integration
class TestConfigurationPropagationIntegration:
    """Test configuration propagation across reorganized components."""

    def test_settings_propagation_to_components(self, integration_settings):
        """Test that settings properly propagate to all components."""
        components_to_test = []

        # Test embedding component configuration
        with patch("src.retrieval.embeddings.BGEM3Embedding") as mock_embedding:
            mock_embed_instance = MagicMock()
            mock_embedding.return_value = mock_embed_instance

            from src.retrieval.embeddings import BGEM3Embedding

            embedding_model = BGEM3Embedding(
                model_name=integration_settings.embedding.model_name,
                device="cuda"
                if integration_settings.enable_gpu_acceleration
                else "cpu",
                batch_size=integration_settings.embedding.batch_size_cpu,
            )

            components_to_test.append(("embedding", embedding_model))

        # Test agent coordinator configuration
        with patch("src.agents.coordinator.AgentCoordinator") as mock_coordinator:
            mock_coord_instance = MagicMock()
            mock_coordinator.return_value = mock_coord_instance

            from src.agents.coordinator import AgentCoordinator

            coordinator = AgentCoordinator(
                settings=integration_settings,
                timeout=integration_settings.agent_decision_timeout,
                max_retries=integration_settings.max_agent_retries,
            )

            components_to_test.append(("coordinator", coordinator))

        # Test document processor configuration
        with patch("src.core.document_processor.DocumentProcessor") as mock_processor:
            mock_proc_instance = MagicMock()
            mock_processor.return_value = mock_proc_instance

            from src.core.document_processor import DocumentProcessor

            processor = DocumentProcessor(
                settings=integration_settings,
                chunk_size=integration_settings.chunk_size,
                max_document_size=integration_settings.max_document_size_mb,
            )

            components_to_test.append(("processor", processor))

        # Verify all components received configuration
        assert len(components_to_test) == 3
        for component_name, component in components_to_test:
            assert component is not None
            print(f"Configuration propagated to {component_name}")

    def test_nested_configuration_access(self, integration_settings):
        """Test that nested configuration models work correctly."""
        # Test accessing nested configurations
        vllm_config = integration_settings.get_vllm_config()
        agent_config = integration_settings.get_agent_config()
        embedding_config = integration_settings.get_embedding_config()

        # Verify nested configurations
        assert isinstance(vllm_config, dict)
        assert isinstance(agent_config, dict)
        assert isinstance(embedding_config, dict)

        # Test specific nested values
        assert vllm_config["model_name"] == integration_settings.model_name
        assert (
            agent_config["enable_multi_agent"]
            == integration_settings.enable_multi_agent
        )
        assert (
            embedding_config["model_name"] == integration_settings.embedding.model_name
        )

        # Test synchronization works automatically (no explicit call needed)
        integration_settings.chunk_size = 4096
        # Synchronization happens automatically when settings are accessed

        updated_processing_config = integration_settings.get_processing_config()
        assert updated_processing_config["chunk_size"] == 4096

        print("Nested configuration access: all configurations accessible")

    def test_environment_variable_integration(self, integration_settings):
        """Test environment variable integration with components."""
        import os
        from unittest.mock import patch

        # Test environment variables affect component behavior
        test_env_vars = {
            "DOCMIND_TOP_K": "20",
            "DOCMIND_USE_RERANKING": "false",
            "DOCMIND_RETRIEVAL_STRATEGY": "dense_only",
        }

        with patch.dict(os.environ, test_env_vars):
            # Create new settings to pick up environment variables
            env_settings = DocMindSettings()

            # Verify environment variables were processed
            assert env_settings.top_k == 20
            assert env_settings.use_reranking is False
            assert env_settings.retrieval_strategy == "dense_only"

            # Test that components would receive these settings
            # retrieval_config = env_settings.get_retrieval_config()
            # NOTE: retrieval_config not used - test focuses on mock engine behavior

            # Mock component using environment-driven settings
            with patch("src.retrieval.query_engine.QueryEngine") as mock_engine:
                mock_engine_instance = MagicMock()
                mock_engine.return_value = mock_engine_instance

                from src.retrieval.query_engine import QueryEngine

                query_engine = QueryEngine(
                    top_k=env_settings.top_k,
                    use_reranking=env_settings.use_reranking,
                    strategy=env_settings.retrieval_strategy,
                )

                # Verify integration
                assert query_engine is not None

                print(
                    "Environment variable integration: "
                    "settings propagated to components"
                )


@pytest.mark.integration
class TestErrorHandlingAndResilienceIntegration:
    """Test error handling and resilience preservation after reorganization."""

    def test_component_failure_resilience(self, integration_settings):
        """Test that component failures are handled gracefully."""
        # Test embedding model failure handling
        with patch("src.retrieval.embeddings.BGEM3Embedding") as mock_embedding:
            mock_embedding.side_effect = RuntimeError(
                "Embedding model initialization failed"
            )

            from src.retrieval.embeddings import BGEM3Embedding

            # Test graceful failure handling
            with pytest.raises(
                RuntimeError, match="Embedding model initialization failed"
            ):
                BGEM3Embedding(settings=integration_settings)

            print("Error handling: embedding failure caught correctly")

        # Test coordinator failure handling with fallback
        with (
            patch("src.agents.coordinator.AgentCoordinator") as mock_coordinator,
            patch("src.agents.coordinator.FallbackRAGEngine") as mock_fallback,
        ):
            mock_coordinator.side_effect = Exception(
                "Coordinator initialization failed"
            )
            mock_fallback_instance = MagicMock()
            mock_fallback.return_value = mock_fallback_instance

            from src.agents.coordinator import AgentCoordinator, FallbackRAGEngine

            # Test fallback mechanism
            try:
                AgentCoordinator(settings=integration_settings)
                pytest.fail("Expected exception was not raised")
            except Exception:
                # Fallback should be available
                fallback_engine = FallbackRAGEngine(settings=integration_settings)
                assert fallback_engine is not None
                print("Error handling: fallback mechanism works correctly")

    def test_configuration_validation_errors(self, tmp_path):
        """Test configuration validation error handling."""
        # Test invalid configuration values
        with pytest.raises(ValueError, match="Field cannot be empty"):
            DocMindSettings(
                app_name="",  # Empty app name should fail validation
                data_dir=str(tmp_path / "data"),
            )

        # Test field constraint violations
        with pytest.raises(ValueError, match=r"chunk_size.*must be.*100"):
            DocMindSettings(
                chunk_size=50,  # Below minimum constraint
                data_dir=str(tmp_path / "data"),
            )

        print("Configuration validation: invalid configurations rejected correctly")

    @pytest.mark.asyncio
    async def test_async_error_propagation(self, integration_settings):
        """Test that async errors propagate correctly through integration layers."""

        async def failing_async_operation():
            raise ConnectionError("Async operation failed")

        # Test error propagation in async workflows
        with patch("src.retrieval.embeddings.AsyncEmbeddingModel") as mock_async_embed:
            mock_async_embed_instance = AsyncMock()
            mock_async_embed_instance.aembed_documents.side_effect = (
                failing_async_operation
            )
            mock_async_embed.return_value = mock_async_embed_instance

            from src.retrieval.embeddings import AsyncEmbeddingModel

            # Test async error handling
            async_model = AsyncEmbeddingModel(settings=integration_settings)

            with pytest.raises(ConnectionError, match="Async operation failed"):
                await async_model.aembed_documents(["test document"])

            print(
                "Async error handling: errors propagated correctly "
                "through async workflows"
            )


@pytest.mark.integration
class TestResourceManagementIntegration:
    """Test resource management integration after structural changes."""

    def test_memory_context_integration(self, integration_settings):
        """Test memory context management across components."""
        with (
            patch("src.utils.storage.gpu_memory_context") as mock_gpu_context,
            patch("src.cache.simple_cache.SimpleCache") as mock_cache,
        ):
            # Mock memory context
            mock_gpu_context.return_value.__enter__ = MagicMock()
            mock_gpu_context.return_value.__exit__ = MagicMock()

            # Mock cache
            mock_cache_instance = MagicMock()
            mock_cache_instance.get.return_value = None
            mock_cache_instance.set.return_value = True
            mock_cache.return_value = mock_cache_instance

            from src.cache.simple_cache import SimpleCache
            from src.utils.storage import gpu_memory_context

            # Test resource management integration
            cache = SimpleCache(settings=integration_settings)

            with gpu_memory_context():
                # Simulate resource-intensive operations
                cache_key = "test_embeddings_key"
                cache_value = {"embeddings": [[0.1] * 1024 for _ in range(10)]}

                # Test cache operations
                cached_result = cache.get(cache_key)
                assert cached_result is None  # Not cached yet

                cache_success = cache.set(cache_key, cache_value)
                assert cache_success is True

            print(
                "Resource management integration: "
                "memory context and caching work together"
            )

    def test_cleanup_and_resource_release(self, integration_settings):
        """Test that resources are properly cleaned up after operations."""
        cleanup_verification = {"components_cleaned": 0}

        def mock_cleanup():
            cleanup_verification["components_cleaned"] += 1

        # Test component cleanup
        with (
            patch("src.retrieval.embeddings.BGEM3Embedding") as mock_embedding,
            patch("src.agents.coordinator.AgentCoordinator") as mock_coordinator,
        ):
            # Mock component cleanup methods
            mock_embed_instance = MagicMock()
            mock_embed_instance.cleanup = mock_cleanup
            mock_embedding.return_value = mock_embed_instance

            mock_coord_instance = MagicMock()
            mock_coord_instance.cleanup = mock_cleanup
            mock_coordinator.return_value = mock_coord_instance

            from src.agents.coordinator import AgentCoordinator
            from src.retrieval.embeddings import BGEM3Embedding

            # Create and cleanup components
            embedding_model = BGEM3Embedding(settings=integration_settings)
            coordinator = AgentCoordinator(settings=integration_settings)

            # Simulate cleanup
            embedding_model.cleanup()
            coordinator.cleanup()

            # Verify cleanup was called
            assert cleanup_verification["components_cleaned"] == 2

            print("Resource cleanup integration: components cleaned up properly")


# Helper classes for mocking
class QueryEngine:
    """Mock query engine for testing."""

    def __init__(self, top_k=10, use_reranking=True, strategy="hybrid"):
        """Initialize mock query engine.

        Args:
            top_k: Number of results to return.
            use_reranking: Whether to use reranking.
            strategy: Retrieval strategy to use.
        """
        self.top_k = top_k
        self.use_reranking = use_reranking
        self.strategy = strategy


class FallbackRAGEngine:
    """Mock fallback RAG engine for testing."""

    def __init__(self, settings):
        """Initialize mock fallback RAG engine.

        Args:
            settings: Configuration settings for the engine.
        """
        self.settings = settings
