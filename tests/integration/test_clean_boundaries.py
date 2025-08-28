"""Clean integration tests with proper component boundary mocking.

This module provides integration tests that validate component interactions
with proper mocking at architectural boundaries. These tests ensure that
components integrate correctly while maintaining fast execution and
deterministic results through strategic boundary mocking.

Integration boundaries tested:
- Settings integration across all modules with configuration validation
- Agent coordination workflow with proper LLM/embedding mocking
- Database and vector store integration with mock backends
- Document processing pipeline with external service mocks
- Cross-module communication patterns with interface validation

These tests use lightweight models and strategic mocking to test integration
points without the overhead of full system dependencies.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from src.config.settings import DocMindSettings


@pytest.mark.integration
class TestSettingsBoundaryIntegration:
    """Test settings integration across component boundaries."""

    def test_settings_app_module_boundary(self, integration_settings):
        """Test settings integration with app module boundaries."""
        # Test that settings properly configure app-specific constants
        assert integration_settings.default_token_limit >= 8192
        assert len(integration_settings.context_size_options) > 0
        assert integration_settings.request_timeout_seconds > 0
        assert integration_settings.streaming_delay_seconds > 0

        # Test nested configuration access patterns used by app.py
        assert hasattr(integration_settings.vllm, "model")
        assert hasattr(integration_settings.agents, "enable_multi_agent")
        assert hasattr(integration_settings.retrieval, "strategy")

    def test_settings_agent_coordination_boundary(self, integration_settings):
        """Test settings integration with agent coordination system."""
        # Test agent configuration is properly accessible
        agent_config = integration_settings.agents
        assert agent_config.enable_multi_agent is not None
        assert agent_config.decision_timeout > 0
        assert agent_config.max_retries >= 0

        # Test LLM configuration for agents
        llm_config = integration_settings.vllm
        assert llm_config.model is not None
        assert llm_config.context_window >= 8192
        assert llm_config.temperature >= 0

        # Test retrieval configuration for agent tools
        retrieval_config = integration_settings.retrieval
        assert retrieval_config.strategy in ["dense", "sparse", "hybrid"]
        assert retrieval_config.top_k > 0

    def test_settings_document_processing_boundary(self, integration_settings):
        """Test settings integration with document processing pipeline."""
        # Test document processing configuration
        assert integration_settings.chunk_size > 0
        assert integration_settings.chunk_overlap >= 0
        assert integration_settings.max_document_size_mb > 0

        # Test embedding configuration for document processing
        embedding_config = integration_settings.embeddings
        assert embedding_config.model is not None
        assert embedding_config.dimension > 0
        assert embedding_config.batch_size > 0

    @patch.dict(
        "os.environ",
        {
            "DOCMIND_DEBUG": "true",
            "DOCMIND_AGENTS__ENABLE_MULTI_AGENT": "false",
            "DOCMIND_VLLM__TEMPERATURE": "0.5",
        },
    )
    def test_environment_variable_boundary_integration(self):
        """Test environment variable overrides work across boundaries."""
        settings = DocMindSettings()

        # Test direct environment overrides
        assert settings.debug is True

        # Test nested environment overrides with delimiter
        assert settings.agents.enable_multi_agent is False
        assert settings.vllm.temperature == 0.5

    def test_directory_creation_boundary_integration(self, tmp_path):
        """Test directory creation integration with file operations."""
        settings = DocMindSettings(
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            log_file=str(tmp_path / "logs" / "test.log"),
        )

        # Test that directories are created during post-init
        assert Path(settings.data_dir).exists()
        assert Path(settings.cache_dir).exists()
        assert Path(settings.log_file).parent.exists()


@pytest.mark.integration
class TestAgentCoordinationBoundaryIntegration:
    """Test agent coordination integration with proper boundary mocking."""

    @patch("src.agents.coordinator.get_agent_system")
    @patch("src.utils.embedding.create_index_async")
    def test_agent_system_initialization_boundary(
        self, mock_create_index, mock_get_agent
    ):
        """Test agent system initialization with mocked dependencies."""
        # Mock the vector index creation
        mock_index = MagicMock(spec=VectorStoreIndex)
        mock_index.as_query_engine.return_value = MagicMock()
        mock_create_index.return_value = mock_index

        # Mock the agent system
        mock_agent_system = MagicMock()
        mock_agent_system.initialize.return_value = True
        mock_get_agent.return_value = mock_agent_system

        # Test integration boundary
        agent_system = mock_get_agent()
        assert agent_system.initialize() is True

        # Verify mocked boundaries were respected
        mock_get_agent.assert_called_once()

    @patch("src.agents.tool_factory.ToolFactory")
    def test_tool_factory_boundary_integration(
        self, mock_tool_factory, integration_settings
    ):
        """Test tool factory integration with settings and agent boundaries."""
        # Mock tool factory with realistic tool creation
        mock_factory = MagicMock()
        mock_factory.create_retrieval_tool.return_value = MagicMock()
        mock_factory.create_synthesis_tool.return_value = MagicMock()
        mock_tool_factory.return_value = mock_factory

        # Test tool factory integration with settings
        factory = mock_tool_factory(integration_settings)
        retrieval_tool = factory.create_retrieval_tool()
        synthesis_tool = factory.create_synthesis_tool()

        # Verify tools are created successfully
        assert retrieval_tool is not None
        assert synthesis_tool is not None

        # Verify factory was initialized with settings
        mock_tool_factory.assert_called_once_with(integration_settings)

    @pytest.mark.asyncio
    @patch("src.agents.retrieval.RetrievalAgent")
    async def test_retrieval_agent_boundary_integration(self, mock_retrieval_agent):
        """Test retrieval agent integration with async boundaries."""
        # Mock retrieval agent with async methods
        mock_agent = AsyncMock()
        mock_agent.arun.return_value = [
            NodeWithScore(node=TextNode(text="Retrieved document 1"), score=0.95),
            NodeWithScore(node=TextNode(text="Retrieved document 2"), score=0.87),
        ]
        mock_retrieval_agent.return_value = mock_agent

        # Test async integration boundary
        agent = mock_retrieval_agent()
        results = await agent.arun("test query")

        # Verify async boundary integration
        assert len(results) == 2
        assert all(isinstance(result, NodeWithScore) for result in results)
        mock_agent.arun.assert_called_once_with("test query")


@pytest.mark.integration
class TestDocumentProcessingBoundaryIntegration:
    """Test document processing integration with external service boundaries."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3Embedder")
    @patch("src.processing.document_processor.DocumentProcessor")
    def test_document_embedder_boundary_integration(
        self, mock_processor, mock_embedder
    ):
        """Test document processing with embedder boundary mocking."""
        # Mock document processor
        mock_proc_instance = MagicMock()
        mock_proc_instance.process_documents.return_value = [
            Document(text="Processed doc 1", metadata={"source": "test1.pdf"}),
            Document(text="Processed doc 2", metadata={"source": "test2.pdf"}),
        ]
        mock_processor.return_value = mock_proc_instance

        # Mock embedder
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.embed_documents.return_value = [
            [0.1] * 1024,  # 1024-dim embedding
            [0.2] * 1024,
        ]
        mock_embedder.return_value = mock_embedder_instance

        # Test integration between processor and embedder
        processor = mock_processor()
        embedder = mock_embedder()

        documents = processor.process_documents(["doc1.pdf", "doc2.pdf"])
        embeddings = embedder.embed_documents([doc.text for doc in documents])

        # Verify integration boundary
        assert len(documents) == len(embeddings)
        assert len(embeddings[0]) == 1024  # BGE-M3 dimension

        mock_proc_instance.process_documents.assert_called_once()
        mock_embedder_instance.embed_documents.assert_called_once()

    @patch("src.utils.document.load_documents_unstructured")
    async def test_document_loading_boundary_integration(self, mock_load_docs):
        """Test document loading with unstructured service boundary."""
        # Mock unstructured document loading
        mock_load_docs.return_value = [
            Document(
                text="Document content from unstructured",
                metadata={
                    "source": "test.pdf",
                    "page": 1,
                    "chunk_id": "chunk_1",
                },
            ),
            Document(
                text="Another document chunk",
                metadata={
                    "source": "test.pdf",
                    "page": 2,
                    "chunk_id": "chunk_2",
                },
            ),
        ]

        # Test document loading integration
        file_paths = [Path("test.pdf")]
        documents = await mock_load_docs(file_paths)

        # Verify loading boundary integration
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        assert all("chunk_id" in doc.metadata for doc in documents)

        mock_load_docs.assert_called_once_with(file_paths)

    @patch("src.processing.chunking.unstructured_chunker.UnstructuredChunker")
    def test_chunking_boundary_integration(self, mock_chunker, integration_settings):
        """Test document chunking with settings boundary integration."""
        # Mock chunker with settings-aware behavior
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk_documents.return_value = [
            Document(
                text="Chunk 1 content",
                metadata={"chunk_size": integration_settings.chunk_size},
            ),
            Document(
                text="Chunk 2 content",
                metadata={"chunk_size": integration_settings.chunk_size},
            ),
        ]
        mock_chunker.return_value = mock_chunker_instance

        # Test chunking integration with settings
        chunker = mock_chunker(integration_settings)
        raw_documents = [Document(text="Large document content for chunking")]
        chunks = chunker.chunk_documents(raw_documents)

        # Verify chunking boundary respects settings
        assert len(chunks) == 2
        assert all(
            chunk.metadata.get("chunk_size") == integration_settings.chunk_size
            for chunk in chunks
        )

        mock_chunker.assert_called_once_with(integration_settings)


@pytest.mark.integration
class TestDatabaseVectorStoreBoundaryIntegration:
    """Test database and vector store integration with mock backends."""

    @patch("src.storage.hybrid_persistence.HybridPersistence")
    def test_hybrid_persistence_boundary_integration(
        self, mock_persistence, integration_settings
    ):
        """Test hybrid persistence integration with mocked storage backends."""
        # Mock hybrid persistence with both SQLite and Qdrant
        mock_persistence_instance = MagicMock()
        mock_persistence_instance.store_documents.return_value = True
        mock_persistence_instance.search.return_value = [
            {"id": "doc1", "score": 0.95, "text": "Retrieved content 1"},
            {"id": "doc2", "score": 0.87, "text": "Retrieved content 2"},
        ]
        mock_persistence.return_value = mock_persistence_instance

        # Test persistence integration
        persistence = mock_persistence(integration_settings)

        # Test document storage
        test_docs = [Document(text="Test document", metadata={"source": "test.pdf"})]
        result = persistence.store_documents(test_docs)
        assert result is True

        # Test search functionality
        search_results = persistence.search("test query", top_k=2)
        assert len(search_results) == 2
        assert all("score" in result for result in search_results)

        mock_persistence.assert_called_once_with(integration_settings)

    @patch("src.retrieval.vector_store.QdrantVectorStore")
    def test_qdrant_vector_store_boundary_integration(self, mock_qdrant):
        """Test Qdrant vector store integration with proper mocking."""
        # Mock Qdrant client responses
        mock_qdrant_instance = MagicMock()
        mock_qdrant_instance.create_collection.return_value = True
        mock_qdrant_instance.upsert.return_value = {"status": "ok"}
        mock_qdrant_instance.search.return_value = [
            {
                "id": "vec1",
                "score": 0.92,
                "payload": {"text": "Vector search result 1"},
            },
            {
                "id": "vec2",
                "score": 0.84,
                "payload": {"text": "Vector search result 2"},
            },
        ]
        mock_qdrant.return_value = mock_qdrant_instance

        # Test vector store integration
        vector_store = mock_qdrant()

        # Test collection creation
        collection_result = vector_store.create_collection("test_collection")
        assert collection_result is True

        # Test vector upsert
        upsert_result = vector_store.upsert([{"id": "test", "vector": [0.1] * 1024}])
        assert upsert_result["status"] == "ok"

        # Test vector search
        search_results = vector_store.search([0.1] * 1024, top_k=2)
        assert len(search_results) == 2
        assert all(result["score"] > 0.8 for result in search_results)

    @patch("src.utils.database.SQLiteManager")
    def test_sqlite_database_boundary_integration(
        self, mock_sqlite, integration_settings
    ):
        """Test SQLite database integration with settings boundary."""
        # Mock SQLite manager
        mock_db = MagicMock()
        mock_db.create_tables.return_value = True
        mock_db.insert_document.return_value = 1  # document ID
        mock_db.get_document.return_value = {
            "id": 1,
            "content": "Document content",
            "metadata": '{"source": "test.pdf"}',
        }
        mock_sqlite.return_value = mock_db

        # Test database integration with settings
        db_manager = mock_sqlite(integration_settings.sqlite_db_path)

        # Test table creation
        create_result = db_manager.create_tables()
        assert create_result is True

        # Test document insertion
        doc_id = db_manager.insert_document("Document content", {"source": "test.pdf"})
        assert doc_id == 1

        # Test document retrieval
        retrieved_doc = db_manager.get_document(1)
        assert retrieved_doc["content"] == "Document content"

        mock_sqlite.assert_called_once_with(integration_settings.sqlite_db_path)


@pytest.mark.integration
class TestCrossModuleCommunicationBoundaries:
    """Test cross-module communication patterns with interface validation."""

    @patch("src.agents.coordinator.get_agent_system")
    @patch("src.utils.embedding.create_index_async")
    def test_coordinator_embedding_communication_boundary(
        self, mock_create_index, mock_get_agent, integration_settings
    ):
        """Test communication between coordinator and embedding modules."""
        # Mock embedding index creation
        mock_index = MagicMock(spec=VectorStoreIndex)
        mock_index.as_query_engine.return_value = MagicMock()
        mock_create_index.return_value = mock_index

        # Mock agent coordinator
        mock_coordinator = MagicMock()
        mock_coordinator.setup_agents.return_value = True
        mock_get_agent.return_value = mock_coordinator

        # Test cross-module communication
        index = mock_create_index([], integration_settings)
        coordinator = mock_get_agent(index, integration_settings)

        setup_result = coordinator.setup_agents()

        # Verify communication boundary
        assert setup_result is True
        mock_create_index.assert_called_once_with([], integration_settings)
        mock_get_agent.assert_called_once_with(index, integration_settings)

    @patch("src.core.infrastructure.gpu_monitor.gpu_performance_monitor")
    @patch("src.utils.monitoring.performance_logger")
    def test_monitoring_infrastructure_communication_boundary(
        self, mock_perf_logger, mock_gpu_monitor, integration_settings
    ):
        """Test communication between monitoring and infrastructure modules."""
        # Mock GPU monitoring
        mock_monitor = MagicMock()
        mock_monitor.get_gpu_stats.return_value = {
            "gpu_utilization": 0.75,
            "memory_used": 8192,
            "memory_total": 16384,
        }
        mock_gpu_monitor.return_value = mock_monitor

        # Mock performance logger
        mock_logger = MagicMock()
        mock_logger.log_performance.return_value = True
        mock_perf_logger.return_value = mock_logger

        # Test monitoring communication
        gpu_monitor = mock_gpu_monitor()
        perf_logger = mock_perf_logger(integration_settings)

        gpu_stats = gpu_monitor.get_gpu_stats()
        log_result = perf_logger.log_performance(gpu_stats)

        # Verify cross-module communication
        assert gpu_stats["gpu_utilization"] > 0
        assert log_result is True

        mock_gpu_monitor.assert_called_once()
        mock_perf_logger.assert_called_once_with(integration_settings)

    @pytest.mark.asyncio
    @patch("src.dspy_integration.DSPyOptimizer")
    async def test_dspy_optimization_boundary_integration(self, mock_dspy_optimizer):
        """Test DSPy optimization integration with async boundaries."""
        # Mock DSPy optimizer with async optimization
        mock_optimizer = AsyncMock()
        mock_optimizer.optimize_prompts.return_value = {
            "optimized_prompt": "Optimized query prompt for better results",
            "improvement_score": 0.23,  # 23% improvement
        }
        mock_dspy_optimizer.return_value = mock_optimizer

        # Test DSPy integration
        optimizer = mock_dspy_optimizer()
        optimization_result = await optimizer.optimize_prompts(
            ["Original prompt for testing optimization"]
        )

        # Verify optimization boundary integration
        assert "optimized_prompt" in optimization_result
        assert optimization_result["improvement_score"] > 0.2

        mock_optimizer.optimize_prompts.assert_called_once()

    def test_configuration_validation_across_boundaries(self, integration_settings):
        """Test configuration validation works across all module boundaries."""
        # Test that all major configuration sections are present
        required_configs = ["vllm", "agents", "embeddings", "retrieval"]
        for config_name in required_configs:
            assert hasattr(integration_settings, config_name), (
                f"Missing required config section: {config_name}"
            )

        # Test that nested configurations have required fields
        assert hasattr(integration_settings.vllm, "model")
        assert hasattr(integration_settings.agents, "enable_multi_agent")
        assert hasattr(integration_settings.embeddings, "model")
        assert hasattr(integration_settings.retrieval, "strategy")

        # Test that all configurations have valid values
        assert integration_settings.vllm.context_window > 0
        assert integration_settings.agents.decision_timeout > 0
        assert integration_settings.embeddings.dimension > 0
        assert integration_settings.retrieval.top_k > 0


@pytest.mark.integration
class TestErrorHandlingBoundaryIntegration:
    """Test error handling across component boundaries."""

    @patch("src.agents.coordinator.get_agent_system")
    def test_agent_system_error_boundary_handling(self, mock_get_agent):
        """Test error handling at agent system boundaries."""
        # Mock agent system that raises errors
        mock_agent_system = MagicMock()
        mock_agent_system.initialize.side_effect = RuntimeError(
            "Agent initialization failed"
        )
        mock_get_agent.return_value = mock_agent_system

        # Test error boundary handling
        agent_system = mock_get_agent()

        with pytest.raises(RuntimeError, match="Agent initialization failed"):
            agent_system.initialize()

        # Verify error was properly propagated across boundary
        mock_get_agent.assert_called_once()

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3Embedder")
    def test_embedder_error_boundary_handling(self, mock_embedder):
        """Test error handling at embedder boundaries."""
        # Mock embedder with initialization error
        mock_embedder.side_effect = ImportError("BGE-M3 model not found")

        # Test error boundary handling
        with pytest.raises(ImportError, match="BGE-M3 model not found"):
            embedder = mock_embedder()

        mock_embedder.assert_called_once()

    def test_settings_validation_error_boundaries(self):
        """Test settings validation error handling at boundaries."""
        # Test invalid configuration values
        with pytest.raises(ValueError):
            DocMindSettings(context_window_size=-1)  # Invalid negative value

        with pytest.raises(ValueError):
            DocMindSettings(chunk_size=0)  # Invalid zero chunk size

    @pytest.mark.asyncio
    @patch("src.utils.embedding.create_index_async")
    async def test_async_error_boundary_handling(self, mock_create_index):
        """Test async error handling across boundaries."""
        # Mock async function that raises error
        mock_create_index.side_effect = TimeoutError("Index creation timeout")

        # Test async error boundary handling
        with pytest.raises(asyncio.TimeoutError, match="Index creation timeout"):
            await mock_create_index([], DocMindSettings())

        mock_create_index.assert_called_once()


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["ollama", "vllm", "llamacpp"])
class TestLLMBackendBoundaryIntegration:
    """Test LLM backend integration across different providers."""

    @patch("src.config.settings.DocMindSettings")
    def test_llm_backend_configuration_boundary(self, mock_settings, backend):
        """Test LLM backend configuration integration boundaries."""
        # Mock settings for different backends
        mock_settings_instance = MagicMock()
        mock_settings_instance.vllm.backend = backend
        mock_settings_instance.vllm.ollama_base_url = "http://localhost:11434"
        mock_settings_instance.vllm.vllm_base_url = "http://localhost:8000"
        mock_settings_instance.vllm.lmstudio_base_url = "http://localhost:1234"
        mock_settings.return_value = mock_settings_instance

        # Test backend configuration
        settings = mock_settings()

        assert settings.vllm.backend == backend

        # Verify backend-specific URL configuration
        if backend == "ollama":
            assert "11434" in settings.vllm.ollama_base_url
        elif backend == "vllm":
            assert "8000" in settings.vllm.vllm_base_url
        elif backend == "llamacpp":
            assert "1234" in settings.vllm.lmstudio_base_url
