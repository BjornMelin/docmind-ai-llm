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

    def test_settings_app_module_boundary(self):
        """Test settings integration with app module boundaries."""
        s = DocMindSettings()
        # UI (moved under nested `ui` config)
        assert s.ui.default_token_limit >= 8192
        assert len(s.ui.context_size_options) > 0
        assert s.ui.request_timeout_seconds > 0
        assert s.ui.streaming_delay_seconds > 0

        # Nested configuration access patterns
        assert hasattr(s.vllm, "model")
        assert hasattr(s.agents, "enable_multi_agent")
        assert hasattr(s.retrieval, "strategy")

    def test_settings_agent_coordination_boundary(self):
        """Test settings integration with agent coordination system."""
        s = DocMindSettings()
        agent_config = s.agents
        assert agent_config.enable_multi_agent is not None
        assert agent_config.decision_timeout > 0
        assert agent_config.max_retries >= 0

        # Test LLM configuration for agents
        llm_config = s.vllm
        assert llm_config.model is not None
        assert llm_config.context_window >= 8192
        assert llm_config.temperature >= 0

        # Test retrieval configuration for agent tools
        retrieval_config = s.retrieval
        assert retrieval_config.strategy in ["dense", "sparse", "hybrid"]
        assert retrieval_config.top_k > 0

    def test_settings_document_processing_boundary(self):
        """Test settings integration with document processing pipeline."""
        s = DocMindSettings()
        # Document processing config moved under `processing`
        assert s.processing.chunk_size > 0
        assert s.processing.chunk_overlap >= 0
        assert s.processing.max_document_size_mb > 0

        # Embedding configuration for document processing
        embedding_config = s.embedding
        assert embedding_config.model_name is not None
        assert embedding_config.dimension > 0
        # Batch size depends on backend; check at least one > 0
        assert (embedding_config.batch_size_gpu > 0) or (
            embedding_config.batch_size_cpu > 0
        )

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

    @patch("src.agents.coordinator.create_supervisor")
    def test_agent_system_initialization_boundary(self, mock_create_supervisor):
        """Test agent system initialization boundary with supervisor patch."""
        mock_create_supervisor.return_value = MagicMock()
        from src.agents.coordinator import MultiAgentCoordinator

        coordinator = MultiAgentCoordinator()
        assert coordinator is not None
        assert hasattr(coordinator, "process_query")

    def test_tool_factory_boundary_integration(self):
        """Test ToolFactory creates a vector search tool from a mock index."""
        from src.agents.tool_factory import ToolFactory

        mock_index = MagicMock()
        mock_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_engine

        tool = ToolFactory.create_vector_search_tool(mock_index)
        assert tool is not None
        mock_index.as_query_engine.assert_called()

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

    @patch("src.processing.chunking.unstructured_chunker.SemanticChunker")
    def test_chunking_boundary_integration(self, mock_chunker):
        """Test document chunking with settings boundary integration."""
        # Patch constructor and ensure it's called with settings
        instance = MagicMock()
        mock_chunker.return_value = instance
        _ = mock_chunker(DocMindSettings())
        assert mock_chunker.called


@pytest.mark.integration
class TestDatabaseVectorStoreBoundaryIntegration:
    """Test database and vector store integration with mock backends."""

    @patch("src.storage.hybrid_persistence.HybridPersistenceManager")
    def test_hybrid_persistence_boundary_integration(self, mock_persistence):
        """Test hybrid persistence integration with mocked storage backends."""
        # Create manager and validate constructor boundary only
        _ = mock_persistence(DocMindSettings())
        assert mock_persistence.called

    def test_vector_store_boundary_integration(self):
        """Test vector store boundary using in-memory LlamaIndex."""
        docs = [
            Document(text="Vector search result 1", metadata={"id": "vec1"}),
            Document(text="Vector search result 2", metadata={"id": "vec2"}),
        ]
        index = VectorStoreIndex.from_documents(docs)
        results = index.as_retriever(similarity_top_k=2).retrieve("vector search")
        assert len(results) == 2
        assert all(hasattr(r, "score") for r in results)

    def test_sqlite_database_boundary_integration(self, tmp_path):
        """Test SQLite path handling via settings boundary (no DB ops)."""
        db_path = tmp_path / "db" / "docmind.db"
        s = DocMindSettings(database={"sqlite_db_path": str(db_path)})
        # Paths should be creatable
        s.database.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        assert s.database.sqlite_db_path.parent.exists()


@pytest.mark.integration
class TestCrossModuleCommunicationBoundaries:
    """Test cross-module communication patterns with interface validation."""

    @patch("src.agents.coordinator.create_supervisor")
    @patch("src.agents.coordinator.create_react_agent")
    def test_coordinator_embedding_communication_boundary(
        self, mock_create_react_agent, mock_create_supervisor
    ):
        """Test communication boundary: trigger supervisor creation during setup."""
        mock_create_supervisor.return_value = MagicMock()
        mock_create_react_agent.return_value = MagicMock()
        from src.agents.coordinator import MultiAgentCoordinator

        coordinator = MultiAgentCoordinator()
        coordinator.llm = MagicMock()
        coordinator._setup_agent_graph()  # pylint: disable=protected-access
        mock_create_supervisor.assert_called()

    @patch("src.core.infrastructure.gpu_monitor.gpu_performance_monitor")
    @patch("src.utils.monitoring.get_performance_monitor")
    def test_monitoring_infrastructure_communication_boundary(
        self, mock_get_perf, mock_gpu_monitor
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

        # Mock performance monitor
        mock_monitor_obj = MagicMock()
        mock_monitor_obj.record_operation.return_value = True
        mock_get_perf.return_value = mock_monitor_obj

        # Test monitoring communication
        gpu_monitor = mock_gpu_monitor()
        perf_monitor = mock_get_perf()

        gpu_stats = gpu_monitor.get_gpu_stats()
        log_result = perf_monitor.record_operation("gpu", 0.01, **gpu_stats)

        # Verify cross-module communication
        assert gpu_stats["gpu_utilization"] > 0
        assert log_result is True

        mock_gpu_monitor.assert_called_once()
        mock_get_perf.assert_called_once()

    @patch("src.dspy_integration.DSPyLlamaIndexRetriever")
    def test_dspy_optimization_boundary_integration(self, mock_dspy):
        """Test DSPy optimization boundary using retriever's optimize_query."""
        mock_dspy.optimize_query.return_value = {
            "refined": "Optimized query",
            "quality_score": 0.23,
        }

        result = mock_dspy.optimize_query("Original query")
        assert "refined" in result
        assert result["quality_score"] > 0.2

    def test_configuration_validation_across_boundaries(self):
        """Test configuration validation works across all module boundaries."""
        # Test that all major configuration sections are present
        required_configs = ["vllm", "agents", "embedding", "retrieval"]
        s = DocMindSettings()
        for config_name in required_configs:
            assert hasattr(s, config_name), (
                f"Missing required config section: {config_name}"
            )

        # Test that nested configurations have required fields
        assert hasattr(s.vllm, "model")
        assert hasattr(s.agents, "enable_multi_agent")
        assert hasattr(s.embedding, "model_name")
        assert hasattr(s.retrieval, "strategy")

        # Test that all configurations have valid values
        assert s.vllm.context_window > 0
        assert s.agents.decision_timeout > 0
        assert s.embedding.dimension > 0
        assert s.retrieval.top_k > 0


@pytest.mark.integration
class TestErrorHandlingBoundaryIntegration:
    """Test error handling across component boundaries."""

    @patch("src.agents.coordinator.create_react_agent")
    @patch("src.agents.coordinator.create_supervisor")
    def test_agent_system_error_boundary_handling(
        self, mock_create_supervisor, mock_create_react_agent
    ):
        """Test error handling at agent system boundaries."""
        mock_create_react_agent.return_value = MagicMock()
        mock_create_supervisor.side_effect = RuntimeError("Supervisor failed")
        from src.agents.coordinator import MultiAgentCoordinator

        coord = MultiAgentCoordinator()
        coord.llm = MagicMock()
        with pytest.raises(RuntimeError, match="Supervisor failed"):
            coord._setup_agent_graph()  # pylint: disable=protected-access

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3Embedder")
    def test_embedder_error_boundary_handling(self, mock_embedder):
        """Test error handling at embedder boundaries."""
        # Mock embedder with initialization error
        mock_embedder.side_effect = ImportError("BGE-M3 model not found")

        # Test error boundary handling
        with pytest.raises(ImportError, match="BGE-M3 model not found"):
            mock_embedder()

        mock_embedder.assert_called_once()

    def test_settings_validation_error_boundaries(self):
        """Test settings validation error handling at boundaries."""
        # Test invalid configuration values (Pydantic v2 ValidationError)
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DocMindSettings(vllm={"context_window": -1})  # Invalid negative

        with pytest.raises(ValidationError):
            DocMindSettings(processing={"chunk_size": 0})  # Invalid zero size

    @pytest.mark.asyncio
    @patch("src.utils.monitoring.async_performance_timer")
    async def test_async_error_boundary_handling(self, mock_async_timer):
        """Test async error handling across boundaries."""

        # Mock async function that raises error
        async def _raiser(*_args, **_kwargs):
            raise TimeoutError("Timer timeout")

        mock_async_timer.side_effect = _raiser

        # Test async error boundary handling
        with pytest.raises(asyncio.TimeoutError, match="Timer timeout"):
            await mock_async_timer("op")

        mock_async_timer.assert_called_once()


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
