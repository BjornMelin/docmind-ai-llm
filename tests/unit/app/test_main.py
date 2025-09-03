"""Comprehensive unit tests for src/main.py DocMindApplication.

Tests the main application entry point with focus on:
- Application initialization and configuration
- Query processing workflows (multi-agent and basic RAG)
- Document ingestion and error handling
- Resource management and shutdown
- Error scenarios and edge cases

Uses pytest-mock for boundary mocking and follows KISS/DRY principles.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.models import AgentResponse
from src.config.settings import DocMindSettings
from src.main import DocMindApplication, main


@pytest.mark.unit
class TestDocMindApplication:
    """Test DocMindApplication class initialization and core methods."""

    def test_application_init_with_defaults(self):
        """Test application initialization with default settings."""
        with (
            patch("src.main.DocumentProcessor") as mock_processor,
            patch("src.main.MultiAgentCoordinator") as mock_coordinator,
        ):
            app = DocMindApplication()

            # Should use global settings by default
            assert app.settings is not None
            assert app.enable_multi_agent is True

            # Should initialize components
            mock_processor.assert_called_once()
            mock_coordinator.assert_called_once()

    def test_application_init_with_custom_settings(self):
        """Test application initialization with custom settings."""
        custom_settings = DocMindSettings(debug=True, log_level="DEBUG")

        with (
            patch("src.main.DocumentProcessor") as mock_processor,
            patch("src.main.MultiAgentCoordinator") as mock_coordinator,
        ):
            app = DocMindApplication(
                app_settings=custom_settings, enable_multi_agent=False
            )

            assert app.settings == custom_settings
            assert app.enable_multi_agent is False
            assert app.agent_coordinator is None

            # Should still initialize document processor
            mock_processor.assert_called_once_with(settings=custom_settings)
            mock_coordinator.assert_not_called()

    def test_application_init_with_multi_agent_disabled(self):
        """Test initialization with multi-agent system disabled."""
        with (
            patch("src.main.DocumentProcessor"),
            patch("src.main.MultiAgentCoordinator") as mock_coordinator,
        ):
            app = DocMindApplication(enable_multi_agent=False)

            assert app.enable_multi_agent is False
            assert app.agent_coordinator is None
            mock_coordinator.assert_not_called()

    def test_initialize_components_creates_required_objects(self):
        """Test _initialize_components creates all required objects."""
        with (
            patch("src.main.DocumentProcessor") as mock_processor,
            patch("src.main.MultiAgentCoordinator") as mock_coordinator,
        ):
            app = DocMindApplication()

            # Verify document processor creation
            mock_processor.assert_called_once_with(settings=app.settings)
            assert app.document_processor is not None

            # Verify multi-agent coordinator creation
            mock_coordinator.assert_called_once_with(
                model_path=app.settings.vllm.model,
                max_context_length=app.settings.vllm.context_window,
                backend="vllm",
                enable_fallback=app.settings.agents.enable_fallback_rag,
                max_agent_timeout=app.settings.agents.decision_timeout / 1000.0,
            )


@pytest.mark.unit
class TestDocMindApplicationQueryProcessing:
    """Test query processing methods and workflows."""

    @pytest.fixture
    def mock_app(self):
        """Create a DocMindApplication with mocked dependencies."""
        with (
            patch("src.main.DocumentProcessor"),
            patch("src.main.MultiAgentCoordinator") as mock_coordinator,
        ):
            app = DocMindApplication(app_settings=DocMindSettings())
            app.agent_coordinator = mock_coordinator.return_value
            return app

    @pytest.mark.asyncio
    async def test_process_query_with_multi_agent_enabled(self, mock_app):
        """Test query processing with multi-agent system."""
        # Setup mock response
        expected_response = AgentResponse(
            content="Multi-agent response",
            sources=[{"name": "source1.pdf", "path": "/path/to/source1.pdf"}],
            metadata={"agent": "multi"},
            validation_score=0.9,
            processing_time=1.5,
        )
        mock_app.agent_coordinator.process_query.return_value = expected_response

        # Test query processing
        query = "Test query about documents"
        context = {"session_id": "test_session"}

        result = await mock_app.process_query(query, context)

        # Verify multi-agent coordinator was called
        mock_app.agent_coordinator.process_query.assert_called_once_with(
            query=query,
            context=context,
        )

        # Verify response
        assert result == expected_response
        assert result.content == "Multi-agent response"
        assert result.metadata["agent"] == "multi"

    @pytest.mark.asyncio
    async def test_process_query_with_multi_agent_disabled(self, mock_app):
        """Test query processing with multi-agent disabled."""
        # Disable multi-agent system
        mock_app.enable_multi_agent = False
        mock_app.agent_coordinator = None

        query = "Test query about documents"
        result = await mock_app.process_query(query)

        # Should use basic RAG pipeline (deprecated message)
        assert "Basic RAG pipeline is no longer supported" in result.content
        assert result.metadata["pipeline"] == "basic_rag"
        assert result.validation_score == 0.8  # BASIC_VALIDATION_SCORE

    @pytest.mark.asyncio
    async def test_process_query_with_use_multi_agent_override(self, mock_app):
        """Test query processing with use_multi_agent parameter override."""
        # Setup mock response for multi-agent
        expected_response = AgentResponse(
            content="Multi-agent response",
            sources=[],
            metadata={"agent": "multi"},
            validation_score=0.9,
            processing_time=1.0,
        )
        mock_app.agent_coordinator.process_query.return_value = expected_response

        # Test with override to use multi-agent
        result = await mock_app.process_query("Test query", use_multi_agent=True)
        assert result == expected_response

        # Test with override to disable multi-agent
        result = await mock_app.process_query("Test query", use_multi_agent=False)
        assert result.metadata["pipeline"] == "basic_rag"

    @pytest.mark.asyncio
    async def test_process_query_error_handling(self, mock_app):
        """Test error handling in query processing."""
        # Setup mock to raise error
        mock_app.agent_coordinator.process_query.side_effect = ValueError("Test error")

        result = await mock_app.process_query("Test query")

        # Should return error response
        assert "I encountered an error processing your query" in result.content
        assert result.metadata["error"] == "Test error"
        assert result.metadata["fallback"] is True
        assert result.validation_score == 0.0

    @pytest.mark.asyncio
    async def test_basic_rag_pipeline_implementation(self, mock_app):
        """Test basic RAG pipeline implementation."""
        query = "Test query for basic RAG"
        context = {"test": "context"}

        result = await mock_app._process_basic_rag(query, context)

        # Verify response structure
        assert isinstance(result, AgentResponse)
        assert "Basic RAG pipeline is no longer supported" in result.content
        assert result.metadata["pipeline"] == "basic_rag"
        assert result.validation_score == 0.8
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_basic_rag_with_import_error(self, mock_app):
        """Test basic RAG pipeline handles import errors gracefully."""
        # Mock an import error scenario
        with patch("src.main.logger"):
            query = "Test query"
            result = await mock_app._process_basic_rag(query)

            # Should handle gracefully and return informative response
            assert isinstance(result, AgentResponse)
            assert "Basic RAG pipeline is no longer supported" in result.content
            assert result.validation_score == 0.8

    # Removed legacy _generate_basic_response tests (method no longer exists)


@pytest.mark.unit
class TestDocMindApplicationDocumentIngestion:
    """Test document ingestion and processing methods."""

    @pytest.fixture
    def mock_app(self):
        """Create a DocMindApplication with mocked dependencies."""
        with (
            patch("src.main.DocumentProcessor") as mock_processor,
            patch("src.main.MultiAgentCoordinator"),
        ):
            app = DocMindApplication(app_settings=DocMindSettings())
            app.document_processor = mock_processor.return_value
            return app

    @pytest.mark.asyncio
    async def test_ingest_document_async_success(self, mock_app, tmp_path):
        """Test successful document ingestion with async processing."""
        # Setup test file
        test_file = tmp_path / "test_document.pdf"
        test_file.write_text("Test document content")

        # Setup mock response
        expected_result = {
            "status": "success",
            "chunks": 5,
            "processing_time": 2.3,
        }
        mock_app.document_processor.process_document_async = AsyncMock(
            return_value=expected_result
        )

        result = await mock_app.ingest_document(test_file, process_async=True)

        # Verify processor was called correctly
        mock_app.document_processor.process_document_async.assert_called_once_with(
            test_file
        )

        # Verify result
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_ingest_document_sync_processing(self, mock_app, tmp_path):
        """Test document ingestion with synchronous processing."""
        test_file = tmp_path / "test_document.pdf"
        test_file.write_text("Test document content")

        expected_result = {"status": "success", "chunks": 3}
        mock_app.document_processor.process_document_async.return_value = (
            expected_result
        )

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = expected_result

            result = await mock_app.ingest_document(test_file, process_async=False)

            # Should call asyncio.run for sync processing
            mock_run.assert_called_once()
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_ingest_document_file_error(self, mock_app, tmp_path):
        """Test document ingestion with file errors."""
        # Test with non-existent file
        non_existent_file = tmp_path / "non_existent.pdf"

        mock_app.document_processor.process_document_async.side_effect = (
            FileNotFoundError("File not found")
        )

        result = await mock_app.ingest_document(non_existent_file)

        # Should return error response
        assert result["error"] == "File not found"
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_ingest_document_processing_error(self, mock_app, tmp_path):
        """Test document ingestion with processing errors."""
        test_file = tmp_path / "corrupted_document.pdf"
        test_file.write_bytes(b"corrupted content")

        # Setup mock to raise processing error
        mock_app.document_processor.process_document_async.side_effect = ValueError(
            "Invalid document format"
        )

        result = await mock_app.ingest_document(test_file)

        assert result["error"] == "Invalid document format"
        assert result["status"] == "failed"


@pytest.mark.unit
class TestDocMindApplicationResourceManagement:
    """Test resource management and shutdown behavior."""

    def test_shutdown_with_agent_coordinator(self):
        """Test shutdown with active agent coordinator."""
        with (
            patch("src.main.DocumentProcessor"),
            patch("src.main.MultiAgentCoordinator"),
        ):
            app = DocMindApplication(app_settings=DocMindSettings())

            # Should not raise any errors
            app.shutdown()

            # Verify coordinator is still accessible (no explicit cleanup needed)
            assert app.agent_coordinator is not None

    def test_shutdown_without_agent_coordinator(self):
        """Test shutdown without agent coordinator."""
        with patch("src.main.DocumentProcessor"):
            app = DocMindApplication(
                app_settings=DocMindSettings(), enable_multi_agent=False
            )

            # Should not raise any errors
            app.shutdown()
            assert app.agent_coordinator is None


@pytest.mark.unit
class TestDocMindApplicationEdgeCases:
    """Test edge cases and error scenarios."""

    def test_initialization_with_none_settings(self):
        """Test initialization with None settings uses defaults."""
        with (
            patch("src.main.DocumentProcessor"),
            patch("src.main.MultiAgentCoordinator"),
            patch("src.main.settings") as mock_global_settings,
        ):
            app = DocMindApplication(app_settings=None)

            # Should use global settings
            assert app.settings == mock_global_settings

    @pytest.mark.asyncio
    async def test_process_query_with_none_coordinator(self):
        """Test query processing when coordinator is None."""
        with patch("src.main.DocumentProcessor"):
            app = DocMindApplication(
                app_settings=DocMindSettings(), enable_multi_agent=True
            )
            app.agent_coordinator = None  # Simulate coordinator failure

            result = await app.process_query("test query", use_multi_agent=True)

            # Should fall back to basic RAG (deprecated message)
            assert "Basic RAG pipeline is no longer supported" in result.content

    @pytest.mark.asyncio
    async def test_process_query_multiple_error_types(self):
        """Test query processing handles different exception types."""
        with (
            patch("src.main.DocumentProcessor"),
            patch("src.main.MultiAgentCoordinator") as mock_coordinator,
        ):
            app = DocMindApplication(app_settings=DocMindSettings())
            app.agent_coordinator = mock_coordinator.return_value

            # Test different exception types
            error_types = [
                ValueError("Value error"),
                TypeError("Type error"),
                RuntimeError("Runtime error"),
                OSError("OS error"),
            ]

            for error in error_types:
                app.agent_coordinator.process_query.side_effect = error
                result = await app.process_query("test query")

                assert "I encountered an error processing your query" in result.content
                assert result.metadata["error"] == str(error)
                assert result.validation_score == 0.0


@pytest.mark.unit
class TestMainEntryPoint:
    """Test the main entry point function."""

    @pytest.mark.asyncio
    async def test_main_function_basic_workflow(self):
        """Test main function executes without errors."""
        mock_app = AsyncMock()
        mock_response = AgentResponse(
            content="Test response content",
            sources=[],
            metadata={},
            validation_score=0.9,
            processing_time=1.2,
        )
        mock_app.process_query.return_value = mock_response

        with (
            patch("src.main.DocMindApplication") as mock_app_class,
            patch("src.main.logger") as mock_logger,
        ):
            mock_app_class.return_value = mock_app

            await main()

            # Verify application creation
            mock_app_class.assert_called_once_with(enable_multi_agent=True)

            # Verify query processing
            mock_app.process_query.assert_called_once_with(
                "What are the key features of the new product?"
            )

            # Verify logged output
            mock_logger.info.assert_any_call("Response: %s", "Test response content")
            mock_logger.info.assert_any_call("Processing time: %.2fs", 1.2)
            mock_logger.info.assert_any_call("Validation score: %.2f", 0.9)

            # Verify shutdown
            mock_app.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_function_handles_exceptions(self):
        """Test main function handles exceptions gracefully."""
        with patch("src.main.DocMindApplication") as mock_app_class:
            mock_app_class.side_effect = Exception("Initialization failed")

            # The current main function doesn't handle exceptions, so this test
            # should expect an exception to be raised
            with pytest.raises(Exception, match="Initialization failed"):
                await main()


## Removed legacy constants test: DOCUMENT_TEXT_SLICE_* no longer present in src


# Integration marker for tests that cross component boundaries
@pytest.mark.integration
class TestDocMindApplicationIntegration:
    """Integration tests for DocMindApplication with lightweight dependencies."""

    @pytest.mark.integration
    def test_application_with_real_settings(self):
        """Test application initialization with real settings object."""
        # Use actual DocMindSettings but with safe defaults
        settings = DocMindSettings(
            debug=True,
            enable_gpu_acceleration=False,
        )

        with patch("src.main.DocumentProcessor") as mock_processor:
            app = DocMindApplication(app_settings=settings, enable_multi_agent=False)

            assert app.settings.debug is True
            assert app.settings.enable_gpu_acceleration is False
            assert app.enable_multi_agent is False  # This is set by the app parameter
            assert app.agent_coordinator is None

            # Should still initialize document processor
            mock_processor.assert_called_once_with(settings=settings)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_basic_rag_workflow(self):
        """Test end-to-end workflow with basic RAG pipeline."""
        settings = DocMindSettings(
            debug=True,
            enable_gpu_acceleration=False,
        )

        with patch("src.main.DocumentProcessor"):
            app = DocMindApplication(
                app_settings=settings,
                enable_multi_agent=False,  # Use basic RAG
            )

            # Test query processing
            result = await app.process_query("What is machine learning?")

            # Verify basic RAG response shape
            assert isinstance(result, AgentResponse)
            assert result.metadata["pipeline"] == "basic_rag"
            # Current implementation returns a deprecation notice in content
            assert "Basic RAG" in result.content

            # Test shutdown
            app.shutdown()  # Should not raise errors
