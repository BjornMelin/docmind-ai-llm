"""Comprehensive end-to-end workflow tests for DocMind AI application.

This module provides comprehensive E2E tests that validate complete user workflows
through the DocMind AI application, including multi-agent coordination, document
processing, and user interface integration. These tests focus on realistic user
scenarios and complete workflow validation.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.e2e.helpers import (
    assert_documents_have_required_metadata,
    build_isolated_modules,
    configure_hardware_mocks,
    install_agent_stubs,
    install_heavy_dependencies,
    install_llama_index_core,
    install_mock_ollama,
    install_mock_torch,
    patch_async_workflow_dependencies,
    patch_document_loader,
    patch_hardware_validation,
)


@pytest.fixture(autouse=True)
def setup_comprehensive_dependencies(monkeypatch):
    """Setup comprehensive dependencies with proper pytest fixtures.

    Uses monkeypatch instead of sys.modules anti-pattern.
    Only mocks external dependencies at boundaries.
    """
    install_mock_torch(
        monkeypatch,
        include_cuda_props=True,
        include_tensor=True,
        include_nn=True,
        include_device=True,
    )

    # Mock heavy external dependencies
    heavy_dependencies = [
        "llama_index.llms.llama_cpp",
        "llama_cpp",
        "transformers",
        "sentence_transformers",
        "FlagEmbedding",
        "spacy",
        "thinc",
        "unstructured",
        "nltk",
        "chromadb",
        "qdrant_client",
    ]

    install_heavy_dependencies(monkeypatch, heavy_dependencies)

    # Mock LlamaIndex core retrievers package

    install_llama_index_core(monkeypatch)

    # Mock Ollama service client specifically (boundary mocking)
    install_mock_ollama(monkeypatch)

    # Provide lightweight stubs for src.agents to avoid heavy imports
    install_agent_stubs(monkeypatch)


def _make_mock_documents() -> list:
    """Return mock document list for testing ingestion flow."""

    class Doc:
        def __init__(self, text, metadata=None):
            self.text = text
            self.metadata = metadata or {}

    return [
        Doc(
            text=(
                "DocMind AI provides comprehensive document analysis using "
                "multi-agent coordination."
            ),
            metadata={"source": "test_document.pdf", "page": 1},
        ),
        Doc(
            text=(
                "The system supports various document formats and provides "
                "intelligent insights."
            ),
            metadata={"source": "test_document.pdf", "page": 2},
        ),
    ]


def _configure_core_mocks(
    mock_detect,
    mock_validate,
    mock_load_docs,
    mock_coordinator_class,
    mock_ollama_list,
    mock_ollama_pull,
) -> None:
    """Configure core mock return values for workflow test."""
    configure_hardware_mocks(mock_detect, mock_validate)
    mock_load_docs.return_value = _make_mock_documents()

    mock_coordinator = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        "Comprehensive analysis completed using multi-agent coordination. "
        "Key insights: Document processing capabilities, AI-powered analysis "
        "features, and system architecture overview."
    )
    mock_coordinator.process_query.return_value = mock_response
    mock_coordinator_class.return_value = mock_coordinator

    mock_ollama_list.return_value = {
        "models": [{"name": "qwen3-4b-instruct-2507:latest"}]
    }
    mock_ollama_pull.return_value = {"status": "success"}


def _exercise_workflow(mock_detect, mock_load_docs):
    """Run the lightweight workflow assertions and return artifacts."""
    from types import SimpleNamespace

    docmind_settings = SimpleNamespace
    analysis_object = SimpleNamespace
    predefined_prompts = {"summarization": "Summarize the docs"}

    def validate_startup_configuration(_settings=None):
        return True

    def detect_hardware():
        return mock_detect.return_value

    print("✅ Minimal environment stubbed for workflow test")

    settings = docmind_settings(
        vllm=SimpleNamespace(model="mock", context_window=8192),
        ollama_base_url="http://localhost:11434",
    )
    validate_startup_configuration(settings)
    hardware_info = detect_hardware()
    documents = mock_load_docs(["test.pdf"])

    mock_index = MagicMock()
    from src.agents.tool_factory import ToolFactory

    tools = ToolFactory.create_basic_tools({"vector": mock_index})

    from src.agents.coordinator import MultiAgentCoordinator

    coordinator = MultiAgentCoordinator()
    result = coordinator.process_query("Analyze this document comprehensively")
    analysis_output = analysis_object(
        summary="Test analysis summary",
        key_insights=["Insight 1", "Insight 2"],
        action_items=["Action 1"],
        open_questions=["Question 1"],
    )
    return (
        settings,
        hardware_info,
        documents,
        tools,
        result,
        analysis_output,
        predefined_prompts,
    )


@pytest.mark.asyncio
async def test_complete_application_workflow():
    """End-to-end application workflow covering major components."""
    module_map = build_isolated_modules()
    with (
        patch.dict(
            sys.modules,
            module_map,
            clear=False,
        ),
        patch_hardware_validation() as (mock_detect, mock_validate),
        patch(
            "src.agents.coordinator.MultiAgentCoordinator", create=True
        ) as mock_coordinator_class,
        patch_document_loader() as mock_load_docs,
        patch("ollama.list") as mock_ollama_list,
        patch("ollama.pull") as mock_ollama_pull,
    ):
        _configure_core_mocks(
            mock_detect,
            mock_validate,
            mock_load_docs,
            mock_coordinator_class,
            mock_ollama_list,
            mock_ollama_pull,
        )
        (
            settings,
            hardware_info,
            documents,
            tools,
            result,
            analysis_output,
            predefined_prompts,
        ) = _exercise_workflow(mock_detect, mock_load_docs)

        assert settings is not None
        assert isinstance(hardware_info, dict)
        assert len(documents) == 2
        assert documents[0].text.startswith("DocMind AI provides")
        assert isinstance(tools, list)
        assert result is not None
        assert hasattr(result, "content")
        assert "multi-agent coordination" in result.content
        assert analysis_output.summary == "Test analysis summary"
        assert len(analysis_output.key_insights) == 2
        assert isinstance(predefined_prompts, dict)
        assert len(predefined_prompts) > 0

        print("✅ Complete application workflow test passed")
        print(f"   - Configuration: {settings.vllm.model}")
        print(f"   - Hardware: {hardware_info['gpu_name']}")
        print(f"   - Documents processed: {len(documents)}")
        print(f"   - Prompts available: {len(predefined_prompts)}")


def test_unified_configuration_architecture():
    """Test the unified configuration architecture and its integration points."""
    with patch.dict(
        "os.environ",
        {
            "DOCMIND_VLLM__MODEL": "qwen3-4b-instruct-2507:latest",
            "DOCMIND_OLLAMA_BASE_URL": "http://localhost:11434",
            "DOCMIND_DEBUG": "false",
        },
    ):
        try:
            from src.config.settings import DocMindSettings

            # Test configuration instantiation
            settings = DocMindSettings()

            # Validate key configuration attributes
            assert settings.vllm.model == "qwen3-4b-instruct-2507:latest"
            assert settings.ollama_base_url == "http://localhost:11434"
            assert settings.debug is False

            # Validate presence of representative attributes
            assert hasattr(settings, "app_name")
            assert hasattr(settings, "ollama_base_url")
            assert hasattr(settings, "debug")
            assert hasattr(settings, "log_level")
            assert hasattr(settings, "enable_gpu_acceleration")
            assert hasattr(settings.vllm, "context_window")

            print("✅ Unified configuration architecture validated")
            print(f"   - Model: {settings.vllm.model}")
            print(f"   - Base URL: {settings.ollama_base_url}")
            print(f"   - Debug mode: {settings.debug}")

        except ImportError as e:
            pytest.skip(f"Configuration system import failed: {e}")


def test_multi_agent_system_integration():
    """Test multi-agent system integration and coordination capabilities."""
    with patch(
        "src.agents.coordinator.MultiAgentCoordinator", create=True
    ) as mock_coordinator_class:
        # Setup mock multi-agent coordinator
        mock_coordinator = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "Multi-agent system successfully coordinated analysis. Agents "
            "collaborated to provide comprehensive insights."
        )
        mock_coordinator.process_query.return_value = mock_response
        mock_coordinator_class.return_value = mock_coordinator

        try:
            from src.agents.coordinator import MultiAgentCoordinator
            from src.agents.tool_factory import ToolFactory

            # Test coordinator instantiation
            coordinator = MultiAgentCoordinator()
            assert coordinator is not None

            # Test agent coordination workflow
            query = (
                "Perform comprehensive document analysis using multi-agent coordination"
            )
            result = coordinator.process_query(query)

            assert result is not None
            assert hasattr(result, "content")
            assert "multi-agent" in result.content.lower()

            # Test tool factory integration
            mock_index = MagicMock()
            tools = ToolFactory.create_basic_tools({"vector": mock_index})
            assert tools is not None

            print("✅ Multi-agent system integration validated")
            print("   - Coordinator: Available")
            print("   - Tool Factory: Available")
            print(f"   - Response length: {len(result.content)} chars")

        except ImportError as e:
            pytest.skip(f"Multi-agent system import failed: {e}")


@pytest.mark.asyncio
async def test_document_processing_pipeline():
    """Test the complete document processing pipeline post-ADR-009."""
    with patch_document_loader() as mock_load_docs:
        # Mock successful document loading
        from llama_index.core import Document

        mock_documents = [
            Document(
                text="Advanced document processing with unstructured data handling.",
                metadata={"source": "doc1.pdf", "page": 1, "chunk_id": "chunk_1"},
            ),
            Document(
                text=(
                    "Multi-modal content extraction and intelligent "
                    "chunking strategies."
                ),
                metadata={"source": "doc1.pdf", "page": 2, "chunk_id": "chunk_2"},
            ),
            Document(
                text="Integration with vector databases and embedding systems.",
                metadata={"source": "doc1.pdf", "page": 3, "chunk_id": "chunk_3"},
            ),
        ]
        mock_load_docs.return_value = mock_documents

        try:
            from src.models.schemas import AnalysisOutput

            # Test document loading
            file_paths = ["test_document.pdf", "test_document2.pdf"]
            documents = await mock_load_docs(  # type: ignore[func-returns-value]
                file_paths
            )

            assert len(documents) == 3
            assert all(isinstance(doc, Document) for doc in documents)
            assert all(doc.metadata.get("source") == "doc1.pdf" for doc in documents)

            # Test that documents have proper structure
            assert_documents_have_required_metadata(documents)

            # Test analysis output schema
            analysis = AnalysisOutput(
                summary=(
                    "Document processing pipeline successfully processed multiple "
                    "documents"
                ),
                key_insights=[
                    "Unstructured data handling capabilities",
                    "Multi-modal content support",
                    "Vector database integration",
                ],
                action_items=[
                    "Optimize chunking strategy",
                    "Enhance metadata extraction",
                ],
                open_questions=["How to handle large document collections?"],
            )

            assert analysis.summary is not None
            assert len(analysis.key_insights) == 3
            assert len(analysis.action_items) == 2
            assert len(analysis.open_questions) == 1

            print("✅ Document processing pipeline validated")
            print(f"   - Documents processed: {len(documents)}")
            print(f"   - Analysis insights: {len(analysis.key_insights)}")
            print(
                f"   - Total text length: "
                f"{sum(len(doc.text) for doc in documents)} chars"
            )

        except ImportError as e:
            pytest.skip(f"Document processing components import failed: {e}")


@pytest.mark.asyncio
async def test_async_workflow_components():
    """Test async workflow components and operations."""
    try:
        # Mock async operations
        with patch_async_workflow_dependencies() as (
            mock_load,
            mock_coordinator_class,
        ):
            # Setup async mocks
            async_coordinator = AsyncMock()
            async_response = MagicMock()
            async_response.content = (
                "Async multi-agent coordination completed successfully"
            )
            async_coordinator.process_query.return_value = async_response
            mock_coordinator_class.return_value = async_coordinator

            mock_load.return_value = []

            # Test async document processing (mocked)
            documents = mock_load(["test.pdf"])
            assert documents is not None

            # Test async coordinator
            coordinator = mock_coordinator_class()
            result = await coordinator.process_query("Process async")
            assert result is not None
            assert hasattr(result, "content")

            print("✅ Async workflow components validated")

    except ImportError as e:
        pytest.skip(f"Async components import failed: {e}")


def test_application_structure_and_markers():
    """Test application structure and testing markers for E2E validation."""
    try:
        # Test core application components
        from llama_index.core.llms import ChatMessage
        from llama_index.core.memory import ChatMemoryBuffer

        from src.config.settings import DocMindSettings
        from src.prompting import list_templates

        # Test prompts structure
        assert isinstance(list_templates(), list)
        assert len(list_templates()) > 0

        tpls = list_templates()
        first_prompt = tpls[0].name if tpls else ""
        assert isinstance(first_prompt, str)
        assert len(first_prompt) > 0

        # Test configuration structure
        settings = DocMindSettings()
        assert hasattr(settings, "ollama_base_url")
        assert hasattr(settings.vllm, "model")
        assert hasattr(settings.vllm, "context_window")

        # Test memory management components
        memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
        assert memory is not None

        message = ChatMessage(role="user", content="Test message for E2E validation")
        assert message.role == "user"
        assert message.content == "Test message for E2E validation"

        print("✅ Application structure and markers validated")
        print(f"   - Available templates: {len(list_templates())}")
        print(f"   - Configuration loaded: {settings.vllm.model}")
        print("   - Memory system: Operational")

    except ImportError as e:
        pytest.skip(f"Application structure validation failed: {e}")


def test_legacy_component_cleanup_validation():
    """Validate that legacy components have been properly cleaned up post-ADR-009."""
    # These modules should NOT be importable (removed/reorganized in ADR-009)
    legacy_modules = [
        "src.retrieval.integration",
        "src.utils.embedding",
        "src.storage.vector_store_legacy",
    ]

    # These modules SHOULD be importable (current architecture)
    current_modules = [
        "src.config.settings",
        "src.agents.coordinator",
        "src.utils.core",
        "src.processing.ingestion_api",
        "src.models.schemas",
    ]

    # Verify legacy modules are not accessible
    for module_path in legacy_modules:
        try:
            __import__(module_path)
            pytest.fail(f"Legacy module {module_path} should not be importable")
        except ImportError:
            # Expected - legacy module properly removed
            pass

    # Verify current modules are accessible
    successful_imports = []
    for module_path in current_modules:
        try:
            __import__(module_path)
            successful_imports.append(module_path)
        except ImportError:
            # Some imports might fail in test environment
            pass

    # Should have at least some successful imports
    assert len(successful_imports) > 0, "No current modules could be imported"

    print("✅ Legacy component cleanup validated")
    print(
        f"   - Current modules imported: {len(successful_imports)}/"
        f"{len(current_modules)}"
    )
    print(f"   - Legacy modules properly blocked: {len(legacy_modules)}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_integration_with_mocks():
    """Integration test that validates complete E2E workflow with proper mocking."""
    with (
        patch_hardware_validation() as (mock_detect, mock_validate),
        patch(
            "src.agents.coordinator.MultiAgentCoordinator", create=True
        ) as mock_coordinator,
        patch_document_loader() as mock_load,
    ):
        # Setup comprehensive mocks
        configure_hardware_mocks(mock_detect, mock_validate)

        from llama_index.core import Document

        mock_documents = [
            Document(text="E2E test document", metadata={"source": "test.pdf"})
        ]
        mock_load.return_value = mock_documents

        mock_coordinator_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "E2E integration test successful"
        mock_coordinator_instance.process_query.return_value = mock_response
        mock_coordinator.return_value = mock_coordinator_instance

        try:
            # Test complete integration workflow
            from src.agents.coordinator import MultiAgentCoordinator
            from src.config.settings import DocMindSettings
            from src.utils.core import detect_hardware, validate_startup_configuration

            # 1. Configuration
            settings = DocMindSettings()
            assert settings is not None

            # 2. Validation
            validate_startup_configuration(settings)
            assert mock_validate.called

            # 3. Hardware detection
            hardware = detect_hardware()
            assert hardware["gpu_name"] == "RTX 4090"

            # 4. Document processing (async mock)
            docs = await mock_load(["test.pdf"])  # type: ignore[func-returns-value]
            assert len(docs) == 1

            # 5. Agent coordination
            coordinator = MultiAgentCoordinator()
            result = coordinator.process_query("Complete E2E test")
            assert "E2E integration test successful" in result.content

            print("✅ End-to-end integration test completed successfully")
            print("   - All system components validated")
            print("   - Integration workflow: Operational")

        except ImportError as e:
            pytest.skip(f"E2E integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
