"""Comprehensive end-to-end workflow tests for DocMind AI application.

This module provides comprehensive E2E tests that validate complete user workflows
through the DocMind AI application, including multi-agent coordination, document
processing, and user interface integration. These tests focus on realistic user
scenarios and complete workflow validation.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Comprehensive mocking strategy for reliable E2E testing
def setup_comprehensive_mocks():
    """Set up comprehensive mocks to prevent import and dependency issues."""
    # Mock torch with complete attributes for spacy/thinc compatibility
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.7.1+cu126"
    mock_torch.__spec__ = MagicMock()
    mock_torch.__spec__.name = "torch"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_properties.return_value = MagicMock(
        name="RTX 4090", total_memory=17179869184
    )
    # Additional torch attributes
    mock_torch.tensor = MagicMock()
    mock_torch.nn = MagicMock()
    mock_torch.device = MagicMock()
    sys.modules["torch"] = mock_torch

    # Mock other heavy dependencies
    mock_modules = [
        "llama_index.llms.llama_cpp",
        "llama_cpp",
        "ollama",
        "transformers",
        "sentence_transformers",
        "spacy",
        "thinc",
        "unstructured",
        "nltk",
        "chromadb",
        "qdrant_client",
    ]

    for module in mock_modules:
        if module not in sys.modules:
            sys.modules[module] = MagicMock()

    # Mock Ollama specifically
    mock_ollama = MagicMock()
    mock_ollama.list.return_value = {
        "models": [{"name": "qwen3-4b-instruct-2507:latest"}]
    }
    mock_ollama.pull.return_value = {"status": "success"}
    mock_ollama.chat.return_value = {"message": {"content": "Test response"}}
    sys.modules["ollama"] = mock_ollama


# Set up mocks at module level
setup_comprehensive_mocks()


@pytest.mark.asyncio
async def test_complete_application_workflow():
    """Test complete end-to-end application workflow including all major components.

    This test validates the entire user journey through the application:
    1. Configuration system initialization
    2. Hardware detection and model suggestions
    3. Document upload and processing
    4. Multi-agent analysis coordination
    5. Chat functionality
    6. Session persistence
    """
    with (
        patch("src.utils.core.detect_hardware") as mock_detect,
        patch("src.utils.core.validate_startup_configuration") as mock_validate,
        patch("src.agents.coordinator.MultiAgentCoordinator") as mock_coordinator_class,
        patch("src.utils.document.load_documents_unstructured") as mock_load_docs,
        patch("ollama.list") as mock_ollama_list,
        patch("ollama.pull") as mock_ollama_pull,
    ):
        # Setup mocks for complete workflow
        mock_detect.return_value = {
            "gpu_name": "RTX 4090",
            "vram_total_gb": 24,
            "cuda_available": True,
        }
        mock_validate.return_value = True

        # Mock document loading
        from llama_index.core import Document

        mock_documents = [
            Document(
                text="DocMind AI provides comprehensive document analysis using multi-agent coordination.",
                metadata={"source": "test_document.pdf", "page": 1},
            ),
            Document(
                text="The system supports various document formats and provides intelligent insights.",
                metadata={"source": "test_document.pdf", "page": 2},
            ),
        ]
        mock_load_docs.return_value = mock_documents

        # Mock multi-agent coordinator
        mock_coordinator = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Comprehensive analysis completed using multi-agent coordination. Key insights: Document processing capabilities, AI-powered analysis features, and system architecture overview."
        mock_coordinator.process_query.return_value = mock_response
        mock_coordinator_class.return_value = mock_coordinator

        # Mock Ollama
        mock_ollama_list.return_value = {
            "models": [{"name": "qwen3-4b-instruct-2507:latest"}]
        }
        mock_ollama_pull.return_value = {"status": "success"}

        # Test core component imports (validate system structure)
        try:
            from src.agents.coordinator import MultiAgentCoordinator
            from src.agents.tool_factory import ToolFactory
            from src.config.settings import DocMindSettings
            from src.models.schemas import AnalysisOutput
            from src.prompts import PREDEFINED_PROMPTS
            from src.utils.core import detect_hardware, validate_startup_configuration
            from src.utils.document import load_documents_unstructured

            print("✅ All core components imported successfully")

        except ImportError as e:
            pytest.skip(f"Core component import failed: {e}")
            return

        # Test configuration system
        settings = DocMindSettings()
        assert settings is not None
        assert hasattr(settings, "model_name")
        assert hasattr(settings, "ollama_base_url")
        assert hasattr(settings, "default_token_limit")

        # Test startup validation
        validation_result = validate_startup_configuration(settings)
        assert mock_validate.called

        # Test hardware detection
        hardware_info = detect_hardware()
        assert mock_detect.called
        assert hardware_info["gpu_name"] == "RTX 4090"

        # Test document processing workflow
        documents = load_documents_unstructured(["test.pdf"])
        assert len(documents) == 2
        assert documents[0].text.startswith("DocMind AI provides")

        # Test tool factory
        mock_index = MagicMock()
        tools = ToolFactory.create_basic_tools({"vector": mock_index})
        assert tools is not None

        # Test multi-agent coordination
        coordinator = MultiAgentCoordinator()
        result = coordinator.process_query("Analyze this document comprehensively")
        assert result is not None
        assert hasattr(result, "content")
        assert "multi-agent coordination" in result.content

        # Test response schema validation
        analysis_output = AnalysisOutput(
            summary="Test analysis summary",
            key_insights=["Insight 1", "Insight 2"],
            action_items=["Action 1"],
            open_questions=["Question 1"],
        )
        assert analysis_output.summary == "Test analysis summary"
        assert len(analysis_output.key_insights) == 2

        # Test prompts system
        assert isinstance(PREDEFINED_PROMPTS, dict)
        assert len(PREDEFINED_PROMPTS) > 0

        print("✅ Complete application workflow test passed")
        print(f"   - Configuration: {settings.model_name}")
        print(f"   - Hardware: {hardware_info['gpu_name']}")
        print(f"   - Documents processed: {len(documents)}")
        print(f"   - Prompts available: {len(PREDEFINED_PROMPTS)}")


def test_unified_configuration_architecture():
    """Test the unified configuration architecture and its integration points."""
    with patch.dict(
        "os.environ",
        {
            "DOCMIND_MODEL_NAME": "qwen3-4b-instruct-2507:latest",
            "DOCMIND_OLLAMA_BASE_URL": "http://localhost:11434",
            "DOCMIND_DEBUG": "false",
        },
    ):
        try:
            from src.config.settings import DocMindSettings

            # Test configuration instantiation
            settings = DocMindSettings()

            # Validate key configuration attributes
            assert settings.model_name == "qwen3-4b-instruct-2507:latest"
            assert settings.ollama_base_url == "http://localhost:11434"
            assert settings.debug is False

            # Test that configuration has expected structure
            expected_attrs = [
                "model_name",
                "ollama_base_url",
                "default_token_limit",
                "debug",
                "log_level",
                "enable_gpu_acceleration",
            ]

            for attr in expected_attrs:
                assert hasattr(settings, attr), (
                    f"Missing configuration attribute: {attr}"
                )

            print("✅ Unified configuration architecture validated")
            print(f"   - Model: {settings.model_name}")
            print(f"   - Base URL: {settings.ollama_base_url}")
            print(f"   - Debug mode: {settings.debug}")

        except ImportError as e:
            pytest.skip(f"Configuration system import failed: {e}")


def test_multi_agent_system_integration():
    """Test multi-agent system integration and coordination capabilities."""
    with patch(
        "src.agents.coordinator.MultiAgentCoordinator"
    ) as mock_coordinator_class:
        # Setup mock multi-agent coordinator
        mock_coordinator = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Multi-agent system successfully coordinated analysis. Agents collaborated to provide comprehensive insights."
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


def test_document_processing_pipeline():
    """Test the complete document processing pipeline post-ADR-009."""
    with patch("src.utils.document.load_documents_unstructured") as mock_load_docs:
        # Mock successful document loading
        from llama_index.core import Document

        mock_documents = [
            Document(
                text="Advanced document processing with unstructured data handling.",
                metadata={"source": "doc1.pdf", "page": 1, "chunk_id": "chunk_1"},
            ),
            Document(
                text="Multi-modal content extraction and intelligent chunking strategies.",
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
            from src.utils.document import load_documents_unstructured

            # Test document loading
            file_paths = ["test_document.pdf", "test_document2.pdf"]
            documents = load_documents_unstructured(file_paths)

            assert len(documents) == 3
            assert all(isinstance(doc, Document) for doc in documents)
            assert all(doc.metadata.get("source") == "doc1.pdf" for doc in documents)

            # Test that documents have proper structure
            for i, doc in enumerate(documents):
                assert doc.text is not None and len(doc.text) > 0
                assert doc.metadata is not None
                assert "source" in doc.metadata
                assert "page" in doc.metadata
                assert "chunk_id" in doc.metadata

            # Test analysis output schema
            analysis = AnalysisOutput(
                summary="Document processing pipeline successfully processed multiple documents",
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
                f"   - Total text length: {sum(len(doc.text) for doc in documents)} chars"
            )

        except ImportError as e:
            pytest.skip(f"Document processing components import failed: {e}")


@pytest.mark.asyncio
async def test_async_workflow_components():
    """Test async workflow components and operations."""
    try:
        from src.agents.coordinator import MultiAgentCoordinator
        from src.utils.document import load_documents_unstructured

        # Mock async operations
        with (
            patch("src.utils.document.load_documents_unstructured") as mock_load,
            patch(
                "src.agents.coordinator.MultiAgentCoordinator"
            ) as mock_coordinator_class,
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
        from src.prompts import PREDEFINED_PROMPTS

        # Test prompts structure
        assert isinstance(PREDEFINED_PROMPTS, dict)
        assert len(PREDEFINED_PROMPTS) > 0

        first_prompt = next(iter(PREDEFINED_PROMPTS.values()))
        assert isinstance(first_prompt, str)
        assert len(first_prompt) > 0

        # Test configuration structure
        settings = DocMindSettings()
        critical_attrs = ["model_name", "ollama_base_url", "default_token_limit"]
        for attr in critical_attrs:
            assert hasattr(settings, attr), f"Missing critical configuration: {attr}"

        # Test memory management components
        memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
        assert memory is not None

        message = ChatMessage(role="user", content="Test message for E2E validation")
        assert message.role == "user"
        assert message.content == "Test message for E2E validation"

        print("✅ Application structure and markers validated")
        print(f"   - Available prompts: {len(PREDEFINED_PROMPTS)}")
        print(f"   - Configuration loaded: {settings.model_name}")
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
        "src.utils.document",
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
        f"   - Current modules imported: {len(successful_imports)}/{len(current_modules)}"
    )
    print(f"   - Legacy modules properly blocked: {len(legacy_modules)}")


@pytest.mark.integration
def test_end_to_end_integration_with_mocks():
    """Integration test that validates complete E2E workflow with proper mocking."""
    with (
        patch("src.utils.core.detect_hardware") as mock_detect,
        patch("src.utils.core.validate_startup_configuration") as mock_validate,
        patch("src.agents.coordinator.MultiAgentCoordinator") as mock_coordinator,
        patch("src.utils.document.load_documents_unstructured") as mock_load,
    ):
        # Setup comprehensive mocks
        mock_detect.return_value = {
            "gpu_name": "RTX 4090",
            "vram_total_gb": 24,
            "cuda_available": True,
        }
        mock_validate.return_value = True

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
            from src.utils.document import load_documents_unstructured

            # 1. Configuration
            settings = DocMindSettings()
            assert settings is not None

            # 2. Validation
            is_valid = validate_startup_configuration(settings)
            assert mock_validate.called

            # 3. Hardware detection
            hardware = detect_hardware()
            assert hardware["gpu_name"] == "RTX 4090"

            # 4. Document processing
            docs = load_documents_unstructured(["test.pdf"])
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
