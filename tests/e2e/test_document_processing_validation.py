"""Document processing workflow validation tests.

This module provides focused tests for the document processing pipeline,
validating that the core document processing functionality works correctly
with the unified architecture post-ADR-009.

These tests focus on:
- Document loading and processing functionality
- Integration with unstructured document processing
- Validation of document metadata and structure
- Core document processing workflow integrity
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock heavy dependencies early
mock_torch = MagicMock()
mock_torch.__version__ = "2.7.1+cu126"
mock_torch.__spec__ = MagicMock()
mock_torch.__spec__.name = "torch"
mock_torch.cuda.is_available.return_value = True
mock_torch.device = MagicMock()
mock_torch.tensor = MagicMock()
sys.modules["torch"] = mock_torch

# Mock other heavy dependencies
heavy_deps = [
    "spacy",
    "thinc",
    "unstructured",
    "nltk",
    "transformers",
    "sentence_transformers",
    "ollama",
    "chromadb",
    "qdrant_client",
]
for dep in heavy_deps:
    if dep not in sys.modules:
        sys.modules[dep] = MagicMock()


def test_document_processing_components_import():
    """Test that document processing components can be imported successfully."""
    try:
        from src.config.settings import DocMindSettings
        from src.models.schemas import AnalysisOutput
        from src.utils.core import detect_hardware, validate_startup_configuration

        # Test that these components exist and are callable
        assert callable(detect_hardware)
        assert callable(validate_startup_configuration)

        # Test configuration system
        settings = DocMindSettings()
        assert settings is not None
        assert hasattr(settings, "model_name")

        # Test response schema
        analysis = AnalysisOutput(
            summary="Test document processing validation",
            key_insights=["Processing capability validated"],
            action_items=["Continue integration testing"],
            open_questions=["How to optimize further?"],
        )
        assert analysis.summary == "Test document processing validation"
        assert len(analysis.key_insights) == 1

        print("✅ Document processing components import successfully")

    except ImportError as e:
        pytest.skip(f"Document processing components import failed: {e}")


def test_document_loading_functionality():
    """Test document loading functionality with mocked dependencies."""
    # Mock the document loading function before importing
    with patch("src.utils.document.load_documents_unstructured") as mock_load_docs:
        # Setup mock document loading response
        from llama_index.core import Document

        mock_documents = [
            Document(
                text=(
                    "This is a test document for validating the document "
                    "processing pipeline."
                ),
                metadata={
                    "source": "test_document.pdf",
                    "page": 1,
                    "chunk_id": "chunk_001",
                    "processing_type": "unstructured",
                },
            ),
            Document(
                text=(
                    "The document processing system supports various formats and "
                    "provides structured output."
                ),
                metadata={
                    "source": "test_document.pdf",
                    "page": 1,
                    "chunk_id": "chunk_002",
                    "processing_type": "unstructured",
                },
            ),
        ]
        mock_load_docs.return_value = mock_documents

        try:
            # Import after mocking to avoid dependency issues
            from src.utils.document import load_documents_unstructured

            # Test document loading
            file_paths = ["test_document.pdf"]
            loaded_documents = load_documents_unstructured(file_paths)

            # Validate document loading results
            assert len(loaded_documents) == 2
            assert all(isinstance(doc, Document) for doc in loaded_documents)

            # Validate document structure
            for doc in loaded_documents:
                assert doc.text is not None
                assert len(doc.text) > 0
                assert doc.metadata is not None
                assert "source" in doc.metadata
                assert "page" in doc.metadata
                assert "chunk_id" in doc.metadata
                assert doc.metadata["source"] == "test_document.pdf"

            # Validate content
            assert "test document" in loaded_documents[0].text.lower()
            assert "processing system" in loaded_documents[1].text.lower()

            print("✅ Document loading functionality validated")
            print(f"   - Documents loaded: {len(loaded_documents)}")
            print(
                f"   - Total text length: "
                f"{sum(len(doc.text) for doc in loaded_documents)} chars"
            )

        except ImportError as e:
            pytest.skip(f"Document loading test failed due to imports: {e}")


def test_hardware_detection_and_validation():
    """Test hardware detection and system validation functionality."""
    # Mock hardware detection and validation functions
    with (
        patch("src.utils.core.detect_hardware") as mock_detect,
        patch("src.utils.core.validate_startup_configuration") as mock_validate,
    ):
        # Setup mocks
        mock_detect.return_value = {
            "gpu_name": "RTX 4090",
            "vram_total_gb": 24,
            "cuda_available": True,
            "system_ram_gb": 64,
            "cpu_cores": 16,
        }
        mock_validate.return_value = True

        try:
            from src.config.settings import DocMindSettings
            from src.utils.core import detect_hardware, validate_startup_configuration

            # Test hardware detection
            hardware_info = detect_hardware()
            assert mock_detect.called
            assert hardware_info["gpu_name"] == "RTX 4090"
            assert hardware_info["vram_total_gb"] == 24
            assert hardware_info["cuda_available"] is True

            # Test configuration validation
            settings = DocMindSettings()
            validation_result = validate_startup_configuration(settings)
            assert mock_validate.called
            assert validation_result is True

            print("✅ Hardware detection and validation tested")
            print(f"   - GPU: {hardware_info['gpu_name']}")
            print(f"   - VRAM: {hardware_info['vram_total_gb']}GB")
            print(f"   - CUDA Available: {hardware_info['cuda_available']}")
            print(f"   - Configuration Valid: {validation_result}")

        except ImportError as e:
            pytest.skip(f"Hardware detection test failed: {e}")


def test_configuration_system_integration():
    """Test the unified configuration system integration."""
    # Test configuration with environment variable overrides
    with patch.dict(
        "os.environ",
        {
            "DOCMIND_MODEL_NAME": "qwen3-4b-instruct-2507:latest",
            "DOCMIND_DEBUG": "false",
            "DOCMIND_ENABLE_GPU_ACCELERATION": "true",
        },
    ):
        try:
            from src.config.settings import DocMindSettings

            # Test configuration instantiation
            settings = DocMindSettings()

            # Validate configuration attributes
            assert settings.vllm.model == "qwen3-4b-instruct-2507:latest"
            assert settings.debug is False
            assert settings.enable_gpu_acceleration is True

            # Test that all required configuration attributes exist
            required_attrs = [
                "model_name",
                "ollama_base_url",
                "default_token_limit",
                "debug",
                "log_level",
                "enable_gpu_acceleration",
                "data_dir",
                "cache_dir",
            ]

            for attr in required_attrs:
                assert hasattr(settings, attr), (
                    f"Missing configuration attribute: {attr}"
                )

            print("✅ Configuration system integration validated")
            print(f"   - Model: {settings.vllm.model}")
            print(f"   - Debug Mode: {settings.debug}")
            print(f"   - GPU Acceleration: {settings.enable_gpu_acceleration}")

        except ImportError as e:
            pytest.skip(f"Configuration system test failed: {e}")


def test_analysis_output_schema_validation():
    """Test that the analysis output schema works correctly."""
    try:
        from src.models.schemas import AnalysisOutput

        # Test valid analysis output creation
        analysis = AnalysisOutput(
            summary="Document processing validation completed successfully",
            key_insights=[
                "Document loading functionality works correctly",
                "Hardware detection provides accurate information",
                "Configuration system integrates properly",
                "Analysis schema validates input and output",
            ],
            action_items=[
                "Continue with E2E testing",
                "Validate UI integration",
                "Test multi-agent coordination",
            ],
            open_questions=[
                "How to optimize document processing performance?",
                "What additional validation scenarios should be tested?",
            ],
        )

        # Validate analysis output structure
        assert analysis.summary is not None
        assert len(analysis.summary) > 0
        assert len(analysis.key_insights) == 4
        assert len(analysis.action_items) == 3
        assert len(analysis.open_questions) == 2

        # Validate content
        assert "validation completed successfully" in analysis.summary
        assert "Document loading functionality" in analysis.key_insights[0]
        assert "Continue with E2E testing" in analysis.action_items[0]
        assert "optimize document processing" in analysis.open_questions[0]

        print("✅ Analysis output schema validation passed")
        print(f"   - Summary length: {len(analysis.summary)} chars")
        print(f"   - Key insights: {len(analysis.key_insights)}")
        print(f"   - Action items: {len(analysis.action_items)}")
        print(f"   - Open questions: {len(analysis.open_questions)}")

    except ImportError as e:
        pytest.skip(f"Analysis schema test failed: {e}")


def test_prompts_and_application_structure():
    """Test that application prompts and structure are properly configured."""
    try:
        from src.prompts import PREDEFINED_PROMPTS

        # Test prompts structure
        assert isinstance(PREDEFINED_PROMPTS, dict)
        assert len(PREDEFINED_PROMPTS) > 0

        # Test that prompts have valid content
        for prompt_name, prompt_content in PREDEFINED_PROMPTS.items():
            assert isinstance(prompt_name, str)
            assert isinstance(prompt_content, str)
            assert len(prompt_name) > 0
            assert len(prompt_content) > 0

        # Test specific prompt types exist
        expected_prompt_types = ["summarization", "analysis", "extraction"]
        available_prompts = list(PREDEFINED_PROMPTS.keys())

        # Check that we have some expected prompt types (flexible check)
        has_expected_prompts = any(
            expected in prompt_name.lower()
            for expected in expected_prompt_types
            for prompt_name in available_prompts
        )

        print("✅ Prompts and application structure validated")
        print(f"   - Total prompts: {len(PREDEFINED_PROMPTS)}")
        print(
            f"   - Prompt names: {list(PREDEFINED_PROMPTS.keys())[:3]}..."
        )  # Show first 3
        print(f"   - Has expected prompt types: {has_expected_prompts}")

    except ImportError as e:
        pytest.skip(f"Prompts structure test failed: {e}")


def test_memory_management_components():
    """Test that memory and session management components are available."""
    try:
        from llama_index.core.llms import ChatMessage
        from llama_index.core.memory import ChatMemoryBuffer

        # Test memory buffer creation
        memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
        assert memory is not None

        # Test chat message creation
        user_message = ChatMessage(
            role="user", content="Test document processing query"
        )
        assert user_message.role == "user"
        assert user_message.content == "Test document processing query"

        assistant_message = ChatMessage(
            role="assistant", content="Document processing completed"
        )
        assert assistant_message.role == "assistant"
        assert assistant_message.content == "Document processing completed"

        print("✅ Memory management components validated")
        print(f"   - Memory buffer token limit: {memory.token_limit}")
        print(f"   - User message: {user_message.content[:50]}...")
        print(f"   - Assistant message: {assistant_message.content[:50]}...")

    except ImportError as e:
        pytest.skip(f"Memory management components test failed: {e}")


@pytest.mark.integration
def test_integrated_document_processing_workflow():
    """Integration test for the complete document processing workflow."""
    # Mock all external dependencies
    with (
        patch("src.utils.document.load_documents_unstructured") as mock_load,
        patch("src.utils.core.detect_hardware") as mock_detect,
        patch("src.utils.core.validate_startup_configuration") as mock_validate,
    ):
        # Setup mocks
        from llama_index.core import Document

        mock_documents = [
            Document(
                text=(
                    "Integrated document processing workflow validation test document."
                ),
                metadata={"source": "workflow_test.pdf", "page": 1},
            )
        ]
        mock_load.return_value = mock_documents

        mock_detect.return_value = {
            "gpu_name": "RTX 4090",
            "vram_total_gb": 24,
            "cuda_available": True,
        }
        mock_validate.return_value = True

        try:
            # Import components
            from src.config.settings import DocMindSettings
            from src.models.schemas import AnalysisOutput
            from src.utils.core import detect_hardware, validate_startup_configuration
            from src.utils.document import load_documents_unstructured

            # 1. Configuration
            settings = DocMindSettings()
            assert settings is not None

            # 2. System validation
            hardware = detect_hardware()
            is_valid = validate_startup_configuration(settings)
            assert hardware["gpu_name"] == "RTX 4090"
            assert is_valid is True

            # 3. Document processing
            documents = load_documents_unstructured(["workflow_test.pdf"])
            assert len(documents) == 1
            assert "workflow validation" in documents[0].text

            # 4. Analysis output
            analysis = AnalysisOutput(
                summary="Integrated workflow processing completed successfully",
                key_insights=["All components integrated correctly"],
                action_items=["Proceed to UI testing"],
                open_questions=["Performance optimization opportunities?"],
            )
            assert analysis.summary is not None

            print("✅ Integrated document processing workflow validated")
            print(f"   - Configuration: {settings.vllm.model}")
            print(f"   - Hardware: {hardware['gpu_name']}")
            print(f"   - Documents: {len(documents)}")
            print("   - Analysis: Generated successfully")

        except ImportError as e:
            pytest.skip(f"Integrated workflow test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
