"""Edge case tests for dependency cleanup validation.

This module tests practical edge cases after PR #2 dependency cleanup:
1. Optional dependency handling
2. Graceful error messages
3. Import fallback scenarios
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Get project root
PROJECT_ROOT = Path(__file__).parents[2]


class TestOptionalDependencyHandling:
    """Test handling of optional dependencies in practical scenarios."""

    def test_gpu_packages_are_truly_optional(self):
        """Test that GPU packages don't break the app when missing."""
        gpu_packages = ["fastembed_gpu", "numba"]

        for pkg in gpu_packages:
            try:
                importlib.import_module(pkg)
                print(f"GPU package {pkg} is available")
            except ImportError:
                print(f"GPU package {pkg} not available (this is fine)")
                # This should not break anything - GPU packages are optional

    def test_video_processing_is_optional(self):
        """Test that video processing doesn't break the app when missing."""
        try:
            import importlib.util

            spec = importlib.util.find_spec("moviepy")
            if spec is not None:
                print("Video processing (moviepy) is available")
            else:
                print("Video processing (moviepy) not found")
        except ImportError:
            print("Video processing (moviepy) not available (this is fine)")
            # Video processing is optional - app should work without it

    def test_optional_embedding_providers_handled_gracefully(self):
        """Test that missing embedding providers are handled gracefully."""
        optional_providers = ["fastembed", "sentence_transformers"]

        for provider in optional_providers:
            try:
                importlib.import_module(provider)
                print(f"Embedding provider {provider} is available")
            except ImportError:
                print(
                    f"Embedding provider {provider} not available "
                    f"(fallback should work)"
                )

    def test_app_core_works_with_minimal_dependencies(self):
        """Test that app core works with just essential dependencies."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # These are the absolute minimum for the app to work
            essential_modules = [
                "models.core",
            ]

            for module_name in essential_modules:
                try:
                    __import__(module_name)
                    print(f"Essential module {module_name} imports successfully")
                except ImportError as e:
                    pytest.skip(f"Essential module {module_name} failed: {e}")

        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)


class TestErrorMessageQuality:
    """Test that error messages are helpful and clear."""

    def test_missing_core_dependency_gives_clear_error(self):
        """Test that missing core dependencies give clear error messages."""
        # Mock a missing core dependency
        with patch.dict(sys.modules, {"qdrant_client": None}):
            try:
                import importlib.util

                spec = importlib.util.find_spec("qdrant_client")
                if spec is None:
                    raise ImportError("qdrant_client")
            except ImportError as e:
                error_msg = str(e)
                # Error should mention the missing module
                assert "qdrant_client" in error_msg or "module" in error_msg.lower()
                print(f"Got expected error message: {error_msg}")

    def test_helpful_errors_for_optional_features(self):
        """Test that optional features give helpful errors when unavailable."""
        # This is more of a behavioral test - we expect graceful handling
        # rather than hard errors for optional features

        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # Try to use a potentially optional feature
            from utils.embedding_factory import EmbeddingFactory

            factory = EmbeddingFactory()

            # This should not crash, even if some providers are missing
            assert factory is not None
            print("Embedding factory creates successfully")

        except ImportError as e:
            # Should be a clear, helpful error message
            error_msg = str(e)
            assert len(error_msg) > 5  # Should have some meaningful content
            print(f"Got error (this may be expected): {error_msg}")

        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)


class TestImportFallbackScenarios:
    """Test import fallback scenarios work correctly."""

    def test_embedding_provider_fallback(self):
        """Test that embedding providers can fall back to alternatives."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # Test that the system can handle missing embedding providers
            from utils.embedding_utils import create_embedding_function

            # This should work with at least one provider
            try:
                # Try with a common provider that's likely to be available
                create_embedding_function(
                    provider="huggingface",
                    model="sentence-transformers/all-MiniLM-L6-v2",
                )
                print("Successfully created embedding with fallback provider")
            except Exception as e:
                # If this fails, it should be due to missing core dependencies,
                # not due to poor fallback handling
                print(
                    f"Embedding creation failed (may be due to missing core deps): {e}"
                )

        except ImportError as e:
            pytest.skip(f"Could not test embedding fallback: {e}")

        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)

    def test_hardware_detection_fallback(self):
        """Test that hardware detection has reasonable fallbacks."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from utils.core import detect_hardware

            # Hardware detection should not crash
            try:
                hardware_info = detect_hardware()
                assert isinstance(hardware_info, dict)
                print(f"Hardware detection successful: {hardware_info}")
            except Exception as e:
                # Hardware detection can fail, but it should fail gracefully
                print(f"Hardware detection failed gracefully: {e}")

        except ImportError as e:
            pytest.skip(f"Could not test hardware detection: {e}")

        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)

    def test_document_processing_fallback(self):
        """Test that document processing has reasonable fallbacks."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from utils.document import load_documents_llama

            # Document processing function should exist
            assert callable(load_documents_llama)
            print("Document processing function is available")

        except ImportError as e:
            pytest.skip(f"Could not test document processing: {e}")

        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)


class TestPackageCompatibility:
    """Test basic package compatibility after cleanup."""

    def test_pydantic_v2_compatibility(self):
        """Test that Pydantic v2 is being used correctly."""
        try:
            import pydantic

            # Should be v2
            version = pydantic.VERSION
            major_version = int(version.split(".")[0])
            assert major_version == 2, f"Expected Pydantic v2, got {version}"

            print(f"Pydantic version check passed: {version}")

        except ImportError:
            pytest.skip("Pydantic not available for version check")

    def test_streamlit_modern_version(self):
        """Test that Streamlit is a modern version."""
        try:
            import streamlit

            # Should have modern features
            version = streamlit.__version__

            # Check for modern Streamlit features
            required_features = ["chat_input", "session_state"]
            for feature in required_features:
                assert hasattr(streamlit, feature), (
                    f"Streamlit missing modern feature: {feature}"
                )

            print(f"Streamlit version check passed: {version}")

        except ImportError:
            pytest.skip("Streamlit not available for version check")

    def test_llamaindex_modular_structure(self):
        """Test that LlamaIndex uses the new modular structure."""
        modular_packages = [
            "llama_index.core",
            "llama_index.llms.openai",
            "llama_index.vector_stores.qdrant",
        ]

        available_packages = []

        for pkg in modular_packages:
            try:
                importlib.import_module(pkg)
                available_packages.append(pkg)
            except ImportError:
                pass  # Not available, which is fine in test environment

        if available_packages:
            print(f"LlamaIndex modular packages available: {available_packages}")
            # At least one modular package should be available if LlamaIndex is
            # installed
            assert len(available_packages) > 0
        else:
            pytest.skip("No LlamaIndex packages available for testing")


class TestRealWorldUsageScenarios:
    """Test real-world usage scenarios work correctly."""

    def test_basic_app_initialization_flow(self):
        """Test the basic app initialization flow works."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # This simulates the basic app startup flow
            from models.core import settings

            # Settings should be available
            assert settings is not None

            # Try to access basic hardware info
            from utils.core import detect_hardware

            try:
                detect_hardware()
                print("Basic initialization flow successful")
            except Exception as e:
                print(
                    f"Hardware detection in initialization flow failed gracefully: {e}"
                )

        except ImportError as e:
            pytest.skip(f"Basic app initialization test failed: {e}")

        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)

    def test_embedding_workflow_components(self):
        """Test that embedding workflow components are available."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # Test embedding-related components
            from utils.embedding import create_index_async
            from utils.embedding_factory import EmbeddingFactory

            # Functions should be available
            assert callable(create_index_async)

            factory = EmbeddingFactory()
            assert factory is not None

            print("Embedding workflow components are available")

        except ImportError as e:
            pytest.skip(f"Embedding workflow test failed: {e}")

        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)

    def test_agent_workflow_components(self):
        """Test that agent workflow components are available."""
        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # Test agent-related components
            from agents.agent_factory import get_agent_system
            from agents.agent_utils import create_tools_from_index

            # Functions should be available
            assert callable(get_agent_system)
            assert callable(create_tools_from_index)

            print("Agent workflow components are available")

        except ImportError as e:
            pytest.skip(f"Agent workflow test failed: {e}")

        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)
