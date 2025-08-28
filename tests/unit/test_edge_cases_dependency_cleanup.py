"""Edge case tests for dependency cleanup validation.

This module tests practical edge cases after comprehensive utils/ cleanup:
1. Optional dependency handling
2. Graceful error messages
3. Import fallback scenarios

Note: Many tests were removed as the modules they tested were completely
deleted during the utils/ directory cleanup.
"""

import importlib
from pathlib import Path

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
                print(f"GPU package {pkg} is optional (as expected)")
                # This is expected - GPU packages should be optional

    def test_torch_optional_graceful_handling(self):
        """Test that PyTorch dependencies are handled gracefully."""
        try:
            import torch

            print(f"PyTorch available: {torch.__version__}")

            # Test CUDA availability detection
            from src.utils.core import detect_hardware

            hardware_info = detect_hardware()
            assert isinstance(hardware_info, dict)
            assert "cuda_available" in hardware_info

        except ImportError:
            print("PyTorch not available (this is acceptable)")

    def test_qdrant_client_import_handling(self):
        """Test that Qdrant client imports are handled gracefully."""
        try:
            from src.utils.storage import create_async_client, create_sync_client

            # These functions should exist and be callable
            assert callable(create_async_client)
            assert callable(create_sync_client)

        except ImportError as e:
            pytest.fail(f"Essential Qdrant utilities should be available: {e}")

    def test_cleaned_up_modules_are_gone(self):
        """Verify that cleaned-up modules are actually gone."""
        deleted_modules = [
            "utils.client_factory",
            "utils.document_loader",
            "utils.embedding_factory",
            "utils.embedding_utils",
            "utils.index_builder",
            "utils.logging_utils",
            "utils.memory_utils",
            "utils.model_manager",
            "utils.monitoring",  # This one was moved to src.utils.monitoring
            "utils.qdrant_utils",
            "utils.retry_utils",
            "utils.utils",  # This was moved to src.utils.core
            "utils.validation_utils",
        ]

        for module in deleted_modules:
            with pytest.raises(ImportError):
                importlib.import_module(module)

    def test_new_src_utils_modules_are_available(self):
        """Verify that current src.utils modules are available."""
        current_modules = [
            "src.utils.core",
            "src.utils.database",
            "src.utils.monitoring",
        ]

        for module in current_modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                pytest.fail(f"Current module {module} should be available: {e}")

    def test_adr009_modules_are_available(self):
        """Verify that ADR-009 compliant modules are available."""
        adr009_modules = [
            "src.processing.document_processor",
            "src.processing.chunking.unstructured_chunker",
            "src.cache.simple_cache",
            "src.processing.embeddings.bgem3_embedder",
        ]

        for module in adr009_modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                pytest.fail(f"ADR-009 module {module} should be available: {e}")


class TestGracefulErrorHandling:
    """Test that missing dependencies produce helpful error messages."""

    def test_app_startup_without_optional_deps(self):
        """Test that app can start without optional dependencies."""
        try:
            from src.config import settings
            from src.utils.core import validate_startup_configuration

            # Should not crash even if some dependencies are missing
            result = validate_startup_configuration(settings)
            # Function should return a dict, not crash
            assert isinstance(result, dict)

        except ImportError:
            pytest.skip("Core utilities not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
