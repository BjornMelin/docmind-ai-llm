#!/usr/bin/env python3
"""Real validation tests for DocMind AI system components.

This module tests actual functionality with minimal mocking to validate
that the system works correctly in real scenarios. Tests that require
external services are marked with appropriate pytest marks.
"""

import logging

import pytest
from llama_index.core import Document

# Import DocMind AI components
from src.models.core import AppSettings
from src.utils.core import (
    detect_hardware,
    verify_rrf_configuration,
)


class TestRealConfiguration:
    """Test real configuration loading and validation."""

    def test_settings_loading(self):
        """Test that settings load correctly from environment."""
        settings = AppSettings()

        # Verify core configuration is loaded
        assert isinstance(settings.backend, str)
        assert isinstance(settings.ollama_base_url, str)
        assert isinstance(settings.default_model, str)

        # Verify embedding configuration
        assert isinstance(settings.dense_embedding_model, str)
        assert isinstance(settings.sparse_embedding_model, str)
        assert settings.dense_embedding_dimension > 0

        # Verify SPLADE++ is configured
        assert "Splade_PP_en_v1" in settings.sparse_embedding_model

        # Verify BGE-Large is configured
        assert "bge-large-en-v1.5" in settings.dense_embedding_model

    def test_rrf_configuration_real_settings(self):
        """Test RRF configuration with real settings."""
        settings = AppSettings()
        verification = verify_rrf_configuration(settings)

        # Log verification results for debugging
        logging.info(f"RRF Verification: {verification}")

        # Check that configuration meets requirements
        assert verification["weights_correct"] is True, (
            f"Weights incorrect: {verification['issues']}"
        )
        assert verification["alpha_in_range"] is True, (
            f"Alpha out of range: {verification['issues']}"
        )

        # Verify computed alpha is reasonable
        if "computed_hybrid_alpha" in verification:
            computed_alpha = verification["computed_hybrid_alpha"]
            assert 0.0 <= computed_alpha <= 1.0
            assert (
                abs(computed_alpha - 0.7) < 0.05
            )  # Should be close to 0.7 for 0.7/0.3 split

    def test_gpu_configuration_consistency(self):
        """Test GPU configuration is consistent across settings."""
        settings = AppSettings()

        # Verify GPU settings are boolean
        assert isinstance(settings.gpu_acceleration, bool)
        # Check if enable_quantization exists (it may not be in simplified settings)
        if hasattr(settings, "enable_quantization"):
            assert isinstance(settings.enable_quantization, bool)

        # Verify batch size is reasonable
        assert 1 <= settings.embedding_batch_size <= 512

        # Test other GPU-related settings if available
        if hasattr(settings, "cuda_device_id"):
            assert settings.cuda_device_id >= 0


class TestHardwareDetectionReal:
    """Test real hardware detection functionality."""

    def test_hardware_detection_runs(self):
        """Test that hardware detection runs without errors."""
        hardware_info = detect_hardware()

        # Verify return structure
        required_keys = [
            "cuda_available",
            "gpu_name",
            "vram_total_gb",
        ]
        # fastembed_providers might not be in the simplified detect_hardware function
        if "fastembed_providers" in hardware_info:
            required_keys.append("fastembed_providers")
        for key in required_keys:
            assert key in hardware_info

        # Verify types
        assert isinstance(hardware_info["cuda_available"], bool)
        assert isinstance(hardware_info["gpu_name"], str)
        assert hardware_info["vram_total_gb"] is None or isinstance(
            hardware_info["vram_total_gb"], int | float
        )
        # Only check fastembed_providers if it exists
        if "fastembed_providers" in hardware_info:
            assert isinstance(hardware_info["fastembed_providers"], list)

    def test_gpu_detection_consistency(self):
        """Test GPU detection is consistent."""
        hardware_info = detect_hardware()

        # If CUDA is available, should have GPU info
        if hardware_info["cuda_available"]:
            assert hardware_info["gpu_name"] != "Unknown"
            assert hardware_info["vram_total_gb"] is not None
            assert hardware_info["vram_total_gb"] > 0


# Query analysis tests removed since analyze_query_complexity function no longer exists
# in the simplified agent_factory.py. The refactoring removed complex query analysis
# in favor of a simple ReActAgent that handles all query types uniformly.


class TestMultiAgentCoordinatorReal:
    """Test real MultiAgentCoordinator functionality."""

    def test_coordinator_functions_exist(self):
        """Test that coordinator functions exist and are callable."""
        from src.agents.coordinator import (
            MultiAgentCoordinator,
            create_multi_agent_coordinator,
        )

        # Verify functions exist and are callable
        assert callable(MultiAgentCoordinator)
        assert callable(create_multi_agent_coordinator)

        # Test that MultiAgentCoordinator has required methods
        assert hasattr(MultiAgentCoordinator, "process_query")
        assert callable(MultiAgentCoordinator.process_query)

    def test_coordinator_initialization(self):
        """Test that MultiAgentCoordinator can be initialized."""
        import inspect

        from src.agents.coordinator import MultiAgentCoordinator

        # Verify coordinator constructor signature
        sig = inspect.signature(MultiAgentCoordinator.__init__)
        # MultiAgentCoordinator should have standard parameters
        expected_params = ["self"]
        # The constructor should at least have self parameter
        assert all(param in sig.parameters for param in expected_params)

        # Test that process_query method exists with correct signature
        process_query_sig = inspect.signature(MultiAgentCoordinator.process_query)
        assert "self" in process_query_sig.parameters
        assert "query" in process_query_sig.parameters


class TestDocumentHandlingReal:
    """Test real document handling and processing."""

    def test_document_creation_with_metadata(self):
        """Test document creation with proper metadata."""
        test_metadata = {
            "source": "test_document.pdf",
            "page_number": 1,
            "has_images": True,
            "document_type": "research_paper",
        }

        doc = Document(
            text="This is a test document for validating metadata handling.",
            metadata=test_metadata,
        )

        # Verify document creation
        assert doc.text == "This is a test document for validating metadata handling."
        assert doc.metadata == test_metadata

        # Verify metadata access
        assert doc.metadata["source"] == "test_document.pdf"
        assert doc.metadata["page_number"] == 1
        assert doc.metadata["has_images"] is True

    def test_document_list_handling(self):
        """Test handling of document lists for batch processing."""
        documents = []
        for i in range(5):
            doc = Document(
                text=f"Document {i} content for testing batch processing.",
                metadata={"doc_id": i, "batch": "test_batch"},
            )
            documents.append(doc)

        assert len(documents) == 5

        # Verify each document is properly formed
        for i, doc in enumerate(documents):
            assert f"Document {i}" in doc.text
            assert doc.metadata["doc_id"] == i
            assert doc.metadata["batch"] == "test_batch"


class TestConfigurationValidation:
    """Test configuration validation and consistency."""

    def test_embedding_dimension_consistency(self):
        """Test that embedding dimensions are consistent."""
        settings = AppSettings()

        # BGE-Large should be 1024 dimensions
        if "bge-large" in settings.dense_embedding_model.lower():
            assert settings.dense_embedding_dimension == 1024

        # Verify dimension is positive
        assert settings.dense_embedding_dimension > 0

    def test_batch_size_configurations(self):
        """Test batch size configurations are reasonable."""
        settings = AppSettings()

        # Embedding batch size should be reasonable for most hardware
        assert 1 <= settings.embedding_batch_size <= 512

        # Prefetch factor should be small
        assert 1 <= settings.prefetch_factor <= 8

    def test_reranking_configuration(self):
        """Test reranking configuration meets Phase 2.2 requirements."""
        settings = AppSettings()

        # Phase 2.2: retrieve 20, rerank to 5
        assert settings.reranking_top_k == 5

        # ColBERT should be enabled
        assert settings.enable_colbert_reranking is True


class TestAsyncPatterns:
    """Test async patterns and compatibility."""

    @pytest.mark.asyncio
    async def test_async_function_availability(self):
        """Test that async functions are available and callable."""
        from src.retrieval.integration import create_index_async
        from src.utils.database import setup_hybrid_collection_async

        # Verify functions exist and are callable
        assert callable(create_index_async)
        assert callable(setup_hybrid_collection_async)

        # Verify they are coroutine functions
        import inspect

        assert inspect.iscoroutinefunction(create_index_async)
        assert inspect.iscoroutinefunction(setup_hybrid_collection_async)

    def test_sync_async_compatibility(self):
        """Test that key functions exist for embedding operations."""
        # Skip removed embedding functions - replaced with FEAT-002 retrieval system
        pytest.skip("Embedding utility functions removed from src.utils.embedding")


class TestErrorHandlingReal:
    """Test real error handling scenarios."""

    def test_settings_validation_errors(self):
        """Test settings validation with invalid values."""
        # Test with invalid RRF weights
        settings = AppSettings()

        # Weights should be between 0 and 1
        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            settings.rrf_fusion_weight_dense = 1.5

        with pytest.raises(ValueError, match="Weight must be between 0 and 1"):
            settings.rrf_fusion_weight_sparse = -0.1

    def test_batch_size_validation(self):
        """Test batch size validation."""
        settings = AppSettings()

        # Test invalid batch sizes
        with pytest.raises(ValueError, match="Batch size must be greater than 0"):
            settings.embedding_batch_size = 0

        with pytest.raises(ValueError, match="Batch size cannot exceed 512"):
            settings.embedding_batch_size = 1000  # Too large

    def test_graceful_degradation_patterns(self):
        """Test graceful degradation patterns exist."""
        # Test that fallback mechanisms are in place
        from src.utils.core import detect_hardware

        # Hardware detection should always return a dict
        hardware_info = detect_hardware()
        assert isinstance(hardware_info, dict)

        # Should have fallback values
        assert "cuda_available" in hardware_info
        assert "gpu_name" in hardware_info


class TestIntegrationReadiness:
    """Test system readiness for integration."""

    def test_all_required_models_configured(self):
        """Test that all required models are properly configured."""
        settings = AppSettings()

        # Dense embedding model
        assert settings.dense_embedding_model is not None
        assert len(settings.dense_embedding_model) > 0
        assert "bge" in settings.dense_embedding_model.lower()

        # Sparse embedding model
        assert settings.sparse_embedding_model is not None
        assert len(settings.sparse_embedding_model) > 0
        assert "splade" in settings.sparse_embedding_model.lower()

    def test_qdrant_configuration(self):
        """Test Qdrant configuration is properly set."""
        settings = AppSettings()

        assert settings.qdrant_url is not None
        assert len(settings.qdrant_url) > 0
        assert settings.qdrant_url.startswith("http")

    def test_llm_backend_configuration(self):
        """Test LLM backend configuration."""
        settings = AppSettings()

        # Check if backend attribute exists (may not be in simplified settings)
        if hasattr(settings, "backend"):
            assert settings.backend in ["ollama", "lmstudio", "llamacpp"]

        # Check default_model attribute
        assert hasattr(settings, "default_model")
        assert settings.default_model is not None
        assert len(settings.default_model) > 0

    def test_system_components_importable(self):
        """Test that all system components can be imported."""
        # Skip removed embedding functions - replaced with FEAT-002 retrieval system
        pytest.skip("create_vector_index and get_embed_model functions removed")


# Test configuration
def pytest_configure():
    """Configure pytest for real validation tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
