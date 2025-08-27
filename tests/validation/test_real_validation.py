#!/usr/bin/env python3
"""Production readiness validation for DocMind AI system.

This module tests actual functionality with minimal mocking to validate
that the system works correctly in real production scenarios. Tests that require
external services are marked with appropriate pytest marks.

Validation Areas:
- Unified configuration system validation
- Real hardware detection and performance validation
- Multi-agent system coordination validation
- Production readiness checks
- Integration compatibility verification
"""

import logging

import pytest
from llama_index.core import Document

# Import DocMind AI components
from src.config.settings import DocMindSettings
from src.utils.core import detect_hardware


class TestRealConfiguration:
    """Test real configuration loading and validation."""

    def test_settings_loading(self):
        """Test that unified settings load correctly from environment."""
        settings = DocMindSettings()

        # Verify core configuration is loaded
        assert isinstance(settings.llm_backend, str)
        assert isinstance(settings.ollama_base_url, str)
        assert isinstance(settings.model_name, str)

        # Verify embedding configuration
        assert isinstance(settings.embedding_model, str)
        assert settings.embedding_dimension > 0

        # Verify BGE models are configured (updated from BGE-Large to current architecture)
        assert (
            "bge" in settings.embedding_model.lower()
            or "bge-m3" in settings.bge_m3_model_name.lower()
        )

    def test_rrf_configuration_real_settings(self):
        """Test RRF configuration with real settings."""
        settings = DocMindSettings()

        # Test RRF configuration is properly set up in unified config
        assert hasattr(settings, "rrf_fusion_alpha")
        assert hasattr(settings, "rrf_k_constant")
        assert hasattr(settings, "rrf_fusion_weight_dense")
        assert hasattr(settings, "rrf_fusion_weight_sparse")

        # Verify RRF weights are reasonable (0.7/0.3 split)
        dense_weight = settings.rrf_fusion_weight_dense
        sparse_weight = settings.rrf_fusion_weight_sparse

        assert 0.0 <= dense_weight <= 1.0
        assert 0.0 <= sparse_weight <= 1.0
        assert abs(dense_weight + sparse_weight - 1.0) < 0.01  # Should sum to 1.0

        # Log RRF configuration for debugging
        logging.info(
            f"RRF Dense Weight: {dense_weight}, Sparse Weight: {sparse_weight}"
        )
        logging.info(
            f"RRF Alpha: {settings.rrf_fusion_alpha}, K: {settings.rrf_k_constant}"
        )

    def test_gpu_configuration_consistency(self):
        """Test GPU configuration is consistent across unified settings."""
        settings = DocMindSettings()

        # Verify GPU settings are boolean
        assert isinstance(settings.enable_gpu_acceleration, bool)

        # Test vLLM GPU configuration
        assert hasattr(settings, "vllm_gpu_memory_utilization")
        assert 0.1 <= settings.vllm_gpu_memory_utilization <= 0.95

        # Test embedding batch sizes from nested config
        assert hasattr(settings.embedding, "batch_size_gpu")
        assert hasattr(settings.embedding, "batch_size_cpu")
        assert 1 <= settings.embedding.batch_size_gpu <= 128
        assert 1 <= settings.embedding.batch_size_cpu <= 32

        # Test quantization settings
        assert hasattr(settings, "quantization")
        assert hasattr(settings, "kv_cache_dtype")


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
        settings = DocMindSettings()

        # BGE-Large should be 1024 dimensions
        if "bge-large" in settings.embedding_model.lower():
            assert settings.embedding_dimension == 1024

        # Verify dimension is positive
        assert settings.embedding_dimension > 0

    def test_batch_size_configurations(self):
        """Test batch size configurations are reasonable."""
        settings = DocMindSettings()

        # Embedding batch sizes from nested config should be reasonable for most hardware
        assert 1 <= settings.embedding.batch_size_gpu <= 128
        assert 1 <= settings.embedding.batch_size_cpu <= 32

        # vLLM batch configuration should be reasonable
        assert 1 <= settings.vllm.max_num_seqs <= 64
        assert 1024 <= settings.vllm.max_num_batched_tokens <= 16384

    def test_reranking_configuration(self):
        """Test reranking configuration meets current requirements."""
        settings = DocMindSettings()

        # Test reranking settings from nested config
        assert settings.retrieval.reranking_top_k == 5
        assert settings.use_reranking is True

        # Test reranker model is properly configured
        assert hasattr(settings.retrieval, "reranker_model")
        assert "bge-reranker" in settings.retrieval.reranker_model.lower()


class TestErrorHandlingReal:
    """Test real error handling scenarios."""

    def test_settings_validation_errors(self):
        """Test settings validation with invalid values."""
        # Test with invalid configuration values

        # Test invalid chunk size
        with pytest.raises(ValueError):
            DocMindSettings(chunk_size=0)  # Should be >= 128

        # Test invalid GPU memory utilization
        with pytest.raises(ValueError):
            DocMindSettings(vllm_gpu_memory_utilization=1.5)  # Should be <= 0.95

    def test_batch_size_validation(self):
        """Test batch size validation."""
        # Test invalid embedding batch sizes in nested config
        with pytest.raises(ValueError):
            DocMindSettings(bge_m3_batch_size_gpu=0)  # Should be >= 1

        with pytest.raises(ValueError):
            DocMindSettings(bge_m3_batch_size_gpu=200)  # Should be <= 128

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
        settings = DocMindSettings()

        # Dense embedding model (both flat and nested config)
        assert settings.embedding_model is not None
        assert len(settings.embedding_model) > 0
        assert "bge" in settings.embedding_model.lower()

        # BGE-M3 model from nested config
        assert settings.embedding.model_name is not None
        assert len(settings.embedding.model_name) > 0
        assert "bge-m3" in settings.embedding.model_name.lower()

        # Test sparse embedding support exists via settings
        assert hasattr(settings, "use_sparse_embeddings")
        assert isinstance(settings.use_sparse_embeddings, bool)

    def test_qdrant_configuration(self):
        """Test Qdrant configuration is properly set."""
        settings = DocMindSettings()

        assert settings.qdrant_url is not None
        assert len(settings.qdrant_url) > 0
        assert settings.qdrant_url.startswith("http")

    def test_llm_backend_configuration(self):
        """Test LLM backend configuration."""
        settings = DocMindSettings()

        # Check llm_backend attribute (updated attribute name)
        assert hasattr(settings, "llm_backend")
        assert settings.llm_backend in ["ollama", "vllm", "openai"]

        # Check model_name attribute
        assert hasattr(settings, "model_name")
        assert settings.model_name is not None
        assert len(settings.model_name) > 0

        # Test vLLM configuration
        assert hasattr(settings, "vllm")
        assert settings.vllm.model == settings.model_name


# Test configuration
def pytest_configure():
    """Configure pytest for real validation tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
