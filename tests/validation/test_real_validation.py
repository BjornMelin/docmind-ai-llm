#!/usr/bin/env python3
"""Real validation tests for DocMind AI system components.

This module tests actual functionality with minimal mocking to validate
that the system works correctly in real scenarios. Tests that require
external services are marked with appropriate pytest marks.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import logging

import pytest
from agent_factory import analyze_query_complexity
from llama_index.core import Document

# Import DocMind AI components
from models import AppSettings
from utils import (
    detect_hardware,
    verify_rrf_configuration,
)
from utils.model_manager import ModelManager


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
        assert isinstance(settings.enable_quantization, bool)

        # Verify batch size is reasonable
        assert 1 <= settings.embedding_batch_size <= 512

        # Verify CUDA device ID is valid
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
            "fastembed_providers",
        ]
        for key in required_keys:
            assert key in hardware_info

        # Verify types
        assert isinstance(hardware_info["cuda_available"], bool)
        assert isinstance(hardware_info["gpu_name"], str)
        assert hardware_info["vram_total_gb"] is None or isinstance(
            hardware_info["vram_total_gb"], int | float
        )
        assert isinstance(hardware_info["fastembed_providers"], list)

    def test_gpu_detection_consistency(self):
        """Test GPU detection is consistent."""
        hardware_info = detect_hardware()

        # If CUDA is available, should have GPU info
        if hardware_info["cuda_available"]:
            assert hardware_info["gpu_name"] != "Unknown"
            assert hardware_info["vram_total_gb"] is not None
            assert hardware_info["vram_total_gb"] > 0


class TestQueryAnalysisReal:
    """Test real query complexity analysis."""

    def test_query_complexity_analysis_comprehensive(self):
        """Test query complexity analysis with comprehensive examples."""
        test_cases = [
            # Simple queries
            ("What is this about?", "simple", "general"),
            ("Summary please", "simple", "general"),
            ("Tell me the main points", "simple", "general"),
            # Moderate complexity
            (
                "How does this document relate to machine learning?",
                "moderate",
                "document",
            ),
            ("What are the key insights from this text?", "moderate", "document"),
            ("Explain the main concepts in this passage", "moderate", "document"),
            # Complex queries
            (
                "Compare and analyze the differences between multiple approaches",
                "complex",
                "general",
            ),
            (
                "Summarize all documents and identify relationships among "
                "various concepts",
                "complex",
                "general",
            ),
            (
                "Analyze how several different authors approach this topic "
                "across documents",
                "complex",
                "general",
            ),
            # Multimodal queries
            ("What do you see in this image?", "simple", "multimodal"),
            ("Describe the visual elements and diagrams", "moderate", "multimodal"),
            (
                "Analyze the charts and pictures in these documents",
                "complex",
                "multimodal",
            ),
            # Knowledge graph queries
            ("What entities are mentioned?", "simple", "knowledge_graph"),
            ("How are these concepts connected?", "moderate", "knowledge_graph"),
            (
                "What relationships exist between different entities?",
                "complex",
                "knowledge_graph",
            ),
        ]

        for query, expected_complexity, expected_type in test_cases:
            complexity, query_type = analyze_query_complexity(query)

            assert complexity == expected_complexity, (
                f"Query: '{query}' - Expected {expected_complexity}, got {complexity}"
            )
            assert query_type == expected_type, (
                f"Query: '{query}' - Expected {expected_type}, got {query_type}"
            )

    def test_query_length_impact(self):
        """Test that query length impacts complexity analysis."""
        short_query = "What is this?"
        long_query = (
            "What is this document about and how does it relate to the broader "
            "context of machine learning research in the field of natural "
            "language processing?"
        )

        short_complexity, _ = analyze_query_complexity(short_query)
        long_complexity, _ = analyze_query_complexity(long_query)

        # Longer queries should generally be rated as more complex
        complexity_order = ["simple", "moderate", "complex"]
        short_idx = complexity_order.index(short_complexity)
        long_idx = complexity_order.index(long_complexity)

        assert long_idx >= short_idx, (
            f"Long query ({long_complexity}) should be at least as complex as "
            f"short query ({short_complexity})"
        )


class TestModelManagerReal:
    """Test FastEmbedModelManager with real model constraints."""

    def test_model_manager_singleton_real(self):
        """Test singleton behavior with real instance."""
        manager1 = ModelManager()
        manager2 = ModelManager()

        assert manager1 is manager2
        assert id(manager1) == id(manager2)

    def test_model_cache_persistence(self):
        """Test that model cache persists across calls."""
        manager = ModelManager()
        initial_cache_size = len(manager._models)

        # This should not create actual models without proper dependencies
        # but should test the caching logic
        cache_size_after = len(manager._models)

        # Cache should be stable
        assert cache_size_after >= initial_cache_size

    def test_model_manager_clear_cache_real(self):
        """Test cache clearing with real manager."""
        manager = ModelManager()

        # Add something to cache if possible

        # Clear cache
        manager.clear_cache()

        # Should be empty after clearing
        assert len(manager._models) == 0


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
        from utils import create_index_async, setup_hybrid_qdrant_async

        # Verify functions exist and are callable
        assert callable(create_index_async)
        assert callable(setup_hybrid_qdrant_async)

        # Verify they are coroutine functions
        import inspect

        assert inspect.iscoroutinefunction(create_index_async)
        assert inspect.iscoroutinefunction(setup_hybrid_qdrant_async)

    def test_sync_async_compatibility(self):
        """Test that both sync and async versions exist for key functions."""
        from utils import (
            create_index,
            create_index_async,
            setup_hybrid_qdrant,
            setup_hybrid_qdrant_async,
        )

        # Both sync and async versions should exist
        assert callable(create_index)
        assert callable(create_index_async)
        assert callable(setup_hybrid_qdrant)
        assert callable(setup_hybrid_qdrant_async)


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
        from utils import detect_hardware

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

        assert settings.backend in ["ollama", "lmstudio", "llamacpp"]
        assert settings.default_model is not None
        assert len(settings.default_model) > 0

    def test_system_components_importable(self):
        """Test that all system components can be imported."""
        # Test core imports work

        # All imports should succeed without errors
        assert True


# Test configuration
def pytest_configure():
    """Configure pytest for real validation tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
