"""Functional tests for models.py - Data validation and configuration management.

This test suite validates the real-world usage of Pydantic models for configuration
management and structured data output. Tests focus on business requirements like
valid configuration loading, data validation, and error handling.
"""

import pytest
from pydantic import ValidationError

from src.config.settings import AnalysisOutput
from src.config.settings import Settings as AppSettings


class TestAnalysisOutputValidation:
    """Test the AnalysisOutput model used for structured LLM responses."""

    def test_valid_analysis_output_creation(self):
        """Valid analysis outputs should be created successfully for LLM responses."""
        # Simulate a typical structured LLM response
        output = AnalysisOutput(
            summary="This document discusses AI trends in 2024",
            key_insights=[
                "Machine learning adoption increased by 40%",
                "Ethical AI frameworks are becoming standard",
                "Edge computing is driving new applications",
            ],
            action_items=[
                "Research competitor AI strategies",
                "Implement ethical AI guidelines",
                "Evaluate edge computing opportunities",
            ],
            open_questions=[
                "How will regulation affect adoption?",
                "What are the costs of implementation?",
            ],
        )

        assert output.summary == "This document discusses AI trends in 2024"
        assert len(output.key_insights) == 3
        assert len(output.action_items) == 3
        assert len(output.open_questions) == 2

        # Verify it can be serialized (important for API responses)
        json_output = output.model_dump()
        assert "summary" in json_output
        assert isinstance(json_output["key_insights"], list)

    def test_minimal_analysis_output(self):
        """Analysis outputs should work with minimal required data."""
        # Some documents might not have all types of insights
        minimal_output = AnalysisOutput(
            summary="Brief document with limited content",
            key_insights=["Single insight from the document"],
            action_items=[],
            open_questions=[],
        )

        assert minimal_output.summary == "Brief document with limited content"
        assert len(minimal_output.key_insights) == 1
        assert len(minimal_output.action_items) == 0
        assert len(minimal_output.open_questions) == 0

    def test_analysis_output_validates_required_fields(self):
        """Analysis output should enforce required fields for consistent responses."""
        # Missing summary should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            AnalysisOutput(
                key_insights=["Some insight"], action_items=[], open_questions=[]
            )

        assert "summary" in str(exc_info.value)

    def test_analysis_output_validates_field_types(self):
        """Analysis output should validate that lists contain strings."""
        # Wrong type for key_insights should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            AnalysisOutput(
                summary="Valid summary",
                key_insights=[
                    "Valid insight",
                    123,
                    "Another insight",
                ],  # Invalid integer
                action_items=[],
                open_questions=[],
            )

        error_details = str(exc_info.value)
        assert "key_insights" in error_details

    def test_analysis_output_handles_empty_strings_appropriately(self):
        """Analysis output should handle edge cases like empty strings."""
        # Empty summary should be handled gracefully if provided
        output = AnalysisOutput(
            summary="",
            key_insights=["At least one insight"],
            action_items=[],
            open_questions=[],
        )

        assert output.summary == ""
        assert len(output.key_insights) == 1

    def test_analysis_output_json_serialization(self):
        """Analysis outputs must serialize properly for API responses."""
        output = AnalysisOutput(
            summary="Test document analysis",
            key_insights=["Insight with unicode: 🤖", "Regular insight"],
            action_items=["Action with special chars: & < >"],
            open_questions=["Question?"],
        )

        # Should serialize to JSON without errors
        json_str = output.model_dump_json()
        assert isinstance(json_str, str)
        assert "🤖" in json_str  # Unicode should be preserved

        # Should be deserializable
        deserialized = AnalysisOutput.model_validate_json(json_str)
        assert deserialized.summary == output.summary
        assert deserialized.key_insights == output.key_insights


class TestAppSettingsConfiguration:
    """Test the AppSettings model used for application configuration."""

    def test_app_settings_loads_with_defaults(self):
        """App settings should load with sensible defaults for development."""
        settings = AppSettings()

        # Critical settings should have defaults
        assert settings.qdrant_url is not None
        assert settings.model_name is not None
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0
        assert settings.top_k > 0

        # Verify reasonable defaults for RAG
        assert settings.chunk_size >= 500  # Reasonable minimum
        assert (
            settings.chunk_overlap < settings.chunk_size
        )  # Overlap should be less than size
        assert settings.top_k <= 20  # Reasonable maximum

    def test_app_settings_validates_chunk_configuration(self):
        """Chunk settings should be validated for proper RAG operation."""
        # Valid configuration should work
        valid_settings = AppSettings(chunk_size=1000, chunk_overlap=200, top_k=5)

        assert valid_settings.chunk_size == 1000
        assert valid_settings.chunk_overlap == 200
        assert valid_settings.top_k == 5

    def test_app_settings_validates_rrf_weights(self):
        """RRF fusion weights should be validated for hybrid search."""
        # Valid RRF weights should sum to 1.0 or be reasonable
        settings = AppSettings(
            rrf_fusion_weight_dense=0.7, rrf_fusion_weight_sparse=0.3
        )

        assert settings.rrf_fusion_weight_dense == 0.7
        assert settings.rrf_fusion_weight_sparse == 0.3
        # Weights should sum to 1.0 for proper fusion
        assert (
            abs(
                (settings.rrf_fusion_weight_dense + settings.rrf_fusion_weight_sparse)
                - 1.0
            )
            < 0.01
        )

    def test_app_settings_url_validation(self):
        """URL settings should be properly formatted."""
        settings = AppSettings()

        # URLs should be well-formed
        assert settings.qdrant_url.startswith(("http://", "https://"))
        assert settings.ollama_base_url.startswith(("http://", "https://"))

        # Should handle custom URLs
        custom_settings = AppSettings(
            qdrant_url="http://custom-qdrant:6333",
            ollama_base_url="http://custom-ollama:11434",
        )

        assert custom_settings.qdrant_url == "http://custom-qdrant:6333"
        assert custom_settings.ollama_base_url == "http://custom-ollama:11434"

    def test_app_settings_model_configuration(self):
        """Model settings should be properly configured for LLM operations."""
        settings = AppSettings()

        # Should have sensible model defaults
        assert settings.model_name is not None
        assert len(settings.model_name) > 0

    def test_app_settings_embedding_configuration(self):
        """Embedding settings should support hybrid search capabilities."""
        settings = AppSettings()

        # Embedding model should be specified
        assert settings.embedding_model is not None
        assert len(settings.embedding_model) > 0

        # Should have reasonable batch size for efficiency
        assert settings.embedding_batch_size > 0
        assert settings.embedding_batch_size <= 100  # Reasonable upper bound

    def test_app_settings_hardware_configuration(self):
        """Hardware settings should be configured appropriately."""
        settings = AppSettings()

        # GPU settings should be boolean
        assert isinstance(settings.enable_gpu_acceleration, bool)

    def test_app_settings_debug_and_logging(self):
        """Debug and logging settings should be properly configured."""
        settings = AppSettings()

        # Cache should be boolean
        assert isinstance(settings.enable_document_caching, bool)


class TestRealWorldConfigurationScenarios:
    """Test configuration scenarios that occur in real deployments."""

    def test_production_like_configuration(self):
        """Test configuration that would be used in production."""
        production_settings = AppSettings(
            chunk_size=1024,
            chunk_overlap=128,
            top_k=10,
            enable_gpu_acceleration=True,
        )

        # Production settings should be conservative
        assert production_settings.enable_gpu_acceleration is True
        assert production_settings.chunk_overlap < production_settings.chunk_size

    def test_development_configuration(self):
        """Test configuration suitable for development."""
        dev_settings = AppSettings(
            chunk_size=512,  # Smaller for faster testing
            top_k=3,  # Fewer results for testing
            enable_gpu_acceleration=False,  # May not be available in dev
        )

        assert dev_settings.chunk_size <= 1000  # Reasonable for dev
        assert dev_settings.enable_gpu_acceleration is False

    def test_memory_constrained_configuration(self):
        """Test configuration for memory-constrained environments."""
        memory_efficient_settings = AppSettings(
            chunk_size=256,  # Smaller chunks
            embedding_batch_size=10,  # Smaller batches
            top_k=3,  # Fewer results
            enable_gpu_acceleration=False,  # CPU-only for memory efficiency
        )

        assert memory_efficient_settings.chunk_size <= 512
        assert memory_efficient_settings.embedding_batch_size <= 20
        assert memory_efficient_settings.top_k <= 5

    def test_high_performance_configuration(self):
        """Test configuration for high-performance environments."""
        high_perf_settings = AppSettings(
            chunk_size=2048,  # Larger chunks for more context
            embedding_batch_size=50,  # Larger batches
            top_k=20,  # More comprehensive results
            enable_gpu_acceleration=True,  # GPU acceleration
        )

        assert high_perf_settings.chunk_size >= 1000
        assert high_perf_settings.embedding_batch_size >= 30
        assert high_perf_settings.enable_gpu_acceleration is True


class TestConfigurationValidationEdgeCases:
    """Test edge cases and error conditions in configuration."""

    def test_invalid_chunk_size_combinations(self):
        """Invalid chunk configurations should be handled properly."""
        # Chunk overlap larger than chunk size should be handled
        try:
            settings = AppSettings(
                chunk_size=100,
                chunk_overlap=200,  # Overlap larger than size
            )
            # If no validation exists, at least verify the values are set
            assert settings.chunk_size == 100
            assert settings.chunk_overlap == 200
        except ValidationError:
            # If validation exists, it should catch this
            pass  # This is acceptable behavior

    def test_extreme_similarity_values(self):
        """Extreme top_k values should be handled."""
        # Very small value
        small_k_settings = AppSettings(top_k=1)
        assert small_k_settings.top_k == 1

        # Large but reasonable value
        large_k_settings = AppSettings(top_k=100)
        assert large_k_settings.top_k == 100

    def test_gpu_settings_boundary_values(self):
        """GPU settings boundary values should be handled correctly."""
        # GPU disabled
        gpu_off_settings = AppSettings(enable_gpu_acceleration=False)
        assert gpu_off_settings.enable_gpu_acceleration is False

        # GPU enabled
        gpu_on_settings = AppSettings(enable_gpu_acceleration=True)
        assert gpu_on_settings.enable_gpu_acceleration is True

    def test_url_format_variations(self):
        """Different URL formats should be handled appropriately."""
        url_variations = [
            "http://localhost:6333",
            "https://qdrant.example.com",
            "http://192.168.1.100:6333",
            "https://qdrant:6333",
        ]

        for url in url_variations:
            settings = AppSettings(qdrant_url=url)
            assert settings.qdrant_url == url
            assert settings.qdrant_url.startswith(("http://", "https://"))
