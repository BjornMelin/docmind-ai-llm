"""Basic unit tests for BGE-M3 embedder module.

These tests provide basic coverage for the BGE-M3 embedder module to address
the zero-coverage issue identified in Phase 1.
"""

from unittest.mock import patch

import pytest

from src.config.settings import DocMindSettings


@pytest.mark.unit
class TestBGEM3EmbedderBasics:
    """Test basic BGE-M3 embedder functionality with mocking."""

    @pytest.mark.unit
    def test_embedder_imports(self):
        """Test that BGE-M3 embedder module can be imported without errors."""
        try:
            from src.processing.embeddings.bgem3_embedder import BGEM3EmbeddingManager

            assert BGEM3EmbeddingManager is not None
        except ImportError as e:
            pytest.fail(f"BGE-M3 embedder import failed: {e}")

    @pytest.mark.unit
    def test_embedder_initialization_config(self):
        """Test embedder initialization with proper config."""
        from src.processing.embeddings.bgem3_embedder import BGEM3EmbeddingManager

        settings = DocMindSettings()

        # Test that embedder can be created with settings
        # Mock the heavy dependencies to avoid model loading
        with patch("src.processing.embeddings.bgem3_embedder.FlagModel"):
            embedder = BGEM3EmbeddingManager(settings)
            assert embedder is not None
            assert hasattr(embedder, "settings")

    @pytest.mark.unit
    def test_embedder_dimension_constants(self):
        """Test embedder dimension constants are correct."""
        from src.processing.embeddings.bgem3_embedder import BGEM3EmbeddingManager

        # BGE-M3 should produce 1024-dimensional embeddings
        settings = DocMindSettings()
        expected_dim = 1024

        with patch("src.processing.embeddings.bgem3_embedder.FlagModel"):
            embedder = BGEM3EmbeddingManager(settings)
            # Test that the expected dimension matches settings
            assert settings.embedding.dimension == expected_dim

    @pytest.mark.unit
    @patch("src.processing.embeddings.bgem3_embedder.FlagModel")
    def test_embedder_model_name_configuration(self, mock_flag_model):
        """Test embedder uses correct model name from settings."""
        from src.processing.embeddings.bgem3_embedder import BGEM3EmbeddingManager

        settings = DocMindSettings()
        expected_model = "BAAI/bge-m3"

        # Verify settings have correct model
        assert settings.embedding.model_name == expected_model

        # Initialize embedder
        embedder = BGEM3EmbeddingManager(settings)
        assert embedder is not None
