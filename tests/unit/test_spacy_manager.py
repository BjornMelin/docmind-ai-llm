"""Tests for spaCy native manager with 3.8+ optimizations."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.core.infrastructure.spacy_manager import SpacyManager, get_spacy_manager


class TestSpacyManager:
    """Test SpacyManager with native spaCy 3.8+ APIs."""

    def setup_method(self):
        """Set up test instance."""
        self.manager = SpacyManager()

    @patch("src.core.infrastructure.spacy_manager.is_package", return_value=True)
    @patch("src.core.infrastructure.spacy_manager.spacy.load")
    @patch("src.core.infrastructure.spacy_manager.download")
    def test_ensure_model_existing_package(
        self, mock_download, mock_load, mock_is_package
    ):
        """Test loading an already installed model."""
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp

        result = self.manager.ensure_model("en_core_web_sm")

        assert result is mock_nlp
        mock_is_package.assert_called_once_with("en_core_web_sm")
        mock_download.assert_not_called()  # Should NOT download when package exists
        mock_load.assert_called_once_with("en_core_web_sm")

        # Test caching - second call should not reload
        result2 = self.manager.ensure_model("en_core_web_sm")
        assert result2 is mock_nlp
        assert mock_load.call_count == 1  # Only called once due to caching

    @patch("src.core.infrastructure.spacy_manager.is_package", return_value=False)
    @patch("src.core.infrastructure.spacy_manager.download")
    @patch("src.core.infrastructure.spacy_manager.spacy.load")
    def test_ensure_model_downloads_missing_package(
        self, mock_load, mock_download, mock_is_package
    ):
        """Test downloading and loading a missing model."""
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp

        result = self.manager.ensure_model("en_core_web_md")

        assert result is mock_nlp
        mock_is_package.assert_called_once_with("en_core_web_md")
        mock_download.assert_called_once_with("en_core_web_md")
        mock_load.assert_called_once_with("en_core_web_md")

    @patch("src.core.infrastructure.spacy_manager.is_package", return_value=True)
    @patch("src.core.infrastructure.spacy_manager.spacy.load")
    @patch("src.core.infrastructure.spacy_manager.download")
    def test_memory_optimized_processing(
        self, mock_download, mock_load, mock_is_package
    ):
        """Test memory-optimized processing using memory_zone()."""
        # Create mock nlp with memory_zone context manager
        mock_nlp = MagicMock()
        mock_memory_zone = MagicMock()
        mock_nlp.memory_zone.return_value.__enter__ = Mock(
            return_value=mock_memory_zone
        )
        mock_nlp.memory_zone.return_value.__exit__ = Mock(return_value=None)
        mock_load.return_value = mock_nlp

        with self.manager.memory_optimized_processing("en_core_web_sm") as nlp:
            assert nlp is mock_nlp
            # Verify memory_zone was called
            mock_nlp.memory_zone.assert_called_once()

    def test_default_model_name(self):
        """Test that default model name is en_core_web_sm."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            patch("src.core.infrastructure.spacy_manager.download"),
        ):
            mock_nlp = MagicMock()
            mock_nlp.memory_zone.return_value.__enter__ = Mock(return_value=None)
            mock_nlp.memory_zone.return_value.__exit__ = Mock(return_value=None)
            mock_load.return_value = mock_nlp

            # Test ensure_model with default
            self.manager.ensure_model()
            mock_load.assert_called_with("en_core_web_sm")

            # Test memory_optimized_processing with default
            with self.manager.memory_optimized_processing():
                pass  # Just test that it works with default


class TestGlobalSpacyManager:
    """Test global spaCy manager instance."""

    def test_get_spacy_manager_singleton(self):
        """Test that get_spacy_manager returns the same instance."""
        manager1 = get_spacy_manager()
        manager2 = get_spacy_manager()

        assert manager1 is manager2
        assert isinstance(manager1, SpacyManager)


class TestSpacyManagerIntegration:
    """Integration tests for SpacyManager (requires spaCy installation)."""

    @pytest.mark.integration
    def test_actual_model_loading(self):
        """Test loading actual spaCy model (integration test)."""
        manager = SpacyManager()

        # This test requires en_core_web_sm to be installed
        try:
            nlp = manager.ensure_model("en_core_web_sm")
            assert nlp is not None
            assert hasattr(nlp, "memory_zone")

            # Test memory zone functionality
            with manager.memory_optimized_processing(
                "en_core_web_sm"
            ) as processing_nlp:
                doc = processing_nlp("Test sentence for memory optimization.")
                assert len(doc) > 0

        except OSError:
            pytest.skip("en_core_web_sm not installed - skipping integration test")

    @pytest.mark.integration
    def test_memory_zone_performance(self):
        """Test that memory_zone provides memory management benefits."""
        manager = SpacyManager()

        try:
            test_texts = ["This is a test sentence." for _ in range(100)]

            # Process with memory optimization
            with manager.memory_optimized_processing("en_core_web_sm") as nlp:
                docs = list(nlp.pipe(test_texts))
                assert len(docs) == 100

        except OSError:
            pytest.skip("en_core_web_sm not installed - skipping performance test")
