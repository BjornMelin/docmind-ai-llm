"""Comprehensive tests for spaCy native manager with 3.8+ optimizations.

This module provides comprehensive test coverage for spaCy management functionality,
including native spaCy API usage, error handling, memory optimization, and edge cases.
"""

import time
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
    def test_ensure_model_downloads_missing_package_success(
        self, mock_load, mock_download, mock_is_package
    ):
        """Test downloading and loading a missing model successfully."""
        mock_nlp = Mock()
        mock_load.return_value = mock_nlp

        result = self.manager.ensure_model("en_core_web_md")

        assert result is mock_nlp
        mock_is_package.assert_called_once_with("en_core_web_md")
        mock_download.assert_called_once_with("en_core_web_md")
        mock_load.assert_called_once_with("en_core_web_md")

    @patch("src.core.infrastructure.spacy_manager.is_package", return_value=False)
    @patch("src.core.infrastructure.spacy_manager.download")
    @patch("src.core.infrastructure.spacy_manager.spacy.load")
    def test_ensure_model_download_failure(
        self, mock_load, mock_download, mock_is_package
    ):
        """Test handling of download failures."""
        mock_download.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            self.manager.ensure_model("en_core_web_lg")

        mock_is_package.assert_called_once_with("en_core_web_lg")
        mock_download.assert_called_once_with("en_core_web_lg")
        mock_load.assert_not_called()

    @patch("src.core.infrastructure.spacy_manager.is_package", return_value=False)
    @patch("src.core.infrastructure.spacy_manager.download")
    @patch("src.core.infrastructure.spacy_manager.spacy.load")
    def test_ensure_model_spacy_load_fails(
        self, mock_load, mock_download, mock_is_package
    ):
        """Test when spacy.load fails after successful download."""
        mock_load.side_effect = OSError("Model not found")

        with pytest.raises(OSError, match="Model not found"):
            self.manager.ensure_model("broken_model")

        mock_download.assert_called_once_with("broken_model")
        mock_load.assert_called_once_with("broken_model")

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

    @patch("src.core.infrastructure.spacy_manager.is_package", return_value=True)
    @patch("src.core.infrastructure.spacy_manager.spacy.load")
    @patch("src.core.infrastructure.spacy_manager.download")
    def test_memory_optimized_processing_exception_handling(
        self, mock_download, mock_load, mock_is_package
    ):
        """Test memory-optimized processing with exception in context."""
        mock_nlp = MagicMock()
        mock_nlp.memory_zone.return_value.__enter__ = Mock(return_value=mock_nlp)
        mock_nlp.memory_zone.return_value.__exit__ = Mock(return_value=None)
        mock_load.return_value = mock_nlp

        # Test that exceptions are properly handled and memory_zone cleanup occurs
        # Test exception handling and cleanup (simplified version)
        with (
            pytest.raises(ValueError, match="Test exception"),
            self.manager.memory_optimized_processing("en_core_web_sm"),
        ):
            raise ValueError("Test exception")

        # Verify __exit__ was called for cleanup
        mock_nlp.memory_zone.return_value.__exit__.assert_called_once()

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

    def test_model_caching_different_models(self):
        """Test that different models are cached separately."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            patch("src.core.infrastructure.spacy_manager.download"),
        ):
            mock_nlp_sm = MagicMock()
            mock_nlp_md = MagicMock()
            mock_load.side_effect = [mock_nlp_sm, mock_nlp_md]

            # Load two different models
            result1 = self.manager.ensure_model("en_core_web_sm")
            result2 = self.manager.ensure_model("en_core_web_md")

            assert result1 is mock_nlp_sm
            assert result2 is mock_nlp_md
            assert result1 is not result2

            # Verify both models are in cache
            assert len(self.manager._models) == 2
            assert "en_core_web_sm" in self.manager._models
            assert "en_core_web_md" in self.manager._models

    def test_clear_model_cache(self):
        """Test clearing model cache manually."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            patch("src.core.infrastructure.spacy_manager.download"),
        ):
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp

            # Load model to populate cache
            self.manager.ensure_model("en_core_web_sm")
            assert len(self.manager._models) == 1

            # Clear cache
            self.manager._models.clear()
            assert len(self.manager._models) == 0

            # Load again should call spacy.load again
            self.manager.ensure_model("en_core_web_sm")
            assert mock_load.call_count == 2


class TestSpacyManagerPerformance:
    """Performance and stress tests for SpacyManager."""

    @pytest.mark.performance
    def test_model_loading_performance(self):
        """Benchmark model loading performance."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            patch("src.core.infrastructure.spacy_manager.download"),
        ):
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp
            manager = SpacyManager()

            # Measure time for 100 model retrievals (should be cached after first)
            start_time = time.time()
            for _ in range(100):
                result = manager.ensure_model("en_core_web_sm")
                assert result is mock_nlp

            elapsed_time = time.time() - start_time

            # Should be very fast due to caching (under 100ms for 100 calls)
            assert elapsed_time < 0.1
            # spacy.load should only be called once
            assert mock_load.call_count == 1

    @pytest.mark.performance
    def test_memory_zone_performance(self):
        """Benchmark memory_zone context manager performance."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            patch("src.core.infrastructure.spacy_manager.download"),
        ):
            mock_nlp = MagicMock()
            mock_nlp.memory_zone.return_value.__enter__ = Mock(return_value=mock_nlp)
            mock_nlp.memory_zone.return_value.__exit__ = Mock(return_value=None)
            mock_load.return_value = mock_nlp
            manager = SpacyManager()

            # Measure time for 50 memory zone contexts
            start_time = time.time()
            for _ in range(50):
                with manager.memory_optimized_processing("en_core_web_sm") as nlp:
                    assert nlp is mock_nlp

            elapsed_time = time.time() - start_time

            # Should complete quickly (under 500ms for 50 contexts)
            assert elapsed_time < 0.5
            # memory_zone should be called 50 times
            assert mock_nlp.memory_zone.call_count == 50

    def test_concurrent_model_access(self):
        """Test concurrent access to the same model."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            patch("src.core.infrastructure.spacy_manager.download"),
        ):
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp
            manager = SpacyManager()

            # Simulate concurrent access
            import threading

            results = []
            errors = []

            def load_model():
                try:
                    result = manager.ensure_model("en_core_web_sm")
                    results.append(result)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=load_model) for _ in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # All should succeed and return the same cached model
            assert len(errors) == 0
            assert len(results) == 10
            assert all(result is mock_nlp for result in results)
            # spacy.load should only be called once
            assert mock_load.call_count == 1


class TestSpacyManagerEdgeCases:
    """Edge case tests for SpacyManager."""

    def test_empty_model_name(self):
        """Test handling of empty model name."""
        manager = SpacyManager()

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=False
            ),
            patch("src.core.infrastructure.spacy_manager.download"),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp

            # Should handle empty string model name
            result = manager.ensure_model("")
            assert result is mock_nlp
            mock_load.assert_called_with("")

    def test_special_characters_in_model_name(self):
        """Test handling of model names with special characters."""
        manager = SpacyManager()

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=False
            ),
            patch("src.core.infrastructure.spacy_manager.download"),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp

            special_name = "en_core@web#sm$v1.0"
            result = manager.ensure_model(special_name)
            assert result is mock_nlp
            mock_load.assert_called_with(special_name)

    def test_very_long_model_name(self):
        """Test handling of very long model names."""
        manager = SpacyManager()

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=False
            ),
            patch("src.core.infrastructure.spacy_manager.download"),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp

            long_name = "a" * 1000  # Very long model name
            result = manager.ensure_model(long_name)
            assert result is mock_nlp
            mock_load.assert_called_with(long_name)


class TestGlobalSpacyManager:
    """Test global spaCy manager instance."""

    def test_get_spacy_manager_singleton(self):
        """Test that get_spacy_manager returns the same instance."""
        manager1 = get_spacy_manager()
        manager2 = get_spacy_manager()

        assert manager1 is manager2
        assert isinstance(manager1, SpacyManager)

    def test_global_manager_caching(self):
        """Test that global manager maintains its cache across calls."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            patch("src.core.infrastructure.spacy_manager.download"),
        ):
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp

            # Load model using global manager
            manager1 = get_spacy_manager()
            result1 = manager1.ensure_model("en_core_web_sm")

            # Get global manager again and check cache
            manager2 = get_spacy_manager()
            result2 = manager2.ensure_model("en_core_web_sm")

            assert manager1 is manager2
            assert result1 is result2 is mock_nlp
            # spacy.load should only be called once due to caching
            assert mock_load.call_count == 1

    def test_global_manager_thread_safety(self):
        """Test thread safety of global manager access."""
        import threading

        managers = []
        errors = []

        def get_manager():
            try:
                manager = get_spacy_manager()
                managers.append(manager)
            except Exception as e:
                errors.append(e)

        # Test concurrent access to global manager
        threads = [threading.Thread(target=get_manager) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All should succeed and return the same instance
        assert len(errors) == 0
        assert len(managers) == 20
        assert all(manager is managers[0] for manager in managers)


class TestSpacyManagerIntegration:
    """Integration tests for SpacyManager (basic functionality only)."""

    @pytest.mark.integration
    def test_integration_basic_flow(self):
        """Test basic integration flow with mocked spaCy."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            patch("src.core.infrastructure.spacy_manager.download"),
        ):
            mock_nlp = MagicMock()
            mock_nlp.memory_zone.return_value.__enter__ = Mock(return_value=mock_nlp)
            mock_nlp.memory_zone.return_value.__exit__ = Mock(return_value=None)
            mock_load.return_value = mock_nlp

            manager = SpacyManager()

            # Test complete workflow
            nlp = manager.ensure_model("en_core_web_sm")
            assert nlp is mock_nlp

            # Test memory optimization
            with manager.memory_optimized_processing(
                "en_core_web_sm"
            ) as processing_nlp:
                assert processing_nlp is mock_nlp

            # Test caching works
            nlp2 = manager.ensure_model("en_core_web_sm")
            assert nlp2 is nlp
            assert mock_load.call_count == 1
