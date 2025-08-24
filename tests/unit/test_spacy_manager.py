"""Enhanced unit tests for spaCy manager with realistic scenarios.

Focuses on testing actual SpacyManager behavior with minimal mocking,
realistic test data, and comprehensive edge case coverage.
Tests spaCy model management logic rather than mock interactions.
"""

import threading
import time
from unittest.mock import Mock, patch

import pytest

from src.core.infrastructure.spacy_manager import SpacyManager, get_spacy_manager


class TestSpacyManager:
    """Test SpacyManager with realistic scenarios."""

    @pytest.fixture
    def manager(self):
        """Create a fresh SpacyManager instance for each test."""
        return SpacyManager()

    def test_manager_initialization(self, manager):
        """Test that SpacyManager initializes correctly."""
        assert hasattr(manager, "_models")
        assert hasattr(manager, "_lock")
        assert isinstance(manager._models, dict)
        assert len(manager._models) == 0  # Should start empty
        assert hasattr(manager._lock, "acquire")
        assert hasattr(manager._lock, "release")

    def test_ensure_model_caching_behavior(self, manager):
        """Test that model caching works correctly."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp

            # First call should load the model
            result1 = manager.ensure_model("en_core_web_sm")
            assert result1 is mock_nlp
            assert mock_load.call_count == 1

            # Second call should return cached model
            result2 = manager.ensure_model("en_core_web_sm")
            assert result2 is mock_nlp
            assert result2 is result1  # Same object
            assert mock_load.call_count == 1  # Still only called once

    def test_ensure_model_different_models(self, manager):
        """Test that different models are cached separately."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp_sm = Mock()
            mock_nlp_md = Mock()
            mock_load.side_effect = [mock_nlp_sm, mock_nlp_md]

            # Load two different models
            result_sm = manager.ensure_model("en_core_web_sm")
            result_md = manager.ensure_model("en_core_web_md")

            assert result_sm is mock_nlp_sm
            assert result_md is mock_nlp_md
            assert result_sm is not result_md
            assert mock_load.call_count == 2

            # Verify both are cached
            assert len(manager._models) == 2
            assert "en_core_web_sm" in manager._models
            assert "en_core_web_md" in manager._models

    def test_ensure_model_download_flow(self, manager):
        """Test model download when package is not installed."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=False
            ),
            patch("src.core.infrastructure.spacy_manager.download") as mock_download,
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp

            result = manager.ensure_model("en_core_web_md")

            assert result is mock_nlp
            mock_download.assert_called_once_with("en_core_web_md")
            mock_load.assert_called_once_with("en_core_web_md")

            # Should be cached after download
            assert "en_core_web_md" in manager._models

    def test_ensure_model_error_handling(self, manager):
        """Test error handling in model loading."""
        # Test download failure
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=False
            ),
            patch(
                "src.core.infrastructure.spacy_manager.download",
                side_effect=Exception("Download failed"),
            ),
            pytest.raises(Exception, match="Download failed"),
        ):
            manager.ensure_model("invalid_model")

        # Test spacy.load failure
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch(
                "src.core.infrastructure.spacy_manager.spacy.load",
                side_effect=OSError("Model not found"),
            ),
            pytest.raises(OSError, match="Model not found"),
        ):
            manager.ensure_model("broken_model")

    def test_ensure_model_default_parameter(self, manager):
        """Test ensure_model with default parameter."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp

            # Should default to "en_core_web_sm"
            result = manager.ensure_model()
            assert result is mock_nlp
            mock_load.assert_called_once_with("en_core_web_sm")

    def test_memory_optimized_processing_basic(self, manager):
        """Test memory-optimized processing context manager."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = Mock()
            mock_nlp.memory_zone.return_value.__enter__ = Mock(return_value=mock_nlp)
            mock_nlp.memory_zone.return_value.__exit__ = Mock(return_value=None)
            mock_load.return_value = mock_nlp

            with manager.memory_optimized_processing("en_core_web_sm") as nlp:
                assert nlp is mock_nlp
                mock_nlp.memory_zone.assert_called_once()

    def test_memory_optimized_processing_default_model(self, manager):
        """Test memory-optimized processing with default model."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = Mock()
            mock_nlp.memory_zone.return_value.__enter__ = Mock(return_value=mock_nlp)
            mock_nlp.memory_zone.return_value.__exit__ = Mock(return_value=None)
            mock_load.return_value = mock_nlp

            # Should default to "en_core_web_sm"
            with manager.memory_optimized_processing() as nlp:
                assert nlp is mock_nlp
                mock_load.assert_called_with("en_core_web_sm")

    def test_memory_optimized_processing_exception_handling(self, manager):
        """Test that memory zone cleanup occurs even with exceptions."""
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = Mock()
            mock_exit = Mock(return_value=None)
            mock_nlp.memory_zone.return_value.__enter__ = Mock(return_value=mock_nlp)
            mock_nlp.memory_zone.return_value.__exit__ = mock_exit
            mock_load.return_value = mock_nlp

            # Exception should still trigger cleanup
            with (
                pytest.raises(ValueError, match="Test exception"),
                manager.memory_optimized_processing("en_core_web_sm"),
            ):
                raise ValueError("Test exception")

            # Verify cleanup was called
            mock_exit.assert_called_once()

    def test_model_name_edge_cases(self, manager):
        """Test model loading with edge case model names."""
        test_cases = [
            "",  # Empty string
            "very_long_model_name_" * 10,  # Very long name
            "model-with-dashes",  # Dashes
            "model.with.dots",  # Dots
        ]

        for model_name in test_cases:
            with (
                patch(
                    "src.core.infrastructure.spacy_manager.is_package",
                    return_value=True,
                ),
                patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
            ):
                mock_nlp = Mock()
                mock_load.return_value = mock_nlp

                result = manager.ensure_model(model_name)
                assert result is mock_nlp
                mock_load.assert_called_once_with(model_name)


class TestSpacyManagerConcurrency:
    """Test SpacyManager thread safety and concurrency."""

    def test_concurrent_model_access(self):
        """Test thread safety of model loading."""
        manager = SpacyManager()
        results = []
        errors = []

        def load_model():
            try:
                with (
                    patch(
                        "src.core.infrastructure.spacy_manager.is_package",
                        return_value=True,
                    ),
                    patch(
                        "src.core.infrastructure.spacy_manager.spacy.load"
                    ) as mock_load,
                ):
                    mock_nlp = Mock()
                    mock_load.return_value = mock_nlp
                    result = manager.ensure_model("en_core_web_sm")
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_model) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 5

    @pytest.mark.performance
    def test_caching_performance(self):
        """Test that caching provides performance benefits."""
        manager = SpacyManager()

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp

            # Measure time for multiple calls (should be fast due to caching)
            start_time = time.time()
            for _ in range(100):
                result = manager.ensure_model("en_core_web_sm")
                assert result is mock_nlp
            elapsed_time = time.time() - start_time

            # Should be very fast due to caching
            assert elapsed_time < 0.1
            # spacy.load should only be called once
            assert mock_load.call_count == 1


class TestGlobalSpacyManager:
    """Test global spaCy manager functionality."""

    def test_get_spacy_manager_singleton(self):
        """Test that get_spacy_manager returns the same instance."""
        manager1 = get_spacy_manager()
        manager2 = get_spacy_manager()

        assert manager1 is manager2
        assert isinstance(manager1, SpacyManager)

    def test_global_manager_type(self):
        """Test that global manager is correct type."""
        manager = get_spacy_manager()
        assert isinstance(manager, SpacyManager)
        assert hasattr(manager, "ensure_model")
        assert hasattr(manager, "memory_optimized_processing")
        assert callable(manager.ensure_model)
        assert callable(manager.memory_optimized_processing)

    def test_global_manager_persistence(self):
        """Test that global manager maintains state across calls."""
        # Get the global manager and verify it's persistent
        manager1 = get_spacy_manager()
        manager1._test_attribute = "test_value"  # Add test attribute

        manager2 = get_spacy_manager()
        assert hasattr(manager2, "_test_attribute")
        assert manager2._test_attribute == "test_value"

        # Clean up
        delattr(manager2, "_test_attribute")

    def test_global_manager_thread_safety(self):
        """Test thread safety of global manager access."""
        managers = []
        errors = []

        def get_manager():
            try:
                manager = get_spacy_manager()
                managers.append(manager)
            except Exception as e:
                errors.append(e)

        # Test concurrent access to global manager
        threads = [threading.Thread(target=get_manager) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All should succeed and return the same instance
        assert len(errors) == 0
        assert len(managers) == 10
        assert all(manager is managers[0] for manager in managers)


class TestSpacyManagerIntegration:
    """Integration tests for SpacyManager functionality."""

    @pytest.mark.integration
    def test_complete_workflow(self):
        """Test complete SpacyManager workflow."""
        manager = SpacyManager()

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = Mock()
            mock_nlp.memory_zone.return_value.__enter__ = Mock(return_value=mock_nlp)
            mock_nlp.memory_zone.return_value.__exit__ = Mock(return_value=None)
            mock_load.return_value = mock_nlp

            # Test model loading
            nlp = manager.ensure_model("en_core_web_sm")
            assert nlp is mock_nlp

            # Test memory-optimized processing
            with manager.memory_optimized_processing(
                "en_core_web_sm"
            ) as processing_nlp:
                assert processing_nlp is mock_nlp

            # Test caching
            nlp2 = manager.ensure_model("en_core_web_sm")
            assert nlp2 is nlp
            assert mock_load.call_count == 1  # Only called once due to caching

    def test_manager_state_isolation(self):
        """Test that different manager instances are isolated."""
        manager1 = SpacyManager()
        manager2 = SpacyManager()

        # Should be different instances
        assert manager1 is not manager2
        assert manager1._models is not manager2._models

        # Should have separate caches
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch("src.core.infrastructure.spacy_manager.spacy.load") as mock_load,
        ):
            mock_nlp = Mock()
            mock_load.return_value = mock_nlp

            manager1.ensure_model("test_model")
            assert "test_model" in manager1._models
            assert "test_model" not in manager2._models
