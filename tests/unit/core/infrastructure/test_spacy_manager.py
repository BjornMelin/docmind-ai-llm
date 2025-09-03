"""Unit tests for spaCy manager infrastructure.

Covers:
- ensure_model: download-if-missing and caching
- memory_optimized_processing: memory_zone context
- get_spacy_manager: singleton behavior
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture(autouse=True)
def _mock_spacy_modules(monkeypatch):
    """Provide lightweight spaCy module shims so import succeeds.

    We patch sys.modules before importing SpacyManager to avoid real spaCy.
    """
    mock_spacy = MagicMock()
    mock_cli = MagicMock()
    mock_util = MagicMock()
    monkeypatch.setitem(sys.modules, "spacy", mock_spacy)
    monkeypatch.setitem(sys.modules, "spacy.cli", mock_cli)
    monkeypatch.setitem(sys.modules, "spacy.util", mock_util)


@pytest.mark.unit
class TestSpacyManager:
    """Tests for SpacyManager utilities."""

    def test_ensure_model_downloads_and_caches(self):
        """ensure_model downloads when missing and caches result."""
        from src.core.infrastructure.spacy_manager import SpacyManager

        manager = SpacyManager()

        mock_nlp = MagicMock()
        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=False
            ),
            patch("src.core.infrastructure.spacy_manager.download") as mock_dl,
            patch(
                "src.core.infrastructure.spacy_manager.spacy.load",
                return_value=mock_nlp,
            ),
        ):
            nlp1 = manager.ensure_model("en_core_web_sm")
            nlp2 = manager.ensure_model("en_core_web_sm")  # should hit cache

        assert nlp1 is mock_nlp
        assert nlp2 is mock_nlp
        mock_dl.assert_called_once_with("en_core_web_sm")

    def test_memory_optimized_processing_uses_memory_zone(self):
        """memory_optimized_processing yields nlp and calls memory_zone."""
        from src.core.infrastructure.spacy_manager import SpacyManager

        manager = SpacyManager()
        mock_nlp = Mock()
        mock_zone = Mock()
        mock_zone.__enter__ = Mock(return_value=None)
        mock_zone.__exit__ = Mock(return_value=None)
        mock_nlp.memory_zone.return_value = mock_zone

        with (
            patch(
                "src.core.infrastructure.spacy_manager.is_package", return_value=True
            ),
            patch(
                "src.core.infrastructure.spacy_manager.spacy.load",
                return_value=mock_nlp,
            ),
            manager.memory_optimized_processing("en_core_web_sm") as nlp,
        ):
            assert nlp is mock_nlp
            mock_nlp.memory_zone.assert_called_once()

    def test_get_spacy_manager_singleton(self):
        """get_spacy_manager returns a singleton instance."""
        from src.core.infrastructure.spacy_manager import (
            SpacyManager,
            get_spacy_manager,
        )

        a = get_spacy_manager()
        b = get_spacy_manager()
        assert isinstance(a, SpacyManager)
        assert a is b
