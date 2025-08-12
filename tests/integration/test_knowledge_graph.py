"""Tests for Knowledge Graph creation, spaCy integration, and agent tools.

This module tests:
- SpaCy model management (loading and auto-download)
- Graceful fallback when dependencies missing

Note: Many tests for create_index functions were removed as those functions
were deleted during utils/ cleanup. Only spaCy integration tests remain.

Following PyTestQA-Agent standards for comprehensive testing.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import subprocess
from unittest.mock import MagicMock, patch

import pytest


class TestSpaCyIntegration:
    """Test spaCy model management and integration."""

    def test_ensure_spacy_model_already_installed(self):
        """Test when spaCy model is already available."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp

            from src.utils.document import ensure_spacy_model

            nlp = ensure_spacy_model()

            assert nlp is not None
            assert nlp == mock_nlp
            mock_load.assert_called_once_with("en_core_web_sm")

    def test_ensure_spacy_model_auto_download_success(self):
        """Test automatic model download when missing."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            # First call fails (not installed), second succeeds
            mock_load.side_effect = [OSError("Model not found"), mock_nlp]

            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)

                from src.utils.document import ensure_spacy_model

                nlp = ensure_spacy_model()

                assert nlp is not None
                assert nlp == mock_nlp
                # Should try to load twice (before and after download)
                assert mock_load.call_count == 2
                # Should call subprocess to download
                mock_subprocess.assert_called_once()

    def test_ensure_spacy_model_auto_download_failure(self):
        """Test fallback when auto-download fails."""
        with patch("spacy.load") as mock_load:
            # Always fail to load model
            mock_load.side_effect = OSError("Model not found")

            with patch("subprocess.run") as mock_subprocess:
                # Download also fails
                mock_subprocess.return_value = MagicMock(returncode=1)

                from src.utils.document import ensure_spacy_model

                nlp = ensure_spacy_model()

                # Should return None when both load and download fail
                assert nlp is None
                # Should try to load twice (before and after failed download)
                assert mock_load.call_count == 2
                mock_subprocess.assert_called_once()

    def test_ensure_spacy_model_no_subprocess_available(self):
        """Test fallback when subprocess is not available."""
        with patch("spacy.load") as mock_load:
            mock_load.side_effect = OSError("Model not found")

            with patch("subprocess.run", side_effect=FileNotFoundError):
                from src.utils.document import ensure_spacy_model

                nlp = ensure_spacy_model()

                # Should return None when subprocess not available
                assert nlp is None

    def test_ensure_spacy_model_subprocess_timeout(self):
        """Test fallback when subprocess times out."""
        with patch("spacy.load") as mock_load:
            mock_load.side_effect = OSError("Model not found")

            with patch(
                "subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)
            ):
                from src.utils.document import ensure_spacy_model

                nlp = ensure_spacy_model()

                # Should return None when download times out
                assert nlp is None

    def test_ensure_spacy_model_generic_subprocess_error(self):
        """Test fallback when subprocess has generic error."""
        with patch("spacy.load") as mock_load:
            mock_load.side_effect = OSError("Model not found")

            with patch("subprocess.run", side_effect=Exception("Generic error")):
                from src.utils.document import ensure_spacy_model

                nlp = ensure_spacy_model()

                # Should return None on generic subprocess error
                assert nlp is None

    def test_ensure_spacy_model_import_error_handling(self):
        """Test graceful handling when spaCy itself is not installed."""
        with patch("spacy.load", side_effect=ImportError("spaCy not installed")):
            from src.utils.document import ensure_spacy_model

            nlp = ensure_spacy_model()

            # Should return None when spaCy not installed
            assert nlp is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
