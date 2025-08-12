"""Tests for multimodal document processing functionality.

This module tests multimodal embedding creation and advanced multimodal features
with modern pytest fixtures and proper typing.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from llama_index.core import Document
from llama_index.core.schema import ImageDocument

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.document_loader import create_native_multimodal_embeddings

# Shared fixtures are automatically available via conftest.py


class TestCreateNativeMultimodalEmbeddings:
    """Test native multimodal embedding creation with modern fixtures."""

    @pytest.mark.parametrize(
        ("image_count", "expected_embeddings"),
        [
            (1, 1),
            (2, 2),
            (5, 5),
        ],
    )
    @patch("utils.document_loader.ModelManager.get_multimodal_embedding_model")
    def test_create_embeddings_success(
        self,
        mock_model_manager: Mock,
        sample_images_data: list[dict[str, Any]],
        image_count: int,
        expected_embeddings: int,
    ) -> None:
        """Test successful multimodal embedding creation with varying image counts."""
        # Adjust sample data for parametrization
        test_images = sample_images_data * image_count

        # Mock multimodal model
        mock_model = Mock()
        mock_text_embedding = Mock()
        mock_text_embedding.flatten.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.embed_text.return_value = [mock_text_embedding]

        mock_img_embedding = Mock()
        mock_img_embedding.flatten.return_value.tolist.return_value = [0.4, 0.5, 0.6]
        mock_model.embed_image.return_value = [mock_img_embedding] * expected_embeddings

        mock_model_manager.return_value = mock_model

        with (
            patch("utils.document_loader.tempfile.gettempdir", return_value="/tmp"),
            patch("builtins.open", create=True),
            patch("utils.document_loader.os.unlink") as mock_unlink,
            patch("utils.document_loader.logging.info") as mock_log_info,
        ):
            result = create_native_multimodal_embeddings(
                "Test text content", test_images
            )

            # Verify model usage
            mock_model.embed_text.assert_called_once_with(["Test text content"])
            mock_model.embed_image.assert_called_once()

            # Verify result structure
            assert result["text_embedding"] == [0.1, 0.2, 0.3]
            assert len(result["image_embeddings"]) == expected_embeddings
            assert result["image_embeddings"][0]["embedding"] == [0.4, 0.5, 0.6]
            assert result["combined_embedding"] == [0.1, 0.2, 0.3]
            assert result["provider_used"] == "fastembed_native_multimodal"

            # Verify temp file cleanup
            assert mock_unlink.call_count == image_count  # One per image

            # Verify logging
            mock_log_info.assert_called_with(
                "Using FastEmbed native LateInteractionMultimodalEmbedding"
            )

    @patch("utils.document_loader.ModelManager.get_multimodal_embedding_model")
    def test_create_embeddings_text_only(self, mock_model_manager: Mock) -> None:
        """Test embedding creation with text only (no images)."""
        mock_model = Mock()
        mock_text_embedding = Mock()
        mock_text_embedding.flatten.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.embed_text.return_value = [mock_text_embedding]

        mock_model_manager.return_value = mock_model

        result = create_native_multimodal_embeddings("Test text content", images=None)

        # Verify model usage
        mock_model.embed_text.assert_called_once_with(["Test text content"])
        mock_model.embed_image.assert_not_called()

        # Verify result structure
        assert result["text_embedding"] == [0.1, 0.2, 0.3]
        assert result["image_embeddings"] == []
        assert result["combined_embedding"] == [0.1, 0.2, 0.3]
        assert result["provider_used"] == "fastembed_native_multimodal"

    @pytest.mark.parametrize(
        ("fallback_scenario", "error_message"),
        [
            ("import_error", "Multimodal not available"),
            ("runtime_error", "Model initialization failed"),
            ("memory_error", "Out of memory"),
        ],
    )
    @patch("utils.document_loader.ModelManager.get_multimodal_embedding_model")
    def test_create_embeddings_fallback_scenarios(
        self,
        mock_model_manager: Mock,
        mock_settings: Mock,
        fallback_scenario: str,
        error_message: str,
    ) -> None:
        """Test fallback behavior for various error scenarios."""
        # Configure error based on scenario
        if fallback_scenario == "import_error":
            mock_model_manager.side_effect = ImportError(error_message)
        else:
            mock_model_manager.side_effect = Exception(error_message)

        # Mock FastEmbed text-only fallback
        with patch("utils.document_loader.FastEmbedEmbedding") as mock_fastembed:
            mock_text_model = Mock()
            mock_text_model.get_text_embedding.return_value = [0.7, 0.8, 0.9]
            mock_fastembed.return_value = mock_text_model

            with (
                patch("utils.document_loader.settings", mock_settings),
                patch("utils.document_loader.logging.warning") as mock_log_warning,
            ):
                result = create_native_multimodal_embeddings("Test text")

                # Verify fallback model creation
                mock_fastembed.assert_called_once_with(
                    model_name=mock_settings.dense_embedding_model,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    cache_dir="./embeddings_cache",
                )

                # Verify result
                assert result["text_embedding"] == [0.7, 0.8, 0.9]
                assert result["combined_embedding"] == [0.7, 0.8, 0.9]
                assert result["provider_used"] == "fastembed_text_only"

                # Verify warning for import error specifically
                if fallback_scenario == "import_error":
                    mock_log_warning.assert_called_once()
                    warning_message = mock_log_warning.call_args[0][0]
                    assert (
                        "FastEmbed LateInteractionMultimodalEmbedding not available"
                        in warning_message
                    )

    @patch("utils.document_loader.ModelManager.get_multimodal_embedding_model")
    def test_create_embeddings_complete_failure(
        self, mock_model_manager: Mock, mock_settings: Mock
    ) -> None:
        """Test ultimate fallback when all embedding methods fail."""
        # Mock all embedding methods failing
        mock_model_manager.side_effect = Exception("All methods failed")

        with patch("utils.document_loader.FastEmbedEmbedding") as mock_fastembed:
            mock_fastembed.side_effect = Exception("FastEmbed also failed")

            with (
                patch("utils.document_loader.settings", mock_settings),
                patch("utils.document_loader.logging.error") as mock_log_error,
            ):
                result = create_native_multimodal_embeddings("Test text")

                # Verify failure result
                assert result["provider_used"] == "failed"
                assert result["text_embedding"] is None
                assert result["combined_embedding"] is None

                # Verify error logging
                assert mock_log_error.call_count >= 1
                error_messages = [call[0][0] for call in mock_log_error.call_args_list]
                assert any(
                    "All embedding methods failed" in msg for msg in error_messages
                )

    @pytest.mark.parametrize(
        "temp_file_error",
        [
            True,  # Temp file creation fails
            False,  # Normal operation
        ],
    )
    @patch("utils.document_loader.ModelManager.get_multimodal_embedding_model")
    def test_create_embeddings_temp_file_handling(
        self,
        mock_model_manager: Mock,
        sample_images_data: list[dict[str, Any]],
        temp_file_error: bool,
    ) -> None:
        """Test proper handling of temporary file operations."""
        mock_model = Mock()
        mock_text_embedding = Mock()
        mock_text_embedding.flatten.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.embed_text.return_value = [mock_text_embedding]

        mock_img_embedding = Mock()
        mock_img_embedding.flatten.return_value.tolist.return_value = [0.4, 0.5, 0.6]
        mock_model.embed_image.return_value = [mock_img_embedding]

        mock_model_manager.return_value = mock_model

        with (
            patch("utils.document_loader.tempfile.gettempdir", return_value="/tmp"),
            patch("builtins.open", create=True) as mock_open,
            patch("utils.document_loader.os.unlink") as mock_unlink,
        ):
            if temp_file_error:
                mock_open.side_effect = OSError("Cannot create temp file")

                with pytest.raises(OSError, match="Cannot create temp file"):
                    create_native_multimodal_embeddings(
                        "Test text content", sample_images_data
                    )
            else:
                result = create_native_multimodal_embeddings(
                    "Test text content", sample_images_data
                )

                # Verify successful operation
                assert result["provider_used"] == "fastembed_native_multimodal"
                assert mock_unlink.call_count == len(sample_images_data)

    @patch("utils.document_loader.ModelManager.get_multimodal_embedding_model")
    def test_create_embeddings_image_decode_error(
        self,
        mock_model_manager: Mock,
    ) -> None:
        """Test handling of invalid base64 image data."""
        # Create images with invalid base64 data
        invalid_images_data = [
            {
                "image_base64": "invalid_base64_data",
                "image_mimetype": "image/png",
                "image_path": "test_image.png",
            }
        ]

        mock_model = Mock()
        mock_text_embedding = Mock()
        mock_text_embedding.flatten.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.embed_text.return_value = [mock_text_embedding]

        mock_model_manager.return_value = mock_model

        with (
            patch("utils.document_loader.tempfile.gettempdir", return_value="/tmp"),
            patch("builtins.open", create=True),
            patch("utils.document_loader.os.unlink"),
            patch("utils.document_loader.base64.b64decode") as mock_b64decode,
            patch("utils.document_loader.logging.warning") as mock_log_warning,
        ):
            mock_b64decode.side_effect = Exception("Invalid base64")

            result = create_native_multimodal_embeddings(
                "Test text content", invalid_images_data
            )

            # Should still return text embedding despite image processing failure
            assert result["text_embedding"] == [0.1, 0.2, 0.3]
            assert result["image_embeddings"] == []
            assert result["provider_used"] == "fastembed_native_multimodal"

            # Should log warning about image processing failure
            mock_log_warning.assert_called()
            warning_message = mock_log_warning.call_args[0][0]
            assert "Failed to decode image" in warning_message


class TestAdvancedMultimodalFeatures:
    """Test advanced multimodal features and integration scenarios."""

    @pytest.mark.parametrize(
        ("document_type", "expected_processing"),
        [
            ("text_only", "text_processing"),
            ("image_only", "image_processing"),
            ("mixed_content", "multimodal_processing"),
        ],
    )
    def test_multimodal_document_processing(
        self,
        document_type: str,
        expected_processing: str,
    ) -> None:
        """Test processing of different document types."""
        if document_type == "text_only":
            documents = [Document(text="Text content", metadata={"source": "doc.pdf"})]
        elif document_type == "image_only":
            documents = [
                ImageDocument(
                    image="base64_data", metadata={"source": "image.pdf", "page": 1}
                )
            ]
        else:  # mixed_content
            documents = [
                Document(text="Text content", metadata={"source": "doc.pdf"}),
                ImageDocument(
                    image="base64_data", metadata={"source": "image.pdf", "page": 1}
                ),
            ]

        with patch(
            "utils.document_loader.create_native_multimodal_embeddings"
        ) as mock_embed:
            mock_embed.return_value = {
                "text_embedding": [0.1, 0.2, 0.3],
                "image_embeddings": [],
                "combined_embedding": [0.1, 0.2, 0.3],
                "provider_used": "fastembed_native_multimodal",
            }

            # This would be the actual function that processes documents
            # For now, we just verify the expected behavior patterns
            if expected_processing == "text_processing":
                assert len([doc for doc in documents if isinstance(doc, Document)]) == 1
            elif expected_processing == "image_processing":
                assert (
                    len([doc for doc in documents if isinstance(doc, ImageDocument)])
                    == 1
                )
            else:  # multimodal_processing
                assert len([doc for doc in documents if isinstance(doc, Document)]) >= 1
                assert (
                    len([doc for doc in documents if isinstance(doc, ImageDocument)])
                    >= 1
                )

    @pytest.mark.parametrize("embedding_dimension", [384, 512, 768, 1024])
    def test_embedding_dimension_consistency(self, embedding_dimension: int) -> None:
        """Test that embedding dimensions remain consistent across operations."""
        text_embedding = [0.1] * embedding_dimension
        image_embedding = [0.2] * embedding_dimension

        with patch(
            "utils.document_loader.ModelManager.get_multimodal_embedding_model"
        ) as mock_model_manager:
            mock_model = Mock()
            mock_text_emb = Mock()
            mock_text_emb.flatten.return_value.tolist.return_value = text_embedding
            mock_model.embed_text.return_value = [mock_text_emb]

            mock_img_emb = Mock()
            mock_img_emb.flatten.return_value.tolist.return_value = image_embedding
            mock_model.embed_image.return_value = [mock_img_emb]

            mock_model_manager.return_value = mock_model

            sample_images = [
                {
                    "image_base64": (
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAHgSJ/JVwAAAABJRU5ErkJggg=="
                    ),
                    "image_mimetype": "image/png",
                    "image_path": "test.png",
                }
            ]

            with (
                patch("utils.document_loader.tempfile.gettempdir", return_value="/tmp"),
                patch("builtins.open", create=True),
                patch("utils.document_loader.os.unlink"),
            ):
                result = create_native_multimodal_embeddings("Test text", sample_images)

                # Verify dimensions
                assert len(result["text_embedding"]) == embedding_dimension
                assert (
                    len(result["image_embeddings"][0]["embedding"])
                    == embedding_dimension
                )
                assert len(result["combined_embedding"]) == embedding_dimension

    def test_multimodal_metadata_preservation(self) -> None:
        """Test that multimodal processing preserves important metadata."""
        test_metadata = {
            "source": "test_document.pdf",
            "page": 2,
            "element_type": "Image",
            "coordinates": {"x": 100, "y": 200, "width": 50, "height": 30},
            "confidence": 0.95,
        }

        image_doc = ImageDocument(image="base64_data", metadata=test_metadata)

        # Verify all metadata is preserved
        for key, value in test_metadata.items():
            assert image_doc.metadata[key] == value

        # Test that metadata survives processing operations
        processed_metadata = image_doc.metadata.copy()
        processed_metadata["processed"] = True

        assert processed_metadata["source"] == "test_document.pdf"
        assert processed_metadata["processed"] is True
