"""Tests for core document loading functionality.

This module tests Unstructured document loading, document chunking,
and LlamaParse functionality with modern pytest fixtures and proper typing.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from llama_index.core import Document
from llama_index.core.schema import ImageDocument

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.document_loader import (
    chunk_documents_structured,
    load_documents_llama,
    load_documents_unstructured,
)

# Shared fixtures are automatically available via conftest.py


class TestLoadDocumentsUnstructured:
    """Test Unstructured document loading functionality with modern fixtures."""

    @pytest.mark.parametrize("parse_strategy", ["hi_res", "fast", "ocr_only"])
    @patch("utils.document_loader.partition")
    def test_load_documents_with_different_strategies(
        self,
        mock_partition: Mock,
        tmp_path: Path,
        mock_settings: Mock,
        parse_strategy: str,
    ) -> None:
        """Test document loading with different parse strategies."""
        # Update mock settings for parametrization
        mock_settings.parse_strategy = parse_strategy

        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        # Mock Unstructured elements
        mock_text_element = Mock()
        mock_text_element.category = "NarrativeText"
        mock_text_element.__str__.return_value = "Test paragraph content"
        mock_text_element.metadata.page_number = 1
        mock_text_element.metadata.filename = "test.pdf"
        mock_text_element.metadata.coordinates = {"x": 100, "y": 200}

        mock_partition.return_value = [mock_text_element]

        with (
            patch("utils.document_loader.settings", mock_settings),
            patch("utils.document_loader.chunk_documents_structured") as mock_chunk,
        ):
            mock_chunk.side_effect = lambda x: x  # Return input unchanged

            result = load_documents_unstructured(str(test_file))

            # Verify partition call with correct strategy
            mock_partition.assert_called_once()
            partition_args = mock_partition.call_args[1]
            assert partition_args["filename"] == str(test_file)
            assert partition_args["strategy"] == parse_strategy
            assert partition_args["extract_images_in_pdf"] is True
            assert partition_args["infer_table_structure"] is True
            assert partition_args["chunking_strategy"] == "by_title"

            # Verify result
            assert len(result) == 1
            assert result[0].text == "Test paragraph content"

    @pytest.mark.parametrize(
        ("element_type", "expected_metadata_key"),
        [
            ("Title", "element_type"),
            ("Table", "element_type"),
            ("FigureCaption", "element_type"),
            ("ListItem", "element_type"),
            ("Header", "element_type"),
        ],
    )
    @patch("utils.document_loader.partition")
    def test_load_documents_element_types(
        self,
        mock_partition: Mock,
        tmp_path: Path,
        mock_settings: Mock,
        element_type: str,
        expected_metadata_key: str,
    ) -> None:
        """Test handling of different Unstructured element types."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        # Mock element
        mock_element = Mock()
        mock_element.category = element_type
        mock_element.__str__.return_value = f"{element_type} content"
        mock_element.metadata.page_number = 1
        mock_element.metadata.filename = "test.pdf"

        mock_partition.return_value = [mock_element]

        with (
            patch("utils.document_loader.settings", mock_settings),
            patch("utils.document_loader.chunk_documents_structured") as mock_chunk,
        ):
            mock_chunk.side_effect = lambda x: x

            result = load_documents_unstructured(str(test_file))

            # Verify element type is preserved in metadata
            assert len(result) == 1
            assert result[0].metadata[expected_metadata_key] == element_type
            assert result[0].text == f"{element_type} content"

    @patch("utils.document_loader.partition")
    def test_load_documents_image_processing(
        self, mock_partition: Mock, tmp_path: Path, mock_settings: Mock
    ) -> None:
        """Test image element processing and ImageDocument creation."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        # Mock image elements with different data sources
        mock_image_metadata = Mock()
        mock_image_metadata.category = "Image"
        mock_image_metadata.metadata.page_number = 1
        mock_image_metadata.metadata.filename = "test.pdf"
        mock_image_metadata.metadata.image_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "2mNkYPhfDwAChAHgSJ/JVwAAAABJRU5ErkJggg=="
        )

        mock_image_text = Mock()
        mock_image_text.category = "Image"
        mock_image_text.metadata.page_number = 2
        mock_image_text.metadata.filename = "test.pdf"
        mock_image_text.metadata.image_base64 = None
        mock_image_text.text = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "2mNkYPhfDwAChAHgSJ/JVwAAAABJRU5ErkJggg=="
        )

        mock_partition.return_value = [mock_image_metadata, mock_image_text]

        with (
            patch("utils.document_loader.settings", mock_settings),
            patch("utils.document_loader.chunk_documents_structured") as mock_chunk,
        ):
            mock_chunk.side_effect = lambda x: x

            result = load_documents_unstructured(str(test_file))

            # Should create ImageDocument objects
            assert len(result) == 2
            for doc in result:
                assert isinstance(doc, ImageDocument)
                assert "image_base64" in doc.metadata
                assert doc.metadata["element_type"] == "Image"

    @patch("utils.document_loader.partition")
    def test_load_documents_fallback_on_error(
        self, mock_partition: Mock, tmp_path: Path
    ) -> None:
        """Test fallback to LlamaParse when Unstructured fails."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        # Mock partition failure
        mock_partition.side_effect = Exception("Unstructured parsing failed")

        with patch("utils.document_loader.load_documents_llama") as mock_llama_load:
            mock_llama_load.return_value = [Document(text="Fallback content")]

            with (
                patch("utils.document_loader.logging.error") as mock_log_error,
                patch("utils.document_loader.logging.info") as mock_log_info,
            ):
                result = load_documents_unstructured(str(test_file))

                # Verify error and fallback logging
                mock_log_error.assert_called_once()
                error_message = mock_log_error.call_args[0][0]
                assert "Error loading with Unstructured" in error_message

                mock_log_info.assert_called_with(
                    "Falling back to existing LlamaParse loader"
                )

                # Verify fallback was called and returned result
                mock_llama_load.assert_called_once()
                assert len(result) == 1
                assert result[0].text == "Fallback content"


class TestChunkDocumentsStructured:
    """Test structured document chunking functionality with modern fixtures."""

    @pytest.mark.parametrize(
        ("chunk_size", "chunk_overlap", "expected_chunks"),
        [
            (512, 50, 2),  # Small chunks, expect multiple
            (2048, 200, 1),  # Large chunks, expect fewer
            (1024, 100, 1),  # Medium chunks
        ],
    )
    def test_chunk_text_documents_parametrized(
        self,
        mock_settings: Mock,
        chunk_size: int,
        chunk_overlap: int,
        expected_chunks: int,
    ) -> None:
        """Test chunking with different size configurations."""
        # Update settings for parametrization
        mock_settings.chunk_size = chunk_size
        mock_settings.chunk_overlap = chunk_overlap

        # Create test document with known length
        long_text = "This is a sentence. " * 200  # ~4000 characters
        doc = Document(text=long_text, metadata={"source": "test.pdf", "page": 1})

        with (
            patch("utils.document_loader.settings", mock_settings),
            patch("llama_index.core.node_parser.SentenceSplitter") as mock_splitter,
        ):
            mock_splitter_instance = Mock()
            # Mock different chunk counts based on size
            mock_chunks = [
                Document(text=f"Chunk {i}", metadata=doc.metadata)
                for i in range(expected_chunks)
            ]
            mock_splitter_instance.get_nodes_from_documents.return_value = mock_chunks
            mock_splitter.return_value = mock_splitter_instance

            result = chunk_documents_structured([doc])

            # Verify splitter configuration
            mock_splitter.assert_called_once_with(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                paragraph_separator="\n\n",
                secondary_chunking_regex="[^,.;。]+[,.;。]?",
                tokenizer=None,
            )

            assert len(result) == expected_chunks

    def test_preserve_mixed_document_types(self, mock_settings: Mock) -> None:
        """Test that different document types are handled appropriately."""
        # Create mixed document types
        text_doc = Document(text="Text content", metadata={"source": "text.pdf"})
        image_doc = ImageDocument(
            image="base64_data", metadata={"source": "image.pdf", "page": 1}
        )
        short_text_doc = Document(text="Short", metadata={"source": "short.pdf"})

        documents = [text_doc, image_doc, short_text_doc]

        with (
            patch("utils.document_loader.settings", mock_settings),
            patch("llama_index.core.node_parser.SentenceSplitter") as mock_splitter,
        ):
            mock_splitter_instance = Mock()
            # Return text docs as processed, varying the returns
            mock_splitter_instance.get_nodes_from_documents.side_effect = [
                [text_doc],  # First text doc
                [short_text_doc],  # Second text doc
            ]
            mock_splitter.return_value = mock_splitter_instance

            result = chunk_documents_structured(documents)

            # Should have processed both text docs and preserved ImageDoc
            assert len(result) == 3

            # Verify ImageDocument was preserved unchanged
            image_docs = [doc for doc in result if isinstance(doc, ImageDocument)]
            assert len(image_docs) == 1
            assert image_docs[0] == image_doc

            # Verify text processing was called twice (once per text doc)
            assert mock_splitter_instance.get_nodes_from_documents.call_count == 2

    def test_chunk_error_handling(self, mock_settings: Mock) -> None:
        """Test graceful handling of chunking errors."""
        doc = Document(text="Test content", metadata={"source": "test.pdf"})

        with (
            patch("utils.document_loader.settings", mock_settings),
            patch("llama_index.core.node_parser.SentenceSplitter") as mock_splitter,
            patch("utils.document_loader.logging.warning") as mock_log_warning,
        ):
            mock_splitter_instance = Mock()
            mock_splitter_instance.get_nodes_from_documents.side_effect = Exception(
                "Chunking failed"
            )
            mock_splitter.return_value = mock_splitter_instance

            result = chunk_documents_structured([doc])

            # Should return original document on error
            assert len(result) == 1
            assert result[0] == doc

            # Should log warning
            mock_log_warning.assert_called()
            warning_message = mock_log_warning.call_args[0][0]
            assert "Error chunking document" in warning_message


class TestLoadDocumentsLlama:
    """Test LlamaParse document loading functionality with modern fixtures."""

    @pytest.mark.parametrize(
        ("file_type", "expected_type"),
        [
            ("pdf", "standard_document"),
            ("docx", "standard_document"),
            ("txt", "standard_document"),
        ],
    )
    @patch("utils.document_loader.LlamaParse")
    @patch("utils.document_loader.SimpleDirectoryReader")
    def test_load_standard_documents(
        self,
        mock_reader: Mock,
        mock_llama_parse: Mock,
        file_type: str,
        expected_type: str,
    ) -> None:
        """Test loading different standard document types."""
        # Create mock file
        mock_file = Mock()
        mock_file.name = f"test_document.{file_type}"
        mock_file.type = (
            f"application/{file_type}" if file_type != "txt" else "text/plain"
        )
        mock_file.getvalue.return_value = b"mock file content"

        # Mock parser and reader
        mock_parser = Mock()
        mock_llama_parse.return_value = mock_parser

        mock_reader_instance = Mock()
        mock_doc = Document(text="Parsed content", metadata={})
        mock_reader_instance.load_data.return_value = [mock_doc]
        mock_reader.return_value = mock_reader_instance

        with patch("utils.document_loader.tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = Mock()
            mock_temp_file.name = f"/tmp/test_document.{file_type}"
            mock_temp.__enter__.return_value = mock_temp_file

            with patch("utils.document_loader.os.remove"):
                result = load_documents_llama(
                    [mock_file], parse_media=False, enable_multimodal=False
                )

                # Verify parser creation
                mock_llama_parse.assert_called_once_with(result_type="markdown")

                # Verify document metadata
                assert len(result) == 1
                assert result[0].text == "Parsed content"
                assert result[0].metadata["source"] == mock_file.name
                assert result[0].metadata["type"] == expected_type
                assert result[0].metadata["has_images"] is False

    @pytest.mark.parametrize(
        ("audio_format", "expected_device"),
        [
            ("mp3", "cuda"),  # Will use CUDA if available
            ("wav", "cuda"),
            ("m4a", "cuda"),
        ],
    )
    @patch("utils.document_loader.whisper_load")
    @patch("utils.document_loader.torch.cuda.is_available")
    def test_load_audio_files(
        self,
        mock_cuda_available: Mock,
        mock_whisper_load: Mock,
        audio_format: str,
        expected_device: str,
    ) -> None:
        """Test loading different audio file formats."""
        # Mock CUDA availability
        mock_cuda_available.return_value = True

        # Create mock audio file
        mock_audio_file = Mock()
        mock_audio_file.name = f"test_audio.{audio_format}"
        mock_audio_file.type = f"audio/{audio_format}"
        mock_audio_file.getvalue.return_value = b"mock audio data"

        # Mock Whisper model
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": f"Transcribed {audio_format} content"
        }
        mock_whisper_load.return_value = mock_model

        with patch("utils.document_loader.tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = Mock()
            mock_temp_file.name = f"/tmp/test_audio.{audio_format}"
            mock_temp.__enter__.return_value = mock_temp_file

            with patch("utils.document_loader.os.remove"):
                result = load_documents_llama(
                    [mock_audio_file], parse_media=True, enable_multimodal=False
                )

                # Verify Whisper model loading
                mock_whisper_load.assert_called_once_with(
                    "base", device=expected_device
                )

                # Verify transcription
                mock_model.transcribe.assert_called_once()

                # Verify result
                assert len(result) == 1
                assert result[0].text == f"Transcribed {audio_format} content"
                assert result[0].metadata["source"] == mock_audio_file.name
                assert result[0].metadata["type"] == "audio"

    @patch("utils.document_loader.VideoFileClip")
    @patch("utils.document_loader.whisper_load")
    @patch("utils.document_loader.torch.cuda.is_available")
    def test_load_video_with_frame_extraction(
        self,
        mock_cuda_available: Mock,
        mock_whisper_load: Mock,
        mock_video_clip: Mock,
    ) -> None:
        """Test loading video file with frame extraction."""
        mock_cuda_available.return_value = False  # Test CPU fallback

        # Create mock video file
        mock_video_file = Mock()
        mock_video_file.name = "test_video.mp4"
        mock_video_file.type = "video/mp4"
        mock_video_file.getvalue.return_value = b"mock video data"

        # Mock Whisper
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "Transcribed video audio"}
        mock_whisper_load.return_value = mock_model

        # Mock VideoFileClip with specific duration for predictable frame extraction
        mock_clip = Mock()
        mock_clip.duration = 20  # 20 second video
        mock_clip.get_frame.return_value = "mock_frame_array"
        mock_clip.audio = Mock()
        mock_video_clip.return_value = mock_clip

        with patch("utils.document_loader.tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = Mock()
            mock_temp_file.name = "/tmp/test_video.mp4"
            mock_temp.__enter__.return_value = mock_temp_file

            with (
                patch("utils.document_loader.Image.fromarray") as mock_image,
                patch("utils.document_loader.os.remove"),
            ):
                mock_image.return_value = "mock_pil_image"

                result = load_documents_llama(
                    [mock_video_file], parse_media=True, enable_multimodal=False
                )

                # Verify frame extraction (every 5 seconds)
                expected_frames = 4  # At 0, 5, 10, 15 seconds for 20s video
                assert mock_clip.get_frame.call_count == expected_frames

                # Verify result
                assert len(result) == 1
                assert result[0].text == "Transcribed video audio"
                assert result[0].metadata["source"] == "test_video.mp4"
                assert result[0].metadata["type"] == "video"
                assert "images" in result[0].metadata

    @pytest.mark.parametrize(
        ("error_type", "expected_log"),
        [
            (FileNotFoundError("File not found"), "File not found"),
            (PermissionError("Permission denied"), "Permission denied"),
            (Exception("Generic error"), "Generic error"),
        ],
    )
    def test_load_documents_error_handling(
        self, error_type: Exception, expected_log: str
    ) -> None:
        """Test error handling for various file loading failures."""
        mock_problem_file = Mock()
        mock_problem_file.name = "problem.pdf"
        mock_problem_file.getvalue.side_effect = error_type

        with patch("utils.document_loader.logging.error") as mock_log_error:
            result = load_documents_llama([mock_problem_file])

            # Should handle error gracefully and return empty list
            assert result == []

            # Verify error logging
            mock_log_error.assert_called_once()
            error_message = mock_log_error.call_args[0][0]
            assert (
                "File not found: problem.pdf" in error_message
                or expected_log in error_message
            )
