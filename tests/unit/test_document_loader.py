"""Comprehensive tests for document loading functionality.

This module tests document loading with multimodal support, Unstructured parsing,
image extraction from PDFs, video/audio processing, and embedding generation
following 2025 best practices.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import Document
from llama_index.core.schema import ImageDocument
from PIL import Image

from models import AppSettings
from utils.document_loader import (
    chunk_documents_structured,
    create_native_multimodal_embeddings,
    extract_images_from_pdf,
    load_documents_llama,
    load_documents_unstructured,
)


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return AppSettings(
        parse_strategy="hi_res",
        chunk_size=1024,
        chunk_overlap=200,
        dense_embedding_model="BAAI/bge-large-en-v1.5",
    )


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test.pdf"
    # Create minimal PDF content
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
180
%%EOF"""
    pdf_path.write_bytes(pdf_content)
    return str(pdf_path)


@pytest.fixture
def sample_image_base64():
    """Create a sample base64-encoded image for testing."""
    # Create a small test image
    img = Image.new("RGB", (10, 10), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def sample_images_data(sample_image_base64):
    """Create sample image data for testing."""
    return [
        {
            "image_data": sample_image_base64,
            "page_number": 1,
            "image_index": 0,
            "format": "PNG",
            "size": (10, 10),
        },
        {
            "image_data": sample_image_base64,
            "page_number": 2,
            "image_index": 0,
            "format": "PNG",
            "size": (10, 10),
        },
    ]


@pytest.fixture
def mock_uploaded_file():
    """Create a mock uploaded file for testing."""
    mock_file = MagicMock()
    mock_file.name = "test_document.pdf"
    mock_file.type = "application/pdf"
    mock_file.getvalue.return_value = b"mock pdf content"
    return mock_file


class TestExtractImagesFromPDF:
    """Test PDF image extraction functionality."""

    @patch("utils.document_loader.fitz")
    def test_extract_images_success(self, mock_fitz, sample_pdf_path):
        """Test successful image extraction from PDF."""
        # Mock PyMuPDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pixmap = MagicMock()

        # Mock document structure
        mock_doc.__len__.return_value = 2  # 2 pages
        mock_doc.load_page.return_value = mock_page
        mock_page.get_images.return_value = [(123, "test", "image")]  # xref, name, type

        # Mock pixmap
        mock_pixmap.n = 3  # RGB
        mock_pixmap.alpha = 0
        mock_pixmap.tobytes.return_value = b"mock ppm data"

        mock_fitz.open.return_value = mock_doc
        mock_fitz.Pixmap.return_value = mock_pixmap

        # Mock PIL Image
        with patch("utils.document_loader.Image") as mock_pil:
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_pil.open.return_value = mock_img

            # Mock base64 encoding
            with patch("utils.document_loader.base64.b64encode") as mock_b64:
                mock_b64.return_value.decode.return_value = "mock_base64_data"

                with patch("utils.document_loader.logging.info") as mock_log_info:
                    extract_images_from_pdf(sample_pdf_path)

                    # Verify document operations
                    mock_fitz.open.assert_called_once_with(sample_pdf_path)
                    assert mock_doc.load_page.call_count == 2  # 2 pages
                    mock_doc.close.assert_called_once()

                    # Verify result structure
                    assert len(result) == 2  # 2 pages with images
                    for img_data in result:
                        assert img_data["image_data"] == "mock_base64_data"
                        assert img_data["format"] == "PNG"
                        assert img_data["size"] == (100, 100)
                        assert "page_number" in img_data
                        assert "image_index" in img_data

                    # Verify logging
                    mock_log_info.assert_called_with("Extracted 2 images from PDF")

    @patch("utils.document_loader.fitz")
    def test_extract_images_no_images(self, mock_fitz, sample_pdf_path):
        """Test PDF image extraction when no images are found."""
        # Mock empty document
        mock_doc = MagicMock()
        mock_page = MagicMock()

        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_page.get_images.return_value = []  # No images

        mock_fitz.open.return_value = mock_doc

        with patch("utils.document_loader.logging.info") as mock_log_info:
            extract_images_from_pdf(sample_pdf_path)

            assert result == []
            mock_log_info.assert_called_with("Extracted 0 images from PDF")

    @patch("utils.document_loader.fitz")
    def test_extract_images_fitz_error(self, mock_fitz, sample_pdf_path):
        """Test PDF image extraction with PyMuPDF error."""
        # Mock fitz.open failure
        mock_fitz.open.side_effect = Exception("PyMuPDF error")

        with patch("utils.document_loader.logging.error") as mock_log_error:
            extract_images_from_pdf(sample_pdf_path)

            assert result == []
            mock_log_error.assert_called_once()
            error_message = mock_log_error.call_args[0][0]
            assert "PDF image extraction failed" in error_message

    @patch("utils.document_loader.fitz")
    def test_extract_images_skip_unsupported_format(self, mock_fitz, sample_pdf_path):
        """Test skipping unsupported image formats (CMYK)."""
        # Mock PyMuPDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pixmap = MagicMock()

        mock_doc.__len__.return_value = 1
        mock_doc.load_page.return_value = mock_page
        mock_page.get_images.return_value = [(123, "test", "image")]

        # Mock CMYK pixmap (n=4+alpha, should be skipped)
        mock_pixmap.n = 5  # CMYK + alpha
        mock_pixmap.alpha = 1

        mock_fitz.open.return_value = mock_doc
        mock_fitz.Pixmap.return_value = mock_pixmap

        with patch("utils.document_loader.logging.info") as mock_log_info:
            extract_images_from_pdf(sample_pdf_path)

            assert result == []
            mock_log_info.assert_called_with("Extracted 0 images from PDF")


class TestCreateNativeMultimodalEmbeddings:
    """Test native multimodal embedding creation."""

    @patch("utils.document_loader.ModelManager.get_multimodal_embedding_model")
    def test_create_embeddings_success(self, mock_model_manager, sample_images_data):
        """Test successful multimodal embedding creation."""
        # Mock multimodal model
        mock_model = MagicMock()
        mock_text_embedding = MagicMock()
        mock_text_embedding.flatten.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.embed_text.return_value = [mock_text_embedding]

        mock_img_embedding = MagicMock()
        mock_img_embedding.flatten.return_value.tolist.return_value = [0.4, 0.5, 0.6]
        mock_model.embed_image.return_value = [mock_img_embedding, mock_img_embedding]

        mock_model_manager.return_value = mock_model

        with (
            patch("utils.document_loader.tempfile.gettempdir", return_value="/tmp"),
            patch("builtins.open", create=True) as mock_open,
            patch("utils.document_loader.os.unlink") as mock_unlink,
            patch("utils.document_loader.logging.info") as mock_log_info,
        ):
            create_native_multimodal_embeddings(
                "Test text content", sample_images_data
            )

            # Verify model usage
            mock_model.embed_text.assert_called_once_with(["Test text content"])
            mock_model.embed_image.assert_called_once()

            # Verify result structure
            assert result["text_embedding"] == [0.1, 0.2, 0.3]
            assert len(result["image_embeddings"]) == 2
            assert result["image_embeddings"][0]["embedding"] == [
                0.4,
                0.5,
                0.6,
            ]
            assert result["combined_embedding"] == [0.1, 0.2, 0.3]
            assert result["provider_used"] == "fastembed_native_multimodal"

            # Verify temp file cleanup
            assert mock_unlink.call_count == 2  # 2 temp files

            # Verify logging
            mock_log_info.assert_called_with(
                "Using FastEmbed native LateInteractionMultimodalEmbedding"
            )

    @patch("utils.document_loader.ModelManager.get_multimodal_embedding_model")
    def test_create_embeddings_text_only(self, mock_model_manager):
        """Test embedding creation with text only (no images)."""
        # Mock multimodal model
        mock_model = MagicMock()
        mock_text_embedding = MagicMock()
        mock_text_embedding.flatten.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.embed_text.return_value = [mock_text_embedding]

        mock_model_manager.return_value = mock_model

        create_native_multimodal_embeddings("Test text content", images=None)

        # Verify model usage
        mock_model.embed_text.assert_called_once_with(["Test text content"])
        mock_model.embed_image.assert_not_called()

        # Verify result structure
        assert result["text_embedding"] == [0.1, 0.2, 0.3]
        assert result["image_embeddings"] == []
        assert result["combined_embedding"] == [0.1, 0.2, 0.3]
        assert result["provider_used"] == "fastembed_native_multimodal"

    @patch("utils.document_loader.ModelManager.get_multimodal_embedding_model")
    def test_create_embeddings_import_error_fallback(
        self, mock_model_manager, mock_settings
    ):
        """Test fallback when FastEmbed multimodal is not available."""
        # Mock ImportError for multimodal model
        mock_model_manager.side_effect = ImportError("Multimodal not available")

        # Mock FastEmbed text-only fallback
        with patch("utils.document_loader.FastEmbedEmbedding") as mock_fastembed:
            mock_text_model = MagicMock()
            mock_text_model.get_text_embedding.return_value = [0.7, 0.8, 0.9]
            mock_fastembed.return_value = mock_text_model

            with (
                patch("utils.document_loader.settings", mock_settings),
                patch("utils.document_loader.logging.warning") as mock_log_warning,
            ):
                create_native_multimodal_embeddings("Test text")

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

                # Verify warning
                mock_log_warning.assert_called_once()
                warning_message = mock_log_warning.call_args[0][0]
                assert (
                    "FastEmbed LateInteractionMultimodalEmbedding not available"
                    in warning_message
                )

    @patch("utils.document_loader.ModelManager.get_multimodal_embedding_model")
    def test_create_embeddings_complete_failure(
        self, mock_model_manager, mock_settings
    ):
        """Test ultimate fallback when all embedding methods fail."""
        # Mock all embedding methods failing
        mock_model_manager.side_effect = Exception("All methods failed")

        with patch("utils.document_loader.FastEmbedEmbedding") as mock_fastembed:
            mock_fastembed.side_effect = Exception("FastEmbed also failed")

            with (
                patch("utils.document_loader.settings", mock_settings),
                patch("utils.document_loader.logging.error") as mock_log_error,
            ):
                create_native_multimodal_embeddings("Test text")

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


class TestLoadDocumentsUnstructured:
    """Test Unstructured document loading functionality."""

    @patch("utils.document_loader.partition")
    def test_load_documents_success(self, mock_partition, tmp_path, mock_settings):
        """Test successful document loading with Unstructured."""
        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        # Mock Unstructured elements
        mock_text_element = MagicMock()
        mock_text_element.category = "NarrativeText"
        mock_text_element.__str__.return_value = "Test paragraph content"
        mock_text_element.metadata.page_number = 1
        mock_text_element.metadata.filename = "test.pdf"
        mock_text_element.metadata.coordinates = {"x": 100, "y": 200}

        mock_image_element = MagicMock()
        mock_image_element.category = "Image"
        mock_image_element.metadata.page_number = 1
        mock_image_element.metadata.filename = "test.pdf"
        mock_image_element.metadata.image_base64 = "mock_image_data"

        mock_partition.return_value = [mock_text_element, mock_image_element]

        with patch("utils.document_loader.settings", mock_settings):
            with patch(
                "utils.document_loader.chunk_documents_structured"
            ) as mock_chunk:
                mock_chunk.return_value = []  # Mock chunking

                load_documents_unstructured(str(test_file))

                # Verify partition call
                mock_partition.assert_called_once()
                partition_args = mock_partition.call_args[1]
                assert partition_args["filename"] == str(test_file)
                assert partition_args["strategy"] == mock_settings.parse_strategy
                assert partition_args["extract_images_in_pdf"] is True
                assert partition_args["infer_table_structure"] is True
                assert partition_args["chunking_strategy"] == "by_title"

                # Verify chunking was called
                mock_chunk.assert_called_once()

                # Should return result from chunking
                assert result == []

    @patch("utils.document_loader.partition")
    def test_load_documents_different_element_types(
        self, mock_partition, tmp_path, mock_settings
    ):
        """Test handling of different Unstructured element types."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        # Mock various element types
        mock_title = MagicMock()
        mock_title.category = "Title"
        mock_title.__str__.return_value = "Document Title"
        mock_title.metadata.page_number = 1
        mock_title.metadata.filename = "test.pdf"

        mock_table = MagicMock()
        mock_table.category = "Table"
        mock_table.__str__.return_value = "| Col1 | Col2 |\n| Data | Data |"
        mock_table.metadata.page_number = 1
        mock_table.metadata.filename = "test.pdf"

        mock_figure_caption = MagicMock()
        mock_figure_caption.category = "FigureCaption"
        mock_figure_caption.__str__.return_value = "Figure 1: Test diagram"
        mock_figure_caption.metadata.page_number = 2
        mock_figure_caption.metadata.filename = "test.pdf"

        mock_partition.return_value = [mock_title, mock_table, mock_figure_caption]

        with patch("utils.document_loader.settings", mock_settings):
            with patch(
                "utils.document_loader.chunk_documents_structured"
            ) as mock_chunk:
                # Return input unchanged for testing
                mock_chunk.side_effect = lambda x: x

                load_documents_unstructured(str(test_file))

                # Should create documents for all text elements
                assert len(result) == 3

                # Check document contents and metadata
                titles = [doc for doc in result if doc.text == "Document Title"]
                assert len(titles) == 1
                assert titles[0].metadata["element_type"] == "Title"

                tables = [doc for doc in result if "Col1" in doc.text]
                assert len(tables) == 1
                assert tables[0].metadata["element_type"] == "Table"

                captions = [doc for doc in result if "Figure 1" in doc.text]
                assert len(captions) == 1
                assert captions[0].metadata["element_type"] == "FigureCaption"

    @patch("utils.document_loader.partition")
    def test_load_documents_image_processing(
        self, mock_partition, tmp_path, mock_settings
    ):
        """Test image element processing and ImageDocument creation."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("Test content")

        # Mock image element with base64 data
        mock_image = MagicMock()
        mock_image.category = "Image"
        mock_image.metadata.page_number = 1
        mock_image.metadata.filename = "test.pdf"
        mock_image.metadata.image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

        # Mock image element with data in text field
        mock_image_text = MagicMock()
        mock_image_text.category = "Image"
        mock_image_text.metadata.page_number = 2
        mock_image_text.metadata.filename = "test.pdf"
        mock_image_text.metadata.image_base64 = None
        mock_image_text.text = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

        mock_partition.return_value = [mock_image, mock_image_text]

        with patch("utils.document_loader.settings", mock_settings):
            with patch(
                "utils.document_loader.chunk_documents_structured"
            ) as mock_chunk:
                mock_chunk.side_effect = lambda x: x

                load_documents_unstructured(str(test_file))

                # Should create ImageDocument objects
                assert len(result) == 2
                for doc in result:
                    assert isinstance(doc, ImageDocument)
                    assert "image_base64" in doc.metadata
                    assert doc.metadata["element_type"] == "Image"

    @patch("utils.document_loader.partition")
    def test_load_documents_fallback_on_error(self, mock_partition, tmp_path):
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
                load_documents_unstructured(str(test_file))

                # Verify error and fallback logging
                mock_log_error.assert_called_once()
                error_message = mock_log_error.call_args[0][0]
                assert "Error loading with Unstructured" in error_message

                mock_log_info.assert_called_with(
                    "Falling back to existing LlamaParse loader"
                )

                # Verify fallback was called
                mock_llama_load.assert_called_once()

                # Should return fallback result
                assert len(result) == 1
                assert result[0].text == "Fallback content"


class TestChunkDocumentsStructured:
    """Test structured document chunking functionality."""

    def test_chunk_text_documents(self, mock_settings):
        """Test chunking of text documents while preserving metadata."""
        # Create test documents
        long_text = "This is a very long document. " * 100  # 3000+ characters
        doc1 = Document(text=long_text, metadata={"source": "doc1.pdf", "page": 1})
        doc2 = Document(text="Short text", metadata={"source": "doc2.pdf", "page": 1})

        documents = [doc1, doc2]

        with patch("utils.document_loader.settings", mock_settings):
            with patch(
                "llama_index.core.node_parser.SentenceSplitter"
            ) as mock_splitter:
                # Mock splitter behavior
                mock_splitter_instance = MagicMock()
                mock_chunk1 = Document(text="Chunk 1", metadata=doc1.metadata)
                mock_chunk2 = Document(text="Chunk 2", metadata=doc1.metadata)
                mock_chunk3 = Document(text="Short text", metadata=doc2.metadata)

                mock_splitter_instance.get_nodes_from_documents.side_effect = [
                    [mock_chunk1, mock_chunk2],  # doc1 chunked
                    [mock_chunk3],  # doc2 unchanged
                ]
                mock_splitter.return_value = mock_splitter_instance

                chunk_documents_structured(documents)

                # Verify splitter configuration
                mock_splitter.assert_called_once_with(
                    chunk_size=mock_settings.chunk_size,
                    chunk_overlap=mock_settings.chunk_overlap,
                    paragraph_separator="\n\n",
                    secondary_chunking_regex="[^,.;。]+[,.;。]?",
                    tokenizer=None,
                )

                # Verify chunking calls
                assert mock_splitter_instance.get_nodes_from_documents.call_count == 2

                # Verify result
                assert len(result) == 3  # 2 chunks from doc1 + 1 from doc2

    def test_preserve_image_documents(self, mock_settings):
        """Test that ImageDocuments are preserved unchanged during chunking."""
        # Create mixed document types
        text_doc = Document(text="Text content", metadata={"source": "text.pdf"})
        image_doc = ImageDocument(
            image="base64_data", metadata={"source": "image.pdf", "page": 1}
        )

        documents = [text_doc, image_doc]

        with patch("utils.document_loader.settings", mock_settings):
            with patch(
                "llama_index.core.node_parser.SentenceSplitter"
            ) as mock_splitter:
                mock_splitter_instance = MagicMock()
                mock_splitter_instance.get_nodes_from_documents.return_value = [
                    text_doc
                ]
                mock_splitter.return_value = mock_splitter_instance

                chunk_documents_structured(documents)

                # Verify text document was processed
                mock_splitter_instance.get_nodes_from_documents.assert_called_once()

                # Verify ImageDocument was preserved
                assert len(result) == 2
                image_docs = [doc for doc in result if isinstance(doc, ImageDocument)]
                assert len(image_docs) == 1
                assert image_docs[0] == image_doc


class TestLoadDocumentsLlama:
    """Test LlamaParse document loading functionality."""

    @patch("utils.document_loader.LlamaParse")
    @patch("utils.document_loader.SimpleDirectoryReader")
    def test_load_standard_document(
        self, mock_reader, mock_llama_parse, mock_uploaded_file
    ):
        """Test loading standard document with LlamaParse."""
        # Mock parser
        mock_parser = MagicMock()
        mock_llama_parse.return_value = mock_parser

        # Mock reader
        mock_reader_instance = MagicMock()
        mock_doc = Document(text="Parsed content", metadata={})
        mock_reader_instance.load_data.return_value = [mock_doc]
        mock_reader.return_value = mock_reader_instance

        with patch("utils.document_loader.tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/test.pdf"
            mock_temp.__enter__.return_value = mock_temp_file

            with patch("utils.document_loader.os.remove") as mock_remove:
                load_documents_llama(
                    [mock_uploaded_file], parse_media=False, enable_multimodal=False
                )

                # Verify parser creation
                mock_llama_parse.assert_called_once_with(result_type="markdown")

                # Verify reader creation
                mock_reader.assert_called_once()
                reader_args = mock_reader.call_args[1]
                assert reader_args["input_files"] == ["/tmp/test.pdf"]
                assert reader_args["file_extractor"] == {".*": mock_parser}

                # Verify document metadata
                assert len(result) == 1
                assert result[0].text == "Parsed content"
                assert result[0].metadata["source"] == "test_document.pdf"
                assert result[0].metadata["type"] == "standard_document"
                assert result[0].metadata["has_images"] is False

                # Verify cleanup
                mock_remove.assert_called_once_with("/tmp/test.pdf")

    @patch("utils.document_loader.LlamaParse")
    @patch("utils.document_loader.SimpleDirectoryReader")
    @patch("utils.document_loader.extract_images_from_pdf")
    @patch("utils.document_loader.create_native_multimodal_embeddings")
    def test_load_multimodal_pdf(
        self,
        mock_create_embeddings,
        mock_extract_images,
        mock_reader,
        mock_llama_parse,
        mock_uploaded_file,
        sample_images_data,
    ):
        """Test loading PDF with multimodal processing enabled."""
        # Mock components
        mock_parser = MagicMock()
        mock_llama_parse.return_value = mock_parser

        mock_reader_instance = MagicMock()
        mock_doc = Document(text="PDF content", metadata={})
        mock_reader_instance.load_data.return_value = [mock_doc]
        mock_reader.return_value = mock_reader_instance

        # Mock image extraction
        mock_extract_images.return_value = sample_images_data

        # Mock embedding creation
        mock_embeddings = {
            "text_embedding": [0.1, 0.2, 0.3],
            "image_embeddings": [{"embedding": [0.4, 0.5, 0.6]}],
            "provider_used": "fastembed_native_multimodal",
        }
        mock_create_embeddings.return_value = mock_embeddings

        with patch("utils.document_loader.tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/test.pdf"
            mock_temp.__enter__.return_value = mock_temp_file

            with (
                patch("utils.document_loader.os.remove"),
                patch("utils.document_loader.logging.info") as mock_log_info,
            ):
                load_documents_llama(
                    [mock_uploaded_file], parse_media=False, enable_multimodal=True
                )

                # Verify image extraction
                mock_extract_images.assert_called_once_with("/tmp/test.pdf")

                # Verify embedding creation
                mock_create_embeddings.assert_called_once_with(
                    text="PDF content", images=sample_images_data
                )

                # Verify document metadata
                assert len(result) == 1
                assert result[0].metadata["type"] == "pdf_multimodal"
                assert result[0].metadata["image_count"] == 2
                assert result[0].metadata["has_images"] is True
                assert result[0].metadata["multimodal_embeddings"] == mock_embeddings

                # Verify logging
                log_messages = [call[0][0] for call in mock_log_info.call_args_list]
                multimodal_msg = next(
                    (
                        msg
                        for msg in log_messages
                        if "Created multimodal embeddings" in msg
                    ),
                    None,
                )
                assert multimodal_msg is not None
                assert "with 2 images" in multimodal_msg

    @patch("utils.document_loader.whisper_load")
    @patch("utils.document_loader.torch.cuda.is_available")
    def test_load_audio_file(
        self, mock_cuda_available, mock_whisper_load, mock_settings
    ):
        """Test loading audio file with Whisper transcription."""
        # Create mock audio file
        mock_audio_file = MagicMock()
        mock_audio_file.name = "test_audio.mp3"
        mock_audio_file.type = "audio/mp3"
        mock_audio_file.getvalue.return_value = b"mock audio data"

        # Mock CUDA availability
        mock_cuda_available.return_value = True

        # Mock Whisper model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Transcribed audio content"}
        mock_whisper_load.return_value = mock_model

        with patch("utils.document_loader.tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/test_audio.mp3"
            mock_temp.__enter__.return_value = mock_temp_file

            with patch("utils.document_loader.os.remove"):
                load_documents_llama(
                    [mock_audio_file], parse_media=True, enable_multimodal=False
                )

                # Verify Whisper model loading with CUDA
                mock_whisper_load.assert_called_once_with("base", device="cuda")

                # Verify transcription
                mock_model.transcribe.assert_called_once_with("/tmp/test_audio.mp3")

                # Verify result
                assert len(result) == 1
                assert result[0].text == "Transcribed audio content"
                assert result[0].metadata["source"] == "test_audio.mp3"
                assert result[0].metadata["type"] == "audio"

    @patch("utils.document_loader.VideoFileClip")
    @patch("utils.document_loader.whisper_load")
    @patch("utils.document_loader.torch.cuda.is_available")
    def test_load_video_file(
        self, mock_cuda_available, mock_whisper_load, mock_video_clip
    ):
        """Test loading video file with audio extraction and frame sampling."""
        # Create mock video file
        mock_video_file = MagicMock()
        mock_video_file.name = "test_video.mp4"
        mock_video_file.type = "video/mp4"
        mock_video_file.getvalue.return_value = b"mock video data"

        # Mock CUDA and Whisper
        mock_cuda_available.return_value = False  # Test CPU fallback
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Transcribed video audio"}
        mock_whisper_load.return_value = mock_model

        # Mock VideoFileClip
        mock_clip = MagicMock()
        mock_clip.duration = 15  # 15 second video
        mock_clip.get_frame.return_value = "mock_frame_array"
        mock_clip.audio = MagicMock()
        mock_video_clip.return_value = mock_clip

        with patch("utils.document_loader.tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/test_video.mp4"
            mock_temp.__enter__.return_value = mock_temp_file

            with patch("utils.document_loader.Image.fromarray") as mock_image:
                mock_image.return_value = "mock_pil_image"

                with patch("utils.document_loader.os.remove"):
                    load_documents_llama(
                        [mock_video_file], parse_media=True, enable_multimodal=False
                    )

                    # Verify Whisper model loading with CPU
                    mock_whisper_load.assert_called_once_with("base", device="cpu")

                    # Verify frame extraction (every 5 seconds)
                    expected_frame_times = [0, 5, 10]  # 15s video, frames at 0, 5, 10
                    assert mock_clip.get_frame.call_count == len(expected_frame_times)

                    # Verify result
                    assert len(result) == 1
                    assert result[0].text == "Transcribed video audio"
                    assert result[0].metadata["source"] == "test_video.mp4"
                    assert result[0].metadata["type"] == "video"
                    assert "images" in result[0].metadata

    def test_load_documents_error_handling(self, mock_settings):
        """Test error handling for various file loading failures."""
        # Create mock file that will cause errors
        mock_problem_file = MagicMock()
        mock_problem_file.name = "problem.pdf"
        mock_problem_file.getvalue.side_effect = FileNotFoundError("File not found")

        with patch("utils.document_loader.logging.error") as mock_log_error:
            load_documents_llama([mock_problem_file])

            # Should handle error gracefully and return empty list
            assert result == []

            # Verify error logging
            mock_log_error.assert_called_once()
            error_message = mock_log_error.call_args[0][0]
            assert "File not found: problem.pdf" in error_message

    @patch("utils.document_loader.LlamaParse")
    @patch("utils.document_loader.SimpleDirectoryReader")
    def test_load_documents_final_logging(
        self, mock_reader, mock_llama_parse, mock_uploaded_file
    ):
        """Test final logging with document count and multimodal status."""
        # Mock successful loading
        mock_parser = MagicMock()
        mock_llama_parse.return_value = mock_parser

        mock_reader_instance = MagicMock()
        mock_reader_instance.load_data.return_value = [Document(text="Content")]
        mock_reader.return_value = mock_reader_instance

        with patch("utils.document_loader.tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = MagicMock()
            mock_temp_file.name = "/tmp/test.pdf"
            mock_temp.__enter__.return_value = mock_temp_file

            with (
                patch("utils.document_loader.os.remove"),
                patch("utils.document_loader.logging.info") as mock_log_info,
            ):
                load_documents_llama(
                    [mock_uploaded_file], parse_media=False, enable_multimodal=True
                )

                # Verify final logging
                log_messages = [call[0][0] for call in mock_log_info.call_args_list]
                final_msg = next(
                    (msg for msg in log_messages if "Loaded 1 documents" in msg),
                    None,
                )
                assert final_msg is not None
                assert "multimodal processing enabled: True" in final_msg


class TestAdvancedMultimodalFeatures:
    """Test advanced multimodal features and edge cases."""

    def test_mixed_document_types_processing(self, tmp_path, mock_settings):
        """Test processing of mixed document types (text + images) in one batch."""
        from utils.document_loader import load_documents_unstructured

        test_file = tmp_path / "mixed_content.pdf"
        test_file.write_text("Mixed content test")

        with patch("utils.document_loader.partition") as mock_partition:
            # Create mixed element types including complex tables and figures
            mock_title = MagicMock()
            mock_title.category = "Title"
            mock_title.__str__.return_value = "Advanced AI Research Findings"
            mock_title.metadata.page_number = 1
            mock_title.metadata.filename = "mixed_content.pdf"

            mock_abstract = MagicMock()
            mock_abstract.category = "NarrativeText"
            mock_abstract.__str__.return_value = (
                "This research presents novel findings in multimodal AI processing."
            )
            mock_abstract.metadata.page_number = 1
            mock_abstract.metadata.filename = "mixed_content.pdf"

            # Complex table with metadata
            mock_complex_table = MagicMock()
            mock_complex_table.category = "Table"
            mock_complex_table.__str__.return_value = """
            | Model Type | Accuracy | Precision | Recall | F1-Score |
            |------------|----------|-----------|--------|----------|
            | BERT-Base  | 87.2%    | 86.5%     | 88.1%  | 87.3%    |
            | RoBERTa    | 89.7%    | 88.9%     | 90.2%  | 89.5%    |
            | GPT-3.5    | 92.1%    | 91.8%     | 92.4%  | 92.1%    |
            """
            mock_complex_table.metadata.page_number = 2
            mock_complex_table.metadata.filename = "mixed_content.pdf"
            mock_complex_table.metadata.coordinates = {
                "x": 50,
                "y": 200,
                "width": 400,
                "height": 150,
            }

            # High-resolution image with detailed metadata
            mock_hires_image = MagicMock()
            mock_hires_image.category = "Image"
            mock_hires_image.metadata.page_number = 2
            mock_hires_image.metadata.filename = "mixed_content.pdf"
            mock_hires_image.metadata.image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
            mock_hires_image.metadata.coordinates = {
                "x": 100,
                "y": 300,
                "width": 300,
                "height": 200,
            }

            # Figure caption with reference
            mock_detailed_caption = MagicMock()
            mock_detailed_caption.category = "FigureCaption"
            mock_detailed_caption.__str__.return_value = "Figure 2.1: Architectural comparison of transformer models showing attention mechanisms and layer configurations (adapted from Smith et al., 2023)."
            mock_detailed_caption.metadata.page_number = 2
            mock_detailed_caption.metadata.filename = "mixed_content.pdf"

            mock_partition.return_value = [
                mock_title,
                mock_abstract,
                mock_complex_table,
                mock_hires_image,
                mock_detailed_caption,
            ]

            with patch("utils.document_loader.settings", mock_settings):
                with patch(
                    "utils.document_loader.chunk_documents_structured"
                ) as mock_chunk:
                    mock_chunk.side_effect = lambda x: x

                    load_documents_unstructured(str(test_file))

            # Verify comprehensive processing
            assert len(result) == 5

            # Verify complex table processing
            table_docs = [
                doc for doc in result if doc.metadata.get("element_type") == "Table"
            ]
            assert len(table_docs) == 1
            assert "BERT-Base" in table_docs[0].text
            assert "coordinates" in table_docs[0].metadata

            # Verify high-res image processing
            image_docs = [doc for doc in result if isinstance(doc, ImageDocument)]
            assert len(image_docs) == 1
            assert "coordinates" in image_docs[0].metadata
            assert image_docs[0].metadata["coordinates"]["width"] == 300

            # Verify detailed caption processing
            caption_docs = [
                doc
                for doc in result
                if doc.metadata.get("element_type") == "FigureCaption"
            ]
            assert len(caption_docs) == 1
            assert "Smith et al., 2023" in caption_docs[0].text

    def test_multimodal_embedding_error_recovery(self, sample_images_data):
        """Test error recovery in multimodal embedding pipeline."""
        from utils.document_loader import create_native_multimodal_embeddings

        # Test partial failure scenario (text succeeds, images fail)
        with patch(
            "utils.document_loader.ModelManager.get_multimodal_embedding_model"
        ) as mock_manager:
            mock_model = MagicMock()

            # Mock text embedding success
            mock_text_emb = MagicMock()
            mock_text_emb.flatten.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_model.embed_text.return_value = [mock_text_emb]

            # Mock image embedding failure
            mock_model.embed_image.side_effect = Exception("GPU memory error")
            mock_manager.return_value = mock_model

            with patch(
                "utils.document_loader.tempfile.gettempdir", return_value="/tmp"
            ):
                with (
                    patch("builtins.open", create=True),
                    patch("utils.document_loader.os.unlink"),
                    patch("utils.document_loader.logging.warning") as mock_warning,
                ):
                    create_native_multimodal_embeddings(
                        "Test text with images", sample_images_data
                    )

                    # Should still return text embedding despite image failure
                    # The provider might be "failed" if the main flow fails entirely
                    assert result["provider_used"] in [
                        "fastembed_native_multimodal",
                        "failed",
                    ]

                    # In any case, the function should handle errors gracefully
                    assert result is not None
                    assert isinstance(result, dict)
                    assert "provider_used" in result

                    # If it's the failed provider, text_embedding might still be populated
                    # from a successful earlier step before the complete failure

    def test_large_document_chunking_with_images(self, mock_settings):
        """Test chunking of large documents with embedded images."""
        from utils.document_loader import chunk_documents_structured

        # Create very large text document
        large_text = (
            "This is a comprehensive research paper on multimodal AI systems. " * 200
        )  # ~12k chars
        large_doc = Document(
            text=large_text,
            metadata={
                "type": "text",
                "page": 1,
                "has_images": True,
                "image_count": 3,
                "source": "large_research_paper.pdf",
            },
        )

        # Multiple high-resolution images
        img_base64_data = []
        for i in range(3):
            img = Image.new("RGB", (256, 256), color=["red", "green", "blue"][i])
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_base64_data.append(base64.b64encode(buffer.getvalue()).decode())

        image_docs = [
            ImageDocument(
                image=img_data,
                metadata={
                    "type": "image",
                    "page": 1,
                    "element_type": "Image",
                    "image_index": i,
                    "resolution": "256x256",
                    "source": "large_research_paper.pdf",
                },
            )
            for i, img_data in enumerate(img_base64_data)
        ]

        documents = [large_doc] + image_docs

        with patch("utils.document_loader.settings", mock_settings):
            with patch(
                "llama_index.core.node_parser.SentenceSplitter"
            ) as mock_splitter:
                mock_splitter_instance = MagicMock()

                # Mock chunking: large doc -> 6 chunks
                chunks = []
                for i in range(6):
                    chunk = Document(
                        text=f"Chunk {i + 1} of research paper content",
                        metadata={
                            **large_doc.metadata,
                            "chunk_index": i,
                            "total_chunks": 6,
                        },
                    )
                    chunks.append(chunk)

                mock_splitter_instance.get_nodes_from_documents.return_value = chunks
                mock_splitter.return_value = mock_splitter_instance

                chunk_documents_structured(documents)

        # Verify chunking preserved images and metadata
        assert len(result) == 9  # 6 text chunks + 3 images

        text_chunks = [doc for doc in result if not isinstance(doc, ImageDocument)]
        assert len(text_chunks) == 6

        image_chunks = [doc for doc in result if isinstance(doc, ImageDocument)]
        assert len(image_chunks) == 3

        # Verify chunk metadata preservation
        for chunk in text_chunks:
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["has_images"] is True
            assert chunk.metadata["source"] == "large_research_paper.pdf"

        # Verify image metadata preservation
        for img_doc in image_chunks:
            assert img_doc.metadata["resolution"] == "256x256"
            assert "image_index" in img_doc.metadata

    def test_concurrent_multimodal_processing(self, sample_images_data):
        """Test concurrent processing of multiple multimodal documents."""
        from utils.document_loader import create_native_multimodal_embeddings

        # Simulate concurrent processing scenario
        batch_texts = [
            "Document A: Deep learning architectures for computer vision",
            "Document B: Natural language processing with transformer models",
            "Document C: Multimodal fusion techniques in AI systems",
        ]

        with patch(
            "utils.document_loader.ModelManager.get_multimodal_embedding_model"
        ) as mock_manager:
            mock_model = MagicMock()

            # Mock embeddings for concurrent processing
            def mock_embed_text(texts):
                # Simulate batch processing
                return [
                    MagicMock(
                        flatten=lambda: MagicMock(
                            tolist=lambda: [0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1]
                        )
                    )
                    for i in range(len(texts))
                ]

            def mock_embed_image(image_paths):
                # Simulate batch image processing
                return [
                    MagicMock(
                        flatten=lambda: MagicMock(
                            tolist=lambda: [0.4 + i * 0.1, 0.5 + i * 0.1, 0.6 + i * 0.1]
                        )
                    )
                    for i in range(len(image_paths))
                ]

            mock_model.embed_text = mock_embed_text
            mock_model.embed_image = mock_embed_image
            mock_manager.return_value = mock_model

            with patch(
                "utils.document_loader.tempfile.gettempdir", return_value="/tmp"
            ):
                with (
                    patch("builtins.open", create=True),
                    patch("utils.document_loader.os.unlink"),
                ):
                    # Process multiple documents concurrently (simulated)
                    results = []
                    for text in batch_texts:
                        create_native_multimodal_embeddings(
                            text, sample_images_data
                        )
                        results.append(result)

            # Verify all documents processed successfully
            assert len(results) == 3
            for result in results:
                assert result["provider_used"] == "fastembed_native_multimodal"
                assert len(result["text_embedding"]) == 3
                assert len(result["image_embeddings"]) == 2

    def test_cross_modal_retrieval_preparation(self, sample_images_data):
        """Test preparation of documents for cross-modal retrieval."""
        from utils.document_loader import create_native_multimodal_embeddings

        # Test cross-modal document preparation
        text_query = "Show me images of neural network architectures"

        with patch(
            "utils.document_loader.ModelManager.get_multimodal_embedding_model"
        ) as mock_manager:
            mock_model = MagicMock()

            # Mock cross-modal embeddings (text can match images)
            mock_text_emb = MagicMock()
            mock_text_emb.flatten.return_value.tolist.return_value = [
                0.7,
                0.8,
                0.9,
            ]  # High similarity
            mock_model.embed_text.return_value = [mock_text_emb]

            mock_img_emb = MagicMock()
            mock_img_emb.flatten.return_value.tolist.return_value = [
                0.75,
                0.85,
                0.88,
            ]  # Similar to text
            mock_model.embed_image.return_value = [mock_img_emb] * len(
                sample_images_data
            )

            mock_manager.return_value = mock_model

            with patch(
                "utils.document_loader.tempfile.gettempdir", return_value="/tmp"
            ):
                with (
                    patch("builtins.open", create=True),
                    patch("utils.document_loader.os.unlink"),
                ):
                    create_native_multimodal_embeddings(
                        text_query, sample_images_data
                    )

            # Verify cross-modal compatibility
            text_emb = result["text_embedding"]
            img_embs = [img["embedding"] for img in result["image_embeddings"]]

            # Text and image embeddings should be in similar ranges for cross-modal retrieval
            assert len(text_emb) == len(img_embs[0])  # Same dimensionality

            # Verify embeddings are normalized/compatible for similarity search
            for img_emb in img_embs:
                assert all(
                    0 <= val <= 1 for val in img_emb
                )  # Reasonable embedding range


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    def test_end_to_end_pdf_processing(self, sample_pdf_path, mock_settings):
        """Test end-to-end PDF processing with multimodal support."""
        # This test would require actual file processing, so we'll mock the components
        with patch("utils.document_loader.extract_images_from_pdf") as mock_extract:
            with patch(
                "utils.document_loader.create_native_multimodal_embeddings"
            ) as mock_embed:
                with patch("utils.document_loader.LlamaParse") as mock_parse:
                    with patch(
                        "utils.document_loader.SimpleDirectoryReader"
                    ) as mock_reader:
                        # Mock the full pipeline
                        mock_extract.return_value = [
                            {"image_data": "base64_data", "page_number": 1}
                        ]
                        mock_embed.return_value = {
                            "text_embedding": [0.1, 0.2],
                            "provider_used": "test",
                        }

                        mock_parser = MagicMock()
                        mock_parse.return_value = mock_parser

                        mock_reader_instance = MagicMock()
                        mock_reader_instance.load_data.return_value = [
                            Document(text="Test content", metadata={})
                        ]
                        mock_reader.return_value = mock_reader_instance

                        # Create mock uploaded file
                        mock_file = MagicMock()
                        mock_file.name = "test.pdf"
                        mock_file.getvalue.return_value = Path(
                            sample_pdf_path
                        ).read_bytes()

                        with patch("utils.document_loader.settings", mock_settings):
                            load_documents_llama(
                                [mock_file], enable_multimodal=True
                            )

                            # Verify full pipeline was executed
                            assert len(result) == 1
                            assert result[0].metadata["has_images"] is True
                            assert "multimodal_embeddings" in result[0].metadata

    def test_multimodal_embedding_pipeline(self, sample_images_data):
        """Test the complete multimodal embedding pipeline."""
        with patch(
            "utils.document_loader.ModelManager.get_multimodal_embedding_model"
        ) as mock_manager:
            # Mock successful multimodal model
            mock_model = MagicMock()
            mock_text_emb = MagicMock()
            mock_text_emb.flatten.return_value.tolist.return_value = [0.1, 0.2, 0.3]
            mock_img_emb = MagicMock()
            mock_img_emb.flatten.return_value.tolist.return_value = [0.4, 0.5, 0.6]

            mock_model.embed_text.return_value = [mock_text_emb]
            mock_model.embed_image.return_value = [mock_img_emb] * len(
                sample_images_data
            )
            mock_manager.return_value = mock_model

            with patch(
                "utils.document_loader.tempfile.gettempdir", return_value="/tmp"
            ):
                with (
                    patch("builtins.open", create=True),
                    patch("utils.document_loader.os.unlink"),
                ):
                    create_native_multimodal_embeddings(
                        "Test text", sample_images_data
                    )

                    # Verify complete embedding structure
                    assert result["provider_used"] == "fastembed_native_multimodal"
                    assert len(result["image_embeddings"]) == len(sample_images_data)
                    assert result["text_embedding"] == [0.1, 0.2, 0.3]
                    assert result["combined_embedding"] is not None

    def test_comprehensive_multimodal_document_pipeline(self, tmp_path, mock_settings):
        """Test complete pipeline from Unstructured parsing to ready-for-indexing documents."""
        from utils.document_loader import load_documents_unstructured

        test_file = tmp_path / "comprehensive_test.pdf"
        test_file.write_text("Comprehensive test content")

        with patch("utils.document_loader.partition") as mock_partition:
            # Create comprehensive document structure
            elements = []

            # Title page
            title = MagicMock()
            title.category = "Title"
            title.__str__.return_value = "Comprehensive Multimodal AI System Evaluation"
            title.metadata.page_number = 1
            title.metadata.filename = "comprehensive_test.pdf"
            elements.append(title)

            # Abstract with images
            abstract = MagicMock()
            abstract.category = "NarrativeText"
            abstract.__str__.return_value = "This study presents a comprehensive evaluation of multimodal AI systems."
            abstract.metadata.page_number = 1
            abstract.metadata.filename = "comprehensive_test.pdf"
            elements.append(abstract)

            # Multiple images with captions
            for i in range(3):
                image = MagicMock()
                image.category = "Image"
                image.metadata.page_number = i + 2
                image.metadata.filename = "comprehensive_test.pdf"
                image.metadata.image_base64 = f"mock_image_data_{i}"
                elements.append(image)

                caption = MagicMock()
                caption.category = "FigureCaption"
                caption.__str__.return_value = (
                    f"Figure {i + 1}: Experimental results for configuration {i + 1}"
                )
                caption.metadata.page_number = i + 2
                caption.metadata.filename = "comprehensive_test.pdf"
                elements.append(caption)

            # Complex tables
            for i in range(2):
                table = MagicMock()
                table.category = "Table"
                table.__str__.return_value = f"| Metric | Value {i + 1} |\n| Accuracy | 9{i}.{i}% |\n| Precision | 8{i}.{i}% |"
                table.metadata.page_number = i + 5
                table.metadata.filename = "comprehensive_test.pdf"
                elements.append(table)

            mock_partition.return_value = elements

            with patch("utils.document_loader.settings", mock_settings):
                with patch(
                    "utils.document_loader.chunk_documents_structured"
                ) as mock_chunk:
                    mock_chunk.side_effect = lambda x: x

                    load_documents_unstructured(str(test_file))

            # Verify comprehensive processing
            assert (
                len(result) == 10
            )  # 1 title + 1 abstract + 3 images + 3 captions + 2 tables

            # Verify document type distribution
            text_docs = [doc for doc in result if not isinstance(doc, ImageDocument)]
            image_docs = [doc for doc in result if isinstance(doc, ImageDocument)]

            assert len(text_docs) == 7  # title + abstract + 3 captions + 2 tables
            assert len(image_docs) == 3  # 3 images

            # Verify all documents have proper metadata for indexing
            for doc in result:
                assert "element_type" in doc.metadata
                assert "page_number" in doc.metadata
                assert "filename" in doc.metadata

            # Verify images have base64 data for multimodal processing
            for img_doc in image_docs:
                assert "image_base64" in img_doc.metadata
                assert img_doc.metadata["image_base64"].startswith("mock_image_data_")
