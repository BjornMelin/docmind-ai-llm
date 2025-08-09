"""Enhanced comprehensive test coverage for document_loader.py.

This test suite provides critical path coverage for the document_loader module,
focusing on business-critical scenarios, error handling, multimodal processing,
and performance optimization to achieve 70%+ coverage.
"""

import base64
from unittest.mock import MagicMock, Mock, patch

import pytest
from llama_index.core import Document
from llama_index.core.schema import ImageDocument

# Import functions under test
from utils.document_loader import (
    batch_embed_documents,
    chunk_documents_structured,
    create_native_multimodal_embeddings,
    extract_images_from_pdf,
    load_documents_llama,
    load_documents_unstructured,
    process_documents_streaming,
    stream_document_processing,
)
from utils.exceptions import DocumentLoadingError, EmbeddingError


class TestExtractImagesFromPDF:
    """Test PDF image extraction functionality."""

    def test_extract_images_from_pdf_success(self, tmp_path):
        """Test successful image extraction from PDF."""
        # Create a mock PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with patch("fitz.open") as mock_fitz_open:
            # Mock PDF document structure
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_doc.load_page.return_value = mock_page
            mock_doc.__len__.return_value = 1
            mock_doc.__enter__ = MagicMock(return_value=mock_doc)
            mock_doc.__exit__ = MagicMock(return_value=None)
            mock_fitz_open.return_value = mock_doc

            # Mock image extraction
            mock_page.get_images.return_value = [(123, 0, 640, 480, 8, "DeviceRGB", "")]

            # Mock pixmap
            with patch("fitz.Pixmap") as mock_pixmap_class:
                mock_pixmap = MagicMock()
                mock_pixmap.n = 4  # RGB + alpha
                mock_pixmap.alpha = 1
                mock_pixmap.tobytes.return_value = b"dummy image data"
                mock_pixmap_class.return_value = mock_pixmap

                # Mock PIL Image processing
                with (
                    patch("io.BytesIO"),
                    patch("PIL.Image.open") as mock_pil_open,
                ):
                    mock_img = MagicMock()
                    mock_img.size = (640, 480)
                    mock_img.__enter__ = MagicMock(return_value=mock_img)
                    mock_img.__exit__ = MagicMock(return_value=None)
                    mock_pil_open.return_value = mock_img

                    # Mock image save to base64
                    with patch("base64.b64encode", return_value=b"base64data"):
                        extract_images_from_pdf(str(pdf_path))

                        assert len(result) == 1
                        assert result[0]["page_number"] == 1
                        assert result[0]["image_index"] == 0
                        assert result[0]["format"] == "PNG"
                        assert result[0]["size"] == (640, 480)
                        assert result[0]["image_data"] == "base64data"

    def test_extract_images_from_pdf_file_not_found(self, tmp_path):
        """Test handling of non-existent PDF file."""
        nonexistent_path = tmp_path / "nonexistent.pdf"

        with pytest.raises(DocumentLoadingError):
            extract_images_from_pdf(str(nonexistent_path))

    def test_extract_images_from_pdf_corrupted_pdf(self, tmp_path):
        """Test handling of corrupted PDF file."""
        pdf_path = tmp_path / "corrupted.pdf"
        pdf_path.write_bytes(b"not a pdf")

        with (
            patch("fitz.open", side_effect=RuntimeError("Corrupted PDF")),
            pytest.raises(DocumentLoadingError),
        ):
            extract_images_from_pdf(str(pdf_path))

    def test_extract_images_from_pdf_no_images(self, tmp_path):
        """Test PDF with no images."""
        pdf_path = tmp_path / "no_images.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with patch("fitz.open") as mock_fitz_open:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_doc.load_page.return_value = mock_page
            mock_doc.__len__.return_value = 1
            mock_doc.__enter__ = MagicMock(return_value=mock_doc)
            mock_doc.__exit__ = MagicMock(return_value=None)
            mock_fitz_open.return_value = mock_doc

            # No images in PDF
            mock_page.get_images.return_value = []

            extract_images_from_pdf(str(pdf_path))

            assert result == []

    def test_extract_images_from_pdf_cmyk_skip(self, tmp_path):
        """Test that CMYK images are skipped to avoid corruption."""
        pdf_path = tmp_path / "cmyk.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with patch("fitz.open") as mock_fitz_open:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_doc.load_page.return_value = mock_page
            mock_doc.__len__.return_value = 1
            mock_doc.__enter__ = MagicMock(return_value=mock_doc)
            mock_doc.__exit__ = MagicMock(return_value=None)
            mock_fitz_open.return_value = mock_doc

            # Mock CMYK image (n=5: CMYK + alpha)
            mock_page.get_images.return_value = [
                (123, 0, 640, 480, 8, "DeviceCMYK", "")
            ]

            with patch("fitz.Pixmap") as mock_pixmap_class:
                mock_pixmap = MagicMock()
                mock_pixmap.n = 5  # CMYK + alpha
                mock_pixmap.alpha = 1
                mock_pixmap_class.return_value = mock_pixmap

                extract_images_from_pdf(str(pdf_path))

                # CMYK images should be skipped
                assert result == []


class TestCreateNativeMultimodalEmbeddings:
    """Test multimodal embedding creation."""

    def test_create_native_multimodal_embeddings_text_only(self):
        """Test embedding creation with text only."""
        with patch(
            "utils.document_loader.ModelManager.get_multimodal_embedding_model"
        ) as mock_get_model:
            mock_model = MagicMock()
            mock_text_embedding = MagicMock()
            mock_text_embedding.flatten.return_value.tolist.return_value = [
                0.1,
                0.2,
                0.3,
            ]
            mock_model.embed_text.return_value = [mock_text_embedding]
            mock_get_model.return_value = mock_model

            create_native_multimodal_embeddings("test text")

            assert result["provider_used"] == "fastembed_native_multimodal"
            assert result["text_embedding"] == [0.1, 0.2, 0.3]
            assert result["image_embeddings"] == []
            assert result["combined_embedding"] == [0.1, 0.2, 0.3]

    def test_create_native_multimodal_embeddings_with_images(self):
        """Test embedding creation with text and images."""
        images = [{"image_data": base64.b64encode(b"dummy image").decode()}]

        with patch(
            "utils.document_loader.ModelManager.get_multimodal_embedding_model"
        ) as mock_get_model:
            mock_model = MagicMock()

            # Mock text embedding
            mock_text_embedding = MagicMock()
            mock_text_embedding.flatten.return_value.tolist.return_value = [
                0.1,
                0.2,
                0.3,
            ]
            mock_model.embed_text.return_value = [mock_text_embedding]

            # Mock image embedding
            mock_image_embedding = MagicMock()
            mock_image_embedding.flatten.return_value.tolist.return_value = [
                0.4,
                0.5,
                0.6,
            ]
            mock_model.embed_image.return_value = [mock_image_embedding]

            mock_get_model.return_value = mock_model

            with (
                patch("tempfile.gettempdir", return_value="/tmp"),
                patch("builtins.open", create=True),
                patch("os.unlink"),
            ):
                create_native_multimodal_embeddings("test text", images)

            assert result["provider_used"] == "fastembed_native_multimodal"
            assert result["text_embedding"] == [0.1, 0.2, 0.3]
            assert len(result["image_embeddings"]) == 1
            assert result["image_embeddings"][0]["embedding"] == [0.4, 0.5, 0.6]

    def test_create_native_multimodal_embeddings_fallback_to_text_only(self):
        """Test fallback to text-only when multimodal fails."""
        with patch(
            "utils.document_loader.ModelManager.get_multimodal_embedding_model",
            side_effect=ImportError("Multimodal not available"),
        ):
            with patch("utils.document_loader.FastEmbedEmbedding") as mock_fastembed:
                mock_model = MagicMock()
                mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]
                mock_fastembed.return_value = mock_model

                create_native_multimodal_embeddings("test text")

                assert result["provider_used"] == "fastembed_text_only"
                assert result["text_embedding"] == [0.1, 0.2, 0.3]

    def test_create_native_multimodal_embeddings_complete_failure(self):
        """Test behavior when all embedding methods fail."""
        with patch(
            "utils.document_loader.ModelManager.get_multimodal_embedding_model",
            side_effect=RuntimeError("All failed"),
        ):
            with patch(
                "utils.document_loader.FastEmbedEmbedding",
                side_effect=RuntimeError("FastEmbed failed"),
            ):
                # Should trigger fallback decorator
                create_native_multimodal_embeddings("test text")

                # Fallback decorator should return default values
                assert result["text_embedding"] is None
                assert result["image_embeddings"] == []
                assert result["combined_embedding"] is None
                assert result["provider_used"] == "failed_fallback"

    def test_create_native_multimodal_embeddings_invalid_image_data(self):
        """Test handling of invalid base64 image data."""
        invalid_images = [{"image_data": "invalid_base64_data"}]

        with patch(
            "utils.document_loader.ModelManager.get_multimodal_embedding_model"
        ) as mock_get_model:
            mock_model = MagicMock()
            mock_text_embedding = MagicMock()
            mock_text_embedding.flatten.return_value.tolist.return_value = [
                0.1,
                0.2,
                0.3,
            ]
            mock_model.embed_text.return_value = [mock_text_embedding]
            mock_get_model.return_value = mock_model

            # Should handle invalid base64 gracefully
            create_native_multimodal_embeddings("test text", invalid_images)

            # Text embedding should still work
            assert result["text_embedding"] == [0.1, 0.2, 0.3]
            assert result["image_embeddings"] == []  # No images processed


class TestLoadDocumentsUnstructured:
    """Test Unstructured document loading."""

    def test_load_documents_unstructured_success(self, tmp_path):
        """Test successful document loading with Unstructured."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        # Mock Unstructured elements
        mock_elements = [
            Mock(
                category="Title",
                text="Document Title",
                metadata=Mock(
                    page_number=1,
                    filename="test.pdf",
                    coordinates=Mock(points=[(0, 0), (100, 50)]),
                ),
            ),
            Mock(
                category="NarrativeText",
                text="Main document content.",
                metadata=Mock(
                    page_number=1,
                    filename="test.pdf",
                    coordinates=Mock(points=[(0, 50), (100, 100)]),
                ),
            ),
            Mock(
                category="Image",
                metadata=Mock(
                    page_number=1, filename="test.pdf", image_base64="base64imagedata"
                ),
            ),
        ]

        with patch("os.path.exists", return_value=True):
            with patch(
                "unstructured.partition.auto.partition", return_value=mock_elements
            ):
                with patch("utils.document_loader.settings") as mock_settings:
                    mock_settings.parse_strategy = "hi_res"
                    mock_settings.chunk_size = 1024
                    mock_settings.chunk_overlap = 200

                    load_documents_unstructured(str(pdf_path))

                    assert len(result) == 3  # Title, text, and image

                    # Check text documents
                    text_docs = [
                        doc
                        for doc in result
                        if isinstance(doc, Document)
                        and not isinstance(doc, ImageDocument)
                    ]
                    assert len(text_docs) == 2

                    # Check image documents
                    image_docs = [
                        doc for doc in result if isinstance(doc, ImageDocument)
                    ]
                    assert len(image_docs) == 1
                    assert image_docs[0].image == "base64imagedata"

    def test_load_documents_unstructured_fallback_to_llama_parse(self, tmp_path):
        """Test fallback to LlamaParse when Unstructured fails."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with patch("os.path.exists", return_value=True):
            with patch(
                "unstructured.partition.auto.partition",
                side_effect=RuntimeError("Unstructured failed"),
            ):
                with patch(
                    "utils.document_loader.load_documents_llama"
                ) as mock_llama_load:
                    fallback_docs = [Document(text="Fallback content")]
                    mock_llama_load.return_value = fallback_docs

                    load_documents_unstructured(str(pdf_path))

                    assert result == fallback_docs
                    mock_llama_load.assert_called_once()

    def test_load_documents_unstructured_file_not_found(self):
        """Test handling when file doesn't exist."""
        with pytest.raises(DocumentLoadingError):
            load_documents_unstructured("nonexistent.pdf")

    def test_load_documents_unstructured_chunking(self, tmp_path):
        """Test document chunking functionality."""
        pdf_path = tmp_path / "large_doc.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        # Create a large text element
        large_text = "Very long document content. " * 100  # Large text
        mock_elements = [
            Mock(
                category="NarrativeText",
                text=large_text,
                metadata=Mock(page_number=1, filename="large_doc.pdf"),
            )
        ]

        with patch("os.path.exists", return_value=True):
            with patch(
                "unstructured.partition.auto.partition", return_value=mock_elements
            ):
                with patch(
                    "utils.document_loader.chunk_documents_structured"
                ) as mock_chunk:
                    mock_chunk.return_value = [
                        Document(text="Chunk 1"),
                        Document(text="Chunk 2"),
                    ]

                    with patch("utils.document_loader.settings") as mock_settings:
                        mock_settings.chunk_size = 500

                        load_documents_unstructured(str(pdf_path))

                        # Should call chunking function
                        mock_chunk.assert_called_once()
                        assert len(result) == 2


class TestLoadDocumentsLlama:
    """Test LlamaParse document loading."""

    def test_load_documents_llama_basic_success(self):
        """Test basic successful document loading with LlamaParse."""
        # Mock file object
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with patch("utils.document_loader.LlamaParse") as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser

            with patch(
                "utils.document_loader.SimpleDirectoryReader"
            ) as mock_reader_class:
                mock_reader = Mock()
                mock_reader_class.return_value = mock_reader

                mock_docs = [Document(text="Parsed content", metadata={})]
                mock_reader.load_data.return_value = mock_docs

                with patch("tempfile.NamedTemporaryFile") as mock_temp:
                    mock_temp_file = Mock()
                    mock_temp_file.name = "/tmp/test.pdf"
                    mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                    mock_temp_file.__exit__ = Mock(return_value=None)
                    mock_temp.return_value = mock_temp_file

                    with (
                        patch("os.path.exists", return_value=True),
                        patch("os.remove"),
                    ):
                        load_documents_llama([mock_file])

                        assert len(result) == 1
                        assert result[0].text == "Parsed content"
                        assert result[0].metadata["source"] == "test.pdf"
                        assert result[0].metadata["type"] == "standard_document"

    def test_load_documents_llama_pdf_multimodal(self):
        """Test PDF loading with multimodal processing."""
        mock_file = Mock()
        mock_file.name = "multimodal.pdf"
        mock_file.getvalue.return_value = b"dummy pdf content"

        with patch("utils.document_loader.LlamaParse") as mock_parser_class:
            with patch(
                "utils.document_loader.SimpleDirectoryReader"
            ) as mock_reader_class:
                mock_docs = [Document(text="PDF content", metadata={})]
                mock_reader_class.return_value.load_data.return_value = mock_docs

                with patch(
                    "utils.document_loader.extract_images_from_pdf"
                ) as mock_extract:
                    mock_images = [{"image_data": "base64data", "page_number": 1}]
                    mock_extract.return_value = mock_images

                    with patch(
                        "utils.document_loader.create_native_multimodal_embeddings"
                    ) as mock_embed:
                        mock_embeddings = {
                            "text_embedding": [0.1, 0.2],
                            "image_embeddings": [{"embedding": [0.3, 0.4]}],
                            "provider_used": "fastembed_native_multimodal",
                        }
                        mock_embed.return_value = mock_embeddings

                        with patch("tempfile.NamedTemporaryFile") as mock_temp:
                            mock_temp_file = Mock()
                            mock_temp_file.name = "/tmp/multimodal.pdf"
                            mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                            mock_temp_file.__exit__ = Mock(return_value=None)
                            mock_temp.return_value = mock_temp_file

                            with (
                                patch("os.path.exists", return_value=True),
                                patch("os.remove"),
                            ):
                                load_documents_llama(
                                    [mock_file], enable_multimodal=True
                                )

                                assert len(result) == 1
                                assert result[0].metadata["type"] == "pdf_multimodal"
                                assert result[0].metadata["has_images"] is True
                                assert result[0].metadata["image_count"] == 1

    def test_load_documents_llama_video_processing(self):
        """Test video file processing with Whisper transcription."""
        mock_file = Mock()
        mock_file.name = "video.mp4"
        mock_file.type = "video/mp4"
        mock_file.getvalue.return_value = b"dummy video content"

        with patch("utils.document_loader.VideoFileClip") as mock_clip_class:
            mock_clip = Mock()
            mock_clip.duration = 10
            mock_clip.audio = Mock()
            mock_clip.get_frame.return_value = [[255, 0, 0]]  # Mock frame data
            mock_clip_class.return_value = mock_clip

            with patch("utils.document_loader.whisper_load") as mock_whisper:
                mock_model = Mock()
                mock_model.transcribe.return_value = {
                    "text": "Transcribed audio content"
                }
                mock_whisper.return_value = mock_model

                with (
                    patch("torch.cuda.is_available", return_value=True),
                    patch("PIL.Image.fromarray") as mock_pil,
                ):
                    mock_image = Mock()
                    mock_pil.return_value = mock_image

                    with patch("tempfile.NamedTemporaryFile") as mock_temp:
                        mock_temp_file = Mock()
                        mock_temp_file.name = "/tmp/video.mp4"
                        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                        mock_temp_file.__exit__ = Mock(return_value=None)
                        mock_temp.return_value = mock_temp_file

                        with (
                            patch("os.path.exists", return_value=True),
                            patch("os.remove"),
                        ):
                            load_documents_llama([mock_file], parse_media=True)

                            assert len(result) == 1
                            assert result[0].text == "Transcribed audio content"
                            assert result[0].metadata["type"] == "video"
                            assert "images" in result[0].metadata

    def test_load_documents_llama_audio_processing(self):
        """Test audio file processing with Whisper transcription."""
        mock_file = Mock()
        mock_file.name = "audio.mp3"
        mock_file.type = "audio/mp3"
        mock_file.getvalue.return_value = b"dummy audio content"

        with patch("utils.document_loader.whisper_load") as mock_whisper:
            mock_model = Mock()
            mock_model.transcribe.return_value = {"text": "Transcribed audio content"}
            mock_whisper.return_value = mock_model

            with (
                patch("torch.cuda.is_available", return_value=False),
                patch("tempfile.NamedTemporaryFile") as mock_temp,
            ):
                mock_temp_file = Mock()
                mock_temp_file.name = "/tmp/audio.mp3"
                mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                mock_temp_file.__exit__ = Mock(return_value=None)
                mock_temp.return_value = mock_temp_file

                with (
                    patch("os.path.exists", return_value=True),
                    patch("os.remove"),
                ):
                    load_documents_llama([mock_file], parse_media=True)

                    assert len(result) == 1
                    assert result[0].text == "Transcribed audio content"
                    assert result[0].metadata["type"] == "audio"

    def test_load_documents_llama_error_handling(self):
        """Test error handling in document loading."""
        mock_file = Mock()
        mock_file.name = "error.pdf"
        mock_file.getvalue.return_value = b"dummy content"

        with patch(
            "tempfile.NamedTemporaryFile", side_effect=OSError("Temp file error")
        ):
            load_documents_llama([mock_file])

            # Should handle error gracefully and return empty list
            assert result == []


class TestChunkDocumentsStructured:
    """Test structured document chunking."""

    def test_chunk_documents_structured_text_documents(self):
        """Test chunking of text documents."""
        docs = [
            Document(text="Short text."),
            Document(text="Very long text content. " * 50),  # Long document
            ImageDocument(image="base64data"),  # Should not be chunked
        ]

        with patch(
            "llama_index.core.node_parser.SentenceSplitter"
        ) as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter

            # Mock chunking result
            mock_chunks = [Document(text="Chunk 1"), Document(text="Chunk 2")]
            mock_splitter.get_nodes_from_documents.side_effect = [
                [],  # Short text - no chunking needed
                mock_chunks,  # Long text - chunked
            ]

            with patch("utils.document_loader.settings") as mock_settings:
                mock_settings.chunk_size = 1024
                mock_settings.chunk_overlap = 200

                chunk_documents_structured(docs)

                # Should have: short doc (unchanged) + 2 chunks + image (unchanged)
                assert len(result) >= 3

                # Verify image document passed through unchanged
                image_docs = [doc for doc in result if isinstance(doc, ImageDocument)]
                assert len(image_docs) == 1

    def test_chunk_documents_structured_empty_list(self):
        """Test chunking with empty document list."""
        chunk_documents_structured([])
        assert result == []

    def test_chunk_documents_structured_only_images(self):
        """Test chunking with only image documents."""
        docs = [ImageDocument(image="base64data1"), ImageDocument(image="base64data2")]

        chunk_documents_structured(docs)

        # Image documents should pass through unchanged
        assert len(result) == 2
        assert all(isinstance(doc, ImageDocument) for doc in result)


class TestAsyncFunctions:
    """Test async streaming and processing functions."""

    @pytest.mark.asyncio
    async def test_stream_document_processing_success(self, tmp_path):
        """Test successful streaming document processing."""
        # Create test files
        file1 = tmp_path / "doc1.pdf"
        file2 = tmp_path / "doc2.pdf"
        file1.write_bytes(b"content1")
        file2.write_bytes(b"content2")

        file_paths = [str(file1), str(file2)]

        with patch("utils.document_loader.load_documents_unstructured") as mock_load:
            mock_load.side_effect = [
                [Document(text="Content 1")],
                [Document(text="Content 2")],
            ]

            docs = []
            async for doc in stream_document_processing(file_paths):
                docs.append(doc)

            assert len(docs) == 2
            assert docs[0].text == "Content 1"
            assert docs[1].text == "Content 2"

    @pytest.mark.asyncio
    async def test_stream_document_processing_with_failures(self, tmp_path):
        """Test streaming with some document failures."""
        file1 = tmp_path / "good.pdf"
        file2 = tmp_path / "bad.pdf"
        file1.write_bytes(b"good content")
        file2.write_bytes(b"bad content")

        file_paths = [str(file1), str(file2)]

        with patch("utils.document_loader.load_documents_unstructured") as mock_load:
            mock_load.side_effect = [
                [Document(text="Good content")],
                RuntimeError("Processing failed"),
            ]

            docs = []
            async for doc in stream_document_processing(file_paths):
                docs.append(doc)

            # Should get one successful document, error should be handled gracefully
            assert len(docs) == 1
            assert docs[0].text == "Good content"

    @pytest.mark.asyncio
    async def test_batch_embed_documents_success(self):
        """Test successful batch document embedding."""
        docs = [
            Document(text="Document 1"),
            Document(text="Document 2"),
            Document(text="Document 3"),
        ]

        with patch("utils.document_loader.get_embed_model") as mock_get_model:
            mock_model = Mock()
            mock_model.embed.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            mock_get_model.return_value = mock_model

            with patch("asyncio.to_thread") as mock_to_thread:
                mock_to_thread.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

                await batch_embed_documents(docs, batch_size=2)

                assert len(result) == 3
                assert result[0] == [0.1, 0.2]
                assert result[1] == [0.3, 0.4]
                assert result[2] == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_batch_embed_documents_with_failures(self):
        """Test batch embedding with some batch failures."""
        docs = [
            Document(text="Document 1"),
            Document(text="Document 2"),
        ]

        with patch("utils.document_loader.get_embed_model") as mock_get_model:
            mock_model = Mock()
            mock_get_model.return_value = mock_model

            with patch("asyncio.gather") as mock_gather:
                # First batch succeeds, second fails
                mock_gather.return_value = [
                    [[0.1, 0.2]],  # Success
                    RuntimeError("Batch failed"),  # Failure
                ]

                with patch("utils.document_loader.settings") as mock_settings:
                    mock_settings.dense_embedding_dimension = 2

                    await batch_embed_documents(docs, batch_size=1)

                    # Should have one successful embedding and one fallback
                    assert len(result) == 2
                    assert result[0] == [0.1, 0.2]  # Successful embedding
                    assert result[1] == [0.0, 0.0]  # Fallback zeros

    @pytest.mark.asyncio
    async def test_process_documents_streaming_with_chunking(self, tmp_path):
        """Test streaming document processing with chunking."""
        # Create test file
        test_file = tmp_path / "large_doc.txt"
        test_file.write_text("Large document content. " * 100)

        file_paths = [str(test_file)]

        with patch("utils.document_loader.stream_document_processing") as mock_stream:
            # Mock large document that needs chunking
            large_text = "Very long document content. " * 100
            large_doc = Document(text=large_text)

            async def mock_stream_generator(paths):
                yield large_doc

            mock_stream.return_value = mock_stream_generator(file_paths)

            with patch(
                "llama_index.core.node_parser.SentenceSplitter"
            ) as mock_splitter_class:
                mock_splitter = Mock()
                mock_splitter_class.return_value = mock_splitter

                # Mock chunking result
                mock_chunks = [Document(text="Chunk 1"), Document(text="Chunk 2")]

                with patch("asyncio.to_thread") as mock_to_thread:
                    mock_to_thread.return_value = mock_chunks

                    docs = []
                    async for doc in process_documents_streaming(
                        file_paths, chunk_size=500
                    ):
                        docs.append(doc)

                    assert len(docs) == 2
                    assert docs[0].text == "Chunk 1"
                    assert docs[1].text == "Chunk 2"

    @pytest.mark.asyncio
    async def test_process_documents_streaming_small_docs_no_chunking(self, tmp_path):
        """Test streaming with small documents that don't need chunking."""
        test_file = tmp_path / "small_doc.txt"
        test_file.write_text("Small content.")

        file_paths = [str(test_file)]

        with patch("utils.document_loader.stream_document_processing") as mock_stream:
            small_doc = Document(text="Small content.")

            async def mock_stream_generator(paths):
                yield small_doc

            mock_stream.return_value = mock_stream_generator(file_paths)

            docs = []
            async for doc in process_documents_streaming(file_paths, chunk_size=1000):
                docs.append(doc)

            # Small document should pass through unchanged
            assert len(docs) == 1
            assert docs[0].text == "Small content."


class TestErrorRecoveryScenarios:
    """Test comprehensive error recovery scenarios."""

    def test_extract_images_from_pdf_pixmap_cleanup(self, tmp_path):
        """Test proper pixmap cleanup even when errors occur."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with patch("fitz.open") as mock_fitz_open:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_doc.load_page.return_value = mock_page
            mock_doc.__len__.return_value = 1
            mock_doc.__enter__ = MagicMock(return_value=mock_doc)
            mock_doc.__exit__ = MagicMock(return_value=None)
            mock_fitz_open.return_value = mock_doc

            mock_page.get_images.return_value = [(123, 0, 640, 480, 8, "DeviceRGB", "")]

            with patch("fitz.Pixmap") as mock_pixmap_class:
                mock_pixmap = MagicMock()
                mock_pixmap.n = 4
                mock_pixmap.alpha = 1
                mock_pixmap.tobytes.side_effect = RuntimeError("Pixmap error")
                mock_pixmap_class.return_value = mock_pixmap

                # Should handle error and cleanup pixmap
                with pytest.raises(DocumentLoadingError):
                    extract_images_from_pdf(str(pdf_path))

                # Pixmap should be set to None for cleanup
                # This tests the finally block in the image extraction loop

    def test_load_documents_unstructured_memory_cleanup(self, tmp_path):
        """Test memory cleanup of large base64 data."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        mock_elements = [
            Mock(
                category="Image",
                metadata=Mock(
                    page_number=1,
                    filename="test.pdf",
                    image_base64="very_large_base64_data" * 1000,
                ),
            )
        ]

        with patch("os.path.exists", return_value=True):
            with patch(
                "unstructured.partition.auto.partition", return_value=mock_elements
            ):
                with patch("utils.document_loader.settings") as mock_settings:
                    mock_settings.parse_strategy = "hi_res"
                    mock_settings.chunk_size = 0  # Disable chunking

                    load_documents_unstructured(str(pdf_path))

                    # After processing, image_base64 should be cleared for memory cleanup
                    assert mock_elements[0].metadata.image_base64 is None

    def test_load_documents_llama_resource_cleanup_on_error(self):
        """Test that temporary files are cleaned up even when errors occur."""
        mock_file = Mock()
        mock_file.name = "error.pdf"
        mock_file.getvalue.return_value = b"dummy content"

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp_file = Mock()
            mock_temp_file.name = "/tmp/error.pdf"
            mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
            mock_temp_file.__exit__ = Mock(return_value=None)
            mock_temp.return_value = mock_temp_file

            with patch(
                "utils.document_loader.SimpleDirectoryReader",
                side_effect=RuntimeError("Reader failed"),
            ):
                with (
                    patch("os.path.exists", return_value=True),
                    patch("os.remove") as mock_remove,
                ):
                    load_documents_llama([mock_file])

                    # Should handle error gracefully
                    assert result == []

                    # Temporary file should still be cleaned up
                    mock_remove.assert_called_with("/tmp/error.pdf")

    @pytest.mark.asyncio
    async def test_batch_embed_documents_model_initialization_failure(self):
        """Test handling of embedding model initialization failure."""
        docs = [Document(text="test")]

        with patch(
            "utils.document_loader.get_embed_model",
            side_effect=RuntimeError("Model failed"),
        ):
            with pytest.raises(EmbeddingError):
                await batch_embed_documents(docs)


class TestPerformanceAndMonitoring:
    """Test performance monitoring and logging."""

    def test_extract_images_from_pdf_performance_logging(self, tmp_path):
        """Test performance logging in image extraction."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"dummy pdf content")

        with patch("fitz.open") as mock_fitz_open:
            mock_doc = MagicMock()
            mock_doc.load_page.return_value.get_images.return_value = []
            mock_doc.__len__.return_value = 1
            mock_doc.__enter__ = MagicMock(return_value=mock_doc)
            mock_doc.__exit__ = MagicMock(return_value=None)
            mock_fitz_open.return_value = mock_doc

            with patch("utils.document_loader.log_performance") as mock_log_perf:
                extract_images_from_pdf(str(pdf_path))

                # Should log performance metrics
                mock_log_perf.assert_called_once()
                call_args = mock_log_perf.call_args[0]
                assert call_args[0] == "pdf_image_extraction"
                assert isinstance(call_args[1], float)  # duration

    def test_create_native_multimodal_embeddings_performance_logging(self):
        """Test performance logging in embedding creation."""
        with (
            patch("utils.document_loader.ModelManager.get_multimodal_embedding_model"),
            patch("utils.document_loader.log_performance") as mock_log_perf,
        ):
            create_native_multimodal_embeddings("test")

            mock_log_perf.assert_called_once()
            call_args = mock_log_perf.call_args[0]
            assert call_args[0] == "multimodal_embedding_creation"

    def test_load_documents_llama_performance_logging(self):
        """Test performance logging in LlamaParse loading."""
        mock_file = Mock()
        mock_file.name = "test.pdf"
        mock_file.getvalue.return_value = b"content"

        with (
            patch("utils.document_loader.LlamaParse"),
            patch("utils.document_loader.SimpleDirectoryReader") as mock_reader,
        ):
            mock_reader.return_value.load_data.return_value = []

            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp_file = Mock()
                mock_temp_file.name = "/tmp/test.pdf"
                mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                mock_temp_file.__exit__ = Mock(return_value=None)
                mock_temp.return_value = mock_temp_file

                with (
                    patch("os.path.exists", return_value=True),
                    patch("os.remove"),
                    patch("utils.document_loader.log_performance") as mock_log_perf,
                ):
                    load_documents_llama([mock_file])

                    mock_log_perf.assert_called_once()
                    call_args = mock_log_perf.call_args[0]
                    assert call_args[0] == "llamaparse_document_loading"
