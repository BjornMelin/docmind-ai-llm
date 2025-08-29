"""Comprehensive test suite for document utility functions.

This test suite covers the utility functions in utils/document.py that provide
convenient wrappers for document processing operations, including loading,
batch processing, caching, and knowledge graph functionality.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.processing import ProcessingResult, ProcessingStrategy
from src.processing.document_processor import ProcessingError
from src.utils.document import (
    clear_cache,
    clear_document_cache,
    clear_document_cache_sync,
    create_knowledge_graph_data,
    ensure_spacy_model,
    extract_entities_with_spacy,
    extract_relationships_with_spacy,
    get_cache_stats,
    get_cache_stats_sync,
    get_doc_info,
    get_document_info,
    load_documents_from_directory,
    load_documents_unstructured,
)


@pytest.fixture
def mock_processing_result():
    """Mock processing result for testing."""
    return ProcessingResult(
        elements=[],
        processing_time=1.5,
        strategy_used=ProcessingStrategy.FAST,
        metadata={"test": "data"},
        document_hash="test_hash",
    )


@pytest.fixture
def sample_directory_structure(tmp_path):
    """Create sample directory structure with various file types."""
    # Create subdirectories
    sub_dir = tmp_path / "subdirectory"
    sub_dir.mkdir()

    # Create files in main directory
    (tmp_path / "document1.pdf").write_text("PDF content 1")
    (tmp_path / "document2.txt").write_text("Text content 2")
    (tmp_path / "document3.docx").write_text("DOCX content 3")
    (tmp_path / "image.jpg").write_text("JPEG content")
    (tmp_path / "unsupported.xyz").write_text("Unsupported content")

    # Create files in subdirectory
    (sub_dir / "sub_document.pdf").write_text("Sub PDF content")
    (sub_dir / "sub_document.html").write_text("<html>Sub HTML content</html>")

    return tmp_path


@pytest.fixture
def mock_spacy_nlp():
    """Mock spaCy language model for testing."""
    nlp = Mock()

    # Mock doc processing
    mock_doc = Mock()

    # Mock entities
    mock_entity1 = Mock()
    mock_entity1.text = "Apple Inc."
    mock_entity1.label_ = "ORG"
    mock_entity1.start_char = 0
    mock_entity1.end_char = 10

    mock_entity2 = Mock()
    mock_entity2.text = "California"
    mock_entity2.label_ = "GPE"
    mock_entity2.start_char = 20
    mock_entity2.end_char = 30

    mock_doc.ents = [mock_entity1, mock_entity2]

    # Mock tokens for dependency parsing
    mock_token1 = Mock()
    mock_token1.dep_ = "ROOT"
    mock_token1.pos_ = "VERB"
    mock_token1.text = "founded"

    mock_subject = Mock()
    mock_subject.dep_ = "nsubj"
    mock_subject.text = "Apple"

    mock_object = Mock()
    mock_object.dep_ = "dobj"
    mock_object.text = "company"

    mock_token1.children = [mock_subject, mock_object]
    mock_doc.__iter__ = Mock(return_value=iter([mock_token1]))

    nlp.return_value = mock_doc
    return nlp


@pytest.mark.unit
class TestDocumentLoading:
    """Test document loading functions."""

    @pytest.mark.asyncio
    async def test_load_documents_unstructured_success(
        self, tmp_path, mock_processing_result
    ):
        """Test successful document loading."""
        test_files = [
            tmp_path / "test1.txt",
            tmp_path / "test2.pdf",
        ]

        for file in test_files:
            file.write_text("Test content")

        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_document_async = AsyncMock(
                return_value=mock_processing_result
            )
            mock_processor_class.return_value = mock_processor

            results = await load_documents_unstructured(test_files)

            assert len(results) == 2
            assert all(result == mock_processing_result for result in results)
            assert mock_processor.process_document_async.call_count == 2

    @pytest.mark.asyncio
    async def test_load_documents_unstructured_with_errors(self, tmp_path):
        """Test document loading with processing errors."""
        test_files = [
            tmp_path / "test1.txt",
            tmp_path / "test2.pdf",
            tmp_path / "test3.docx",
        ]

        for file in test_files:
            file.write_text("Test content")

        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()

            # First file succeeds, second fails, third succeeds
            async def mock_process(file_path):
                if "test2.pdf" in str(file_path):
                    raise ProcessingError("Processing failed")
                return ProcessingResult(
                    elements=[],
                    processing_time=1.0,
                    strategy_used=ProcessingStrategy.FAST,
                    metadata={},
                    document_hash="hash",
                )

            mock_processor.process_document_async = AsyncMock(side_effect=mock_process)
            mock_processor_class.return_value = mock_processor

            results = await load_documents_unstructured(test_files)

            # Should return 2 results (skipping the failed one)
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_load_documents_from_directory_recursive(
        self, sample_directory_structure, mock_processing_result
    ):
        """Test loading documents from directory recursively."""
        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_document_async = AsyncMock(
                return_value=mock_processing_result
            )
            mock_processor_class.return_value = mock_processor

            results = await load_documents_from_directory(
                sample_directory_structure, recursive=True
            )

            # Should find and process supported files (excluding .xyz)
            assert len(results) >= 5  # pdf, txt, docx, jpg, html files
            mock_processor.process_document_async.assert_called()

    @pytest.mark.asyncio
    async def test_load_documents_from_directory_non_recursive(
        self, sample_directory_structure, mock_processing_result
    ):
        """Test loading documents from directory non-recursively."""
        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_document_async = AsyncMock(
                return_value=mock_processing_result
            )
            mock_processor_class.return_value = mock_processor

            results = await load_documents_from_directory(
                sample_directory_structure, recursive=False
            )

            # Should only find files in root directory
            assert len(results) >= 4  # pdf, txt, docx, jpg files
            # Should be fewer than recursive version

    @pytest.mark.asyncio
    async def test_load_documents_from_directory_custom_extensions(
        self, sample_directory_structure, mock_processing_result
    ):
        """Test loading documents with custom file extensions."""
        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_document_async = AsyncMock(
                return_value=mock_processing_result
            )
            mock_processor_class.return_value = mock_processor

            # Only load PDF and TXT files
            custom_extensions = {".pdf", ".txt"}
            results = await load_documents_from_directory(
                sample_directory_structure,
                supported_extensions=custom_extensions,
                recursive=True,
            )

            # Should find PDF and TXT files only
            assert len(results) >= 2
            mock_processor.process_document_async.assert_called()


@pytest.mark.unit
class TestDocumentInfo:
    """Test document information functions."""

    def test_get_document_info_existing_file(self, tmp_path):
        """Test getting info for existing file."""
        test_file = tmp_path / "test_document.pdf"
        test_content = "Test PDF content"
        test_file.write_text(test_content)

        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.get_strategy_for_file.return_value = (
                ProcessingStrategy.HI_RES
            )
            mock_processor_class.return_value = mock_processor

            info = get_document_info(test_file)

            assert info["file_path"] == str(test_file)
            assert info["file_name"] == "test_document.pdf"
            assert info["file_extension"] == ".pdf"
            assert info["file_size_bytes"] == len(test_content.encode())
            assert info["is_readable"] is True
            assert info["supported"] is True
            assert info["processing_strategy"] == "hi_res"

    def test_get_document_info_non_existent_file(self):
        """Test getting info for non-existent file."""
        non_existent = Path("/non/existent/file.pdf")

        with pytest.raises(FileNotFoundError):
            get_document_info(non_existent)

    def test_get_document_info_unsupported_format(self, tmp_path):
        """Test getting info for unsupported file format."""
        test_file = tmp_path / "unsupported.xyz"
        test_file.write_text("Unsupported content")

        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.get_strategy_for_file.side_effect = ValueError(
                "Unsupported format"
            )
            mock_processor_class.return_value = mock_processor

            info = get_document_info(test_file)

            assert info["supported"] is False
            assert info["processing_strategy"] is None

    def test_get_doc_info_alias(self, tmp_path):
        """Test get_doc_info alias function."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.get_strategy_for_file.return_value = ProcessingStrategy.FAST
            mock_processor_class.return_value = mock_processor

            # get_doc_info is an alias for get_document_info, so it should return the actual result
            result = get_doc_info(test_file)

            # Verify it's actually calling the underlying function
            assert result["file_name"] == "test.txt"
            assert result["file_extension"] == ".txt"
            assert result["supported"] is True


@pytest.mark.unit
class TestCacheFunctions:
    """Test cache-related functions."""

    @pytest.mark.asyncio
    async def test_clear_document_cache_success(self):
        """Test successful cache clearing."""
        with patch("src.utils.document.SimpleCache") as mock_cache_class:
            mock_cache = Mock()
            mock_cache.clear_cache = AsyncMock(return_value=True)
            mock_cache_class.return_value = mock_cache

            result = await clear_document_cache()

            assert result is True
            mock_cache.clear_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_document_cache_failure(self):
        """Test cache clearing failure."""
        with patch("src.utils.document.SimpleCache") as mock_cache_class:
            mock_cache = Mock()
            mock_cache.clear_cache = AsyncMock(return_value=False)
            mock_cache_class.return_value = mock_cache

            result = await clear_document_cache()

            assert result is False

    def test_clear_document_cache_sync_wrapper(self):
        """Test synchronous cache clearing wrapper."""
        with patch(
            "src.utils.document._run_async_in_sync_context"
        ) as mock_async_wrapper:
            mock_async_wrapper.return_value = True

            result = clear_document_cache_sync()
            assert result is True

    def test_clear_cache_alias(self):
        """Test clear_cache alias function."""
        with patch("src.utils.document.clear_document_cache") as mock_clear:
            mock_clear.return_value = True

            # clear_cache is an alias for clear_document_cache (async function)
            # Since we're calling it in a sync context, it should reference the async function
            assert clear_cache is clear_document_cache

    @pytest.mark.asyncio
    async def test_get_cache_stats_success(self):
        """Test successful cache statistics retrieval."""
        with patch("src.utils.document.SimpleCache") as mock_cache_class:
            mock_cache = Mock()
            expected_stats = {
                "cache_type": "simple",
                "hits": 10,
                "misses": 5,
                "size": 15,
            }
            mock_cache.get_cache_stats = AsyncMock(return_value=expected_stats)
            mock_cache_class.return_value = mock_cache

            stats = await get_cache_stats()

            assert stats == expected_stats
            mock_cache.get_cache_stats.assert_called_once()

    def test_get_cache_stats_sync_wrapper(self):
        """Test synchronous cache stats wrapper."""
        expected_stats = {"test": "stats"}

        with patch(
            "src.utils.document._run_async_in_sync_context"
        ) as mock_async_wrapper:
            mock_async_wrapper.return_value = expected_stats

            result = get_cache_stats_sync()
            assert result == expected_stats


@pytest.mark.unit
class TestSpacyFunctions:
    """Test spaCy-related functions."""

    def test_ensure_spacy_model_success(self, mock_spacy_nlp):
        """Test successful spaCy model loading."""
        with patch("src.utils.document.get_spacy_manager") as mock_get_manager:
            mock_manager = Mock()
            mock_manager.ensure_model.return_value = mock_spacy_nlp
            mock_get_manager.return_value = mock_manager

            nlp = ensure_spacy_model("en_core_web_sm")

            assert nlp == mock_spacy_nlp
            mock_manager.ensure_model.assert_called_once_with("en_core_web_sm")

    def test_extract_entities_with_spacy_success(self, mock_spacy_nlp):
        """Test successful entity extraction with spaCy."""
        test_text = "Apple Inc. is located in California."

        entities = extract_entities_with_spacy(test_text, mock_spacy_nlp)

        assert len(entities) == 2

        # Check first entity
        assert entities[0]["text"] == "Apple Inc."
        assert entities[0]["label"] == "ORG"
        assert entities[0]["start"] == 0
        assert entities[0]["end"] == 10

        # Check second entity
        assert entities[1]["text"] == "California"
        assert entities[1]["label"] == "GPE"
        assert entities[1]["start"] == 20
        assert entities[1]["end"] == 30

    def test_extract_entities_without_nlp_model(self):
        """Test entity extraction without providing nlp model."""
        with patch("src.utils.document.ensure_spacy_model") as mock_ensure:
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_doc.ents = []
            mock_nlp.return_value = mock_doc
            mock_ensure.return_value = mock_nlp

            entities = extract_entities_with_spacy("Test text")

            assert entities == []
            mock_ensure.assert_called_once()

    def test_extract_entities_error_handling(self):
        """Test entity extraction error handling."""
        mock_nlp = Mock()
        mock_nlp.side_effect = ValueError("spaCy processing failed")

        entities = extract_entities_with_spacy("Test text", mock_nlp)

        assert entities == []

    def test_extract_relationships_with_spacy_success(self, mock_spacy_nlp):
        """Test successful relationship extraction with spaCy."""
        test_text = "Apple founded the company."

        relationships = extract_relationships_with_spacy(test_text, mock_spacy_nlp)

        assert len(relationships) == 1
        assert relationships[0]["subject"] == "Apple"
        assert relationships[0]["predicate"] == "founded"
        assert relationships[0]["object"] == "company"

    def test_extract_relationships_without_nlp_model(self):
        """Test relationship extraction without providing nlp model."""
        with patch("src.utils.document.ensure_spacy_model") as mock_ensure:
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_doc.__iter__ = Mock(return_value=iter([]))
            mock_nlp.return_value = mock_doc
            mock_ensure.return_value = mock_nlp

            relationships = extract_relationships_with_spacy("Test text")

            assert relationships == []
            mock_ensure.assert_called_once()

    def test_extract_relationships_error_handling(self):
        """Test relationship extraction error handling."""
        mock_nlp = Mock()
        mock_nlp.side_effect = OSError("spaCy model not found")

        relationships = extract_relationships_with_spacy("Test text", mock_nlp)

        assert relationships == []

    def test_create_knowledge_graph_data_success(self, mock_spacy_nlp):
        """Test successful knowledge graph data creation."""
        test_text = "Apple Inc. is located in California."

        with (
            patch(
                "src.utils.document.extract_entities_with_spacy"
            ) as mock_extract_entities,
            patch(
                "src.utils.document.extract_relationships_with_spacy"
            ) as mock_extract_relations,
        ):
            mock_entities = [
                {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10}
            ]
            mock_relationships = [
                {"subject": "Apple", "predicate": "located", "object": "California"}
            ]

            mock_extract_entities.return_value = mock_entities
            mock_extract_relations.return_value = mock_relationships

            kg_data = create_knowledge_graph_data(test_text, mock_spacy_nlp)

            assert kg_data["entities"] == mock_entities
            assert kg_data["relationships"] == mock_relationships
            assert kg_data["metadata"]["entity_count"] == 1
            assert kg_data["metadata"]["relationship_count"] == 1
            assert kg_data["metadata"]["text_length"] == len(test_text)
            assert kg_data["metadata"]["processing_method"] == "spacy_knowledge_graph"

    def test_create_knowledge_graph_data_error_handling(self):
        """Test knowledge graph creation error handling."""
        with patch(
            "src.utils.document.extract_entities_with_spacy"
        ) as mock_extract_entities:
            mock_extract_entities.side_effect = ValueError("Processing failed")

            kg_data = create_knowledge_graph_data("Test text")

            assert kg_data["entities"] == []
            assert kg_data["relationships"] == []
            assert "error" in kg_data["metadata"]


@pytest.mark.unit
class TestAsyncSyncWrappers:
    """Test async/sync context wrapper functions."""

    def test_run_async_in_sync_context_no_loop(self):
        """Test running async function when no event loop exists."""

        async def mock_async_func():
            return "test_result"

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("No event loop")

            with patch("asyncio.run") as mock_run:
                mock_run.return_value = "test_result"

                from src.utils.document import _run_async_in_sync_context

                result = _run_async_in_sync_context(mock_async_func())

                assert result == "test_result"
                mock_run.assert_called_once()

    def test_run_async_in_sync_context_running_loop(self):
        """Test running async function when loop is already running."""

        async def mock_async_func():
            return "test_result"

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = Mock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop

            from src.utils.document import _run_async_in_sync_context

            result = _run_async_in_sync_context(mock_async_func())

            # Should return None when loop is running
            assert result is None

    def test_run_async_in_sync_context_with_loop(self):
        """Test running async function with existing stopped loop."""

        async def mock_async_func():
            return "test_result"

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = Mock()
            mock_loop.is_running.return_value = False
            mock_loop.run_until_complete.return_value = "test_result"
            mock_get_loop.return_value = mock_loop

            from src.utils.document import _run_async_in_sync_context

            result = _run_async_in_sync_context(mock_async_func())

            assert result == "test_result"
            mock_loop.run_until_complete.assert_called_once()


@pytest.mark.unit
class TestDocumentUtilitiesEdgeCases:
    """Test edge cases and boundary conditions for document utilities."""

    @pytest.mark.parametrize(
        ("filename", "expected_valid"),
        [
            ("file.txt", True),  # Valid file
            ("file.PDF", True),  # Uppercase extension
            ("file with spaces.docx", True),  # Spaces in name
            ("very_long_filename_" + "x" * 100 + ".txt", True),  # Long name
        ],
    )
    def test_get_document_info_filename_edge_cases(
        self, tmp_path, filename, expected_valid
    ):
        """Test document info with various filename edge cases."""
        test_file = tmp_path / filename
        test_file.write_text("Test content")

        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()

            if expected_valid and test_file.suffix.lower() in [".txt", ".pdf", ".docx"]:
                mock_processor.get_strategy_for_file.return_value = (
                    ProcessingStrategy.FAST
                )
            else:
                mock_processor.get_strategy_for_file.side_effect = ValueError(
                    "Unsupported"
                )

            mock_processor_class.return_value = mock_processor

            info = get_document_info(test_file)
            assert info["supported"] == (
                expected_valid and test_file.suffix.lower() in [".txt", ".pdf", ".docx"]
            )

    @pytest.mark.parametrize(
        ("file_size", "expected_readable"),
        [
            (0, False),  # Empty file - not readable
            (1, True),  # 1 byte
            (1024, True),  # 1 KB
            (1048576, True),  # 1 MB
        ],
    )
    def test_get_document_info_file_size_edge_cases(
        self, tmp_path, file_size, expected_readable
    ):
        """Test document info with various file sizes."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"x" * file_size)

        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.get_strategy_for_file.return_value = ProcessingStrategy.FAST
            mock_processor_class.return_value = mock_processor

            info = get_document_info(test_file)
            assert info["file_size_bytes"] == file_size
            assert info["is_readable"] == expected_readable

    @pytest.mark.asyncio
    async def test_load_documents_unstructured_empty_file_list(self):
        """Test loading documents with empty file list."""
        results = await load_documents_unstructured([])
        assert results == []

    @pytest.mark.asyncio
    async def test_load_documents_unstructured_duplicate_files(self, tmp_path):
        """Test loading documents with duplicate file paths."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        duplicate_files = [test_file, test_file, test_file]  # Same file 3 times

        mock_result = ProcessingResult(
            elements=[],
            processing_time=1.0,
            strategy_used=ProcessingStrategy.FAST,
            metadata={},
            document_hash="hash",
        )

        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_document_async = AsyncMock(return_value=mock_result)
            mock_processor_class.return_value = mock_processor

            results = await load_documents_unstructured(duplicate_files)

            # Should process all instances, even duplicates
            assert len(results) == 3
            assert mock_processor.process_document_async.call_count == 3

    @pytest.mark.parametrize(
        ("text", "expected_entity_count"),
        [
            ("", 0),  # Empty text
            ("   ", 0),  # Whitespace only
            ("No entities here", 0),  # Text without entities
            ("A" * 1000, 0),  # Very long text
        ],
    )
    def test_extract_entities_with_spacy_edge_cases(self, text, expected_entity_count):
        """Test entity extraction with edge case text inputs."""
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc

        entities = extract_entities_with_spacy(text, mock_nlp)
        assert len(entities) == expected_entity_count

    def test_ensure_spacy_model_error_handling(self):
        """Test spaCy model loading error scenarios."""
        with patch("src.utils.document.get_spacy_manager") as mock_get_manager:
            mock_manager = Mock()
            mock_manager.ensure_model.side_effect = OSError("Model not found")
            mock_get_manager.return_value = mock_manager

            # Should raise the original exception
            with pytest.raises(OSError):
                ensure_spacy_model("nonexistent_model")


@pytest.mark.integration
class TestDocumentUtilitiesIntegration:
    """Integration tests for document utilities with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_full_document_processing_workflow(self, tmp_path):
        """Integration test for complete document processing workflow."""
        # Create test files
        test_files = []
        for i in range(3):
            file_path = tmp_path / f"document_{i}.txt"
            file_path.write_text(f"This is test document {i} with some content.")
            test_files.append(file_path)

        # Mock processing results
        mock_results = []
        for i, _file_path in enumerate(test_files):
            mock_result = ProcessingResult(
                elements=[],
                processing_time=0.5 + i * 0.1,
                strategy_used=ProcessingStrategy.FAST,
                metadata={"file_index": i},
                document_hash=f"hash_{i}",
            )
            mock_results.append(mock_result)

        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_document_async = AsyncMock(side_effect=mock_results)
            mock_processor_class.return_value = mock_processor

            # Test batch loading
            results = await load_documents_unstructured(test_files)

            # Verify results
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.metadata["file_index"] == i
                assert result.processing_time > 0

    def test_document_info_batch_analysis(self, sample_directory_structure):
        """Test batch document information analysis."""
        # Get all files from directory
        all_files = list(sample_directory_structure.rglob("*"))
        document_files = [f for f in all_files if f.is_file()]

        with patch("src.utils.document.DocumentProcessor") as mock_processor_class:
            mock_processor = Mock()

            def mock_strategy_selection(file_path):
                ext = Path(file_path).suffix.lower()
                if ext in [".pdf", ".docx"]:
                    return ProcessingStrategy.HI_RES
                elif ext in [".txt", ".html"]:
                    return ProcessingStrategy.FAST
                elif ext in [".jpg"]:
                    return ProcessingStrategy.OCR_ONLY
                else:
                    raise ValueError("Unsupported format")

            mock_processor.get_strategy_for_file = Mock(
                side_effect=mock_strategy_selection
            )
            mock_processor_class.return_value = mock_processor

            # Analyze each file
            file_info = []
            for file_path in document_files:
                try:
                    info = get_document_info(file_path)
                    file_info.append(info)
                except FileNotFoundError:
                    # Skip directories
                    continue

            # Verify analysis
            supported_files = [
                info for info in file_info if info.get("supported", False)
            ]
            assert len(supported_files) > 0

            # Check that different strategies are assigned
            strategies = {info["processing_strategy"] for info in supported_files}
            assert len(strategies) >= 2  # Should have multiple strategies

    @pytest.mark.asyncio
    async def test_cache_operations_workflow(self):
        """Integration test for cache operations workflow."""
        mock_stats_before = {"cache_type": "simple", "hits": 5, "misses": 3, "size": 8}

        mock_stats_after = {"cache_type": "simple", "hits": 0, "misses": 0, "size": 0}

        with patch("src.utils.document.SimpleCache") as mock_cache_class:
            mock_cache = Mock()

            # First call returns stats before clearing
            # Second call returns stats after clearing
            mock_cache.get_cache_stats = AsyncMock(
                side_effect=[mock_stats_before, mock_stats_after]
            )
            mock_cache.clear_cache = AsyncMock(return_value=True)
            mock_cache_class.return_value = mock_cache

            # Get initial stats
            initial_stats = await get_cache_stats()
            assert initial_stats == mock_stats_before
            assert initial_stats["size"] == 8

            # Clear cache
            clear_result = await clear_document_cache()
            assert clear_result is True

            # Get stats after clearing
            final_stats = await get_cache_stats()
            assert final_stats == mock_stats_after
            assert final_stats["size"] == 0

    def test_knowledge_graph_integration(self, mock_spacy_nlp):
        """Integration test for knowledge graph extraction workflow."""
        test_text = """
        Apple Inc. is an American multinational technology company based in California.
        The company was founded by Steve Jobs and Steve Wozniak in 1976.
        Apple designs and manufactures consumer electronics and software.
        """

        # Create more realistic mock entities and relationships
        mock_entities = [
            {
                "text": "Apple Inc.",
                "label": "ORG",
                "start": 9,
                "end": 19,
                "confidence": 1.0,
            },
            {
                "text": "American",
                "label": "NORP",
                "start": 26,
                "end": 34,
                "confidence": 0.9,
            },
            {
                "text": "California",
                "label": "GPE",
                "start": 83,
                "end": 93,
                "confidence": 0.95,
            },
            {
                "text": "Steve Jobs",
                "label": "PERSON",
                "start": 127,
                "end": 137,
                "confidence": 1.0,
            },
            {
                "text": "Steve Wozniak",
                "label": "PERSON",
                "start": 142,
                "end": 155,
                "confidence": 1.0,
            },
            {
                "text": "1976",
                "label": "DATE",
                "start": 159,
                "end": 163,
                "confidence": 0.8,
            },
        ]

        mock_relationships = [
            {"subject": "Apple", "predicate": "founded", "object": "company"},
            {"subject": "Apple", "predicate": "designs", "object": "electronics"},
            {"subject": "Apple", "predicate": "manufactures", "object": "software"},
        ]

        with (
            patch(
                "src.utils.document.extract_entities_with_spacy"
            ) as mock_extract_entities,
            patch(
                "src.utils.document.extract_relationships_with_spacy"
            ) as mock_extract_relations,
        ):
            mock_extract_entities.return_value = mock_entities
            mock_extract_relations.return_value = mock_relationships

            kg_data = create_knowledge_graph_data(test_text, mock_spacy_nlp)

            # Verify comprehensive knowledge graph data
            assert len(kg_data["entities"]) == 6
            assert len(kg_data["relationships"]) == 3

            # Check metadata completeness
            metadata = kg_data["metadata"]
            assert metadata["entity_count"] == 6
            assert metadata["relationship_count"] == 3
            assert metadata["text_length"] == len(test_text)
            assert metadata["processing_method"] == "spacy_knowledge_graph"

            # Verify entity types are diverse
            entity_labels = {entity["label"] for entity in kg_data["entities"]}
            assert len(entity_labels) >= 3  # Should have multiple entity types
