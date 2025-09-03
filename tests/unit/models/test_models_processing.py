"""Unit tests for Pydantic models in processing.py.

Covers document processing strategies, elements, results, and hash generation.
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models.processing import (
    DocumentElement,
    ProcessingError,
    ProcessingResult,
    ProcessingStrategy,
)

# --- merged from test_models_processing_coverage.py ---


class TestProcessingResultHashMethodCoverage:
    """Additional coverage for ProcessingResult.create_hash_for_document (merged)."""

    @pytest.mark.unit
    def test_create_hash_for_document_path_conversion(self):
        """Test hash creation for document path conversion."""
        test_content = b"Path conversion test content"
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(test_content)
            p = tmp.name
        try:
            h1 = ProcessingResult.create_hash_for_document(p)
            h2 = ProcessingResult.create_hash_for_document(Path(p))
            assert isinstance(h1, str)
            assert len(h1) == 64
            assert h1 == h2
        finally:
            Path(p).unlink()

    @pytest.mark.unit
    def test_create_hash_for_document_chunked_reading(self):
        """Test hash creation for document chunked reading."""
        import hashlib
        import tempfile
        from pathlib import Path

        large = b"x" * (8192 * 3 + 1000)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(large)
            p = tmp.name
        try:
            got = ProcessingResult.create_hash_for_document(p)
            manual = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    manual.update(chunk)
            stat = Path(p).stat()
            meta = f"{Path(p).name}:{stat.st_size}:{stat.st_mtime}".encode()
            manual.update(meta)
            assert got == manual.hexdigest()
        finally:
            Path(p).unlink()


class TestProcessingStrategy:
    """Test suite for ProcessingStrategy enum."""

    @pytest.mark.unit
    def test_processing_strategy_values(self):
        """Test ProcessingStrategy enum values."""
        assert ProcessingStrategy.HI_RES == "hi_res"
        assert ProcessingStrategy.FAST == "fast"
        assert ProcessingStrategy.OCR_ONLY == "ocr_only"

    @pytest.mark.unit
    def test_processing_strategy_string_behavior(self):
        """Test ProcessingStrategy string behavior."""
        strategy = ProcessingStrategy.HI_RES
        # The __str__ method may return the full enum representation
        assert strategy.value == "hi_res"
        assert strategy == "hi_res"  # Equality comparison should work

    @pytest.mark.unit
    def test_processing_strategy_equality(self):
        """Test ProcessingStrategy equality comparisons."""
        strategy1 = ProcessingStrategy.HI_RES
        strategy2 = ProcessingStrategy.HI_RES
        strategy3 = ProcessingStrategy.FAST

        assert strategy1 == strategy2
        assert strategy1 != strategy3
        assert strategy1 == "hi_res"
        assert strategy1 != "fast"

    @pytest.mark.unit
    def test_processing_strategy_iteration(self):
        """Test ProcessingStrategy can be iterated."""
        strategies = list(ProcessingStrategy)
        assert len(strategies) == 3
        assert ProcessingStrategy.HI_RES in strategies
        assert ProcessingStrategy.FAST in strategies
        assert ProcessingStrategy.OCR_ONLY in strategies

    @pytest.mark.unit
    def test_processing_strategy_creation_from_string(self):
        """Test ProcessingStrategy creation from string values."""
        assert ProcessingStrategy("hi_res") == ProcessingStrategy.HI_RES
        assert ProcessingStrategy("fast") == ProcessingStrategy.FAST
        assert ProcessingStrategy("ocr_only") == ProcessingStrategy.OCR_ONLY

    @pytest.mark.unit
    def test_processing_strategy_invalid_value(self):
        """Test ProcessingStrategy with invalid value."""
        with pytest.raises(ValueError, match="invalid_strategy") as exc_info:
            ProcessingStrategy("invalid_strategy")

        assert "invalid_strategy" in str(exc_info.value)

    @pytest.mark.unit
    def test_processing_strategy_in_pydantic_model(self):
        """Test ProcessingStrategy used in Pydantic model validation."""
        # Valid strategy
        result = ProcessingResult(
            elements=[],
            processing_time=1.0,
            strategy_used=ProcessingStrategy.HI_RES,
            document_hash="test_hash",
        )
        assert result.strategy_used == ProcessingStrategy.HI_RES

        # Valid strategy from string
        result2 = ProcessingResult(
            elements=[],
            processing_time=1.0,
            strategy_used="fast",
            document_hash="test_hash",
        )
        assert result2.strategy_used == ProcessingStrategy.FAST

    @pytest.mark.unit
    def test_processing_strategy_serialization(self):
        """Test ProcessingStrategy serialization behavior."""
        result = ProcessingResult(
            elements=[],
            processing_time=1.0,
            strategy_used=ProcessingStrategy.OCR_ONLY,
            document_hash="test_hash",
        )

        json_data = result.model_dump()
        assert json_data["strategy_used"] == "ocr_only"

        # Deserialize
        restored = ProcessingResult.model_validate(json_data)
        assert restored.strategy_used == ProcessingStrategy.OCR_ONLY


class TestDocumentElement:
    """Test suite for DocumentElement model."""

    @pytest.mark.unit
    def test_document_element_creation_basic(self):
        """Test DocumentElement creation with basic data."""
        element = DocumentElement(
            text="This is a paragraph of text.",
            category="NarrativeText",
            metadata={"page_number": 1, "bbox": [100, 200, 300, 400]},
        )

        assert element.text == "This is a paragraph of text."
        assert element.category == "NarrativeText"
        assert element.metadata["page_number"] == 1
        assert element.metadata["bbox"] == [100, 200, 300, 400]

    @pytest.mark.unit
    def test_document_element_creation_minimal(self):
        """Test DocumentElement creation with minimal data."""
        element = DocumentElement(text="Title", category="Title")

        assert element.text == "Title"
        assert element.category == "Title"
        assert element.metadata == {}

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "category",
        [
            "Title",
            "NarrativeText",
            "Table",
            "Image",
            "ListItem",
            "Header",
            "Footer",
            "Formula",
            "FigureCaption",
            "PageBreak",
            "Address",
        ],
    )
    def test_document_element_common_categories(self, category):
        """Test DocumentElement with common unstructured.io categories."""
        element = DocumentElement(text="Test content", category=category)
        assert element.category == category

    @pytest.mark.unit
    def test_document_element_empty_text(self):
        """Test DocumentElement with empty text."""
        element = DocumentElement(text="", category="Image")
        assert element.text == ""

    @pytest.mark.unit
    def test_document_element_large_text(self):
        """Test DocumentElement with large text content."""
        large_text = "Lorem ipsum " * 1000  # ~11KB text
        element = DocumentElement(text=large_text, category="NarrativeText")
        assert len(element.text) > 10000

    @pytest.mark.unit
    def test_document_element_unicode_text(self):
        """Test DocumentElement with unicode content."""
        element = DocumentElement(
            text="文档内容 with émojis 🚀 and symbols: α, β, γ",
            category="NarrativeText",
            metadata={"language": "multi"},
        )

        assert "文档内容" in element.text
        assert "🚀" in element.text
        assert element.metadata["language"] == "multi"

    @pytest.mark.unit
    def test_document_element_complex_metadata(self):
        """Test DocumentElement with complex metadata structures."""
        complex_metadata = {
            "coordinates": {
                "bbox": {"x": 100, "y": 200, "width": 300, "height": 50},
                "page_dimensions": {"width": 612, "height": 792},
            },
            "text_analysis": {
                "word_count": 25,
                "confidence_scores": [0.9, 0.8, 0.95, 0.7],
                "font_info": {"family": "Arial", "size": 12, "bold": False},
            },
            "extraction_info": {
                "method": "ocr",
                "model_version": "v2.1",
                "processing_time": 0.15,
            },
        }

        element = DocumentElement(
            text="Complex document element",
            category="NarrativeText",
            metadata=complex_metadata,
        )

        assert element.metadata["coordinates"]["bbox"]["x"] == 100
        assert element.metadata["text_analysis"]["word_count"] == 25
        assert element.metadata["extraction_info"]["method"] == "ocr"

    @pytest.mark.unit
    def test_document_element_serialization(self):
        """Test DocumentElement serialization and deserialization."""
        original = DocumentElement(
            text="Serialization test content",
            category="Header",
            metadata={"level": 2, "style": "bold", "tags": ["important", "chapter"]},
        )

        # Serialize and deserialize
        json_data = original.model_dump()
        restored = DocumentElement.model_validate(json_data)

        assert restored.text == original.text
        assert restored.category == original.category
        assert restored.metadata == original.metadata

    @pytest.mark.unit
    def test_document_element_validation_types(self):
        """Test DocumentElement validation with invalid types."""
        with pytest.raises(ValidationError):
            DocumentElement(text=123, category="Title")  # Invalid text type

        with pytest.raises(ValidationError):
            DocumentElement(text="Valid", category=None)  # Invalid category type

        with pytest.raises(ValidationError):
            DocumentElement(
                text="Valid", category="Title", metadata="invalid"
            )  # Invalid metadata type


class TestProcessingResult:
    """Test suite for ProcessingResult model."""

    @pytest.mark.unit
    def test_processing_result_creation_basic(self):
        """Test ProcessingResult creation with basic data."""
        elements = [
            DocumentElement(text="Title", category="Title"),
            DocumentElement(text="Content", category="NarrativeText"),
        ]

        result = ProcessingResult(
            elements=elements,
            processing_time=2.5,
            strategy_used=ProcessingStrategy.HI_RES,
            document_hash="abc123def456",
            metadata={"total_pages": 5, "file_size": 1024000},
        )

        assert len(result.elements) == 2
        assert result.processing_time == 2.5
        assert result.strategy_used == ProcessingStrategy.HI_RES
        assert result.document_hash == "abc123def456"
        assert result.metadata["total_pages"] == 5

    @pytest.mark.unit
    def test_processing_result_empty_elements(self):
        """Test ProcessingResult with empty elements list."""
        result = ProcessingResult(
            elements=[],
            processing_time=0.1,
            strategy_used=ProcessingStrategy.FAST,
            document_hash="empty_doc_hash",
        )

        assert result.elements == []
        assert result.processing_time == 0.1

    @pytest.mark.unit
    def test_processing_result_large_elements_list(self):
        """Test ProcessingResult with large elements list."""
        elements = [
            DocumentElement(text=f"Element {i}", category="NarrativeText")
            for i in range(1000)
        ]

        result = ProcessingResult(
            elements=elements,
            processing_time=15.0,
            strategy_used=ProcessingStrategy.HI_RES,
            document_hash="large_doc_hash",
        )

        assert len(result.elements) == 1000
        assert result.elements[999].text == "Element 999"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "strategy",
        [
            ProcessingStrategy.HI_RES,
            ProcessingStrategy.FAST,
            ProcessingStrategy.OCR_ONLY,
        ],
    )
    def test_processing_result_all_strategies(self, strategy):
        """Test ProcessingResult with all processing strategies."""
        result = ProcessingResult(
            elements=[DocumentElement(text="Test", category="Title")],
            processing_time=1.0,
            strategy_used=strategy,
            document_hash="test_hash",
        )

        assert result.strategy_used == strategy

    @pytest.mark.unit
    def test_processing_result_performance_edge_cases(self):
        """Test ProcessingResult with performance edge cases."""
        # Very fast processing
        fast_result = ProcessingResult(
            elements=[],
            processing_time=0.001,
            strategy_used=ProcessingStrategy.FAST,
            document_hash="fast_hash",
        )
        assert fast_result.processing_time == 0.001

        # Very slow processing
        slow_result = ProcessingResult(
            elements=[],
            processing_time=300.0,
            strategy_used=ProcessingStrategy.HI_RES,
            document_hash="slow_hash",
        )
        assert slow_result.processing_time == 300.0

        # Zero time (edge case)
        zero_result = ProcessingResult(
            elements=[],
            processing_time=0.0,
            strategy_used=ProcessingStrategy.FAST,
            document_hash="zero_hash",
        )
        assert zero_result.processing_time == 0.0

    @pytest.mark.unit
    def test_processing_result_serialization(self):
        """Test ProcessingResult serialization and deserialization."""
        elements = [
            DocumentElement(text="Title", category="Title", metadata={"level": 1}),
            DocumentElement(
                text="Paragraph", category="NarrativeText", metadata={"page": 1}
            ),
        ]

        original = ProcessingResult(
            elements=elements,
            processing_time=3.5,
            strategy_used=ProcessingStrategy.OCR_ONLY,
            document_hash="serialization_test_hash",
            metadata={"source": "test.pdf", "version": "1.0"},
        )

        # Serialize and deserialize
        json_data = original.model_dump()
        restored = ProcessingResult.model_validate(json_data)

        assert len(restored.elements) == len(original.elements)
        assert restored.processing_time == original.processing_time
        assert restored.strategy_used == original.strategy_used
        assert restored.document_hash == original.document_hash
        assert restored.metadata == original.metadata

    @pytest.mark.unit
    def test_processing_result_create_hash_for_document_basic(self):
        """Test ProcessingResult.create_hash_for_document with basic file."""
        test_content = b"This is test file content for hashing."

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            hash_result = ProcessingResult.create_hash_for_document(temp_file_path)

            # Should be valid SHA-256 hex string
            assert len(hash_result) == 64
            assert all(c in "0123456789abcdef" for c in hash_result)

            # Same file should produce same hash
            hash_result2 = ProcessingResult.create_hash_for_document(temp_file_path)
            assert hash_result == hash_result2

        finally:
            Path(temp_file_path).unlink()

    @pytest.mark.unit
    def test_processing_result_create_hash_for_document_empty_file(self):
        """Test ProcessingResult.create_hash_for_document with empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name  # Empty file

        try:
            hash_result = ProcessingResult.create_hash_for_document(temp_file_path)
            assert len(hash_result) == 64
        finally:
            Path(temp_file_path).unlink()

    @pytest.mark.unit
    def test_processing_result_create_hash_for_document_large_file(self):
        """Test ProcessingResult.create_hash_for_document with large file."""
        # Create a large test file (1MB)
        large_content = b"x" * (1024 * 1024)

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(large_content)
            temp_file_path = temp_file.name

        try:
            hash_result = ProcessingResult.create_hash_for_document(temp_file_path)
            assert len(hash_result) == 64
        finally:
            Path(temp_file_path).unlink()

    @pytest.mark.unit
    def test_processing_result_create_hash_for_document_different_files(self):
        """Hashes differ for different files in create_hash_for_document."""
        content1 = b"First file content"
        content2 = b"Second file content"

        with tempfile.NamedTemporaryFile(delete=False) as temp_file1:
            temp_file1.write(content1)
            temp_file1_path = temp_file1.name

        with tempfile.NamedTemporaryFile(delete=False) as temp_file2:
            temp_file2.write(content2)
            temp_file2_path = temp_file2.name

        try:
            hash1 = ProcessingResult.create_hash_for_document(temp_file1_path)
            hash2 = ProcessingResult.create_hash_for_document(temp_file2_path)

            assert hash1 != hash2
            assert len(hash1) == 64
            assert len(hash2) == 64

        finally:
            Path(temp_file1_path).unlink()
            Path(temp_file2_path).unlink()

    @pytest.mark.unit
    def test_processing_result_create_hash_for_document_path_types(self):
        """Test ProcessingResult.create_hash_for_document with different path types."""
        test_content = b"Path type test content"

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            # Test with string path
            hash_str = ProcessingResult.create_hash_for_document(temp_file_path)

            # Test with Path object
            hash_path = ProcessingResult.create_hash_for_document(Path(temp_file_path))

            assert hash_str == hash_path

        finally:
            Path(temp_file_path).unlink()

    @pytest.mark.unit
    def test_processing_result_create_hash_includes_metadata(self):
        """Test ProcessingResult.create_hash_for_document includes file metadata."""
        test_content = b"Metadata test content"

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            hash1 = ProcessingResult.create_hash_for_document(temp_file_path)

            # Modify the file (different mtime)
            with open(temp_file_path, "ab") as f:
                f.write(b" additional content")

            hash2 = ProcessingResult.create_hash_for_document(temp_file_path)

            # Should be different because content and mtime changed
            assert hash1 != hash2

        finally:
            Path(temp_file_path).unlink()

    @pytest.mark.unit
    def test_processing_result_create_hash_file_not_found(self):
        """Test ProcessingResult.create_hash_for_document with non-existent file."""
        with pytest.raises(FileNotFoundError):
            ProcessingResult.create_hash_for_document("/non/existent/file.txt")

    @pytest.mark.unit
    def test_processing_result_validation_types(self):
        """Test ProcessingResult validation with invalid types."""
        with pytest.raises(ValidationError):
            ProcessingResult(
                elements="invalid",  # Should be list
                processing_time=1.0,
                strategy_used=ProcessingStrategy.FAST,
                document_hash="hash",
            )

        with pytest.raises(ValidationError):
            ProcessingResult(
                elements=[],
                processing_time="invalid",  # Should be float
                strategy_used=ProcessingStrategy.FAST,
                document_hash="hash",
            )

        with pytest.raises(ValidationError):
            ProcessingResult(
                elements=[],
                processing_time=1.0,
                strategy_used="invalid_strategy",  # Should be valid enum
                document_hash="hash",
            )


class TestProcessingError:
    """Test suite for ProcessingError exception."""

    @pytest.mark.unit
    def test_processing_error_creation_basic(self):
        """Test ProcessingError creation and basic functionality."""
        error = ProcessingError("Processing failed")

        assert str(error) == "Processing failed"
        assert isinstance(error, Exception)

    @pytest.mark.unit
    def test_processing_error_creation_empty(self):
        """Test ProcessingError with empty message."""
        error = ProcessingError("")
        assert str(error) == ""

    @pytest.mark.unit
    def test_processing_error_creation_unicode(self):
        """Test ProcessingError with unicode characters."""
        error = ProcessingError("处理失败 with émojis 📄")
        assert "处理失败" in str(error)
        assert "📄" in str(error)

    @pytest.mark.unit
    def test_processing_error_inheritance(self):
        """Test ProcessingError inheritance from Exception."""
        error = ProcessingError("Test")
        assert isinstance(error, Exception)
        assert issubclass(ProcessingError, Exception)

    @pytest.mark.unit
    def test_processing_error_raising(self):
        """Test raising and catching ProcessingError."""
        with pytest.raises(ProcessingError) as exc_info:
            raise ProcessingError("Document processing failed")

        assert str(exc_info.value) == "Document processing failed"

    @pytest.mark.unit
    def test_processing_error_with_args(self):
        """Test ProcessingError with multiple arguments."""
        error = ProcessingError("Processing failed", "Additional context", 500)

        # Exception args should contain all arguments
        assert error.args == ("Processing failed", "Additional context", 500)

    @pytest.mark.unit
    def test_processing_error_chaining(self):
        """Test ProcessingError exception chaining."""

        def _raise_processing_error():
            try:
                raise OSError("File read error")
            except OSError as err:
                raise ProcessingError("Document processing failed") from err

        with pytest.raises(ProcessingError) as exc_info:
            _raise_processing_error()

        processing_error = exc_info.value
        assert isinstance(processing_error.__cause__, IOError)
        assert str(processing_error.__cause__) == "File read error"

    @pytest.mark.unit
    def test_processing_error_in_context(self):
        """Test ProcessingError used in realistic processing context."""

        def mock_processing_function(success: bool):
            if not success:
                raise ProcessingError("Failed to extract text from PDF")
            return "Success"

        # Test successful case
        result = mock_processing_function(True)
        assert result == "Success"

        # Test error case
        with pytest.raises(ProcessingError) as exc_info:
            mock_processing_function(False)

        assert "extract text" in str(exc_info.value)
