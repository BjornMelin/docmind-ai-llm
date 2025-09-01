"""Comprehensive exception handling tests for all custom exception classes.

This module provides thorough testing of all custom exception classes across
the DocMind AI system, focusing on proper inheritance, error messages,
exception chaining, and real-world usage scenarios.

Tests cover:
- Custom exception class instantiation
- Exception inheritance hierarchy
- Error message handling (including Unicode)
- Exception chaining and context propagation
- Exception raising and catching patterns
- Integration with system error handling
"""

import pytest

from src.models.embeddings import EmbeddingError
from src.models.processing import ProcessingError
from src.models.storage import PersistenceError


class TestEmbeddingError:
    """Comprehensive tests for EmbeddingError exception class."""

    @pytest.mark.unit
    def test_embedding_error_basic_instantiation(self):
        """Test basic EmbeddingError instantiation and properties."""
        error = EmbeddingError("Basic embedding error")

        assert str(error) == "Basic embedding error"
        assert isinstance(error, Exception)
        assert isinstance(error, EmbeddingError)
        assert error.args == ("Basic embedding error",)

    @pytest.mark.unit
    def test_embedding_error_empty_message(self):
        """Test EmbeddingError with empty message."""
        error = EmbeddingError("")

        assert str(error) == ""
        assert error.args == ("",)

    @pytest.mark.unit
    def test_embedding_error_none_message(self):
        """Test EmbeddingError with None message."""
        error = EmbeddingError(None)

        assert str(error) == "None"
        assert error.args == (None,)

    @pytest.mark.unit
    def test_embedding_error_unicode_message(self):
        """Test EmbeddingError with Unicode characters."""
        unicode_message = (
            "嵌入错误: Failed to process 文档 with émojis 🚀 and математика"
        )
        error = EmbeddingError(unicode_message)

        assert str(error) == unicode_message
        assert "嵌入错误" in str(error)
        assert "文档" in str(error)
        assert "🚀" in str(error)
        assert "математика" in str(error)

    @pytest.mark.unit
    def test_embedding_error_long_message(self):
        """Test EmbeddingError with very long message."""
        long_message = (
            "Embedding processing failed: " + "A" * 10000 + " (detailed error context)"
        )
        error = EmbeddingError(long_message)

        assert len(str(error)) > 10000
        assert str(error).startswith("Embedding processing failed:")
        assert str(error).endswith(" (detailed error context)")

    @pytest.mark.unit
    def test_embedding_error_multiple_arguments(self):
        """Test EmbeddingError with multiple arguments."""
        error = EmbeddingError(
            "Primary error", "Secondary context", 42, {"detail": "extra info"}
        )

        assert error.args == (
            "Primary error",
            "Secondary context",
            42,
            {"detail": "extra info"},
        )
        # Python formats Exception(*args) as a tuple string when multiple args are provided  # noqa: E501
        # in the exception constructor
        assert error.args[0] == "Primary error"
        assert "Primary error" in str(error)

    @pytest.mark.unit
    def test_embedding_error_inheritance(self):
        """Test EmbeddingError inheritance hierarchy."""
        error = EmbeddingError("Inheritance test")

        # Should inherit from Exception
        assert isinstance(error, Exception)
        assert isinstance(error, BaseException)
        assert issubclass(EmbeddingError, Exception)
        assert issubclass(EmbeddingError, BaseException)

    @pytest.mark.unit
    def test_embedding_error_raising_and_catching(self):
        """Test raising and catching EmbeddingError."""
        with pytest.raises(EmbeddingError) as exc_info:
            raise EmbeddingError("Test embedding failure")

        assert str(exc_info.value) == "Test embedding failure"
        assert exc_info.type == EmbeddingError

    @pytest.mark.unit
    def test_embedding_error_exception_chaining_from(self):
        """Test EmbeddingError exception chaining with 'raise from'."""

        def _raise_chained():
            try:
                raise ValueError("Original embedding model error")
            except ValueError as original_error:
                raise EmbeddingError(
                    "Failed to initialize BGE-M3 embeddings"
                ) from original_error

        with pytest.raises(EmbeddingError) as exc_info:
            _raise_chained()
        chained_error = exc_info.value
        assert str(chained_error) == "Failed to initialize BGE-M3 embeddings"
        assert isinstance(chained_error.__cause__, ValueError)
        assert str(chained_error.__cause__) == "Original embedding model error"

    @pytest.mark.unit
    def test_embedding_error_exception_chaining_during(self):
        """Test EmbeddingError exception context (implicit chaining)."""

        def _raise_context():
            try:
                raise RuntimeError("Embedding computation failed")
            except RuntimeError as err:
                raise EmbeddingError("Embedding pipeline error") from err

        with pytest.raises(EmbeddingError) as exc_info:
            _raise_context()
        context_error = exc_info.value
        assert str(context_error) == "Embedding pipeline error"
        assert isinstance(context_error.__context__, RuntimeError)
        assert str(context_error.__context__) == "Embedding computation failed"

    @pytest.mark.unit
    def test_embedding_error_realistic_scenarios(self):
        """Test EmbeddingError in realistic usage scenarios."""

        def mock_embedding_function(model_available: bool, memory_sufficient: bool):
            """Mock function that can raise EmbeddingError in various scenarios."""
            if not model_available:
                raise EmbeddingError("BGE-M3 model not found or not downloaded")
            if not memory_sufficient:
                raise EmbeddingError(
                    "Insufficient GPU memory for embedding batch processing"
                )
            return "Embeddings computed successfully"

        # Test successful case
        result = mock_embedding_function(model_available=True, memory_sufficient=True)
        assert result == "Embeddings computed successfully"

        # Test model unavailable
        with pytest.raises(
            EmbeddingError, match="model not found|not downloaded"
        ) as exc_info:
            mock_embedding_function(model_available=False, memory_sufficient=True)
        assert "BGE-M3 model not found" in str(exc_info.value)

        # Test insufficient memory
        with pytest.raises(EmbeddingError, match="Insufficient GPU memory") as exc_info:
            mock_embedding_function(model_available=True, memory_sufficient=False)
        assert "Insufficient GPU memory" in str(exc_info.value)

    @pytest.mark.unit
    def test_embedding_error_with_technical_details(self):
        """Test EmbeddingError with technical error details."""
        technical_error = EmbeddingError(
            "BGE-M3 embedding failed",
            {
                "model": "BAAI/bge-m3",
                "input_tokens": 8192,
                "batch_size": 32,
                "device": "cuda:0",
                "memory_used_mb": 12000,
                "error_code": "CUDA_OUT_OF_MEMORY",
            },
        )

        # Primary message should be first arg; overall str may include all args
        assert technical_error.args[0] == "BGE-M3 embedding failed"
        assert "BGE-M3 embedding failed" in str(technical_error)
        assert technical_error.args[1]["model"] == "BAAI/bge-m3"
        assert technical_error.args[1]["error_code"] == "CUDA_OUT_OF_MEMORY"


class TestProcessingError:
    """Comprehensive tests for ProcessingError exception class."""

    @pytest.mark.unit
    def test_processing_error_basic_instantiation(self):
        """Test basic ProcessingError instantiation and properties."""
        error = ProcessingError("Document processing failed")

        assert str(error) == "Document processing failed"
        assert isinstance(error, Exception)
        assert isinstance(error, ProcessingError)
        assert error.args == ("Document processing failed",)

    @pytest.mark.unit
    def test_processing_error_with_file_context(self):
        """Test ProcessingError with file processing context."""
        error = ProcessingError(
            "Failed to parse PDF document",
            {
                "file_path": "/documents/research_paper.pdf",
                "file_size_mb": 45.2,
                "processing_strategy": "hi_res",
                "page_number": 127,
                "element_type": "Table",
            },
        )

        # Primary message present; full str may include all args formatting
        assert error.args[0] == "Failed to parse PDF document"
        assert "Failed to parse PDF document" in str(error)
        assert error.args[1]["file_path"] == "/documents/research_paper.pdf"
        assert error.args[1]["page_number"] == 127

    @pytest.mark.unit
    def test_processing_error_unicode_filenames(self):
        """Test ProcessingError with Unicode filenames and content."""
        unicode_error = ProcessingError(
            "处理文档失败: Cannot process 文档.pdf with special characters",
            {
                "filename": "研究论文_2024年.pdf",
                "text_excerpt": "第一章：人工智能概述 🤖",
                "encoding": "utf-8",
            },
        )

        assert "处理文档失败" in str(unicode_error)
        assert unicode_error.args[1]["filename"] == "研究论文_2024年.pdf"
        assert "🤖" in unicode_error.args[1]["text_excerpt"]

    @pytest.mark.unit
    def test_processing_error_inheritance(self):
        """Test ProcessingError inheritance hierarchy."""
        error = ProcessingError("Processing inheritance test")

        assert isinstance(error, Exception)
        assert isinstance(error, BaseException)
        assert issubclass(ProcessingError, Exception)
        assert issubclass(ProcessingError, BaseException)

    @pytest.mark.unit
    def test_processing_error_chaining_io_errors(self):
        """Test ProcessingError chaining with I/O errors."""

        def simulate_io_error_processing():
            try:
                raise FileNotFoundError("Document file not accessible")
            except FileNotFoundError as io_error:
                raise ProcessingError(
                    "Cannot process document due to file system error"
                ) from io_error

        with pytest.raises(ProcessingError) as exc_info:
            simulate_io_error_processing()

        processing_error = exc_info.value
        assert "Cannot process document" in str(processing_error)
        assert isinstance(processing_error.__cause__, FileNotFoundError)
        assert "not accessible" in str(processing_error.__cause__)

    @pytest.mark.unit
    def test_processing_error_realistic_scenarios(self):
        """Test ProcessingError in realistic document processing scenarios."""

        def mock_document_processor(
            file_exists: bool, file_readable: bool, memory_available: bool
        ):
            """Mock doc processor that raises ProcessingError in various scenarios."""
            if not file_exists:
                raise ProcessingError("Document file does not exist")
            if not file_readable:
                raise ProcessingError("Document file is corrupted or encrypted")
            if not memory_available:
                raise ProcessingError(
                    "Insufficient memory for large document processing"
                )
            return "Document processed successfully"

        # Test successful processing
        result = mock_document_processor(
            file_exists=True, file_readable=True, memory_available=True
        )
        assert result == "Document processed successfully"

        # Test file not found
        with pytest.raises(ProcessingError) as exc_info:
            mock_document_processor(
                file_exists=False, file_readable=True, memory_available=True
            )
        assert "does not exist" in str(exc_info.value)

        # Test corrupted file
        with pytest.raises(ProcessingError) as exc_info:
            mock_document_processor(
                file_exists=True, file_readable=False, memory_available=True
            )
        assert "corrupted or encrypted" in str(exc_info.value)

    @pytest.mark.unit
    def test_processing_error_with_stack_trace_context(self):
        """Test ProcessingError preserves stack trace context."""

        def level_3_function():
            raise ValueError("Deep processing error")

        def level_2_function():
            try:
                level_3_function()
            except ValueError as e:
                raise ProcessingError("Level 2 processing failed") from e

        def level_1_function():
            try:
                level_2_function()
            except ProcessingError as e:
                raise ProcessingError("Level 1 processing pipeline failed") from e

        with pytest.raises(ProcessingError) as exc_info:
            level_1_function()

        # Verify exception chaining preserves original error
        assert "Level 1 processing pipeline failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ProcessingError)
        assert "Level 2 processing failed" in str(exc_info.value.__cause__)
        assert isinstance(exc_info.value.__cause__.__cause__, ValueError)
        assert "Deep processing error" in str(exc_info.value.__cause__.__cause__)


class TestPersistenceError:
    """Comprehensive tests for PersistenceError exception class."""

    @pytest.mark.unit
    def test_persistence_error_basic_instantiation(self):
        """Test basic PersistenceError instantiation and properties."""
        error = PersistenceError("Database connection failed")

        assert error.args[0] == "Database connection failed"
        assert "Database connection failed" in str(error)
        assert isinstance(error, Exception)
        assert isinstance(error, PersistenceError)
        assert error.args == ("Database connection failed",)

    @pytest.mark.unit
    def test_persistence_error_with_database_context(self):
        """Test PersistenceError with database operation context."""
        error = PersistenceError(
            "Failed to insert document into vector database",
            {
                "operation": "vector_insert",
                "database": "qdrant",
                "collection": "docmind_docs",
                "document_id": "doc_12345",
                "vector_dimension": 1024,
                "batch_size": 100,
                "connection_url": "http://localhost:6333",
            },
        )

        assert error.args[0] == "Failed to insert document into vector database"
        assert "Failed to insert document into vector database" in str(error)
        assert error.args[1]["database"] == "qdrant"
        assert error.args[1]["collection"] == "docmind_docs"
        assert error.args[1]["vector_dimension"] == 1024

    @pytest.mark.unit
    def test_persistence_error_unicode_data(self):
        """Test PersistenceError with Unicode data and paths."""
        unicode_error = PersistenceError(
            "数据库错误: Failed to persist document with Chinese text",
            {
                "document_text": "机器学习是人工智能的一个重要分支 🤖",
                "database_path": "/data/数据库/文档.db",
                "encoding_used": "utf-8",
                "character_count": 1500,
            },
        )

        assert "数据库错误" in str(unicode_error)
        assert (
            unicode_error.args[1]["document_text"]
            == "机器学习是人工智能的一个重要分支 🤖"
        )
        assert "/数据库/" in unicode_error.args[1]["database_path"]

    @pytest.mark.unit
    def test_persistence_error_inheritance(self):
        """Test PersistenceError inheritance hierarchy."""
        error = PersistenceError("Persistence inheritance test")

        assert isinstance(error, Exception)
        assert isinstance(error, BaseException)
        assert issubclass(PersistenceError, Exception)
        assert issubclass(PersistenceError, BaseException)

    @pytest.mark.unit
    def test_persistence_error_chaining_connection_errors(self):
        """Test PersistenceError chaining with connection errors."""

        def _raise_persistence_chain():
            try:
                raise ConnectionError("Qdrant server unreachable at localhost:6333")
            except ConnectionError as conn_error:
                raise PersistenceError(
                    "Vector database persistence failed"
                ) from conn_error

        with pytest.raises(PersistenceError) as exc_info:
            _raise_persistence_chain()
        persistence_error = exc_info.value
        assert "persistence failed" in str(persistence_error)
        assert isinstance(persistence_error.__cause__, ConnectionError)
        assert "localhost:6333" in str(persistence_error.__cause__)

    @pytest.mark.unit
    def test_persistence_error_chaining_sqlite_errors(self):
        """Test PersistenceError chaining with SQLite errors."""
        import sqlite3

        def simulate_sqlite_persistence_error():
            try:
                raise sqlite3.OperationalError("database is locked")
            except sqlite3.Error as sqlite_error:
                raise PersistenceError(
                    "SQLite metadata persistence failed"
                ) from sqlite_error

        with pytest.raises(PersistenceError) as exc_info:
            simulate_sqlite_persistence_error()

        persistence_error = exc_info.value
        assert "metadata persistence failed" in str(persistence_error)
        assert isinstance(persistence_error.__cause__, sqlite3.Error)
        assert "database is locked" in str(persistence_error.__cause__)

    @pytest.mark.unit
    def test_persistence_error_realistic_scenarios(self):
        """Test PersistenceError in realistic database operation scenarios."""

        def mock_database_operations(
            qdrant_available: bool, sqlite_writable: bool, disk_space: bool
        ):
            """Mock db operations that raise PersistenceError in various scenarios."""
            if not qdrant_available:
                raise PersistenceError("Qdrant vector database is not available")
            if not sqlite_writable:
                raise PersistenceError("SQLite database file is read-only or locked")
            if not disk_space:
                raise PersistenceError(
                    "Insufficient disk space for database operations"
                )
            return "Database operations completed successfully"

        # Test successful operations
        result = mock_database_operations(
            qdrant_available=True, sqlite_writable=True, disk_space=True
        )
        assert result == "Database operations completed successfully"

        # Test Qdrant unavailable
        with pytest.raises(PersistenceError) as exc_info:
            mock_database_operations(
                qdrant_available=False, sqlite_writable=True, disk_space=True
            )
        assert "Qdrant vector database is not available" in str(exc_info.value)

        # Test SQLite locked
        with pytest.raises(PersistenceError) as exc_info:
            mock_database_operations(
                qdrant_available=True, sqlite_writable=False, disk_space=True
            )
        assert "read-only or locked" in str(exc_info.value)

    @pytest.mark.unit
    def test_persistence_error_with_transaction_context(self):
        """Test PersistenceError within database transaction context."""
        error = PersistenceError(
            "Transaction rolled back due to constraint violation",
            {
                "transaction_id": "tx_789012",
                "operation": "document_insert",
                "affected_tables": [
                    "documents",
                    "document_vectors",
                    "document_metadata",
                ],
                "constraint_violated": "UNIQUE(document_hash)",
                "rollback_successful": True,
                "cleanup_required": False,
            },
        )

        assert "Transaction rolled back" in str(error)
        assert error.args[1]["transaction_id"] == "tx_789012"
        assert "UNIQUE(document_hash)" in error.args[1]["constraint_violated"]
        assert error.args[1]["rollback_successful"] is True


class TestExceptionInteractionPatterns:
    """Test interaction patterns between different exception types."""

    @pytest.mark.unit
    def test_exception_hierarchy_catching(self):
        """Test catching specific vs general exceptions."""

        def raise_random_error(error_type: str):
            """Raise different types of errors for testing."""
            if error_type == "embedding":
                raise EmbeddingError("Embedding computation failed")
            elif error_type == "processing":
                raise ProcessingError("Document processing failed")
            elif error_type == "persistence":
                raise PersistenceError("Database operation failed")
            elif error_type == "general":
                raise ValueError("General error")

        # Test specific exception catching
        with pytest.raises(EmbeddingError):
            raise_random_error("embedding")

        with pytest.raises(ProcessingError):
            raise_random_error("processing")

        with pytest.raises(PersistenceError):
            raise_random_error("persistence")

        # Test specific mapping for all error types
        import re

        matrix = [
            ("embedding", EmbeddingError, r"embedding"),
            ("processing", ProcessingError, r"processing"),
            # Persistence error message may not include the word 'persistence'
            # explicitly; accept broader database/persist phrasing.
            ("persistence", PersistenceError, r"database|persist"),
            ("general", ValueError, r"general error"),
        ]
        for error_type, exc, pattern in matrix:
            with pytest.raises(exc, match=re.compile(pattern, re.IGNORECASE)):
                raise_random_error(error_type)

    @pytest.mark.unit
    def test_complex_exception_chaining_pipeline(self):
        """Test complex exception chaining through processing pipeline."""

        def embedding_stage():
            """Simulate embedding stage failure."""
            raise EmbeddingError("BGE-M3 model initialization failed")

        def processing_stage():
            """Simulate processing stage that depends on embedding."""
            try:
                embedding_stage()
            except EmbeddingError as e:
                raise ProcessingError(
                    "Document processing aborted due to embedding failure"
                ) from e

        def persistence_stage():
            """Simulate persistence stage that depends on processing."""
            try:
                processing_stage()
            except ProcessingError as e:
                raise PersistenceError(
                    "Cannot persist document due to processing failure"
                ) from e

        # Test full pipeline failure with chained exceptions
        with pytest.raises(PersistenceError) as exc_info:
            persistence_stage()

        # Verify complete exception chain
        persistence_error = exc_info.value
        assert "Cannot persist document" in str(persistence_error)

        processing_error = persistence_error.__cause__
        assert isinstance(processing_error, ProcessingError)
        assert "processing aborted" in str(processing_error)

        embedding_error = processing_error.__cause__
        assert isinstance(embedding_error, EmbeddingError)
        assert "BGE-M3 model initialization failed" in str(embedding_error)

    @pytest.mark.unit
    def test_exception_suppression_patterns(self):
        """Test exception suppression and context manager patterns."""
        errors_encountered = []

        def safe_operation(operation_name: str, should_fail: bool):
            """Simulate operations that may fail but should not stop pipeline."""
            try:
                if should_fail:
                    if operation_name == "embedding":
                        raise EmbeddingError(
                            f"Embedding operation {operation_name} failed"
                        )
                    elif operation_name == "processing":
                        raise ProcessingError(
                            f"Processing operation {operation_name} failed"
                        )
                    elif operation_name == "persistence":
                        raise PersistenceError(
                            f"Persistence operation {operation_name} failed"
                        )
                return f"Operation {operation_name} succeeded"
            except (EmbeddingError, ProcessingError, PersistenceError) as e:
                errors_encountered.append((operation_name, type(e).__name__, str(e)))
                return f"Operation {operation_name} failed gracefully"

        # Test pipeline continues despite individual failures
        results = []
        results.append(safe_operation("embedding", should_fail=True))
        results.append(safe_operation("processing", should_fail=False))
        results.append(safe_operation("persistence", should_fail=True))

        # Verify operations completed with graceful failure handling
        assert "failed gracefully" in results[0]
        assert "succeeded" in results[1]
        assert "failed gracefully" in results[2]

        # Verify errors were captured
        assert len(errors_encountered) == 2
        assert errors_encountered[0][1] == "EmbeddingError"
        assert errors_encountered[1][1] == "PersistenceError"

    @pytest.mark.unit
    def test_exception_context_preservation(self):
        """Test that exception context is preserved through complex operations."""

        def create_detailed_error(error_class, message: str, **context):
            """Create errors with rich context information."""
            return error_class(message, context)

        # Create nested operation with rich context
        def _raise_nested():
            try:
                embedding_error = create_detailed_error(
                    EmbeddingError,
                    "CUDA device not available",
                    device="cuda:0",
                    required_memory_gb=8.5,
                    available_memory_gb=2.1,
                    model="BAAI/bge-m3",
                )
                raise embedding_error
            except EmbeddingError as e:
                processing_error = create_detailed_error(
                    ProcessingError,
                    "Cannot process documents without embeddings",
                    document_count=150,
                    processing_strategy="hi_res",
                    fallback_available=False,
                )
                raise processing_error from e

        def _wrap_to_persistence():
            try:
                _raise_nested()
            except ProcessingError as e:
                persistence_error = create_detailed_error(
                    PersistenceError,
                    "Pipeline failure prevents data persistence",
                    transaction_id="tx_456789",
                    cleanup_performed=True,
                    data_integrity_verified=True,
                )
                raise persistence_error from e

        with pytest.raises(PersistenceError) as exc_info:
            _wrap_to_persistence()

        final_error = exc_info.value
        # Verify all context is preserved through the chain
        assert "Pipeline failure prevents data persistence" in str(final_error)
        assert final_error.args[1]["transaction_id"] == "tx_456789"

        # Check processing error context
        processing = final_error.__cause__
        assert processing.args[1]["document_count"] == 150
        assert processing.args[1]["processing_strategy"] == "hi_res"

        # Check embedding error context
        embedding = processing.__cause__
        assert embedding.args[1]["device"] == "cuda:0"
        assert embedding.args[1]["model"] == "BAAI/bge-m3"
        assert embedding.args[1]["required_memory_gb"] == 8.5
