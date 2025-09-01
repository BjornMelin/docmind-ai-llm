"""Comprehensive boundary validation tests for all Pydantic models.

This module provides high-value boundary testing for numeric fields across
all models in the DocMind AI system, focusing on edge cases and validation
boundaries that improve coverage with minimal effort.

Tests cover:
- Numeric field boundary conditions
- Type validation edge cases
- Required vs optional field validation
- Field constraint validation
- Input sanitization boundaries
"""

import pytest
from pydantic import ValidationError

from src.models.embeddings import EmbeddingParameters, EmbeddingResult
from src.models.processing import DocumentElement, ProcessingResult
from src.models.schemas import Document, PerformanceMetrics, QueryRequest
from src.models.storage import (
    DocumentMetadata,
    StorageStats,
    VectorRecord,
)


class TestEmbeddingModelBoundaries:
    """Test boundary conditions for embedding models."""

    @pytest.mark.unit
    def test_embedding_parameters_max_length_boundaries(self):
        """Test EmbeddingParameters max_length field boundaries."""
        # Valid boundaries
        EmbeddingParameters(max_length=512)  # Minimum
        EmbeddingParameters(max_length=8192)  # Default
        EmbeddingParameters(max_length=16384)  # Maximum

        # Invalid boundaries
        with pytest.raises(ValidationError):
            EmbeddingParameters(max_length=256)  # Below minimum

        with pytest.raises(ValidationError):
            EmbeddingParameters(max_length=32768)  # Above maximum

    @pytest.mark.unit
    def test_embedding_parameters_weights_boundary_values(self):
        """Test EmbeddingParameters weights with boundary values."""
        # All zero weights (edge case)
        params = EmbeddingParameters(weights_for_different_modes=[0.0, 0.0, 0.0])
        assert params.weights_for_different_modes == [0.0, 0.0, 0.0]

        # All maximum weights (edge case)
        params = EmbeddingParameters(weights_for_different_modes=[1.0, 1.0, 1.0])
        assert params.weights_for_different_modes == [1.0, 1.0, 1.0]

        # Single weight with maximum value
        params = EmbeddingParameters(weights_for_different_modes=[1.0])
        assert params.weights_for_different_modes == [1.0]

        # Negative weights (should be allowed - no explicit constraints)
        params = EmbeddingParameters(weights_for_different_modes=[-0.1, 0.5, 0.6])
        assert params.weights_for_different_modes == [-0.1, 0.5, 0.6]

    @pytest.mark.unit
    def test_embedding_result_numeric_boundaries(self):
        """Test EmbeddingResult with extreme numeric values."""
        # Minimum values
        result = EmbeddingResult(processing_time=0.0, batch_size=0, memory_usage_mb=0.0)
        assert result.processing_time == 0.0
        assert result.batch_size == 0
        assert result.memory_usage_mb == 0.0

        # Very small values
        result = EmbeddingResult(
            processing_time=0.001,  # 1ms
            batch_size=1,
            memory_usage_mb=0.1,
        )
        assert result.processing_time == 0.001
        assert result.batch_size == 1
        assert result.memory_usage_mb == 0.1

        # Very large values
        result = EmbeddingResult(
            processing_time=3600.0,  # 1 hour
            batch_size=10000,
            memory_usage_mb=100000.0,  # 100GB
        )
        assert result.processing_time == 3600.0
        assert result.batch_size == 10000
        assert result.memory_usage_mb == 100000.0


class TestProcessingModelBoundaries:
    """Test boundary conditions for processing models."""

    @pytest.mark.unit
    def test_processing_result_processing_time_boundaries(self):
        """Test ProcessingResult processing_time boundaries."""
        from src.models.processing import ProcessingStrategy

        # Zero processing time (instant processing)
        result = ProcessingResult(
            elements=[],
            processing_time=0.0,
            strategy_used=ProcessingStrategy.FAST,
            document_hash="test_hash",
        )
        assert result.processing_time == 0.0

        # Very fast processing
        result = ProcessingResult(
            elements=[],
            processing_time=0.001,  # 1ms
            strategy_used=ProcessingStrategy.FAST,
            document_hash="test_hash",
        )
        assert result.processing_time == 0.001

        # Very slow processing
        result = ProcessingResult(
            elements=[],
            processing_time=7200.0,  # 2 hours
            strategy_used=ProcessingStrategy.HI_RES,
            document_hash="test_hash",
        )
        assert result.processing_time == 7200.0

    @pytest.mark.unit
    def test_document_element_text_length_boundaries(self):
        """Test DocumentElement with various text lengths."""
        # Empty text
        element = DocumentElement(text="", category="Empty")
        assert element.text == ""

        # Single character
        element = DocumentElement(text="A", category="Short")
        assert element.text == "A"

        # Very long text (1MB)
        long_text = "A" * (1024 * 1024)
        element = DocumentElement(text=long_text, category="VeryLong")
        assert len(element.text) == 1024 * 1024

        # Unicode boundary cases
        unicode_text = "🚀" * 1000  # Emoji characters
        element = DocumentElement(text=unicode_text, category="Unicode")
        assert len(element.text) == 1000


class TestStorageModelBoundaries:
    """Test boundary conditions for storage models."""

    @pytest.mark.unit
    def test_document_metadata_file_size_boundaries(self):
        """Test DocumentMetadata file_size boundaries."""
        # Zero-size file
        metadata = DocumentMetadata(
            id="empty_file",
            file_path="/test/empty.txt",
            file_hash="empty_hash",
            file_size=0,
            processing_time=0.1,
            strategy_used="fast",
            element_count=0,
            created_at=1000.0,
            updated_at=1000.0,
        )
        assert metadata.file_size == 0

        # Large file (1TB)
        metadata = DocumentMetadata(
            id="huge_file",
            file_path="/test/huge.pdf",
            file_hash="huge_hash",
            file_size=1099511627776,  # 1TB
            processing_time=3600.0,
            strategy_used="hi_res",
            element_count=100000,
            created_at=1000.0,
            updated_at=1000.0,
        )
        assert metadata.file_size == 1099511627776

    @pytest.mark.unit
    def test_vector_record_chunk_index_boundaries(self):
        """Test VectorRecord chunk_index boundaries."""
        # First chunk
        record = VectorRecord(
            id="chunk_0",
            document_id="doc_001",
            chunk_index=0,
            text="First chunk",
            embedding=[0.1, 0.2],
        )
        assert record.chunk_index == 0

        # Large chunk index
        record = VectorRecord(
            id="chunk_large",
            document_id="doc_001",
            chunk_index=999999,
            text="Large index chunk",
            embedding=[0.1, 0.2],
        )
        assert record.chunk_index == 999999

    @pytest.mark.unit
    def test_storage_stats_large_numbers(self):
        """Test StorageStats with very large numbers."""
        # Empty stats
        stats = StorageStats(
            total_documents=0,
            total_vectors=0,
            sqlite_size_mb=0.0,
            qdrant_size_mb=0.0,
            avg_processing_time=0.0,
            last_indexed_at=None,
        )
        assert stats.total_documents == 0
        assert stats.total_vectors == 0

        # Very large corpus
        stats = StorageStats(
            total_documents=10_000_000,  # 10M documents
            total_vectors=1_000_000_000,  # 1B vectors
            sqlite_size_mb=100_000.0,  # 100GB SQLite
            qdrant_size_mb=1_000_000.0,  # 1TB Qdrant
            avg_processing_time=60.0,  # 1 minute average
            last_indexed_at=1704067200.0,
        )
        assert stats.total_documents == 10_000_000
        assert stats.total_vectors == 1_000_000_000


class TestSchemaModelBoundaries:
    """Test boundary conditions for schema models."""

    @pytest.mark.unit
    def test_document_boundaries(self):
        """Test Document with boundary conditions."""
        # Minimal document
        doc = Document(id="", text="", metadata={})
        assert doc.id == ""
        assert doc.text == ""

        # Very long content
        long_content = "Test content " * 100000  # ~1.3M characters
        doc = Document(id="long_doc", text=long_content, metadata={"size": "large"})
        assert len(doc.text) > 1000000

    @pytest.mark.unit
    def test_query_request_boundaries(self):
        """Test QueryRequest with boundary conditions."""
        # Empty query
        request = QueryRequest(query="", top_k=1)
        assert request.query == ""
        assert request.top_k == 1

        # Very long query
        long_query = "What is " * 10000  # ~70K character query
        request = QueryRequest(query=long_query, top_k=50)
        assert len(request.query) > 50000

    @pytest.mark.unit
    def test_performance_metrics_boundaries(self):
        """Test PerformanceMetrics with boundary conditions."""
        # Zero performance metrics
        metrics = PerformanceMetrics(
            query_latency_ms=0.0,
            agent_overhead_ms=0.0,
            retrieval_latency_ms=0.0,
            llm_latency_ms=0.0,
            memory_usage_mb=0.0,
            vram_usage_mb=0.0,
            tokens_per_second=0.0,
            cache_hit_rate=0.0,
        )
        assert metrics.query_latency_ms == 0.0
        assert metrics.cache_hit_rate == 0.0

        # Very high performance metrics
        metrics = PerformanceMetrics(
            query_latency_ms=3600000.0,  # 1 hour in ms
            agent_overhead_ms=1000.0,
            retrieval_latency_ms=5000.0,
            llm_latency_ms=10000.0,
            memory_usage_mb=100000.0,  # 100GB
            vram_usage_mb=24000.0,
            tokens_per_second=10000.0,
            cache_hit_rate=99.9,
        )
        assert metrics.query_latency_ms == 3600000.0
        assert metrics.tokens_per_second == 10000.0


class TestFieldTypeCoercionBoundaries:
    """Test field type coercion and validation boundaries."""

    @pytest.mark.unit
    def test_integer_field_type_coercion(self):
        """Test integer field type coercion boundaries."""
        # Float to int coercion
        result = EmbeddingResult(
            processing_time=1.0,
            batch_size=8.0,  # Float that should become int
            memory_usage_mb=1024.0,
        )
        assert isinstance(result.batch_size, int)
        assert result.batch_size == 8

    @pytest.mark.unit
    def test_float_field_type_coercion(self):
        """Test float field type coercion boundaries."""
        # Int to float coercion
        result = EmbeddingResult(
            processing_time=1,  # Int that should become float
            batch_size=8,
            memory_usage_mb=1024,  # Int that should become float
        )
        assert isinstance(result.processing_time, float)
        assert isinstance(result.memory_usage_mb, float)
        assert result.processing_time == 1.0
        assert result.memory_usage_mb == 1024.0

    @pytest.mark.unit
    def test_string_field_validation(self):
        """Test string field validation boundaries."""
        # Empty strings should be valid for most fields
        element = DocumentElement(text="", category="")
        assert element.text == ""
        assert element.category == ""

        # Very long strings
        very_long_string = "a" * 1000000  # 1M characters
        element = DocumentElement(text=very_long_string, category="VeryLong")
        assert len(element.text) == 1000000

        # Unicode strings
        unicode_string = "🚀 测试 émojis αβγ математика"
        element = DocumentElement(text=unicode_string, category="Unicode")
        assert "🚀" in element.text
        assert "测试" in element.text
        assert "математика" in element.text
