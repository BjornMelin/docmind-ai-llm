"""Comprehensive test suite for Pydantic models in storage.py.

This module provides thorough testing of all models in src/models/storage.py,
focusing on hybrid persistence system with SQLite and Qdrant integration.
"""

import time

import pytest
from pydantic import ValidationError

from src.models.storage import (
    DocumentMetadata,
    PersistenceError,
    SearchResult,
    StorageStats,
    VectorRecord,
)


class TestDocumentMetadata:
    """Test suite for DocumentMetadata model."""

    @pytest.mark.unit
    def test_document_metadata_creation_basic(self):
        """Test DocumentMetadata creation with basic data."""
        timestamp = time.time()
        metadata = DocumentMetadata(
            id="doc_001",
            file_path="/path/to/document.pdf",
            file_hash="abc123def456",
            file_size=1024000,
            processing_time=2.5,
            strategy_used="hi_res",
            element_count=25,
            created_at=timestamp,
            updated_at=timestamp,
            metadata={"author": "John Doe", "pages": 10},
        )

        assert metadata.id == "doc_001"
        assert metadata.file_path == "/path/to/document.pdf"
        assert metadata.file_hash == "abc123def456"
        assert metadata.file_size == 1024000
        assert metadata.processing_time == 2.5
        assert metadata.strategy_used == "hi_res"
        assert metadata.element_count == 25
        assert metadata.created_at == timestamp
        assert metadata.updated_at == timestamp
        assert metadata.metadata["author"] == "John Doe"

    @pytest.mark.unit
    def test_document_metadata_creation_minimal(self):
        """Test DocumentMetadata creation with minimal data."""
        metadata = DocumentMetadata(
            id="doc_minimal",
            file_path="/min/path.txt",
            file_hash="hash123",
            file_size=100,
            processing_time=0.1,
            strategy_used="fast",
            element_count=1,
            created_at=1000.0,
            updated_at=1000.0,
        )

        assert metadata.metadata == {}  # Default empty dict

    @pytest.mark.unit
    def test_document_metadata_file_paths_various(self):
        """Test DocumentMetadata with various file path formats."""
        paths = [
            "/absolute/path/document.pdf",
            "relative/path/document.docx",
            "C:\\Windows\\path\\document.txt",
            "/path/with spaces/document name.pdf",
            "/path/with-special_chars/document@2024.txt",
            "/very/long/nested/deep/path/structure/with/many/levels/document.pdf",
        ]

        for path in paths:
            metadata = DocumentMetadata(
                id=f"doc_{hash(path)}",
                file_path=path,
                file_hash="test_hash",
                file_size=1000,
                processing_time=1.0,
                strategy_used="fast",
                element_count=5,
                created_at=1000.0,
                updated_at=1000.0,
            )
            assert metadata.file_path == path

    @pytest.mark.unit
    def test_document_metadata_file_sizes_edge_cases(self):
        """Test DocumentMetadata with various file sizes."""
        test_cases = [
            0,  # Empty file
            1,  # Single byte
            1024,  # 1KB
            1048576,  # 1MB
            1073741824,  # 1GB
            2**63 - 1,  # Max int64
        ]

        for file_size in test_cases:
            metadata = DocumentMetadata(
                id="doc_size_test",
                file_path="/test/path.txt",
                file_hash="size_hash",
                file_size=file_size,
                processing_time=1.0,
                strategy_used="fast",
                element_count=1,
                created_at=1000.0,
                updated_at=1000.0,
            )
            assert metadata.file_size == file_size

    @pytest.mark.unit
    def test_document_metadata_processing_times_edge_cases(self):
        """Test DocumentMetadata with various processing times."""
        test_times = [
            0.0,  # Instant processing
            0.001,  # 1ms
            1.0,  # 1 second
            60.0,  # 1 minute
            3600.0,  # 1 hour
            86400.0,  # 1 day
        ]

        for proc_time in test_times:
            metadata = DocumentMetadata(
                id="doc_time_test",
                file_path="/test/path.txt",
                file_hash="time_hash",
                file_size=1000,
                processing_time=proc_time,
                strategy_used="fast",
                element_count=1,
                created_at=1000.0,
                updated_at=1000.0,
            )
            assert metadata.processing_time == proc_time

    @pytest.mark.unit
    def test_document_metadata_strategies(self):
        """Test DocumentMetadata with different processing strategies."""
        strategies = ["hi_res", "fast", "ocr_only", "custom_strategy"]

        for strategy in strategies:
            metadata = DocumentMetadata(
                id="doc_strategy_test",
                file_path="/test/path.txt",
                file_hash="strategy_hash",
                file_size=1000,
                processing_time=1.0,
                strategy_used=strategy,
                element_count=5,
                created_at=1000.0,
                updated_at=1000.0,
            )
            assert metadata.strategy_used == strategy

    @pytest.mark.unit
    def test_document_metadata_element_counts(self):
        """Test DocumentMetadata with various element counts."""
        test_counts = [0, 1, 10, 100, 1000, 10000]

        for count in test_counts:
            metadata = DocumentMetadata(
                id="doc_count_test",
                file_path="/test/path.txt",
                file_hash="count_hash",
                file_size=1000,
                processing_time=1.0,
                strategy_used="fast",
                element_count=count,
                created_at=1000.0,
                updated_at=1000.0,
            )
            assert metadata.element_count == count

    @pytest.mark.unit
    def test_document_metadata_complex_metadata(self):
        """Test DocumentMetadata with complex metadata structures."""
        complex_metadata = {
            "document_info": {
                "title": "Research Paper",
                "author": "Dr. Smith",
                "publication_date": "2024-01-15",
                "doi": "10.1000/example",
            },
            "extraction_stats": {
                "figures": 5,
                "tables": 3,
                "references": 45,
                "words": 12500,
            },
            "processing_details": {
                "ocr_confidence": 0.95,
                "layout_analysis": True,
                "language_detected": ["en", "es"],
                "errors": [],
            },
        }

        metadata = DocumentMetadata(
            id="doc_complex",
            file_path="/research/paper.pdf",
            file_hash="complex_hash",
            file_size=2048000,
            processing_time=15.7,
            strategy_used="hi_res",
            element_count=150,
            created_at=1000.0,
            updated_at=1001.0,
            metadata=complex_metadata,
        )

        assert metadata.metadata["document_info"]["title"] == "Research Paper"
        assert metadata.metadata["extraction_stats"]["words"] == 12500
        assert metadata.metadata["processing_details"]["ocr_confidence"] == 0.95

    @pytest.mark.unit
    def test_document_metadata_serialization(self):
        """Test DocumentMetadata serialization and deserialization."""
        timestamp = time.time()
        original = DocumentMetadata(
            id="doc_serialization",
            file_path="/path/to/serialize.pdf",
            file_hash="serialization_hash",
            file_size=5120000,
            processing_time=8.3,
            strategy_used="hi_res",
            element_count=75,
            created_at=timestamp,
            updated_at=timestamp + 10,
            metadata={"test": "value", "nested": {"key": "data"}},
        )

        # Serialize and deserialize
        json_data = original.model_dump()
        restored = DocumentMetadata.model_validate(json_data)

        assert restored.id == original.id
        assert restored.file_path == original.file_path
        assert restored.file_hash == original.file_hash
        assert restored.file_size == original.file_size
        assert restored.processing_time == original.processing_time
        assert restored.strategy_used == original.strategy_used
        assert restored.element_count == original.element_count
        assert restored.created_at == original.created_at
        assert restored.updated_at == original.updated_at
        assert restored.metadata == original.metadata

    @pytest.mark.unit
    def test_document_metadata_validation_types(self):
        """Test DocumentMetadata validation with invalid types."""
        with pytest.raises(ValidationError):
            DocumentMetadata(
                id=123,  # Should be string
                file_path="/path/test.txt",
                file_hash="hash",
                file_size=1000,
                processing_time=1.0,
                strategy_used="fast",
                element_count=5,
                created_at=1000.0,
                updated_at=1000.0,
            )

        with pytest.raises(ValidationError):
            DocumentMetadata(
                id="valid_id",
                file_path="/path/test.txt",
                file_hash="hash",
                file_size="invalid",  # Should be int
                processing_time=1.0,
                strategy_used="fast",
                element_count=5,
                created_at=1000.0,
                updated_at=1000.0,
            )


class TestVectorRecord:
    """Test suite for VectorRecord model."""

    @pytest.mark.unit
    def test_vector_record_creation_basic(self):
        """Test VectorRecord creation with basic data."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        record = VectorRecord(
            id="vec_001",
            document_id="doc_001",
            chunk_index=0,
            text="This is the first chunk of text.",
            embedding=embedding,
            metadata={"chunk_type": "paragraph", "page": 1},
        )

        assert record.id == "vec_001"
        assert record.document_id == "doc_001"
        assert record.chunk_index == 0
        assert record.text == "This is the first chunk of text."
        assert record.embedding == embedding
        assert record.metadata["chunk_type"] == "paragraph"

    @pytest.mark.unit
    def test_vector_record_creation_minimal(self):
        """Test VectorRecord creation with minimal data."""
        record = VectorRecord(
            id="vec_minimal",
            document_id="doc_minimal",
            chunk_index=0,
            text="Minimal text",
            embedding=[0.1, 0.2],
        )

        assert record.metadata == {}  # Default empty dict

    @pytest.mark.unit
    def test_vector_record_various_chunk_indices(self):
        """Test VectorRecord with various chunk indices."""
        test_indices = [0, 1, 10, 100, 1000, 10000]

        for idx in test_indices:
            record = VectorRecord(
                id=f"vec_{idx}",
                document_id="doc_chunks",
                chunk_index=idx,
                text=f"Chunk {idx} text",
                embedding=[0.1] * 10,
            )
            assert record.chunk_index == idx

    @pytest.mark.unit
    def test_vector_record_various_embedding_dimensions(self):
        """Test VectorRecord with different embedding dimensions."""
        test_dimensions = [
            [0.1],  # 1D
            [0.1, 0.2],  # 2D
            [0.1] * 384,  # 384D (sentence-transformers)
            [0.1] * 768,  # 768D (BERT)
            [0.1] * 1024,  # 1024D (BGE-M3)
            [0.1] * 1536,  # 1536D (OpenAI)
            [0.1] * 4096,  # 4096D (large models)
        ]

        for embedding in test_dimensions:
            record = VectorRecord(
                id=f"vec_dim_{len(embedding)}",
                document_id="doc_dimensions",
                chunk_index=0,
                text="Dimension test text",
                embedding=embedding,
            )
            assert len(record.embedding) == len(embedding)

    @pytest.mark.unit
    def test_vector_record_empty_embedding(self):
        """Test VectorRecord with empty embedding."""
        record = VectorRecord(
            id="vec_empty",
            document_id="doc_empty",
            chunk_index=0,
            text="Text with empty embedding",
            embedding=[],
        )
        assert record.embedding == []

    @pytest.mark.unit
    def test_vector_record_various_text_lengths(self):
        """Test VectorRecord with various text lengths."""
        test_texts = [
            "",  # Empty
            "Short",  # Short
            "This is a medium length text chunk.",  # Medium
            " ".join(["word"] * 100),  # Long (100 words)
            "A" * 10000,  # Very long (10K chars)
        ]

        for i, text in enumerate(test_texts):
            record = VectorRecord(
                id=f"vec_text_{i}",
                document_id="doc_text_lengths",
                chunk_index=i,
                text=text,
                embedding=[0.1, 0.2],
            )
            assert record.text == text

    @pytest.mark.unit
    def test_vector_record_unicode_text(self):
        """Test VectorRecord with unicode text content."""
        unicode_text = "文档内容 avec des accents éñ español 🚀 математика"
        record = VectorRecord(
            id="vec_unicode",
            document_id="doc_unicode",
            chunk_index=0,
            text=unicode_text,
            embedding=[0.1] * 1024,
        )

        assert "文档内容" in record.text
        assert "🚀" in record.text
        assert "математика" in record.text

    @pytest.mark.unit
    def test_vector_record_complex_metadata(self):
        """Test VectorRecord with complex metadata."""
        complex_metadata = {
            "source_info": {
                "page": 5,
                "paragraph": 3,
                "bbox": [100, 200, 300, 250],
                "confidence": 0.95,
            },
            "processing": {
                "chunking_method": "semantic",
                "overlap_tokens": 50,
                "original_length": 1500,
                "compressed_length": 512,
            },
            "semantic_info": {
                "topic": "machine_learning",
                "keywords": ["neural", "network", "training"],
                "relevance_score": 0.87,
            },
        }

        record = VectorRecord(
            id="vec_complex_meta",
            document_id="doc_complex_meta",
            chunk_index=15,
            text="Complex metadata text chunk with detailed information.",
            embedding=[0.1] * 1024,
            metadata=complex_metadata,
        )

        assert record.metadata["source_info"]["page"] == 5
        assert record.metadata["processing"]["chunking_method"] == "semantic"
        assert record.metadata["semantic_info"]["keywords"] == [
            "neural",
            "network",
            "training",
        ]

    @pytest.mark.unit
    def test_vector_record_serialization(self):
        """Test VectorRecord serialization and deserialization."""
        original = VectorRecord(
            id="vec_serialization",
            document_id="doc_serialization",
            chunk_index=42,
            text="Text for serialization testing with various characters: αβγ",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata={"test": "serialize", "numbers": [1, 2, 3]},
        )

        # Serialize and deserialize
        json_data = original.model_dump()
        restored = VectorRecord.model_validate(json_data)

        assert restored.id == original.id
        assert restored.document_id == original.document_id
        assert restored.chunk_index == original.chunk_index
        assert restored.text == original.text
        assert restored.embedding == original.embedding
        assert restored.metadata == original.metadata

    @pytest.mark.unit
    def test_vector_record_validation_types(self):
        """Test VectorRecord validation with invalid types."""
        with pytest.raises(ValidationError):
            VectorRecord(
                id="vec_invalid",
                document_id="doc_invalid",
                chunk_index="invalid",  # Should be int
                text="Valid text",
                embedding=[0.1, 0.2],
            )

        with pytest.raises(ValidationError):
            VectorRecord(
                id="vec_invalid2",
                document_id="doc_invalid2",
                chunk_index=0,
                text=123,  # Should be string
                embedding=[0.1, 0.2],
            )

        with pytest.raises(ValidationError):
            VectorRecord(
                id="vec_invalid3",
                document_id="doc_invalid3",
                chunk_index=0,
                text="Valid text",
                embedding="invalid",  # Should be list
            )


class TestSearchResult:
    """Test suite for SearchResult model."""

    def _create_test_document_metadata(self) -> DocumentMetadata:
        """Helper to create test DocumentMetadata."""
        return DocumentMetadata(
            id="doc_search_test",
            file_path="/search/test.pdf",
            file_hash="search_hash",
            file_size=2048000,
            processing_time=3.5,
            strategy_used="hi_res",
            element_count=50,
            created_at=1000.0,
            updated_at=1001.0,
            metadata={"title": "Search Test Document"},
        )

    @pytest.mark.unit
    def test_search_result_creation_basic(self):
        """Test SearchResult creation with basic data."""
        doc_meta = self._create_test_document_metadata()

        result = SearchResult(
            document_id="doc_search_test",
            chunk_id="chunk_001",
            text="This is the matching text content from the search.",
            score=0.85,
            document_metadata=doc_meta,
            chunk_metadata={"page": 3, "paragraph": 2},
        )

        assert result.document_id == "doc_search_test"
        assert result.chunk_id == "chunk_001"
        assert result.text == "This is the matching text content from the search."
        assert result.score == 0.85
        assert result.document_metadata.id == "doc_search_test"
        assert result.chunk_metadata["page"] == 3

    @pytest.mark.unit
    def test_search_result_creation_minimal(self):
        """Test SearchResult creation with minimal data."""
        doc_meta = self._create_test_document_metadata()

        result = SearchResult(
            document_id="doc_minimal",
            chunk_id="chunk_minimal",
            text="Minimal search result",
            score=0.5,
            document_metadata=doc_meta,
        )

        assert result.chunk_metadata == {}  # Default empty dict

    @pytest.mark.unit
    def test_search_result_various_scores(self):
        """Test SearchResult with various similarity scores."""
        doc_meta = self._create_test_document_metadata()
        test_scores = [0.0, 0.1, 0.5, 0.99, 1.0, 1.5, -0.1]  # Include edge cases

        for score in test_scores:
            result = SearchResult(
                document_id="doc_scores",
                chunk_id=f"chunk_score_{score}",
                text=f"Text with score {score}",
                score=score,
                document_metadata=doc_meta,
            )
            assert result.score == score

    @pytest.mark.unit
    def test_search_result_various_text_matches(self):
        """Test SearchResult with various types of matching text."""
        doc_meta = self._create_test_document_metadata()
        test_texts = [
            "",  # Empty match
            "Single word",  # Simple match
            "Multi-word phrase with context",  # Phrase match
            "A very long matching passage that contains " * 10,  # Long match
            "🔍 Unicode match with émojis and 中文 content",  # Unicode match
        ]

        for i, text in enumerate(test_texts):
            result = SearchResult(
                document_id="doc_text_match",
                chunk_id=f"chunk_text_{i}",
                text=text,
                score=0.8,
                document_metadata=doc_meta,
            )
            assert result.text == text

    @pytest.mark.unit
    def test_search_result_complex_chunk_metadata(self):
        """Test SearchResult with complex chunk metadata."""
        doc_meta = self._create_test_document_metadata()

        complex_chunk_metadata = {
            "location": {
                "page": 7,
                "section": "methodology",
                "paragraph": 4,
                "bbox": [150, 300, 450, 380],
            },
            "matching": {
                "query_terms": ["machine learning", "neural networks"],
                "match_type": "semantic",
                "match_positions": [10, 25, 67],
                "highlight_spans": [(10, 25), (67, 82)],
            },
            "ranking": {
                "bm25_score": 0.75,
                "vector_score": 0.92,
                "fusion_method": "rrf",
                "rank_position": 3,
            },
            "context": {
                "surrounding_chunks": ["chunk_006", "chunk_008"],
                "topic_coherence": 0.88,
                "document_relevance": 0.91,
            },
        }

        result = SearchResult(
            document_id="doc_complex_chunk",
            chunk_id="chunk_007",
            text="Complex chunk metadata text with detailed matching information.",
            score=0.89,
            document_metadata=doc_meta,
            chunk_metadata=complex_chunk_metadata,
        )

        assert result.chunk_metadata["location"]["page"] == 7
        assert result.chunk_metadata["matching"]["query_terms"][0] == "machine learning"
        assert result.chunk_metadata["ranking"]["fusion_method"] == "rrf"
        assert result.chunk_metadata["context"]["topic_coherence"] == 0.88

    @pytest.mark.unit
    def test_search_result_document_metadata_composition(self):
        """Test SearchResult with comprehensive DocumentMetadata composition."""
        # Create rich document metadata
        rich_doc_meta = DocumentMetadata(
            id="doc_rich",
            file_path="/research/advanced_ai.pdf",
            file_hash="rich_doc_hash_12345",
            file_size=15728640,  # 15MB
            processing_time=45.7,
            strategy_used="hi_res",
            element_count=300,
            created_at=1704067200.0,  # 2024-01-01
            updated_at=1704153600.0,  # 2024-01-02
            metadata={
                "title": "Advanced AI Techniques in Modern Applications",
                "authors": ["Dr. Alice Smith", "Prof. Bob Johnson"],
                "publication_year": 2024,
                "journal": "Journal of AI Research",
                "doi": "10.1000/ai.2024.001",
                "pages": 45,
                "language": "en",
                "keywords": [
                    "artificial intelligence",
                    "machine learning",
                    "deep learning",
                ],
            },
        )

        result = SearchResult(
            document_id="doc_rich",
            chunk_id="chunk_rich_007",
            text="Advanced techniques for neural network optimization in distributed systems.",
            score=0.94,
            document_metadata=rich_doc_meta,
            chunk_metadata={
                "page": 12,
                "section": "optimization_techniques",
                "relevance": "high",
            },
        )

        # Verify composition works correctly
        assert result.document_metadata.id == "doc_rich"
        assert result.document_metadata.file_size == 15728640
        assert (
            result.document_metadata.metadata["title"]
            == "Advanced AI Techniques in Modern Applications"
        )
        assert len(result.document_metadata.metadata["authors"]) == 2
        assert result.chunk_metadata["section"] == "optimization_techniques"

    @pytest.mark.unit
    def test_search_result_serialization(self):
        """Test SearchResult serialization and deserialization."""
        doc_meta = DocumentMetadata(
            id="doc_serialize",
            file_path="/serialize/test.pdf",
            file_hash="serialize_hash",
            file_size=1024000,
            processing_time=5.2,
            strategy_used="fast",
            element_count=30,
            created_at=2000.0,
            updated_at=2001.0,
            metadata={"test": "serialization"},
        )

        original = SearchResult(
            document_id="doc_serialize",
            chunk_id="chunk_serialize",
            text="Serialization test content with special characters: αβγ 🔍",
            score=0.77,
            document_metadata=doc_meta,
            chunk_metadata={"context": "serialization_test", "values": [1, 2, 3]},
        )

        # Serialize and deserialize
        json_data = original.model_dump()
        restored = SearchResult.model_validate(json_data)

        assert restored.document_id == original.document_id
        assert restored.chunk_id == original.chunk_id
        assert restored.text == original.text
        assert restored.score == original.score
        assert restored.document_metadata.id == original.document_metadata.id
        assert (
            restored.document_metadata.file_path == original.document_metadata.file_path
        )
        assert restored.chunk_metadata == original.chunk_metadata

    @pytest.mark.unit
    def test_search_result_validation_types(self):
        """Test SearchResult validation with invalid types."""
        doc_meta = self._create_test_document_metadata()

        with pytest.raises(ValidationError):
            SearchResult(
                document_id=123,  # Should be string
                chunk_id="chunk_001",
                text="Valid text",
                score=0.8,
                document_metadata=doc_meta,
            )

        with pytest.raises(ValidationError):
            SearchResult(
                document_id="doc_valid",
                chunk_id="chunk_001",
                text="Valid text",
                score="invalid",  # Should be float
                document_metadata=doc_meta,
            )

        with pytest.raises(ValidationError):
            SearchResult(
                document_id="doc_valid",
                chunk_id="chunk_001",
                text="Valid text",
                score=0.8,
                document_metadata="invalid",  # Should be DocumentMetadata
            )


class TestStorageStats:
    """Test suite for StorageStats model."""

    @pytest.mark.unit
    def test_storage_stats_creation_default(self):
        """Test StorageStats creation with default values."""
        stats = StorageStats()

        assert stats.total_documents == 0
        assert stats.total_vectors == 0
        assert stats.sqlite_size_mb == 0.0
        assert stats.qdrant_size_mb == 0.0
        assert stats.avg_processing_time == 0.0
        assert stats.last_indexed_at is None

    @pytest.mark.unit
    def test_storage_stats_creation_custom(self):
        """Test StorageStats creation with custom values."""
        timestamp = time.time()
        stats = StorageStats(
            total_documents=1500,
            total_vectors=15000,
            sqlite_size_mb=128.5,
            qdrant_size_mb=2048.7,
            avg_processing_time=3.2,
            last_indexed_at=timestamp,
        )

        assert stats.total_documents == 1500
        assert stats.total_vectors == 15000
        assert stats.sqlite_size_mb == 128.5
        assert stats.qdrant_size_mb == 2048.7
        assert stats.avg_processing_time == 3.2
        assert stats.last_indexed_at == timestamp

    @pytest.mark.unit
    def test_storage_stats_large_numbers(self):
        """Test StorageStats with large numbers."""
        stats = StorageStats(
            total_documents=1000000,  # 1M documents
            total_vectors=100000000,  # 100M vectors
            sqlite_size_mb=10240.0,  # 10GB SQLite
            qdrant_size_mb=102400.0,  # 100GB Qdrant
            avg_processing_time=125.7,  # ~2 minutes average
            last_indexed_at=1704067200.0,
        )

        assert stats.total_documents == 1000000
        assert stats.total_vectors == 100000000
        assert stats.sqlite_size_mb == 10240.0
        assert stats.qdrant_size_mb == 102400.0

    @pytest.mark.unit
    def test_storage_stats_zero_values(self):
        """Test StorageStats with explicit zero values."""
        stats = StorageStats(
            total_documents=0,
            total_vectors=0,
            sqlite_size_mb=0.0,
            qdrant_size_mb=0.0,
            avg_processing_time=0.0,
            last_indexed_at=0.0,
        )

        assert all(
            [
                stats.total_documents == 0,
                stats.total_vectors == 0,
                stats.sqlite_size_mb == 0.0,
                stats.qdrant_size_mb == 0.0,
                stats.avg_processing_time == 0.0,
                stats.last_indexed_at == 0.0,
            ]
        )

    @pytest.mark.unit
    def test_storage_stats_negative_values(self):
        """Test StorageStats allows negative values (no constraints defined)."""
        stats = StorageStats(
            total_documents=-1,
            total_vectors=-100,
            sqlite_size_mb=-5.0,
            qdrant_size_mb=-10.0,
            avg_processing_time=-1.0,
            last_indexed_at=-1000.0,
        )

        # Should be allowed since no explicit constraints are defined
        assert stats.total_documents == -1
        assert stats.total_vectors == -100
        assert stats.sqlite_size_mb == -5.0
        assert stats.qdrant_size_mb == -10.0
        assert stats.avg_processing_time == -1.0
        assert stats.last_indexed_at == -1000.0

    @pytest.mark.unit
    def test_storage_stats_float_precision(self):
        """Test StorageStats with high precision float values."""
        stats = StorageStats(
            sqlite_size_mb=123.456789,
            qdrant_size_mb=9876.543210,
            avg_processing_time=1.23456789,
            last_indexed_at=1704067200.123456,
        )

        assert abs(stats.sqlite_size_mb - 123.456789) < 1e-6
        assert abs(stats.qdrant_size_mb - 9876.543210) < 1e-6
        assert abs(stats.avg_processing_time - 1.23456789) < 1e-8

    @pytest.mark.unit
    def test_storage_stats_serialization(self):
        """Test StorageStats serialization and deserialization."""
        timestamp = 1704067200.5
        original = StorageStats(
            total_documents=2500,
            total_vectors=50000,
            sqlite_size_mb=256.75,
            qdrant_size_mb=4096.25,
            avg_processing_time=7.8,
            last_indexed_at=timestamp,
        )

        # Serialize and deserialize
        json_data = original.model_dump()
        restored = StorageStats.model_validate(json_data)

        assert restored.total_documents == original.total_documents
        assert restored.total_vectors == original.total_vectors
        assert restored.sqlite_size_mb == original.sqlite_size_mb
        assert restored.qdrant_size_mb == original.qdrant_size_mb
        assert restored.avg_processing_time == original.avg_processing_time
        assert restored.last_indexed_at == original.last_indexed_at

    @pytest.mark.unit
    def test_storage_stats_validation_types(self):
        """Test StorageStats validation with invalid types."""
        with pytest.raises(ValidationError):
            StorageStats(total_documents="invalid")  # Should be int

        with pytest.raises(ValidationError):
            StorageStats(sqlite_size_mb="invalid")  # Should be float

        with pytest.raises(ValidationError):
            StorageStats(last_indexed_at="invalid")  # Should be float or None


class TestPersistenceError:
    """Test suite for PersistenceError exception."""

    @pytest.mark.unit
    def test_persistence_error_creation_basic(self):
        """Test PersistenceError creation and basic functionality."""
        error = PersistenceError("Database connection failed")

        assert str(error) == "Database connection failed"
        assert isinstance(error, Exception)

    @pytest.mark.unit
    def test_persistence_error_creation_empty(self):
        """Test PersistenceError with empty message."""
        error = PersistenceError("")
        assert str(error) == ""

    @pytest.mark.unit
    def test_persistence_error_creation_unicode(self):
        """Test PersistenceError with unicode characters."""
        error = PersistenceError("存储错误 with émojis 💾")
        assert "存储错误" in str(error)
        assert "💾" in str(error)

    @pytest.mark.unit
    def test_persistence_error_inheritance(self):
        """Test PersistenceError inheritance from Exception."""
        error = PersistenceError("Test")
        assert isinstance(error, Exception)
        assert issubclass(PersistenceError, Exception)

    @pytest.mark.unit
    def test_persistence_error_raising(self):
        """Test raising and catching PersistenceError."""
        with pytest.raises(PersistenceError) as exc_info:
            raise PersistenceError("Vector store connection lost")

        assert str(exc_info.value) == "Vector store connection lost"

    @pytest.mark.unit
    def test_persistence_error_with_args(self):
        """Test PersistenceError with multiple arguments."""
        error = PersistenceError("Connection failed", "timeout", 30)

        # Exception args should contain all arguments
        assert error.args == ("Connection failed", "timeout", 30)

    @pytest.mark.unit
    def test_persistence_error_chaining(self):
        """Test PersistenceError exception chaining."""
        try:
            try:
                raise ConnectionError("Network timeout")
            except ConnectionError as e:
                raise PersistenceError("Failed to persist document") from e
        except PersistenceError as persistence_error:
            assert isinstance(persistence_error.__cause__, ConnectionError)
            assert str(persistence_error.__cause__) == "Network timeout"

    @pytest.mark.unit
    def test_persistence_error_in_storage_context(self):
        """Test PersistenceError used in realistic storage context."""

        def mock_storage_function(should_fail: bool):
            if should_fail:
                raise PersistenceError("Qdrant collection not found")
            return {"status": "success"}

        # Test successful case
        result = mock_storage_function(False)
        assert result["status"] == "success"

        # Test error case
        with pytest.raises(PersistenceError) as exc_info:
            mock_storage_function(True)

        assert "Qdrant collection" in str(exc_info.value)
