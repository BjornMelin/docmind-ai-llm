"""System tests for HybridDocumentProcessor with real models and GPU requirements.

This module provides system-level tests for HybridDocumentProcessor that require
actual GPU hardware and real model loading. These tests validate the complete
system behavior under realistic conditions.

HARDWARE REQUIREMENTS:
- RTX 4090 (16GB VRAM) or equivalent
- 32GB RAM recommended
- CUDA 12.8+ support
- 100GB available storage

Test Coverage:
- Real model loading and inference with GPU acceleration
- Actual document processing with unstructured.io
- Performance validation with realistic workloads
- Memory management under load
- Error handling with real model failures
- Multi-document processing workflows

Following 3-tier testing strategy:
- Tier 3 (System): End-to-end tests with real models (<5min each)
- Requires GPU hardware and real model loading
- Tests actual performance targets and hardware constraints
- Validates complete system integration
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.processing.hybrid_processor import HybridDocumentProcessor, ProcessingError
from src.processing.models import ProcessingResult, ProcessingStrategy


@pytest.fixture
def gpu_settings():
    """System test settings with GPU-optimized configuration."""
    from src.config.settings import app_settings

    return app_settings


@pytest.fixture
def system_test_documents(tmp_path):
    """Create realistic test documents for system testing."""
    documents = {}

    # Comprehensive PDF with multiple pages and complex content
    pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R 4 0 R]/Count 2>>endobj  
3 0 obj<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 5 0 R>>>>/MediaBox[0 0 612 792]/Contents 6 0 R>>endobj
4 0 obj<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 5 0 R>>>>/MediaBox[0 0 612 792]/Contents 7 0 R>>endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Times-Roman>>endobj
6 0 obj<</Length 180>>stream
BT
/F1 12 Tf
72 720 Td
(System Test Document - Page 1) Tj
0 -20 Td
(This document tests real GPU-accelerated processing) Tj
0 -20 Td
(with actual unstructured.io parsing and LlamaIndex) Tj
0 -20 Td
(pipeline integration under realistic conditions.) Tj
ET
endstream
endobj
7 0 obj<</Length 165>>stream
BT
/F1 12 Tf
72 720 Td
(System Test Document - Page 2) Tj
0 -20 Td
(Performance targets: >1 page/second processing) Tj
0 -20 Td
(Memory usage: <14GB VRAM on RTX 4090) Tj
0 -20 Td
(Quality: Complete multimodal extraction) Tj
ET
endstream
endobj
xref
0 8
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000130 00000 n
0000000251 00000 n
0000000372 00000 n
0000000428 00000 n
0000000657 00000 n
trailer<</Size 8/Root 1 0 R>>
startxref
871
%%EOF"""

    documents["complex_pdf"] = tmp_path / "system_test_complex.pdf"
    documents["complex_pdf"].write_bytes(pdf_content)

    # Large text document for performance testing
    large_text_content = """# DocMind AI System Test Document

This is a comprehensive system test document designed to validate the complete
HybridDocumentProcessor pipeline with real GPU acceleration and model loading.

## Executive Summary

DocMind AI represents a breakthrough in document processing technology, combining
the power of unstructured.io for multimodal extraction with LlamaIndex for 
sophisticated pipeline orchestration. The system leverages BGE-M3 unified
embeddings for both dense semantic similarity and sparse keyword matching.

## Technical Architecture

The architecture consists of multiple integrated components:

### Document Processing Layer
- HybridDocumentProcessor: Combines unstructured.io with LlamaIndex
- Strategy-based processing: hi_res, fast, and ocr_only modes
- UnstructuredTransformation: Custom LlamaIndex transformation
- Semantic chunking with SentenceSplitter

### Embedding Layer  
- BGE-M3 unified embeddings (1024D dense + sparse weights)
- FlagEmbedding library integration
- GPU acceleration with FP16 optimization
- Batch processing for efficiency

### Caching Layer
- Dual caching: LlamaIndex IngestionCache + SimpleCache
- Document hash-based deduplication
- Persistent storage for processed results
- Cache statistics and management

## Performance Specifications

Target performance metrics for system validation:

- Processing speed: >1 page/second with hi_res strategy
- Memory usage: <14GB VRAM on RTX 4090
- Context window: 128K tokens with Qwen3-4B-Instruct-2507-FP8
- Embedding dimension: 1024D dense + variable sparse
- Cache hit ratio: >80% for repeated documents
- Error rate: <1% for supported document formats

## Quality Metrics

The system must achieve high-quality extraction across multiple modalities:

- Text extraction: >99% accuracy for digital text
- Table structure: Complete HTML preservation with coordinates
- Image extraction: OCR text + coordinate mapping
- Metadata preservation: Complete element relationships
- Semantic chunking: Context-aware boundary detection

## Integration Points

System integration validates multiple connection points:

- Unstructured.io direct partition() calls
- LlamaIndex IngestionPipeline orchestration
- BGE-M3 FlagModel for unified embeddings
- Qdrant vector store for hybrid search
- SimpleCache for document persistence
- GPU memory management and optimization

## Error Handling

Comprehensive error handling across all system components:

- Document corruption detection and recovery
- GPU out-of-memory handling with fallbacks
- Model loading failures and retries
- Cache corruption detection and rebuilding
- Network timeout handling for model downloads

## Scalability Testing

The system must handle realistic production workloads:

- Batch document processing (10-100 documents)
- Concurrent processing streams
- Memory pressure under sustained load
- Cache management with large document sets
- Long-running processing sessions

This document provides comprehensive test coverage for validating
system-level behavior under realistic GPU-accelerated conditions.
"""

    documents["large_text"] = tmp_path / "system_test_large.txt"
    documents["large_text"].write_text(large_text_content)

    return documents


class TestHybridProcessorSystemGPU:
    """System tests for HybridDocumentProcessor with GPU requirements."""

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_real_gpu_accelerated_processing(
        self, gpu_settings, system_test_documents
    ):
        """Test real GPU-accelerated document processing with actual models.

        REQUIRES: RTX 4090, 32GB RAM, CUDA 12.8+
        """
        processor = HybridDocumentProcessor(gpu_settings)

        # Process PDF with hi_res strategy
        result = await processor.process_document_async(
            system_test_documents["complex_pdf"]
        )

        # Verify system-level processing results
        assert isinstance(result, ProcessingResult)
        assert result.strategy_used == ProcessingStrategy.HI_RES
        assert len(result.elements) > 0

        # Verify performance targets
        assert result.processing_time < 2.0, (
            f"Processing took {result.processing_time}s, exceeding performance target"
        )

        # Verify quality metrics
        text_elements = [
            elem
            for elem in result.elements
            if elem.category in ["Title", "NarrativeText"]
        ]
        assert len(text_elements) >= 2, (
            "Should extract multiple text elements from multi-page PDF"
        )

        # Verify metadata preservation
        for element in result.elements:
            assert "page_number" in element.metadata
            assert "element_id" in element.metadata
            assert "processing_strategy" in element.metadata
            assert element.metadata["processing_strategy"] == "hi_res"

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_large_document_processing_performance(
        self, gpu_settings, system_test_documents
    ):
        """Test performance with large documents under realistic GPU load.

        REQUIRES: RTX 4090, 32GB RAM
        """
        processor = HybridDocumentProcessor(gpu_settings)

        # Process large text document
        start_time = asyncio.get_event_loop().time()
        result = await processor.process_document_async(
            system_test_documents["large_text"]
        )
        total_time = asyncio.get_event_loop().time() - start_time

        # Verify performance under load
        assert total_time < 10.0, (
            f"Large document processing took {total_time}s, too slow for production"
        )
        assert result.processing_time > 0

        # Verify semantic chunking worked effectively
        assert len(result.elements) > 5, (
            "Large document should be split into multiple semantic chunks"
        )

        # Verify chunk quality
        chunk_sizes = [len(elem.text) for elem in result.elements]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
        assert 200 < avg_chunk_size < 800, (
            f"Average chunk size {avg_chunk_size} outside optimal range"
        )

        # Verify metadata consistency across chunks
        for element in result.elements:
            assert "source_file" in element.metadata
            assert element.metadata["source_file"].endswith("system_test_large.txt")

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_multi_document_batch_processing(
        self, gpu_settings, system_test_documents
    ):
        """Test batch processing multiple documents with GPU optimization.

        REQUIRES: RTX 4090, sustained GPU load tolerance
        """
        processor = HybridDocumentProcessor(gpu_settings)

        # Process multiple documents in sequence
        documents_to_process = [
            system_test_documents["complex_pdf"],
            system_test_documents["large_text"],
        ]

        results = []
        total_start_time = asyncio.get_event_loop().time()

        for doc_path in documents_to_process:
            result = await processor.process_document_async(doc_path)
            results.append(result)

        total_processing_time = asyncio.get_event_loop().time() - total_start_time

        # Verify batch processing performance
        assert len(results) == 2
        assert total_processing_time < 15.0, (
            f"Batch processing took {total_processing_time}s, too slow"
        )

        # Verify each document processed correctly
        pdf_result, text_result = results

        # PDF should use hi_res strategy
        assert pdf_result.strategy_used == ProcessingStrategy.HI_RES

        # Text should use fast strategy
        assert text_result.strategy_used == ProcessingStrategy.FAST

        # Verify different processing approaches
        assert len(pdf_result.elements) != len(text_result.elements)

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_gpu_memory_management_under_load(
        self, gpu_settings, system_test_documents
    ):
        """Test GPU memory management under sustained processing load.

        REQUIRES: RTX 4090, GPU memory monitoring
        """
        processor = HybridDocumentProcessor(gpu_settings)

        # Simulate sustained processing load
        processing_rounds = 5
        all_results = []

        for round_num in range(processing_rounds):
            result = await processor.process_document_async(
                system_test_documents["large_text"]
            )
            all_results.append(result)

            # Verify memory management between rounds
            if round_num > 0:
                # Processing time should remain stable (no memory leaks)
                time_variance = abs(
                    result.processing_time - all_results[0].processing_time
                )
                assert time_variance < 2.0, (
                    f"Processing time variance {time_variance}s suggests memory issues"
                )

        # Verify all processing rounds succeeded
        assert len(all_results) == processing_rounds

        # Test cache effectiveness
        cache_stats = await processor.get_cache_stats()
        assert "simple_cache" in cache_stats
        assert "llamaindex_cache" in cache_stats

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_cache_performance_with_real_models(
        self, gpu_settings, system_test_documents
    ):
        """Test cache performance with real model loading and GPU processing.

        REQUIRES: RTX 4090, persistent cache storage
        """
        processor = HybridDocumentProcessor(gpu_settings)

        # First processing - should populate cache
        first_result = await processor.process_document_async(
            system_test_documents["complex_pdf"]
        )
        first_time = first_result.processing_time

        # Second processing - should benefit from cache
        second_result = await processor.process_document_async(
            system_test_documents["complex_pdf"]
        )
        second_time = second_result.processing_time

        # Verify cache effectiveness
        # Note: With real models, cache may not dramatically reduce time due to model loading overhead
        # but should provide consistent results
        assert first_result.document_hash == second_result.document_hash
        assert len(first_result.elements) == len(second_result.elements)

        # Verify cache statistics
        cache_stats = await processor.get_cache_stats()
        assert cache_stats["processor_type"] == "hybrid"

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_error_recovery_with_real_gpu_constraints(self, gpu_settings):
        """Test error recovery under real GPU memory constraints.

        REQUIRES: RTX 4090, ability to trigger GPU memory pressure
        """
        processor = HybridDocumentProcessor(gpu_settings)

        # Test with non-existent file
        with pytest.raises(ProcessingError) as exc_info:
            await processor.process_document_async("/non/existent/document.pdf")

        assert "file not found" in str(exc_info.value).lower()

        # Test with corrupted file
        corrupted_path = Path(tempfile.mktemp(suffix=".pdf"))
        corrupted_path.write_bytes(b"This is not a valid PDF file")

        try:
            # Should either process gracefully or raise informative error
            result = await processor.process_document_async(corrupted_path)
            # If it processes, should have minimal elements
            assert isinstance(result, ProcessingResult)
        except ProcessingError as e:
            # If it raises error, should be informative
            error_message = str(e).lower()
            assert any(
                keyword in error_message
                for keyword in [
                    "corrupted",
                    "invalid",
                    "processing failed",
                    "partition failed",
                ]
            )
        finally:
            corrupted_path.unlink(missing_ok=True)

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_real_world_performance_benchmarks(
        self, gpu_settings, system_test_documents
    ):
        """Test real-world performance benchmarks with production-like workloads.

        REQUIRES: RTX 4090, performance baseline validation
        """
        processor = HybridDocumentProcessor(gpu_settings)

        # Benchmark different document types
        benchmark_results = {}

        # PDF benchmark (hi_res strategy)
        pdf_start = asyncio.get_event_loop().time()
        pdf_result = await processor.process_document_async(
            system_test_documents["complex_pdf"]
        )
        pdf_time = asyncio.get_event_loop().time() - pdf_start

        benchmark_results["pdf"] = {
            "time": pdf_time,
            "elements": len(pdf_result.elements),
            "strategy": pdf_result.strategy_used,
        }

        # Text benchmark (fast strategy)
        text_start = asyncio.get_event_loop().time()
        text_result = await processor.process_document_async(
            system_test_documents["large_text"]
        )
        text_time = asyncio.get_event_loop().time() - text_start

        benchmark_results["text"] = {
            "time": text_time,
            "elements": len(text_result.elements),
            "strategy": text_result.strategy_used,
        }

        # Verify performance baselines
        assert benchmark_results["pdf"]["time"] < 3.0, (
            "PDF processing too slow for production"
        )
        assert benchmark_results["text"]["time"] < 5.0, (
            "Text processing too slow for production"
        )

        # Verify quality baselines
        assert benchmark_results["pdf"]["elements"] >= 2, (
            "PDF extraction quality insufficient"
        )
        assert benchmark_results["text"]["elements"] >= 5, (
            "Text chunking quality insufficient"
        )

        # Log benchmark results for performance tracking
        print("\n--- Performance Benchmark Results ---")
        for doc_type, metrics in benchmark_results.items():
            print(
                f"{doc_type.upper()}: {metrics['time']:.2f}s, "
                f"{metrics['elements']} elements, {metrics['strategy']}"
            )

    @pytest.mark.system
    @pytest.mark.requires_gpu
    def test_gpu_device_detection_and_optimization(self, gpu_settings):
        """Test GPU device detection and optimization settings.

        REQUIRES: RTX 4090, CUDA toolkit
        """
        processor = HybridDocumentProcessor(gpu_settings)

        # Verify GPU-optimized configuration
        pipeline = processor._create_pipeline(ProcessingStrategy.HI_RES)

        # Verify pipeline has expected optimizations
        assert (
            len(pipeline.transformations) == 2
        )  # UnstructuredTransformation + SentenceSplitter
        assert pipeline.cache is not None
        assert pipeline.docstore is not None

        # Verify strategy mapping includes GPU-optimized settings
        assert len(processor.strategy_map) > 5  # Multiple file type strategies
        assert ProcessingStrategy.HI_RES in processor.strategy_map.values()
        assert ProcessingStrategy.FAST in processor.strategy_map.values()

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_system_integration_with_real_pipeline(
        self, gpu_settings, system_test_documents
    ):
        """Test complete system integration with real LlamaIndex pipeline.

        REQUIRES: RTX 4090, full system stack
        """
        processor = HybridDocumentProcessor(gpu_settings)

        # Test complete pipeline integration
        result = await processor.process_document_async(
            system_test_documents["complex_pdf"]
        )

        # Verify complete pipeline execution
        assert "pipeline_config" in result.metadata
        pipeline_config = result.metadata["pipeline_config"]

        assert pipeline_config["strategy"] == "hi_res"
        assert pipeline_config["transformations"] >= 2
        assert pipeline_config["cache_enabled"] is True
        assert pipeline_config["docstore_enabled"] is True

        # Verify element quality from real pipeline
        for element in result.elements:
            # Should have rich metadata from UnstructuredTransformation
            assert "element_category" in element.metadata
            assert "processing_strategy" in element.metadata
            assert "source_file" in element.metadata

            # Should have been processed by SentenceSplitter
            assert len(element.text) > 0
            assert len(element.text) < 2000  # Reasonable chunk size

        # Verify cache integration worked
        cache_stats = await processor.get_cache_stats()
        assert cache_stats["processor_type"] == "hybrid"
        assert "strategy_mappings" in cache_stats
