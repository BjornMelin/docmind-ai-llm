"""System tests for BGE-M3 Embedder with real GPU models and hardware validation.

This module provides system-level tests for BGEM3Embedder that require actual
hardware, real model loading, and FlagEmbedding library integration.
These tests validate the complete embedding system under realistic conditions.

HARDWARE REQUIREMENTS:
- GPU: RTX 3060 (8GB) minimum, RTX 4060/4090 recommended
- RAM: 16GB minimum, 32GB recommended
- CUDA: 11.8+ support
- Storage: ~3GB for BGE-M3 model download
- Network: Initial model download required

Test Coverage:
- Real BGE-M3 model loading with FlagEmbedding
- Actual GPU-accelerated embedding generation
- BOTH dense AND sparse embeddings validation (ADR-002)
- Performance benchmarks with realistic targets
- Memory management under sustained load
- Batch processing optimization with available hardware
- Integration with document processing workflows
- GPU memory monitoring and cleanup validation

Following 3-tier testing strategy:
- Tier 3 (System): End-to-end tests with real models (<5min each)
- Requires GPU hardware and actual model loading
- Tests realistic performance targets based on available hardware
- Validates ADR-002 compliance with live models
- Includes proper cleanup and resource management
"""

import time
from contextlib import asynccontextmanager

import pytest
import torch
from loguru import logger

# Import models and classes with proper error handling
try:
    from src.config.settings import DocMindSettings
    from src.core.infrastructure.gpu_monitor import (
        GPUMetrics,
        gpu_performance_monitor,
    )
    from src.models.embeddings import EmbeddingParameters, EmbeddingResult
    from src.processing.embeddings.bgem3_embedder import BGEM3Embedder
except ImportError as e:
    logger.error(f"Import error in system tests: {e}")
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)

# Check for FlagEmbedding availability
try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    pytest.skip("FlagEmbedding not available for system tests", allow_module_level=True)

# Skip all tests if no GPU available
if not torch.cuda.is_available():
    pytest.skip("GPU not available for system tests", allow_module_level=True)


@pytest.fixture
def gpu_settings():
    """System test settings with GPU-optimized configuration."""
    from src.config.settings import app_settings

    return app_settings


@pytest.fixture
def system_parameters():
    """System test parameters optimized for real GPU performance."""
    return EmbeddingParameters(
        max_length=8192,  # Full 8K context
        batch_size_gpu=12,  # RTX 4090 optimized
        use_fp16=True,  # Enable FP16 acceleration
        return_dense=True,
        return_sparse=True,
        return_colbert=False,  # Disable for faster system tests
    )


@pytest.fixture
def system_test_texts():
    """Comprehensive text samples for system testing."""
    return [
        "DocMind AI leverages advanced BGE-M3 unified embeddings for document "
        "processing with both dense semantic similarity and sparse keyword matching "
        "capabilities. The system architecture combines unstructured.io for multimodal "
        "document parsing with LlamaIndex for pipeline orchestration and caching "
        "optimization.",
        "Performance targets include processing over one page per second with hi-res "
        "strategy while maintaining under 14GB VRAM usage on RTX 4090 hardware.",
        "Integration testing validates end-to-end functionality from document "
        "ingestion through embedding generation to vector storage and hybrid search.",
        "Quality metrics ensure over 99% text extraction accuracy, complete table "
        "structure preservation, and accurate OCR processing for image-based content.",
        "The embedding pipeline supports 8K context windows enabling processing of "
        "large document chunks with full semantic context preservation.",
        "Sparse embeddings provide learned token weights for precise keyword matching "
        "while dense embeddings capture deep semantic relationships.",
        "System validation includes GPU memory management, batch processing "
        "optimization, and sustained performance under production workloads.",
        "Cache integration provides dual-layer caching with LlamaIndex IngestionCache "
        "and SimpleCache for document processing persistence.",
        "Error handling mechanisms ensure graceful degradation under memory pressure "
        "with automatic fallback strategies and retry logic.",
    ]


@asynccontextmanager
async def gpu_memory_tracker():
    """Track GPU memory usage during test execution with proper cleanup.

    Yields:
        dict: Memory statistics including initial, peak, and final memory usage
    """
    if not torch.cuda.is_available():
        yield {"available": False}
        return

    # Clear cache before starting
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    initial_memory = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()

    stats = {
        "available": True,
        "initial_memory_gb": initial_memory / (1024**3),
        "initial_reserved_gb": initial_reserved / (1024**3),
    }

    try:
        yield stats
    finally:
        final_memory = torch.cuda.memory_allocated()
        final_reserved = torch.cuda.memory_reserved()
        peak_memory = torch.cuda.max_memory_allocated()

        stats.update(
            {
                "final_memory_gb": final_memory / (1024**3),
                "final_reserved_gb": final_reserved / (1024**3),
                "peak_memory_gb": peak_memory / (1024**3),
                "memory_used_gb": (final_memory - initial_memory) / (1024**3),
            }
        )

        logger.info(
            f"GPU Memory Stats - Used: {stats['memory_used_gb']:.2f}GB, "
            f"Peak: {stats['peak_memory_gb']:.2f}GB"
        )

        # Cleanup
        torch.cuda.empty_cache()


class TestBGEM3EmbedderSystemGPU:
    """System tests for BGEM3Embedder with real GPU hardware."""

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_real_bge_m3_model_loading_and_inference(
        self, gpu_settings, system_parameters
    ):
        """Test real BGE-M3 model loading and GPU-accelerated inference.

        REQUIRES: RTX 4090, FlagEmbedding library, BGE-M3 model download
        """
        embedder = BGEM3Embedder(gpu_settings, system_parameters)

        # Verify real model loaded successfully
        assert embedder.device == "cuda"
        assert embedder.batch_size == 12  # RTX 4090 optimized
        assert embedder.model is not None

        # Test real embedding generation
        test_text = "System test for real BGE-M3 model inference with GPU acceleration."
        result = await embedder.embed_single_text_async(test_text)

        # Verify real embedding properties
        assert isinstance(result, list)
        assert len(result) == 1024  # BGE-M3 dimension
        assert all(isinstance(val, float) for val in result)

        # Verify embeddings contain meaningful values (not all zeros/ones)
        assert not all(val == 0.0 for val in result)
        assert not all(val == 1.0 for val in result)
        assert min(result) != max(result)  # Should have variance

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_real_unified_dense_and_sparse_embeddings(
        self, gpu_settings, system_parameters, system_test_texts
    ):
        """Test real unified dense AND sparse embedding generation.

        REQUIRES: RTX 4090, real BGE-M3 model
        """
        embedder = BGEM3Embedder(gpu_settings, system_parameters)

        # Generate unified embeddings with real model
        result = await embedder.embed_texts_async(
            system_test_texts[:3]
        )  # 3 texts for system test

        # CRITICAL: Verify BOTH dense AND sparse embeddings from real model
        assert isinstance(result, EmbeddingResult)
        assert result.dense_embeddings is not None, (
            "Real dense embeddings missing - ADR-002 violation"
        )
        assert result.sparse_embeddings is not None, (
            "Real sparse embeddings missing - ADR-002 violation"
        )

        # Verify real dense embeddings structure
        assert len(result.dense_embeddings) == 3
        for dense_emb in result.dense_embeddings:
            assert isinstance(dense_emb, list)
            assert len(dense_emb) == 1024
            assert all(isinstance(val, float) for val in dense_emb)
            # Verify real embeddings have meaningful variance
            assert min(dense_emb) != max(dense_emb)
            assert abs(max(dense_emb) - min(dense_emb)) > 0.1

        # Verify real sparse embeddings structure
        assert len(result.sparse_embeddings) == 3
        for sparse_emb in result.sparse_embeddings:
            assert isinstance(sparse_emb, dict)
            assert len(sparse_emb) > 0  # Real model should generate sparse weights
            assert all(
                isinstance(k, int) and isinstance(v, float)
                for k, v in sparse_emb.items()
            )
            # Verify realistic sparse weights
            assert all(0.0 <= v <= 1.0 for v in sparse_emb.values())

        # Verify performance metrics
        assert result.processing_time > 0
        assert result.memory_usage_mb > 0
        assert result.model_info["dense_enabled"] is True
        assert result.model_info["sparse_enabled"] is True

    @pytest.mark.system
    @pytest.mark.requires_gpu
    def test_real_sparse_similarity_computation(self, gpu_settings, system_test_texts):
        """Test sparse similarity computation with real BGE-M3 model.

        REQUIRES: RTX 4090, real sparse embeddings
        """
        embedder = BGEM3Embedder(gpu_settings)

        # Generate real sparse embeddings
        sparse_embeddings = embedder.get_sparse_embeddings(system_test_texts[:2])
        assert sparse_embeddings is not None
        assert len(sparse_embeddings) == 2

        sparse1, sparse2 = sparse_embeddings

        # Test real sparse similarity computation
        similarity = embedder.compute_sparse_similarity(sparse1, sparse2)

        # Verify realistic similarity score
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0  # Valid similarity range

        # Test with identical sparse embeddings (should be high similarity)
        identical_similarity = embedder.compute_sparse_similarity(sparse1, sparse1)
        assert identical_similarity > similarity  # Self-similarity should be higher

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_gpu_performance_benchmarks(
        self, gpu_settings, system_parameters, system_test_texts
    ):
        """Test performance benchmarks with real GPU acceleration.

        REQUIRES: RTX 4090, performance validation
        """
        embedder = BGEM3Embedder(gpu_settings, system_parameters)

        # Benchmark single text embedding
        start_time = time.time()
        single_result = await embedder.embed_single_text_async(system_test_texts[0])
        single_time = time.time() - start_time

        # Benchmark batch embedding
        batch_start = time.time()
        batch_result = await embedder.embed_texts_async(system_test_texts[:5])
        batch_time = time.time() - batch_start

        # Verify performance targets
        assert single_time < 1.0, f"Single embedding took {single_time}s, too slow"
        assert batch_time < 3.0, f"Batch embedding took {batch_time}s, too slow"

        # Verify batch efficiency (should be more efficient per text)
        per_text_batch_time = batch_time / 5
        assert per_text_batch_time < single_time * 0.8, "Batch processing not efficient"

        # Verify embedding quality
        assert len(single_result) == 1024
        assert len(batch_result.dense_embeddings) == 5

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_sustained_gpu_memory_management(
        self, gpu_settings, system_test_texts
    ):
        """Test GPU memory management under sustained processing load.

        REQUIRES: RTX 4090, memory monitoring
        """
        embedder = BGEM3Embedder(gpu_settings)

        # Process multiple rounds to test memory management
        processing_rounds = 10
        processing_times = []

        for _round_num in range(processing_rounds):
            start_time = time.time()
            result = await embedder.embed_texts_async(system_test_texts[:3])
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Verify successful processing
            assert result.dense_embeddings is not None
            assert result.sparse_embeddings is not None
            assert len(result.dense_embeddings) == 3

        # Verify memory stability (no significant performance degradation)
        avg_early = sum(processing_times[:3]) / 3
        avg_late = sum(processing_times[-3:]) / 3
        performance_degradation = (avg_late - avg_early) / avg_early

        assert performance_degradation < 0.2, (
            f"Performance degraded {performance_degradation * 100:.1f}%, "
            "suggests memory issues"
        )

        # Verify performance statistics tracking
        stats = embedder.get_performance_stats()
        assert stats["total_texts_embedded"] == processing_rounds * 3
        assert stats["total_processing_time"] > 0

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_large_batch_processing_gpu_optimization(
        self, gpu_settings, system_test_texts
    ):
        """Test large batch processing with GPU memory optimization.

        REQUIRES: RTX 4090, batch processing optimization
        """
        embedder = BGEM3Embedder(gpu_settings)

        # Create larger text batch for stress testing
        large_batch = system_test_texts * 3  # 30 texts total

        start_time = time.time()
        result = await embedder.embed_texts_async(large_batch)
        processing_time = time.time() - start_time

        # Verify large batch processing
        assert len(result.dense_embeddings) == 30
        assert len(result.sparse_embeddings) == 30
        assert processing_time < 10.0, f"Large batch took {processing_time}s, too slow"

        # Verify all embeddings are valid
        for i, (dense_emb, sparse_emb) in enumerate(
            zip(result.dense_embeddings, result.sparse_embeddings, strict=False)
        ):
            assert len(dense_emb) == 1024, f"Invalid dense embedding at index {i}"
            assert isinstance(sparse_emb, dict), (
                f"Invalid sparse embedding at index {i}"
            )
            assert len(sparse_emb) > 0, f"Empty sparse embedding at index {i}"

        # Verify GPU memory efficiency
        assert result.memory_usage_mb > 0
        assert result.memory_usage_mb < 14000  # Should stay under 14GB for RTX 4090

    @pytest.mark.system
    @pytest.mark.requires_gpu
    def test_device_optimization_and_configuration(self, gpu_settings):
        """Test device optimization and hardware configuration.

        REQUIRES: RTX 4090, CUDA optimization
        """
        embedder = BGEM3Embedder(gpu_settings)

        # Verify GPU optimization settings
        assert embedder.device == "cuda"
        assert embedder.batch_size >= 8  # Should be optimized for RTX 4090
        assert embedder.parameters.use_fp16 is True

        # Verify model configuration
        assert embedder.model is not None

        # Test performance characteristics
        stats = embedder.get_performance_stats()
        assert stats["device"] == "cuda"
        assert stats["unified_embeddings_enabled"] is True
        assert stats["model_library"] == "FlagEmbedding.BGEM3FlagModel"

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_embedding_consistency_across_runs(
        self, gpu_settings, system_test_texts
    ):
        """Test embedding consistency across multiple runs with real model.

        REQUIRES: RTX 4090, deterministic model behavior
        """
        embedder = BGEM3Embedder(gpu_settings)

        test_text = system_test_texts[0]

        # Generate embeddings multiple times
        embeddings = []
        for _ in range(3):
            result = await embedder.embed_single_text_async(test_text)
            embeddings.append(result)

        # Verify consistency (real models should be deterministic with same input)
        for i in range(1, len(embeddings)):
            # Allow for small floating point differences
            for j, (val1, val2) in enumerate(
                zip(embeddings[0], embeddings[i], strict=False)
            ):
                diff = abs(val1 - val2)
                assert diff < 1e-6, f"Inconsistent embedding at position {j}: {diff}"

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_integration_with_document_processing_system(
        self, gpu_settings, system_test_texts
    ):
        """Test integration with document processing system workflows.

        REQUIRES: RTX 4090, full system integration
        """
        embedder = BGEM3Embedder(gpu_settings)

        # Simulate document processing workflow with various text types
        document_chunks = [
            "# Document Title\nThis is a title section from a processed document.",
            "## Section Header\nThis narrative text contains important information "
            "extracted from PDF.",
            "Table data: | Column 1 | Column 2 | Column 3 |\n|----------|----------|"
            "----------|",
            "Image caption: Chart showing performance metrics and system benchmarks.",
            "Footer text: End of document section with metadata and references.",
        ]

        # Process as document workflow would
        workflow_start = time.time()

        # Process in batches as document processor would
        batch_size = 3
        all_embeddings = []

        for i in range(0, len(document_chunks), batch_size):
            batch = document_chunks[i : i + batch_size]
            result = await embedder.embed_texts_async(batch)
            all_embeddings.extend(
                list(
                    zip(
                        result.dense_embeddings,
                        result.sparse_embeddings,
                        strict=False,
                    )
                )
            )

        workflow_time = time.time() - workflow_start

        # Verify workflow integration
        assert len(all_embeddings) == len(document_chunks)
        assert workflow_time < 5.0, f"Document workflow took {workflow_time}s, too slow"

        # Verify embedding quality for different content types
        for i, (dense_emb, sparse_emb) in enumerate(all_embeddings):
            assert len(dense_emb) == 1024, f"Invalid dense embedding for chunk {i}"
            assert isinstance(sparse_emb, dict), (
                f"Invalid sparse embedding for chunk {i}"
            )
            assert len(sparse_emb) > 0, f"No sparse features for chunk {i}"

            # Verify meaningful embeddings (different chunks should have different
            # embeddings)
            if i > 0:
                prev_dense = all_embeddings[i - 1][0]
                similarity = sum(
                    a * b
                    for a, b in zip(dense_emb[:100], prev_dense[:100], strict=False)
                )  # Quick similarity check
                assert abs(similarity) < 50, (
                    f"Chunks {i - 1} and {i} too similar, embeddings may be invalid"
                )

    @pytest.mark.system
    @pytest.mark.requires_gpu
    def test_model_unloading_and_cleanup(self, gpu_settings):
        """Test model unloading and GPU memory cleanup.

        REQUIRES: RTX 4090, memory management validation
        """
        embedder = BGEM3Embedder(gpu_settings)

        # Verify model is loaded
        assert embedder.model is not None

        # Test statistics before unloading
        embedder._embedding_count = 100
        embedder._total_processing_time = 10.0

        stats_before = embedder.get_performance_stats()
        assert stats_before["total_texts_embedded"] == 100

        # Test model unloading
        embedder.unload_model()

        # Verify cleanup
        stats_after = embedder.get_performance_stats()
        assert stats_after["total_texts_embedded"] == 0
        assert stats_after["total_processing_time"] == 0.0

    @pytest.mark.system
    @pytest.mark.requires_gpu
    @pytest.mark.asyncio
    async def test_real_world_production_simulation(
        self, gpu_settings, system_test_texts
    ):
        """Test production-like workload simulation with real performance targets.

        REQUIRES: RTX 4090, production performance validation
        """
        embedder = BGEM3Embedder(gpu_settings)

        # Simulate production workload patterns
        production_tasks = [
            system_test_texts[:2],  # Small batch
            system_test_texts[2:5],  # Medium batch
            system_test_texts[5:8],  # Medium batch
            system_test_texts[8:],  # Remaining batch
        ]

        total_start = time.time()
        total_texts = 0
        all_results = []

        for batch in production_tasks:
            batch_start = time.time()
            result = await embedder.embed_texts_async(batch)
            batch_time = time.time() - batch_start

            all_results.append((result, batch_time))
            total_texts += len(batch)

            # Verify each batch meets production requirements
            assert batch_time < 2.0, f"Batch processing too slow: {batch_time}s"
            assert result.dense_embeddings is not None
            assert result.sparse_embeddings is not None

        total_time = time.time() - total_start

        # Verify overall production performance
        assert total_time < 8.0, (
            f"Total production simulation took {total_time}s, too slow"
        )
        assert total_texts == len(system_test_texts)

        # Verify sustained performance
        batch_times = [batch_time for _, batch_time in all_results]
        performance_variance = (max(batch_times) - min(batch_times)) / min(batch_times)
        assert performance_variance < 0.5, (
            f"Performance variance {performance_variance * 100:.1f}% too high for "
            "production"
        )

        # Verify final performance statistics
        final_stats = embedder.get_performance_stats()
        assert final_stats["total_texts_embedded"] == total_texts
        assert final_stats["avg_time_per_text_ms"] > 0
