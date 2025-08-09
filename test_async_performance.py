#!/usr/bin/env python3
"""Performance test for async optimizations in DocMind AI.

This script tests the async performance optimizations implemented in Phase 3
to verify that they provide the required 50%+ performance improvement for
multi-document processing scenarios.

Features tested:
- Parallel embedding generation with asyncio.gather()
- Parallel document processing with concurrency limits
- Optimized index creation pipeline
- Connection pooling for Qdrant
- Batch operations for vector stores

Usage:
    python test_async_performance.py
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from llama_index.core import Document

from models import AppSettings
from utils.document_loader import load_documents_parallel, load_documents_unstructured
from utils.index_builder import (
    create_index_async,
    create_index_async_optimized,
    generate_embeddings_parallel,
)
from utils.logging_config import setup_logging
from utils.utils import EnhancedPerformanceMonitor

# Setup
settings = AppSettings()
setup_logging()
logger = logging.getLogger(__name__)


async def create_test_documents(count: int = 50) -> list[Document]:
    """Create test documents for performance testing."""
    documents = []
    base_text = """
    This is a test document for performance benchmarking. It contains enough text
    to simulate realistic document processing scenarios. The document includes
    multiple paragraphs with various content types to test the embedding generation
    and indexing performance effectively. This content is replicated to create
    documents of sufficient length for meaningful performance testing.
    """

    for i in range(count):
        # Vary document length to simulate real-world scenarios
        multiplier = (i % 5) + 1
        text = base_text * multiplier

        doc = Document(
            text=f"Document {i + 1}: {text}",
            metadata={
                "doc_id": i + 1,
                "source": f"test_doc_{i + 1}.txt",
                "length": len(text),
                "type": "test_document",
            },
        )
        documents.append(doc)

    logger.info(f"Created {len(documents)} test documents")
    return documents


async def benchmark_sequential_processing(documents: list[Document]) -> dict[str, Any]:
    """Benchmark sequential document processing (baseline)."""
    logger.info("Starting sequential processing benchmark...")
    monitor = EnhancedPerformanceMonitor()

    async with monitor.measure("sequential_embedding_generation"):
        # Simulate sequential embedding generation
        from utils.embedding_factory import EmbeddingFactory

        embed_model = EmbeddingFactory.create_dense_embedding(
            use_gpu=settings.gpu_acceleration
        )

        all_embeddings = []
        for doc in documents:
            embeddings = await asyncio.to_thread(embed_model.embed, [doc.text])
            all_embeddings.extend(embeddings)

    async with monitor.measure("sequential_index_creation"):
        # Use the standard async index creation
        result = await create_index_async(documents, use_gpu=settings.gpu_acceleration)

    report = monitor.get_report()
    logger.info("Sequential processing completed")
    return {
        "method": "sequential",
        "total_time": report["summary"]["total_time"],
        "embedding_time": monitor.metrics["sequential_embedding_generation"][
            "duration_seconds"
        ],
        "index_time": monitor.metrics["sequential_index_creation"]["duration_seconds"],
        "document_count": len(documents),
        "performance_report": report,
    }


async def benchmark_parallel_processing(documents: list[Document]) -> dict[str, Any]:
    """Benchmark parallel document processing (optimized)."""
    logger.info("Starting parallel processing benchmark...")
    monitor = EnhancedPerformanceMonitor()

    async with monitor.measure("parallel_embedding_generation"):
        # Use optimized parallel embedding generation
        from utils.embedding_factory import EmbeddingFactory

        embed_model = EmbeddingFactory.create_dense_embedding(
            use_gpu=settings.gpu_acceleration
        )

        embeddings = await generate_embeddings_parallel(
            documents, embed_model, batch_size=16
        )

    async with monitor.measure("parallel_index_creation"):
        # Use optimized parallel index creation
        result = await create_index_async_optimized(
            documents, use_gpu=settings.gpu_acceleration
        )

    report = monitor.get_report()
    logger.info("Parallel processing completed")
    return {
        "method": "parallel",
        "total_time": report["summary"]["total_time"],
        "embedding_time": monitor.metrics["parallel_embedding_generation"][
            "duration_seconds"
        ],
        "index_time": monitor.metrics["parallel_index_creation"]["duration_seconds"],
        "document_count": len(documents),
        "performance_report": report,
    }


async def benchmark_document_loading() -> dict[str, Any]:
    """Benchmark document loading performance."""
    logger.info("Starting document loading benchmark...")

    # Create some test files
    test_files = []
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)

    try:
        # Create test files
        for i in range(10):
            file_path = test_dir / f"test_doc_{i + 1}.txt"
            with open(file_path, "w") as f:
                f.write(f"Test document {i + 1} content. " * 100)
            test_files.append(str(file_path))

        monitor = EnhancedPerformanceMonitor()

        # Sequential loading
        async with monitor.measure("sequential_document_loading"):
            seq_docs = []
            for file_path in test_files:
                docs = load_documents_unstructured(file_path)
                seq_docs.extend(docs)

        # Parallel loading
        async with monitor.measure("parallel_document_loading"):
            par_docs = await load_documents_parallel(test_files, max_concurrent=5)

        report = monitor.get_report()

        return {
            "sequential_time": monitor.metrics["sequential_document_loading"][
                "duration_seconds"
            ],
            "parallel_time": monitor.metrics["parallel_document_loading"][
                "duration_seconds"
            ],
            "sequential_docs": len(seq_docs),
            "parallel_docs": len(par_docs),
            "improvement": (
                monitor.metrics["sequential_document_loading"]["duration_seconds"]
                - monitor.metrics["parallel_document_loading"]["duration_seconds"]
            )
            / monitor.metrics["sequential_document_loading"]["duration_seconds"]
            * 100,
            "performance_report": report,
        }

    finally:
        # Cleanup test files
        for file_path in test_files:
            try:
                Path(file_path).unlink()
            except Exception:
                pass
        test_dir.rmdir()


def calculate_performance_improvement(
    sequential: dict[str, Any], parallel: dict[str, Any]
) -> dict[str, Any]:
    """Calculate performance improvement metrics."""
    seq_total = sequential["total_time"]
    par_total = parallel["total_time"]

    improvement_pct = (seq_total - par_total) / seq_total * 100
    speedup_factor = seq_total / par_total

    seq_embed = sequential["embedding_time"]
    par_embed = parallel["embedding_time"]
    embed_improvement = (seq_embed - par_embed) / seq_embed * 100

    seq_index = sequential["index_time"]
    par_index = parallel["index_time"]
    index_improvement = (seq_index - par_index) / seq_index * 100

    return {
        "total_improvement_pct": improvement_pct,
        "speedup_factor": speedup_factor,
        "embedding_improvement_pct": embed_improvement,
        "index_improvement_pct": index_improvement,
        "sequential_total_time": seq_total,
        "parallel_total_time": par_total,
        "target_met": improvement_pct >= 50.0,
    }


async def run_performance_tests():
    """Run comprehensive performance tests."""
    logger.info("🚀 Starting DocMind AI Async Performance Tests")
    logger.info("=" * 60)

    # Test 1: Document loading performance
    logger.info("Test 1: Document Loading Performance")
    doc_loading_results = await benchmark_document_loading()
    logger.info(f"Sequential loading: {doc_loading_results['sequential_time']:.2f}s")
    logger.info(f"Parallel loading: {doc_loading_results['parallel_time']:.2f}s")
    logger.info(f"Loading improvement: {doc_loading_results['improvement']:.1f}%")
    logger.info("-" * 40)

    # Test 2: Multi-document processing performance
    logger.info("Test 2: Multi-Document Processing Performance")
    documents = await create_test_documents(30)  # Reduced for faster testing

    # Run sequential benchmark
    sequential_results = await benchmark_sequential_processing(documents)
    logger.info(f"Sequential processing: {sequential_results['total_time']:.2f}s")

    # Run parallel benchmark
    parallel_results = await benchmark_parallel_processing(documents)
    logger.info(f"Parallel processing: {parallel_results['total_time']:.2f}s")

    # Calculate improvements
    improvements = calculate_performance_improvement(
        sequential_results, parallel_results
    )

    logger.info("-" * 40)
    logger.info("📊 PERFORMANCE RESULTS")
    logger.info("=" * 60)
    logger.info(
        f"Total Performance Improvement: {improvements['total_improvement_pct']:.1f}%"
    )
    logger.info(f"Speedup Factor: {improvements['speedup_factor']:.2f}x")
    logger.info(
        f"Embedding Generation Improvement: {improvements['embedding_improvement_pct']:.1f}%"
    )
    logger.info(
        f"Index Creation Improvement: {improvements['index_improvement_pct']:.1f}%"
    )
    logger.info(
        f"Document Loading Improvement: {doc_loading_results['improvement']:.1f}%"
    )
    logger.info("-" * 40)

    # Check if target is met
    if improvements["target_met"]:
        logger.info("✅ SUCCESS: 50%+ performance improvement achieved!")
        logger.info("   Target: 50% improvement")
        logger.info(
            f"   Actual: {improvements['total_improvement_pct']:.1f}% improvement"
        )
    else:
        logger.warning("❌ TARGET NOT MET: Performance improvement below 50%")
        logger.warning("   Target: 50% improvement")
        logger.warning(
            f"   Actual: {improvements['total_improvement_pct']:.1f}% improvement"
        )

    logger.info("=" * 60)

    return {
        "document_loading": doc_loading_results,
        "sequential_processing": sequential_results,
        "parallel_processing": parallel_results,
        "improvements": improvements,
        "target_met": improvements["target_met"],
    }


if __name__ == "__main__":
    try:
        results = asyncio.run(run_performance_tests())

        # Print final summary
        print("\n" + "=" * 60)
        print("DOCMIND AI ASYNC PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        print(
            f"Total Performance Improvement: {results['improvements']['total_improvement_pct']:.1f}%"
        )
        print(
            f"Target Achievement: {'✅ PASSED' if results['target_met'] else '❌ FAILED'}"
        )
        print("=" * 60)

    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise
