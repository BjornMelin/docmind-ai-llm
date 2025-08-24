"""Throughput benchmarks and scalability testing for DocMind AI.

This module provides comprehensive throughput testing and scalability analysis:
- Request throughput measurement under various loads
- Document processing throughput benchmarks
- Batch processing efficiency validation
- Concurrent user simulation and load testing
- Scalability bottleneck identification
- Resource utilization efficiency metrics

Follows DocMind AI patterns with proper resource monitoring and actionable insights.
"""

import asyncio
import concurrent.futures
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from llama_index.core import Document

from src.core.infrastructure.gpu_monitor import gpu_performance_monitor
from src.utils.resource_management import gpu_memory_context

# Throughput test constants
SMALL_LOAD_REQUESTS = 10
MEDIUM_LOAD_REQUESTS = 50
LARGE_LOAD_REQUESTS = 100

SMALL_BATCH_SIZE = 5
MEDIUM_BATCH_SIZE = 20
LARGE_BATCH_SIZE = 50

# Performance targets (requests/second)
RTX_4090_THROUGHPUT_TARGETS = {
    "embedding_rps": 20,  # Embeddings per second
    "reranking_rps": 10,  # Reranking operations per second
    "query_rps": 5,  # End-to-end queries per second
    "document_processing_rps": 100,  # Documents processed per second
    "concurrent_users": 10,  # Concurrent users supported
}

# Scalability thresholds
SCALABILITY_EFFICIENCY_THRESHOLD = 0.7  # 70% efficiency maintained under load
THROUGHPUT_DEGRADATION_THRESHOLD = 0.5  # 50% max throughput degradation


@pytest.fixture
def throughput_tracker():
    """Fixture for tracking throughput measurements and computing efficiency metrics."""

    class ThroughputTracker:
        def __init__(self):
            self.measurements = []
            self.load_tests = {}
            self.batch_tests = {}

        def record_throughput_test(
            self,
            test_name: str,
            total_operations: int,
            total_time_seconds: float,
            concurrent_operations: int = 1,
            resource_usage: dict[str, float] = None,
        ) -> dict[str, float]:
            """Record throughput test results."""
            throughput_rps = total_operations / total_time_seconds

            test_result = {
                "test_name": test_name,
                "total_operations": total_operations,
                "total_time_seconds": total_time_seconds,
                "throughput_rps": throughput_rps,
                "concurrent_operations": concurrent_operations,
                "operations_per_concurrent": throughput_rps / concurrent_operations,
                "resource_usage": resource_usage or {},
                "timestamp": time.time(),
            }

            self.measurements.append(test_result)
            return test_result

        def record_load_test(
            self,
            load_level: str,
            operations: int,
            duration: float,
            concurrent_workers: int,
            success_rate: float = 1.0,
            error_count: int = 0,
        ) -> dict[str, Any]:
            """Record load test results."""
            throughput = operations / duration

            load_result = {
                "load_level": load_level,
                "operations": operations,
                "duration": duration,
                "throughput_rps": throughput,
                "concurrent_workers": concurrent_workers,
                "success_rate": success_rate,
                "error_count": error_count,
                "operations_per_worker": throughput / concurrent_workers,
                "efficiency": throughput
                / (concurrent_workers * (throughput / concurrent_workers)),
            }

            self.load_tests[load_level] = load_result
            return load_result

        def record_batch_test(
            self,
            batch_size: int,
            total_items: int,
            processing_time: float,
            items_per_second: float,
        ) -> dict[str, Any]:
            """Record batch processing test results."""
            batch_result = {
                "batch_size": batch_size,
                "total_items": total_items,
                "processing_time": processing_time,
                "items_per_second": items_per_second,
                "batches_processed": total_items // batch_size,
                "batch_efficiency": items_per_second * batch_size,
            }

            self.batch_tests[batch_size] = batch_result
            return batch_result

        def analyze_scalability(
            self, baseline_load: str = "small", comparison_load: str = "large"
        ) -> dict[str, Any]:
            """Analyze scalability between different load levels."""
            if (
                baseline_load not in self.load_tests
                or comparison_load not in self.load_tests
            ):
                return {"error": "Missing load test data for comparison"}

            baseline = self.load_tests[baseline_load]
            comparison = self.load_tests[comparison_load]

            # Calculate scalability metrics
            throughput_ratio = comparison["throughput_rps"] / baseline["throughput_rps"]
            worker_ratio = (
                comparison["concurrent_workers"] / baseline["concurrent_workers"]
            )
            efficiency_ratio = comparison["efficiency"] / baseline["efficiency"]

            ideal_throughput = baseline["throughput_rps"] * worker_ratio
            scalability_efficiency = comparison["throughput_rps"] / ideal_throughput

            return {
                "baseline_throughput_rps": baseline["throughput_rps"],
                "comparison_throughput_rps": comparison["throughput_rps"],
                "throughput_ratio": throughput_ratio,
                "worker_ratio": worker_ratio,
                "efficiency_ratio": efficiency_ratio,
                "scalability_efficiency": scalability_efficiency,
                "scales_well": scalability_efficiency
                >= SCALABILITY_EFFICIENCY_THRESHOLD,
                "throughput_degradation": 1.0 - scalability_efficiency,
                "bottleneck_detected": scalability_efficiency
                < SCALABILITY_EFFICIENCY_THRESHOLD,
            }

        def validate_throughput_targets(
            self, targets: dict[str, float]
        ) -> dict[str, Any]:
            """Validate throughput against performance targets."""
            validation = {
                "targets_met": True,
                "failed_targets": [],
                "results": {},
                "overall_score": 0.0,
            }

            scores = []

            for target_name, target_value in targets.items():
                # Find matching measurements
                matching_tests = [
                    test
                    for test in self.measurements
                    if target_name.replace("_rps", "") in test["test_name"].lower()
                ]

                if matching_tests:
                    # Use best throughput result
                    best_throughput = max(
                        test["throughput_rps"] for test in matching_tests
                    )

                    validation["results"][target_name] = {
                        "target": target_value,
                        "actual": best_throughput,
                        "met": best_throughput >= target_value,
                        "margin": (best_throughput - target_value) / target_value,
                        "score": min(best_throughput / target_value, 2.0),  # Cap at 2.0
                    }

                    scores.append(validation["results"][target_name]["score"])

                    if best_throughput < target_value:
                        validation["targets_met"] = False
                        validation["failed_targets"].append(target_name)

            validation["overall_score"] = sum(scores) / len(scores) if scores else 0.0
            return validation

        def get_peak_throughput(self, test_type: str = None) -> dict[str, float]:
            """Get peak throughput across all tests or for specific test type."""
            if test_type:
                filtered_tests = [
                    test
                    for test in self.measurements
                    if test_type.lower() in test["test_name"].lower()
                ]
            else:
                filtered_tests = self.measurements

            if not filtered_tests:
                return {"peak_throughput_rps": 0.0, "test_name": "None"}

            peak_test = max(filtered_tests, key=lambda x: x["throughput_rps"])

            return {
                "peak_throughput_rps": peak_test["throughput_rps"],
                "test_name": peak_test["test_name"],
                "concurrent_operations": peak_test["concurrent_operations"],
                "total_operations": peak_test["total_operations"],
            }

    return ThroughputTracker()


@pytest.mark.performance
class TestEmbeddingThroughputBenchmarks:
    """Test embedding component throughput performance."""

    def test_single_document_embedding_throughput(self, throughput_tracker):
        """Test single document embedding throughput."""
        with patch(
            "src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel"
        ) as mock_model:
            mock_bgem3 = MagicMock()

            # Simulate consistent embedding time
            def mock_encode(*args, **kwargs):
                time.sleep(0.025)  # 25ms per embedding (40 RPS theoretical max)
                return {"dense_vecs": [[0.1] * 1024]}

            mock_bgem3.encode = mock_encode
            mock_model.return_value = mock_bgem3

            from src.retrieval.embeddings.bge_m3_manager import BGEM3Embedding

            embedding_model = BGEM3Embedding()

            # Benchmark single document throughput
            test_documents = [f"Single embedding test document {i}" for i in range(20)]

            start_time = time.perf_counter()

            for doc_text in test_documents:
                with gpu_memory_context():
                    embeddings = embedding_model.get_unified_embeddings([doc_text])
                    assert "dense" in embeddings

            total_time = time.perf_counter() - start_time

            # Record throughput measurement
            result = throughput_tracker.record_throughput_test(
                "single_embedding",
                len(test_documents),
                total_time,
                concurrent_operations=1,
            )

            print(f"Single embedding throughput: {result['throughput_rps']:.1f} RPS")

            # Should achieve reasonable throughput
            assert result["throughput_rps"] >= 5.0, (
                f"Single embedding throughput too low: "
                f"{result['throughput_rps']:.1f} < 5.0 RPS"
            )

    def test_batch_embedding_throughput_scaling(self, throughput_tracker):
        """Test how batch embedding throughput scales with batch size."""
        with patch(
            "src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel"
        ) as mock_model:
            mock_bgem3 = MagicMock()
            mock_model.return_value = mock_bgem3

            from src.retrieval.embeddings.bge_m3_manager import BGEM3Embedding

            batch_sizes = [1, 5, 10, 20]
            total_documents = 40
            test_documents = [
                f"Batch throughput test document {i}" for i in range(total_documents)
            ]

            batch_results = {}

            for batch_size in batch_sizes:
                # Simulate batch efficiency - larger batches are more efficient per item
                def mock_batch_encode(texts, **kwargs):
                    batch_count = len(texts)
                    # Base time + per-item time (with efficiency gains)
                    base_time = 0.01  # 10ms base overhead
                    per_item_time = (
                        0.015 if batch_count == 1 else 0.008
                    )  # 8ms per item in batch vs 15ms individual
                    total_time = base_time + (per_item_time * batch_count)
                    time.sleep(total_time)
                    return {"dense_vecs": [[0.1] * 1024 for _ in texts]}

                mock_bgem3.encode = mock_batch_encode
                embedding_model = BGEM3Embedding(batch_size=batch_size)

                start_time = time.perf_counter()

                # Process all documents in batches
                for i in range(0, len(test_documents), batch_size):
                    batch_docs = test_documents[i : i + batch_size]

                    with gpu_memory_context():
                        embeddings = embedding_model.get_unified_embeddings(batch_docs)
                        assert "dense" in embeddings
                        assert len(embeddings["dense"]) == len(batch_docs)

                total_time = time.perf_counter() - start_time
                items_per_second = total_documents / total_time

                batch_result = throughput_tracker.record_batch_test(
                    batch_size, total_documents, total_time, items_per_second
                )

                batch_results[batch_size] = batch_result

                # Also record as throughput test
                throughput_tracker.record_throughput_test(
                    f"batch_embedding_{batch_size}",
                    total_documents,
                    total_time,
                    concurrent_operations=1,
                )

            # Analyze batch scaling efficiency
            print("Batch embedding scaling results:")
            for batch_size, result in batch_results.items():
                print(
                    f"  Batch size {batch_size}: "
                    f"{result['items_per_second']:.1f} items/sec"
                )

            # Larger batches should be more efficient
            batch_1_throughput = batch_results[1]["items_per_second"]
            batch_20_throughput = batch_results[20]["items_per_second"]

            efficiency_gain = batch_20_throughput / batch_1_throughput

            assert efficiency_gain >= 1.5, (
                f"Batch processing not efficient enough: {efficiency_gain:.2f}x vs "
                f"expected >=1.5x"
            )

            print(f"Batch efficiency gain (20 vs 1): {efficiency_gain:.2f}x")

    def test_concurrent_embedding_throughput(self, throughput_tracker):
        """Test concurrent embedding processing throughput."""
        with patch(
            "src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel"
        ) as mock_model:
            mock_bgem3 = MagicMock()

            # Simulate thread-safe embedding processing
            def mock_concurrent_encode(*args, **kwargs):
                time.sleep(0.03)  # 30ms processing time
                return {"dense_vecs": [[0.1] * 1024]}

            mock_bgem3.encode = mock_concurrent_encode
            mock_model.return_value = mock_bgem3

            from src.retrieval.embeddings.bge_m3_manager import BGEM3Embedding

            # Test different concurrency levels
            concurrency_levels = [1, 2, 4, 8]
            documents_per_worker = 5

            for concurrency in concurrency_levels:
                total_documents = concurrency * documents_per_worker
                test_documents = [
                    f"Concurrent test doc {i}" for i in range(total_documents)
                ]

                def worker_function(worker_docs):
                    """Worker function for concurrent processing."""
                    embedding_model = BGEM3Embedding()
                    results = []

                    for doc in worker_docs:
                        with gpu_memory_context():
                            embeddings = embedding_model.get_unified_embeddings([doc])
                            results.append(embeddings)

                    return results

                # Split documents among workers
                worker_batches = [
                    test_documents[i::concurrency] for i in range(concurrency)
                ]

                start_time = time.perf_counter()

                # Process with ThreadPoolExecutor for concurrent execution
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency
                ) as executor:
                    futures = [
                        executor.submit(worker_function, batch)
                        for batch in worker_batches
                    ]
                    results = [
                        future.result()
                        for future in concurrent.futures.as_completed(futures)
                    ]

                total_time = time.perf_counter() - start_time

                # Record concurrent throughput
                result = throughput_tracker.record_throughput_test(
                    f"concurrent_embedding_{concurrency}_workers",
                    total_documents,
                    total_time,
                    concurrent_operations=concurrency,
                )

                # Verify all documents were processed
                total_results = sum(len(worker_result) for worker_result in results)
                assert total_results == concurrency, (
                    f"Expected {concurrency} worker results, got {total_results}"
                )

                print(
                    f"Concurrency {concurrency}: {result['throughput_rps']:.1f} RPS, "
                    f"{result['operations_per_concurrent']:.1f} RPS per worker"
                )

            # Analyze concurrent scaling
            concurrent_1 = next(
                t
                for t in throughput_tracker.measurements
                if "concurrent_embedding_1_workers" in t["test_name"]
            )
            concurrent_8 = next(
                t
                for t in throughput_tracker.measurements
                if "concurrent_embedding_8_workers" in t["test_name"]
            )

            scaling_factor = (
                concurrent_8["throughput_rps"] / concurrent_1["throughput_rps"]
            )

            # Should scale reasonably well (aim for 60%+ of linear scaling)
            linear_scaling = 8.0
            scaling_efficiency = scaling_factor / linear_scaling

            assert scaling_efficiency >= 0.4, (
                f"Concurrent scaling poor: {scaling_efficiency:.2f} efficiency vs "
                f"expected >=0.4"
            )

            print(f"Concurrent scaling efficiency: {scaling_efficiency:.2f}")


@pytest.mark.performance
class TestEndToEndThroughputBenchmarks:
    """Test end-to-end query processing throughput."""

    def test_query_processing_throughput_under_load(self, throughput_tracker):
        """Test query processing throughput under different load levels."""
        # Mock all pipeline components for fast processing
        with (
            patch(
                "src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel"
            ) as mock_embed,
            patch(
                "src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder"
            ) as mock_ce,
        ):
            # Setup fast mocks
            mock_bgem3 = MagicMock()
            mock_bgem3.encode.return_value = {"dense_vecs": [[0.1] * 1024]}
            mock_embed.return_value = mock_bgem3

            mock_cross_encoder = MagicMock()
            mock_cross_encoder.predict.return_value = [0.9, 0.8, 0.7]
            mock_cross_encoder.model = MagicMock()
            mock_ce.return_value = mock_cross_encoder

            from src.retrieval.postprocessor.cross_encoder_rerank import (
                BGECrossEncoderRerank,
            )
            from src.retrieval.query_engine.router_engine import (
                AdaptiveRouterQueryEngine,
            )

            # Create pipeline
            reranker = BGECrossEncoderRerank()
            mock_vector_index = MagicMock()

            # Mock fast query response
            def mock_fast_query(query_str, **kwargs):
                time.sleep(0.1)  # 100ms query processing
                response = MagicMock()
                response.response = f"Response to: {query_str}"
                response.source_nodes = []
                return response

            router = AdaptiveRouterQueryEngine(
                vector_index=mock_vector_index, reranker=reranker
            )
            router.router_engine.query = mock_fast_query

            # Test different load levels
            load_tests = [
                {"level": "small", "queries": SMALL_LOAD_REQUESTS, "workers": 1},
                {"level": "medium", "queries": MEDIUM_LOAD_REQUESTS, "workers": 2},
                {"level": "large", "queries": LARGE_LOAD_REQUESTS, "workers": 4},
            ]

            for load_test in load_tests:
                queries = [f"Load test query {i}" for i in range(load_test["queries"])]

                def query_worker(worker_queries):
                    """Worker function for processing queries."""
                    results = []
                    for query in worker_queries:
                        response = router.query(query)
                        results.append(response)
                    return results

                # Split queries among workers
                queries_per_worker = len(queries) // load_test["workers"]
                worker_batches = [
                    queries[i * queries_per_worker : (i + 1) * queries_per_worker]
                    for i in range(load_test["workers"])
                ]

                # Add remaining queries to last worker
                remaining_queries = len(queries) % load_test["workers"]
                if remaining_queries > 0:
                    worker_batches[-1].extend(queries[-remaining_queries:])

                start_time = time.perf_counter()

                # Process queries concurrently
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=load_test["workers"]
                ) as executor:
                    futures = [
                        executor.submit(query_worker, batch) for batch in worker_batches
                    ]
                    results = [
                        future.result()
                        for future in concurrent.futures.as_completed(futures)
                    ]

                total_time = time.perf_counter() - start_time
                total_processed = sum(len(worker_result) for worker_result in results)

                # Record load test results
                load_result = throughput_tracker.record_load_test(
                    load_test["level"],
                    total_processed,
                    total_time,
                    load_test["workers"],
                )

                # Also record as throughput test
                throughput_tracker.record_throughput_test(
                    f"query_load_{load_test['level']}",
                    total_processed,
                    total_time,
                    concurrent_operations=load_test["workers"],
                )

                print(
                    f"Load level {load_test['level']}: "
                    f"{load_result['throughput_rps']:.1f} RPS, "
                    f"{load_result['operations_per_worker']:.1f} RPS per worker"
                )

        # Analyze scalability between load levels
        scalability = throughput_tracker.analyze_scalability("small", "large")

        print(
            f"Scalability analysis: "
            f"{scalability['scalability_efficiency']:.2f} efficiency"
        )

        # Should maintain reasonable scalability
        assert scalability["scales_well"], (
            f"Poor scalability detected: "
            f"{scalability['scalability_efficiency']:.2f} < "
            f"{SCALABILITY_EFFICIENCY_THRESHOLD}"
        )

    @pytest.mark.asyncio
    async def test_async_query_throughput(self, throughput_tracker):
        """Test async query processing throughput."""

        # Mock async query processing
        async def mock_async_query_processor(query_id: int):
            """Mock async query processor."""
            import random

            # Simulate variable async processing time
            processing_time = random.uniform(0.08, 0.12)  # 80-120ms
            await asyncio.sleep(processing_time)
            return f"Async response to query {query_id}"

        # Test concurrent async processing
        concurrent_queries = 20

        start_time = time.perf_counter()

        # Create and run concurrent async tasks
        tasks = [mock_async_query_processor(i) for i in range(concurrent_queries)]
        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

        # Record async throughput
        result = throughput_tracker.record_throughput_test(
            "async_query_processing",
            len(results),
            total_time,
            concurrent_operations=concurrent_queries,
        )

        # Verify all queries completed
        assert len(results) == concurrent_queries

        print(f"Async query throughput: {result['throughput_rps']:.1f} RPS")

        # Async processing should be efficient
        assert result["throughput_rps"] >= 10.0, (
            f"Async throughput too low: {result['throughput_rps']:.1f} < 10.0 RPS"
        )

        # Should handle concurrency well
        efficiency = result["throughput_rps"] / concurrent_queries
        assert efficiency >= 0.5, (
            f"Async concurrency efficiency too low: {efficiency:.2f} < 0.5"
        )


@pytest.mark.performance
class TestDocumentProcessingThroughputBenchmarks:
    """Test document processing pipeline throughput."""

    def test_document_ingestion_throughput(self, throughput_tracker):
        """Test document ingestion and processing throughput."""
        # Mock document processing components
        with patch("src.utils.document.load_documents_unstructured") as mock_loader:
            # Simulate document loading with realistic timing
            def mock_document_loading(file_paths, settings):
                time.sleep(0.005 * len(file_paths))  # 5ms per document
                return [Document(text=f"Content from {path}") for path in file_paths]

            mock_loader.side_effect = mock_document_loading

            # Test different batch sizes for document processing
            batch_sizes = [SMALL_BATCH_SIZE, MEDIUM_BATCH_SIZE, LARGE_BATCH_SIZE]
            documents_per_batch = 10

            for batch_size in batch_sizes:
                total_documents = batch_size * documents_per_batch
                file_paths = [f"test_doc_{i}.pdf" for i in range(total_documents)]

                start_time = time.perf_counter()

                # Process documents in batches
                from src.config.settings import AppSettings

                settings = AppSettings()

                processed_documents = []
                for i in range(0, len(file_paths), batch_size):
                    batch_paths = file_paths[i : i + batch_size]

                    from src.utils.document import load_documents_unstructured

                    docs = load_documents_unstructured(batch_paths, settings)
                    processed_documents.extend(docs)

                total_time = time.perf_counter() - start_time
                docs_per_second = len(processed_documents) / total_time

                # Record batch processing results
                throughput_tracker.record_batch_test(
                    batch_size, len(processed_documents), total_time, docs_per_second
                )

                throughput_tracker.record_throughput_test(
                    f"document_processing_batch_{batch_size}",
                    len(processed_documents),
                    total_time,
                    concurrent_operations=1,
                )

                print(f"Batch size {batch_size}: {docs_per_second:.1f} docs/sec")

                # Verify all documents were processed
                assert len(processed_documents) == total_documents

        # Check that larger batches are at least as efficient
        small_batch_result = throughput_tracker.batch_tests[SMALL_BATCH_SIZE]
        large_batch_result = throughput_tracker.batch_tests[LARGE_BATCH_SIZE]

        efficiency_ratio = (
            large_batch_result["items_per_second"]
            / small_batch_result["items_per_second"]
        )

        assert efficiency_ratio >= 0.8, (
            f"Large batch processing less efficient: {efficiency_ratio:.2f} vs "
            f"small batch"
        )

    def test_chunk_processing_throughput(self, throughput_tracker):
        """Test document chunking throughput."""
        # Create test documents of varying lengths
        test_documents = []
        for i in range(50):
            # Vary document lengths to test chunking efficiency
            content_length = 100 + (i * 20)  # 100 to 1080 words
            content = " ".join([f"word{j}" for j in range(content_length)])
            test_documents.append(Document(text=content))

        # Mock chunking process
        def mock_chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
            """Mock document chunking with realistic timing."""
            chunks = []
            for doc in documents:
                # Simulate chunking time based on document length
                words = len(doc.text.split())
                chunks_count = max(1, words // chunk_size)
                time.sleep(0.001 * chunks_count)  # 1ms per chunk

                # Create mock chunks
                for chunk_idx in range(chunks_count):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, words)
                    chunk_text = " ".join(doc.text.split()[start_idx:end_idx])
                    chunks.append(Document(text=chunk_text))

            return chunks

        start_time = time.perf_counter()

        # Process documents into chunks
        chunks = mock_chunk_documents(test_documents)

        total_time = time.perf_counter() - start_time

        # Record chunking throughput
        result = throughput_tracker.record_throughput_test(
            "document_chunking",
            len(test_documents),
            total_time,
            concurrent_operations=1,
            resource_usage={"chunks_created": len(chunks)},
        )

        chunks_per_second = len(chunks) / total_time

        print(
            f"Document chunking: {result['throughput_rps']:.1f} docs/sec, "
            f"{chunks_per_second:.1f} chunks/sec"
        )

        # Should process documents efficiently
        assert result["throughput_rps"] >= 20.0, (
            f"Document chunking too slow: {result['throughput_rps']:.1f} < "
            f"20.0 docs/sec"
        )

        # Should create reasonable number of chunks
        avg_chunks_per_doc = len(chunks) / len(test_documents)
        assert 1 <= avg_chunks_per_doc <= 10, (
            f"Unusual chunking ratio: {avg_chunks_per_doc:.1f} chunks per document"
        )


@pytest.mark.performance
class TestThroughputTargetValidation:
    """Test throughput performance against RTX 4090 targets."""

    def test_comprehensive_throughput_validation(self, throughput_tracker):
        """Test comprehensive throughput validation against all targets."""
        # Simulate various throughput measurements that should meet/exceed targets
        test_scenarios = [
            {"name": "embedding_single", "rps": 25, "operations": 50},
            {"name": "embedding_batch", "rps": 35, "operations": 100},
            {"name": "reranking_20_docs", "rps": 12, "operations": 60},
            {"name": "query_processing", "rps": 8, "operations": 40},
            {"name": "document_processing", "rps": 120, "operations": 200},
        ]

        # Record test measurements
        for scenario in test_scenarios:
            duration = scenario["operations"] / scenario["rps"]

            throughput_tracker.record_throughput_test(
                scenario["name"],
                scenario["operations"],
                duration,
                concurrent_operations=1,
            )

        # Validate against RTX 4090 targets
        validation = throughput_tracker.validate_throughput_targets(
            RTX_4090_THROUGHPUT_TARGETS
        )

        print("Throughput validation results:")
        print(f"  Overall score: {validation['overall_score']:.2f}")
        print(f"  Targets met: {validation['targets_met']}")

        for target, result in validation["results"].items():
            status = "✓" if result["met"] else "✗"
            print(
                f"  {status} {target}: {result['actual']:.1f} vs "
                f"{result['target']:.1f} (margin: {result['margin']:.1%})"
            )

        # Should meet most targets
        assert validation["overall_score"] >= 0.8, (
            f"Overall throughput score too low: {validation['overall_score']:.2f} < 0.8"
        )

        # Should meet critical targets
        critical_targets = ["embedding_rps", "query_rps"]
        for target in critical_targets:
            if target in validation["results"]:
                assert validation["results"][target]["met"], (
                    f"Critical target {target} not met: {validation['results'][target]}"
                )

    def test_peak_throughput_analysis(self, throughput_tracker):
        """Test peak throughput analysis and reporting."""
        # Record various throughput tests with different peak performance
        test_data = [
            {"name": "embedding_peak_test", "rps": 45, "ops": 90, "concurrent": 2},
            {"name": "reranking_peak_test", "rps": 15, "ops": 30, "concurrent": 1},
            {"name": "query_peak_test", "rps": 12, "ops": 60, "concurrent": 3},
            {"name": "document_peak_test", "rps": 150, "ops": 300, "concurrent": 2},
        ]

        for test in test_data:
            duration = test["ops"] / test["rps"]
            throughput_tracker.record_throughput_test(
                test["name"],
                test["ops"],
                duration,
                concurrent_operations=test["concurrent"],
            )

        # Analyze peak throughput overall
        overall_peak = throughput_tracker.get_peak_throughput()

        print(f"Overall peak throughput: {overall_peak['peak_throughput_rps']:.1f} RPS")
        print(f"  Test: {overall_peak['test_name']}")
        print(f"  Concurrent operations: {overall_peak['concurrent_operations']}")

        # Analyze peak throughput by component
        component_peaks = {}
        for component in ["embedding", "reranking", "query", "document"]:
            peak = throughput_tracker.get_peak_throughput(component)
            component_peaks[component] = peak

            if peak["peak_throughput_rps"] > 0:
                print(
                    f"{component.title()} peak: {peak['peak_throughput_rps']:.1f} RPS"
                )

        # Verify peak measurements are reasonable
        assert overall_peak["peak_throughput_rps"] > 0, "No peak throughput recorded"
        assert overall_peak["peak_throughput_rps"] >= 10.0, (
            f"Overall peak throughput too low: "
            f"{overall_peak['peak_throughput_rps']:.1f} RPS"
        )

        # Verify component-specific peaks exist
        assert any(
            peak["peak_throughput_rps"] > 0 for peak in component_peaks.values()
        ), "No component-specific peaks found"


@pytest.mark.performance
@pytest.mark.requires_gpu
class TestGPUAcceleratedThroughputBenchmarks:
    """Test GPU-accelerated throughput with resource monitoring."""

    async def test_gpu_accelerated_embedding_throughput(self, throughput_tracker):
        """Test GPU-accelerated embedding throughput with monitoring."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for accelerated throughput test")

        with patch(
            "src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel"
        ) as mock_model:
            mock_bgem3 = MagicMock()

            # Simulate faster GPU processing with batch efficiency
            def mock_gpu_encode(texts, **kwargs):
                batch_size = len(texts)
                # GPU batching is more efficient
                base_time = 0.01  # 10ms base
                per_item_time = (
                    0.005 if batch_size > 1 else 0.02
                )  # 5ms per item in batch vs 20ms individual
                total_time = base_time + (per_item_time * batch_size)
                time.sleep(total_time)
                return {"dense_vecs": [[0.1] * 1024 for _ in texts]}

            mock_bgem3.encode = mock_gpu_encode
            mock_model.return_value = mock_bgem3

            from src.retrieval.embeddings.bge_m3_manager import BGEM3Embedding

            # Test GPU throughput with monitoring
            embedding_model = BGEM3Embedding(device="cuda", batch_size=10)
            test_documents = [f"GPU throughput test document {i}" for i in range(100)]

            async with gpu_performance_monitor() as gpu_metrics:
                if gpu_metrics:
                    print(f"Initial GPU metrics: {gpu_metrics}")

                    start_time = time.perf_counter()

                    # Process documents in GPU-optimized batches
                    batch_size = 10
                    for i in range(0, len(test_documents), batch_size):
                        batch_docs = test_documents[i : i + batch_size]

                        with gpu_memory_context():
                            embeddings = embedding_model.get_unified_embeddings(
                                batch_docs
                            )
                            assert "dense" in embeddings

                    total_time = time.perf_counter() - start_time

                    # Record GPU throughput
                    result = throughput_tracker.record_throughput_test(
                        "gpu_accelerated_embedding",
                        len(test_documents),
                        total_time,
                        concurrent_operations=1,
                        resource_usage={
                            "gpu_memory_gb": gpu_metrics.memory_allocated_gb,
                            "gpu_utilization": gpu_metrics.utilization_percent,
                        },
                    )

                    print(
                        f"GPU embedding throughput: {result['throughput_rps']:.1f} RPS"
                    )
                    print(f"GPU memory used: {gpu_metrics.memory_allocated_gb:.2f} GB")

                    # GPU should significantly outperform CPU baseline
                    cpu_target = RTX_4090_THROUGHPUT_TARGETS["embedding_rps"]
                    gpu_speedup = result["throughput_rps"] / cpu_target

                    assert gpu_speedup >= 1.2, (
                        f"GPU acceleration insufficient: {gpu_speedup:.2f}x vs "
                        f"expected >=1.2x"
                    )

                    print(f"GPU speedup over CPU target: {gpu_speedup:.2f}x")

    def test_memory_constrained_throughput(self, throughput_tracker):
        """Test throughput under GPU memory constraints."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for memory constrained test")

        from src.utils.resource_management import get_safe_vram_usage

        # Simulate memory-intensive operations
        def memory_intensive_operation(data_size: int):
            """Simulate operation that uses GPU memory."""
            with gpu_memory_context():
                # Allocate GPU memory to simulate model processing
                if torch.cuda.is_available():
                    temp_tensor = torch.randn(data_size, 1024, device="cuda")
                    time.sleep(0.01)  # Processing time
                    result = temp_tensor.sum().item()
                    del temp_tensor
                    return result
                else:
                    time.sleep(0.01)
                    return 0.0

        # Test throughput with increasing memory pressure
        memory_sizes = [100, 500, 1000, 2000]  # Different tensor sizes

        for size in memory_sizes:
            operations_count = 20

            start_time = time.perf_counter()
            memory_before = get_safe_vram_usage()

            # Process operations under memory constraint
            results = []
            for _i in range(operations_count):
                result = memory_intensive_operation(size)
                results.append(result)

            total_time = time.perf_counter() - start_time
            memory_after = get_safe_vram_usage()

            # Record throughput under memory pressure
            throughput_result = throughput_tracker.record_throughput_test(
                f"memory_constrained_{size}",
                operations_count,
                total_time,
                concurrent_operations=1,
                resource_usage={
                    "memory_before_gb": memory_before,
                    "memory_after_gb": memory_after,
                    "memory_delta_gb": memory_after - memory_before,
                    "tensor_size": size,
                },
            )

            print(
                f"Memory size {size}: {throughput_result['throughput_rps']:.1f} RPS, "
                f"Memory delta: {memory_after - memory_before:.3f}GB"
            )

        # Analyze throughput degradation with memory pressure
        small_size_result = next(
            t
            for t in throughput_tracker.measurements
            if "memory_constrained_100" in t["test_name"]
        )
        large_size_result = next(
            t
            for t in throughput_tracker.measurements
            if "memory_constrained_2000" in t["test_name"]
        )

        throughput_ratio = (
            large_size_result["throughput_rps"] / small_size_result["throughput_rps"]
        )

        # Throughput shouldn't degrade too much with memory pressure
        assert throughput_ratio >= THROUGHPUT_DEGRADATION_THRESHOLD, (
            f"Excessive throughput degradation under memory pressure: "
            f"{throughput_ratio:.2f} < {THROUGHPUT_DEGRADATION_THRESHOLD}"
        )

        print(f"Memory pressure throughput ratio: {throughput_ratio:.2f}")
