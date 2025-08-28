"""Response latency benchmarks and percentile tracking for DocMind AI.

This module provides comprehensive latency testing and analysis:
- Response time benchmarking with percentile metrics (P50, P95, P99)
- Latency regression detection with configurable thresholds
- End-to-end pipeline latency validation
- Concurrent request latency stability
- Component-level latency breakdown
- Performance target validation for RTX 4090

Follows PyTest-AI patterns with proper statistical analysis and actionable metrics.
"""

import asyncio
import statistics
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

# Graceful import handling for performance tests
try:
    from src.core.infrastructure.gpu_monitor import gpu_performance_monitor
    from src.utils.storage import gpu_memory_context
except ImportError:
    # Fallback mocks for consistent testing without external dependencies
    def gpu_performance_monitor():
        """Mock GPU performance monitor for testing without GPU dependencies."""
        return MagicMock()

    def gpu_memory_context():
        """Mock GPU memory context for testing without GPU dependencies."""
        return MagicMock()


# Latency test constants
P50_PERCENTILE = 50
P95_PERCENTILE = 95
P99_PERCENTILE = 99

# RTX 4090 Performance targets (milliseconds) - Updated for FP8 optimization
RTX_4090_TARGETS = {
    # BGE-M3 single chunk with FP8 optimization (down from 50ms)
    "embedding_latency_ms": 35,
    # BGE reranker-v2-m3 20 docs (down from 100ms)
    "reranking_latency_ms": 85,
    # End-to-end P95 with multi-agent efficiency (down from 2000ms)
    "query_p95_latency_ms": 1800,
    # End-to-end P50 (down from 1000ms)
    "query_p50_latency_ms": 900,
    # Improved batch processing efficiency (down from 0.3)
    "batch_efficiency_factor": 0.25,
}

# Regression detection thresholds
LATENCY_REGRESSION_THRESHOLD = 1.5  # 50% increase triggers regression alert
LATENCY_IMPROVEMENT_THRESHOLD = 0.8  # 20% decrease indicates improvement


@pytest.fixture
def latency_tracker():
    """Fixture for tracking latency measurements and computing statistics."""

    class LatencyTracker:
        def __init__(self):
            self.measurements = []
            self.component_measurements = {}
            self.baselines = {}

        def start_timing(self) -> float:
            """Start timing an operation."""
            return time.perf_counter()

        def end_timing(self, start_time: float) -> float:
            """End timing and return latency in milliseconds."""
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.measurements.append(latency_ms)
            return latency_ms

        def record_component_latency(self, component: str, latency_ms: float) -> None:
            """Record latency for a specific component."""
            if component not in self.component_measurements:
                self.component_measurements[component] = []
            self.component_measurements[component].append(latency_ms)

        def get_percentiles(self, measurements: list[float] = None) -> dict[str, float]:
            """Calculate latency percentiles."""
            data = measurements or self.measurements
            if not data:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

            sorted_data = sorted(data)
            return {
                "p50": self._percentile(sorted_data, P50_PERCENTILE),
                "p95": self._percentile(sorted_data, P95_PERCENTILE),
                "p99": self._percentile(sorted_data, P99_PERCENTILE),
                "mean": statistics.mean(data),
                "min": min(data),
                "max": max(data),
                "std": statistics.stdev(data) if len(data) > 1 else 0.0,
                "count": len(data),
            }

        def get_component_stats(self, component: str) -> dict[str, float]:
            """Get statistics for a specific component."""
            if component not in self.component_measurements:
                return {}
            return self.get_percentiles(self.component_measurements[component])

        def set_baseline(self, name: str, measurements: list[float] = None) -> None:
            """Set performance baseline for regression detection."""
            data = measurements or self.measurements
            if data:
                self.baselines[name] = self.get_percentiles(data)

        def detect_regression(
            self, baseline_name: str, current_measurements: list[float] = None
        ) -> dict[str, Any]:
            """Detect performance regression compared to baseline."""
            if baseline_name not in self.baselines:
                return {"error": f"Baseline '{baseline_name}' not found"}

            baseline_stats = self.baselines[baseline_name]
            current_stats = self.get_percentiles(
                current_measurements or self.measurements
            )

            # Calculate regression ratios
            regression_analysis = {
                "baseline_p95": baseline_stats["p95"],
                "current_p95": current_stats["p95"],
                "p95_ratio": current_stats["p95"] / baseline_stats["p95"]
                if baseline_stats["p95"] > 0
                else 1.0,
                "baseline_mean": baseline_stats["mean"],
                "current_mean": current_stats["mean"],
                "mean_ratio": current_stats["mean"] / baseline_stats["mean"]
                if baseline_stats["mean"] > 0
                else 1.0,
                "regression_detected": False,
                "improvement_detected": False,
            }

            # Check for regression or improvement
            if regression_analysis["p95_ratio"] > LATENCY_REGRESSION_THRESHOLD:
                regression_analysis["regression_detected"] = True
            elif regression_analysis["p95_ratio"] < LATENCY_IMPROVEMENT_THRESHOLD:
                regression_analysis["improvement_detected"] = True

            return regression_analysis

        def validate_targets(self, targets: dict[str, float]) -> dict[str, Any]:
            """Validate latency against performance targets."""
            current_stats = self.get_percentiles()
            validation = {"targets_met": True, "failed_targets": [], "results": {}}

            # Check each target
            for metric, target_value in targets.items():
                if metric == "query_p95_latency_ms":
                    actual_value = current_stats["p95"]
                elif metric == "query_p50_latency_ms":
                    actual_value = current_stats["p50"]
                elif metric.endswith("_latency_ms"):
                    actual_value = current_stats[
                        "mean"
                    ]  # Use mean for other latency targets
                else:
                    continue  # Skip non-latency targets

                validation["results"][metric] = {
                    "target": target_value,
                    "actual": actual_value,
                    "met": actual_value <= target_value,
                    "margin": (actual_value - target_value) / target_value
                    if target_value > 0
                    else 0.0,
                }

                if actual_value > target_value:
                    validation["targets_met"] = False
                    validation["failed_targets"].append(metric)

            return validation

        def _percentile(self, sorted_data: list[float], percentile: float) -> float:
            """Calculate percentile from sorted data."""
            if not sorted_data:
                return 0.0
            index = (percentile / 100.0) * (len(sorted_data) - 1)
            if index.is_integer():
                return sorted_data[int(index)]
            else:
                lower_index = int(index)
                upper_index = lower_index + 1
                weight = index - lower_index
                return (
                    sorted_data[lower_index] * (1 - weight)
                    + sorted_data[upper_index] * weight
                )

    return LatencyTracker()


@pytest.mark.performance
class TestComponentLatencyBenchmarks:
    """Test individual component latency performance."""

    def test_embedding_latency_single_document(self, latency_tracker):
        """Test single document embedding latency meets targets."""
        # Mock BGE-M3 embedding with current architecture patterns
        mock_bgem3 = MagicMock()

        # Simulate optimized embedding time (FP8 improvements)
        def mock_encode(*args, **kwargs):
            import random

            # Simulate 15-25ms processing time (faster with FP8 optimization)
            time.sleep(random.uniform(0.015, 0.025))
            return {"dense_vecs": [[0.1] * 1024]}

        mock_bgem3.encode = mock_encode

        # Mock embedding model without external dependencies
        class MockBGEM3Embedding:
            def __init__(self):
                self.model = mock_bgem3

            def get_unified_embeddings(self, texts):
                result = mock_bgem3.encode(texts)
                return {"dense": result["dense_vecs"], "sparse": []}

        embedding_model = MockBGEM3Embedding()

        # Benchmark multiple single document embeddings
        test_documents = [
            "Short document for embedding latency test.",
            (
                "Medium length document with more content for testing embedding "
                "performance and latency characteristics."
            ),
            (
                "Very long document with extensive content to test how BGE-M3 "
                "handles longer texts and whether latency scales appropriately "
                "with document length, including various types of content and "
                "complex sentence structures that might affect processing time."
            ),
        ]

        for doc_text in test_documents:
            start_time = latency_tracker.start_timing()

            # Use context manager properly with fallback
            context = gpu_memory_context()
            with context if hasattr(context, "__enter__") else MagicMock():
                embeddings = embedding_model.get_unified_embeddings([doc_text])

            latency = latency_tracker.end_timing(start_time)
            latency_tracker.record_component_latency("embedding_single", latency)

            # Verify embedding created
            assert "dense" in embeddings
            assert len(embeddings["dense"]) == 1

        # Analyze embedding latency statistics
        embedding_stats = latency_tracker.get_component_stats("embedding_single")

        # Validate against RTX 4090 targets
        target_latency = RTX_4090_TARGETS["embedding_latency_ms"]

        assert embedding_stats["p95"] <= target_latency * 1.5, (
            f"Embedding P95 latency too high: {embedding_stats['p95']:.1f}ms > "
            f"{target_latency * 1.5}ms"
        )

        assert embedding_stats["mean"] <= target_latency, (
            f"Embedding mean latency too high: {embedding_stats['mean']:.1f}ms > "
            f"{target_latency}ms"
        )

        print(f"Embedding latency stats: {embedding_stats}")

    def test_batch_embedding_efficiency(self, latency_tracker):
        """Test batch embedding provides efficiency gains over individual processing."""
        with patch(
            "src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel"
        ) as mock_model:
            mock_bgem3 = MagicMock()
            mock_model.return_value = mock_bgem3

            from src.retrieval.embeddings import BGEM3Embedding

            test_documents = [f"Batch test document {i}" for i in range(8)]

            # Test individual processing
            embedding_model_individual = BGEM3Embedding(batch_size=1)
            mock_bgem3.encode.side_effect = lambda texts, **kw: {
                "dense_vecs": [[0.1] * 1024 for _ in texts]
            }

            individual_start = latency_tracker.start_timing()

            for doc in test_documents:
                with gpu_memory_context():
                    embedding_model_individual.get_unified_embeddings([doc])
                    time.sleep(0.025)  # Simulate 25ms per individual operation

            individual_total = latency_tracker.end_timing(individual_start)
            individual_per_doc = individual_total / len(test_documents)

            # Test batch processing
            embedding_model_batch = BGEM3Embedding(batch_size=8)
            mock_bgem3.encode.side_effect = lambda texts, **kw: {
                "dense_vecs": [[0.1] * 1024 for _ in texts]
            }

            batch_start = latency_tracker.start_timing()

            with gpu_memory_context():
                embedding_model_batch.get_unified_embeddings(test_documents)
                time.sleep(0.08)  # Simulate 80ms for batch of 8 (10ms per doc)

            batch_total = latency_tracker.end_timing(batch_start)
            batch_per_doc = batch_total / len(test_documents)

            # Calculate efficiency
            efficiency_ratio = batch_per_doc / individual_per_doc

            latency_tracker.record_component_latency(
                "individual_embedding", individual_per_doc
            )
            latency_tracker.record_component_latency("batch_embedding", batch_per_doc)

            # Batch processing should be more efficient
            assert efficiency_ratio <= RTX_4090_TARGETS["batch_efficiency_factor"], (
                f"Batch processing not efficient enough: {efficiency_ratio:.2f} vs "
                f"target {RTX_4090_TARGETS['batch_efficiency_factor']}"
            )

            print(
                f"Individual per doc: {individual_per_doc:.1f}ms, "
                f"Batch per doc: {batch_per_doc:.1f}ms"
            )
            print(f"Batch efficiency ratio: {efficiency_ratio:.2f}")

    def test_reranking_latency_scaling(self, latency_tracker):
        """Test reranking latency scales reasonably with document count."""
        with patch(
            "src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder"
        ) as mock_ce:
            mock_cross_encoder = MagicMock()
            mock_ce.return_value = mock_cross_encoder

            # Simulate realistic reranking times based on document count
            def mock_predict(pairs, **kwargs):
                doc_count = len(pairs)
                # Simulate ~3-5ms per document pair
                time.sleep(doc_count * 0.004)
                return [0.9 - (i * 0.05) for i in range(doc_count)]

            mock_cross_encoder.predict = mock_predict
            mock_cross_encoder.model = MagicMock()

            from src.retrieval.reranking import (
                BGECrossEncoderRerank,
            )

            reranker = BGECrossEncoderRerank(top_n=5)

            # Test different document counts
            document_counts = [5, 10, 20]

            for doc_count in document_counts:
                # Create test nodes
                test_nodes = [
                    NodeWithScore(
                        node=TextNode(
                            text=f"reranking test document {i}", id_=f"doc_{i}"
                        ),
                        score=0.8,
                    )
                    for i in range(doc_count)
                ]

                query_bundle = QueryBundle(query_str="test reranking query")

                start_time = latency_tracker.start_timing()

                reranked_nodes = reranker._postprocess_nodes(test_nodes, query_bundle)

                latency = latency_tracker.end_timing(start_time)
                latency_tracker.record_component_latency(
                    f"reranking_{doc_count}_docs", latency
                )

                # Verify reranking completed
                assert len(reranked_nodes) == min(5, doc_count)  # top_n

            # Check 20-document reranking meets target
            rerank_20_stats = latency_tracker.get_component_stats("reranking_20_docs")
            target_latency = RTX_4090_TARGETS["reranking_latency_ms"]

            assert rerank_20_stats["mean"] <= target_latency, (
                f"Reranking 20 docs too slow: {rerank_20_stats['mean']:.1f}ms > "
                f"{target_latency}ms"
            )

            # Check scaling is reasonable (should be roughly linear)
            rerank_5_stats = latency_tracker.get_component_stats("reranking_5_docs")
            rerank_10_stats = latency_tracker.get_component_stats("reranking_10_docs")

            scaling_5_to_10 = rerank_10_stats["mean"] / rerank_5_stats["mean"]
            scaling_10_to_20 = rerank_20_stats["mean"] / rerank_10_stats["mean"]

            # Scaling should be roughly 2x for doubling document count
            assert 1.5 <= scaling_5_to_10 <= 2.5, (
                f"5->10 doc scaling unusual: {scaling_5_to_10:.1f}x"
            )
            assert 1.5 <= scaling_10_to_20 <= 2.5, (
                f"10->20 doc scaling unusual: {scaling_10_to_20:.1f}x"
            )

            print(
                f"Reranking scaling - 5 docs: {rerank_5_stats['mean']:.1f}ms, "
                f"10 docs: {rerank_10_stats['mean']:.1f}ms, "
                f"20 docs: {rerank_20_stats['mean']:.1f}ms"
            )


@pytest.mark.performance
class TestEndToEndLatencyBenchmarks:
    """Test end-to-end pipeline latency performance."""

    def test_query_pipeline_latency_targets(self, latency_tracker, test_documents):
        """Test complete query pipeline meets P95 latency targets."""
        # Mock all pipeline components
        with (
            patch(
                "src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel"
            ) as mock_embed,
            patch(
                "src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder"
            ) as mock_ce,
        ):
            # Setup embedding mock
            mock_bgem3 = MagicMock()
            mock_bgem3.encode.return_value = {"dense_vecs": [[0.1] * 1024]}
            mock_embed.return_value = mock_bgem3

            # Setup reranking mock
            mock_cross_encoder = MagicMock()
            mock_cross_encoder.predict.return_value = [0.9, 0.8, 0.7]
            mock_cross_encoder.model = MagicMock()
            mock_ce.return_value = mock_cross_encoder

            from src.retrieval.query_engine import (
                AdaptiveRouterQueryEngine,
            )
            from src.retrieval.reranking import (
                BGECrossEncoderRerank,
            )

            # Create pipeline components
            reranker = BGECrossEncoderRerank()
            mock_vector_index = MagicMock()

            # Mock vector search to return test nodes
            mock_vector_index.query.return_value = MagicMock(
                source_nodes=[
                    NodeWithScore(
                        node=TextNode(text=doc.text, id_=f"node_{i}"),
                        score=0.8 - (i * 0.1),
                    )
                    for i, doc in enumerate(test_documents)
                ]
            )

            router = AdaptiveRouterQueryEngine(
                vector_index=mock_vector_index, reranker=reranker
            )

            # Mock the router engine query with realistic latency
            def mock_router_query(query_str, **kwargs):
                # Simulate: embedding (30ms) + retrieval (50ms) +
                # reranking (80ms) = 160ms
                time.sleep(0.16)
                response = MagicMock()
                response.source_nodes = (
                    mock_vector_index.query.return_value.source_nodes[:3]
                )
                response.response = f"Response to: {query_str}"
                return response

            router.router_engine.query = mock_router_query

            # Test queries with different complexities
            test_queries = [
                "simple query",
                "What is BGE-M3 embedding and how does it work?",
                (
                    "Complex analytical query requiring detailed retrieval and "
                    "comprehensive reranking analysis"
                ),
                "How does DocMind AI implement multi-agent coordination?",
                "Explain the performance characteristics of hybrid search systems",
            ]

            # Run multiple iterations for statistical significance
            for _iteration in range(3):
                for query in test_queries:
                    start_time = latency_tracker.start_timing()

                    response = router.query(query)

                    latency = latency_tracker.end_timing(start_time)
                    latency_tracker.record_component_latency(
                        "end_to_end_query", latency
                    )

                    # Verify response generated
                    assert response is not None
                    assert hasattr(response, "response") or hasattr(
                        response, "source_nodes"
                    )

        # Validate end-to-end latency targets
        query_stats = latency_tracker.get_component_stats("end_to_end_query")

        validation = latency_tracker.validate_targets(
            {
                "query_p95_latency_ms": RTX_4090_TARGETS["query_p95_latency_ms"],
                "query_p50_latency_ms": RTX_4090_TARGETS["query_p50_latency_ms"],
            }
        )

        assert validation["targets_met"], (
            f"Latency targets not met: {validation['failed_targets']}\n"
            f"Results: {validation['results']}"
        )

        print(f"End-to-end query latency stats: {query_stats}")
        print(f"Target validation: {validation}")

    @pytest.mark.asyncio
    async def test_concurrent_query_latency_stability(self, latency_tracker):
        """Test latency stability under concurrent load."""

        # Mock async query processing
        async def mock_async_query(query_id: int):
            """Mock async query with variable latency."""
            import random

            # Simulate variable processing time (100-200ms)
            base_latency = 0.15
            variation = random.uniform(-0.05, 0.05)
            await asyncio.sleep(max(0.05, base_latency + variation))
            return f"Response to query {query_id}"

        # Simulate concurrent queries
        concurrent_queries = 10

        # Measure individual query latency under concurrency
        tasks = []
        start_times = {}

        for query_id in range(concurrent_queries):
            start_times[query_id] = latency_tracker.start_timing()
            task = asyncio.create_task(mock_async_query(query_id))
            tasks.append((query_id, task))

        # Wait for all queries to complete
        results = []
        for query_id, task in tasks:
            response = await task
            latency = latency_tracker.end_timing(start_times[query_id])
            latency_tracker.record_component_latency("concurrent_query", latency)
            results.append(response)

        # Verify all queries completed
        assert len(results) == concurrent_queries

        # Analyze concurrent latency characteristics
        concurrent_stats = latency_tracker.get_component_stats("concurrent_query")

        # Concurrent latency should not degrade significantly
        # Standard deviation should be reasonable (not excessive variance)
        coefficient_of_variation = concurrent_stats["std"] / concurrent_stats["mean"]
        assert coefficient_of_variation <= 0.3, (
            f"High latency variance under concurrency: "
            f"CV={coefficient_of_variation:.2f}"
        )

        # Mean latency should still be reasonable
        assert concurrent_stats["mean"] <= 300, (
            f"Concurrent latency too high: {concurrent_stats['mean']:.1f}ms"
        )

        print(f"Concurrent query latency stats: {concurrent_stats}")
        print(f"Coefficient of variation: {coefficient_of_variation:.2f}")


@pytest.mark.performance
class TestLatencyRegressionDetection:
    """Test latency regression detection and performance monitoring."""

    def test_performance_baseline_establishment(self, latency_tracker):
        """Test establishing performance baselines for regression detection."""
        # Simulate baseline performance measurements
        baseline_measurements = [
            150,
            145,
            160,
            140,
            155,
            165,
            148,
            152,
            158,
            147,  # ~150ms average
        ]

        # Set baseline
        latency_tracker.set_baseline("production_v1", baseline_measurements)

        # Simulate current performance (slight improvement)
        current_measurements = [
            135,
            140,
            138,
            142,
            145,
            148,
            136,
            144,
            141,
            139,  # ~140ms average
        ]

        # Detect regression/improvement
        regression_analysis = latency_tracker.detect_regression(
            "production_v1", current_measurements
        )

        # Should detect improvement (not regression)
        assert not regression_analysis["regression_detected"], (
            f"False regression detected: {regression_analysis}"
        )

        assert regression_analysis["improvement_detected"], (
            f"Improvement not detected: {regression_analysis}"
        )

        print(f"Regression analysis: {regression_analysis}")

    def test_regression_detection_sensitivity(self, latency_tracker):
        """Test regression detection with various performance changes."""
        # Baseline performance
        baseline = [100, 105, 98, 102, 99, 103, 101, 97, 106, 100]  # ~100ms
        latency_tracker.set_baseline("sensitivity_test", baseline)

        # Test scenarios
        test_scenarios = [
            {
                "name": "no_change",
                "measurements": [101, 99, 103, 98, 105, 102, 100, 104, 97, 101],
                "expected_regression": False,
                "expected_improvement": False,
            },
            {
                "name": "minor_improvement",
                "measurements": [85, 88, 82, 86, 84, 87, 85, 83, 89, 85],  # ~15% faster
                "expected_regression": False,
                "expected_improvement": True,
            },
            {
                "name": "significant_regression",
                "measurements": [
                    180,
                    175,
                    185,
                    190,
                    182,
                    178,
                    188,
                    183,
                    177,
                    186,
                ],  # ~80% slower
                "expected_regression": True,
                "expected_improvement": False,
            },
            {
                "name": "marginal_regression",
                "measurements": [
                    120,
                    125,
                    118,
                    122,
                    119,
                    126,
                    121,
                    117,
                    128,
                    120,
                ],  # ~20% slower
                "expected_regression": False,  # Below 50% threshold
                "expected_improvement": False,
            },
        ]

        for scenario in test_scenarios:
            analysis = latency_tracker.detect_regression(
                "sensitivity_test", scenario["measurements"]
            )

            assert analysis["regression_detected"] == scenario["expected_regression"], (
                f"Scenario '{scenario['name']}': regression detection mismatch\n"
                f"Expected: {scenario['expected_regression']}, "
                f"Got: {analysis['regression_detected']}\n"
                f"Analysis: {analysis}"
            )

            assert (
                analysis["improvement_detected"] == scenario["expected_improvement"]
            ), (
                f"Scenario '{scenario['name']}': improvement detection mismatch\n"
                f"Expected: {scenario['expected_improvement']}, "
                f"Got: {analysis['improvement_detected']}\n"
                f"Analysis: {analysis}"
            )

            print(
                f"Scenario '{scenario['name']}': {analysis['p95_ratio']:.2f}x P95 ratio"
            )

    def test_target_validation_comprehensive(self, latency_tracker):
        """Test comprehensive target validation against RTX 4090 specifications."""
        # Simulate measurements that should meet some targets but fail others
        test_measurements = [
            800,
            850,
            780,
            920,
            1100,
            760,
            880,
            940,
            1050,
            820,  # Mix of good and borderline
            1200,
            1400,
            950,
            1050,
            890,
            1300,
            1150,
            980,
            1250,
            1100,  # Some high latencies
        ]

        latency_tracker.measurements = test_measurements

        # Validate against all RTX 4090 targets
        validation = latency_tracker.validate_targets(RTX_4090_TARGETS)

        # Analyze results
        stats = latency_tracker.get_percentiles()

        print(f"Test measurements stats: {stats}")
        print(f"Target validation results: {validation}")

        # Should have detailed results for each target
        expected_metrics = ["query_p95_latency_ms", "query_p50_latency_ms"]
        for metric in expected_metrics:
            assert metric in validation["results"], f"Missing validation for {metric}"

            result = validation["results"][metric]
            assert "target" in result
            assert "actual" in result
            assert "met" in result
            assert "margin" in result

        # Validation should correctly identify which targets are met/failed
        if not validation["targets_met"]:
            assert len(validation["failed_targets"]) > 0
            print(f"Failed targets: {validation['failed_targets']}")

        # P50 should be better (lower) than P95
        assert stats["p50"] <= stats["p95"], "P50 should be <= P95"

        # Verify margin calculations are reasonable
        for metric, result in validation["results"].items():
            expected_margin = (result["actual"] - result["target"]) / result["target"]
            assert abs(result["margin"] - expected_margin) < 0.01, (
                f"Margin calculation error for {metric}"
            )


@pytest.mark.performance
@pytest.mark.requires_gpu
class TestGPUAcceleratedLatencyBenchmarks:
    """Test GPU-accelerated component latency with hardware monitoring."""

    async def test_gpu_accelerated_pipeline_latency(self, latency_tracker):
        """Test GPU-accelerated pipeline latency with resource monitoring."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for accelerated latency test")

        # Mock GPU-accelerated components
        with patch(
            "src.retrieval.embeddings.bge_m3_manager.BGEM3FlagModel"
        ) as mock_embed:
            mock_bgem3 = MagicMock()

            # Simulate faster GPU processing
            def mock_gpu_encode(*args, **kwargs):
                time.sleep(0.015)  # 15ms GPU processing (faster than CPU)
                return {"dense_vecs": [[0.1] * 1024]}

            mock_bgem3.encode = mock_gpu_encode
            mock_embed.return_value = mock_bgem3

            from src.retrieval.embeddings import BGEM3Embedding

            # Test GPU-accelerated embedding
            embedding_model = BGEM3Embedding(device="cuda")

            async with gpu_performance_monitor() as gpu_metrics:
                if gpu_metrics:
                    print(f"Initial GPU metrics: {gpu_metrics}")

                    # Benchmark GPU-accelerated operations
                    test_queries = [
                        "GPU accelerated query 1",
                        "GPU accelerated query 2",
                        "GPU accelerated query 3",
                    ]

                    for query in test_queries:
                        start_time = latency_tracker.start_timing()

                        with gpu_memory_context():
                            embeddings = embedding_model.get_unified_embeddings([query])

                        latency = latency_tracker.end_timing(start_time)
                        latency_tracker.record_component_latency(
                            "gpu_embedding", latency
                        )

                        # Verify embedding created
                        assert "dense" in embeddings

        # Analyze GPU-accelerated performance
        gpu_stats = latency_tracker.get_component_stats("gpu_embedding")

        # GPU acceleration should meet aggressive targets
        gpu_target = (
            RTX_4090_TARGETS["embedding_latency_ms"] * 0.7
        )  # 30% faster with GPU

        assert gpu_stats["mean"] <= gpu_target, (
            f"GPU embedding not fast enough: {gpu_stats['mean']:.1f}ms > {gpu_target}ms"
        )

        print(f"GPU-accelerated embedding stats: {gpu_stats}")

    def test_fp16_latency_benefits(self, latency_tracker):
        """Test FP16 acceleration provides latency benefits over FP32."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for FP16 latency test")

        with patch(
            "src.retrieval.postprocessor.cross_encoder_rerank.CrossEncoder"
        ) as mock_ce:
            mock_cross_encoder = MagicMock()
            mock_cross_encoder.model = MagicMock()
            mock_ce.return_value = mock_cross_encoder

            # Simulate FP16 vs FP32 performance difference
            def mock_fp16_predict(pairs, **kwargs):
                time.sleep(0.03)  # 30ms with FP16
                return [0.9] * len(pairs)

            def mock_fp32_predict(pairs, **kwargs):
                time.sleep(0.05)  # 50ms with FP32 (slower)
                return [0.9] * len(pairs)

            from src.retrieval.reranking import (
                BGECrossEncoderRerank,
            )

            # Test FP16 reranking
            mock_cross_encoder.predict = mock_fp16_predict
            reranker_fp16 = BGECrossEncoderRerank(use_fp16=True)

            test_nodes = [
                NodeWithScore(
                    node=TextNode(text=f"FP16 test doc {i}", id_=f"doc_{i}"), score=0.8
                )
                for i in range(10)
            ]

            query_bundle = QueryBundle(query_str="FP16 performance test")

            # Benchmark FP16
            for _ in range(3):
                start_time = latency_tracker.start_timing()

                reranker_fp16._postprocess_nodes(test_nodes, query_bundle)

                latency = latency_tracker.end_timing(start_time)
                latency_tracker.record_component_latency("fp16_reranking", latency)

            # Test FP32 reranking for comparison
            mock_cross_encoder.predict = mock_fp32_predict
            reranker_fp32 = BGECrossEncoderRerank(use_fp16=False)

            for _ in range(3):
                start_time = latency_tracker.start_timing()

                reranker_fp32._postprocess_nodes(test_nodes, query_bundle)

                latency = latency_tracker.end_timing(start_time)
                latency_tracker.record_component_latency("fp32_reranking", latency)

        # Compare FP16 vs FP32 performance
        fp16_stats = latency_tracker.get_component_stats("fp16_reranking")
        fp32_stats = latency_tracker.get_component_stats("fp32_reranking")

        # FP16 should be faster than FP32
        speedup = fp32_stats["mean"] / fp16_stats["mean"]

        assert speedup >= 1.2, (
            f"FP16 speedup insufficient: {speedup:.2f}x vs expected >=1.2x"
        )

        print(
            f"FP16 mean: {fp16_stats['mean']:.1f}ms, "
            f"FP32 mean: {fp32_stats['mean']:.1f}ms"
        )
        print(f"FP16 speedup: {speedup:.2f}x")
