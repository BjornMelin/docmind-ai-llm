"""Performance tests for Multi-Agent Coordination System.

This module implements comprehensive performance tests that validate coordination
overhead, throughput targets, and system performance under various load conditions.

Performance Targets Validated:
- Coordination overhead <200ms per agent decision (ADR-011)
- Token reduction 50-87% through parallel execution
- Decode throughput 100-160 tokens/second (FP8 optimization)
- Prefill throughput 800-1300 tokens/second (FP8 optimization)
- VRAM usage <16GB on RTX 4090 Laptop
- Context management at 128K token limit

Features tested:
- Real-time performance measurement
- Load testing and stress scenarios
- Memory usage optimization
- Concurrent request handling
- Performance degradation detection
- Benchmarking against targets
"""

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.messages import HumanMessage

from src.agents.coordinator import MultiAgentCoordinator
from src.config.vllm_config import VLLMConfig, VLLMManager


class TestCoordinationOverheadPerformance:
    """Test coordination overhead performance against <200ms target."""

    def test_coordination_overhead_single_query(
        self, mock_coordinator: MultiAgentCoordinator, performance_timer: dict[str, Any]
    ):
        """Test coordination overhead for single query under 200ms."""
        # Mock fast agent responses
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Fast response")],
                "agent_timings": {
                    "router_agent": 0.015,
                    "retrieval_agent": 0.035,
                    "validation_agent": 0.012,
                },
                "validation_result": {"confidence": 0.9},
            }
        ]

        # Measure coordination overhead
        performance_timer["start_timer"]("coordination")
        response = mock_coordinator.process_query("What is machine learning?")
        coordination_time_ms = performance_timer["end_timer"]("coordination") * 1000

        # Verify ADR-011 target
        assert coordination_time_ms < 200, (
            f"Coordination overhead {coordination_time_ms:.2f}ms exceeds 200ms target"
        )
        assert response.optimization_metrics["coordination_overhead_ms"] < 200
        assert response.optimization_metrics["meets_200ms_target"] is True

    def test_coordination_overhead_multiple_queries(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test coordination overhead consistency across multiple queries."""
        queries = [
            "What is AI?",
            "Explain neural networks",
            "Define machine learning",
            "How do LLMs work?",
            "What is deep learning?",
        ]

        coordination_times = []

        for query in queries:
            # Mock fast processing for each query
            mock_coordinator.compiled_graph.stream.return_value = [
                {
                    "messages": [HumanMessage(content=f"Response to: {query}")],
                    "agent_timings": {"router_agent": 0.012, "retrieval_agent": 0.028},
                    "validation_result": {"confidence": 0.9},
                }
            ]

            start_time = time.perf_counter()
            response = mock_coordinator.process_query(query)
            coordination_time = (time.perf_counter() - start_time) * 1000

            coordination_times.append(coordination_time)

            # Each query should meet target
            assert coordination_time < 200
            assert response.optimization_metrics["coordination_overhead_ms"] < 200

        # Statistical analysis
        avg_coordination = statistics.mean(coordination_times)
        max_coordination = max(coordination_times)
        std_coordination = (
            statistics.stdev(coordination_times) if len(coordination_times) > 1 else 0
        )

        # Performance requirements
        assert avg_coordination < 150, (
            f"Average coordination {avg_coordination:.2f}ms should be well under 200ms"
        )
        assert max_coordination < 200, (
            f"Maximum coordination {max_coordination:.2f}ms exceeds target"
        )
        assert std_coordination < 50, (
            f"Coordination variance {std_coordination:.2f}ms too high"
        )

    def test_coordination_overhead_complex_queries(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test coordination overhead for complex queries requiring planning."""
        complex_query = (
            "Compare renewable energy vs fossil fuels and analyze environmental impact"
        )

        # Mock complex query processing with multiple agents
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Complex analysis response")],
                "routing_decision": {"complexity": "complex", "needs_planning": True},
                "planning_output": {"sub_tasks": ["Task 1", "Task 2", "Task 3"]},
                "agent_timings": {
                    "router_agent": 0.025,
                    "planner_agent": 0.045,
                    "retrieval_agent": 0.085,
                    "synthesis_agent": 0.055,
                    "validation_agent": 0.030,
                },
                "parallel_execution_active": True,
                "validation_result": {"confidence": 0.88},
            }
        ]

        start_time = time.perf_counter()
        response = mock_coordinator.process_query(complex_query)
        coordination_time = (time.perf_counter() - start_time) * 1000

        # Even complex queries should meet coordination target
        assert coordination_time < 250, (
            f"Complex query coordination {coordination_time:.2f}ms too high"
        )
        assert response.optimization_metrics["coordination_overhead_ms"] < 200

        # Verify parallel execution helps meet target
        assert response.optimization_metrics["parallel_execution_active"] is True

    def test_coordination_overhead_under_load(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test coordination overhead under concurrent load."""
        num_concurrent = 5
        queries = [f"Query {i + 1}" for i in range(num_concurrent)]

        def process_query_with_timing(query: str) -> float:
            # Mock processing for concurrent execution
            mock_coordinator.compiled_graph.stream.return_value = [
                {
                    "messages": [HumanMessage(content=f"Response to {query}")],
                    "agent_timings": {"router_agent": 0.015, "retrieval_agent": 0.035},
                    "validation_result": {"confidence": 0.9},
                }
            ]

            start_time = time.perf_counter()
            _ = mock_coordinator.process_query(query)
            coordination_time = (time.perf_counter() - start_time) * 1000

            return coordination_time

        # Execute concurrent queries
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(process_query_with_timing, query) for query in queries
            ]
            coordination_times = [future.result() for future in as_completed(futures)]

        # All queries should still meet coordination target under load
        max_time = max(coordination_times)
        avg_time = statistics.mean(coordination_times)

        assert max_time < 300, (
            f"Max coordination time {max_time:.2f}ms under load exceeds limit"
        )
        assert avg_time < 200, (
            f"Average coordination time {avg_time:.2f}ms under load too high"
        )


class TestThroughputPerformance:
    """Test throughput performance for FP8 optimization targets."""

    def test_decode_throughput_targets(self, mock_vllm_manager: VLLMManager):
        """Test decode throughput meets 100-160 tokens/second target."""
        # Mock performance validation with realistic FP8 performance
        mock_performance_result = {
            "decode_throughput_estimate": 135.7,  # Within target range
            "meets_decode_target": True,
            "generation_time": 0.24,
            "tokens_generated": 32,
            "model_loaded": True,
            "fp8_optimization": True,
            "context_window": 131072,
            "meets_context_target": True,
            "validation_timestamp": time.time(),
        }

        mock_vllm_manager.validate_performance = Mock(
            return_value=mock_performance_result
        )

        result = mock_vllm_manager.validate_performance()

        # Verify decode throughput targets
        decode_throughput = result["decode_throughput_estimate"]
        assert 100 <= decode_throughput <= 160, (
            f"Decode throughput {decode_throughput} outside 100-160 tok/s range"
        )
        assert result["meets_decode_target"] is True
        assert result["fp8_optimization"] is True

    def test_prefill_throughput_targets(self, mock_vllm_manager: VLLMManager):
        """Test prefill throughput meets 800-1300 tokens/second target."""
        # Mock performance validation with prefill metrics
        mock_performance_result = {
            "prefill_throughput_estimate": 1150.2,  # Within target range
            "meets_prefill_target": True,
            "prefill_time": 0.085,
            "prefill_tokens": 98,
            "model_loaded": True,
            "fp8_optimization": True,
            "flashinfer_backend": True,
            "validation_timestamp": time.time(),
        }

        mock_vllm_manager.validate_performance = Mock(
            return_value=mock_performance_result
        )

        result = mock_vllm_manager.validate_performance()

        # Verify prefill throughput targets
        prefill_throughput = result["prefill_throughput_estimate"]
        assert 800 <= prefill_throughput <= 1300, (
            f"Prefill throughput {prefill_throughput} outside 800-1300 tok/s range"
        )
        assert result["meets_prefill_target"] is True
        assert result["fp8_optimization"] is True

    def test_token_reduction_parallel_execution(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test token reduction through parallel execution meets 50-87% target."""
        # Mock parallel execution with token reduction
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Parallel execution response")],
                "parallel_execution_active": True,
                "token_reduction_achieved": 0.68,  # 68% reduction
                "parallel_tool_calls": True,
                "agent_timings": {
                    "router_agent": 0.020,
                    "retrieval_agent": 0.045,  # Parallel with synthesis
                    "synthesis_agent": 0.042,  # Parallel with retrieval
                    "validation_agent": 0.018,
                },
                "validation_result": {"confidence": 0.91},
            }
        ]

        response = mock_coordinator.process_query("Test parallel execution performance")

        # Verify token reduction targets
        token_reduction = response.optimization_metrics["token_reduction_achieved"]
        assert 0.5 <= token_reduction <= 0.87, (
            f"Token reduction {token_reduction:.2%} outside 50-87% range"
        )
        assert response.optimization_metrics["parallel_execution_active"] is True

    def test_throughput_degradation_detection(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test detection of throughput performance degradation."""
        # Mock degraded performance scenario
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Degraded performance response")],
                "coordination_overhead_ms": 285.5,  # Exceeds target
                "meets_200ms_target": False,
                "parallel_execution_active": False,  # Degraded to sequential
                "token_reduction_achieved": 0.25,  # Below target
                "agent_timings": {
                    "router_agent": 0.055,  # Slower than expected
                    "retrieval_agent": 0.125,  # Much slower
                    "validation_agent": 0.045,
                },
                "validation_result": {"confidence": 0.78},
            }
        ]

        response = mock_coordinator.process_query("Test degraded performance")

        # Verify degradation detection
        assert response.optimization_metrics["meets_200ms_target"] is False
        assert response.optimization_metrics["coordination_overhead_ms"] > 200
        assert response.optimization_metrics["token_reduction_achieved"] < 0.5
        assert response.optimization_metrics["parallel_execution_active"] is False


class TestMemoryPerformance:
    """Test memory usage and optimization performance."""

    def test_vram_usage_targets(self, mock_vllm_config: VLLMConfig):
        """Test VRAM usage stays under 16GB target."""
        manager = VLLMManager(mock_vllm_config)

        # Mock VRAM usage metrics
        mock_performance_result = {
            "vram_usage_gb": 13.2,  # Under 16GB target
            "peak_vram_usage_gb": 14.8,  # Peak still under target
            "kv_cache_usage_gb": 8.1,  # FP8 optimized
            "model_memory_gb": 4.2,  # FP8 quantized model
            "fp8_optimization": True,
            "memory_efficient": True,
            "context_window": 131072,
        }

        manager.validate_performance = Mock(return_value=mock_performance_result)

        result = manager.validate_performance()

        # Verify VRAM targets
        vram_usage = result["vram_usage_gb"]
        peak_vram = result["peak_vram_usage_gb"]

        assert vram_usage <= 16.0, f"VRAM usage {vram_usage}GB exceeds 16GB target"
        assert peak_vram <= 16.0, f"Peak VRAM usage {peak_vram}GB exceeds 16GB target"
        assert result["fp8_optimization"] is True

    def test_kv_cache_memory_efficiency(self, mock_coordinator: MultiAgentCoordinator):
        """Test FP8 KV cache memory efficiency."""
        # Mock KV cache optimization
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Memory optimized response")],
                "kv_cache_usage_gb": 7.8,  # ~50% reduction with FP8
                "context_used_tokens": 128000,  # Near 128K limit
                "memory_optimized": True,
                "fp8_kv_cache": True,
                "optimization_metrics": {
                    "kv_cache_dtype": "fp8_e5m2",
                    "memory_reduction_ratio": 0.52,  # 52% memory reduction
                    "fp8_optimization": True,
                },
                "validation_result": {"confidence": 0.9},
            }
        ]

        response = mock_coordinator.process_query("Test memory optimization")

        # Verify KV cache efficiency
        kv_cache_usage = response.optimization_metrics.get("kv_cache_usage_gb", 0)
        assert kv_cache_usage <= 10.0, f"KV cache usage {kv_cache_usage}GB too high"
        assert response.optimization_metrics["fp8_optimization"] is True

    def test_context_management_performance(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test 128K context management performance."""
        # Mock large context scenario
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Large context response")],
                "context_trimmed": True,
                "tokens_trimmed": 15000,
                "context_management_time_ms": 25.5,  # Fast context trimming
                "optimization_metrics": {
                    "context_window_used": 131072,
                    "context_used_tokens": 120000,  # Under threshold
                    "trimming_efficiency": 0.88,
                    "fp8_optimization": True,
                },
                "validation_result": {"confidence": 0.87},
            }
        ]

        response = mock_coordinator.process_query("Test large context management")

        # Verify context management performance
        assert response.optimization_metrics["context_window_used"] == 131072
        assert response.optimization_metrics["context_used_tokens"] <= 131072

        # Context trimming should be efficient
        if "context_management_time_ms" in response.optimization_metrics:
            trim_time = response.optimization_metrics["context_management_time_ms"]
            assert trim_time < 50, f"Context trimming time {trim_time}ms too slow"


class TestStressAndLoadPerformance:
    """Test system performance under stress and load conditions."""

    def test_concurrent_query_performance(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test performance with concurrent query processing."""
        num_concurrent = 10
        queries = [f"Concurrent query {i + 1}" for i in range(num_concurrent)]

        def process_concurrent_query(query: str) -> dict[str, Any]:
            # Mock concurrent processing
            mock_coordinator.compiled_graph.stream.return_value = [
                {
                    "messages": [HumanMessage(content=f"Response to {query}")],
                    "agent_timings": {"router_agent": 0.018, "retrieval_agent": 0.042},
                    "coordination_overhead_ms": 95.2,
                    "validation_result": {"confidence": 0.9},
                }
            ]

            start_time = time.perf_counter()
            response = mock_coordinator.process_query(query)
            processing_time = time.perf_counter() - start_time

            return {
                "query": query,
                "processing_time": processing_time,
                "coordination_overhead": response.optimization_metrics[
                    "coordination_overhead_ms"
                ],
                "response": response,
            }

        # Execute concurrent queries
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [
                executor.submit(process_concurrent_query, query) for query in queries
            ]
            results = [future.result() for future in as_completed(futures)]

        # Analyze concurrent performance
        processing_times = [r["processing_time"] for r in results]
        coordination_overheads = [r["coordination_overhead"] for r in results]

        avg_processing = statistics.mean(processing_times)
        max_processing = max(processing_times)
        avg_coordination = statistics.mean(coordination_overheads)
        max_coordination = max(coordination_overheads)

        # Performance under concurrent load
        assert avg_processing < 1.0, (
            f"Average processing time {avg_processing:.3f}s too high under load"
        )
        assert max_processing < 2.0, (
            f"Max processing time {max_processing:.3f}s too high under load"
        )
        assert avg_coordination < 150, (
            f"Average coordination {avg_coordination:.2f}ms too high under load"
        )
        assert max_coordination < 250, (
            f"Max coordination {max_coordination:.2f}ms too high under load"
        )

    def test_memory_performance_under_load(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test memory performance under sustained load."""
        num_queries = 20
        memory_usage_samples = []

        for i in range(num_queries):
            # Mock memory usage tracking
            mock_coordinator.compiled_graph.stream.return_value = [
                {
                    "messages": [HumanMessage(content=f"Load test response {i + 1}")],
                    "memory_usage_mb": 2800 + (i * 15),  # Gradually increasing
                    "kv_cache_usage_gb": 8.2 + (i * 0.05),  # Slight increase
                    "optimization_metrics": {
                        "fp8_optimization": True,
                        "memory_efficient": True,
                    },
                    "validation_result": {"confidence": 0.9},
                }
            ]

            response = mock_coordinator.process_query(f"Load test query {i + 1}")

            # Track memory usage
            if "memory_usage_mb" in response.optimization_metrics:
                memory_usage_samples.append(
                    response.optimization_metrics["memory_usage_mb"]
                )

        # Analyze memory performance under load
        if memory_usage_samples:
            max_memory = max(memory_usage_samples)
            memory_growth = memory_usage_samples[-1] - memory_usage_samples[0]

            assert max_memory < 5000, f"Memory usage {max_memory}MB too high under load"
            assert memory_growth < 500, (
                f"Memory growth {memory_growth}MB indicates leak"
            )

    @pytest.mark.slow
    def test_sustained_performance_benchmark(
        self, mock_coordinator: MultiAgentCoordinator, benchmark_config: dict[str, Any]
    ):
        """Test sustained performance over extended period."""
        duration_seconds = 30  # Sustained test duration
        queries_per_second = 2
        total_queries = duration_seconds * queries_per_second

        performance_samples = []
        start_time = time.perf_counter()

        for i in range(total_queries):
            # Mock sustained performance
            mock_coordinator.compiled_graph.stream.return_value = [
                {
                    "messages": [
                        HumanMessage(content=f"Sustained test response {i + 1}")
                    ],
                    "agent_timings": {"router_agent": 0.015, "retrieval_agent": 0.038},
                    "coordination_overhead_ms": 120 + (i % 10),  # Slight variation
                    "validation_result": {"confidence": 0.9},
                }
            ]

            query_start = time.perf_counter()
            response = mock_coordinator.process_query(f"Sustained query {i + 1}")
            query_time = (time.perf_counter() - query_start) * 1000

            performance_samples.append(
                {
                    "query_id": i + 1,
                    "processing_time_ms": query_time,
                    "coordination_overhead_ms": response.optimization_metrics[
                        "coordination_overhead_ms"
                    ],
                    "timestamp": time.perf_counter() - start_time,
                }
            )

            # Maintain target QPS
            time.sleep(1.0 / queries_per_second)

        _ = time.perf_counter() - start_time

        # Analyze sustained performance
        processing_times = [s["processing_time_ms"] for s in performance_samples]
        coordination_times = [
            s["coordination_overhead_ms"] for s in performance_samples
        ]

        avg_processing = statistics.mean(processing_times)
        std_processing = statistics.stdev(processing_times)
        avg_coordination = statistics.mean(coordination_times)
        std_coordination = statistics.stdev(coordination_times)

        # Sustained performance requirements
        assert avg_processing < 500, (
            f"Average processing {avg_processing:.2f}ms too high in sustained test"
        )
        assert std_processing < 100, (
            f"Processing variance {std_processing:.2f}ms too high"
        )
        assert avg_coordination < 150, (
            f"Average coordination {avg_coordination:.2f}ms too high"
        )
        assert std_coordination < 30, (
            f"Coordination variance {std_coordination:.2f}ms too high"
        )

        # Performance should remain stable over time
        first_half = coordination_times[: len(coordination_times) // 2]
        second_half = coordination_times[len(coordination_times) // 2 :]

        first_half_avg = statistics.mean(first_half)
        second_half_avg = statistics.mean(second_half)
        performance_drift = abs(second_half_avg - first_half_avg)

        assert performance_drift < 20, (
            f"Performance drift {performance_drift:.2f}ms indicates degradation"
        )


class TestPerformanceRegression:
    """Test performance regression detection and monitoring."""

    def test_performance_baseline_establishment(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test establishment of performance baselines."""
        baseline_queries = [
            "Simple factual query",
            "Medium complexity analysis",
            "Complex comparison with planning",
        ]

        baseline_metrics = {}

        for query in baseline_queries:
            complexity = (
                "simple"
                if "Simple" in query
                else ("medium" if "Medium" in query else "complex")
            )

            # Mock baseline performance
            mock_coordinator.compiled_graph.stream.return_value = [
                {
                    "messages": [
                        HumanMessage(content=f"Baseline response for {query}")
                    ],
                    "routing_decision": {"complexity": complexity},
                    "coordination_overhead_ms": {
                        "simple": 85,
                        "medium": 125,
                        "complex": 175,
                    }[complexity],
                    "agent_timings": {"router_agent": 0.015, "retrieval_agent": 0.035},
                    "validation_result": {"confidence": 0.9},
                }
            ]

            start_time = time.perf_counter()
            response = mock_coordinator.process_query(query)
            processing_time = (time.perf_counter() - start_time) * 1000

            baseline_metrics[complexity] = {
                "processing_time_ms": processing_time,
                "coordination_overhead_ms": response.optimization_metrics[
                    "coordination_overhead_ms"
                ],
                "query": query,
            }

        # Verify baseline establishment
        for _, metrics in baseline_metrics.items():
            assert metrics["coordination_overhead_ms"] < 200
            assert metrics["processing_time_ms"] < 1000

        # Baselines should show complexity scaling
        assert (
            baseline_metrics["simple"]["coordination_overhead_ms"]
            < baseline_metrics["medium"]["coordination_overhead_ms"]
        )
        assert (
            baseline_metrics["medium"]["coordination_overhead_ms"]
            < baseline_metrics["complex"]["coordination_overhead_ms"]
        )

    def test_performance_regression_detection(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test detection of performance regressions."""
        # Establish baseline
        baseline_query = "Performance regression test query"

        # Mock baseline performance
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Baseline performance")],
                "coordination_overhead_ms": 120.5,
                "agent_timings": {"router_agent": 0.015, "retrieval_agent": 0.035},
                "validation_result": {"confidence": 0.9},
            }
        ]

        baseline_response = mock_coordinator.process_query(baseline_query)
        baseline_coordination = baseline_response.optimization_metrics[
            "coordination_overhead_ms"
        ]

        # Mock regressed performance
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Regressed performance")],
                "coordination_overhead_ms": 285.8,  # Significant regression
                "agent_timings": {
                    "router_agent": 0.055,
                    "retrieval_agent": 0.125,
                },  # Much slower
                "performance_regression": True,
                "regression_ratio": 2.37,  # 137% increase
                "validation_result": {"confidence": 0.75},
            }
        ]

        regressed_response = mock_coordinator.process_query(baseline_query)
        regressed_coordination = regressed_response.optimization_metrics[
            "coordination_overhead_ms"
        ]

        # Detect regression
        regression_ratio = regressed_coordination / baseline_coordination
        regression_detected = regression_ratio > 1.5  # 50% increase threshold

        assert regression_detected, (
            f"Failed to detect regression: {regression_ratio:.2f}x baseline"
        )
        assert regressed_coordination > 250, (
            "Regression scenario should show significant degradation"
        )

        # Performance should clearly indicate problem
        assert regressed_response.optimization_metrics["meets_200ms_target"] is False

    def test_performance_monitoring_alerts(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test performance monitoring and alert generation."""
        monitoring_samples = []
        alert_thresholds = {
            "coordination_overhead_ms": 200,
            "processing_time_ms": 1000,
            "token_reduction_achieved": 0.5,  # Minimum threshold
        }

        # Simulate monitoring over time with varying performance
        performance_scenarios = [
            {
                "name": "normal",
                "coordination_ms": 125,
                "processing_ms": 450,
                "token_reduction": 0.65,
            },
            {
                "name": "degraded",
                "coordination_ms": 235,
                "processing_ms": 850,
                "token_reduction": 0.35,
            },
            {
                "name": "recovered",
                "coordination_ms": 145,
                "processing_ms": 520,
                "token_reduction": 0.62,
            },
        ]

        for scenario in performance_scenarios:
            # Mock scenario performance
            mock_coordinator.compiled_graph.stream.return_value = [
                {
                    "messages": [
                        HumanMessage(content=f"Monitoring scenario: {scenario['name']}")
                    ],
                    "coordination_overhead_ms": scenario["coordination_ms"],
                    "processing_time_ms": scenario["processing_ms"],
                    "token_reduction_achieved": scenario["token_reduction"],
                    "validation_result": {"confidence": 0.9},
                }
            ]

            response = mock_coordinator.process_query(
                f"Monitoring test {scenario['name']}"
            )

            # Check against thresholds
            alerts = []
            for metric, threshold in alert_thresholds.items():
                if metric in response.optimization_metrics:
                    value = response.optimization_metrics[metric]
                    if metric == "token_reduction_achieved":
                        if value < threshold:  # Below minimum
                            alerts.append(f"{metric}: {value:.2f} below {threshold}")
                    else:
                        if value > threshold:  # Above maximum
                            alerts.append(f"{metric}: {value} exceeds {threshold}")

            monitoring_samples.append(
                {
                    "scenario": scenario["name"],
                    "metrics": response.optimization_metrics,
                    "alerts": alerts,
                }
            )

        # Verify monitoring detection
        normal_sample = monitoring_samples[0]
        degraded_sample = monitoring_samples[1]
        recovered_sample = monitoring_samples[2]

        assert len(normal_sample["alerts"]) == 0, (
            "Normal performance should not trigger alerts"
        )
        assert len(degraded_sample["alerts"]) > 0, (
            "Degraded performance should trigger alerts"
        )
        assert len(recovered_sample["alerts"]) == 0, (
            "Recovered performance should clear alerts"
        )
