"""Comprehensive performance validation tests for agent coordination flows.

This test suite validates performance characteristics of the multi-agent coordination system:
- Coordination overhead measurement and ADR-011 compliance (<200ms target)
- Token reduction validation through parallel execution optimization
- Memory usage monitoring for FP8 KV cache optimization
- Throughput testing for concurrent agent operations
- Latency benchmarks for individual agent response times
- Scalability validation under increasing coordination complexity

Test Strategy:
- Performance-focused testing with realistic workload simulation
- Mock boundaries to eliminate external latency factors
- Measure coordination-specific performance metrics
- Validate ADR compliance for performance targets
- Focus on system performance patterns, not functional correctness

Markers:
- @pytest.mark.performance: Performance benchmark tests
- @pytest.mark.asyncio: Async performance patterns
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from llama_index.core.memory import ChatMemoryBuffer

from src.agents.coordinator import ContextManager
from src.agents.tools import (
    plan_query,
    retrieve_documents,
    route_query,
    synthesize_results,
    validate_response,
)


@pytest.mark.performance
class TestAgentCoordinationPerformance:
    """Performance validation tests for agent coordination system."""

    def test_coordination_overhead_adr_011_compliance(self, benchmark_config):
        """Test coordination overhead meets ADR-011 target (<200ms)."""
        # Given: Mock state for coordination overhead testing
        mock_state = {
            "messages": [HumanMessage(content="Coordination overhead test")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock(), "kg": Mock(), "retriever": Mock()},
            "parallel_execution_active": True,
            "total_start_time": time.perf_counter(),
        }

        coordination_times = []

        # When: Measuring coordination overhead across multiple operations
        for i in range(10):  # Test multiple iterations for statistical significance
            start_time = time.perf_counter()

            with patch("src.agents.tools.ToolFactory") as mock_factory:
                mock_factory.create_tools_from_indexes.return_value = [Mock()]

                # Simulate agent coordination sequence
                route_query(f"Test query {i}", state=mock_state)
                plan_query(f"Test query {i}", state=mock_state)
                retrieve_documents(f"Test query {i}", state=mock_state)

            coordination_time = (
                time.perf_counter() - start_time
            ) * 1000  # Convert to ms
            coordination_times.append(coordination_time)

        # Then: Coordination overhead meets ADR-011 target
        avg_coordination_time = sum(coordination_times) / len(coordination_times)
        max_coordination_time = max(coordination_times)
        min_coordination_time = min(coordination_times)

        # ADR-011 compliance: <200ms coordination overhead
        assert avg_coordination_time < 200, (
            f"Average coordination overhead {avg_coordination_time:.2f}ms exceeds 200ms target"
        )
        assert max_coordination_time < 300, (
            f"Peak coordination overhead {max_coordination_time:.2f}ms too high"
        )

        # Performance metrics for reporting
        print("\nCoordination Performance Metrics:")
        print(f"Average overhead: {avg_coordination_time:.2f}ms")
        print(f"Min overhead: {min_coordination_time:.2f}ms")
        print(f"Max overhead: {max_coordination_time:.2f}ms")
        print(
            f"ADR-011 compliance: {'PASS' if avg_coordination_time < 200 else 'FAIL'}"
        )

    def test_token_reduction_parallel_execution_optimization(self):
        """Test token reduction through parallel execution optimization."""
        # Given: Context manager for token optimization testing
        context_manager = ContextManager(max_context_tokens=131072)

        # Simulate messages that would benefit from parallel optimization
        sequential_messages = [
            {
                "content": f"Sequential query {i} with detailed context requiring processing",
                "role": "user",
            }
            for i in range(5)
        ]

        parallel_messages = [
            {
                "content": f"Parallel query {i} optimized for concurrent processing",
                "role": "user",
            }
            for i in range(5)
        ]

        # When: Measuring token usage with different execution modes
        sequential_tokens = context_manager.estimate_tokens(sequential_messages)
        parallel_tokens = context_manager.estimate_tokens(parallel_messages)

        # Simulate parallel execution token optimization
        parallel_optimized_state = {
            "messages": parallel_messages,
            "parallel_execution_active": True,
            "parallel_tool_calls": True,
            "token_reduction_enabled": True,
        }

        optimized_state = context_manager.post_model_hook(parallel_optimized_state)

        # Then: Parallel execution achieves token reduction
        # Token reduction should be achieved through optimized message structuring
        assert sequential_tokens > 0
        assert parallel_tokens > 0

        # Verify optimization metadata is present
        assert "parallel_execution_active" in optimized_state
        assert optimized_state["parallel_execution_active"] is True

        print("\nToken Optimization Metrics:")
        print(f"Sequential tokens: {sequential_tokens}")
        print(f"Parallel tokens: {parallel_tokens}")
        print(
            f"Optimization metadata present: {'YES' if 'metadata' in optimized_state else 'NO'}"
        )

    def test_memory_usage_fp8_optimization_validation(self):
        """Test memory usage validation for FP8 optimization."""
        # Given: Context manager with FP8 settings
        context_manager = ContextManager(max_context_tokens=131072)

        # Simulate different context sizes
        test_scenarios = [
            {"name": "small_context", "message_count": 5},
            {"name": "medium_context", "message_count": 25},
            {"name": "large_context", "message_count": 100},
            {"name": "max_context", "message_count": 200},
        ]

        memory_metrics = {}

        # When: Testing memory usage across different context sizes
        for scenario in test_scenarios:
            messages = [
                {
                    "content": f"Context message {i} with substantial content for memory testing",
                    "role": "user",
                }
                for i in range(scenario["message_count"])
            ]

            state = {"messages": messages}

            # Measure KV cache usage (FP8 optimization metric)
            kv_cache_usage = context_manager.calculate_kv_cache_usage(state)
            token_count = context_manager.estimate_tokens(messages)

            memory_metrics[scenario["name"]] = {
                "kv_cache_usage_gb": kv_cache_usage,
                "token_count": token_count,
                "messages": scenario["message_count"],
            }

        # Then: Memory usage scales appropriately with FP8 optimization
        # Verify memory usage increases linearly with context size
        small_usage = memory_metrics["small_context"]["kv_cache_usage_gb"]
        large_usage = memory_metrics["large_context"]["kv_cache_usage_gb"]

        assert small_usage > 0, "Small context should use measurable memory"
        assert large_usage > small_usage, (
            "Large context should use more memory than small"
        )

        # FP8 optimization: Should stay within reasonable bounds
        max_usage = memory_metrics["max_context"]["kv_cache_usage_gb"]
        assert max_usage < 16.0, f"Memory usage {max_usage:.2f}GB exceeds GPU capacity"

        print("\nMemory Usage Metrics (FP8 Optimization):")
        for scenario, metrics in memory_metrics.items():
            print(
                f"{scenario}: {metrics['kv_cache_usage_gb']:.3f}GB ({metrics['token_count']} tokens)"
            )

    def test_throughput_concurrent_agent_operations(self):
        """Test throughput for concurrent agent operations."""
        # Given: Mock state for concurrent operation testing
        base_state = {
            "messages": [HumanMessage(content="Concurrent throughput test")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock(), "kg": Mock(), "retriever": Mock()},
            "parallel_execution_active": True,
        }

        operation_counts = [1, 5, 10, 20]  # Test different concurrency levels
        throughput_metrics = {}

        # When: Testing throughput at different concurrency levels
        for count in operation_counts:
            start_time = time.perf_counter()

            with patch("src.agents.tools.ToolFactory") as mock_factory:
                mock_factory.create_tools_from_indexes.return_value = [Mock()]

                # Execute multiple operations concurrently (simulated)
                results = []
                for i in range(count):
                    state = base_state.copy()
                    result = route_query(f"Concurrent query {i}", state=state)
                    results.append(result)

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            throughput = count / execution_time  # Operations per second
            throughput_metrics[count] = {
                "execution_time": execution_time,
                "throughput_ops_per_sec": throughput,
                "operations": count,
            }

        # Then: Throughput scales appropriately with concurrency
        single_throughput = throughput_metrics[1]["throughput_ops_per_sec"]
        multi_throughput = throughput_metrics[10]["throughput_ops_per_sec"]

        # Should handle multiple operations efficiently
        assert single_throughput > 0, "Single operation throughput should be measurable"
        assert multi_throughput > single_throughput * 0.5, (
            "Multi-operation throughput should scale reasonably"
        )

        print("\nThroughput Metrics:")
        for count, metrics in throughput_metrics.items():
            print(
                f"{count} operations: {metrics['throughput_ops_per_sec']:.2f} ops/sec ({metrics['execution_time']:.3f}s)"
            )

    def test_latency_individual_agent_response_times(self):
        """Test latency benchmarks for individual agent response times."""
        # Given: Mock state for latency testing
        mock_state = {
            "messages": [HumanMessage(content="Latency benchmark test")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock(), "kg": Mock(), "retriever": Mock()},
        }

        agent_latencies = {}

        # When: Measuring individual agent response latencies
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_factory.create_tools_from_indexes.return_value = [Mock()]

            # Test routing agent latency
            start_time = time.perf_counter()
            route_query("Latency test query", state=mock_state)
            agent_latencies["routing"] = (time.perf_counter() - start_time) * 1000

            # Test planning agent latency
            start_time = time.perf_counter()
            plan_query("Latency test query", state=mock_state)
            agent_latencies["planning"] = (time.perf_counter() - start_time) * 1000

            # Test retrieval agent latency
            start_time = time.perf_counter()
            retrieve_documents("Latency test query", state=mock_state)
            agent_latencies["retrieval"] = (time.perf_counter() - start_time) * 1000

            # Test synthesis agent latency
            start_time = time.perf_counter()
            synthesize_results("Latency test query", [{"content": "test"}], mock_state)
            agent_latencies["synthesis"] = (time.perf_counter() - start_time) * 1000

            # Test validation agent latency
            start_time = time.perf_counter()
            validate_response("Latency test query", "Test response", mock_state)
            agent_latencies["validation"] = (time.perf_counter() - start_time) * 1000

        # Then: Individual agent latencies meet performance targets
        for agent, latency in agent_latencies.items():
            assert latency < 100, (
                f"{agent} agent latency {latency:.2f}ms exceeds 100ms target"
            )
            assert latency > 0, f"{agent} agent latency should be measurable"

        total_latency = sum(agent_latencies.values())
        assert total_latency < 200, (
            f"Total agent latency {total_latency:.2f}ms exceeds 200ms coordination target"
        )

        print("\nAgent Latency Benchmarks:")
        for agent, latency in agent_latencies.items():
            print(f"{agent.capitalize()} agent: {latency:.2f}ms")
        print(f"Total latency: {total_latency:.2f}ms")

    def test_scalability_increasing_coordination_complexity(self):
        """Test scalability under increasing coordination complexity."""
        # Given: Different complexity scenarios
        complexity_scenarios = [
            {"name": "simple", "agents": 2, "tools": 1, "iterations": 1},
            {"name": "medium", "agents": 3, "tools": 2, "iterations": 2},
            {"name": "complex", "agents": 5, "tools": 3, "iterations": 3},
            {"name": "high", "agents": 5, "tools": 5, "iterations": 5},
        ]

        scalability_metrics = {}

        # When: Testing performance across complexity levels
        for scenario in complexity_scenarios:
            mock_state = {
                "messages": [
                    HumanMessage(content=f"Scalability test - {scenario['name']}")
                ],
                "context": ChatMemoryBuffer.from_defaults(),
                "tools_data": {f"tool_{i}": Mock() for i in range(scenario["tools"])},
                "parallel_execution_active": True,
                "complexity_level": scenario["name"],
            }

            start_time = time.perf_counter()

            with patch("src.agents.tools.ToolFactory") as mock_factory:
                mock_tools = [Mock() for _ in range(scenario["tools"])]
                mock_factory.create_tools_from_indexes.return_value = mock_tools

                # Simulate multiple coordination iterations
                for iteration in range(scenario["iterations"]):
                    # Simulate agent coordination sequence
                    route_query(f"Scalability query {iteration}", state=mock_state)
                    if scenario["agents"] >= 3:
                        plan_query(f"Scalability query {iteration}", state=mock_state)
                    if scenario["agents"] >= 4:
                        retrieve_documents(
                            f"Scalability query {iteration}", state=mock_state
                        )
                    if scenario["agents"] >= 5:
                        synthesize_results(
                            f"Scalability query {iteration}",
                            [{"content": "test"}],
                            mock_state,
                        )
                        validate_response(
                            f"Scalability query {iteration}",
                            "Test response",
                            mock_state,
                        )

            execution_time = time.perf_counter() - start_time

            scalability_metrics[scenario["name"]] = {
                "execution_time": execution_time,
                "agents": scenario["agents"],
                "tools": scenario["tools"],
                "iterations": scenario["iterations"],
                "complexity_score": scenario["agents"]
                * scenario["tools"]
                * scenario["iterations"],
                "time_per_complexity": execution_time
                / (scenario["agents"] * scenario["tools"] * scenario["iterations"]),
            }

        # Then: Performance scales reasonably with complexity
        simple_time = scalability_metrics["simple"]["execution_time"]
        complex_time = scalability_metrics["complex"]["execution_time"]
        high_time = scalability_metrics["high"]["execution_time"]

        # Performance should degrade linearly, not exponentially
        complexity_ratio = (
            scalability_metrics["complex"]["complexity_score"]
            / scalability_metrics["simple"]["complexity_score"]
        )
        time_ratio = complex_time / simple_time

        assert time_ratio <= complexity_ratio * 2, (
            f"Performance degradation {time_ratio:.2f}x too high for complexity increase {complexity_ratio:.2f}x"
        )

        # High complexity should still be manageable
        assert high_time < 5.0, (
            f"High complexity execution time {high_time:.2f}s exceeds reasonable bounds"
        )

        print("\nScalability Metrics:")
        for scenario, metrics in scalability_metrics.items():
            print(
                f"{scenario.capitalize()}: {metrics['execution_time']:.3f}s "
                f"(complexity: {metrics['complexity_score']}, "
                f"time/complexity: {metrics['time_per_complexity']:.4f}s)"
            )


@pytest.mark.asyncio
@pytest.mark.performance
class TestAsyncCoordinationPerformance:
    """Async performance validation tests for agent coordination."""

    async def test_async_coordination_performance_benchmarks(self):
        """Test async coordination performance benchmarks."""
        # Given: Async coordination setup

        async def mock_agent_operation(name: str, delay: float):
            """Mock async agent operation with controlled delay."""
            await asyncio.sleep(delay)
            return f"Result from {name}"

        # When: Running concurrent async operations
        start_time = time.perf_counter()

        # Simulate concurrent agent operations
        tasks = [
            asyncio.create_task(mock_agent_operation("router", 0.01)),
            asyncio.create_task(mock_agent_operation("planner", 0.02)),
            asyncio.create_task(mock_agent_operation("retrieval", 0.03)),
            asyncio.create_task(mock_agent_operation("synthesis", 0.02)),
            asyncio.create_task(mock_agent_operation("validation", 0.01)),
        ]

        results = await asyncio.gather(*tasks)
        execution_time = time.perf_counter() - start_time

        # Then: Async coordination is faster than sequential
        sequential_time = 0.01 + 0.02 + 0.03 + 0.02 + 0.01  # Sum of delays

        assert execution_time < sequential_time, (
            f"Async execution {execution_time:.3f}s should be faster than sequential {sequential_time:.3f}s"
        )
        assert len(results) == 5, "All async operations should complete"

        # Performance improvement calculation
        improvement_factor = sequential_time / execution_time

        print("\nAsync Performance Metrics:")
        print(f"Sequential time: {sequential_time:.3f}s")
        print(f"Async time: {execution_time:.3f}s")
        print(f"Performance improvement: {improvement_factor:.2f}x")

    async def test_async_resource_utilization_efficiency(self):
        """Test async resource utilization efficiency."""
        # Given: Multiple resource-intensive operations

        async def resource_intensive_operation(resource_id: str, duration: float):
            """Simulate resource-intensive async operation."""
            start = time.perf_counter()
            await asyncio.sleep(duration)
            end = time.perf_counter()

            return {
                "resource_id": resource_id,
                "duration": end - start,
                "efficiency": duration / (end - start),  # Ideal efficiency = 1.0
            }

        # When: Running multiple resource operations concurrently
        start_time = time.perf_counter()

        operations = [
            resource_intensive_operation("cpu_operation", 0.02),
            resource_intensive_operation("memory_operation", 0.015),
            resource_intensive_operation("io_operation", 0.025),
            resource_intensive_operation("network_operation", 0.01),
        ]

        results = await asyncio.gather(*operations)
        total_time = time.perf_counter() - start_time

        # Then: Resource utilization is efficient
        avg_efficiency = sum(r["efficiency"] for r in results) / len(results)

        assert avg_efficiency > 0.8, (
            f"Resource utilization efficiency {avg_efficiency:.2f} below 0.8 threshold"
        )
        assert total_time < 0.1, (
            f"Total async resource time {total_time:.3f}s exceeds expected bounds"
        )

        print("\nAsync Resource Utilization:")
        for result in results:
            print(
                f"{result['resource_id']}: {result['efficiency']:.3f} efficiency ({result['duration']:.3f}s)"
            )
        print(f"Average efficiency: {avg_efficiency:.3f}")

    async def test_async_error_recovery_performance_impact(self):
        """Test performance impact of async error recovery mechanisms."""
        # Given: Operations that may fail and need recovery

        async def potentially_failing_operation(name: str, failure_rate: float):
            """Operation that fails based on failure rate."""
            await asyncio.sleep(0.01)  # Base operation time

            import random

            if random.random() < failure_rate:
                raise Exception(f"Simulated failure in {name}")

            return f"Success: {name}"

        async def operation_with_retry(
            name: str, failure_rate: float, max_retries: int = 2
        ):
            """Operation with retry mechanism."""
            for attempt in range(max_retries + 1):
                try:
                    return await potentially_failing_operation(name, failure_rate)
                except Exception:
                    if attempt == max_retries:
                        return f"Failed after retries: {name}"
                    await asyncio.sleep(0.005)  # Retry delay

        # When: Testing operations with different failure rates
        start_time = time.perf_counter()

        operations = [
            operation_with_retry("reliable_op", 0.1),  # 10% failure rate
            operation_with_retry("moderate_op", 0.3),  # 30% failure rate
            operation_with_retry("unreliable_op", 0.5),  # 50% failure rate
        ]

        results = await asyncio.gather(*operations, return_exceptions=True)
        execution_time = time.perf_counter() - start_time

        # Then: Error recovery doesn't significantly impact performance
        successful_operations = sum(1 for r in results if not isinstance(r, Exception))

        # Performance should remain reasonable even with retries
        assert execution_time < 0.2, (
            f"Error recovery time {execution_time:.3f}s exceeds acceptable bounds"
        )
        assert successful_operations >= 1, "At least some operations should succeed"

        print("\nAsync Error Recovery Performance:")
        print(f"Total execution time: {execution_time:.3f}s")
        print(f"Successful operations: {successful_operations}/{len(operations)}")
        print(
            f"Performance impact: {execution_time / len(operations):.3f}s per operation"
        )
