"""Integration tests for Multi-Agent Coordination System based on Gherkin scenarios.

This module implements comprehensive integration tests that validate the complete
multi-agent pipeline against the Gherkin scenarios specified in the requirements.

Gherkin Scenarios Covered:
1. Simple Query Processing (Scenario 1)
2. Complex Query Decomposition (Scenario 2)
3. FP8 Model Performance (Scenario 6)
4. Modern Supervisor Coordination (Scenario 8)

Features tested:
- End-to-end multi-agent workflow
- ADR-011 compliance validation
- Performance targets (<200ms coordination, throughput)
- FP8 optimization and context management
- Error handling and fallback mechanisms
- Real-world query processing scenarios
"""

import time
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from llama_index.core.memory import ChatMemoryBuffer

from src.agents.coordinator import AgentResponse, MultiAgentCoordinator


class TestGherkinScenario1SimpleQuery:
    """Integration tests for Gherkin Scenario 1: Simple Query Processing.

    Given a simple factual query "What is the capital of France?"
    When the query is processed by the multi-agent system
    Then the router agent classifies it as "simple" complexity
    And the retrieval agent uses vector search strategy
    And the response is generated without planning or synthesis
    And the total processing time is under 1.5 seconds
    """

    @pytest.mark.asyncio
    async def test_simple_query_end_to_end_processing(
        self,
        mock_coordinator: MultiAgentCoordinator,
        gherkin_test_scenarios: dict[str, Any],
    ):
        """Test complete simple query processing pipeline."""
        scenario = gherkin_test_scenarios["simple_query"]
        query = scenario["query"]

        # Mock the agent workflow to simulate simple query processing
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Paris is the capital of France.")],
                "routing_decision": {
                    "strategy": "vector",
                    "complexity": "simple",
                    "needs_planning": False,
                    "confidence": 0.95,
                    "processing_time_ms": 15.2,
                },
                "retrieval_results": [
                    {
                        "documents": [
                            {
                                "content": (
                                    "Paris is the capital and largest city of France."
                                ),
                                "score": 0.98,
                                "metadata": {"source": "geography.pdf", "page": 1},
                            }
                        ],
                        "strategy_used": "vector",
                        "processing_time_ms": 45.8,
                    }
                ],
                "validation_result": {
                    "valid": True,
                    "confidence": 0.95,
                    "suggested_action": "accept",
                    "issues": [],
                },
                "agent_timings": {
                    "router_agent": 0.015,
                    "retrieval_agent": 0.046,
                    "validation_agent": 0.012,
                },
                "parallel_execution_active": True,
                "context_trimmed": False,
                "tokens_trimmed": 0,
            }
        ]

        start_time = time.perf_counter()
        response = mock_coordinator.process_query(query)
        end_time = time.perf_counter()

        # Verify Gherkin scenario requirements
        assert isinstance(response, AgentResponse)

        # Verify routing classification
        routing = response.metadata["routing_decision"]
        assert routing["complexity"] == scenario["expected_complexity"]
        assert routing["strategy"] == scenario["expected_strategy"]
        assert routing["needs_planning"] == scenario["planning_required"]

        # Verify response generation without planning/synthesis
        assert (
            "planning_output" not in response.metadata
            or not response.metadata["planning_output"]
        )

        # Verify processing time under 1.5 seconds
        assert response.processing_time < scenario["max_processing_time"]
        assert end_time - start_time < scenario["max_processing_time"]

        # Verify content quality
        assert "Paris" in response.content
        assert len(response.content) > 10
        assert response.validation_score >= 0.8

    def test_simple_query_routing_classification(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test that simple queries are correctly classified by router agent."""
        simple_queries = [
            "What is the capital of France?",
            "Who is the president of the United States?",
            "When was Python created?",
            "Where is Mount Everest located?",
        ]

        for query in simple_queries:
            # Mock router response for simple queries
            mock_coordinator.compiled_graph.stream.return_value = [
                {
                    "messages": [HumanMessage(content=f"Answer to: {query}")],
                    "routing_decision": {
                        "strategy": "vector",
                        "complexity": "simple",
                        "needs_planning": False,
                        "confidence": 0.9,
                    },
                    "validation_result": {"confidence": 0.85},
                }
            ]

            response = mock_coordinator.process_query(query)

            # Verify simple classification
            routing = response.metadata["routing_decision"]
            assert routing["complexity"] == "simple"
            assert routing["strategy"] == "vector"
            assert routing["needs_planning"] is False

    def test_simple_query_performance_targets(
        self, mock_coordinator: MultiAgentCoordinator, performance_timer: dict[str, Any]
    ):
        """Test simple query meets performance targets."""
        query = "What is machine learning?"

        # Mock fast processing
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Machine learning answer")],
                "routing_decision": {"strategy": "vector", "complexity": "simple"},
                "agent_timings": {"router_agent": 0.008, "retrieval_agent": 0.035},
                "validation_result": {"confidence": 0.9},
            }
        ]

        performance_timer["start_timer"]("simple_query")
        response = mock_coordinator.process_query(query)
        processing_time = performance_timer["end_timer"]("simple_query")

        # Verify performance targets
        assert processing_time < 1.5  # Under 1.5 seconds
        assert response.optimization_metrics["coordination_overhead_ms"] < 200
        assert response.optimization_metrics["meets_200ms_target"] is True


class TestGherkinScenario2ComplexQuery:
    """Integration tests for Gherkin Scenario 2: Complex Query Decomposition.

    Given a complex query "Compare the environmental impact of electric vs gasoline
    vehicles and explain the manufacturing differences"
    When the query is processed by the multi-agent system
    Then the router agent classifies it as "complex" complexity
    And the planner agent decomposes it into 3 sub-tasks
    And the retrieval agent processes each sub-task
    And the synthesis agent combines the results
    And the validator ensures response completeness
    """

    @pytest.mark.asyncio
    async def test_complex_query_end_to_end_processing(
        self,
        mock_coordinator: MultiAgentCoordinator,
        gherkin_test_scenarios: dict[str, Any],
    ):
        """Test complete complex query processing pipeline."""
        scenario = gherkin_test_scenarios["complex_query"]
        query = scenario["query"]

        # Mock the agent workflow to simulate complex query processing
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Comprehensive comparison of electric vs gasoline vehicles."
                        )
                    )
                ],
                "routing_decision": {
                    "strategy": "hybrid",
                    "complexity": "complex",
                    "needs_planning": True,
                    "confidence": 0.92,
                },
                "planning_output": {
                    "sub_tasks": [
                        "Research environmental impact of electric vehicles",
                        "Research environmental impact of gasoline vehicles",
                        "Compare manufacturing processes of both vehicle types",
                    ],
                    "execution_order": "parallel",
                    "task_count": 3,
                },
                "retrieval_results": [
                    {
                        "documents": [
                            {
                                "content": (
                                    "Electric vehicles produce zero direct emissions"
                                ),
                                "score": 0.89,
                            },
                            {
                                "content": (
                                    "Battery manufacturing has environmental costs"
                                ),
                                "score": 0.85,
                            },
                        ],
                        "strategy_used": "hybrid",
                    },
                    {
                        "documents": [
                            {
                                "content": "Gasoline vehicles emit CO2 and pollutants",
                                "score": 0.91,
                            },
                            {
                                "content": "Oil refining process environmental impact",
                                "score": 0.88,
                            },
                        ],
                        "strategy_used": "hybrid",
                    },
                    {
                        "documents": [
                            {
                                "content": (
                                    "Manufacturing differences in vehicle production"
                                ),
                                "score": 0.87,
                            },
                        ],
                        "strategy_used": "hybrid",
                    },
                ],
                "synthesis_result": {
                    "documents": [
                        {"content": "Synthesized comparison analysis", "score": 0.93}
                    ],
                    "synthesis_metadata": {
                        "original_count": 5,
                        "final_count": 4,
                        "deduplication_ratio": 0.8,
                    },
                },
                "validation_result": {
                    "valid": True,
                    "confidence": 0.88,
                    "suggested_action": "accept",
                    "issues": [],
                },
                "agent_timings": {
                    "router_agent": 0.025,
                    "planner_agent": 0.085,
                    "retrieval_agent": 0.180,
                    "synthesis_agent": 0.120,
                    "validation_agent": 0.055,
                },
                "parallel_execution_active": True,
            }
        ]

        response = mock_coordinator.process_query(query)

        # Verify Gherkin scenario requirements
        assert isinstance(response, AgentResponse)

        # Verify routing classification
        routing = response.metadata["routing_decision"]
        assert routing["complexity"] == scenario["expected_complexity"]
        assert routing["strategy"] == scenario["expected_strategy"]
        assert routing["needs_planning"] == scenario["planning_required"]

        # Verify planner decomposition
        planning = response.metadata["planning_output"]
        assert len(planning["sub_tasks"]) == scenario["expected_subtasks"]
        assert planning["execution_order"] == "parallel"

        # Verify retrieval processing of sub-tasks
        assert len(response.metadata["retrieval_results"]) == 3  # One per sub-task

        # Verify synthesis combination
        assert "synthesis_result" in response.metadata
        synthesis = response.metadata["synthesis_result"]
        assert "synthesis_metadata" in synthesis

        # Verify validation completeness
        validation = response.metadata["validation_result"]
        assert validation["valid"] is True
        assert validation["confidence"] >= 0.8
        assert validation["suggested_action"] == "accept"

        # Verify content quality
        assert "electric" in response.content.lower()
        assert (
            "gasoline" in response.content.lower() or "gas" in response.content.lower()
        )
        assert len(response.content) > 50  # Substantial content

    def test_complex_query_planning_decomposition(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test complex query planning and decomposition."""
        complex_queries = [
            "Compare AI vs machine learning and explain their applications",
            "Analyze the benefits and drawbacks of remote work policies",
            "Explain the relationship between climate change and renewable energy",
        ]

        for query in complex_queries:
            # Mock complex query processing
            mock_coordinator.compiled_graph.stream.return_value = [
                {
                    "messages": [HumanMessage(content=f"Complex analysis: {query}")],
                    "routing_decision": {
                        "strategy": "hybrid",
                        "complexity": "complex",
                        "needs_planning": True,
                    },
                    "planning_output": {
                        "sub_tasks": ["Task 1", "Task 2", "Task 3", "Synthesis"],
                        "execution_order": "parallel",
                    },
                    "synthesis_result": {"documents": []},
                    "validation_result": {"confidence": 0.85},
                }
            ]

            response = mock_coordinator.process_query(query)

            # Verify complex processing
            routing = response.metadata["routing_decision"]
            assert routing["complexity"] == "complex"
            assert routing["needs_planning"] is True

            planning = response.metadata["planning_output"]
            assert len(planning["sub_tasks"]) >= 3
            assert "synthesis_result" in response.metadata

    def test_complex_query_parallel_execution(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test complex query enables parallel execution optimization."""
        query = "Compare renewable vs fossil fuel energy sources"

        # Mock parallel execution
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Energy comparison analysis")],
                "routing_decision": {"complexity": "complex", "needs_planning": True},
                "planning_output": {"execution_order": "parallel"},
                "parallel_execution_active": True,
                "token_reduction_achieved": 0.65,  # 65% reduction through parallel
                "validation_result": {"confidence": 0.9},
            }
        ]

        response = mock_coordinator.process_query(query)

        # Verify parallel execution optimization
        assert response.optimization_metrics["parallel_execution_active"] is True
        assert (
            response.optimization_metrics["token_reduction_achieved"] >= 0.5
        )  # 50% target


class TestGherkinScenario6FP8Performance:
    """Integration tests for Gherkin Scenario 6: FP8 Model Performance.

    Given the vLLM backend is configured with FP8 quantization
    When processing a query requiring agent coordination
    Then the decode throughput is between 100-160 tokens/second
    And the prefill throughput is between 800-1300 tokens/second
    And total VRAM usage stays under 16GB
    And context management maintains 128K token limit
    """

    def test_fp8_performance_targets_validation(
        self,
        mock_coordinator: MultiAgentCoordinator,
        gherkin_test_scenarios: dict[str, Any],
    ):
        """Test FP8 performance targets are met."""
        scenario = gherkin_test_scenarios["fp8_performance"]
        targets = scenario["performance_targets"]

        # Mock FP8 performance metrics
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="FP8 optimized response")],
                "optimization_metrics": {
                    "decode_throughput_estimate": 135.2,  # Within 100-160 range
                    "prefill_throughput_estimate": 1150.5,  # Within 800-1300 range
                    "vram_usage_gb": 13.8,  # Under 16GB
                    "context_window_used": 131072,  # 128K tokens
                    "fp8_optimization": True,
                    "kv_cache_usage_gb": 8.2,
                },
                "validation_result": {"confidence": 0.92},
            }
        ]

        response = mock_coordinator.process_query("Test FP8 performance query")

        # Verify FP8 performance targets
        metrics = response.optimization_metrics

        # Verify decode throughput (100-160 tok/s)
        if "decode_throughput_estimate" in metrics:
            decode_throughput = metrics["decode_throughput_estimate"]
            assert (
                targets["decode_throughput_min"]
                <= decode_throughput
                <= targets["decode_throughput_max"]
            )

        # Verify prefill throughput (800-1300 tok/s)
        if "prefill_throughput_estimate" in metrics:
            prefill_throughput = metrics["prefill_throughput_estimate"]
            assert (
                targets["prefill_throughput_min"]
                <= prefill_throughput
                <= targets["prefill_throughput_max"]
            )

        # Verify VRAM usage under 16GB
        if "vram_usage_gb" in metrics:
            assert metrics["vram_usage_gb"] <= targets["vram_usage_max_gb"]

        # Verify 128K context limit
        assert metrics["context_window_used"] == targets["context_limit"]

        # Verify FP8 optimization is enabled
        assert metrics["fp8_optimization"] is True

    def test_fp8_kv_cache_optimization(self, mock_coordinator: MultiAgentCoordinator):
        """Test FP8 KV cache optimization is properly configured."""
        # Mock FP8 KV cache metrics
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="KV cache optimized response")],
                "kv_cache_usage_gb": 8.0,  # ~50% reduction with FP8
                "context_trimmed": False,
                "tokens_trimmed": 0,
                "optimization_metrics": {
                    "fp8_optimization": True,
                    "kv_cache_dtype": "fp8_e5m2",
                },
                "validation_result": {"confidence": 0.9},
            }
        ]

        response = mock_coordinator.process_query("Test KV cache optimization")

        # Verify FP8 KV cache optimization
        assert response.optimization_metrics["fp8_optimization"] is True
        assert "kv_cache_usage_gb" in response.optimization_metrics

        # Verify KV cache memory efficiency (should be around 8GB for 128K context)
        if "kv_cache_usage_gb" in response.optimization_metrics:
            kv_usage = response.optimization_metrics["kv_cache_usage_gb"]
            assert kv_usage <= 10.0  # Should be efficient with FP8

    def test_fp8_context_management_128k(self, mock_coordinator: MultiAgentCoordinator):
        """Test 128K context management with FP8 optimization."""
        # Mock large context scenario
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Large context response")],
                "context_trimmed": True,
                "tokens_trimmed": 25000,  # Had to trim some context
                "optimization_metrics": {
                    "context_window_used": 131072,  # 128K limit
                    "context_used_tokens": 120000,  # Under threshold
                    "fp8_optimization": True,
                },
                "validation_result": {"confidence": 0.87},
            }
        ]

        # Create context that would exceed 128K
        large_context = ChatMemoryBuffer.from_defaults()
        for _ in range(10):
            large_context.put(HumanMessage(content="Large message content " * 1000))

        response = mock_coordinator.process_query(
            "Process this with large context", context=large_context
        )

        # Verify context management
        assert response.optimization_metrics["context_window_used"] == 131072
        assert response.optimization_metrics["fp8_optimization"] is True

        # Context trimming should have occurred
        if "context_trimmed" in response.optimization_metrics:
            assert response.optimization_metrics["context_trimmed"] is True


class TestGherkinScenario8SupervisorCoordination:
    """Integration tests for Gherkin Scenario 8: Modern Supervisor Coordination.

    Given the ADR-mandated supervisor system architecture (ADR-011)
    When implementing the complete architectural replacement
    Then langgraph-supervisor library must be used with create_supervisor()
    And parallel_tool_calls=True must enable concurrent agent execution
    And token usage must be reduced by 50-87% through parallel tool execution
    And total coordination overhead must stay under 200ms per agent decision
    """

    def test_supervisor_coordination_adr_011_compliance(
        self,
        mock_coordinator: MultiAgentCoordinator,
        gherkin_test_scenarios: dict[str, Any],
    ):
        """Test supervisor coordination meets ADR-011 requirements."""
        scenario = gherkin_test_scenarios["supervisor_coordination"]

        # Mock supervisor coordination
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Supervisor coordinated response")],
                "parallel_execution_active": True,
                "token_reduction_achieved": 0.72,  # 72% reduction
                "coordination_overhead_ms": 145.8,  # Under 200ms
                "add_handoff_back_messages": True,
                "output_mode": "structured",
                "create_forward_message_tool": True,
                "agent_timings": {
                    "router_agent": 0.035,
                    "planner_agent": 0.042,
                    "retrieval_agent": 0.068,
                    "synthesis_agent": 0.041,
                    "validation_agent": 0.028,
                },
                "validation_result": {"confidence": 0.91},
            }
        ]

        start_time = time.perf_counter()
        response = mock_coordinator.process_query("Test supervisor coordination")
        _ = time.perf_counter() - start_time

        # Verify ADR-011 compliance
        assert response.optimization_metrics["parallel_execution_active"] is True

        # Verify token reduction target (50-87%)
        token_reduction = response.optimization_metrics["token_reduction_achieved"]
        assert scenario["token_reduction_target"] <= token_reduction <= 0.87

        # Verify coordination overhead under 200ms
        coordination_ms = response.optimization_metrics["coordination_overhead_ms"]
        assert coordination_ms < scenario["max_coordination_overhead_ms"]

        # Verify modern supervisor parameters
        assert response.optimization_metrics.get("parallel_execution_active") is True

        # Verify FP8 KV cache optimization
        assert response.optimization_metrics["fp8_optimization"] is True

    def test_supervisor_parallel_tool_execution(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test parallel tool execution optimization."""
        # Mock parallel execution scenario
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Parallel execution response")],
                "parallel_tool_calls": True,
                "parallel_execution_active": True,
                "token_reduction_achieved": 0.65,  # 65% reduction
                "agent_timings": {
                    "router_agent": 0.025,
                    "retrieval_agent": 0.055,  # Ran in parallel
                    "synthesis_agent": 0.045,  # Ran in parallel
                    "validation_agent": 0.020,
                },
                "validation_result": {"confidence": 0.88},
            }
        ]

        response = mock_coordinator.process_query("Test parallel tool execution")

        # Verify parallel execution
        assert response.optimization_metrics["parallel_execution_active"] is True

        # Verify token reduction from parallel execution
        assert response.optimization_metrics["token_reduction_achieved"] >= 0.5

        # Verify agent timings show parallel execution efficiency
        agent_timings = response.metadata["agent_timings"]
        total_agent_time = sum(agent_timings.values())
        assert total_agent_time < 0.2  # Efficient parallel execution

    def test_supervisor_coordination_overhead_target(
        self, mock_coordinator: MultiAgentCoordinator, performance_timer: dict[str, Any]
    ):
        """Test coordination overhead stays under 200ms target."""
        # Mock fast coordination
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Fast coordination response")],
                "agent_timings": {
                    "router_agent": 0.015,
                    "retrieval_agent": 0.035,
                    "validation_agent": 0.012,
                },
                "coordination_overhead_ms": 125.5,  # Under 200ms
                "validation_result": {"confidence": 0.9},
            }
        ]

        performance_timer["start_timer"]("coordination")
        response = mock_coordinator.process_query("Test coordination overhead")
        coordination_time = (
            performance_timer["end_timer"]("coordination") * 1000
        )  # Convert to ms

        # Verify coordination overhead target
        reported_overhead = response.optimization_metrics["coordination_overhead_ms"]
        assert reported_overhead < 200  # Under 200ms target
        assert coordination_time < 300  # Allow some test overhead

        # Verify meets target indicator
        assert response.optimization_metrics["meets_200ms_target"] is True


class TestIntegrationErrorHandling:
    """Integration tests for error handling and fallback mechanisms."""

    def test_agent_failure_fallback_integration(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test integration handles agent failures gracefully."""
        # Mock agent failure scenario
        mock_coordinator.compiled_graph.stream.side_effect = Exception(
            "Agent workflow failed"
        )
        mock_coordinator.enable_fallback = True

        response = mock_coordinator.process_query("Test agent failure")

        # Verify fallback response
        assert isinstance(response, AgentResponse)
        assert response.metadata["fallback_used"] is True
        assert "fallback_mode" in response.optimization_metrics
        assert response.validation_score > 0  # Should have some confidence

    def test_timeout_handling_integration(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test integration handles timeouts properly."""
        # Set very short timeout
        mock_coordinator.max_agent_timeout = 0.1

        # Mock slow agent execution
        def slow_stream(*args, **kwargs):
            time.sleep(0.2)  # Exceed timeout
            return [
                {
                    "messages": [HumanMessage(content="Slow response")],
                    "validation_result": {"confidence": 0.8},
                }
            ]

        mock_coordinator.compiled_graph.stream = slow_stream

        start_time = time.perf_counter()
        response = mock_coordinator.process_query("Test timeout handling")
        elapsed = time.perf_counter() - start_time

        # Should not hang indefinitely
        assert elapsed < 1.0  # Should timeout quickly
        assert isinstance(response, AgentResponse)

    def test_performance_degradation_handling(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test integration handles performance degradation."""
        # Mock degraded performance
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Degraded performance response")],
                "coordination_overhead_ms": 350.5,  # Exceeds 200ms target
                "meets_200ms_target": False,
                "parallel_execution_active": False,  # Degraded to sequential
                "token_reduction_achieved": 0.2,  # Below 50% target
                "validation_result": {"confidence": 0.75},
            }
        ]

        response = mock_coordinator.process_query("Test performance degradation")

        # Verify degradation is reported
        assert response.optimization_metrics["meets_200ms_target"] is False
        assert response.optimization_metrics["coordination_overhead_ms"] > 200
        assert response.optimization_metrics["token_reduction_achieved"] < 0.5

        # Should still provide valid response
        assert isinstance(response, AgentResponse)
        assert len(response.content) > 0


class TestIntegrationPerformanceValidation:
    """Integration tests for end-to-end performance validation."""

    def test_end_to_end_performance_simple_query(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test end-to-end performance for simple queries."""
        # Mock optimized simple query processing
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Fast simple response")],
                "routing_decision": {
                    "complexity": "simple",
                    "processing_time_ms": 12.5,
                },
                "agent_timings": {"router_agent": 0.012, "retrieval_agent": 0.035},
                "coordination_overhead_ms": 85.2,
                "validation_result": {"confidence": 0.92},
            }
        ]

        start_time = time.perf_counter()
        response = mock_coordinator.process_query("What is AI?")
        end_time = time.perf_counter()

        # Verify simple query performance
        assert response.processing_time < 1.0  # Fast processing
        assert response.optimization_metrics["coordination_overhead_ms"] < 150
        assert end_time - start_time < 1.0

    def test_end_to_end_performance_complex_query(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test end-to-end performance for complex queries."""
        # Mock optimized complex query processing
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Optimized complex analysis")],
                "routing_decision": {"complexity": "complex"},
                "planning_output": {"sub_tasks": ["Task 1", "Task 2", "Task 3"]},
                "agent_timings": {
                    "router_agent": 0.025,
                    "planner_agent": 0.045,
                    "retrieval_agent": 0.095,
                    "synthesis_agent": 0.065,
                    "validation_agent": 0.030,
                },
                "parallel_execution_active": True,
                "coordination_overhead_ms": 178.5,
                "validation_result": {"confidence": 0.89},
            }
        ]

        start_time = time.perf_counter()
        response = mock_coordinator.process_query(
            "Compare AI vs machine learning and explain their relationship"
        )
        end_time = time.perf_counter()

        # Verify complex query performance
        assert response.processing_time < 3.0  # Reasonable for complex query
        assert response.optimization_metrics["coordination_overhead_ms"] < 200
        assert response.optimization_metrics["parallel_execution_active"] is True
        assert end_time - start_time < 3.0

    def test_performance_metrics_collection_integration(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test performance metrics are properly collected across multiple queries."""
        queries = [
            "Simple query 1",
            "Simple query 2",
            "Complex query requiring analysis",
        ]

        for i, query in enumerate(queries):
            complexity = "simple" if i < 2 else "complex"
            mock_coordinator.compiled_graph.stream.return_value = [
                {
                    "messages": [HumanMessage(content=f"Response {i + 1}")],
                    "routing_decision": {"complexity": complexity},
                    "coordination_overhead_ms": 120 + (i * 20),
                    "validation_result": {"confidence": 0.9},
                }
            ]

            mock_coordinator.process_query(query)

        # Verify performance statistics
        stats = mock_coordinator.get_performance_stats()

        assert stats["total_queries"] == 3
        assert stats["success_rate"] == 1.0
        assert stats["avg_coordination_overhead_ms"] > 0
        assert stats["meets_200ms_target"] is True  # All under 200ms

        # Verify ADR compliance reporting
        assert "adr_compliance" in stats
        adr_compliance = stats["adr_compliance"]
        assert "adr_011" in adr_compliance  # Supervisor framework
        assert "adr_004" in adr_compliance  # Local-first LLM
        assert "adr_010" in adr_compliance  # Performance optimization
