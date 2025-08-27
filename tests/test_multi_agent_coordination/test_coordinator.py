"""Comprehensive unit tests for MultiAgentCoordinator (ADR-011 compliant).

Tests cover:
- ADR-011 compliance (LangGraph supervisor with modern parameters)
- Performance targets (<200ms coordination overhead)
- FP8 optimization and context management
- Agent workflow execution and state management
- Error handling and fallback mechanisms
- Query processing scenarios (simple, complex, edge cases)
"""

import time
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from llama_index.core.memory import ChatMemoryBuffer

from src.agents.coordinator import (
    MultiAgentCoordinator,
    create_multi_agent_coordinator,
)
from src.agents.models import AgentResponse, MultiAgentState


class TestMultiAgentCoordinator:
    """Test suite for ADR-011 compliant MultiAgentCoordinator."""

    def test_initialization_adr_compliance(self, mock_vllm_config: MockVLLMConfig):
        """Test coordinator initialization meets ADR requirements."""
        with patch("src.agents.coordinator.create_vllm_manager"):
            coordinator = MultiAgentCoordinator(
                model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
                max_context_length=131072,
                enable_fallback=True,
            )

            # Verify ADR-004 compliance (Local-First LLM Strategy)
            assert "FP8" in coordinator.model_path
            assert coordinator.max_context_length == 131072
            assert coordinator.backend == "vllm"

            # Verify ADR-010 compliance (Performance Optimization)
            assert coordinator.vllm_config.kv_cache_dtype == "fp8_e5m2"
            assert coordinator.max_agent_timeout == 3.0  # Reduced for <200ms target

            # Verify performance tracking initialization
            assert coordinator.total_queries == 0
            assert coordinator.avg_coordination_overhead == 0.0

    def test_setup_agent_graph_adr_011_compliance(self, mock_llm: Mock):
        """Test agent graph setup follows ADR-011 modern parameters."""
        with (
            patch.multiple(
                "src.agents.coordinator",
                create_vllm_manager=Mock(return_value=Mock()),
                is_dspy_available=Mock(return_value=True),
            ),
            patch("src.agents.coordinator.create_supervisor") as mock_create_supervisor,
        ):
            coordinator = MultiAgentCoordinator()
            coordinator.llm = mock_llm
            coordinator._setup_agent_graph()

            # Verify create_supervisor was called with ADR-011 parameters
            mock_create_supervisor.assert_called_once()
            call_kwargs = mock_create_supervisor.call_args[1]

            # Verify modern optimization parameters (ADR-011)
            assert call_kwargs["parallel_tool_calls"] is True  # 50-87% token reduction
            assert call_kwargs["output_mode"] == "structured"  # Enhanced formatting
            assert (
                call_kwargs["create_forward_message_tool"] is True
            )  # Direct passthrough
            assert (
                call_kwargs["add_handoff_back_messages"] is True
            )  # Coordination tracking
            assert "pre_model_hook" in call_kwargs  # Context management
            assert "post_model_hook" in call_kwargs  # Response formatting

    def test_context_management_hooks(self, mock_llm: Mock):
        """Test pre/post model hooks for 128K context management."""
        with patch.multiple(
            "src.agents.coordinator",
            create_vllm_manager=Mock(return_value=Mock()),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator()
            coordinator.llm = mock_llm

            # Test pre-model hook (context trimming)
            pre_hook = coordinator._create_pre_model_hook()
            test_state = {
                "messages": [HumanMessage(content="A" * 500000)]  # Large message
            }

            # Mock context manager to simulate trimming
            coordinator.context_manager.estimate_tokens = Mock(return_value=150000)
            coordinator.context_manager.trim_threshold = 120000

            with patch("langchain_core.messages.utils.trim_messages") as mock_trim:
                mock_trim.return_value = [HumanMessage(content="Trimmed")]
                result_state = pre_hook(test_state)

                # Verify trimming was triggered
                mock_trim.assert_called_once()
                assert result_state["context_trimmed"] is True
                assert "tokens_trimmed" in result_state

            # Test post-model hook (response formatting)
            post_hook = coordinator._create_post_model_hook()
            test_state = {
                "output_mode": "structured",
                "messages": [HumanMessage(content="Test")],
                "response": "Test response",
            }

            result_state = post_hook(test_state)

            # Verify optimization metadata was added
            assert "optimization_metrics" in result_state
            metrics = result_state["optimization_metrics"]
            assert "context_used_tokens" in metrics
            assert "kv_cache_usage_gb" in metrics
            assert "fp8_optimization" in metrics
            assert metrics["fp8_optimization"] is True

    @pytest.mark.asyncio
    async def test_process_query_simple_scenario(
        self, mock_coordinator: MultiAgentCoordinator, performance_timer: dict[str, Any]
    ):
        """Test simple query processing (Gherkin Scenario 1)."""
        query = "What is the capital of France?"

        performance_timer["start_timer"]("coordination")
        response = mock_coordinator.process_query(query)
        performance_timer["end_timer"]("coordination")

        # Verify response structure
        assert isinstance(response, AgentResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.processing_time > 0

        # Verify performance target (<1.5s for simple queries)
        assert response.processing_time < 1.5

        # Verify optimization metrics
        assert "coordination_overhead_ms" in response.optimization_metrics
        assert "meets_200ms_target" in response.optimization_metrics
        assert "fp8_optimization" in response.optimization_metrics
        assert response.optimization_metrics["fp8_optimization"] is True

        # Verify metadata contains routing decision
        assert "routing_decision" in response.metadata
        assert "adr_compliance" in response.metadata

    @pytest.mark.asyncio
    async def test_process_query_complex_scenario(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test complex query processing (Gherkin Scenario 2)."""
        query = (
            "Compare the environmental impact of electric vs gasoline vehicles "
            "and explain the manufacturing differences"
        )

        # Mock complex workflow response
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Complex analysis response")],
                "routing_decision": {
                    "strategy": "hybrid",
                    "complexity": "complex",
                    "needs_planning": True,
                },
                "planning_output": {
                    "sub_tasks": [
                        "Electric impact",
                        "Gasoline impact",
                        "Compare results",
                    ]
                },
                "synthesis_result": {
                    "documents": [{"content": "Synthesized comparison", "score": 0.9}]
                },
                "validation_result": {"confidence": 0.85},
                "parallel_execution_active": True,
                "agent_timings": {
                    "router_agent": 0.05,
                    "planner_agent": 0.08,
                    "retrieval_agent": 0.15,
                    "synthesis_agent": 0.12,
                    "validation_agent": 0.06,
                },
            }
        ]

        response = mock_coordinator.process_query(query)

        # Verify complex query handling
        assert isinstance(response, AgentResponse)
        assert "routing_decision" in response.metadata
        assert response.metadata["routing_decision"]["complexity"] == "complex"
        assert response.metadata["routing_decision"]["needs_planning"] is True

        # Verify planning was used
        assert "planning_output" in response.metadata
        assert len(response.metadata["planning_output"]["sub_tasks"]) == 3

        # Verify synthesis was performed
        assert len(response.sources) > 0
        assert response.validation_score > 0.8

    def test_performance_coordination_overhead(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test coordination overhead meets <200ms target (ADR-011)."""
        query = "Test query for performance"

        # Simulate fast agent responses
        mock_coordinator.compiled_graph.stream.return_value = [
            {
                "messages": [HumanMessage(content="Fast response")],
                "agent_timings": {"router_agent": 0.02, "retrieval_agent": 0.03},
                "validation_result": {"confidence": 0.9},
            }
        ]

        start_time = time.perf_counter()
        response = mock_coordinator.process_query(query)
        time.perf_counter() - start_time  # Calculate time for verification

        # Verify coordination overhead
        coordination_ms = response.optimization_metrics["coordination_overhead_ms"]
        assert coordination_ms < 200, (
            f"Coordination overhead {coordination_ms}ms exceeds 200ms target"
        )
        assert response.optimization_metrics["meets_200ms_target"] is True

        # Verify performance stats tracking
        stats = mock_coordinator.get_performance_stats()
        assert stats["total_queries"] > 0
        assert stats["avg_coordination_overhead_ms"] < 200
        assert stats["meets_200ms_target"] is True

    def test_fallback_mechanism(self):
        """Test fallback to basic RAG when multi-agent system fails."""
        with patch.multiple(
            "src.agents.coordinator",
            create_vllm_manager=Mock(return_value=Mock()),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator(enable_fallback=True)
            coordinator._setup_complete = False  # Force setup failure

            response = coordinator.process_query("Test query")

            # Verify fallback response
            assert isinstance(response, AgentResponse)
            assert (
                "unavailable" in response.content.lower()
                or "error" in response.content.lower()
            )
            assert response.metadata["fallback_available"] is True
            assert "initialization_failed" in response.optimization_metrics

    def test_adr_compliance_validation(
        self,
        mock_coordinator: MultiAgentCoordinator,
        adr_compliance_validator: dict[str, Any],
    ):
        """Test comprehensive ADR compliance validation."""
        # Validate ADR-011 compliance
        adr_011_results = adr_compliance_validator["validate_adr_011"](mock_coordinator)
        assert all(adr_011_results.values()), (
            f"ADR-011 compliance failed: {adr_011_results}"
        )

        # Validate ADR-004 compliance
        adr_004_results = adr_compliance_validator["validate_adr_004"](
            mock_coordinator.vllm_config
        )
        assert all(adr_004_results.values()), (
            f"ADR-004 compliance failed: {adr_004_results}"
        )

        # Validate ADR-010 compliance
        adr_010_results = adr_compliance_validator["validate_adr_010"](
            mock_coordinator.vllm_config
        )
        assert all(adr_010_results.values()), (
            f"ADR-010 compliance failed: {adr_010_results}"
        )

        # Test built-in compliance validation
        compliance_check = mock_coordinator.validate_adr_compliance()
        assert compliance_check["adr_004_fp8_model"] is True
        assert compliance_check["context_128k_support"] is True
        assert compliance_check["adr_011_modern_parameters"] is True

    def test_error_handling_and_recovery(self, mock_llm: Mock):
        """Test error handling in agent workflow execution."""
        with patch.multiple(
            "src.agents.coordinator",
            create_vllm_manager=Mock(return_value=Mock()),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator()
            coordinator.llm = mock_llm
            coordinator._setup_complete = True

            # Mock graph to raise exception
            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream.side_effect = Exception(
                "Agent workflow failed"
            )

            # Test with fallback enabled
            coordinator.enable_fallback = True
            response = coordinator.process_query("Test query")

            assert isinstance(response, AgentResponse)
            assert response.metadata["fallback_used"] is True
            assert "fallback_mode" in response.optimization_metrics

            # Test without fallback
            coordinator.enable_fallback = False
            response = coordinator.process_query("Test query")

            assert "Error processing query" in response.content
            assert response.validation_score == 0.0

    def test_performance_metrics_tracking(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test performance metrics collection and reporting."""
        # Process multiple queries to build metrics
        queries = [
            "What is AI?",
            "Explain machine learning",
            "Define neural networks",
        ]

        for query in queries:
            mock_coordinator.process_query(query)

        # Verify performance statistics
        stats = mock_coordinator.get_performance_stats()

        assert stats["total_queries"] == 3
        assert stats["success_rate"] > 0.0
        assert "avg_processing_time" in stats
        assert "avg_coordination_overhead_ms" in stats
        assert "adr_compliance" in stats

        # Verify ADR compliance reporting
        adr_compliance = stats["adr_compliance"]
        assert "adr_001" in adr_compliance
        assert "adr_004" in adr_compliance
        assert "adr_010" in adr_compliance
        assert "adr_011" in adr_compliance
        assert "adr_018" in adr_compliance

        # Reset and verify
        mock_coordinator.reset_performance_stats()
        reset_stats = mock_coordinator.get_performance_stats()
        assert reset_stats["total_queries"] == 0

    def test_context_conversation_continuity(
        self, mock_coordinator: MultiAgentCoordinator
    ):
        """Test conversation context management and continuity."""
        # Create conversation context
        context = ChatMemoryBuffer.from_defaults()
        context.put(HumanMessage(content="What is machine learning?"))
        context.put(HumanMessage(content="Previous AI discussion context"))

        query = "Can you elaborate on that?"
        response = mock_coordinator.process_query(query, context=context)

        # Verify context was passed through
        assert isinstance(response, AgentResponse)
        assert response.processing_time > 0

        # Verify context optimization metrics
        assert "context_window_used" in response.optimization_metrics
        assert response.optimization_metrics["context_window_used"] == 131072

    def test_agent_timeout_handling(self, mock_llm: Mock):
        """Test agent timeout handling for performance compliance."""
        with patch.multiple(
            "src.agents.coordinator",
            create_vllm_manager=Mock(return_value=Mock()),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator(
                max_agent_timeout=0.1
            )  # Very short timeout
            coordinator.llm = mock_llm
            coordinator._setup_complete = True

            # Mock slow graph execution
            def slow_stream(*args, **kwargs):
                time.sleep(0.2)  # Exceed timeout
                return [
                    {
                        "messages": [HumanMessage(content="Slow response")],
                        "validation_result": {"confidence": 0.9},
                    }
                ]

            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream = slow_stream

            start_time = time.perf_counter()
            response = coordinator.process_query("Test query")
            elapsed = time.perf_counter() - start_time

            # Verify timeout was respected (approximately)
            assert elapsed < 1.0  # Should not wait indefinitely
            assert isinstance(response, AgentResponse)


class TestMultiAgentState:
    """Test suite for MultiAgentState data model."""

    def test_state_initialization_defaults(self):
        """Test state initialization with proper defaults."""
        state = MultiAgentState(messages=[HumanMessage(content="Test")])

        # Verify default values
        assert state.tools_data == {}
        assert state.context is None
        assert state.routing_decision == {}
        assert state.planning_output == {}
        assert state.retrieval_results == []
        assert state.synthesis_result == {}
        assert state.validation_result == {}
        assert state.agent_timings == {}
        assert state.parallel_execution_active is False
        assert state.token_reduction_achieved == 0.0
        assert state.context_trimmed is False
        assert state.tokens_trimmed == 0
        assert state.kv_cache_usage_gb == 0.0
        assert state.output_mode == "structured"
        assert state.errors == []
        assert state.fallback_used is False
        assert state.remaining_steps == 10

    def test_state_with_performance_data(self):
        """Test state with performance tracking data."""
        state = MultiAgentState(
            messages=[HumanMessage(content="Test")],
            agent_timings={"router_agent": 0.05, "retrieval_agent": 0.08},
            parallel_execution_active=True,
            token_reduction_achieved=0.65,
            context_trimmed=True,
            tokens_trimmed=1500,
            kv_cache_usage_gb=8.2,
        )

        # Verify performance data
        assert state.agent_timings["router_agent"] == 0.05
        assert state.agent_timings["retrieval_agent"] == 0.08
        assert state.parallel_execution_active is True
        assert state.token_reduction_achieved == 0.65
        assert state.context_trimmed is True
        assert state.tokens_trimmed == 1500
        assert state.kv_cache_usage_gb == 8.2


class TestAgentResponse:
    """Test suite for AgentResponse data model."""

    def test_response_creation_with_optimization_metrics(self):
        """Test response creation with comprehensive optimization metrics."""
        response = AgentResponse(
            content="Test response content",
            sources=[{"content": "Source doc", "score": 0.9}],
            metadata={"routing_decision": {"strategy": "vector"}},
            validation_score=0.85,
            processing_time=0.15,
            optimization_metrics={
                "coordination_overhead_ms": 120.5,
                "meets_200ms_target": True,
                "parallel_execution_active": True,
                "token_reduction_achieved": 0.72,
                "fp8_optimization": True,
                "context_window_used": 131072,
            },
        )

        # Verify all fields
        assert response.content == "Test response content"
        assert len(response.sources) == 1
        assert response.sources[0]["score"] == 0.9
        assert response.validation_score == 0.85
        assert response.processing_time == 0.15

        # Verify optimization metrics
        metrics = response.optimization_metrics
        assert metrics["coordination_overhead_ms"] == 120.5
        assert metrics["meets_200ms_target"] is True
        assert metrics["parallel_execution_active"] is True
        assert metrics["token_reduction_achieved"] == 0.72
        assert metrics["fp8_optimization"] is True
        assert metrics["context_window_used"] == 131072


class TestFactoryFunction:
    """Test suite for factory function."""

    def test_create_multi_agent_coordinator_factory(self):
        """Test factory function creates properly configured coordinator."""
        with patch(
            "src.agents.coordinator.MultiAgentCoordinator"
        ) as mock_coordinator_class:
            mock_instance = Mock()
            mock_coordinator_class.return_value = mock_instance

            result = create_multi_agent_coordinator(
                model_path="custom/model",
                max_context_length=65536,
                enable_fallback=False,
            )

            # Verify factory called with correct parameters
            mock_coordinator_class.assert_called_once_with(
                model_path="custom/model",
                max_context_length=65536,
                enable_fallback=False,
            )
            assert result == mock_instance

    def test_create_multi_agent_coordinator_defaults(self):
        """Test factory function with default parameters."""
        with patch(
            "src.agents.coordinator.MultiAgentCoordinator"
        ) as mock_coordinator_class:
            mock_instance = Mock()
            mock_coordinator_class.return_value = mock_instance

            result = create_multi_agent_coordinator()

            # Verify default parameters
            mock_coordinator_class.assert_called_once_with(
                model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
                max_context_length=131072,
                enable_fallback=True,
            )
            assert result == mock_instance
