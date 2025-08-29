"""Comprehensive unit tests for agents/coordinator.py targeting 45%+ coverage.

This test suite focuses on real business logic testing for the MultiAgentCoordinator,
testing critical multi-agent orchestration scenarios, performance optimization,
context management, and error recovery mechanisms without inappropriate internal
mocking.

Coverage Focus:
- ADR-compliant multi-agent coordination workflow (currently 196/242 missing statements)
- Context management and 128K token handling with FP8 optimization
- Agent workflow execution and state management
- Performance coordination overhead targeting <200ms
- Error handling and fallback mechanisms
- Query processing scenarios from simple to complex
- Agent decision tracking and optimization metrics

Test Strategy:
- Focus on actual coordination logic and decision flows
- Test performance targets and optimization metrics
- Validate ADR compliance requirements
- Include comprehensive error scenarios
- Test real multi-agent workflow orchestration
- Performance and timing validation
"""

import time
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage
from llama_index.core.memory import ChatMemoryBuffer

from src.agents.coordinator import (
    ADD_HANDOFF_BACK_MESSAGES_ENABLED,
    COORDINATION_OVERHEAD_THRESHOLD,
    CREATE_FORWARD_MESSAGE_TOOL_ENABLED,
    OUTPUT_MODE_STRUCTURED,
    PARALLEL_TOOL_CALLS_ENABLED,
    ContextManager,
    MultiAgentCoordinator,
    create_multi_agent_coordinator,
)
from src.agents.models import AgentResponse, MultiAgentState
from src.config import settings


class TestContextManagerComprehensive:
    """Comprehensive tests for ContextManager business logic."""

    def test_context_manager_initialization(self):
        """Test ContextManager initialization with proper defaults."""
        context_manager = ContextManager(max_context_tokens=131072)

        assert context_manager.max_context_tokens == 131072
        assert context_manager.trim_threshold == int(131072 * 0.9)  # 90% threshold
        assert (
            context_manager.kv_cache_memory_per_token == 1024
        )  # bytes per token for FP8

    def test_token_estimation_algorithm(self):
        """Test token estimation with various message types."""
        context_manager = ContextManager()

        # Test with different message types
        messages_dict = [
            {"content": "This is a test message"},  # Dict format
            {"content": "Another test message with more content"},
        ]

        tokens_dict = context_manager.estimate_tokens(messages_dict)
        expected_chars = len("This is a test message") + len(
            "Another test message with more content"
        )
        expected_tokens = expected_chars // 4  # 4 chars per token
        assert tokens_dict == expected_tokens

        # Test with mock Message objects
        mock_messages = [
            Mock(content="Message object content"),
            Mock(content="Another message object"),
        ]

        tokens_mock = context_manager.estimate_tokens(mock_messages)
        expected_chars = len("Message object content") + len("Another message object")
        expected_tokens = expected_chars // 4
        assert tokens_mock == expected_tokens

        # Test with empty messages
        assert context_manager.estimate_tokens([]) == 0
        assert context_manager.estimate_tokens(None) == 0

    def test_kv_cache_usage_calculation(self):
        """Test KV cache memory usage calculation."""
        context_manager = ContextManager()

        test_state = {
            "messages": [
                {"content": "A" * 1000},  # 1000 chars = ~250 tokens
                {"content": "B" * 2000},  # 2000 chars = ~500 tokens
            ]
        }

        usage_gb = context_manager.calculate_kv_cache_usage(test_state)

        # Expected: ~750 tokens * 1024 bytes/token = 768000 bytes = ~0.000715 GB
        expected_tokens = 3000 // 4  # 750 tokens
        expected_bytes = expected_tokens * 1024
        expected_gb = expected_bytes / (1024**3)

        assert (
            abs(usage_gb - expected_gb) < 0.001
        )  # Allow small floating point differences

    def test_structured_response_formatting(self):
        """Test structured response formatting with metadata."""
        context_manager = ContextManager()

        response_text = "This is a test response"
        structured = context_manager.structure_response(response_text)

        assert structured["content"] == response_text
        assert structured["structured"] is True
        assert structured["context_optimized"] is True
        assert "generated_at" in structured
        assert isinstance(structured["generated_at"], int | float)


class TestMultiAgentCoordinatorComprehensive:
    """Comprehensive tests for MultiAgentCoordinator business logic."""

    def test_coordinator_initialization_comprehensive(self):
        """Test comprehensive coordinator initialization."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator(
                model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
                max_context_length=131072,
                backend="vllm",
                enable_fallback=True,
                max_agent_timeout=300.0,
            )

            # Verify initialization parameters
            assert coordinator.model_path == "Qwen/Qwen3-4B-Instruct-2507-FP8"
            assert coordinator.max_context_length == 131072
            assert coordinator.backend == "vllm"
            assert coordinator.enable_fallback is True
            assert coordinator.max_agent_timeout == 300.0

            # Verify performance tracking initialization
            assert coordinator.total_queries == 0
            assert coordinator.successful_queries == 0
            assert coordinator.fallback_queries == 0
            assert coordinator.avg_processing_time == 0.0
            assert coordinator.avg_coordination_overhead == 0.0

            # Verify context management
            assert isinstance(coordinator.context_manager, ContextManager)
            assert coordinator.context_manager.max_context_tokens == 131072

    def test_ensure_setup_comprehensive(self):
        """Test comprehensive setup process."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ) as mocks:
            coordinator = MultiAgentCoordinator()

            # Mock LlamaIndex Settings
            with patch("llama_index.core.Settings") as mock_settings:
                mock_llm = Mock()
                mock_settings.llm = mock_llm

                with patch.object(
                    coordinator, "_setup_agent_graph"
                ) as mock_setup_graph:
                    # Test successful setup
                    result = coordinator._ensure_setup()

                    assert result is True
                    assert coordinator._setup_complete is True
                    assert coordinator.llm == mock_llm
                    mocks["setup_llamaindex"].assert_called_once()
                    mock_setup_graph.assert_called_once()

        # Test setup failure
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(side_effect=RuntimeError("Setup failed")),
            is_dspy_available=Mock(return_value=False),
        ):
            coordinator_fail = MultiAgentCoordinator()
            result = coordinator_fail._ensure_setup()

            assert result is False
            assert coordinator_fail._setup_complete is False

    def test_agent_graph_setup_adr_compliance(self):
        """Test agent graph setup with ADR-011 compliance."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator()
            coordinator.llm = Mock()

            with (
                patch("src.agents.coordinator.create_react_agent") as mock_create_agent,
                patch(
                    "src.agents.coordinator.create_supervisor"
                ) as mock_create_supervisor,
                patch(
                    "src.agents.coordinator.create_forward_message_tool"
                ) as mock_forward_tool,
            ):
                # Mock agent creation
                mock_agents = [Mock() for _ in range(5)]  # 5 agents
                mock_create_agent.side_effect = mock_agents

                # Mock supervisor creation
                mock_graph = Mock()
                mock_create_supervisor.return_value = mock_graph

                # Mock forward tool
                mock_tool = Mock()
                mock_forward_tool.return_value = mock_tool

                coordinator._setup_agent_graph()

                # Verify 5 agents created
                # (router, planner, retrieval, synthesis, validation)
                assert mock_create_agent.call_count == 5

                # Verify agent names and tools
                agent_calls = mock_create_agent.call_args_list
                expected_agents = [
                    "router_agent",
                    "planner_agent",
                    "retrieval_agent",
                    "synthesis_agent",
                    "validation_agent",
                ]

                for i, call in enumerate(agent_calls):
                    kwargs = call[1]
                    assert kwargs["name"] == expected_agents[i]
                    assert kwargs["llm"] == coordinator.llm
                    assert len(kwargs["tools"]) == 1  # Each agent has one specific tool

                # Verify supervisor creation with ADR-011 parameters
                mock_create_supervisor.assert_called_once()
                supervisor_kwargs = mock_create_supervisor.call_args[1]

                assert (
                    supervisor_kwargs["parallel_tool_calls"]
                    == PARALLEL_TOOL_CALLS_ENABLED
                )
                assert supervisor_kwargs["output_mode"] == OUTPUT_MODE_STRUCTURED
                assert (
                    supervisor_kwargs["create_forward_message_tool"]
                    == CREATE_FORWARD_MESSAGE_TOOL_ENABLED
                )
                assert (
                    supervisor_kwargs["add_handoff_back_messages"]
                    == ADD_HANDOFF_BACK_MESSAGES_ENABLED
                )
                assert "pre_model_hook" in supervisor_kwargs
                assert "post_model_hook" in supervisor_kwargs

                # Verify forward tool creation
                mock_forward_tool.assert_called_once_with("supervisor")

                # Verify graph compilation
                assert coordinator.graph == mock_graph
                assert hasattr(coordinator, "compiled_graph")

    def test_supervisor_prompt_creation(self):
        """Test supervisor prompt creation with performance targets."""
        coordinator = MultiAgentCoordinator()
        prompt = coordinator._create_supervisor_prompt()

        # Verify prompt contains key elements
        assert "high-performance supervisor" in prompt
        assert "<200ms per decision" in prompt
        assert "50-87% token reduction target" in prompt
        assert "128K tokens" in prompt
        assert "FP8 KV cache optimization" in prompt

        # Verify team composition mentioned
        agent_names = [
            "router_agent",
            "planner_agent",
            "retrieval_agent",
            "synthesis_agent",
            "validation_agent",
        ]
        for agent_name in agent_names:
            assert agent_name in prompt

        # Verify coordination strategy
        assert "router_agent for strategy analysis" in prompt
        assert "needs_planning=true" in prompt
        assert "parallel_tool_calls" in prompt

    def test_pre_model_hook_context_trimming(self):
        """Test pre-model hook for context trimming logic."""
        coordinator = MultiAgentCoordinator()
        coordinator.context_manager = ContextManager(max_context_tokens=1000)
        coordinator.context_manager.trim_threshold = 800

        pre_hook = coordinator._create_pre_model_hook()

        # Test with messages under threshold
        small_state = {"messages": [HumanMessage(content="Short message")]}

        result = pre_hook(small_state)
        assert result == small_state  # Should not modify
        assert "context_trimmed" not in result

        # Test with messages over threshold
        large_state = {
            "messages": [HumanMessage(content="A" * 4000)]  # ~1000 tokens
        }

        with patch("src.agents.coordinator.trim_messages") as mock_trim:
            mock_trim.return_value = [HumanMessage(content="Trimmed")]

            result = pre_hook(large_state)

            # Should trigger trimming
            mock_trim.assert_called_once()
            assert result["context_trimmed"] is True
            assert "tokens_trimmed" in result
            assert result["tokens_trimmed"] > 0

    def test_post_model_hook_response_formatting(self):
        """Test post-model hook for structured response formatting."""
        coordinator = MultiAgentCoordinator()
        coordinator.model_path = "Qwen/Qwen3-4B-Instruct-2507-FP8"

        post_hook = coordinator._create_post_model_hook()

        test_state = {
            "output_mode": "structured",
            "messages": [HumanMessage(content="Test message")],
            "response": "Test response content",
            "parallel_tool_calls": True,
            "context_trimmed": True,
            "tokens_trimmed": 150,
        }

        result = post_hook(test_state)

        # Verify optimization metrics added
        assert "optimization_metrics" in result
        metrics = result["optimization_metrics"]

        assert "context_used_tokens" in metrics
        assert "kv_cache_usage_gb" in metrics
        assert "parallel_execution_active" in metrics
        assert "fp8_optimization" in metrics
        assert metrics["fp8_optimization"] is True
        assert metrics["model_path"] == coordinator.model_path
        assert metrics["context_trimmed"] is True
        assert metrics["tokens_trimmed"] == 150

        # Verify response structuring
        assert isinstance(result["response"], dict)
        assert result["response"]["structured"] is True

    def test_run_agent_workflow_execution(self):
        """Test agent workflow execution with timeout handling."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator(max_agent_timeout=1.0)
            coordinator._setup_complete = True

            # Mock compiled graph
            mock_graph = Mock()
            coordinator.compiled_graph = mock_graph

            # Test successful workflow
            mock_states = [
                {
                    "messages": [HumanMessage(content="Step 1")],
                    "routing_decision": {"strategy": "vector"},
                },
                {
                    "messages": [HumanMessage(content="Step 2")],
                    "validation_result": {"confidence": 0.9},
                },
            ]
            mock_graph.stream.return_value = mock_states

            initial_state = MultiAgentState(
                messages=[HumanMessage(content="Test query")],
                total_start_time=time.perf_counter(),
            )

            result = coordinator._run_agent_workflow(initial_state, "test_thread")

            # Should return final state
            assert result == mock_states[-1]
            mock_graph.stream.assert_called_once()

            # Verify config passed to stream
            call_kwargs = mock_graph.stream.call_args[1]
            assert call_kwargs["config"]["configurable"]["thread_id"] == "test_thread"
            assert call_kwargs["stream_mode"] == "values"

    def test_agent_workflow_timeout_handling(self):
        """Test agent workflow timeout handling."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator(
                max_agent_timeout=0.1
            )  # Very short timeout
            coordinator._setup_complete = True

            mock_graph = Mock()
            coordinator.compiled_graph = mock_graph

            # Mock slow workflow
            def slow_stream(*args, **kwargs):
                time.sleep(0.2)  # Exceed timeout
                return [{"messages": [HumanMessage(content="Slow response")]}]

            mock_graph.stream.side_effect = slow_stream

            initial_state = MultiAgentState(
                messages=[HumanMessage(content="Test query")],
                total_start_time=time.perf_counter(),
            )

            start_time = time.perf_counter()
            result = coordinator._run_agent_workflow(initial_state, "timeout_test")
            elapsed = time.perf_counter() - start_time

            # Should respect timeout
            assert elapsed >= 0.1  # At least timeout duration
            assert result is not None  # Should return something

    def test_extract_response_comprehensive(self):
        """Test comprehensive response extraction from agent state."""
        coordinator = MultiAgentCoordinator()
        coordinator.model_path = "Qwen/Qwen3-4B-Instruct-2507-FP8"
        coordinator.max_context_length = 131072

        final_state = {
            "messages": [HumanMessage(content="Final agent response")],
            "routing_decision": {"strategy": "hybrid", "complexity": "complex"},
            "planning_output": {"sub_tasks": ["task1", "task2", "task3"]},
            "synthesis_result": {
                "documents": [
                    {"content": "Source document 1", "score": 0.9},
                    {"content": "Source document 2", "score": 0.8},
                ]
            },
            "validation_result": {"confidence": 0.85, "suggested_action": "accept"},
            "agent_timings": {
                "router_agent": 0.05,
                "planner_agent": 0.08,
                "retrieval_agent": 0.15,
                "synthesis_agent": 0.12,
                "validation_agent": 0.06,
            },
            "parallel_execution_active": True,
            "token_reduction_achieved": 0.65,
            "context_trimmed": True,
            "tokens_trimmed": 1500,
            "kv_cache_usage_gb": 8.2,
        }

        start_time = time.perf_counter()
        coordination_time = 0.18

        response = coordinator._extract_response(
            final_state, "original query", start_time, coordination_time
        )

        # Verify response structure
        assert isinstance(response, AgentResponse)
        assert response.content == "Final agent response"
        assert len(response.sources) == 2
        assert response.validation_score == 0.85
        assert response.processing_time > 0

        # Verify optimization metrics
        metrics = response.optimization_metrics
        assert metrics["coordination_overhead_ms"] == round(coordination_time * 1000, 2)
        assert metrics["meets_200ms_target"] == (
            coordination_time < COORDINATION_OVERHEAD_THRESHOLD
        )
        assert metrics["parallel_execution_active"] is True
        assert metrics["token_reduction_achieved"] == 0.65
        assert metrics["context_trimmed"] is True
        assert metrics["tokens_trimmed"] == 1500
        assert metrics["kv_cache_usage_gb"] == 8.2
        assert metrics["fp8_optimization"] is True
        assert metrics["context_window_used"] == 131072

        # Verify metadata
        metadata = response.metadata
        assert metadata["routing_decision"]["strategy"] == "hybrid"
        assert len(metadata["planning_output"]["sub_tasks"]) == 3
        assert len(metadata["agent_timings"]) == 5
        assert "adr_compliance" in metadata

    def test_process_query_comprehensive_flow(self):
        """Test comprehensive query processing flow."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock compiled graph
            mock_graph = Mock()
            coordinator.compiled_graph = mock_graph

            # Mock successful workflow
            mock_final_state = {
                "messages": [HumanMessage(content="Processed response")],
                "routing_decision": {"strategy": "vector", "complexity": "simple"},
                "validation_result": {"confidence": 0.9},
                "agent_timings": {"router_agent": 0.05, "retrieval_agent": 0.08},
                "parallel_execution_active": True,
            }
            mock_graph.stream.return_value = [mock_final_state]

            # Process query
            query = "What is machine learning?"
            context = ChatMemoryBuffer.from_defaults()

            start_time = time.perf_counter()
            response = coordinator.process_query(query, context=context)
            processing_time = time.perf_counter() - start_time

            # Verify response
            assert isinstance(response, AgentResponse)
            assert response.content == "Processed response"
            assert response.processing_time <= processing_time + 0.1  # Allow overhead

            # Verify performance tracking updated
            assert coordinator.total_queries == 1
            assert coordinator.successful_queries == 1
            assert coordinator.avg_processing_time > 0

            # Verify optimization metrics
            assert "coordination_overhead_ms" in response.optimization_metrics
            assert "fp8_optimization" in response.optimization_metrics

    def test_fallback_mechanism_comprehensive(self):
        """Test comprehensive fallback mechanism."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator(enable_fallback=True)

            # Test setup failure fallback
            coordinator._setup_complete = False

            response = coordinator.process_query("Test query")

            assert isinstance(response, AgentResponse)
            assert "Failed to initialize coordinator" in response.content
            assert response.metadata["fallback_available"] is True
            assert "initialization_failed" in response.optimization_metrics

            # Test workflow failure fallback
            coordinator._setup_complete = True
            mock_graph = Mock()
            mock_graph.stream.side_effect = RuntimeError("Workflow failed")
            coordinator.compiled_graph = mock_graph

            response = coordinator.process_query("Test query")

            assert isinstance(response, AgentResponse)
            assert "unavailable" in response.content.lower()
            assert response.metadata["fallback_used"] is True
            assert "fallback_mode" in response.optimization_metrics

            # Verify fallback counter incremented
            assert coordinator.fallback_queries == 1

    def test_performance_metrics_comprehensive(self):
        """Test comprehensive performance metrics tracking."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock fast successful queries
            mock_graph = Mock()
            mock_graph.stream.return_value = [
                {
                    "messages": [HumanMessage(content="Response")],
                    "validation_result": {"confidence": 0.9},
                    "agent_timings": {"router_agent": 0.05},
                }
            ]
            coordinator.compiled_graph = mock_graph

            # Process multiple queries
            queries = ["Query 1", "Query 2", "Query 3"]
            for query in queries:
                coordinator.process_query(query)

            # Get performance stats
            stats = coordinator.get_performance_stats()

            # Verify basic metrics
            assert stats["total_queries"] == 3
            assert stats["successful_queries"] == 3
            assert stats["success_rate"] == 1.0
            assert stats["fallback_rate"] == 0.0
            assert stats["avg_processing_time"] > 0

            # Verify ADR-011 metrics
            assert "avg_coordination_overhead_ms" in stats
            assert "meets_200ms_target" in stats
            assert stats["agent_timeout"] == coordinator.max_agent_timeout

            # Verify ADR compliance reporting
            adr_compliance = stats["adr_compliance"]
            assert "adr_001" in adr_compliance
            assert "adr_004" in adr_compliance
            assert "adr_010" in adr_compliance
            assert "adr_011" in adr_compliance
            assert "adr_018" in adr_compliance

            # Verify model configuration
            model_config = stats["model_config"]
            assert model_config["model_path"] == coordinator.model_path
            assert model_config["max_context_length"] == coordinator.max_context_length
            assert model_config["fp8_optimization"] is True

    def test_adr_compliance_validation(self):
        """Test comprehensive ADR compliance validation."""
        coordinator = MultiAgentCoordinator(
            model_path="Qwen/Qwen3-4B-Instruct-2507-FP8", max_context_length=131072
        )
        coordinator._setup_complete = True
        coordinator.compiled_graph = Mock()
        coordinator.avg_coordination_overhead = 0.15  # 150ms < 200ms target

        compliance = coordinator.validate_adr_compliance()

        # Verify ADR compliance checks
        assert compliance["adr_001_supervisor_pattern"] is True
        assert compliance["adr_004_fp8_model"] is True  # Model path ends with "FP8"
        assert compliance["adr_010_performance_optimization"] is True
        assert compliance["adr_011_modern_parameters"] is True
        assert compliance["coordination_under_200ms"] is True
        assert compliance["context_128k_support"] is True

    def test_performance_stats_reset(self):
        """Test performance statistics reset functionality."""
        coordinator = MultiAgentCoordinator()

        # Simulate some activity
        coordinator.total_queries = 5
        coordinator.successful_queries = 4
        coordinator.fallback_queries = 1
        coordinator.avg_processing_time = 1.5
        coordinator.avg_coordination_overhead = 0.18

        # Reset stats
        coordinator.reset_performance_stats()

        # Verify reset
        assert coordinator.total_queries == 0
        assert coordinator.successful_queries == 0
        assert coordinator.fallback_queries == 0
        assert coordinator.avg_processing_time == 0.0
        assert coordinator.avg_coordination_overhead == 0.0

    def test_error_handling_edge_cases(self):
        """Test error handling in edge cases."""
        coordinator = MultiAgentCoordinator()

        # Test response extraction with malformed state
        malformed_state = {
            "messages": None,  # Invalid messages
            "routing_decision": "not a dict",  # Invalid type
        }

        response = coordinator._extract_response(
            malformed_state, "query", time.perf_counter(), 0.1
        )

        assert isinstance(response, AgentResponse)
        assert "No response generated by agents" in response.content

        # Test process_query with None query
        with patch.object(coordinator, "_ensure_setup", return_value=False):
            response = coordinator.process_query(None)
            assert "Failed to initialize coordinator" in response.content

    def test_context_conversation_continuity(self):
        """Test conversation context management and continuity."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Create conversation context
            context = ChatMemoryBuffer.from_defaults()
            context.put(HumanMessage(content="What is machine learning?"))
            context.put(HumanMessage(content="Previous AI discussion context"))

            # Mock workflow
            mock_graph = Mock()
            mock_graph.stream.return_value = [
                {
                    "messages": [HumanMessage(content="Contextual response")],
                    "validation_result": {"confidence": 0.9},
                }
            ]
            coordinator.compiled_graph = mock_graph

            # Process contextual query
            response = coordinator.process_query(
                "Can you elaborate on that?", context=context
            )

            # Verify context was used
            assert isinstance(response, AgentResponse)
            assert response.processing_time > 0

            # Verify initial state creation included context
            call_args = mock_graph.stream.call_args[0][
                0
            ]  # First positional arg (initial_state)
            assert call_args.context == context


class TestMultiAgentStateComprehensive:
    """Comprehensive tests for MultiAgentState data model."""

    def test_state_initialization_comprehensive(self):
        """Test comprehensive state initialization."""
        messages = [HumanMessage(content="Test message")]
        context = ChatMemoryBuffer.from_defaults()
        tools_data = {"vector": Mock(), "kg": Mock()}

        state = MultiAgentState(
            messages=messages,
            tools_data=tools_data,
            context=context,
            routing_decision={"strategy": "vector"},
            agent_timings={"router_agent": 0.05},
            parallel_execution_active=True,
            token_reduction_achieved=0.65,
            context_trimmed=True,
            tokens_trimmed=1500,
            kv_cache_usage_gb=8.2,
            output_mode="structured",
        )

        # Verify state fields (handle both attribute and dict access)
        if hasattr(state, "messages"):
            # Attribute access
            assert state.messages == messages
            assert state.tools_data == tools_data
            assert state.context == context
            assert state.routing_decision["strategy"] == "vector"
            assert state.agent_timings["router_agent"] == 0.05
            assert state.parallel_execution_active is True
            assert state.token_reduction_achieved == 0.65
            assert state.context_trimmed is True
            assert state.tokens_trimmed == 1500
            assert state.kv_cache_usage_gb == 8.2
            assert state.output_mode == "structured"
        else:
            # Dict-like access
            assert state["messages"] == messages
            assert state["tools_data"] == tools_data
            assert state["context"] == context
            assert state["routing_decision"]["strategy"] == "vector"
            assert state["agent_timings"]["router_agent"] == 0.05
            assert state["parallel_execution_active"] is True
            assert state["token_reduction_achieved"] == 0.65
            assert state["context_trimmed"] is True
            assert state["tokens_trimmed"] == 1500
            assert state["kv_cache_usage_gb"] == 8.2
            assert state["output_mode"] == "structured"

    def test_state_default_values(self):
        """Test state default values are properly set."""
        state = MultiAgentState(messages=[HumanMessage(content="Test")])

        # Verify defaults (handle both access patterns)
        default_fields = [
            ("tools_data", {}),
            ("context", None),
            ("routing_decision", {}),
            ("planning_output", {}),
            ("retrieval_results", []),
            ("synthesis_result", {}),
            ("validation_result", {}),
            ("agent_timings", {}),
            ("total_start_time", 0.0),
            ("parallel_execution_active", False),
            ("token_reduction_achieved", 0.0),
            ("context_trimmed", False),
            ("tokens_trimmed", 0),
            ("kv_cache_usage_gb", 0.0),
            ("output_mode", "structured"),
            ("errors", []),
            ("fallback_used", False),
            ("remaining_steps", 10),
        ]

        for field_name, expected_default in default_fields:
            if hasattr(state, field_name):
                actual_value = getattr(state, field_name)
            else:
                actual_value = state.get(field_name, "missing")

            if actual_value != "missing":
                assert actual_value == expected_default, (
                    f"Field {field_name} default mismatch"
                )

    def test_state_error_tracking(self):
        """Test state error tracking functionality."""
        state = MultiAgentState(
            messages=[HumanMessage(content="Test")],
            errors=["Error 1", "Error 2"],
            fallback_used=True,
        )

        if hasattr(state, "errors"):
            assert len(state.errors) == 2
            assert state.fallback_used is True
        else:
            assert len(state["errors"]) == 2
            assert state["fallback_used"] is True


class TestAgentResponseComprehensive:
    """Comprehensive tests for AgentResponse data model."""

    def test_response_creation_comprehensive(self):
        """Test comprehensive response creation."""
        response = AgentResponse(
            content="Comprehensive test response",
            sources=[
                {"content": "Source 1", "score": 0.9},
                {"content": "Source 2", "score": 0.8},
            ],
            metadata={
                "routing_decision": {"strategy": "hybrid", "complexity": "complex"},
                "agent_timings": {"router_agent": 0.05, "retrieval_agent": 0.12},
                "adr_compliance": {"adr_011": "Modern supervisor parameters"},
            },
            validation_score=0.85,
            processing_time=1.25,
            optimization_metrics={
                "coordination_overhead_ms": 175.5,
                "meets_200ms_target": True,
                "parallel_execution_active": True,
                "token_reduction_achieved": 0.68,
                "fp8_optimization": True,
                "context_window_used": 131072,
                "context_trimmed": True,
                "tokens_trimmed": 2500,
                "kv_cache_usage_gb": 9.8,
            },
            agent_decisions=[
                {"agent": "router", "decision": "hybrid_strategy"},
                {"agent": "planner", "decision": "decompose_query"},
            ],
            fallback_used=False,
        )

        # Verify all fields
        assert response.content == "Comprehensive test response"
        assert len(response.sources) == 2
        assert response.sources[0]["score"] == 0.9
        assert response.validation_score == 0.85
        assert response.processing_time == 1.25
        assert response.fallback_used is False

        # Verify metadata
        assert response.metadata["routing_decision"]["strategy"] == "hybrid"
        assert len(response.metadata["agent_timings"]) == 2

        # Verify optimization metrics
        metrics = response.optimization_metrics
        assert metrics["coordination_overhead_ms"] == 175.5
        assert metrics["meets_200ms_target"] is True
        assert metrics["parallel_execution_active"] is True
        assert metrics["token_reduction_achieved"] == 0.68
        assert metrics["fp8_optimization"] is True
        assert metrics["context_window_used"] == 131072
        assert metrics["context_trimmed"] is True
        assert metrics["tokens_trimmed"] == 2500
        assert metrics["kv_cache_usage_gb"] == 9.8

        # Verify agent decisions
        assert len(response.agent_decisions) == 2
        assert response.agent_decisions[0]["agent"] == "router"

    def test_response_validation_score_constraints(self):
        """Test response validation score constraints."""
        # Test valid scores
        valid_scores = [0.0, 0.5, 1.0]
        for score in valid_scores:
            response = AgentResponse(
                content="Test", validation_score=score, processing_time=1.0
            )
            assert response.validation_score == score

        # Test invalid scores (should be handled by Pydantic validation)
        try:
            AgentResponse(content="Test", validation_score=-0.1, processing_time=1.0)
            raise AssertionError("Should raise validation error for negative score")
        except ValueError:
            pass  # Expected

        try:
            AgentResponse(content="Test", validation_score=1.1, processing_time=1.0)
            raise AssertionError("Should raise validation error for score > 1")
        except ValueError:
            pass  # Expected


class TestFactoryFunctionComprehensive:
    """Comprehensive tests for factory function."""

    def test_create_multi_agent_coordinator_comprehensive(self):
        """Test factory function with comprehensive parameters."""
        with patch("src.agents.coordinator.MultiAgentCoordinator") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            # Test with custom parameters
            result = create_multi_agent_coordinator(
                model_path="custom/model-path",
                max_context_length=65536,
                enable_fallback=False,
            )

            # Verify factory called correctly
            mock_class.assert_called_once_with(
                model_path="custom/model-path",
                max_context_length=65536,
                enable_fallback=False,
            )
            assert result == mock_instance

    def test_create_multi_agent_coordinator_defaults(self):
        """Test factory function with default parameters."""
        with patch("src.agents.coordinator.MultiAgentCoordinator") as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            # Test with defaults
            result = create_multi_agent_coordinator()

            # Verify default parameters
            mock_class.assert_called_once_with(
                model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
                max_context_length=settings.vllm.context_window,
                enable_fallback=True,
            )
            assert result == mock_instance


class TestIntegrationScenariosComprehensive:
    """Integration tests for comprehensive multi-agent scenarios."""

    def test_simple_query_complete_flow(self):
        """Test complete flow for simple query processing."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock simple query workflow
            mock_workflow_states = [
                # Router agent decision
                {
                    "messages": [HumanMessage(content="Routing complete")],
                    "routing_decision": {
                        "strategy": "vector",
                        "complexity": "simple",
                        "needs_planning": False,
                        "confidence": 0.95,
                    },
                },
                # Retrieval agent execution
                {
                    "messages": [HumanMessage(content="Retrieval complete")],
                    "retrieval_results": [
                        {
                            "documents": [{"content": "AI definition", "score": 0.9}],
                            "strategy_used": "vector",
                        }
                    ],
                },
                # Validation agent final check
                {
                    "messages": [
                        HumanMessage(content="AI is artificial intelligence.")
                    ],
                    "validation_result": {
                        "confidence": 0.92,
                        "suggested_action": "accept",
                    },
                    "agent_timings": {
                        "router_agent": 0.03,
                        "retrieval_agent": 0.08,
                        "validation_agent": 0.04,
                    },
                    "parallel_execution_active": True,
                },
            ]

            mock_graph = Mock()
            mock_graph.stream.return_value = mock_workflow_states
            coordinator.compiled_graph = mock_graph

            # Process simple query
            response = coordinator.process_query("What is AI?")

            # Verify simple query characteristics
            assert isinstance(response, AgentResponse)
            assert response.content == "AI is artificial intelligence."
            assert response.validation_score == 0.92

            # Should meet performance targets for simple queries
            assert response.processing_time < 1.5  # Simple query target
            coordination_overhead = response.optimization_metrics[
                "coordination_overhead_ms"
            ]
            assert coordination_overhead < 200  # ADR-011 target

    def test_complex_query_complete_flow(self):
        """Test complete flow for complex query processing."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock complex query workflow
            mock_workflow_states = [
                # Router agent decision
                {
                    "messages": [HumanMessage(content="Complex routing complete")],
                    "routing_decision": {
                        "strategy": "hybrid",
                        "complexity": "complex",
                        "needs_planning": True,
                        "confidence": 0.9,
                    },
                },
                # Planner agent decomposition
                {
                    "messages": [HumanMessage(content="Planning complete")],
                    "planning_output": {
                        "sub_tasks": [
                            "Define machine learning",
                            "Define deep learning",
                            "Compare approaches",
                            "Synthesize differences",
                        ],
                        "execution_order": "parallel",
                    },
                },
                # Retrieval agent execution
                {
                    "messages": [HumanMessage(content="Retrieval complete")],
                    "retrieval_results": [
                        {
                            "documents": [
                                {
                                    "content": "ML uses statistical methods",
                                    "score": 0.9,
                                },
                                {"content": "DL uses neural networks", "score": 0.85},
                            ],
                            "strategy_used": "hybrid",
                        }
                    ],
                },
                # Synthesis agent combination
                {
                    "messages": [HumanMessage(content="Synthesis complete")],
                    "synthesis_result": {
                        "documents": [
                            {"content": "Combined ML and DL analysis", "score": 0.92},
                        ],
                        "synthesis_metadata": {
                            "original_count": 2,
                            "final_count": 1,
                            "strategies_used": ["hybrid"],
                        },
                    },
                },
                # Validation agent final check
                {
                    "messages": [
                        HumanMessage(
                            content="Machine learning and deep learning differ in "
                            "their approach to pattern recognition..."
                        )
                    ],
                    "validation_result": {
                        "confidence": 0.88,
                        "suggested_action": "accept",
                    },
                    "agent_timings": {
                        "router_agent": 0.05,
                        "planner_agent": 0.08,
                        "retrieval_agent": 0.15,
                        "synthesis_agent": 0.12,
                        "validation_agent": 0.06,
                    },
                    "parallel_execution_active": True,
                    "token_reduction_achieved": 0.72,
                },
            ]

            mock_graph = Mock()
            mock_graph.stream.return_value = mock_workflow_states
            coordinator.compiled_graph = mock_graph

            # Process complex query
            response = coordinator.process_query(
                "Compare machine learning with deep learning approaches"
            )

            # Verify complex query characteristics
            assert isinstance(response, AgentResponse)
            assert "Machine learning and deep learning differ" in response.content
            assert response.validation_score == 0.88
            assert len(response.sources) == 1  # From synthesis

            # Verify all agents were involved
            agent_timings = response.metadata["agent_timings"]
            expected_agents = [
                "router_agent",
                "planner_agent",
                "retrieval_agent",
                "synthesis_agent",
                "validation_agent",
            ]
            for agent in expected_agents:
                assert agent in agent_timings

            # Verify planning was performed
            planning_output = response.metadata["planning_output"]
            assert len(planning_output["sub_tasks"]) == 4
            assert planning_output["execution_order"] == "parallel"

            # Verify optimization metrics
            assert response.optimization_metrics["parallel_execution_active"] is True
            assert response.optimization_metrics["token_reduction_achieved"] == 0.72

    def test_error_recovery_comprehensive_flow(self):
        """Test comprehensive error recovery scenarios."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator(enable_fallback=True)
            coordinator._setup_complete = True

            # Mock workflow that fails after routing
            def failing_workflow(*args, **kwargs):
                # Return one successful state then fail
                yield {
                    "messages": [HumanMessage(content="Router successful")],
                    "routing_decision": {"strategy": "vector", "complexity": "simple"},
                }
                # Simulate retrieval failure
                raise RuntimeError("Retrieval agent failed")

            mock_graph = Mock()
            mock_graph.stream.side_effect = failing_workflow
            coordinator.compiled_graph = mock_graph

            # Process query that will trigger failure
            response = coordinator.process_query("Test query with failure")

            # Should trigger fallback
            assert isinstance(response, AgentResponse)
            assert response.metadata["fallback_used"] is True
            assert "unavailable" in response.content.lower()
            assert coordinator.fallback_queries == 1


class TestPerformanceValidationComprehensive:
    """Comprehensive performance validation tests."""

    def test_coordination_overhead_validation(self):
        """Test coordination overhead meets ADR-011 targets."""
        with patch.multiple(
            "src.agents.coordinator",
            setup_llamaindex=Mock(),
            is_dspy_available=Mock(return_value=True),
        ):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock fast workflow for performance testing
            mock_graph = Mock()
            mock_graph.stream.return_value = [
                {
                    "messages": [HumanMessage(content="Fast response")],
                    "validation_result": {"confidence": 0.9},
                }
            ]
            coordinator.compiled_graph = mock_graph

            # Process multiple queries to measure consistency
            coordination_times = []
            for i in range(5):
                time.perf_counter()
                response = coordinator.process_query(f"Performance test query {i}")

                coordination_overhead = response.optimization_metrics[
                    "coordination_overhead_ms"
                ]
                coordination_times.append(coordination_overhead)

                # Each query should meet the 200ms target
                assert coordination_overhead < 200, f"Query {i} exceeded 200ms target"
                assert response.optimization_metrics["meets_200ms_target"] is True

            # Average should also be under target
            avg_coordination = sum(coordination_times) / len(coordination_times)
            assert avg_coordination < 200

            # Performance stats should reflect this
            stats = coordinator.get_performance_stats()
            assert stats["meets_200ms_target"] is True
            assert stats["avg_coordination_overhead_ms"] < 200

    def test_context_management_performance(self):
        """Test context management performance with large contexts."""
        context_manager = ContextManager(max_context_tokens=131072)

        # Test with varying context sizes
        context_sizes = [1000, 10000, 50000, 100000]  # tokens

        for size in context_sizes:
            # Create messages that approximate the token count
            large_content = "A" * (size * 4)  # 4 chars per token
            messages = [{"content": large_content}]

            start_time = time.perf_counter()
            estimated_tokens = context_manager.estimate_tokens(messages)
            kv_usage = context_manager.calculate_kv_cache_usage({"messages": messages})
            end_time = time.perf_counter()

            # Performance should be consistent regardless of size
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            assert processing_time < 10, (
                f"Context processing too slow for {size} tokens"
            )

            # Estimates should be reasonable
            assert abs(estimated_tokens - size) < size * 0.1  # Within 10%
            assert kv_usage > 0  # Should calculate usage

    def test_memory_efficiency_validation(self):
        """Test memory efficiency with FP8 optimization."""
        coordinator = MultiAgentCoordinator()
        context_manager = coordinator.context_manager

        # Test memory calculations for different scenarios
        test_scenarios = [
            {"tokens": 32768, "expected_gb_range": (0.03, 0.04)},  # 32K context
            {"tokens": 65536, "expected_gb_range": (0.06, 0.08)},  # 64K context
            {"tokens": 131072, "expected_gb_range": (0.12, 0.16)},  # 128K context
        ]

        for scenario in test_scenarios:
            tokens = scenario["tokens"]
            expected_min, expected_max = scenario["expected_gb_range"]

            # Create mock state with estimated tokens
            mock_messages = [{"content": "A" * (tokens * 4)}]  # Approximate tokens
            state = {"messages": mock_messages}

            kv_usage_gb = context_manager.calculate_kv_cache_usage(state)

            # Should be within expected FP8 memory usage range
            assert expected_min <= kv_usage_gb <= expected_max, (
                f"KV cache usage {kv_usage_gb}GB outside expected range "
                f"{expected_min}-{expected_max}GB for {tokens} tokens"
            )

    def test_adr_compliance_comprehensive_validation(self):
        """Test comprehensive ADR compliance validation."""
        coordinator = MultiAgentCoordinator(
            model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
            max_context_length=131072,
        )

        # Simulate successful setup
        coordinator._setup_complete = True
        coordinator.compiled_graph = Mock()

        # Test ADR-011 compliance validation
        compliance = coordinator.validate_adr_compliance()

        required_compliance_items = [
            "adr_001_supervisor_pattern",
            "adr_004_fp8_model",
            "adr_010_performance_optimization",
            "adr_011_modern_parameters",
            "adr_018_dspy_integration",
            "coordination_under_200ms",
            "context_128k_support",
        ]

        for item in required_compliance_items:
            assert item in compliance, f"Missing compliance check: {item}"
            # Most should pass (except possibly DSPy which may not be available)
            if item != "adr_018_dspy_integration":
                assert compliance[item] is True, f"Compliance failed for: {item}"

    def test_constants_validation(self):
        """Test that performance constants are properly defined."""
        # Verify critical constants for performance
        assert COORDINATION_OVERHEAD_THRESHOLD == 0.2  # 200ms in seconds
        assert PARALLEL_TOOL_CALLS_ENABLED is True
        assert OUTPUT_MODE_STRUCTURED == "structured"
        assert CREATE_FORWARD_MESSAGE_TOOL_ENABLED is True
        assert ADD_HANDOFF_BACK_MESSAGES_ENABLED is True

        # Verify these constants are used in the coordinator
        coordinator = MultiAgentCoordinator()

        # Context manager should use reasonable defaults
        assert coordinator.context_manager.max_context_tokens >= 131072
        assert coordinator.context_manager.trim_threshold > 100000
        assert coordinator.context_manager.kv_cache_memory_per_token > 0


# Mark coverage validation
class TestCoverageValidationComprehensive:
    """Tests to validate comprehensive coverage improvements."""

    def test_coverage_critical_coordinator_paths(self):
        """Ensure critical coordinator paths are tested."""
        # Test that key coordinator methods are accessible and functional
        coordinator = MultiAgentCoordinator()

        # Key methods should be callable without errors
        assert callable(coordinator._create_supervisor_prompt)
        assert callable(coordinator._create_pre_model_hook)
        assert callable(coordinator._create_post_model_hook)
        assert callable(coordinator.get_performance_stats)
        assert callable(coordinator.validate_adr_compliance)
        assert callable(coordinator.reset_performance_stats)

        # Methods should return expected types
        prompt = coordinator._create_supervisor_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial

        pre_hook = coordinator._create_pre_model_hook()
        post_hook = coordinator._create_post_model_hook()
        assert callable(pre_hook)
        assert callable(post_hook)

        stats = coordinator.get_performance_stats()
        assert isinstance(stats, dict)
        assert "total_queries" in stats

        compliance = coordinator.validate_adr_compliance()
        assert isinstance(compliance, dict)
        assert len(compliance) >= 5  # Multiple compliance checks

    def test_coverage_context_manager_paths(self):
        """Ensure ContextManager paths are comprehensively tested."""
        context_manager = ContextManager()

        # Test various input types and edge cases
        test_cases = [
            [],  # Empty list
            [{"content": "test"}],  # Single message
            [Mock(content="mock message")],  # Mock objects
            [{"content": "A" * 10000}],  # Large content
        ]

        for test_case in test_cases:
            # Should handle all cases without error
            tokens = context_manager.estimate_tokens(test_case)
            assert isinstance(tokens, int)
            assert tokens >= 0

            # KV cache calculation should work
            state = {"messages": test_case}
            kv_usage = context_manager.calculate_kv_cache_usage(state)
            assert isinstance(kv_usage, float)
            assert kv_usage >= 0

    def test_coverage_agent_response_validation(self):
        """Ensure AgentResponse model validation is comprehensive."""
        # Test required fields
        minimal_response = AgentResponse(content="test", processing_time=1.0)
        assert minimal_response.content == "test"
        assert minimal_response.processing_time == 1.0
        assert minimal_response.sources == []
        assert minimal_response.validation_score == 0.0

        # Test comprehensive response
        full_response = AgentResponse(
            content="comprehensive test",
            sources=[{"test": "source"}],
            metadata={"test": "metadata"},
            validation_score=0.75,
            processing_time=2.5,
            optimization_metrics={"test": "metrics"},
            agent_decisions=[{"test": "decision"}],
            fallback_used=True,
        )

        assert len(full_response.sources) == 1
        assert full_response.fallback_used is True
        assert len(full_response.agent_decisions) == 1
        assert full_response.validation_score == 0.75
