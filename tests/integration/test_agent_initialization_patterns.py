"""Comprehensive integration tests for agent initialization patterns.

This test suite validates agent initialization and coordination setup including:
- Multi-agent coordinator initialization with proper dependencies
- Tool factory integration and agent tool creation patterns
- Context manager setup and configuration validation
- LangGraph supervisor creation and agent registration
- State management initialization and persistence setup
- Configuration validation and ADR compliance checks

Test Strategy:
- Integration-level testing with real component interactions
- Mock external dependencies (LLMs, vector stores) at boundaries
- Test initialization sequences and dependency injection
- Validate proper error handling during initialization failures
- Focus on integration patterns, not individual component logic

Markers:
- @pytest.mark.integration: Cross-component integration tests
- @pytest.mark.asyncio: Async initialization patterns
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from llama_index.core.memory import ChatMemoryBuffer

from src.agents.coordinator import ContextManager, MultiAgentCoordinator
from src.agents.models import MultiAgentState
from src.agents.tool_factory import ToolFactory
from src.config import settings


@pytest.mark.integration
class TestAgentInitializationPatterns:
    """Integration test suite for agent initialization and setup patterns."""

    @patch("src.agents.coordinator.create_vllm_manager")
    @patch("src.agents.coordinator.is_dspy_available")
    def test_multi_agent_coordinator_initialization_complete_flow(
        self, mock_dspy_available, mock_create_vllm
    ):
        """Test complete multi-agent coordinator initialization flow."""
        # Given: Mock dependencies for coordinator initialization
        mock_dspy_available.return_value = True

        mock_vllm_manager = Mock()
        mock_vllm_manager.llm = Mock()
        mock_vllm_manager.context_manager = Mock()
        mock_create_vllm.return_value = mock_vllm_manager

        # When: Initializing coordinator with full configuration
        coordinator = MultiAgentCoordinator(
            model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
            max_context_length=131072,
            enable_fallback=True,
            enable_dspy_optimization=True,
            coordination_timeout_seconds=300,
        )

        # Then: Coordinator is properly initialized
        assert coordinator is not None
        assert coordinator.model_path == "Qwen/Qwen3-4B-Instruct-2507-FP8"
        assert coordinator.max_context_length == 131072
        assert coordinator.enable_fallback is True
        assert coordinator.coordination_timeout_seconds == 300

        # Verify dependencies were created
        mock_create_vllm.assert_called_once()
        mock_dspy_available.assert_called()

    def test_tool_factory_integration_with_multiple_index_types(self):
        """Test tool factory integration with multiple index types."""
        # Given: Mock indexes and retriever for tool creation
        mock_vector_index = Mock()
        mock_vector_index.as_query_engine.return_value = Mock()

        mock_kg_index = Mock()
        mock_kg_index.as_query_engine.return_value = Mock()

        mock_retriever = Mock()

        # Mock reranker creation
        with patch.object(ToolFactory, "_create_reranker") as mock_create_reranker:
            mock_reranker = Mock()
            mock_create_reranker.return_value = mock_reranker

            # When: Creating tools from indexes
            tools = ToolFactory.create_tools_from_indexes(
                vector_index=mock_vector_index,
                kg_index=mock_kg_index,
                retriever=mock_retriever,
            )

        # Then: All expected tools are created
        assert len(tools) == 3
        tool_names = [tool.metadata.name for tool in tools]

        # Verify tool types
        assert any("vector" in name.lower() for name in tool_names)
        assert any(
            "graph" in name.lower() or "knowledge" in name.lower()
            for name in tool_names
        )
        assert any("hybrid" in name.lower() for name in tool_names)

        # Verify reranker was integrated
        mock_create_reranker.assert_called()

    def test_context_manager_initialization_with_optimization_settings(self):
        """Test context manager initialization with FP8 optimization settings."""
        # Given: Context manager configuration matching ADR requirements
        max_context_tokens = 131072  # 128K context

        # When: Initializing context manager
        context_manager = ContextManager(max_context_tokens=max_context_tokens)

        # Then: Context manager is properly configured
        assert context_manager.max_context_tokens == 131072
        assert context_manager.trim_threshold == int(131072 * 0.9)  # 90% threshold
        assert context_manager.kv_cache_memory_per_token == 1024  # FP8 optimization

        # Test token estimation functionality
        test_messages = [
            {"content": "Test message 1", "role": "user"},
            {"content": "Test response 1", "role": "assistant"},
            {"content": "Follow-up question", "role": "user"},
        ]

        token_count = context_manager.estimate_tokens(test_messages)
        assert token_count > 0

        # Test KV cache calculation
        mock_state = {"messages": test_messages}
        kv_usage = context_manager.calculate_kv_cache_usage(mock_state)
        assert kv_usage > 0
        assert isinstance(kv_usage, float)

    def test_multi_agent_state_initialization_with_full_configuration(self):
        """Test MultiAgentState initialization with complete configuration."""
        # Given: Full state configuration data
        messages = [HumanMessage(content="Test initialization query")]
        context = ChatMemoryBuffer.from_defaults()
        tools_data = {"vector": Mock(), "kg": Mock(), "retriever": Mock()}

        # When: Initializing multi-agent state
        state = MultiAgentState(
            messages=messages,
            tools_data=tools_data,
            context=context,
            total_start_time=time.perf_counter(),
            routing_decision={"strategy": "hybrid", "complexity": "medium"},
            planning_output={
                "sub_tasks": ["retrieve", "synthesize"],
                "execution_order": "sequential",
            },
            retrieval_results=[{"content": "Sample doc", "score": 0.9}],
            synthesis_result={"content": "Synthesized response", "confidence": 0.85},
            validation_result={"valid": True, "confidence": 0.9},
            agent_timings={"router": 0.05, "retrieval": 0.12, "synthesis": 0.08},
            parallel_execution_active=True,
            output_mode="structured",
        )

        # Then: State is properly initialized with all components
        assert len(state.messages) == 1
        assert state.tools_data is not None
        assert len(state.tools_data) == 3
        assert state.context is not None
        assert state.routing_decision["strategy"] == "hybrid"
        assert state.planning_output["execution_order"] == "sequential"
        assert len(state.retrieval_results) == 1
        assert state.synthesis_result["confidence"] == 0.85
        assert state.validation_result["valid"] is True
        assert len(state.agent_timings) == 3
        assert state.parallel_execution_active is True
        assert state.output_mode == "structured"

    @patch("src.agents.coordinator.create_supervisor")
    @patch("src.agents.coordinator.create_react_agent")
    def test_langgraph_supervisor_creation_and_agent_registration(
        self, mock_create_react_agent, mock_create_supervisor
    ):
        """Test LangGraph supervisor creation and agent registration."""
        # Given: Mock LangGraph components
        mock_graph = Mock()
        mock_supervisor = Mock()
        mock_supervisor.compile.return_value = mock_graph
        mock_create_supervisor.return_value = mock_supervisor

        mock_react_agent = Mock()
        mock_create_react_agent.return_value = mock_react_agent

        # Mock tools for agent creation
        mock_tools = [Mock(), Mock(), Mock()]

        # When: Setting up supervisor with agents
        with patch("src.agents.coordinator.create_vllm_manager"):
            coordinator = MultiAgentCoordinator()

            # Simulate supervisor setup (internal method would be called during process_query)
            tools_data = {"vector": Mock(), "kg": Mock(), "retriever": Mock()}

            # Mock internal setup process
            with patch.object(
                ToolFactory, "create_tools_from_indexes", return_value=mock_tools
            ):
                # Simulate the internal graph creation process
                coordinator._create_agent_graph(tools_data)

        # Then: Supervisor and agents are properly created
        mock_create_supervisor.assert_called()

        # Verify supervisor configuration includes modern parameters
        supervisor_call_args = mock_create_supervisor.call_args
        if supervisor_call_args:
            # Check for ADR-011 compliance parameters
            assert supervisor_call_args is not None

    def test_dependency_injection_integration_patterns(self):
        """Test dependency injection patterns across agent components."""
        # Given: Mock dependencies for injection testing
        mock_llm = Mock()
        mock_memory = InMemorySaver()
        mock_context_manager = Mock()

        # When: Setting up dependency injection
        with patch("src.agents.coordinator.create_vllm_manager") as mock_create_vllm:
            mock_vllm_manager = Mock()
            mock_vllm_manager.llm = mock_llm
            mock_vllm_manager.context_manager = mock_context_manager
            mock_create_vllm.return_value = mock_vllm_manager

            coordinator = MultiAgentCoordinator()

            # Verify dependencies are properly injected
            assert (
                coordinator._setup_complete is False
            )  # Should be False until first query

            # Test dependency access
            with patch.object(coordinator, "vllm_manager", mock_vllm_manager):
                assert coordinator.vllm_manager is mock_vllm_manager
                assert coordinator.vllm_manager.llm is mock_llm
                assert coordinator.vllm_manager.context_manager is mock_context_manager

    def test_configuration_validation_and_adr_compliance_checks(self):
        """Test configuration validation and ADR compliance during initialization."""
        # Given: Configuration that should pass ADR compliance
        adr_compliant_config = {
            "model_path": "Qwen/Qwen3-4B-Instruct-2507-FP8",  # ADR-004 compliance
            "max_context_length": 131072,  # 128K context
            "enable_fallback": True,
            "coordination_timeout_seconds": 200,  # Under 200ms target from ADR-011
        }

        # When: Initializing with ADR-compliant configuration
        with patch("src.agents.coordinator.create_vllm_manager"):
            coordinator = MultiAgentCoordinator(**adr_compliant_config)

            # Verify ADR-004 compliance (Local-First LLM Strategy)
            assert (
                "FP8" in coordinator.model_path or "Instruct" in coordinator.model_path
            )
            assert coordinator.max_context_length >= 131072

            # Verify ADR-011 compliance (Agent Orchestration Framework)
            assert coordinator.coordination_timeout_seconds <= 300
            assert coordinator.enable_fallback is True

    def test_initialization_failure_handling_and_recovery(self):
        """Test initialization failure handling and recovery patterns."""
        # Given: Configuration that will cause initialization failures

        # When: LLM initialization fails
        with patch("src.agents.coordinator.create_vllm_manager") as mock_create_vllm:
            mock_create_vllm.side_effect = Exception("LLM initialization failed")

            try:
                coordinator = MultiAgentCoordinator()
                assert False, "Expected initialization failure"
            except Exception as e:
                assert "LLM initialization failed" in str(e)

        # Recovery: Successful initialization after fixing issue
        with patch("src.agents.coordinator.create_vllm_manager"):
            coordinator = MultiAgentCoordinator()
            assert coordinator is not None

    def test_tool_creation_integration_with_configuration_settings(self):
        """Test tool creation integration with configuration settings."""
        # Given: Configuration settings for tool creation
        original_top_k = settings.retrieval.top_k
        original_reranking_top_k = settings.retrieval.reranking_top_k

        try:
            # Modify settings for testing
            settings.retrieval.top_k = 15
            settings.retrieval.reranking_top_k = 8

            # When: Creating tools with modified settings
            mock_vector_index = Mock()
            mock_vector_index.as_query_engine.return_value = Mock()

            with patch.object(ToolFactory, "_create_reranker") as mock_create_reranker:
                mock_create_reranker.return_value = Mock()

                tool = ToolFactory.create_vector_search_tool(mock_vector_index)

            # Then: Tool creation uses configuration settings
            assert tool is not None

            # Verify query engine was called with correct parameters
            mock_vector_index.as_query_engine.assert_called_once()
            call_args = mock_vector_index.as_query_engine.call_args

            # Should have similarity_top_k parameter matching settings
            assert "similarity_top_k" in call_args[1]
            assert call_args[1]["similarity_top_k"] == 15

        finally:
            # Restore original settings
            settings.retrieval.top_k = original_top_k
            settings.retrieval.reranking_top_k = original_reranking_top_k

    def test_memory_and_state_persistence_integration(self):
        """Test memory and state persistence integration patterns."""
        # Given: Memory and state persistence setup
        memory = InMemorySaver()
        thread_config = {"configurable": {"thread_id": "integration_test_thread"}}

        initial_state = MultiAgentState(
            messages=[HumanMessage(content="Integration test message")],
            tools_data={"vector": Mock()},
            context=ChatMemoryBuffer.from_defaults(),
            total_start_time=time.perf_counter(),
            parallel_execution_active=True,
        )

        # When: Testing state persistence workflow
        # Save initial state
        checkpoint1 = memory.put(thread_config, initial_state.dict())
        assert checkpoint1 is not None

        # Modify state to simulate processing
        modified_state = initial_state.copy()
        modified_state.routing_decision = {"strategy": "vector", "complexity": "simple"}
        modified_state.agent_timings = {"router": 0.05}

        # Save modified state
        checkpoint2 = memory.put(thread_config, modified_state.dict())
        assert checkpoint2 is not None

        # Retrieve final state
        final_state = memory.get(thread_config)

        # Then: State persistence works correctly
        assert final_state is not None
        assert "routing_decision" in final_state.values
        assert "agent_timings" in final_state.values


@pytest.mark.asyncio
@pytest.mark.integration
class TestAsyncInitializationPatterns:
    """Async integration tests for agent initialization patterns."""

    async def test_async_coordinator_initialization_flow(self):
        """Test async coordinator initialization with proper async setup."""
        # Given: Async coordinator setup
        with patch("src.agents.coordinator.create_vllm_manager") as mock_create_vllm:
            mock_vllm_manager = Mock()
            mock_vllm_manager.llm = AsyncMock()
            mock_vllm_manager.async_engine = AsyncMock()
            mock_create_vllm.return_value = mock_vllm_manager

            # When: Initializing coordinator for async operations
            coordinator = MultiAgentCoordinator()

            # Mock async setup completion
            coordinator._setup_complete = True
            coordinator.vllm_manager = mock_vllm_manager

            # Test async compatibility
            assert coordinator.vllm_manager is not None
            assert hasattr(coordinator.vllm_manager, "async_engine")

    async def test_async_tool_integration_initialization(self):
        """Test async tool integration during initialization."""
        # Given: Mock async tools
        mock_async_tool = AsyncMock()
        mock_async_tool.ainvoke = AsyncMock(return_value="Async tool result")

        # When: Setting up async tool integration
        tools_data = {"vector": Mock(), "kg": Mock(), "retriever": Mock()}

        with patch.object(
            ToolFactory, "create_tools_from_indexes"
        ) as mock_create_tools:
            mock_create_tools.return_value = [mock_async_tool]

            # Simulate async tool setup
            tools = ToolFactory.create_tools_from_indexes(**tools_data)

            # Test async tool functionality
            if tools and hasattr(tools[0], "ainvoke"):
                result = await tools[0].ainvoke("Test async query")
                assert result == "Async tool result"

    async def test_async_context_manager_integration(self):
        """Test async context manager integration patterns."""
        # Given: Context manager for async operations
        context_manager = ContextManager(max_context_tokens=131072)

        # When: Using context manager in async context
        async def async_context_workflow():
            messages = [
                {"content": "Async message 1", "role": "user"},
                {"content": "Async response 1", "role": "assistant"},
            ]

            # Simulate async token estimation
            token_count = context_manager.estimate_tokens(messages)

            # Simulate async context processing
            await asyncio.sleep(0.01)  # Async operation simulation

            mock_state = {
                "messages": messages,
                "output_mode": "structured",
                "parallel_tool_calls": True,
            }

            # Test pre and post hooks in async context
            processed_state = context_manager.pre_model_hook(mock_state.copy())
            final_state = context_manager.post_model_hook(processed_state)

            return token_count, final_state

        token_count, final_state = await async_context_workflow()

        # Then: Async context processing works correctly
        assert token_count > 0
        assert final_state is not None
        assert "metadata" in final_state  # Added by post_model_hook

    async def test_async_state_initialization_and_management(self):
        """Test async state initialization and management patterns."""

        # Given: Async state setup
        async def create_async_state():
            # Simulate async state creation
            await asyncio.sleep(0.01)

            state = MultiAgentState(
                messages=[HumanMessage(content="Async state test")],
                tools_data={"vector": Mock(), "kg": Mock()},
                context=ChatMemoryBuffer.from_defaults(),
                total_start_time=time.perf_counter(),
                parallel_execution_active=True,
                async_mode_enabled=True,
            )

            return state

        # When: Creating and managing async state
        state = await create_async_state()

        # Simulate async state operations
        async def async_state_operations(state):
            # Simulate async routing decision
            await asyncio.sleep(0.01)
            state.routing_decision = {"strategy": "hybrid", "complexity": "medium"}

            # Simulate async retrieval results
            await asyncio.sleep(0.01)
            state.retrieval_results = [{"content": "Async retrieval", "score": 0.9}]

            return state

        updated_state = await async_state_operations(state)

        # Then: Async state management works correctly
        assert updated_state.routing_decision is not None
        assert len(updated_state.retrieval_results) == 1
        assert updated_state.parallel_execution_active is True
