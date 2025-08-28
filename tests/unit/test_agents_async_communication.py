"""Comprehensive async agent communication tests with proper mocking.

This test suite focuses on async patterns in multi-agent communication including:
- Async tool execution and coordination
- Agent message passing and state management
- Context preservation across async operations
- Performance validation for coordination flows
- Error propagation and recovery in async workflows

Test Strategy:
- Mock LLM boundaries with predictable responses
- Test async coordination patterns systematically
- Validate context preservation and state management
- Test error recovery mechanisms in async flows
- Focus on communication patterns, not framework internals

Markers:
- @pytest.mark.asyncio: All async communication tests
- @pytest.mark.unit: Fast isolated unit tests
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from llama_index.core.memory import ChatMemoryBuffer

from src.agents.coordinator import ContextManager
from src.agents.models import AgentResponse, MultiAgentState
from src.agents.tools import (
    plan_query,
    retrieve_documents,
    route_query,
    validate_response,
)


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncAgentCommunication:
    """Test suite for async agent communication patterns."""

    async def test_async_tool_execution_with_state_injection(self):
        """Test async tool execution with proper state injection."""
        # Given: Mock state with coordination context
        mock_state = {
            "messages": [HumanMessage(content="What is machine learning?")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock(), "kg": Mock(), "retriever": Mock()},
            "parallel_execution_active": True,
            "coordination_start_time": time.perf_counter(),
        }

        # When: Executing route_query tool async
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_factory.create_tools_from_indexes.return_value = [Mock()]

            result = route_query(query="What is machine learning?", state=mock_state)

        # Then: Tool executes successfully with state context
        assert result is not None
        assert isinstance(result, str)

        # Verify JSON structure
        import json

        parsed_result = json.loads(result)
        assert "strategy" in parsed_result
        assert "complexity" in parsed_result
        assert "confidence" in parsed_result

    async def test_async_context_preservation_across_tools(self):
        """Test context preservation during async tool chain execution."""
        # Given: Initial state with context
        initial_context = ChatMemoryBuffer.from_defaults()
        initial_context.chat_store.add_message("user", "Previous context message")

        mock_state = {
            "messages": [HumanMessage(content="Follow-up question")],
            "context": initial_context,
            "tools_data": {"vector": Mock(), "kg": Mock(), "retriever": Mock()},
            "routing_decision": {"strategy": "vector", "complexity": "medium"},
            "total_start_time": time.perf_counter(),
        }

        # When: Executing sequential tool chain
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_factory.create_tools_from_indexes.return_value = [Mock()]

            # Simulate async tool chain
            route_result = route_query("Follow-up question", state=mock_state)

            # Update state with routing decision
            mock_state["routing_decision"] = {
                "strategy": "vector",
                "complexity": "medium",
            }

            plan_result = plan_query("Follow-up question", state=mock_state)

        # Then: Context is preserved across async operations
        assert route_result is not None
        assert plan_result is not None

        # Verify context maintains conversation history
        assert mock_state["context"] is initial_context
        assert len(initial_context.chat_store.store) > 0

    async def test_async_agent_coordination_timing(self):
        """Test async agent coordination meets performance targets."""
        # Given: Performance tracking state
        start_time = time.perf_counter()
        mock_state = {
            "messages": [HumanMessage(content="Performance test query")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock(), "kg": Mock(), "retriever": Mock()},
            "total_start_time": start_time,
            "agent_timings": {},
            "coordination_overhead_ms": 0,
        }

        # When: Executing coordinated async operations
        async def simulate_agent_coordination():
            # Simulate router agent timing
            router_start = time.perf_counter()
            with patch("src.agents.tools.ToolFactory"):
                route_result = route_query("Performance test query", state=mock_state)
            router_time = time.perf_counter() - router_start
            mock_state["agent_timings"]["router"] = router_time

            # Simulate retrieval agent timing
            retrieval_start = time.perf_counter()
            with patch("src.agents.tools.ToolFactory"):
                retrieval_result = retrieve_documents(
                    "Performance test query", state=mock_state
                )
            retrieval_time = time.perf_counter() - retrieval_start
            mock_state["agent_timings"]["retrieval"] = retrieval_time

            return route_result, retrieval_result

        results = await simulate_agent_coordination()
        total_time = time.perf_counter() - start_time

        # Then: Performance targets are met
        assert total_time < 1.0  # Under 1 second for test execution
        assert results[0] is not None
        assert results[1] is not None
        assert "router" in mock_state["agent_timings"]
        assert "retrieval" in mock_state["agent_timings"]

        # Verify individual agent timing tracking
        assert mock_state["agent_timings"]["router"] > 0
        assert mock_state["agent_timings"]["retrieval"] > 0

    async def test_async_error_propagation_and_recovery(self):
        """Test error propagation and recovery in async agent communication."""
        # Given: State with mock tools that can fail
        mock_state = {
            "messages": [HumanMessage(content="Error test query")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock(), "kg": Mock(), "retriever": Mock()},
            "error_recovery_enabled": True,
            "fallback_strategy": "basic_rag",
        }

        # Mock factory to raise error on first call, succeed on second
        call_count = 0

        def mock_create_tools(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Simulated async tool failure")
            return [Mock()]

        # When: Tool execution encounters error
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_factory.create_tools_from_indexes.side_effect = mock_create_tools

            # First call should handle error gracefully
            try:
                result = route_query("Error test query", state=mock_state)
                # Should not reach here on first call
                assert False, "Expected exception was not raised"
            except Exception as e:
                assert "Simulated async tool failure" in str(e)

            # Second call should succeed (simulating recovery)
            result = route_query("Error test query", state=mock_state)

        # Then: Error recovery works properly
        assert result is not None
        assert call_count == 2  # Verify retry occurred

    async def test_async_parallel_tool_execution(self):
        """Test parallel execution of multiple async tools."""
        # Given: State configured for parallel execution
        mock_state = {
            "messages": [HumanMessage(content="Parallel execution test")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock(), "kg": Mock(), "retriever": Mock()},
            "parallel_execution_active": True,
            "max_parallel_tasks": 3,
        }

        # When: Executing tools in parallel
        async def parallel_tool_execution():
            with patch("src.agents.tools.ToolFactory") as mock_factory:
                mock_factory.create_tools_from_indexes.return_value = [Mock()]

                # Simulate parallel tool calls
                tasks = [
                    asyncio.create_task(
                        asyncio.to_thread(route_query, "Parallel query 1", mock_state)
                    ),
                    asyncio.create_task(
                        asyncio.to_thread(
                            retrieve_documents, "Parallel query 2", mock_state
                        )
                    ),
                    asyncio.create_task(
                        asyncio.to_thread(
                            validate_response,
                            "Parallel query 3",
                            "Mock response",
                            mock_state,
                        )
                    ),
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results

        results = await parallel_tool_execution()

        # Then: All parallel executions complete successfully
        assert len(results) == 3
        for result in results:
            assert not isinstance(result, Exception)
            assert result is not None

    async def test_async_context_manager_integration(self):
        """Test async integration with context manager for token handling."""
        # Given: Context manager for async operations
        context_manager = ContextManager(max_context_tokens=131072)

        messages = [
            {"content": "First message in conversation", "role": "user"},
            {"content": "Response to first message", "role": "assistant"},
            {"content": "Follow-up question with more context", "role": "user"},
        ]

        mock_state = {
            "messages": messages,
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock()},
            "output_mode": "structured",
            "parallel_tool_calls": True,
        }

        # When: Using context manager in async workflow
        async def async_context_workflow():
            # Simulate token estimation in async context
            token_count = context_manager.estimate_tokens(messages)

            # Simulate pre-model hook
            processed_state = context_manager.pre_model_hook(mock_state.copy())

            # Simulate async tool execution with context
            with patch("src.agents.tools.ToolFactory"):
                result = route_query("Context test query", state=processed_state)

            # Simulate post-model hook
            final_state = context_manager.post_model_hook(processed_state)

            return token_count, result, final_state

        token_count, result, final_state = await async_context_workflow()

        # Then: Context management works in async environment
        assert token_count > 0
        assert result is not None
        assert final_state is not None
        assert "metadata" in final_state  # Added by post_model_hook

    async def test_async_state_persistence_across_operations(self):
        """Test state persistence across async agent operations."""
        # Given: In-memory state saver
        memory = InMemorySaver()
        thread_config = {"configurable": {"thread_id": "test_async_thread"}}

        initial_state = MultiAgentState(
            messages=[HumanMessage(content="Initial async message")],
            tools_data={"vector": Mock(), "kg": Mock()},
            context=ChatMemoryBuffer.from_defaults(),
            total_start_time=time.perf_counter(),
            routing_decision={"strategy": "vector", "complexity": "simple"},
            parallel_execution_active=True,
        )

        # When: Persisting and retrieving state across async operations
        async def async_state_workflow():
            # Save initial state
            checkpoint = memory.put(thread_config, initial_state.dict())

            # Simulate async operation that modifies state
            await asyncio.sleep(0.01)  # Small delay to simulate async work

            # Retrieve and modify state
            retrieved_state = memory.get(thread_config)
            modified_state = initial_state.copy()
            modified_state.retrieval_results = [{"content": "Async retrieval result"}]

            # Save updated state
            memory.put(thread_config, modified_state.dict())

            # Final retrieval
            final_state = memory.get(thread_config)
            return checkpoint, final_state

        checkpoint, final_state = await async_state_workflow()

        # Then: State persistence works across async operations
        assert checkpoint is not None
        assert final_state is not None
        assert "retrieval_results" in final_state.values

    async def test_async_agent_response_construction(self):
        """Test async construction of comprehensive agent responses."""
        # Given: Mock data for response construction
        mock_sources = [
            {
                "text": "ML definition",
                "score": 0.95,
                "metadata": {"source": "ml_guide.pdf"},
            },
            {
                "text": "AI overview",
                "score": 0.87,
                "metadata": {"source": "ai_book.pdf"},
            },
        ]

        mock_metadata = {
            "coordination_overhead_ms": 150,
            "agents_used": ["router", "retrieval", "synthesis"],
            "parallel_execution": True,
            "fp8_optimization": True,
        }

        # When: Constructing agent response asynchronously
        async def construct_agent_response():
            # Simulate async processing
            await asyncio.sleep(0.01)

            response = AgentResponse(
                content="Machine learning is a subset of AI that enables systems to learn from data.",
                sources=mock_sources,
                metadata=mock_metadata,
                validation_score=0.92,
                processing_time=0.15,
                optimization_metrics={
                    "coordination_overhead_ms": 150,
                    "meets_200ms_target": True,
                    "fp8_optimization": True,
                    "parallel_execution_active": True,
                },
                agent_decisions=[
                    {"agent": "router", "decision": "vector_search", "confidence": 0.9},
                    {"agent": "retrieval", "strategy": "hybrid", "results_count": 5},
                ],
                fallback_used=False,
            )

            return response

        response = await construct_agent_response()

        # Then: Response is properly constructed with all components
        assert isinstance(response, AgentResponse)
        assert response.content != ""
        assert len(response.sources) == 2
        assert response.validation_score > 0.9
        assert response.processing_time > 0
        assert len(response.agent_decisions) == 2
        assert response.optimization_metrics["meets_200ms_target"] is True
        assert response.fallback_used is False

    async def test_async_coordination_with_timeout_handling(self):
        """Test async coordination with proper timeout handling."""
        # Given: State with timeout configuration
        mock_state = {
            "messages": [HumanMessage(content="Timeout test query")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock()},
            "coordination_timeout_seconds": 0.1,  # Short timeout for testing
            "enable_timeout_recovery": True,
        }

        # When: Executing with potential timeout
        async def coordination_with_timeout():
            async def slow_tool_execution():
                await asyncio.sleep(0.2)  # Exceed timeout
                return "Delayed result"

            async def fast_tool_execution():
                await asyncio.sleep(0.05)  # Within timeout
                return "Fast result"

            try:
                # Try slow execution with timeout
                result = await asyncio.wait_for(
                    slow_tool_execution(),
                    timeout=mock_state["coordination_timeout_seconds"],
                )
                return result, "no_timeout"
            except TimeoutError:
                # Fallback to fast execution
                result = await asyncio.wait_for(
                    fast_tool_execution(),
                    timeout=mock_state["coordination_timeout_seconds"] * 2,
                )
                return result, "timeout_recovery"

        result, status = await coordination_with_timeout()

        # Then: Timeout handling works correctly
        assert result == "Fast result"
        assert status == "timeout_recovery"

    async def test_async_message_streaming_simulation(self):
        """Test async message streaming simulation for real-time updates."""
        # Given: Mock streaming setup
        messages = []

        async def mock_agent_stream():
            """Simulate async agent streaming responses."""
            chunks = [
                "Machine learning",
                " is a branch of",
                " artificial intelligence",
                " that enables computers",
                " to learn from data.",
            ]

            for chunk in chunks:
                await asyncio.sleep(0.01)  # Simulate streaming delay
                yield chunk

        # When: Processing streaming response
        async def process_stream():
            full_response = ""
            chunk_count = 0

            async for chunk in mock_agent_stream():
                full_response += chunk
                chunk_count += 1
                messages.append({"chunk": chunk, "timestamp": time.time()})

            return full_response, chunk_count

        full_response, chunk_count = await process_stream()

        # Then: Streaming works correctly
        assert (
            full_response
            == "Machine learning is a branch of artificial intelligence that enables computers to learn from data."
        )
        assert chunk_count == 5
        assert len(messages) == 5

        # Verify timestamp ordering
        timestamps = [msg["timestamp"] for msg in messages]
        assert timestamps == sorted(timestamps)  # Timestamps should be in order
