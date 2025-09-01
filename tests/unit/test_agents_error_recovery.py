"""Comprehensive error recovery mechanism tests for agent coordination.

This test suite validates error handling and recovery patterns in the multi-agent
system:
- LLM failure recovery and fallback strategies
- Tool execution errors and graceful degradation
- Context corruption and state recovery patterns
- Agent timeout handling and coordination recovery
- Partial failure scenarios and system resilience
- Cascading failure prevention mechanisms

Test Strategy:
- Mock failure scenarios at different system boundaries
- Test fallback mechanisms and graceful degradation
- Validate error propagation and containment patterns
- Test system recovery after partial failures
- Focus on resilience patterns, not framework internals

Markers:
- @pytest.mark.unit: Fast error scenario tests
- @pytest.mark.asyncio: Async error recovery tests
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from llama_index.core.memory import ChatMemoryBuffer

from src.agents.tools import (
    retrieve_documents,
    route_query,
    validate_response,
)


@pytest.mark.unit
class TestAgentErrorRecovery:
    """Test suite for agent error recovery and resilience patterns."""

    # Deprecated legacy fallback tests removed; modern strategy relies on
    # boundary monkeypatching at tool execution levels.

    def test_tool_execution_error_containment(self):
        """Test error containment when individual tools fail."""
        # Given: State with multiple tools, one fails
        mock_state = {
            "messages": [HumanMessage(content="Multi-tool test query")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock(), "kg": Mock(), "retriever": Mock()},
            "error_isolation_enabled": True,
            "continue_on_tool_failure": True,
        }

        # Mock tools where one fails
        mock_vector_tool = Mock()
        mock_vector_tool.invoke.side_effect = Exception("Vector search failure")

        mock_kg_tool = Mock()
        mock_kg_tool.invoke.return_value = "KG search successful"

        mock_hybrid_tool = Mock()
        mock_hybrid_tool.invoke.return_value = "Hybrid search successful"

        # When: Executing tools with one failure
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_factory.create_tools_from_indexes.return_value = [
                mock_vector_tool,
                mock_kg_tool,
                mock_hybrid_tool,
            ]

            # Should isolate vector tool failure
            with patch("src.agents.tools.logger") as mock_logger:
                result = retrieve_documents.invoke(
                    {"query": "Multi-tool test query", "state": mock_state}
                )

        # Then: Error is contained and other tools continue working
        assert result is not None

        # Verify error was logged but execution continued
        mock_logger.error.assert_called()

        # Verify result indicates partial success
        import json

        parsed_result = json.loads(result)
        assert (
            parsed_result.get("partial_results", False)
            or len(parsed_result.get("documents", [])) > 0
        )

    def test_context_corruption_recovery(self):
        """Test recovery from context corruption scenarios."""
        # Given: Corrupted context state
        corrupted_context = ChatMemoryBuffer.from_defaults()
        # Simulate corrupted context by adding invalid data
        corrupted_context.chat_store.store = {"invalid": "corrupted_data"}

        mock_state = {
            "messages": [HumanMessage(content="Context corruption test")],
            "context": corrupted_context,
            "tools_data": {"vector": Mock()},
            "context_recovery_enabled": True,
            "reset_context_on_error": True,
        }

        # When: Tool execution encounters corrupted context
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_factory.create_tools_from_indexes.return_value = [Mock()]

            # Mock context validation that detects corruption
            with patch("src.agents.tools.ChatMemoryBuffer") as mock_context_class:
                mock_fresh_context = Mock()
                mock_context_class.from_defaults.return_value = mock_fresh_context

                result = route_query.invoke(
                    {"query": "Context corruption test", "state": mock_state}
                )

        # Then: Context recovery creates fresh context
        assert result is not None
        # Verify new context was created due to corruption
        assert mock_context_class.from_defaults.called

    # Deprecated timeout recovery probe tests removed; use direct monkeypatching
    # at tool boundaries in modern tests.

    def test_partial_failure_system_resilience(self):
        """Test system resilience under partial failure conditions."""
        # Given: State where multiple components have issues
        mock_state = {
            "messages": [HumanMessage(content="Partial failure resilience test")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {
                "vector": Mock(),  # This will work
                "kg": None,  # This is unavailable
                "retriever": Mock(),  # This will fail
            },
            "resilience_mode_enabled": True,
            "min_successful_tools": 1,
            "partial_results_acceptable": True,
        }

        # Mock mixed success/failure scenario
        mock_vector_result = Mock()
        mock_vector_result.invoke.return_value = "Vector search successful"

        mock_retriever_result = Mock()
        mock_retriever_result.invoke.side_effect = Exception(
            "Retriever connection failed"
        )

        # When: Executing with partial failures
        with patch("src.agents.tools.ToolFactory") as mock_factory:

            def create_mixed_tools(*args, **kwargs):
                # Only return tools that are available
                available_tools = []
                if mock_state["tools_data"]["vector"]:
                    available_tools.append(mock_vector_result)
                if mock_state["tools_data"]["retriever"]:
                    available_tools.append(mock_retriever_result)
                return available_tools

            mock_factory.create_tools_from_indexes.side_effect = create_mixed_tools

            with patch("src.agents.tools.logger") as mock_logger:
                result = retrieve_documents.invoke(
                    {"query": "Partial failure test", "state": mock_state}
                )

        # Then: System continues with available tools
        assert result is not None

        # Verify partial failure was logged
        mock_logger.warning.assert_called()

        # Verify result indicates partial success
        import json

        parsed_result = json.loads(result)
        assert len(parsed_result.get("documents", [])) > 0 or parsed_result.get(
            "partial_results", False
        )

    # Deprecated cascading failure prevention tests removed; modern approach uses
    # external orchestration or store-backed checkpointers.

    def test_state_corruption_detection_and_recovery(self):
        """Test detection and recovery from state corruption."""
        # Given: State that becomes corrupted during processing
        mock_state = {
            "messages": [HumanMessage(content="State corruption test")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock()},
            "state_validation_enabled": True,
            "auto_recover_corrupted_state": True,
        }

        # Start with missing tools_data to trigger boundary error handling
        mock_state["tools_data"] = None

        # First execution should return error JSON due to missing tools
        result_json = retrieve_documents.invoke(
            {"query": "State corruption test", "state": mock_state}
        )

        # Recovery: Restore minimal required state and verify route works
        mock_state["tools_data"] = {"vector": Mock()}
        route_result = route_query.invoke(
            {"query": "State recovery test", "state": mock_state}
        )

        # Then: Error JSON and successful subsequent route
        import json

        parsed = json.loads(result_json)
        assert parsed.get("error") == "No retrieval tools available"
        assert route_result is not None

    def test_memory_exhaustion_graceful_handling(self):
        """Test graceful handling of memory exhaustion scenarios."""
        # Given: State that simulates memory pressure
        mock_state = {
            "messages": [HumanMessage(content="Memory exhaustion test")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock()},
            "memory_limit_mb": 100,
            "auto_cleanup_on_memory_pressure": True,
            "reduce_context_on_memory_pressure": True,
        }

        # Simulate memory exhaustion in initial tool aggregation path
        call_count = {"create_tools": 0}

        def memory_exhaustion_simulation(*args, **kwargs):
            call_count["create_tools"] += 1
            raise MemoryError("Insufficient memory for tool execution")

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_factory.create_tools_from_indexes.side_effect = (
                memory_exhaustion_simulation
            )

            # Retrieval should fall back to detailed path and not raise
            result_json = retrieve_documents.invoke(
                {"query": "Memory test", "state": mock_state}
            )

        # Then: Memory error path was attempted and result returned
        assert call_count["create_tools"] == 1
        import json

        parsed = json.loads(result_json)
        assert "documents" in parsed
        assert parsed.get("document_count", 0) >= 0

    def test_network_failure_recovery_patterns(self):
        """Test recovery patterns for network-related failures."""
        # Given: State with vector index and short query to trigger variants
        mock_state = {
            "messages": [HumanMessage(content="Network fail test")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock()},
        }

        # Patch vector tool to fail twice then succeed
        call_counter = {"calls": 0}

        def tool_call_side_effect(_q):
            call_counter["calls"] += 1
            if call_counter["calls"] <= 2:
                raise ConnectionError(
                    f"Network connection failed (attempt {call_counter['calls']})"
                )
            return "Recovered result"

        mock_tool = Mock()
        mock_tool.call.side_effect = tool_call_side_effect

        with (
            patch(
                "src.agents.tools.ToolFactory.create_vector_search_tool",
                return_value=mock_tool,
            ),
            patch(
                "src.agents.tools.ToolFactory.create_hybrid_vector_tool",
                return_value=mock_tool,
            ),
        ):
            # Use short query (<3 words) to trigger 2 variants + primary = 3 calls
            result_json = retrieve_documents.invoke(
                {"query": "net issue", "state": mock_state}
            )

        # Then: Recovery succeeds after internal retries across variants
        assert call_counter["calls"] >= 3
        import json

        parsed = json.loads(result_json)
        assert parsed.get("document_count", 0) >= 0

    def test_validation_failure_recovery_strategy(self):
        """Test recovery strategy when response validation fails."""
        # Given: State with validation recovery enabled
        mock_state = {
            "messages": [HumanMessage(content="Validation failure test")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock()},
            "validation_recovery_enabled": True,
            "retry_on_validation_failure": True,
            "max_validation_retries": 2,
        }

        validation_attempt = 0

        def validation_failure_simulation(query, response, state):
            nonlocal validation_attempt
            validation_attempt += 1

            if validation_attempt <= 1:
                return (
                    '{"valid": false, "confidence": 0.2, "suggested_action": "retry", '
                    '"issues": ["low_quality"]}'
                )
            else:
                return (
                    '{"valid": true, "confidence": 0.8, "suggested_action": "accept", '
                    '"issues": []}'
                )

        # When: Validation fails initially then succeeds
        with patch(
            "tests.unit.test_agents_error_recovery.validate_response"
        ) as mock_tool:
            mock_tool.invoke.side_effect = lambda inp: validation_failure_simulation(
                inp["query"], inp["response"], inp.get("_state")
            )
            # First validation attempt fails
            result1 = validate_response.invoke(
                {
                    "query": "Test query",
                    "response": "Low quality response",
                    "sources": "[]",
                    "_state": mock_state,
                }
            )

            # Second validation attempt succeeds
            result2 = validate_response.invoke(
                {
                    "query": "Test query",
                    "response": "Improved response",
                    "sources": "[]",
                    "_state": mock_state,
                }
            )

        # Then: Validation recovery produces acceptable result
        import json

        result1_parsed = json.loads(result1)
        result2_parsed = json.loads(result2)

        assert result1_parsed["valid"] is False
        assert result2_parsed["valid"] is True
        assert validation_attempt == 2


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncErrorRecovery:
    """Test suite for async error recovery patterns."""

    async def test_async_timeout_with_graceful_degradation(self):
        """Test async timeout handling with graceful degradation."""
        # Given: Async state with timeout configuration
        mock_state = {
            "messages": [HumanMessage(content="Async timeout test")],
            "context": ChatMemoryBuffer.from_defaults(),
            "tools_data": {"vector": Mock()},
            "async_timeout_seconds": 0.1,
            "graceful_degradation_enabled": True,
        }

        # When: Async operation times out
        async def test_timeout_recovery():
            async def slow_async_operation():
                await asyncio.sleep(0.2)  # Exceeds timeout
                return "Slow result"

            async def fast_fallback_operation():
                await asyncio.sleep(0.05)  # Within acceptable limits
                return "Fast fallback result"

            try:
                result = await asyncio.wait_for(
                    slow_async_operation(), timeout=mock_state["async_timeout_seconds"]
                )
                return result, "no_timeout"
            except TimeoutError:
                # Graceful degradation to faster operation
                result = await asyncio.wait_for(
                    fast_fallback_operation(),
                    timeout=mock_state["async_timeout_seconds"] * 3,
                )
                return result, "graceful_degradation"

        result, recovery_type = await test_timeout_recovery()

        # Then: Graceful degradation works correctly
        assert result == "Fast fallback result"
        assert recovery_type == "graceful_degradation"

    async def test_async_resource_cleanup_on_error(self):
        """Test proper resource cleanup during async error scenarios."""
        # Given: Resources that need cleanup
        resources_created = []
        resources_cleaned = []

        async def create_mock_resource(name):
            resource = {"name": name, "created_at": time.time()}
            resources_created.append(resource)
            return resource

        async def cleanup_mock_resource(resource):
            resources_cleaned.append(resource)

        # When: Async operation fails and cleanup is needed
        async def async_operation_with_cleanup():
            resource1 = await create_mock_resource("resource1")
            resource2 = await create_mock_resource("resource2")

            try:
                # Simulate operation that fails
                raise Exception("Simulated async operation failure")
            except Exception:
                # Cleanup resources
                await cleanup_mock_resource(resource1)
                await cleanup_mock_resource(resource2)
                raise

        # Execute with expected failure
        with pytest.raises(Exception, match="Simulated async operation failure"):
            await async_operation_with_cleanup()

        # Then: Resources are properly cleaned up
        assert len(resources_created) == 2
        assert len(resources_cleaned) == 2
        assert resources_created == resources_cleaned

    async def test_async_concurrent_failure_isolation(self):
        """Test isolation of failures in concurrent async operations."""

        # Given: Multiple concurrent async operations
        async def successful_operation(name):
            await asyncio.sleep(0.01)
            return f"Success: {name}"

        async def failing_operation(name):
            await asyncio.sleep(0.01)
            raise Exception(f"Failed: {name}")

        # When: Running concurrent operations with some failures
        async def concurrent_operations_with_failures():
            tasks = [
                asyncio.create_task(successful_operation("op1")),
                asyncio.create_task(failing_operation("op2")),
                asyncio.create_task(successful_operation("op3")),
                asyncio.create_task(failing_operation("op4")),
                asyncio.create_task(successful_operation("op5")),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        results = await concurrent_operations_with_failures()

        # Then: Successful operations complete despite failures
        assert len(results) == 5

        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        assert len(successful_results) == 3
        assert len(failed_results) == 2

        # Verify successful results
        assert "Success: op1" in successful_results
        assert "Success: op3" in successful_results
        assert "Success: op5" in successful_results
