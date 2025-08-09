"""Enhanced comprehensive test coverage for agent_utils.py.

This test suite provides critical path coverage for the agent_utils module,
focusing on business-critical agent creation, tool integration, error handling,
and performance optimization to achieve 70%+ coverage.

Key areas covered:
- ReActAgent creation with enhanced configurations
- Tool creation from index data with error handling
- Agent analysis with multimodal and streaming capabilities
- Memory management and context preservation
- Performance monitoring and fallback mechanisms
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer

# Import functions under test
from agents.agent_utils import (
    analyze_documents_agentic,
    chat_with_agent,
    create_agent_with_tools,
    create_tools_from_index,
)
from utils.exceptions import AgentError, ConfigurationError


class TestCreateToolsFromIndex:
    """Test tool creation from index data with comprehensive error handling."""

    def test_create_tools_from_index_success(self):
        """Test successful tool creation from complete index data."""
        index_data = {
            "vector": MagicMock(),
            "kg": MagicMock(),
            "retriever": MagicMock(),
        }

        with patch("agents.agent_utils.ToolFactory.create_basic_tools") as mock_create:
            mock_tools = [MagicMock(), MagicMock(), MagicMock()]
            mock_create.return_value = mock_tools

            with patch("agents.agent_utils.log_performance") as mock_log_perf:
                result = create_tools_from_index(index_data)

                assert result == mock_tools
                assert len(result) == 3

                # Verify ToolFactory was called correctly
                mock_create.assert_called_once_with(index_data)

                # Verify performance logging
                mock_log_perf.assert_called_once()
                call_args = mock_log_perf.call_args[0]
                assert call_args[0] == "tool_creation_from_index"
                assert isinstance(call_args[1], float)  # duration

    def test_create_tools_from_index_empty_data(self):
        """Test tool creation with empty index data."""
        with pytest.raises(ConfigurationError) as exc_info:
            create_tools_from_index({})

        # Should provide context about empty data
        assert exc_info.value.context["index_data_keys"] == []
        assert exc_info.value.operation == "tool_creation_validation"

    def test_create_tools_from_index_none_data(self):
        """Test tool creation with None index data."""
        with pytest.raises(ConfigurationError):
            create_tools_from_index(None)

    def test_create_tools_from_index_partial_data(self):
        """Test tool creation with partial index data."""
        index_data = {
            "vector": MagicMock(),
            # Missing kg and retriever
        }

        with patch("agents.agent_utils.ToolFactory.create_basic_tools") as mock_create:
            mock_tools = [MagicMock()]
            mock_create.return_value = mock_tools

            result = create_tools_from_index(index_data)

            assert result == mock_tools
            mock_create.assert_called_once_with(index_data)

    def test_create_tools_from_index_tool_factory_import_error(self):
        """Test handling when ToolFactory import fails."""
        index_data = {"vector": MagicMock()}

        with patch(
            "agents.agent_utils.ToolFactory",
            side_effect=ImportError("ToolFactory not available"),
        ):
            with pytest.raises(AgentError):
                create_tools_from_index(index_data)

    def test_create_tools_from_index_tool_factory_error(self):
        """Test handling when ToolFactory.create_basic_tools fails."""
        index_data = {"vector": MagicMock()}

        with patch(
            "agents.agent_utils.ToolFactory.create_basic_tools",
            side_effect=RuntimeError("Tool creation failed"),
        ):
            with pytest.raises(AgentError) as exc_info:
                create_tools_from_index(index_data)

            # Should have context about the error
            assert "tool_creation_from_index" in str(exc_info.value)

    def test_create_tools_from_index_performance_logging_details(self):
        """Test detailed performance logging information."""
        index_data = {
            "vector": MagicMock(),
            "kg": MagicMock(),
        }

        with patch("agents.agent_utils.ToolFactory.create_basic_tools") as mock_create:
            mock_tools = [MagicMock(), MagicMock()]
            mock_create.return_value = mock_tools

            with patch("agents.agent_utils.log_performance") as mock_log_perf:
                create_tools_from_index(index_data)

                # Verify performance logging includes relevant context
                call_kwargs = mock_log_perf.call_args[1]
                assert call_kwargs["tool_count"] == 2
                assert call_kwargs["has_vector"] is True
                assert call_kwargs["has_kg"] is True

    def test_create_tools_from_index_fallback_decorator_behavior(self):
        """Test fallback decorator returns empty list on critical failure."""
        index_data = {"vector": MagicMock()}

        # Mock a critical failure that should trigger fallback
        with patch(
            "agents.agent_utils.ToolFactory.create_basic_tools",
            side_effect=Exception("Critical failure"),
        ):
            # The @with_fallback decorator should return empty list
            result = create_tools_from_index(index_data)

            assert result == []

    def test_create_tools_from_index_logging_success(self):
        """Test success logging with tool details."""
        index_data = {"vector": MagicMock()}

        with patch("agents.agent_utils.ToolFactory.create_basic_tools") as mock_create:
            mock_tools = [MagicMock(), MagicMock()]
            # Add names to mock tools for logging
            mock_tools[0].metadata.name = "vector_search"
            mock_tools[1].metadata.name = "knowledge_graph"
            mock_create.return_value = mock_tools

            with patch("agents.agent_utils.logger") as mock_logger:
                create_tools_from_index(index_data)

                # Verify success logging
                mock_logger.success.assert_called_once()
                call_args = mock_logger.success.call_args[0]
                assert "Created 2 query tools" in call_args[0]


class TestCreateAgentWithTools:
    """Test ReActAgent creation with comprehensive configurations."""

    def test_create_agent_with_tools_success(self):
        """Test successful agent creation with enhanced configuration."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()

        mock_tools = [MagicMock(), MagicMock()]

        with patch(
            "agents.agent_utils.create_tools_from_index", return_value=mock_tools
        ):
            with patch("agents.agent_utils.ReActAgent.from_tools") as mock_agent_class:
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent

                with patch("agents.agent_utils.log_performance") as mock_log_perf:
                    result = create_agent_with_tools(index_data, mock_llm)

                    assert result == mock_agent

                    # Verify enhanced configuration
                    mock_agent_class.assert_called_once_with(
                        tools=mock_tools,
                        llm=mock_llm,
                        verbose=True,
                        max_iterations=10,
                        memory=pytest.mock.ANY,
                    )

                    # Verify memory configuration
                    call_kwargs = mock_agent_class.call_args[1]
                    memory = call_kwargs["memory"]
                    assert isinstance(memory, ChatMemoryBuffer)

                    # Verify performance logging
                    mock_log_perf.assert_called_once()

    def test_create_agent_with_tools_no_tools(self):
        """Test agent creation when no tools are available."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()

        with patch("agents.agent_utils.create_tools_from_index", return_value=[]):
            with patch("agents.agent_utils.ReActAgent.from_tools") as mock_agent_class:
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent

                with patch("agents.agent_utils.logger") as mock_logger:
                    result = create_agent_with_tools(index_data, mock_llm)

                    assert result == mock_agent

                    # Should warn about no tools
                    mock_logger.warning.assert_called_once()
                    warning_call = mock_logger.warning.call_args[0]
                    assert "No tools created" in warning_call[0]

    def test_create_agent_with_tools_enhanced_config_failure(self):
        """Test fallback to basic config when enhanced config fails."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()
        mock_tools = [MagicMock()]

        with patch(
            "agents.agent_utils.create_tools_from_index", return_value=mock_tools
        ):
            with patch("agents.agent_utils.ReActAgent.from_tools") as mock_agent_class:
                # First call (enhanced) fails, second call (basic) succeeds
                mock_basic_agent = MagicMock()
                mock_agent_class.side_effect = [
                    RuntimeError("Enhanced config failed"),
                    mock_basic_agent,
                ]

                with patch("agents.agent_utils.logger") as mock_logger:
                    result = create_agent_with_tools(index_data, mock_llm)

                    assert result == mock_basic_agent

                    # Should have been called twice (enhanced + fallback)
                    assert mock_agent_class.call_count == 2

                    # Second call should be basic configuration
                    second_call = mock_agent_class.call_args_list[1]
                    assert "max_iterations" not in second_call[1]
                    assert "memory" not in second_call[1]

                    # Should log fallback warning
                    mock_logger.warning.assert_called()

    def test_create_agent_with_tools_complete_failure(self):
        """Test handling when both enhanced and basic config fail."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()
        mock_tools = [MagicMock()]

        with patch(
            "agents.agent_utils.create_tools_from_index", return_value=mock_tools
        ):
            with patch(
                "agents.agent_utils.ReActAgent.from_tools",
                side_effect=RuntimeError("Agent creation failed"),
            ):
                with pytest.raises(AgentError):
                    create_agent_with_tools(index_data, mock_llm)

    def test_create_agent_with_tools_tool_creation_failure(self):
        """Test handling when tool creation fails."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()

        with patch(
            "agents.agent_utils.create_tools_from_index",
            side_effect=RuntimeError("Tool creation failed"),
        ):
            with pytest.raises(AgentError):
                create_agent_with_tools(index_data, mock_llm)

    def test_create_agent_with_tools_performance_metrics(self):
        """Test detailed performance metrics collection."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()
        mock_tools = [MagicMock(), MagicMock(), MagicMock()]

        with patch(
            "agents.agent_utils.create_tools_from_index", return_value=mock_tools
        ):
            with patch(
                "agents.agent_utils.ReActAgent.from_tools", return_value=MagicMock()
            ):
                with patch("agents.agent_utils.log_performance") as mock_log_perf:
                    create_agent_with_tools(index_data, mock_llm)

                    # Verify performance metrics
                    call_kwargs = mock_log_perf.call_args[1]
                    assert call_kwargs["tool_count"] == 3
                    assert call_kwargs["max_iterations"] == 10
                    assert call_kwargs["memory_token_limit"] == 8192

    def test_create_agent_with_tools_fallback_decorator_behavior(self):
        """Test fallback decorator creates basic agent on critical failure."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()

        # Mock complete failure
        with patch("agents.agent_utils.create_tools_from_index", return_value=[]):
            with patch(
                "agents.agent_utils.ReActAgent.from_tools",
                side_effect=Exception("Critical failure"),
            ):
                # Fallback decorator should create basic agent
                result = create_agent_with_tools(index_data, mock_llm)

                # Should be a ReActAgent (from fallback decorator)
                assert isinstance(result, ReActAgent)


class TestAnalyzeDocumentsAgentic:
    """Test agentic document analysis functionality."""

    def test_analyze_documents_agentic_success(self):
        """Test successful agentic document analysis."""
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.response = (
            "Analysis complete: Document contains important information."
        )
        mock_agent.chat.return_value = mock_response

        index_data = {"vector": MagicMock(), "kg": MagicMock()}

        # Configure query engines
        mock_vector_engine = MagicMock()
        mock_kg_engine = MagicMock()
        index_data["vector"].as_query_engine.return_value = mock_vector_engine
        index_data["kg"].as_query_engine.return_value = mock_kg_engine

        with patch("agents.agent_utils.ToolFactory") as mock_tool_factory:
            mock_tool_factory.create_query_tool.return_value = MagicMock()

            with patch("agents.agent_utils.log_performance") as mock_log_perf:
                result = analyze_documents_agentic(mock_agent, index_data, "summary")

                assert (
                    result
                    == "Analysis complete: Document contains important information."
                )
                mock_agent.chat.assert_called_once_with("Analyze with prompt: summary")

                # Verify performance logging
                mock_log_perf.assert_called_once()

    def test_analyze_documents_agentic_empty_index_data(self):
        """Test analysis with empty index data."""
        mock_agent = MagicMock()

        with pytest.raises(ConfigurationError) as exc_info:
            analyze_documents_agentic(mock_agent, {}, "summary")

        assert exc_info.value.operation == "analysis_input_validation"
        assert exc_info.value.context["prompt_type"] == "summary"

    def test_analyze_documents_agentic_none_index_data(self):
        """Test analysis with None index data."""
        mock_agent = MagicMock()

        with pytest.raises(ConfigurationError):
            analyze_documents_agentic(mock_agent, None, "summary")

    def test_analyze_documents_agentic_vector_query_engine_failure(self):
        """Test handling when vector query engine creation fails."""
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.response = "Analysis with limited tools"
        mock_agent.chat.return_value = mock_response

        index_data = {
            "vector": MagicMock(),
        }

        # Vector query engine creation fails
        index_data["vector"].as_query_engine.side_effect = RuntimeError(
            "Query engine failed"
        )

        with patch("agents.agent_utils.ToolFactory") as mock_tool_factory:
            with patch("agents.agent_utils.logger") as mock_logger:
                result = analyze_documents_agentic(mock_agent, index_data, "summary")

                assert result == "Analysis with limited tools"

                # Should warn about query engine failure
                mock_logger.warning.assert_called()

    def test_analyze_documents_agentic_no_agent_fallback(self):
        """Test fallback agent creation when no agent provided."""
        index_data = {"vector": MagicMock()}
        mock_vector_engine = MagicMock()
        index_data["vector"].as_query_engine.return_value = mock_vector_engine

        with patch("agents.agent_utils.ToolFactory") as mock_tool_factory:
            mock_tool = MagicMock()
            mock_tool_factory.create_query_tool.return_value = mock_tool

            with patch("agents.agent_utils.ReActAgent.from_tools") as mock_agent_class:
                mock_fallback_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.response = "Fallback analysis"
                mock_fallback_agent.chat.return_value = mock_response
                mock_agent_class.return_value = mock_fallback_agent

                with patch("agents.agent_utils.Ollama") as mock_ollama:
                    with patch("agents.agent_utils.settings") as mock_settings:
                        mock_settings.default_model = "llama2"

                        result = analyze_documents_agentic(None, index_data, "summary")

                        assert result == "Fallback analysis"

                        # Should create fallback agent with default model
                        mock_ollama.assert_called_once_with(model="llama2")
                        mock_agent_class.assert_called_once()

    def test_analyze_documents_agentic_agent_chat_failure(self):
        """Test handling when agent chat fails."""
        mock_agent = MagicMock()
        mock_agent.chat.side_effect = RuntimeError("Chat failed")

        index_data = {"vector": MagicMock()}
        mock_vector_engine = MagicMock()
        index_data["vector"].as_query_engine.return_value = mock_vector_engine

        with patch("agents.agent_utils.ToolFactory") as mock_tool_factory:
            mock_tool_factory.create_query_tool.return_value = MagicMock()

            with patch("agents.agent_utils.logger") as mock_logger:
                result = analyze_documents_agentic(mock_agent, index_data, "summary")

                # Should provide fallback message
                assert "Analysis partially completed" in result
                assert "summary" in result

                # Should log the error
                mock_logger.warning.assert_called()

    def test_analyze_documents_agentic_performance_metrics(self):
        """Test performance metrics collection in analysis."""
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.response = "Complete analysis"
        mock_agent.chat.return_value = mock_response

        index_data = {"vector": MagicMock(), "kg": MagicMock()}

        # Configure query engines
        index_data["vector"].as_query_engine.return_value = MagicMock()
        index_data["kg"].as_query_engine.return_value = MagicMock()

        with patch("agents.agent_utils.ToolFactory") as mock_tool_factory:
            mock_tool_factory.create_query_tool.return_value = MagicMock()

            with patch("agents.agent_utils.log_performance") as mock_log_perf:
                analyze_documents_agentic(mock_agent, index_data, "detailed")

                # Verify performance metrics
                call_kwargs = mock_log_perf.call_args[1]
                assert call_kwargs["prompt_type"] == "detailed"
                assert call_kwargs["tool_count"] == 2  # vector + kg
                assert call_kwargs["has_vector"] is True
                assert call_kwargs["has_kg"] is True

    def test_analyze_documents_agentic_kg_query_engine_failure(self):
        """Test handling when KG query engine creation fails."""
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.response = "Analysis with vector only"
        mock_agent.chat.return_value = mock_response

        index_data = {"vector": MagicMock(), "kg": MagicMock()}

        # Vector works, KG fails
        index_data["vector"].as_query_engine.return_value = MagicMock()
        index_data["kg"].as_query_engine.side_effect = RuntimeError("KG failed")

        with patch("agents.agent_utils.ToolFactory") as mock_tool_factory:
            mock_tool_factory.create_query_tool.return_value = MagicMock()

            with patch("agents.agent_utils.logger") as mock_logger:
                result = analyze_documents_agentic(mock_agent, index_data, "summary")

                assert result == "Analysis with vector only"

                # Should have warnings for KG failure
                warning_calls = [
                    call[0][0] for call in mock_logger.warning.call_args_list
                ]
                assert any(
                    "Failed to create KG query engine" in call for call in warning_calls
                )

    def test_analyze_documents_agentic_fallback_decorator_behavior(self):
        """Test fallback decorator behavior on critical failure."""
        mock_agent = MagicMock()

        # Complete failure scenario
        with patch(
            "agents.agent_utils.ToolFactory",
            side_effect=ImportError("ToolFactory failed"),
        ):
            # Fallback decorator should return default message
            result = analyze_documents_agentic(
                mock_agent, {"vector": MagicMock()}, "summary"
            )

            assert "Analysis failed for prompt type: summary" in result


class TestChatWithAgent:
    """Test async agent chat functionality."""

    @pytest.mark.asyncio
    async def test_chat_with_agent_success(self):
        """Test successful async agent chat."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()
        mock_memory.token_limit = 4096

        # Mock async streaming response
        mock_response = MagicMock()
        mock_async_gen = AsyncMock()

        async def mock_response_generator():
            yield "Hello"
            yield " there"
            yield "!"

        mock_async_gen.return_value = mock_response_generator()
        mock_response.async_response_gen = mock_async_gen

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_response

            with patch("agents.agent_utils.log_performance") as mock_log_perf:
                chunks = []
                async for chunk in chat_with_agent(mock_agent, "Hello", mock_memory):
                    chunks.append(chunk)

                assert chunks == ["Hello", " there", "!"]

                # Verify asyncio.to_thread was called
                mock_to_thread.assert_called_once_with(
                    mock_agent.async_stream_chat, "Hello", memory=mock_memory
                )

                # Verify performance logging
                mock_log_perf.assert_called_once()
                call_kwargs = mock_log_perf.call_args[1]
                assert call_kwargs["user_input_length"] == 5
                assert call_kwargs["chunk_count"] == 3

    @pytest.mark.asyncio
    async def test_chat_with_agent_empty_input(self):
        """Test chat with empty user input."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()

        with pytest.raises(ConfigurationError) as exc_info:
            async for chunk in chat_with_agent(mock_agent, "", mock_memory):
                pass

        assert exc_info.value.operation == "chat_input_validation"

    @pytest.mark.asyncio
    async def test_chat_with_agent_whitespace_only_input(self):
        """Test chat with whitespace-only input."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()

        with pytest.raises(ConfigurationError):
            async for chunk in chat_with_agent(mock_agent, "   \n  ", mock_memory):
                pass

    @pytest.mark.asyncio
    async def test_chat_with_agent_none_agent(self):
        """Test chat with None agent."""
        mock_memory = MagicMock()

        with pytest.raises(ConfigurationError) as exc_info:
            async for chunk in chat_with_agent(None, "Hello", mock_memory):
                pass

        assert exc_info.value.operation == "chat_agent_validation"

    @pytest.mark.asyncio
    async def test_chat_with_agent_streaming_error(self):
        """Test handling of streaming errors."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()
        mock_memory.token_limit = 4096

        with patch("asyncio.to_thread", side_effect=RuntimeError("Streaming failed")):
            with patch("agents.agent_utils.log_error_with_context") as mock_log_error:
                chunks = []
                async for chunk in chat_with_agent(mock_agent, "Hello", mock_memory):
                    chunks.append(chunk)

                # Should get error message
                assert len(chunks) == 1
                assert "Chat processing encountered an error" in chunks[0]
                assert "Streaming failed" in chunks[0]

                # Should log the error
                mock_log_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_agent_async_response_gen_error(self):
        """Test error in async response generator."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()

        # Mock successful to_thread but failing async_response_gen
        mock_response = MagicMock()

        async def failing_generator():
            yield "Start"
            raise RuntimeError("Generator failed")

        mock_response.async_response_gen.return_value = failing_generator()

        with patch("asyncio.to_thread", return_value=mock_response):
            with patch("agents.agent_utils.log_error_with_context"):
                chunks = []
                async for chunk in chat_with_agent(mock_agent, "Hello", mock_memory):
                    chunks.append(chunk)

                # Should get partial response + error message
                assert len(chunks) == 2
                assert chunks[0] == "Start"
                assert "Chat processing encountered an error" in chunks[1]

    @pytest.mark.asyncio
    async def test_chat_with_agent_performance_metrics_detailed(self):
        """Test detailed performance metrics collection."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()
        mock_memory.token_limit = 8192

        # Mock response with multiple chunks
        mock_response = MagicMock()

        async def mock_generator():
            for i in range(5):
                yield f"chunk{i}"

        mock_response.async_response_gen.return_value = mock_generator()

        with patch("asyncio.to_thread", return_value=mock_response):
            with patch("agents.agent_utils.log_performance") as mock_log_perf:
                chunks = []
                async for chunk in chat_with_agent(
                    mock_agent, "Long user query", mock_memory
                ):
                    chunks.append(chunk)

                # Verify performance metrics
                call_kwargs = mock_log_perf.call_args[1]
                assert call_kwargs["user_input_length"] == 15  # "Long user query"
                assert call_kwargs["chunk_count"] == 5
                assert call_kwargs["agent_type"] == type(mock_agent).__name__

    @pytest.mark.asyncio
    async def test_chat_with_agent_timeout_handling(self):
        """Test timeout handling with async_with_timeout decorator."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()

        # Mock a very slow response that should timeout
        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(300)  # 5 minutes - should exceed 2 minute timeout
            return MagicMock()

        with patch("asyncio.to_thread", side_effect=slow_operation):
            with pytest.raises(asyncio.TimeoutError):
                async for chunk in chat_with_agent(mock_agent, "Hello", mock_memory):
                    pass

    @pytest.mark.asyncio
    async def test_chat_with_agent_memory_context(self):
        """Test that memory context is properly used."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()
        mock_memory.token_limit = 2048

        mock_response = MagicMock()

        async def single_chunk():
            yield "Response with memory context"

        mock_response.async_response_gen.return_value = single_chunk()

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = mock_response

            chunks = []
            async for chunk in chat_with_agent(mock_agent, "Hello", mock_memory):
                chunks.append(chunk)

            # Verify memory was passed to agent
            call_kwargs = mock_to_thread.call_args[1]
            assert call_kwargs["memory"] == mock_memory

    @pytest.mark.asyncio
    async def test_chat_with_agent_critical_error_handling(self):
        """Test handling of critical errors that bypass stream error handling."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()

        # Mock a critical error in the main try block
        with patch(
            "agents.agent_utils.ConfigurationError",
            side_effect=Exception("Critical error"),
        ):
            with pytest.raises(AgentError):
                async for chunk in chat_with_agent(mock_agent, "Hello", mock_memory):
                    pass

    @pytest.mark.asyncio
    async def test_chat_with_agent_multimodal_note(self):
        """Test that multimodal handling note is documented (code comment verification)."""
        # This test verifies the multimodal handling documentation exists
        # by checking the function's implementation includes the note
        import inspect

        source = inspect.getsource(chat_with_agent)
        assert "multimodal handling depends on LLM backend" in source.lower()
        assert "gemma" in source.lower()
        assert "nemotron" in source.lower()


class TestAgentUtilsIntegration:
    """Test integration scenarios between agent utility functions."""

    def test_complete_agent_workflow(self):
        """Test complete workflow from index data to agent analysis."""
        # Setup complete index data
        index_data = {
            "vector": MagicMock(),
            "kg": MagicMock(),
            "retriever": MagicMock(),
        }
        mock_llm = MagicMock()

        # Mock tool creation
        mock_tools = [MagicMock(), MagicMock(), MagicMock()]
        for i, tool in enumerate(mock_tools):
            tool.metadata.name = f"tool_{i}"

        with patch(
            "agents.agent_utils.create_tools_from_index", return_value=mock_tools
        ):
            # Mock agent creation
            mock_agent = MagicMock()
            mock_agent.chat.return_value.response = "Comprehensive analysis complete"

            with patch(
                "agents.agent_utils.ReActAgent.from_tools", return_value=mock_agent
            ):
                # Step 1: Create agent with tools
                agent = create_agent_with_tools(index_data, mock_llm)

                assert agent == mock_agent

                # Step 2: Perform analysis with agent
                # Setup query engines for analysis
                index_data["vector"].as_query_engine.return_value = MagicMock()
                index_data["kg"].as_query_engine.return_value = MagicMock()

                with patch("agents.agent_utils.ToolFactory") as mock_tool_factory:
                    mock_tool_factory.create_query_tool.return_value = MagicMock()

                    result = analyze_documents_agentic(
                        agent, index_data, "comprehensive"
                    )

                    assert result == "Comprehensive analysis complete"

    @pytest.mark.asyncio
    async def test_agent_creation_to_chat_workflow(self):
        """Test workflow from agent creation to chat interaction."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()
        mock_tools = [MagicMock()]

        with patch(
            "agents.agent_utils.create_tools_from_index", return_value=mock_tools
        ):
            mock_agent = MagicMock()

            with patch(
                "agents.agent_utils.ReActAgent.from_tools", return_value=mock_agent
            ):
                # Create agent
                agent = create_agent_with_tools(index_data, mock_llm)

                # Use agent for chat
                mock_memory = MagicMock()
                mock_response = MagicMock()

                async def chat_generator():
                    yield "Chat response"

                mock_response.async_response_gen.return_value = chat_generator()

                with patch("asyncio.to_thread", return_value=mock_response):
                    chunks = []
                    async for chunk in chat_with_agent(agent, "Hello", mock_memory):
                        chunks.append(chunk)

                    assert chunks == ["Chat response"]

    def test_error_propagation_through_workflow(self):
        """Test how errors propagate through the complete workflow."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()

        # Test error in tool creation
        with patch(
            "agents.agent_utils.create_tools_from_index",
            side_effect=RuntimeError("Tool creation failed"),
        ):
            with pytest.raises(AgentError):
                create_agent_with_tools(index_data, mock_llm)

        # Test error propagates to analysis
        mock_agent = MagicMock()

        with patch(
            "agents.agent_utils.ToolFactory", side_effect=ImportError("Import failed")
        ):
            with pytest.raises(AgentError):
                analyze_documents_agentic(mock_agent, index_data, "test")

    def test_performance_monitoring_across_workflow(self):
        """Test that performance is monitored across the complete workflow."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()
        mock_tools = [MagicMock()]

        with patch(
            "agents.agent_utils.create_tools_from_index", return_value=mock_tools
        ):
            with patch(
                "agents.agent_utils.ReActAgent.from_tools", return_value=MagicMock()
            ):
                with patch("agents.agent_utils.log_performance") as mock_log_perf:
                    # Create agent (should log performance)
                    create_agent_with_tools(index_data, mock_llm)

                    # Should have performance logging for agent creation
                    agent_creation_calls = [
                        call
                        for call in mock_log_perf.call_args_list
                        if call[0][0] == "react_agent_creation"
                    ]
                    assert len(agent_creation_calls) == 1

                    # Now test analysis performance logging
                    mock_agent = MagicMock()
                    mock_agent.chat.return_value.response = "Analysis"

                    index_data["vector"].as_query_engine.return_value = MagicMock()

                    with patch("agents.agent_utils.ToolFactory") as mock_tool_factory:
                        mock_tool_factory.create_query_tool.return_value = MagicMock()

                        analyze_documents_agentic(mock_agent, index_data, "test")

                        # Should have performance logging for analysis too
                        analysis_calls = [
                            call
                            for call in mock_log_perf.call_args_list
                            if call[0][0] == "agentic_document_analysis"
                        ]
                        assert len(analysis_calls) == 1


class TestAgentUtilsErrorRecovery:
    """Test error recovery and resilience scenarios."""

    def test_create_tools_from_index_partial_recovery(self):
        """Test recovery when some index components are problematic."""
        index_data = {
            "vector": MagicMock(),
            "kg": "invalid_kg",  # Invalid type
            "retriever": None,  # None value
        }

        with patch("agents.agent_utils.ToolFactory.create_basic_tools") as mock_create:
            # ToolFactory should handle invalid components gracefully
            mock_tools = [MagicMock()]  # Only vector tool created
            mock_create.return_value = mock_tools

            result = create_tools_from_index(index_data)

            assert result == mock_tools
            mock_create.assert_called_once_with(index_data)

    def test_create_agent_memory_fallback(self):
        """Test memory configuration fallback scenarios."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()
        mock_tools = [MagicMock()]

        with patch(
            "agents.agent_utils.create_tools_from_index", return_value=mock_tools
        ):
            with patch(
                "agents.agent_utils.ChatMemoryBuffer.from_defaults",
                side_effect=RuntimeError("Memory creation failed"),
            ):
                with patch(
                    "agents.agent_utils.ReActAgent.from_tools"
                ) as mock_agent_class:
                    # Should fail on first attempt (with memory), succeed on second (without)
                    mock_agent_class.side_effect = [
                        RuntimeError("Enhanced config failed"),
                        MagicMock(),  # Basic config succeeds
                    ]

                    result = create_agent_with_tools(index_data, mock_llm)

                    # Should fall back to basic configuration without memory
                    assert mock_agent_class.call_count == 2
                    basic_call = mock_agent_class.call_args_list[1]
                    assert "memory" not in basic_call[1]

    def test_analyze_documents_resource_cleanup(self):
        """Test resource cleanup in analysis error scenarios."""
        mock_agent = MagicMock()
        index_data = {"vector": MagicMock()}

        # Mock query engine that needs cleanup
        mock_query_engine = MagicMock()
        index_data["vector"].as_query_engine.return_value = mock_query_engine

        with patch("agents.agent_utils.ToolFactory") as mock_tool_factory:
            mock_tool_factory.create_query_tool.side_effect = RuntimeError(
                "Tool creation failed"
            )

            # Even with tool creation failure, query engine creation should complete
            with patch("agents.agent_utils.logger"):
                result = analyze_documents_agentic(mock_agent, index_data, "test")

                # Should still attempt to create query engine
                index_data["vector"].as_query_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_agent_graceful_degradation(self):
        """Test graceful degradation in chat functionality."""
        mock_agent = MagicMock()
        mock_memory = MagicMock()

        # Mock partial streaming failure
        async def partial_failure_generator():
            yield "Success chunk"
            raise RuntimeError("Partial failure")

        mock_response = MagicMock()
        mock_response.async_response_gen.return_value = partial_failure_generator()

        with patch("asyncio.to_thread", return_value=mock_response):
            with patch("agents.agent_utils.log_error_with_context"):
                chunks = []
                async for chunk in chat_with_agent(mock_agent, "Hello", mock_memory):
                    chunks.append(chunk)

                # Should get successful chunk + error message
                assert len(chunks) == 2
                assert chunks[0] == "Success chunk"
                assert "error" in chunks[1].lower()

    def test_settings_fallback_handling(self):
        """Test handling when settings are not properly configured."""
        index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()

        # Mock settings access failure
        with patch(
            "agents.agent_utils.settings",
            side_effect=AttributeError("Settings not configured"),
        ):
            with patch(
                "agents.agent_utils.ToolFactory.create_basic_tools", return_value=[]
            ):
                with patch(
                    "agents.agent_utils.ReActAgent.from_tools", return_value=MagicMock()
                ):
                    # Should handle settings access gracefully
                    result = create_agent_with_tools(index_data, mock_llm)

                    assert result is not None
