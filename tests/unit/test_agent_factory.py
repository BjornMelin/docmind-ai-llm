"""Comprehensive test suite for new ReActAgent implementation.

Tests the simplified agent_factory.py with pure LlamaIndex ReActAgent.
Validates agent creation, tool integration, query processing, and agentic capabilities.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.tools import QueryEngineTool

from src.agents.agent_factory import (
    create_agentic_rag_system,
    create_single_agent,
    get_agent_system,
    process_query_with_agent_system,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.__str__ = lambda x: "MockLLM"
    llm.__repr__ = lambda x: "MockLLM"
    return llm


@pytest.fixture
def mock_query_engine_tools():
    """Create mock QueryEngineTool objects for testing."""
    tools = []
    tool_configs = [
        ("semantic_search", "Semantic similarity search using dense embeddings"),
        ("keyword_search", "Keyword-based search using sparse embeddings"),
        ("hybrid_search", "Hybrid search combining dense and sparse methods"),
    ]

    for name, description in tool_configs:
        tool = MagicMock(spec=QueryEngineTool)
        tool.metadata.name = name
        tool.metadata.description = description
        tools.append(tool)

    return tools


@pytest.fixture
def mock_react_agent():
    """Create a mock ReActAgent for testing."""
    agent = MagicMock()

    # Mock response object with response attribute
    mock_response = MagicMock()
    mock_response.response = "Mock agent response with detailed analysis"
    mock_response.__str__ = lambda x: "Mock agent response with detailed analysis"
    agent.chat.return_value = mock_response

    return agent


@pytest.fixture
def mock_memory():
    """Create a mock ChatMemoryBuffer for testing."""
    memory = MagicMock()
    memory.token_limit = 8192
    return memory


class TestReActAgentCreation:
    """Test ReActAgent creation and configuration."""

    @patch("src.agents.agent_factory.ReActAgent")
    @patch("src.agents.agent_factory.ChatMemoryBuffer")
    def test_create_agentic_rag_system_success(
        self,
        mock_memory_class,
        mock_react_agent_class,
        mock_llm,
        mock_query_engine_tools,
    ):
        """Test successful creation of ReActAgent with proper configuration."""
        # Setup mocks
        mock_memory_instance = MagicMock()
        mock_memory_class.from_defaults.return_value = mock_memory_instance
        mock_agent_instance = MagicMock()
        mock_react_agent_class.from_tools.return_value = mock_agent_instance

        # Call function
        result = create_agentic_rag_system(mock_query_engine_tools, mock_llm)

        # Verify agent creation was called with correct parameters
        mock_react_agent_class.from_tools.assert_called_once()
        call_kwargs = mock_react_agent_class.from_tools.call_args[1]

        assert call_kwargs["tools"] == mock_query_engine_tools
        assert call_kwargs["llm"] == mock_llm
        assert call_kwargs["verbose"] is True
        assert call_kwargs["max_iterations"] == 3
        assert "intelligent document analysis agent" in call_kwargs["system_prompt"]

        # Verify memory was created with correct token limit
        mock_memory_class.from_defaults.assert_called_once_with(token_limit=8192)

        assert result == mock_agent_instance

    @patch("src.agents.agent_factory.ReActAgent")
    @patch("src.agents.agent_factory.ChatMemoryBuffer")
    def test_create_agentic_rag_system_with_custom_memory(
        self,
        mock_memory_class,
        mock_react_agent_class,
        mock_llm,
        mock_query_engine_tools,
        mock_memory,
    ):
        """Test agent creation with custom memory buffer."""
        mock_agent_instance = MagicMock()
        mock_react_agent_class.from_tools.return_value = mock_agent_instance

        result = create_agentic_rag_system(
            mock_query_engine_tools, mock_llm, mock_memory
        )

        # Should use provided memory instead of creating new one
        mock_memory_class.from_defaults.assert_not_called()
        call_kwargs = mock_react_agent_class.from_tools.call_args[1]
        assert call_kwargs["memory"] == mock_memory
        assert result == mock_agent_instance

    @patch("src.agents.agent_factory.ReActAgent")
    @patch("src.agents.agent_factory.ChatMemoryBuffer")
    @patch("src.agents.agent_factory.logger")
    def test_create_agentic_rag_system_empty_tools(
        self,
        mock_logger,
        mock_memory_class,
        mock_react_agent_class,
        mock_llm,  # noqa: ARG002
    ):
        """Test agent creation with empty tools list."""
        mock_agent_instance = MagicMock()
        mock_react_agent_class.from_tools.return_value = mock_agent_instance

        result = create_agentic_rag_system([], mock_llm)

        # Should log warning and create agent with empty tools
        mock_logger.warning.assert_called_once_with(
            "No tools provided for ReActAgent creation"
        )
        mock_react_agent_class.from_tools.assert_called_once_with([], mock_llm)
        assert result == mock_agent_instance

    @patch("src.agents.agent_factory.ReActAgent")
    def test_create_single_agent_legacy_compatibility(
        self, mock_react_agent_class, mock_llm, mock_query_engine_tools
    ):
        """Test legacy compatibility function."""
        mock_agent_instance = MagicMock()
        mock_react_agent_class.from_tools.return_value = mock_agent_instance

        result = create_single_agent(mock_query_engine_tools, mock_llm)

        # Should call create_agentic_rag_system
        assert result == mock_agent_instance
        mock_react_agent_class.from_tools.assert_called_once()

    def test_system_prompt_content(self, mock_llm, mock_query_engine_tools):
        """Test that system prompt contains key agentic instructions."""
        with patch("src.agents.agent_factory.ReActAgent") as mock_react_agent_class:
            mock_react_agent_class.from_tools.return_value = MagicMock()

            create_agentic_rag_system(mock_query_engine_tools, mock_llm)

            call_kwargs = mock_react_agent_class.from_tools.call_args[1]
            system_prompt = call_kwargs["system_prompt"]

            # Check for key agentic behavior instructions
            assert "Think step-by-step" in system_prompt
            assert "most appropriate tools" in system_prompt
            assert "multiple tools" in system_prompt
            assert "Cross-reference results" in system_prompt
            assert "explain your reasoning" in system_prompt


class TestAgentSystemCompatibility:
    """Test get_agent_system() backward compatibility function."""

    @patch("src.agents.agent_factory.create_agentic_rag_system")
    @patch("src.agents.agent_factory.logger")
    def test_get_agent_system_default_behavior(
        self, mock_logger, mock_create_agent, mock_llm, mock_query_engine_tools
    ):
        """Test get_agent_system returns single agent by default."""
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        agent, mode = get_agent_system(mock_query_engine_tools, mock_llm)

        assert agent == mock_agent
        assert mode == "single"
        mock_create_agent.assert_called_once_with(
            mock_query_engine_tools, mock_llm, None
        )
        mock_logger.info.assert_not_called()

    @patch("src.agents.agent_factory.create_agentic_rag_system")
    @patch("src.agents.agent_factory.logger")
    def test_get_agent_system_multi_agent_requested(
        self, mock_logger, mock_create_agent, mock_llm, mock_query_engine_tools
    ):
        """Test get_agent_system logs when multi-agent requested but returns single."""
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        agent, mode = get_agent_system(
            mock_query_engine_tools, mock_llm, enable_multi_agent=True
        )

        assert agent == mock_agent
        assert mode == "single"
        mock_logger.info.assert_called_once_with(
            "Multi-agent requested but using single optimized agent"
        )

    @patch("src.agents.agent_factory.create_agentic_rag_system")
    def test_get_agent_system_with_memory(
        self, mock_create_agent, mock_llm, mock_query_engine_tools, mock_memory
    ):
        """Test get_agent_system passes custom memory to agent creation."""
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        agent, mode = get_agent_system(
            mock_query_engine_tools, mock_llm, memory=mock_memory
        )

        assert agent == mock_agent
        assert mode == "single"
        mock_create_agent.assert_called_once_with(
            mock_query_engine_tools, mock_llm, mock_memory
        )

    @pytest.mark.parametrize(
        ("enable_multi_agent", "enable_human_in_loop", "checkpoint_path"),
        [
            (False, False, None),
            (True, False, None),
            (False, True, "/tmp/checkpoint"),
            (True, True, "/tmp/checkpoint"),
        ],
    )
    @patch("src.agents.agent_factory.create_agentic_rag_system")
    def test_get_agent_system_ignores_unused_parameters(
        self,
        mock_create_agent,
        enable_multi_agent,
        enable_human_in_loop,
        checkpoint_path,
        mock_llm,
        mock_query_engine_tools,
    ):
        """Test that unused parameters are ignored in new implementation."""
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        agent, mode = get_agent_system(
            mock_query_engine_tools,
            mock_llm,
            enable_multi_agent=enable_multi_agent,
            enable_human_in_loop=enable_human_in_loop,
            checkpoint_path=checkpoint_path,
        )

        # Always returns single agent regardless of parameters
        assert agent == mock_agent
        assert mode == "single"
        mock_create_agent.assert_called_once()


class TestQueryProcessing:
    """Test query processing with ReActAgent."""

    def test_process_query_success(self, mock_react_agent):
        """Test successful query processing with response extraction."""
        query = "What is the main topic of the documents?"

        result = process_query_with_agent_system(mock_react_agent, query, "single")

        assert result == "Mock agent response with detailed analysis"
        mock_react_agent.chat.assert_called_once_with(query)

    def test_process_query_with_string_response(self):
        """Test query processing when agent returns string instead of object."""
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "Direct string response"
        query = "Simple query"

        result = process_query_with_agent_system(mock_agent, query, "single")

        assert result == "Direct string response"
        mock_agent.chat.assert_called_once_with(query)

    def test_process_query_with_mode_parameter(self, mock_react_agent):
        """Test that mode parameter is accepted but doesn't affect processing."""
        query = "Test query"

        # Test with different mode values
        for mode in ["single", "multi", "unknown"]:
            result = process_query_with_agent_system(mock_react_agent, query, mode)
            assert result == "Mock agent response with detailed analysis"

        # Should have been called 3 times
        assert mock_react_agent.chat.call_count == 3

    def test_process_query_ignores_unused_parameters(self, mock_react_agent):
        """Test that unused parameters are ignored gracefully."""
        query = "Test query with unused params"
        mock_memory = MagicMock()
        thread_id = "thread_123"

        result = process_query_with_agent_system(
            mock_react_agent, query, "single", memory=mock_memory, thread_id=thread_id
        )

        assert result == "Mock agent response with detailed analysis"
        mock_react_agent.chat.assert_called_once_with(query)

    @pytest.mark.parametrize(
        "query",
        [
            "Simple question about AI",
            "Complex multi-part query about machine learning and deep learning",
            "Technical query about SPLADE++ sparse embeddings",
            "What are the relationships between these concepts?",
            "",  # Empty query
            "?",  # Single character
        ],
    )
    def test_process_query_various_inputs(self, mock_react_agent, query):
        """Test query processing with various input types."""
        result = process_query_with_agent_system(mock_react_agent, query, "single")

        assert result == "Mock agent response with detailed analysis"
        mock_react_agent.chat.assert_called_once_with(query)


class TestErrorHandling:
    """Test error handling and edge cases in agent factory."""

    @patch("src.agents.agent_factory.logger")
    def test_process_query_exception_handling(self, mock_logger):
        """Test graceful error handling when agent.chat() fails."""
        mock_agent = MagicMock()
        mock_agent.chat.side_effect = ValueError("Agent processing failed")
        query = "Test query that causes failure"

        result = process_query_with_agent_system(mock_agent, query, "single")

        # Should return error message instead of raising exception
        assert "Error processing query" in result
        assert "Agent processing failed" in result

        # Should log the error
        mock_logger.error.assert_called_once()
        error_call_args = mock_logger.error.call_args[0][0]
        assert "Query processing failed" in error_call_args

    def test_process_query_with_none_response(self):
        """Test handling when agent returns None response."""
        mock_agent = MagicMock()
        mock_agent.chat.return_value = None

        result = process_query_with_agent_system(mock_agent, "test", "single")

        assert result == "None"

    def test_process_query_with_object_without_response_attr(self):
        """Test handling when agent returns object without response attribute."""
        mock_agent = MagicMock()
        mock_response = MagicMock()
        del mock_response.response  # Remove response attribute
        mock_response.__str__ = lambda x: "String representation of response"
        mock_agent.chat.return_value = mock_response

        result = process_query_with_agent_system(mock_agent, "test", "single")

        assert result == "String representation of response"

    @patch("src.agents.agent_factory.ReActAgent")
    def test_agent_creation_with_invalid_tools(self, mock_react_agent_class, mock_llm):
        """Test agent creation with invalid or malformed tools."""
        invalid_tools = [None, "not a tool", 123]
        mock_react_agent_class.from_tools.return_value = MagicMock()

        # Should not raise exception, just pass invalid tools to ReActAgent
        result = create_agentic_rag_system(invalid_tools, mock_llm)

        assert result is not None
        mock_react_agent_class.from_tools.assert_called_once()
        call_kwargs = mock_react_agent_class.from_tools.call_args[1]
        assert call_kwargs["tools"] == invalid_tools

    @patch("src.agents.agent_factory.ReActAgent")
    def test_agent_creation_with_none_llm(self, mock_react_agent_class):
        """Test agent creation with None LLM."""
        mock_tools = [MagicMock()]
        mock_react_agent_class.from_tools.return_value = MagicMock()

        # Should not raise exception, just pass None LLM to ReActAgent
        result = create_agentic_rag_system(mock_tools, None)

        assert result is not None
        mock_react_agent_class.from_tools.assert_called_once()
        call_kwargs = mock_react_agent_class.from_tools.call_args[1]
        assert call_kwargs["llm"] is None

    @patch("src.agents.agent_factory.ReActAgent")
    @patch("src.agents.agent_factory.ChatMemoryBuffer")
    def test_memory_creation_failure_graceful_handling(
        self, mock_memory_class, mock_react_agent_class, mock_llm
    ):
        """Test graceful handling when memory creation fails."""
        mock_tools = [MagicMock()]
        mock_memory_class.from_defaults.side_effect = Exception(
            "Memory creation failed"
        )
        mock_react_agent_class.from_tools.return_value = MagicMock()

        # Should propagate exception since it can't create proper memory
        with pytest.raises(Exception, match="Memory creation failed"):
            create_agentic_rag_system(mock_tools, mock_llm)


class TestPerformanceScenarios:
    """Test basic performance characteristics and timing."""

    @pytest.mark.performance
    def test_agent_creation_performance(
        self, benchmark, mock_llm, mock_query_engine_tools
    ):
        """Test performance of ReActAgent creation."""
        with patch("src.agents.agent_factory.ReActAgent") as mock_react_agent_class:
            mock_react_agent_class.from_tools.return_value = MagicMock()

            def create_agent():
                return create_agentic_rag_system(mock_query_engine_tools, mock_llm)

            result = benchmark(create_agent)
            assert result is not None

    @pytest.mark.performance
    def test_query_processing_timing(self, mock_react_agent):
        """Test that query processing completes within reasonable time."""
        query = (
            "Complex analytical query about machine learning and document processing"
        )

        start_time = time.time()
        result = process_query_with_agent_system(mock_react_agent, query, "single")
        end_time = time.time()

        # Should complete quickly with mock (real test would use actual agent)
        processing_time = end_time - start_time
        assert processing_time < 0.1  # Mock should be very fast
        assert result == "Mock agent response with detailed analysis"

    def test_multiple_tool_handling(self, mock_llm):
        """Test agent creation with multiple tools."""
        # Create larger tool set
        large_tool_set = []
        tool_names = [
            "semantic_search",
            "keyword_search",
            "hybrid_search",
            "knowledge_graph",
            "document_summary",
            "entity_extraction",
        ]

        for name in tool_names:
            tool = MagicMock(spec=QueryEngineTool)
            tool.metadata.name = name
            tool.metadata.description = f"Mock {name} tool for testing"
            large_tool_set.append(tool)

        with patch("src.agents.agent_factory.ReActAgent") as mock_react_agent_class:
            mock_react_agent_class.from_tools.return_value = MagicMock()

            result = create_agentic_rag_system(large_tool_set, mock_llm)

            assert result is not None
            call_kwargs = mock_react_agent_class.from_tools.call_args[1]
            assert len(call_kwargs["tools"]) == 6

    @pytest.mark.parametrize(
        "query_length",
        [10, 50, 100, 500, 1000],  # Different query lengths
    )
    def test_query_length_handling(self, mock_react_agent, query_length):
        """Test agent handling of various query lengths."""
        # Generate query of specified length
        base_query = "What is artificial intelligence and how does it work? "
        query = (base_query * (query_length // len(base_query) + 1))[:query_length]

        result = process_query_with_agent_system(mock_react_agent, query, "single")

        assert result == "Mock agent response with detailed analysis"
        mock_react_agent.chat.assert_called_once_with(query)

    def test_concurrent_query_simulation(self, mock_react_agent):
        """Test that agent can handle multiple queries (sequential simulation)."""
        queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain neural networks",
            "What is natural language processing?",
            "Describe computer vision",
        ]

        results = []
        for query in queries:
            result = process_query_with_agent_system(mock_react_agent, query, "single")
            results.append(result)

        # All queries should succeed
        assert len(results) == 5
        assert all(r == "Mock agent response with detailed analysis" for r in results)
        assert mock_react_agent.chat.call_count == 5


class TestMemoryManagement:
    """Test memory-related functionality."""

    @patch("src.agents.agent_factory.ReActAgent")
    @patch("src.agents.agent_factory.ChatMemoryBuffer")
    def test_default_memory_configuration(
        self,
        mock_memory_class,
        mock_react_agent_class,
        mock_llm,
        mock_query_engine_tools,
    ):
        """Test that default memory is created with correct token limit."""
        mock_memory_instance = MagicMock()
        mock_memory_class.from_defaults.return_value = mock_memory_instance
        mock_react_agent_class.from_tools.return_value = MagicMock()

        create_agentic_rag_system(mock_query_engine_tools, mock_llm)

        # Verify memory was created with correct token limit
        mock_memory_class.from_defaults.assert_called_once_with(token_limit=8192)

        # Verify memory was passed to agent
        call_kwargs = mock_react_agent_class.from_tools.call_args[1]
        assert call_kwargs["memory"] == mock_memory_instance

    @patch("src.agents.agent_factory.ReActAgent")
    def test_custom_memory_usage(
        self, mock_react_agent_class, mock_llm, mock_query_engine_tools, mock_memory
    ):
        """Test that custom memory is used when provided."""
        mock_react_agent_class.from_tools.return_value = MagicMock()

        result = create_agentic_rag_system(
            mock_query_engine_tools, mock_llm, mock_memory
        )

        assert result is not None
        call_kwargs = mock_react_agent_class.from_tools.call_args[1]
        assert call_kwargs["memory"] == mock_memory

    def test_memory_parameter_ignored_in_query_processing(self, mock_react_agent):
        """Test that memory parameter in process_query is ignored (no longer used)."""
        unused_memory = MagicMock()
        query = "Test query"

        result = process_query_with_agent_system(
            mock_react_agent, query, "single", memory=unused_memory
        )

        # Should work normally and ignore the memory parameter
        assert result == "Mock agent response with detailed analysis"
        mock_react_agent.chat.assert_called_once_with(query)
