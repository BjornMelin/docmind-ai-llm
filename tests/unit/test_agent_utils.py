"""Tests for agent utilities functionality.

This module provides test coverage for agent utilities, focusing on:
- Tool creation from index data
- ReActAgent creation and configuration
- Agentic document analysis
- Async chat capabilities
- Error handling and resilience

Aligned with the simplified architecture using src.agents.agent_utils.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama

from src.agents.agent_utils import (
    analyze_documents_agentic,
    chat_with_agent,
    create_agent_with_tools,
    create_tools_from_index,
)
from src.models.core import Settings as AppSettings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return AppSettings(
        reranker_model="jinaai/jina-reranker-v2-base-multilingual",
        reranking_top_k=5,
        rrf_fusion_alpha=60,
        default_model="google/gemma-3n-E4B-it",
    )


@pytest.fixture
def mock_vector_index():
    """Create a mock vector index for testing."""
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_index.as_query_engine.return_value = mock_query_engine
    return mock_index


@pytest.fixture
def mock_kg_index():
    """Create a mock knowledge graph index for testing."""
    mock_index = MagicMock()
    mock_query_engine = MagicMock()
    mock_index.as_query_engine.return_value = mock_query_engine
    return mock_index


@pytest.fixture
def mock_hybrid_retriever():
    """Create a mock hybrid retriever for testing."""
    return MagicMock()


@pytest.fixture
def mock_index_data_complete(mock_vector_index, mock_kg_index, mock_hybrid_retriever):
    """Create complete index data with all components."""
    return {
        "vector": mock_vector_index,
        "kg": mock_kg_index,
        "retriever": mock_hybrid_retriever,
    }


@pytest.fixture
def mock_index_data_vector_only(mock_vector_index):
    """Create index data with vector index only."""
    return {
        "vector": mock_vector_index,
        "kg": None,
        "retriever": None,
    }


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock_llm = MagicMock(spec=Ollama)
    mock_llm.model = "google/gemma-3n-E4B-it"
    return mock_llm


@pytest.fixture
def mock_agent():
    """Create a mock ReActAgent for testing."""
    mock_agent = MagicMock(spec=ReActAgent)
    mock_response = MagicMock()
    mock_response.response = "Mock agent response"
    mock_agent.chat.return_value = mock_response

    # Mock async stream chat
    mock_async_response = MagicMock()
    mock_async_response.async_response_gen = AsyncMock(
        return_value=["Mock ", "streaming ", "response"]
    )
    mock_agent.async_stream_chat = AsyncMock(return_value=mock_async_response)

    return mock_agent


class TestCreateToolsFromIndex:
    """Test tool creation from indexes."""

    @patch("src.agents.agent_utils.ToolFactory")
    def test_create_tools_with_complete_data(
        self, mock_tool_factory, mock_index_data_complete
    ):
        """Test tool creation with complete index data."""
        # Mock ToolFactory.create_basic_tools
        mock_tools = [MagicMock(), MagicMock()]
        mock_tool_factory.create_basic_tools.return_value = mock_tools

        tools = create_tools_from_index(mock_index_data_complete)

        assert len(tools) == 2
        mock_tool_factory.create_basic_tools.assert_called_once_with(
            mock_index_data_complete
        )

    @patch("src.agents.agent_utils.ToolFactory")
    def test_create_tools_with_vector_only(
        self, mock_tool_factory, mock_index_data_vector_only
    ):
        """Test tool creation with vector index only."""
        mock_tools = [MagicMock()]
        mock_tool_factory.create_basic_tools.return_value = mock_tools

        tools = create_tools_from_index(mock_index_data_vector_only)

        assert len(tools) == 1
        mock_tool_factory.create_basic_tools.assert_called_once_with(
            mock_index_data_vector_only
        )

    def test_create_tools_empty_index_data(self):
        """Test tool creation with empty index data - uses fallback."""
        # Should trigger fallback and return empty list
        tools = create_tools_from_index({})
        assert tools == []

    def test_create_tools_none_data(self):
        """Test tool creation with None index data - uses fallback."""
        # Should trigger fallback and return empty list
        tools = create_tools_from_index(None)
        assert tools == []


class TestCreateAgentWithTools:
    """Test ReActAgent creation with tools."""

    @patch("src.agents.agent_utils.create_tools_from_index")
    @patch("src.agents.agent_utils.ReActAgent")
    @patch("src.agents.agent_utils.ChatMemoryBuffer")
    def test_create_agent_success(
        self,
        mock_memory_buffer,
        mock_react_agent,
        mock_create_tools,
        mock_llm,
        mock_index_data_complete,
    ):
        """Test successful agent creation."""
        # Mock tools and agent
        mock_tools = [MagicMock(), MagicMock()]
        mock_create_tools.return_value = mock_tools
        mock_agent_instance = MagicMock()
        mock_react_agent.from_tools.return_value = mock_agent_instance
        mock_memory_instance = MagicMock()
        mock_memory_buffer.from_defaults.return_value = mock_memory_instance

        result = create_agent_with_tools(mock_index_data_complete, mock_llm)

        assert result == mock_agent_instance
        mock_create_tools.assert_called_once_with(mock_index_data_complete)
        mock_react_agent.from_tools.assert_called_once()

    @patch("src.agents.agent_utils.create_tools_from_index")
    @patch("src.agents.agent_utils.ReActAgent")
    def test_create_agent_empty_tools(
        self, mock_react_agent, mock_create_tools, mock_llm, mock_index_data_complete
    ):
        """Test agent creation with empty tools."""
        mock_create_tools.return_value = []
        mock_agent_instance = MagicMock()
        mock_react_agent.from_tools.return_value = mock_agent_instance

        result = create_agent_with_tools(mock_index_data_complete, mock_llm)

        assert result == mock_agent_instance
        mock_create_tools.assert_called_once_with(mock_index_data_complete)

    def test_create_agent_fallback_on_failure(self, mock_llm):
        """Test agent creation fallback on failure."""
        # Should use fallback decorator and return basic agent
        result = create_agent_with_tools({}, mock_llm)

        # Fallback should create a basic ReActAgent
        assert result is not None


class TestAnalyzeDocumentsAgentic:
    """Test agentic document analysis."""

    def test_analyze_documents_with_agent(self, mock_agent, mock_index_data_complete):
        """Test document analysis with provided agent."""
        result = analyze_documents_agentic(
            mock_agent, mock_index_data_complete, "summary"
        )

        assert "Mock agent response" in result
        mock_agent.chat.assert_called_once()

    @patch("src.agents.agent_utils.ReActAgent")
    @patch("src.agents.agent_utils.Ollama")
    def test_analyze_documents_without_agent(
        self, mock_ollama, mock_react_agent, mock_index_data_complete
    ):
        """Test document analysis without provided agent - creates fallback."""
        mock_agent_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.response = "Fallback analysis"
        mock_agent_instance.chat.return_value = mock_response
        mock_react_agent.from_tools.return_value = mock_agent_instance

        result = analyze_documents_agentic(None, mock_index_data_complete, "summary")

        assert "Fallback analysis" in result

    def test_analyze_documents_fallback_on_failure(self, mock_index_data_complete):
        """Test analysis fallback on failure."""
        # Should use fallback decorator
        result = analyze_documents_agentic(None, {}, "summary")

        assert "Analysis failed for prompt type: summary" in result


class TestChatWithAgent:
    """Test async chat with agent."""

    @pytest.mark.asyncio
    async def test_chat_with_agent_success(self, mock_agent):
        """Test successful chat with agent."""
        from llama_index.core.memory import ChatMemoryBuffer

        memory = ChatMemoryBuffer.from_defaults()
        chunks = []

        async for chunk in chat_with_agent(mock_agent, "Test query", memory):
            chunks.append(chunk)

        assert len(chunks) > 0
        mock_agent.async_stream_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_empty_input(self, mock_agent):
        """Test chat with empty input."""
        from llama_index.core.memory import ChatMemoryBuffer

        memory = ChatMemoryBuffer.from_defaults()

        with pytest.raises(RuntimeError):
            async for chunk in chat_with_agent(mock_agent, "", memory):
                pass

    @pytest.mark.asyncio
    async def test_chat_with_agent_failure(self, mock_agent):
        """Test chat handling when agent fails."""
        from llama_index.core.memory import ChatMemoryBuffer

        memory = ChatMemoryBuffer.from_defaults()
        mock_agent.async_stream_chat.side_effect = Exception("Agent failed")

        chunks = []
        async for chunk in chat_with_agent(mock_agent, "Test query", memory):
            chunks.append(chunk)

        # Should yield error message as fallback
        assert len(chunks) > 0
        assert "Chat processing encountered an error" in chunks[0]


class TestErrorHandling:
    """Test error handling and fallback mechanisms."""

    def test_create_tools_with_import_error(self, mock_index_data_complete):
        """Test tool creation when ToolFactory import fails."""
        with patch("src.agents.agent_utils.ToolFactory", side_effect=ImportError):
            # Should trigger fallback and return empty list
            tools = create_tools_from_index(mock_index_data_complete)
            assert tools == []

    def test_create_agent_with_tool_creation_failure(self, mock_llm):
        """Test agent creation when tool creation fails."""
        with patch(
            "src.agents.agent_utils.create_tools_from_index",
            side_effect=Exception("Tool creation failed"),
        ):
            # Should use fallback and return basic agent
            result = create_agent_with_tools({}, mock_llm)
            assert result is not None

    def test_analyze_with_query_engine_failure(self, mock_agent):
        """Test analysis when query engine creation fails."""
        # Mock index data with failing query engine
        mock_index = MagicMock()
        mock_index.as_query_engine.side_effect = Exception("Query engine failed")
        index_data = {"vector": mock_index}

        result = analyze_documents_agentic(mock_agent, index_data, "summary")

        # Should still complete analysis despite query engine failure
        assert "Mock agent response" in result
