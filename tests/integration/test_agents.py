"""Tests for single ReActAgent integration.

This module tests the simplified ReActAgent system integration with tools
and query processing following 2025 KISS principles.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from unittest.mock import MagicMock, patch

import pytest

from src.agents.agent_factory import get_agent_system, process_query_with_agent_system


class TestReActAgentIntegration:
    """Test single ReActAgent system integration."""

    def test_agent_system_creation(self):
        """Test that agent system creates successfully with minimal setup."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_llm = MagicMock()

        with patch("src.agents.agent_factory.ReActAgent") as mock_react_agent:
            mock_agent = MagicMock()
            mock_react_agent.from_tools.return_value = mock_agent

            agent, mode = get_agent_system(mock_tools, mock_llm)

            assert agent == mock_agent
            assert mode == "single"
            mock_react_agent.from_tools.assert_called_once()

    def test_query_processing_integration(self):
        """Test end-to-end query processing."""
        mock_agent = MagicMock()
        mock_response = MagicMock()
        mock_response.response = "Test response from ReActAgent"
        mock_agent.chat.return_value = mock_response

        result = process_query_with_agent_system(
            mock_agent, "What is machine learning?", "single"
        )

        assert result == "Test response from ReActAgent"
        mock_agent.chat.assert_called_once_with("What is machine learning?")

    def test_tool_integration_workflow(self):
        """Test that tools are properly integrated into agent workflow."""
        mock_tools = [
            MagicMock(metadata=MagicMock(name="semantic_search")),
            MagicMock(metadata=MagicMock(name="keyword_search")),
        ]
        mock_llm = MagicMock()

        with patch("src.agents.agent_factory.ReActAgent") as mock_react_agent:
            mock_agent = MagicMock()
            mock_react_agent.from_tools.return_value = mock_agent

            agent, mode = get_agent_system(mock_tools, mock_llm)

            # Verify tools were passed to ReActAgent
            call_args = mock_react_agent.from_tools.call_args
            assert call_args[1]["tools"] == mock_tools
            assert call_args[1]["llm"] == mock_llm
            assert call_args[1]["verbose"] is True
            assert call_args[1]["max_iterations"] == 3

    @pytest.mark.performance
    def test_agent_performance_characteristics(self, benchmark):
        """Test that agent creation and query processing meet performance targets."""
        mock_tools = [MagicMock(), MagicMock()]
        mock_llm = MagicMock()

        with patch("src.agents.agent_factory.ReActAgent") as mock_react_agent:
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.response = "Fast response"
            mock_agent.chat.return_value = mock_response
            mock_react_agent.from_tools.return_value = mock_agent

            def create_and_query():
                agent, mode = get_agent_system(mock_tools, mock_llm)
                return process_query_with_agent_system(agent, "test query", mode)

            result = benchmark(create_and_query)
            assert result == "Fast response"

    def test_error_handling_integration(self):
        """Test error handling in integrated system."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        with patch("src.agents.agent_factory.ReActAgent") as mock_react_agent:
            # Test agent creation failure
            mock_react_agent.from_tools.side_effect = Exception("Agent creation failed")

            with pytest.raises(Exception, match="Agent creation failed"):
                get_agent_system(mock_tools, mock_llm)

    def test_memory_integration(self):
        """Test memory buffer integration."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()
        mock_memory = MagicMock()

        with patch("src.agents.agent_factory.ReActAgent") as mock_react_agent:
            mock_agent = MagicMock()
            mock_react_agent.from_tools.return_value = mock_agent

            agent, mode = get_agent_system(mock_tools, mock_llm, memory=mock_memory)

            # Verify memory was passed to agent
            call_args = mock_react_agent.from_tools.call_args
            assert call_args[1]["memory"] == mock_memory
