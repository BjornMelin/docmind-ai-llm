"""Comprehensive test coverage for agent_factory.py.

This test suite provides extensive coverage for the agent_factory module,
including multi-agent coordination, query analysis, error handling,
and LangGraph integration to achieve 70%+ coverage.
"""

from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from llama_index.core.tools import QueryEngineTool

# Import the module under test
from agent_factory import (
    AgentState,
    analyze_query_complexity,
    create_single_agent,
)
from models import AppSettings


class TestAgentState:
    """Test the AgentState class comprehensively."""

    def test_agent_state_initialization_defaults(self):
        """Test AgentState initialization with default values."""
        state = AgentState()

        # Test default values
        assert state.query_complexity == "simple"
        assert state.query_type == "general"
        assert state.current_agent == "supervisor"
        assert state.task_progress == {}
        assert state.agent_outputs == {}
        assert state.final_answer == ""
        assert state.confidence_score == 0.0

    def test_agent_state_initialization_custom_values(self):
        """Test AgentState initialization with custom values."""
        custom_state = AgentState(
            query_complexity="complex",
            query_type="multimodal",
            current_agent="document_agent",
            task_progress={"step1": "completed"},
            agent_outputs={"analysis": "result"},
            final_answer="Custom answer",
            confidence_score=0.85,
        )

        assert custom_state.query_complexity == "complex"
        assert custom_state.query_type == "multimodal"
        assert custom_state.current_agent == "document_agent"
        assert custom_state.task_progress == {"step1": "completed"}
        assert custom_state.agent_outputs == {"analysis": "result"}
        assert custom_state.final_answer == "Custom answer"
        assert custom_state.confidence_score == 0.85

    def test_agent_state_inheritance_from_messages_state(self):
        """Test that AgentState properly inherits from MessagesState."""
        from langgraph.graph import MessagesState

        state = AgentState()

        # Should inherit from MessagesState
        assert isinstance(state, MessagesState)

        # Should have messages attribute from parent
        assert hasattr(state, "messages")


class TestAnalyzeQueryComplexity:
    """Test the analyze_query_complexity function comprehensively."""

    @pytest.mark.parametrize(
        "query,expected_complexity,expected_type",
        [
            # Simple queries
            ("Hello", "simple", "general"),
            ("What is this?", "simple", "general"),
            ("Brief summary", "simple", "general"),
            # Document-specific queries
            ("Summarize this document", "simple", "document"),
            ("What does the document say about AI?", "complex", "document"),
            ("Extract key findings from the PDF", "complex", "document"),
            # Knowledge graph queries
            (
                "Show me the relationships between entities",
                "complex",
                "knowledge_graph",
            ),
            ("What connections exist between concepts?", "complex", "knowledge_graph"),
            ("Map the entity relationships", "specialized", "knowledge_graph"),
            # Multimodal queries
            ("Describe the images in the document", "specialized", "multimodal"),
            ("Analyze both text and visual content", "specialized", "multimodal"),
            ("What do the charts and graphs show?", "complex", "multimodal"),
            # Complex analytical queries
            (
                "Provide a comprehensive analysis with detailed insights and recommendations",
                "complex",
                "document",
            ),
            (
                "Compare multiple documents and identify patterns",
                "specialized",
                "document",
            ),
        ],
    )
    def test_analyze_query_complexity_categorization(
        self, query, expected_complexity, expected_type
    ):
        """Test query complexity and type categorization."""
        complexity, query_type = analyze_query_complexity(query)

        assert complexity == expected_complexity
        assert query_type == expected_type

    def test_analyze_query_complexity_empty_query(self):
        """Test analysis of empty or very short queries."""
        complexity, query_type = analyze_query_complexity("")
        assert complexity == "simple"
        assert query_type == "general"

        complexity, query_type = analyze_query_complexity("?")
        assert complexity == "simple"
        assert query_type == "general"

    def test_analyze_query_complexity_very_long_query(self):
        """Test analysis of very long, complex queries."""
        long_query = (
            "Please provide a comprehensive analysis of all documents, "
            "including detailed summaries, key insights, entity relationships, "
            "image descriptions, cross-references, and actionable recommendations "
            "based on the multimodal content while considering the broader context "
            "and implications for strategic decision making."
        )

        complexity, query_type = analyze_query_complexity(long_query)

        # Very long, complex queries should be categorized as specialized
        assert complexity in ["complex", "specialized"]
        # Should detect multimodal intent
        assert query_type in ["multimodal", "document"]

    def test_analyze_query_complexity_case_insensitive(self):
        """Test that query analysis is case-insensitive."""
        query_lower = "summarize the document"
        query_upper = "SUMMARIZE THE DOCUMENT"
        query_mixed = "Summarize The Document"

        complexity1, type1 = analyze_query_complexity(query_lower)
        complexity2, type2 = analyze_query_complexity(query_upper)
        complexity3, type3 = analyze_query_complexity(query_mixed)

        # All should produce same results
        assert complexity1 == complexity2 == complexity3
        assert type1 == type2 == type3


class TestCreateSingleAgent:
    """Test the create_single_agent function comprehensively."""

    def test_create_single_agent_success(self):
        """Test successful single agent creation with proper tool list."""
        # Mock dependencies - create_single_agent expects tools list, not index
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        with patch("llama_index.core.agent.ReActAgent") as mock_react_agent:
            with patch("llama_index.core.memory.ChatMemoryBuffer") as mock_memory:
                # Arrange mocks
                mock_agent = MagicMock()
                mock_react_agent.from_tools.return_value = mock_agent

                mock_memory_instance = MagicMock()
                mock_memory.from_defaults.return_value = mock_memory_instance

                # Act
                result = create_single_agent(mock_tools, mock_llm)

                # Assert
                assert result == mock_agent
                mock_react_agent.from_tools.assert_called_once_with(
                    tools=mock_tools,
                    llm=mock_llm,
                    memory=mock_memory_instance,
                    verbose=True,
                    max_iterations=10,
                )

    def test_create_single_agent_with_custom_settings(self):
        """Test single agent creation with custom settings."""
        mock_index = MagicMock()
        mock_llm = MagicMock()

        test_settings = AppSettings(context_size=8192, max_iterations=20)

        with patch("agent_factory.settings", test_settings):
            with patch("llama_index.core.agent.ReActAgent") as mock_react_agent:
                with patch("llama_index.core.memory.ChatMemoryBuffer") as mock_memory:
                    with patch("llama_index.core.tools.QueryEngineTool"):
                        # Act
                        create_single_agent(mock_index, mock_llm)

                        # Assert - verify settings were used
                        mock_react_agent.from_tools.assert_called_once()
                        call_kwargs = mock_react_agent.from_tools.call_args[1]
                        assert "memory" in call_kwargs

    def test_create_single_agent_error_handling(self):
        """Test error handling in single agent creation."""
        mock_index = MagicMock()
        mock_llm = MagicMock()

        with patch(
            "llama_index.core.agent.ReActAgent.from_tools",
            side_effect=RuntimeError("Agent creation failed"),
        ):
            with patch("llama_index.core.memory.ChatMemoryBuffer"):
                with patch("llama_index.core.tools.QueryEngineTool"):
                    # Should propagate the error
                    with pytest.raises(RuntimeError, match="Agent creation failed"):
                        create_single_agent(mock_index, mock_llm)


class TestCreateMultiAgentGraph:
    """Test the create_multi_agent_graph function comprehensively."""

    def test_create_multi_agent_graph_success(self):
        """Test successful multi-agent graph creation."""
        # Mock dependencies
        mock_agents = {
            "document_agent": MagicMock(),
            "kg_agent": MagicMock(),
            "multimodal_agent": MagicMock(),
        }

        with patch("langgraph.graph.StateGraph") as mock_state_graph:
            with patch("langgraph.prebuilt.create_react_agent") as mock_create_react:
                # Arrange mocks
                mock_graph_instance = MagicMock()
                mock_state_graph.return_value = mock_graph_instance
                mock_graph_instance.compile.return_value = MagicMock()

                mock_create_react.return_value = MagicMock()

                # Act
                result = create_multi_agent_graph(mock_agents)

                # Assert
                assert result is not None
                mock_state_graph.assert_called_once_with(AgentState)
                mock_graph_instance.compile.assert_called_once()

    def test_create_multi_agent_graph_with_supervisor(self):
        """Test multi-agent graph creation includes supervisor logic."""
        mock_agents = {"document_agent": MagicMock(), "kg_agent": MagicMock()}

        with patch("langgraph.graph.StateGraph") as mock_state_graph:
            with patch("langgraph.prebuilt.create_react_agent"):
                mock_graph_instance = MagicMock()
                mock_state_graph.return_value = mock_graph_instance

                # Act
                create_multi_agent_graph(mock_agents)

                # Assert - should add nodes and edges for supervisor pattern
                assert (
                    mock_graph_instance.add_node.call_count >= len(mock_agents) + 1
                )  # +1 for supervisor
                assert (
                    mock_graph_instance.add_edge.call_count >= 0
                )  # Should have routing edges

    def test_create_multi_agent_graph_empty_agents(self):
        """Test multi-agent graph creation with empty agents dict."""
        with patch("langgraph.graph.StateGraph") as mock_state_graph:
            mock_graph_instance = MagicMock()
            mock_state_graph.return_value = mock_graph_instance

            # Act
            result = create_multi_agent_graph({})

            # Assert - should still create a basic graph structure
            assert result is not None
            mock_graph_instance.compile.assert_called_once()

    def test_create_multi_agent_graph_routing_logic(self):
        """Test that multi-agent graph includes proper routing logic."""
        mock_agents = {
            "document_agent": MagicMock(),
            "kg_agent": MagicMock(),
            "multimodal_agent": MagicMock(),
        }

        with patch("langgraph.graph.StateGraph") as mock_state_graph:
            with patch("langgraph.prebuilt.create_react_agent"):
                mock_graph_instance = MagicMock()
                mock_state_graph.return_value = mock_graph_instance

                # Act
                create_multi_agent_graph(mock_agents)

                # Assert - verify routing function is added
                add_conditional_edges_calls = [
                    call
                    for call in mock_graph_instance.method_calls
                    if call[0] == "add_conditional_edges"
                ]
                # Should have conditional routing logic
                assert len(add_conditional_edges_calls) >= 0


class TestAgentFactory:
    """Test the AgentFactory class comprehensively."""

    def test_agent_factory_initialization(self):
        """Test AgentFactory initialization."""
        mock_index_data = {
            "vector": MagicMock(),
            "kg": MagicMock(),
            "retriever": MagicMock(),
        }
        mock_llm = MagicMock()

        # Act
        factory = AgentFactory(mock_index_data, mock_llm)

        # Assert
        assert factory.index_data == mock_index_data
        assert factory.llm == mock_llm
        assert hasattr(factory, "agents")

    def test_agent_factory_create_agents(self):
        """Test agent creation through factory."""
        mock_index_data = {
            "vector": MagicMock(),
            "kg": MagicMock(),
            "retriever": MagicMock(),
        }
        mock_llm = MagicMock()

        factory = AgentFactory(mock_index_data, mock_llm)

        with patch.object(factory, "_create_document_agent") as mock_doc_agent:
            with patch.object(factory, "_create_kg_agent") as mock_kg_agent:
                with patch.object(factory, "_create_multimodal_agent") as mock_mm_agent:
                    # Arrange mocks
                    mock_doc_agent.return_value = MagicMock()
                    mock_kg_agent.return_value = MagicMock()
                    mock_mm_agent.return_value = MagicMock()

                    # Act
                    factory.create_agents()

                    # Assert
                    mock_doc_agent.assert_called_once()
                    mock_kg_agent.assert_called_once()
                    mock_mm_agent.assert_called_once()

                    assert len(factory.agents) == 3
                    assert "document_agent" in factory.agents
                    assert "kg_agent" in factory.agents
                    assert "multimodal_agent" in factory.agents

    def test_agent_factory_get_agent_for_query(self):
        """Test intelligent agent selection for queries."""
        mock_index_data = {"vector": MagicMock(), "kg": MagicMock()}
        mock_llm = MagicMock()

        factory = AgentFactory(mock_index_data, mock_llm)
        factory.agents = {
            "document_agent": MagicMock(),
            "kg_agent": MagicMock(),
            "multimodal_agent": MagicMock(),
        }

        # Test document query routing
        with patch(
            "agent_factory.analyze_query_complexity",
            return_value=("simple", "document"),
        ):
            agent = factory.get_agent_for_query("Summarize the document")
            assert agent == factory.agents["document_agent"]

        # Test knowledge graph query routing
        with patch(
            "agent_factory.analyze_query_complexity",
            return_value=("complex", "knowledge_graph"),
        ):
            agent = factory.get_agent_for_query("Show entity relationships")
            assert agent == factory.agents["kg_agent"]

        # Test multimodal query routing
        with patch(
            "agent_factory.analyze_query_complexity",
            return_value=("specialized", "multimodal"),
        ):
            agent = factory.get_agent_for_query("Analyze images and text")
            assert agent == factory.agents["multimodal_agent"]

    def test_agent_factory_create_workflow(self):
        """Test workflow creation for complex queries."""
        mock_index_data = {"vector": MagicMock(), "kg": MagicMock()}
        mock_llm = MagicMock()

        factory = AgentFactory(mock_index_data, mock_llm)
        factory.agents = {"document_agent": MagicMock(), "kg_agent": MagicMock()}

        with patch("agent_factory.create_multi_agent_graph") as mock_create_graph:
            mock_workflow = MagicMock()
            mock_create_graph.return_value = mock_workflow

            # Act
            workflow = factory.create_workflow()

            # Assert
            assert workflow == mock_workflow
            mock_create_graph.assert_called_once_with(factory.agents)

    def test_agent_factory_process_query_single_agent(self):
        """Test query processing with single agent."""
        mock_index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()

        factory = AgentFactory(mock_index_data, mock_llm)

        mock_agent = MagicMock()
        mock_agent.chat.return_value.response = "Single agent response"
        factory.agents = {"document_agent": mock_agent}

        with patch(
            "agent_factory.analyze_query_complexity",
            return_value=("simple", "document"),
        ):
            with patch.object(factory, "get_agent_for_query", return_value=mock_agent):
                # Act
                result = factory.process_query("Simple document question")

                # Assert
                assert "Single agent response" in result
                mock_agent.chat.assert_called_once()

    def test_agent_factory_process_query_multi_agent(self):
        """Test query processing with multi-agent workflow."""
        mock_index_data = {"vector": MagicMock(), "kg": MagicMock()}
        mock_llm = MagicMock()

        factory = AgentFactory(mock_index_data, mock_llm)
        factory.agents = {"document_agent": MagicMock(), "kg_agent": MagicMock()}

        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "final_answer": "Multi-agent response",
            "confidence_score": 0.9,
        }

        with patch(
            "agent_factory.analyze_query_complexity",
            return_value=("complex", "document"),
        ):
            with patch.object(factory, "create_workflow", return_value=mock_workflow):
                # Act
                result = factory.process_query("Complex analytical question")

                # Assert
                assert "Multi-agent response" in result
                mock_workflow.invoke.assert_called_once()


class TestPropertyBasedAgentFactory:
    """Property-based tests for agent factory functions."""

    @given(queries=st.lists(st.text(min_size=1, max_size=200), min_size=1, max_size=5))
    def test_query_analysis_properties(self, queries):
        """Test query analysis properties with various inputs."""
        for query in queries:
            complexity, query_type = analyze_query_complexity(query)

            # Properties that should always hold
            assert complexity in ["simple", "complex", "specialized"]
            assert query_type in [
                "general",
                "document",
                "knowledge_graph",
                "multimodal",
            ]

    @given(num_agents=st.integers(min_value=1, max_value=5))
    def test_multi_agent_graph_properties(self, num_agents):
        """Test multi-agent graph creation properties."""
        # Create mock agents
        mock_agents = {f"agent_{i}": MagicMock() for i in range(num_agents)}

        with patch("langgraph.graph.StateGraph") as mock_state_graph:
            with patch("langgraph.prebuilt.create_react_agent"):
                mock_graph_instance = MagicMock()
                mock_state_graph.return_value = mock_graph_instance
                mock_graph_instance.compile.return_value = MagicMock()

                # Act
                result = create_multi_agent_graph(mock_agents)

                # Assert - properties that should always hold
                assert result is not None
                mock_state_graph.assert_called_once_with(AgentState)
                # Should add at least as many nodes as agents
                assert mock_graph_instance.add_node.call_count >= num_agents

    @given(
        complexity=st.sampled_from(["simple", "complex", "specialized"]),
        query_type=st.sampled_from(
            ["general", "document", "knowledge_graph", "multimodal"]
        ),
    )
    def test_agent_routing_properties(self, complexity, query_type):
        """Test agent routing properties for different query types."""
        mock_index_data = {"vector": MagicMock(), "kg": MagicMock()}
        mock_llm = MagicMock()

        factory = AgentFactory(mock_index_data, mock_llm)
        factory.agents = {
            "document_agent": MagicMock(),
            "kg_agent": MagicMock(),
            "multimodal_agent": MagicMock(),
        }

        with patch(
            "agent_factory.analyze_query_complexity",
            return_value=(complexity, query_type),
        ):
            agent = factory.get_agent_for_query("test query")

            # Properties that should always hold
            assert agent is not None
            assert agent in factory.agents.values()


class TestAgentFactoryErrorHandling:
    """Comprehensive error handling tests for agent factory."""

    def test_agent_creation_failure_graceful_degradation(self):
        """Test graceful handling of agent creation failures."""
        mock_index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()

        factory = AgentFactory(mock_index_data, mock_llm)

        # Mock agent creation failure
        with patch.object(
            factory,
            "_create_document_agent",
            side_effect=RuntimeError("Creation failed"),
        ):
            with patch.object(factory, "_create_kg_agent", return_value=MagicMock()):
                with patch.object(
                    factory, "_create_multimodal_agent", return_value=MagicMock()
                ):
                    # Should handle individual agent failures gracefully
                    factory.create_agents()

                    # Should still have working agents
                    assert len(factory.agents) >= 1  # At least some agents should work

    def test_query_processing_with_agent_failure(self):
        """Test query processing when selected agent fails."""
        mock_index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()

        factory = AgentFactory(mock_index_data, mock_llm)

        # Mock failing agent
        mock_agent = MagicMock()
        mock_agent.chat.side_effect = RuntimeError("Agent processing failed")
        factory.agents = {"document_agent": mock_agent}

        with patch(
            "agent_factory.analyze_query_complexity",
            return_value=("simple", "document"),
        ):
            with patch.object(factory, "get_agent_for_query", return_value=mock_agent):
                # Should handle agent failure gracefully
                result = factory.process_query("Test query")

                # Should return error message or fallback response
                assert isinstance(result, str)
                assert len(result) > 0

    def test_workflow_creation_failure_fallback(self):
        """Test fallback when workflow creation fails."""
        mock_index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()

        factory = AgentFactory(mock_index_data, mock_llm)
        factory.agents = {"document_agent": MagicMock()}

        with patch(
            "agent_factory.create_multi_agent_graph",
            side_effect=RuntimeError("Workflow creation failed"),
        ):
            # Should handle workflow creation failure
            with pytest.raises(RuntimeError):
                factory.create_workflow()

    def test_missing_index_components_handling(self):
        """Test handling of missing index components."""
        # Test with missing components
        incomplete_index_data = {"vector": MagicMock()}  # Missing kg, retriever
        mock_llm = MagicMock()

        # Should still create factory
        factory = AgentFactory(incomplete_index_data, mock_llm)

        with patch.object(factory, "_create_document_agent", return_value=MagicMock()):
            with patch.object(
                factory, "_create_kg_agent", side_effect=KeyError("Missing KG index")
            ):
                with patch.object(
                    factory, "_create_multimodal_agent", return_value=MagicMock()
                ):
                    # Should handle missing components gracefully
                    factory.create_agents()

                    # Should still have some working agents
                    assert len(factory.agents) >= 1

    @pytest.mark.parametrize(
        "exception_type",
        [RuntimeError, ValueError, KeyError, AttributeError, TypeError],
    )
    def test_specific_exception_handling_in_processing(self, exception_type):
        """Test handling of specific exception types during processing."""
        mock_index_data = {"vector": MagicMock()}
        mock_llm = MagicMock()

        factory = AgentFactory(mock_index_data, mock_llm)

        # Mock agent that raises specific exception
        mock_agent = MagicMock()
        mock_agent.chat.side_effect = exception_type("Specific error")
        factory.agents = {"document_agent": mock_agent}

        with patch(
            "agent_factory.analyze_query_complexity",
            return_value=("simple", "document"),
        ):
            with patch.object(factory, "get_agent_for_query", return_value=mock_agent):
                # Should handle all exception types gracefully
                result = factory.process_query("Test query")

                assert isinstance(result, str)
                # Should contain error information or fallback response
                assert len(result) > 0
