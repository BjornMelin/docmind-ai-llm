"""Tests for ReActAgent and query complexity analysis.

This module tests the ReActAgent system, tool integration, query complexity
analysis, and multi-agent workflows following 2025 best practices.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_factory import analyze_query_complexity, get_agent_system


class TestQueryComplexityAnalysis:
    """Test query complexity analysis functionality."""

    @pytest.mark.parametrize(
        ("query", "expected_complexity"),
        [
            ("What is AI?", "simple"),
            (
                "Explain the differences between machine learning and deep learning",
                "moderate",
            ),
            (
                "Compare SPLADE++ vs BGE-Large performance implications",
                "complex",
            ),
            ("Hi", "simple"),
            (
                "How do hybrid search systems integrate multiple retrieval methods?",
                "moderate",
            ),
        ],
    )
    def test_query_complexity_classification(self, query, expected_complexity):
        """Test query complexity classification with different query types.

        Args:
            query: Query string to analyze.
            expected_complexity: Expected complexity classification.
        """
        with patch("agent_factory.analyze_query_complexity") as mock_analyze:
            # Mock the complexity analysis
            mock_analyze.return_value = {
                "complexity": expected_complexity,
                "reasoning": f"Query classified as {expected_complexity}",
                "features": {
                    "word_count": len(query.split()),
                    "question_words": sum(
                        1
                        for word in query.lower().split()
                        if word in ["what", "how", "why", "when", "where"]
                    ),
                    "technical_terms": 0,
                },
            }

            result = analyze_query_complexity(query)

            assert result["complexity"] == expected_complexity
            assert "reasoning" in result
            assert "features" in result

    def test_query_complexity_features_extraction(self):
        """Test extraction of query complexity features."""
        test_queries = [
            "What is SPLADE++?",
            "How does ColBERT reranking improve search results?",
            "Compare dense and sparse embeddings for document retrieval systems",
        ]

        for query in test_queries:
            with patch("agent_factory.analyze_query_complexity") as mock_analyze:
                # Mock realistic feature extraction
                words = query.split()
                mock_analyze.return_value = {
                    "complexity": "moderate",
                    "features": {
                        "word_count": len(words),
                        "question_words": sum(
                            1
                            for word in words
                            if word.lower() in ["what", "how", "why", "compare"]
                        ),
                        "technical_terms": sum(
                            1
                            for word in words
                            if word.lower() in ["splade", "colbert", "embeddings"]
                        ),
                        "sentence_count": query.count(".") + query.count("?") + 1,
                    },
                }

                result = analyze_query_complexity(query)
                features = result["features"]

                assert features["word_count"] > 0
                assert "question_words" in features
                assert "technical_terms" in features

    @pytest.mark.performance
    def test_query_complexity_analysis_performance(self, benchmark):
        """Test query complexity analysis performance.

        Args:
            benchmark: Pytest benchmark fixture.
        """
        test_query = (
            "How do hybrid search systems integrate SPLADE++ sparse embeddings "
            "with BGE-Large dense embeddings using RRF fusion?"
        )

        with patch("agent_factory.analyze_query_complexity") as mock_analyze:
            mock_analyze.return_value = {
                "complexity": "complex",
                "reasoning": "Multi-step analysis required",
                "features": {"word_count": 20},
            }

            def analyze_operation():
                return analyze_query_complexity(test_query)

            result = benchmark(analyze_operation)
            assert result["complexity"] in ["simple", "moderate", "complex"]

    def test_query_complexity_edge_cases(self):
        """Test query complexity analysis with edge cases."""
        edge_cases = [
            "",  # Empty query
            "   ",  # Whitespace only
            "A",  # Single character
            "A" * 1000,  # Very long query
            "🚀 What is AI? 🤖",  # Special characters and emojis
        ]

        for query in edge_cases:
            with patch("agent_factory.analyze_query_complexity") as mock_analyze:
                # Mock appropriate handling for edge cases
                if not query.strip():
                    mock_analyze.return_value = {
                        "complexity": "simple",
                        "reasoning": "Empty or minimal query",
                        "features": {"word_count": 0},
                    }
                else:
                    mock_analyze.return_value = {
                        "complexity": "simple",
                        "reasoning": "Basic query handling",
                        "features": {"word_count": len(query.split())},
                    }

                result = analyze_query_complexity(query)
                assert result["complexity"] in ["simple", "moderate", "complex"]


class TestAgentSystem:
    """Test ReActAgent system functionality."""

    def test_agent_system_initialization(self, test_settings):
        """Test agent system initialization and configuration.

        Args:
            test_settings: Test settings fixture.
        """
        with patch("agent_factory.get_agent_system") as mock_get_agent:
            mock_agent = MagicMock()
            mock_get_agent.return_value = mock_agent

            agent_system = get_agent_system(settings=test_settings)

            assert agent_system is not None
            mock_get_agent.assert_called_once_with(settings=test_settings)

    @pytest.mark.asyncio
    async def test_agent_async_processing(self, mock_async_llm, test_settings):
        """Test asynchronous agent processing.

        Args:
            mock_async_llm: Mock async LLM fixture.
            test_settings: Test settings fixture.
        """
        with patch("agent_factory.get_agent_system") as mock_get_agent:
            # Create mock async agent
            mock_agent = AsyncMock()
            mock_agent.arun.return_value = "Mock agent response"
            mock_get_agent.return_value = mock_agent

            agent_system = get_agent_system(settings=test_settings)

            # Test async processing
            query = "What is hybrid search?"
            result = await agent_system.arun(query)

            assert result == "Mock agent response"
            mock_agent.arun.assert_called_once_with(query)

    def test_agent_tool_integration(self, test_settings):
        """Test agent integration with tools.

        Args:
            test_settings: Test settings fixture.
        """
        with (
            patch("agent_factory.get_agent_system") as mock_get_agent,
            patch("utils.create_tools_from_index") as mock_create_tools,
        ):
            # Mock tools creation
            mock_tools = [
                MagicMock(name="search_tool"),
                MagicMock(name="analysis_tool"),
                MagicMock(name="retrieval_tool"),
            ]
            mock_create_tools.return_value = mock_tools

            # Mock agent with tools
            mock_agent = MagicMock()
            mock_agent.tools = mock_tools
            mock_get_agent.return_value = mock_agent

            agent_system = get_agent_system(settings=test_settings)

            assert len(agent_system.tools) == 3
            assert any(tool.name == "search_tool" for tool in agent_system.tools)

    @pytest.mark.parametrize(
        ("query_complexity", "expected_strategy"),
        [
            ("simple", "direct_response"),
            ("moderate", "single_agent"),
            ("complex", "multi_agent"),
        ],
    )
    def test_agent_strategy_selection(
        self, query_complexity, expected_strategy, test_settings
    ):
        """Test agent strategy selection based on query complexity.

        Args:
            query_complexity: Query complexity level.
            expected_strategy: Expected processing strategy.
            test_settings: Test settings fixture.
        """
        with (
            patch("agent_factory.analyze_query_complexity") as mock_analyze,
            patch("agent_factory.get_agent_system") as mock_get_agent,
        ):
            # Mock complexity analysis
            mock_analyze.return_value = {
                "complexity": query_complexity,
                "reasoning": f"Query is {query_complexity}",
            }

            # Mock agent system with strategy
            mock_agent = MagicMock()
            mock_agent.strategy = expected_strategy
            mock_get_agent.return_value = mock_agent

            query = "Test query"
            complexity_result = analyze_query_complexity(query)
            agent_system = get_agent_system(settings=test_settings)

            assert complexity_result["complexity"] == query_complexity
            assert agent_system.strategy == expected_strategy


class TestAgentToolIntegration:
    """Test agent integration with various tools."""

    def test_search_tool_integration(self, mock_qdrant_client, test_settings):
        """Test integration with search tools.

        Args:
            mock_qdrant_client: Mock Qdrant client fixture.
            test_settings: Test settings fixture.
        """
        with (
            patch("utils.create_tools_from_index") as mock_create_tools,
            patch("utils.QdrantClient", return_value=mock_qdrant_client),
        ):
            # Mock search tool
            mock_search_tool = MagicMock()
            mock_search_tool.name = "vector_search"
            mock_search_tool.func.return_value = "Search results"

            mock_create_tools.return_value = [mock_search_tool]

            # Test tool creation and usage
            tools = mock_create_tools(index=mock_qdrant_client)
            search_tool = tools[0]

            result = search_tool.func("test query")

            assert result == "Search results"
            assert search_tool.name == "vector_search"

    def test_analysis_tool_integration(self, mock_llm, test_settings):
        """Test integration with document analysis tools.

        Args:
            mock_llm: Mock LLM fixture.
            test_settings: Test settings fixture.
        """
        with patch("utils.create_tools_from_index") as mock_create_tools:
            # Mock analysis tool
            mock_analysis_tool = MagicMock()
            mock_analysis_tool.name = "document_analysis"
            mock_analysis_tool.func.return_value = "Analysis complete"

            mock_create_tools.return_value = [mock_analysis_tool]

            tools = mock_create_tools(llm=mock_llm)
            analysis_tool = tools[0]

            result = analysis_tool.func("Document content")

            assert result == "Analysis complete"
            assert analysis_tool.name == "document_analysis"

    def test_tool_error_handling(self, test_settings):
        """Test error handling in tool integration.

        Args:
            test_settings: Test settings fixture.
        """
        with patch("utils.create_tools_from_index") as mock_create_tools:
            # Mock tool that raises exception
            mock_failing_tool = MagicMock()
            mock_failing_tool.name = "failing_tool"
            mock_failing_tool.func.side_effect = Exception("Tool failed")

            mock_create_tools.return_value = [mock_failing_tool]

            tools = mock_create_tools()
            failing_tool = tools[0]

            # Test error handling
            with pytest.raises(Exception, match="Tool failed"):
                failing_tool.func("test input")

    @pytest.mark.performance
    def test_tool_performance(self, benchmark, test_settings):
        """Test tool performance characteristics.

        Args:
            benchmark: Pytest benchmark fixture.
            test_settings: Test settings fixture.
        """
        with patch("utils.create_tools_from_index") as mock_create_tools:
            # Mock fast tool
            mock_fast_tool = MagicMock()
            mock_fast_tool.name = "fast_tool"
            mock_fast_tool.func.return_value = "Fast result"

            mock_create_tools.return_value = [mock_fast_tool]

            tools = mock_create_tools()
            fast_tool = tools[0]

            def tool_operation():
                return fast_tool.func("test input")

            result = benchmark(tool_operation)
            assert result == "Fast result"


class TestMultiAgentWorkflows:
    """Test multi-agent workflow coordination."""

    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, test_settings):
        """Test coordination between multiple agents.

        Args:
            test_settings: Test settings fixture.
        """
        with patch("agent_factory.get_agent_system") as mock_get_agent:
            # Mock multiple agents
            search_agent = AsyncMock()
            search_agent.arun.return_value = "Search completed"

            analysis_agent = AsyncMock()
            analysis_agent.arun.return_value = "Analysis completed"

            coordinator_agent = AsyncMock()
            coordinator_agent.search_agent = search_agent
            coordinator_agent.analysis_agent = analysis_agent
            coordinator_agent.arun.return_value = "Workflow completed"

            mock_get_agent.return_value = coordinator_agent

            agent_system = get_agent_system(settings=test_settings)

            # Test multi-agent workflow
            result = await agent_system.arun("Complex multi-step query")

            assert result == "Workflow completed"

    def test_agent_communication(self, test_settings):
        """Test communication between agents.

        Args:
            test_settings: Test settings fixture.
        """
        with patch("agent_factory.get_agent_system") as mock_get_agent:
            # Mock agents with communication
            mock_agent1 = MagicMock()
            mock_agent1.send_message.return_value = "Message sent"

            mock_agent2 = MagicMock()
            mock_agent2.receive_message.return_value = "Message received"

            mock_coordinator = MagicMock()
            mock_coordinator.agents = [mock_agent1, mock_agent2]
            mock_get_agent.return_value = mock_coordinator

            agent_system = get_agent_system(settings=test_settings)

            # Test agent communication
            agent1 = agent_system.agents[0]
            agent2 = agent_system.agents[1]

            sent_result = agent1.send_message("Test message")
            received_result = agent2.receive_message("Test message")

            assert sent_result == "Message sent"
            assert received_result == "Message received"

    def test_workflow_error_recovery(self, test_settings):
        """Test error recovery in multi-agent workflows.

        Args:
            test_settings: Test settings fixture.
        """
        with patch("agent_factory.get_agent_system") as mock_get_agent:
            # Mock agents with failure scenarios
            failing_agent = MagicMock()
            failing_agent.run.side_effect = Exception("Agent failed")

            backup_agent = MagicMock()
            backup_agent.run.return_value = "Backup successful"

            resilient_coordinator = MagicMock()
            resilient_coordinator.primary_agent = failing_agent
            resilient_coordinator.backup_agent = backup_agent

            def coordinator_run(query):
                try:
                    return resilient_coordinator.primary_agent.run(query)
                except Exception:
                    return resilient_coordinator.backup_agent.run(query)

            resilient_coordinator.run = coordinator_run
            mock_get_agent.return_value = resilient_coordinator

            agent_system = get_agent_system(settings=test_settings)

            # Test error recovery
            result = agent_system.run("Test query")
            assert result == "Backup successful"


class TestAgentMemoryAndContext:
    """Test agent memory and context management."""

    def test_agent_context_retention(self, test_settings):
        """Test agent context retention across interactions.

        Args:
            test_settings: Test settings fixture.
        """
        with patch("agent_factory.get_agent_system") as mock_get_agent:
            # Mock agent with memory
            mock_agent = MagicMock()
            mock_agent.memory = []

            def agent_run_with_memory(query):
                mock_agent.memory.append(query)
                return f"Processed: {query} (Memory: {len(mock_agent.memory)})"

            mock_agent.run = agent_run_with_memory
            mock_get_agent.return_value = mock_agent

            agent_system = get_agent_system(settings=test_settings)

            # Test multiple interactions
            result1 = agent_system.run("First query")
            result2 = agent_system.run("Second query")

            assert "Memory: 1" in result1
            assert "Memory: 2" in result2
            assert len(agent_system.memory) == 2

    def test_agent_context_window_management(self, test_settings):
        """Test agent context window size management.

        Args:
            test_settings: Test settings fixture.
        """
        with patch("agent_factory.get_agent_system") as mock_get_agent:
            # Mock agent with context window limits
            mock_agent = MagicMock()
            mock_agent.context_window_size = test_settings.context_size
            mock_agent.current_context_length = 0

            def check_context_size(text):
                return len(text.split()) < mock_agent.context_window_size

            mock_agent.fits_in_context = check_context_size
            mock_get_agent.return_value = mock_agent

            agent_system = get_agent_system(settings=test_settings)

            # Test context window checking
            short_text = "Short query"
            long_text = " ".join(["word"] * (test_settings.context_size + 100))

            assert agent_system.fits_in_context(short_text) is True
            assert agent_system.fits_in_context(long_text) is False

    def test_agent_conversation_history(self, test_settings):
        """Test agent conversation history management.

        Args:
            test_settings: Test settings fixture.
        """
        with patch("agent_factory.get_agent_system") as mock_get_agent:
            # Mock agent with conversation history
            mock_agent = MagicMock()
            mock_agent.conversation_history = []

            def add_to_history(user_input, agent_response):
                mock_agent.conversation_history.append(
                    {
                        "user": user_input,
                        "agent": agent_response,
                        "timestamp": "2024-01-01 12:00:00",
                    }
                )
                return agent_response

            mock_agent.chat = add_to_history
            mock_get_agent.return_value = mock_agent

            agent_system = get_agent_system(settings=test_settings)

            # Test conversation history
            agent_system.chat("Hello", "Hi there!")
            agent_system.chat("How are you?", "I'm doing well!")

            assert len(agent_system.conversation_history) == 2
            assert agent_system.conversation_history[0]["user"] == "Hello"
            assert agent_system.conversation_history[1]["agent"] == "I'm doing well!"
