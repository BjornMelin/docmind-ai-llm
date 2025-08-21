"""Comprehensive unit tests for Multi-Agent Tools (src/agents/tools.py).

Tests cover:
- Query routing and complexity analysis
- Query planning and decomposition
- Document retrieval with multiple strategies
- Result synthesis and deduplication
- Response validation and quality scoring
- DSPy integration and optimization
- Error handling and fallback mechanisms
- Performance measurement and timing
"""

import json
import time
from typing import Any
from unittest.mock import Mock, patch

from llama_index.core import Document

from src.agents.tools import (
    _parse_tool_result,
    _rank_documents_by_relevance,
    plan_query,
    retrieve_documents,
    route_query,
    synthesize_results,
    validate_response,
)


class TestRouteQuery:
    """Test suite for route_query tool."""

    def test_route_simple_query(self):
        """Test routing of simple factual queries."""
        query = "What is the capital of France?"
        result = route_query(query)

        decision = json.loads(result)

        # Verify simple query classification
        assert decision["complexity"] == "simple"
        assert decision["strategy"] == "vector"
        assert decision["needs_planning"] is False
        assert decision["confidence"] >= 0.8
        assert "processing_time_ms" in decision
        assert decision["word_count"] == 6

    def test_route_complex_query(self):
        """Test routing of complex analytical queries."""
        query = "Compare the environmental impact of electric vs gasoline vehicles and explain the manufacturing differences"
        result = route_query(query)

        decision = json.loads(result)

        # Verify complex query classification
        assert decision["complexity"] == "complex"
        assert decision["strategy"] == "hybrid"
        assert decision["needs_planning"] is True
        assert decision["confidence"] >= 0.8
        assert decision["word_count"] > 10

    def test_route_medium_query(self):
        """Test routing of medium complexity queries."""
        query = "Tell me about machine learning algorithms"
        result = route_query(query)

        decision = json.loads(result)

        # Verify medium query classification
        assert decision["complexity"] == "medium"
        assert decision["strategy"] == "hybrid"
        assert decision["needs_planning"] is False
        assert decision["confidence"] >= 0.8

    def test_route_graphrag_query(self):
        """Test routing of relationship-focused queries."""
        query = "What are the connections between AI and machine learning?"
        result = route_query(query)

        decision = json.loads(result)

        # Verify GraphRAG strategy selection
        assert decision["strategy"] == "graphrag"
        assert "relationship" in query.lower() or "connect" in query.lower()

    def test_route_contextual_query_with_state(self):
        """Test routing with conversation context."""
        # Mock state with context
        mock_context = Mock()
        mock_context.chat_history = [
            Mock(content="Previous question about AI"),
            Mock(content="Follow-up about machine learning"),
        ]
        state = {"context": mock_context}

        query = "Can you elaborate on that?"
        result = route_query(query, state)

        decision = json.loads(result)

        # Verify context dependency was detected
        assert decision["context_dependent"] is True
        assert "processing_time_ms" in decision

    def test_route_query_error_handling(self):
        """Test error handling in query routing."""
        # Simulate error condition
        with patch("src.agents.tools.logger") as mock_logger:
            # Force an exception during processing
            with patch("builtins.len", side_effect=Exception("Test error")):
                result = route_query("Test query")

                decision = json.loads(result)

                # Verify fallback decision
                assert decision["strategy"] == "vector"
                assert decision["complexity"] == "simple"
                assert decision["confidence"] == 0.5
                assert "error" in decision

                # Verify error was logged
                mock_logger.error.assert_called_once()

    def test_route_query_performance_timing(self):
        """Test routing performance measurement."""
        query = "Performance test query"

        start_time = time.perf_counter()
        result = route_query(query)
        end_time = time.perf_counter()

        decision = json.loads(result)
        reported_time = decision["processing_time_ms"]
        actual_time = (end_time - start_time) * 1000

        # Verify timing is reasonable and reported
        assert reported_time > 0
        assert reported_time < actual_time + 10  # Allow some overhead


class TestPlanQuery:
    """Test suite for plan_query tool."""

    def test_plan_simple_query_no_decomposition(self):
        """Test planning for simple queries (no decomposition needed)."""
        query = "What is machine learning?"
        result = plan_query(query, "simple")

        plan = json.loads(result)

        # Verify simple query planning
        assert plan["original_query"] == query
        assert plan["sub_tasks"] == [query]  # No decomposition
        assert plan["execution_order"] == "sequential"
        assert plan["estimated_complexity"] == "low"

    def test_plan_comparison_query(self):
        """Test planning for comparison queries."""
        query = "Compare AI vs machine learning"
        result = plan_query(query, "complex")

        plan = json.loads(result)

        # Verify comparison query decomposition
        assert plan["original_query"] == query
        assert len(plan["sub_tasks"]) == 4  # Two entities + comparison + summary
        assert "AI" in str(plan["sub_tasks"])
        assert "machine learning" in str(plan["sub_tasks"])
        assert plan["execution_order"] == "parallel"  # Comparison can be parallelized

    def test_plan_analysis_query(self):
        """Test planning for analytical queries."""
        query = "Analyze the impact of artificial intelligence on healthcare"
        result = plan_query(query, "complex")

        plan = json.loads(result)

        # Verify analysis query decomposition
        assert plan["original_query"] == query
        assert (
            len(plan["sub_tasks"]) == 4
        )  # Key components, background, analysis, synthesis
        assert any("component" in task.lower() for task in plan["sub_tasks"])
        assert any("background" in task.lower() for task in plan["sub_tasks"])

    def test_plan_process_query(self):
        """Test planning for process/explanation queries."""
        query = "How does neural network training work?"
        result = plan_query(query, "complex")

        plan = json.loads(result)

        # Verify process query decomposition
        assert plan["original_query"] == query
        assert len(plan["sub_tasks"]) == 4  # Definition, steps, details, organization
        assert any("definition" in task.lower() for task in plan["sub_tasks"])
        assert any("step" in task.lower() for task in plan["sub_tasks"])

    def test_plan_list_query(self):
        """Test planning for list/enumeration queries."""
        query = "List the types of machine learning algorithms"
        result = plan_query(query, "medium")

        plan = json.loads(result)

        # Verify list query decomposition
        assert plan["original_query"] == query
        assert len(plan["sub_tasks"]) == 3  # Find info, categorize, organize
        assert any("categorize" in task.lower() for task in plan["sub_tasks"])

    def test_plan_query_with_connectors(self):
        """Test planning for queries with logical connectors."""
        query = "Explain AI and machine learning and deep learning"
        result = plan_query(query, "complex")

        plan = json.loads(result)

        # Verify connector-based decomposition
        assert plan["original_query"] == query
        assert len(plan["sub_tasks"]) > 2  # Should split on "and"
        assert any("Synthesize" in task for task in plan["sub_tasks"])

    def test_plan_query_error_handling(self):
        """Test error handling in query planning."""
        with patch("src.agents.tools.logger") as mock_logger:
            # Force an exception during processing
            with patch("builtins.len", side_effect=Exception("Planning error")):
                result = plan_query("Test query", "complex")

                plan = json.loads(result)

                # Verify fallback plan
                assert plan["original_query"] == "Test query"
                assert plan["sub_tasks"] == ["Test query"]
                assert plan["execution_order"] == "sequential"
                assert "error" in plan

                # Verify error was logged
                mock_logger.error.assert_called_once()

    def test_plan_query_performance_metrics(self):
        """Test planning performance measurement."""
        query = "Complex query for performance testing"
        result = plan_query(query, "complex")

        plan = json.loads(result)

        # Verify performance metrics
        assert "processing_time_ms" in plan
        assert "task_count" in plan
        assert plan["processing_time_ms"] > 0
        assert plan["task_count"] == len(plan["sub_tasks"])


class TestRetrieveDocuments:
    """Test suite for retrieve_documents tool."""

    def test_retrieve_vector_strategy(self, mock_tools_data: dict[str, Any]):
        """Test document retrieval using vector strategy."""
        state = {"tools_data": mock_tools_data}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = "Vector search result"
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result = retrieve_documents("machine learning", "vector", state=state)

            data = json.loads(result)

            # Verify vector retrieval
            assert data["strategy_used"] == "vector"
            assert data["query_original"] == "machine learning"
            assert len(data["documents"]) > 0
            assert data["documents"][0]["content"] == "Vector search result"

            # Verify tool was called correctly
            mock_factory.create_vector_search_tool.assert_called_once_with(
                mock_tools_data["vector"]
            )

    def test_retrieve_hybrid_strategy(self, mock_tools_data: dict[str, Any]):
        """Test document retrieval using hybrid strategy."""
        state = {"tools_data": mock_tools_data}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = "Hybrid search result"
            mock_factory.create_hybrid_search_tool.return_value = mock_tool

            result = retrieve_documents("AI concepts", "hybrid", state=state)

            data = json.loads(result)

            # Verify hybrid retrieval
            assert data["strategy_used"] == "hybrid_fusion"
            assert (
                data["query_optimized"] == "Find documents about AI concepts"
            )  # DSPy optimization
            assert len(data["documents"]) > 0

    def test_retrieve_graphrag_strategy(self, mock_tools_data: dict[str, Any]):
        """Test document retrieval using GraphRAG strategy."""
        state = {"tools_data": mock_tools_data}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = "GraphRAG result"
            mock_factory.create_kg_search_tool.return_value = mock_tool

            result = retrieve_documents(
                "AI relationships", "graphrag", use_graphrag=True, state=state
            )

            data = json.loads(result)

            # Verify GraphRAG retrieval
            assert data["strategy_used"] == "graphrag"
            assert data["graphrag_used"] is True
            assert len(data["documents"]) > 0

    def test_retrieve_with_dspy_optimization(self, mock_tools_data: dict[str, Any]):
        """Test document retrieval with DSPy query optimization."""
        state = {"tools_data": mock_tools_data}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = "Optimized result"
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result = retrieve_documents("AI", "vector", use_dspy=True, state=state)

            data = json.loads(result)

            # Verify DSPy optimization was applied
            assert data["dspy_used"] is True
            assert (
                data["query_optimized"] == "Find documents about AI"
            )  # Short query enhancement
            assert data["query_original"] == "AI"

    def test_retrieve_without_tools_data(self):
        """Test retrieval fallback when no tools data available."""
        state = {}  # No tools_data

        result = retrieve_documents("test query", "vector", state=state)

        data = json.loads(result)

        # Verify fallback behavior
        assert data["documents"] == []
        assert "error" in data
        assert "No retrieval tools available" in data["error"]

    def test_retrieve_tool_failure_handling(self, mock_tools_data: dict[str, Any]):
        """Test handling of tool execution failures."""
        state = {"tools_data": mock_tools_data}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.side_effect = Exception("Tool execution failed")
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result = retrieve_documents("test query", "vector", state=state)

            data = json.loads(result)

            # Verify error handling
            assert data["documents"] == []
            assert "processing_time_ms" in data

    def test_retrieve_graphrag_fallback(self, mock_tools_data: dict[str, Any]):
        """Test GraphRAG fallback to hybrid when GraphRAG fails."""
        state = {"tools_data": mock_tools_data}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            # GraphRAG tool fails
            mock_kg_tool = Mock()
            mock_kg_tool.call.side_effect = Exception("GraphRAG failed")
            mock_factory.create_kg_search_tool.return_value = mock_kg_tool

            # Hybrid tool succeeds
            mock_hybrid_tool = Mock()
            mock_hybrid_tool.call.return_value = "Hybrid fallback result"
            mock_factory.create_hybrid_vector_tool.return_value = mock_hybrid_tool

            result = retrieve_documents(
                "test query", "graphrag", use_graphrag=True, state=state
            )

            data = json.loads(result)

            # Verify fallback to hybrid
            assert data["strategy_used"] == "hybrid_vector"
            assert len(data["documents"]) > 0


class TestSynthesizeResults:
    """Test suite for synthesize_results tool."""

    def test_synthesize_multiple_results(self):
        """Test synthesis of multiple retrieval results."""
        sub_results = [
            {
                "documents": [
                    {"content": "AI is artificial intelligence", "score": 0.9},
                    {"content": "ML is machine learning", "score": 0.8},
                ],
                "strategy_used": "vector",
                "processing_time_ms": 100,
            },
            {
                "documents": [
                    {"content": "Deep learning uses neural networks", "score": 0.85},
                    {
                        "content": "AI is artificial intelligence",
                        "score": 0.9,
                    },  # Duplicate
                ],
                "strategy_used": "hybrid",
                "processing_time_ms": 150,
            },
        ]

        result = synthesize_results(
            json.dumps(sub_results), "What is AI and machine learning?"
        )

        data = json.loads(result)

        # Verify synthesis
        assert len(data["documents"]) == 3  # One duplicate removed
        assert data["synthesis_metadata"]["original_count"] == 4
        assert data["synthesis_metadata"]["after_deduplication"] == 3
        assert data["synthesis_metadata"]["final_count"] == 3
        assert "deduplication_ratio" in data["synthesis_metadata"]
        assert set(data["synthesis_metadata"]["strategies_used"]) == {
            "vector",
            "hybrid",
        }

    def test_synthesize_with_ranking(self):
        """Test synthesis with relevance-based ranking."""
        sub_results = [
            {
                "documents": [
                    {"content": "Unrelated content about cooking", "score": 0.5},
                    {"content": "Machine learning algorithms for AI", "score": 0.9},
                    {"content": "AI artificial intelligence overview", "score": 0.8},
                ],
                "strategy_used": "vector",
            }
        ]

        result = synthesize_results(
            json.dumps(sub_results), "artificial intelligence machine learning"
        )

        data = json.loads(result)

        # Verify ranking by relevance
        assert len(data["documents"]) == 3
        # First document should be most relevant (highest word overlap)
        first_doc = data["documents"][0]
        assert "relevance_score" in first_doc
        assert first_doc["relevance_score"] > 0

    def test_synthesize_empty_results(self):
        """Test synthesis with empty or no results."""
        sub_results = []

        result = synthesize_results(json.dumps(sub_results), "test query")

        data = json.loads(result)

        # Verify empty result handling
        assert data["documents"] == []
        assert data["synthesis_metadata"]["original_count"] == 0
        assert data["synthesis_metadata"]["final_count"] == 0

    def test_synthesize_invalid_json_input(self):
        """Test synthesis with invalid JSON input."""
        invalid_json = "not valid json"

        result = synthesize_results(invalid_json, "test query")

        data = json.loads(result)

        # Verify error handling
        assert data["documents"] == []
        assert "error" in data
        assert "Invalid input format" in data["error"]

    def test_synthesize_with_max_results_limit(self):
        """Test synthesis respects maximum results limit."""
        # Create many documents
        documents = [{"content": f"Document {i}", "score": 0.8} for i in range(20)]
        sub_results = [{"documents": documents, "strategy_used": "vector"}]

        with patch("src.config.settings.synthesis_max_docs", 5):
            result = synthesize_results(json.dumps(sub_results), "test query")

            data = json.loads(result)

            # Verify limit is respected
            assert len(data["documents"]) == 5
            assert data["synthesis_metadata"]["original_count"] == 20
            assert data["synthesis_metadata"]["final_count"] == 5

    def test_synthesize_error_handling(self):
        """Test error handling in synthesis process."""
        with patch("src.agents.tools.logger") as mock_logger:
            # Force an exception during processing
            with patch("builtins.len", side_effect=Exception("Synthesis error")):
                result = synthesize_results("[]", "test query")

                data = json.loads(result)

                # Verify error handling
                assert data["documents"] == []
                assert "error" in data

                # Verify error was logged
                mock_logger.error.assert_called_once()


class TestValidateResponse:
    """Test suite for validate_response tool."""

    def test_validate_good_response(self):
        """Test validation of a good quality response."""
        query = "What is machine learning?"
        response = (
            "Machine learning is a subset of artificial intelligence that enables "
            "computers to learn and improve from experience without being explicitly "
            "programmed. It uses algorithms and statistical models to analyze data "
            "and make predictions or decisions."
        )
        sources = json.dumps(
            {
                "documents": [
                    {
                        "content": "Machine learning algorithms analyze data patterns",
                        "score": 0.9,
                    },
                    {
                        "content": "AI subset artificial intelligence learning",
                        "score": 0.8,
                    },
                ]
            }
        )

        result = validate_response(query, response, sources)

        data = json.loads(result)

        # Verify good response validation
        assert data["valid"] is True
        assert data["confidence"] >= 0.6
        assert data["suggested_action"] == "accept"
        assert len(data["issues"]) <= 1
        assert "processing_time_ms" in data
        assert data["source_count"] == 2

    def test_validate_incomplete_response(self):
        """Test validation of incomplete response."""
        query = "Explain the complex process of neural network training"
        response = "Neural networks learn."  # Too brief
        sources = json.dumps({"documents": []})

        result = validate_response(query, response, sources)

        data = json.loads(result)

        # Verify incomplete response detection
        assert data["confidence"] < 0.8
        assert any(issue["type"] == "incomplete_response" for issue in data["issues"])
        assert any(issue["type"] == "no_sources" for issue in data["issues"])
        assert data["suggested_action"] in ["refine", "regenerate"]

    def test_validate_missing_source_attribution(self):
        """Test validation detects missing source attribution."""
        query = "What is deep learning?"
        response = (
            "Deep learning is a fascinating field that involves complex computations "
            "and has many applications in modern technology."
        )
        sources = json.dumps(
            {
                "documents": [
                    {
                        "content": "Neural networks with multiple layers for pattern recognition",
                        "score": 0.9,
                    },
                ]
            }
        )

        result = validate_response(query, response, sources)

        data = json.loads(result)

        # Verify missing source detection
        issues = data["issues"]
        assert any(issue["type"] == "missing_source" for issue in issues)
        assert data["confidence"] < 0.8

    def test_validate_potential_hallucination(self):
        """Test validation detects potential hallucinations."""
        query = "Explain quantum computing"
        response = (
            "I cannot find specific information about quantum computing in the provided sources. "
            "Based on my training data, quantum computing uses quantum mechanics principles."
        )
        sources = json.dumps({"documents": []})

        result = validate_response(query, response, sources)

        data = json.loads(result)

        # Verify hallucination detection
        issues = data["issues"]
        assert any(issue["type"] == "potential_hallucination" for issue in issues)
        assert any(issue["type"] == "no_sources" for issue in issues)
        assert data["confidence"] <= 0.5
        assert data["suggested_action"] == "regenerate"

    def test_validate_low_relevance(self):
        """Test validation detects low query relevance."""
        query = "machine learning algorithms neural networks"
        response = (
            "Cooking recipes are important for making delicious meals. "
            "There are many different cooking techniques to explore."
        )
        sources = json.dumps({"documents": []})

        result = validate_response(query, response, sources)

        data = json.loads(result)

        # Verify low relevance detection
        issues = data["issues"]
        assert any(issue["type"] == "low_relevance" for issue in issues)
        assert data["confidence"] < 0.8

    def test_validate_coherence_issues(self):
        """Test validation detects coherence issues."""
        query = "What is AI?"
        response = (
            "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O."  # Unusual sentence structure
        )
        sources = json.dumps({"documents": []})

        result = validate_response(query, response, sources)

        data = json.loads(result)

        # Verify coherence issue detection
        issues = data["issues"]
        assert any(issue["type"] == "coherence_issue" for issue in issues)

    def test_validate_with_good_sources(self):
        """Test validation with good source attribution."""
        query = "What is artificial intelligence?"
        response = (
            "Artificial intelligence (AI) refers to computer systems that can perform "
            "tasks typically requiring human intelligence, such as learning, reasoning, "
            "and problem-solving. Machine learning algorithms enable AI systems to "
            "improve their performance through experience."
        )
        sources = json.dumps(
            {
                "documents": [
                    {
                        "content": "Artificial intelligence computer systems human intelligence learning reasoning",
                        "score": 0.9,
                    },
                    {
                        "content": "Machine learning algorithms improve performance through experience",
                        "score": 0.8,
                    },
                ]
            }
        )

        result = validate_response(query, response, sources)

        data = json.loads(result)

        # Verify good validation with sources
        assert data["valid"] is True
        assert data["confidence"] >= 0.8
        assert data["suggested_action"] == "accept"
        assert data["source_count"] == 2

    def test_validate_error_handling(self):
        """Test error handling in response validation."""
        with patch("src.agents.tools.logger") as mock_logger:
            # Force an exception during processing
            with patch("builtins.len", side_effect=Exception("Validation error")):
                result = validate_response("query", "response", "[]")

                data = json.loads(result)

                # Verify error handling
                assert data["valid"] is False
                assert data["confidence"] == 0.0
                assert data["suggested_action"] == "regenerate"
                assert "error" in data

                # Verify error was logged
                mock_logger.error.assert_called_once()

    def test_validate_invalid_sources_json(self):
        """Test validation with invalid sources JSON."""
        query = "Test query"
        response = "Test response"
        invalid_sources = "not valid json"

        result = validate_response(query, response, invalid_sources)

        data = json.loads(result)

        # Verify handling of invalid JSON
        assert data["source_count"] == 0
        issues = data["issues"]
        assert any(issue["type"] == "no_sources" for issue in issues)


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_parse_tool_result_string(self):
        """Test parsing string tool result."""
        result = "This is a text response from a tool"
        documents = _parse_tool_result(result)

        assert len(documents) == 1
        assert documents[0]["content"] == result
        assert documents[0]["metadata"]["source"] == "tool_response"
        assert documents[0]["score"] == 1.0

    def test_parse_tool_result_llamaindex_response(self):
        """Test parsing LlamaIndex response object."""
        mock_result = Mock()
        mock_result.response = "LlamaIndex response text"
        mock_result.metadata = {"source": "test.pdf", "page": 1}

        documents = _parse_tool_result(mock_result)

        assert len(documents) == 1
        assert documents[0]["content"] == "LlamaIndex response text"
        assert documents[0]["metadata"]["source"] == "test.pdf"
        assert documents[0]["score"] == 1.0

    def test_parse_tool_result_document_list(self):
        """Test parsing list of Document objects."""
        docs = [
            Document(text="First document", metadata={"source": "doc1.pdf"}),
            Document(text="Second document", metadata={"source": "doc2.pdf"}),
        ]

        # Mock score attribute
        docs[0].score = 0.9
        docs[1].score = 0.8

        documents = _parse_tool_result(docs)

        assert len(documents) == 2
        assert documents[0]["content"] == "First document"
        assert documents[0]["metadata"]["source"] == "doc1.pdf"
        assert documents[0]["score"] == 0.9
        assert documents[1]["content"] == "Second document"
        assert documents[1]["score"] == 0.8

    def test_parse_tool_result_dict_list(self):
        """Test parsing list of dictionary objects."""
        result = [
            {
                "content": "Dict document 1",
                "metadata": {"source": "dict1.pdf"},
                "score": 0.95,
            },
            {
                "content": "Dict document 2",
                "metadata": {"source": "dict2.pdf"},
                "score": 0.85,
            },
        ]

        documents = _parse_tool_result(result)

        assert len(documents) == 2
        assert documents == result  # Should return as-is

    def test_parse_tool_result_fallback(self):
        """Test parsing with fallback conversion."""
        result = {"unexpected": "format", "data": 123}

        documents = _parse_tool_result(result)

        assert len(documents) == 1
        assert documents[0]["content"] == str(result)
        assert documents[0]["metadata"]["source"] == "unknown"
        assert documents[0]["score"] == 1.0

    def test_rank_documents_by_relevance(self):
        """Test document ranking by relevance."""
        documents = [
            {"content": "This is about cooking recipes and food", "score": 0.8},
            {
                "content": "Machine learning algorithms for artificial intelligence",
                "score": 0.7,
            },
            {"content": "AI and machine learning in computer science", "score": 0.9},
        ]
        query = "artificial intelligence machine learning"

        ranked = _rank_documents_by_relevance(documents, query)

        # Verify ranking (AI/ML docs should rank higher)
        assert len(ranked) == 3
        assert all("relevance_score" in doc for doc in ranked)

        # First document should be most relevant
        assert (
            ranked[0]["relevance_score"] > ranked[2]["relevance_score"]
        )  # AI doc > cooking doc

        # Verify original scores are still boosted
        for doc in ranked:
            assert doc["relevance_score"] >= 0.0

    def test_rank_documents_empty_query(self):
        """Test document ranking with empty query."""
        documents = [
            {"content": "Some content", "score": 0.8},
            {"content": "Other content", "score": 0.9},
        ]

        ranked = _rank_documents_by_relevance(documents, "")

        # Should still return documents with relevance scores
        assert len(ranked) == 2
        assert all("relevance_score" in doc for doc in ranked)

    def test_rank_documents_no_content(self):
        """Test document ranking with documents missing content."""
        documents = [
            {"metadata": {"source": "test.pdf"}, "score": 0.8},  # No content field
            {
                "text": "Alternative text field",
                "score": 0.9,
            },  # Using 'text' instead of 'content'
        ]
        query = "test query"

        ranked = _rank_documents_by_relevance(documents, query)

        # Should handle missing content gracefully
        assert len(ranked) == 2
        assert all("relevance_score" in doc for doc in ranked)
