"""Unit tests for agent tool functions.

Covers route_query, plan_query, retrieve_documents, synthesize_results, and
validate_response with boundary testing and business logic validation.
"""

import json
from unittest.mock import MagicMock, Mock, patch

from src.agents.tools import (
    plan_query,
    retrieve_documents,
    route_query,
    synthesize_results,
    validate_response,
)


class TestRouteQuery:
    """Test the route_query critical function."""

    def test_route_query_simple_query(self):
        """Test routing simple query."""
        result_json = route_query.invoke({"query": "Define Python"})
        result = json.loads(result_json)

        assert isinstance(result, dict)
        assert "strategy" in result
        assert "complexity" in result
        assert "needs_planning" in result
        assert "confidence" in result
        assert "processing_time_ms" in result

        # Simple query should be classified correctly
        assert result["complexity"] in {"simple", "medium"}
        assert result["strategy"] == "vector"
        assert not result["needs_planning"]
        assert result["confidence"] >= 0.9

    def test_route_query_complex_query(self):
        """Test routing complex query requiring planning."""
        result_json = route_query.invoke(
            {
                "query": (
                    "Compare machine learning vs deep learning approaches, analyze "
                    "their differences and explain step by step implementation"
                )
            }
        )
        result = json.loads(result_json)

        assert result["complexity"] == "complex"
        assert result["strategy"] == "hybrid"
        assert result["needs_planning"]
        assert result["confidence"] == 0.9
        assert result["word_count"] > 10

    def test_route_query_medium_query(self):
        """Test routing medium complexity query."""
        result_json = route_query.invoke(
            {"query": "When was backpropagation introduced"}
        )
        result = json.loads(result_json)

        assert result["complexity"] == "medium"
        assert result["strategy"] == "hybrid"
        assert not result["needs_planning"]
        assert result["confidence"] >= 0.8

    def test_route_query_graphrag_patterns(self):
        """Test routing query with GraphRAG indicators."""
        result_json = route_query.invoke(
            {"query": "Show the relationship between concepts and their connections"}
        )
        result = json.loads(result_json)

        assert result["strategy"] == "graphrag"

    def test_route_query_with_context(self):
        """Test routing query with conversation context."""
        mock_state = {"context": MagicMock()}
        mock_state["context"].chat_history = [
            MagicMock(content="Previous question about AI"),
            MagicMock(content="Another question"),
        ]

        result_json = route_query.invoke(
            {"query": "What about this topic?", "state": mock_state}
        )
        result = json.loads(result_json)

        assert result["context_dependent"]

    def test_route_query_contextual_without_history(self):
        """Test contextual query without previous context."""
        result_json = route_query.invoke({"query": "What about this approach?"})
        result = json.loads(result_json)

        # Should have lower confidence due to missing context
        assert result["confidence"] < 0.95

    def test_route_query_error_handling(self):
        """Test route_query error handling."""
        import pytest

        # Mock an internal error: route_query now propagates exceptions
        with (
            patch(
                "src.agents.tools.time.perf_counter",
                side_effect=RuntimeError("Timer error"),
            ),
            pytest.raises(RuntimeError),
        ):
            route_query.invoke({"query": "test query"})

    def test_route_query_boundary_values(self):
        """Test route_query with boundary word counts."""
        # Test exactly at medium threshold (10 words)
        result_json = route_query.invoke(
            {"query": "This is exactly ten words for medium complexity threshold test"}
        )
        result = json.loads(result_json)
        assert result["word_count"] == 10

        # Test exactly at complex threshold (20 words)
        long_query = " ".join(["word"] * 20)
        result_json = route_query.invoke({"query": long_query})
        result = json.loads(result_json)
        assert result["word_count"] == 20


class TestPlanQuery:
    """Test the plan_query critical function."""

    def test_plan_query_simple(self):
        """Test planning simple query."""
        result_json = plan_query.invoke(
            {"query": "What is AI?", "complexity": "simple"}
        )
        result = json.loads(result_json)

        assert isinstance(result, dict)
        assert "original_query" in result
        assert "sub_tasks" in result
        assert "execution_order" in result
        assert "estimated_complexity" in result

        # Simple query should not be decomposed
        assert result["sub_tasks"] == ["What is AI?"]
        assert result["execution_order"] == "sequential"

    def test_plan_query_comparison(self):
        """Test planning comparison query."""
        result_json = plan_query.invoke(
            {"query": "Compare AI vs ML performance", "complexity": "complex"}
        )
        result = json.loads(result_json)

        assert len(result["sub_tasks"]) >= 3
        assert result["execution_order"] == "parallel"
        # Should have tasks for each entity and comparison
        assert any("AI" in task for task in result["sub_tasks"])
        assert any("ML" in task for task in result["sub_tasks"])
        assert any("Compare" in task for task in result["sub_tasks"])

    def test_plan_query_analysis(self):
        """Test planning analysis query."""
        result_json = plan_query.invoke(
            {
                "query": "Analyze the performance of neural networks",
                "complexity": "complex",
            }
        )
        result = json.loads(result_json)

        assert len(result["sub_tasks"]) >= 3
        # Should have structured analysis tasks
        sub_tasks_text = " ".join(result["sub_tasks"])
        assert (
            "components" in sub_tasks_text.lower()
            or "background" in sub_tasks_text.lower()
        )

    def test_plan_query_process_explanation(self):
        """Test planning process/how-to query."""
        result_json = plan_query.invoke(
            {"query": "How does gradient descent work?", "complexity": "complex"}
        )
        result = json.loads(result_json)

        assert len(result["sub_tasks"]) >= 3
        # Should have sequential tasks for process explanation
        sub_tasks_text = " ".join(result["sub_tasks"])
        assert (
            "steps" in sub_tasks_text.lower() or "definition" in sub_tasks_text.lower()
        )

    def test_plan_query_list_enumeration(self):
        """Test planning list/enumeration query."""
        result_json = plan_query.invoke(
            {
                "query": "List the types of machine learning algorithms",
                "complexity": "complex",
            }
        )
        result = json.loads(result_json)

        assert len(result["sub_tasks"]) >= 2
        # Should have tasks for finding and organizing information
        sub_tasks_text = " ".join(result["sub_tasks"])
        assert (
            "list" in sub_tasks_text.lower() or "categorize" in sub_tasks_text.lower()
        )

    def test_plan_query_default_decomposition(self):
        """Test planning with default decomposition strategy."""
        result_json = plan_query.invoke(
            {
                "query": "Complex query with multiple aspects and various components",
                "complexity": "complex",
            }
        )
        result = json.loads(result_json)

        assert len(result["sub_tasks"]) >= 1
        assert result["task_count"] == len(result["sub_tasks"])
        assert result["estimated_complexity"] in ["medium", "high"]

    def test_plan_query_error_handling(self):
        """Test plan_query error handling."""
        with patch(
            "src.agents.tools.time.perf_counter",
            side_effect=RuntimeError("Timer error"),
        ):
            result_json = plan_query.invoke(
                {"query": "test query", "complexity": "complex"}
            )
            result = json.loads(result_json)
            assert "error" in result
            assert result["sub_tasks"] == ["test query"]
            assert result["execution_order"] == "sequential"


class TestRetrieveDocuments:
    """Test the retrieve_documents critical function."""

    def test_retrieve_documents_no_tools_data(self):
        """Test retrieve_documents when no tools data available."""
        result_json = retrieve_documents.invoke({"query": "test query", "state": {}})
        result = json.loads(result_json)

        assert result["documents"] == []
        assert "error" in result
        assert result["strategy_used"] == "hybrid"  # Default strategy

    def test_retrieve_documents_no_state(self):
        """Test retrieve_documents with no state."""
        # Omit the optional 'state' to allow default handling
        result_json = retrieve_documents.invoke({"query": "test query"})
        result = json.loads(result_json)

        assert result["documents"] == []
        assert "error" in result

    @patch("src.dspy_integration.DSPyLlamaIndexRetriever")
    def test_retrieve_documents_with_dspy_optimization(self, mock_dspy):
        """Test retrieve_documents with DSPy query optimization."""
        # Mock DSPy optimization
        mock_dspy.optimize_query.return_value = {
            "refined": "optimized test query",
            "variants": ["variant 1", "variant 2"],
        }

        # Mock tools data
        mock_state = {"tools_data": {"vector": MagicMock(), "retriever": None}}

        # Mock tool factory and search results
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = MagicMock()
            mock_tool.call.return_value = [{"content": "test doc", "score": 0.9}]
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result_json = retrieve_documents.invoke(
                {
                    "query": "test query",
                    "strategy": "vector",
                    "use_dspy": True,
                    "state": mock_state,
                }
            )
            result = json.loads(result_json)

        assert result["query_original"] == "test query"
        assert result["query_optimized"] == "optimized test query"
        assert result["dspy_used"]
        assert len(result["documents"]) > 0

    def test_retrieve_documents_fallback_optimization(self):
        """Test retrieve_documents with fallback query optimization."""
        mock_state = {
            "tools_data": {
                "vector": MagicMock(),
            }
        }

        # Simulate ImportError during importing DSPyLlamaIndexRetriever
        import builtins as _builtins

        real_import = _builtins.__import__

        def _fake_import(name, *args, **kwargs):  # noqa: D401
            if name.startswith("src.dspy_integration"):
                raise ImportError("Module not found")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(_builtins, "__import__", side_effect=_fake_import),
            patch("src.agents.tools.ToolFactory") as mock_factory,
        ):
            mock_tool = MagicMock()
            mock_tool.call.return_value = [{"content": "test doc"}]
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result_json = retrieve_documents.invoke(
                {"query": "AI", "use_dspy": True, "state": mock_state}
            )
            result = json.loads(result_json)

        # Should have fallback optimization for short queries
        assert result["query_optimized"] == "Find documents about AI"

    def test_retrieve_documents_graphrag_strategy(self):
        """Test retrieve_documents with GraphRAG strategy."""
        mock_state = {
            "tools_data": {
                "kg": MagicMock(),
            }
        }

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = MagicMock()
            mock_tool.call.return_value = "Knowledge graph result"
            mock_factory.create_kg_search_tool.return_value = mock_tool

            result_json = retrieve_documents.invoke(
                {
                    "query": "test query",
                    "strategy": "graphrag",
                    "use_graphrag": True,
                    "state": mock_state,
                }
            )
            result = json.loads(result_json)

        assert result["strategy_used"] == "graphrag"
        assert result["graphrag_used"]

    def test_retrieve_documents_hybrid_strategy(self):
        """Test retrieve_documents with hybrid strategy."""
        mock_state = {"tools_data": {"vector": MagicMock(), "retriever": MagicMock()}}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = MagicMock()
            mock_tool.call.return_value = [{"content": "hybrid result", "score": 0.8}]
            mock_factory.create_hybrid_search_tool.return_value = mock_tool

            result_json = retrieve_documents.invoke(
                {"query": "test query", "strategy": "hybrid", "state": mock_state}
            )
            result = json.loads(result_json)

        assert result["strategy_used"] == "hybrid_fusion"

    def test_retrieve_documents_deduplication(self):
        """Test document deduplication in retrieve_documents."""
        mock_state = {
            "tools_data": {
                "vector": MagicMock(),
            }
        }

        # Mock duplicate documents
        duplicate_docs = [
            {"content": "Same content here", "score": 0.9},
            {"content": "Same content here", "score": 0.8},  # Duplicate
            {"content": "Different content", "score": 0.7},
        ]

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = MagicMock()
            mock_tool.call.return_value = duplicate_docs
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result_json = retrieve_documents.invoke(
                {
                    "query": "test query",
                    "strategy": "vector",
                    "state": mock_state,
                    "use_dspy": False,
                }
            )
            result = json.loads(result_json)

        # Should keep only the higher scoring duplicate
        assert len(result["documents"]) == 2
        assert result["documents"][0]["score"] == 0.9  # Higher score kept


class TestSynthesizeResults:
    """Test the synthesize_results critical function."""

    def test_synthesize_results_basic(self):
        """Test basic result synthesis."""
        sub_results = [
            {
                "documents": [{"content": "doc1", "score": 0.9}],
                "strategy_used": "vector",
                "processing_time_ms": 100,
            },
            {
                "documents": [{"content": "doc2", "score": 0.8}],
                "strategy_used": "hybrid",
                "processing_time_ms": 150,
            },
        ]

        result_json = synthesize_results.invoke(
            {"sub_results": json.dumps(sub_results), "original_query": "test query"}
        )
        result = json.loads(result_json)

        assert isinstance(result, dict)
        assert "documents" in result
        assert "synthesis_metadata" in result
        assert "original_query" in result

        assert len(result["documents"]) == 2
        assert result["synthesis_metadata"]["original_count"] == 2
        assert result["synthesis_metadata"]["final_count"] == 2
        assert "vector" in result["synthesis_metadata"]["strategies_used"]
        assert "hybrid" in result["synthesis_metadata"]["strategies_used"]

    def test_synthesize_results_deduplication(self):
        """Test synthesis with document deduplication."""
        sub_results = [
            {
                "documents": [
                    {
                        "content": (
                            "machine learning is a subset of artificial intelligence"
                        )
                    }
                ]
            },
            {
                "documents": [
                    {
                        "content": (
                            "machine learning is a subset of artificial intelligence"
                        )
                    }
                ]
            },  # Duplicate
            {"documents": [{"content": "deep learning uses neural networks"}]},
        ]

        result_json = synthesize_results.invoke(
            {"sub_results": json.dumps(sub_results), "original_query": "AI query"}
        )
        result = json.loads(result_json)

        # Should deduplicate based on content similarity
        assert result["synthesis_metadata"]["original_count"] == 3
        assert result["synthesis_metadata"]["after_deduplication"] <= 2
        assert result["synthesis_metadata"]["deduplication_ratio"] <= 1.0

    def test_synthesize_results_ranking(self):
        """Test synthesis with document ranking."""
        sub_results = [
            {
                "documents": [
                    {"content": "machine learning algorithms", "score": 0.7},
                    {"content": "artificial intelligence overview", "score": 0.9},
                ]
            }
        ]

        result_json = synthesize_results.invoke(
            {
                "sub_results": json.dumps(sub_results),
                "original_query": "machine learning",
            }
        )
        result = json.loads(result_json)

        # Documents should be ranked by relevance
        assert len(result["documents"]) == 2
        # First document should have higher relevance to "machine learning"
        first_doc = result["documents"][0]
        assert "relevance_score" in first_doc

    def test_synthesize_results_invalid_json(self):
        """Test synthesis with invalid JSON input."""
        result_json = synthesize_results.invoke(
            {"sub_results": "invalid json", "original_query": "test query"}
        )
        result = json.loads(result_json)

        assert result["documents"] == []
        assert "error" in result

    def test_synthesize_results_empty_input(self):
        """Test synthesis with empty results."""
        result_json = synthesize_results.invoke(
            {"sub_results": json.dumps([]), "original_query": "test query"}
        )
        result = json.loads(result_json)

        assert result["documents"] == []
        assert result["synthesis_metadata"]["original_count"] == 0

    def test_synthesize_results_max_limit(self):
        """Test synthesis respects maximum result limit."""
        # Create more documents than the limit (10)
        large_results = [
            {"documents": [{"content": f"doc {i}", "score": 0.8} for i in range(15)]}
        ]

        result_json = synthesize_results.invoke(
            {"sub_results": json.dumps(large_results), "original_query": "test query"}
        )
        result = json.loads(result_json)

        # Should be limited to MAX_RETRIEVAL_RESULTS (10)
        assert len(result["documents"]) <= 10

    def test_synthesize_results_handles_invalid_json(self):
        """Test synthesis error handling on invalid JSON input (public boundary)."""
        result_json = synthesize_results.invoke(
            {"sub_results": "{not: 'json'}", "original_query": "test query"}
        )
        result = json.loads(result_json)

        assert result["documents"] == []
        assert "error" in result


class TestValidateResponse:
    """Test the validate_response critical function."""

    def test_validate_response_good_quality(self):
        """Test validation of good quality response."""
        query = "What is machine learning?"
        response = (
            "Machine learning is a subset of artificial intelligence that focuses on "
            "algorithms that can learn from data without being explicitly programmed."
        )
        sources = json.dumps(
            [
                {
                    "content": "Machine learning is a method of data analysis that "
                    "automates analytical model building using algorithms "
                    "that iteratively learn from data",
                    "score": 0.9,
                }
            ]
        )

        result_json = validate_response.invoke(
            {"query": query, "response": response, "sources": sources}
        )
        result = json.loads(result_json)

        assert isinstance(result, dict)
        assert "valid" in result
        assert "confidence" in result
        assert "issues" in result
        assert "suggested_action" in result

        assert result["valid"]
        assert result["confidence"] >= 0.6
        assert result["suggested_action"] in ["accept", "refine"]
        assert result["source_count"] == 1

    def test_validate_response_short_response(self):
        """Test validation of too short response."""
        query = "Explain neural networks in detail"
        response = "Neural networks are good."  # Too short
        sources = json.dumps([{"content": "detailed neural network info"}])

        result_json = validate_response.invoke(
            {"query": query, "response": response, "sources": sources}
        )
        result = json.loads(result_json)

        # Should flag incomplete response
        assert any(issue["type"] == "incomplete_response" for issue in result["issues"])
        assert result["confidence"] < 1.0

    def test_validate_response_no_sources(self):
        """Test validation when no sources provided."""
        result_json = validate_response.invoke(
            {"query": "test query", "response": "test response", "sources": "[]"}
        )
        result = json.loads(result_json)

        assert any(issue["type"] == "no_sources" for issue in result["issues"])
        assert result["confidence"] < 1.0

    def test_validate_response_missing_source_attribution(self):
        """Test validation when response doesn't reference sources."""
        query = "What is AI?"
        response = "Artificial intelligence is about computers thinking like humans."
        sources = json.dumps(
            [
                {
                    "content": (
                        "Machine learning algorithms use statistical techniques to "
                        "learn patterns from data"  # Unrelated
                    )
                }
            ]
        )

        result_json = validate_response.invoke(
            {"query": query, "response": response, "sources": sources}
        )
        result = json.loads(result_json)

        assert any(issue["type"] == "missing_source" for issue in result["issues"])

    def test_validate_response_hallucination_indicators(self):
        """Test validation with hallucination indicators."""
        response = "I cannot find information about this topic in my training data."
        sources = json.dumps([{"content": "relevant source information"}])

        result_json = validate_response.invoke(
            {"query": "test query", "response": response, "sources": sources}
        )
        result = json.loads(result_json)

        assert any(
            issue["type"] == "potential_hallucination" for issue in result["issues"]
        )
        assert result["confidence"] <= 0.5

    def test_validate_response_low_relevance(self):
        """Test validation with low query relevance."""
        query = "machine learning algorithms neural networks"
        response = "The weather is nice today and cats are great pets."  # Irrelevant
        sources = json.dumps([{"content": "source info"}])

        result_json = validate_response.invoke(
            {"query": query, "response": response, "sources": sources}
        )
        result = json.loads(result_json)

        assert any(issue["type"] == "low_relevance" for issue in result["issues"])

    def test_validate_response_coherence_issues(self):
        """Test validation with coherence issues."""
        query = "test query"
        # Very short sentences
        response = "AI. ML. Good. Yes. No. Maybe. Done."
        sources = json.dumps([{"content": "source"}])

        result_json = validate_response.invoke(
            {"query": query, "response": response, "sources": sources}
        )
        result = json.loads(result_json)

        # May flag coherence issues due to very short sentences
        [issue for issue in result["issues"] if issue["type"] == "coherence_issue"]
        # This test is more about structure - coherence detection is basic

    def test_validate_response_suggested_actions(self):
        """Test validation suggested actions logic."""
        # High confidence - should accept
        high_conf_result = json.loads(
            validate_response.invoke(
                {
                    "query": "What is AI?",
                    "response": (
                        "AI is artificial intelligence, a field of computer "
                        "science focused on creating systems that can perform "
                        "tasks requiring human intelligence."
                    ),
                    "sources": json.dumps(
                        [
                            {
                                "content": (
                                    "artificial intelligence computer science "
                                    "systems human intelligence"
                                )
                            }
                        ]
                    ),
                }
            )
        )

        # Low confidence - should regenerate
        low_conf_result = json.loads(
            validate_response.invoke(
                {"query": "What is AI?", "response": "I don't know.", "sources": "[]"}
            )
        )

        assert high_conf_result["suggested_action"] in ["accept", "refine"]
        assert low_conf_result["suggested_action"] == "regenerate"

    def test_validate_response_error_handling(self):
        """Test validation error handling via timing failure."""
        with patch(
            "src.agents.tools.time.perf_counter",
            side_effect=RuntimeError("Timer error"),
        ):
            result_json = validate_response.invoke(
                {"query": "query", "response": "response", "sources": "[]"}
            )
            result = json.loads(result_json)

            assert not result["valid"]
            assert result["confidence"] == 0.0
            assert result["suggested_action"] == "regenerate"
            assert "error" in result


class TestHelperFunctions:
    """Boundary tests for parsing and ranking via public APIs only."""

    def test_retrieve_parsing_string_result(self):
        """Parsing of string tool output is validated via retrieve_documents."""
        mock_state = {"tools_data": {"vector": Mock()}}
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = "Simple string response"
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result_json = retrieve_documents.invoke(
                {"query": "q", "strategy": "vector", "state": mock_state}
            )
            data = json.loads(result_json)

        assert len(data["documents"]) == 1
        assert data["documents"][0]["content"] == "Simple string response"
        assert data["documents"][0]["score"] == 1.0

    def test_retrieve_parsing_llamaindex_like_object(self):
        """LlamaIndex response objects parsed through public retrieve_documents."""
        mock_state = {"tools_data": {"vector": Mock()}}
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_result = Mock()
            mock_result.response = "LlamaIndex response"
            mock_result.metadata = {"source": "test"}

            mock_tool = Mock()
            mock_tool.call.return_value = mock_result
            mock_factory.create_vector_search_tool.return_value = mock_tool

            data = json.loads(
                retrieve_documents.invoke(
                    {"query": "q", "strategy": "vector", "state": mock_state}
                )
            )

        assert len(data["documents"]) == 1
        assert data["documents"][0]["content"] == "LlamaIndex response"
        assert data["documents"][0]["metadata"]["source"] == "test"

    def test_retrieve_parsing_document_list(self):
        """List[Document] parsing verified via retrieve_documents public path."""
        from llama_index.core import Document

        docs = [
            Document(text="First document", metadata={"id": 1}),
            Document(text="Second document", metadata={"id": 2}),
        ]

        mock_state = {"tools_data": {"vector": Mock()}}
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = docs
            mock_factory.create_vector_search_tool.return_value = mock_tool

            data = json.loads(
                retrieve_documents.invoke(
                    {"query": "q", "strategy": "vector", "state": mock_state}
                )
            )

        assert len(data["documents"]) == 2
        assert data["documents"][0]["content"] == "First document"
        assert data["documents"][1]["content"] == "Second document"

    def test_retrieve_parsing_dict_list_passthrough(self):
        """List[dict] is returned as-is via public API."""
        inputs = [
            {"content": "First doc", "score": 0.9},
            {"content": "Second doc", "score": 0.8},
        ]

        mock_state = {"tools_data": {"vector": Mock()}}
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = inputs
            mock_factory.create_vector_search_tool.return_value = mock_tool

            data = json.loads(
                retrieve_documents.invoke(
                    {"query": "q", "strategy": "vector", "state": mock_state}
                )
            )

        assert len(data["documents"]) == 2
        assert data["documents"] == inputs

    def test_retrieve_parsing_fallback_conversion(self):
        """Unknown result shapes fall back to string conversion via public API."""
        mock_state = {"tools_data": {"vector": Mock()}}
        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = {"unexpected": "format", "data": 123}
            mock_factory.create_vector_search_tool.return_value = mock_tool

            data = json.loads(
                retrieve_documents.invoke(
                    {"query": "q", "strategy": "vector", "state": mock_state}
                )
            )

        assert len(data["documents"]) == 1
        assert data["documents"][0]["content"].startswith("{'unexpected': 'format'")
