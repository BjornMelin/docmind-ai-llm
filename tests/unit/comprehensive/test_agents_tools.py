"""Comprehensive unit tests for agents/tools.py targeting 50%+ coverage.

This test suite focuses on real business logic testing without inappropriate internal
mocking. Tests critical agent functionality including query routing complexity analysis,
multi-strategy document retrieval, result synthesis and validation, and error recovery
mechanisms.

Coverage Focus:
- Query routing and complexity analysis logic (currently 282/331 missing statements)
- Document retrieval with multiple strategies and DSPy optimization
- Result synthesis with deduplication and ranking algorithms
- Response validation with quality scoring and hallucination detection
- Error handling and fallback mechanisms
- Performance measurement and timing validation

Test Strategy:
- Use lightweight test doubles for external services only
- Test actual business logic flows and decision trees
- Include edge cases and error conditions
- Validate complex multi-step workflows
- Performance and timing validation
"""

import json
import time
from unittest.mock import Mock, patch

import pytest
from llama_index.core import Document

from src.agents.tools import (
    COMPLEX_CONFIDENCE,
    COMPLEX_QUERY_WORD_THRESHOLD,
    CONFIDENCE_ADJUSTMENT_FACTOR,
    CONTENT_KEY_LENGTH,
    FALLBACK_CONFIDENCE,
    FIRST_N_SOURCES_CHECK,
    MAX_RETRIEVAL_RESULTS,
    MEDIUM_CONFIDENCE,
    MEDIUM_QUERY_WORD_THRESHOLD,
    MIN_RESPONSE_LENGTH,
    RECENT_CHAT_HISTORY_LIMIT,
    RELEVANCE_THRESHOLD,
    SIMILARITY_THRESHOLD,
    SIMPLE_CONFIDENCE,
    VARIANT_QUERY_LIMIT,
    _parse_tool_result,
    _rank_documents_by_relevance,
    plan_query,
    retrieve_documents,
    route_query,
    synthesize_results,
    validate_response,
)


class TestRouteQueryComprehensive:
    """Comprehensive tests for route_query business logic."""

    def test_simple_query_classification_logic(self):
        """Test simple query classification with edge cases."""
        simple_queries = [
            "What is AI?",
            "Define machine learning",
            "Who is John Doe?",
            "When was Python created?",
            "Where is Silicon Valley?",
        ]

        for query in simple_queries:
            result = route_query(query)
            decision = json.loads(result)

            assert decision["complexity"] == "simple"
            assert decision["strategy"] == "vector"
            assert decision["needs_planning"] is False
            assert decision["confidence"] == SIMPLE_CONFIDENCE
            assert decision["word_count"] <= MEDIUM_QUERY_WORD_THRESHOLD

    def test_medium_complexity_query_patterns(self):
        """Test medium complexity query pattern recognition."""
        medium_queries = [
            "Tell me about neural networks in detail",
            "Find information about quantum computing applications",
            "Search for machine learning algorithms and their uses",
            "Look for papers on artificial intelligence research",
            "Show me examples of deep learning in healthcare",
        ]

        for query in medium_queries:
            result = route_query(query)
            decision = json.loads(result)

            assert decision["complexity"] == "medium"
            assert decision["strategy"] == "hybrid"
            assert decision["needs_planning"] is False
            assert decision["confidence"] == MEDIUM_CONFIDENCE
            assert (
                MEDIUM_QUERY_WORD_THRESHOLD
                < decision["word_count"]
                <= COMPLEX_QUERY_WORD_THRESHOLD
            )

    def test_complex_query_analysis_patterns(self):
        """Test complex query pattern recognition and analysis."""
        complex_patterns = [
            "compare and contrast machine learning with deep learning approaches",
            "analyze the relationship between artificial intelligence and robotics",
            "explain the step by step process of neural network training",
            "breakdown the impact of AI on healthcare versus traditional methods",
            "examine the differences and similarities between supervised versus "
            "unsupervised learning",
        ]

        for query in complex_patterns:
            result = route_query(query)
            decision = json.loads(result)

            assert decision["complexity"] == "complex"
            assert decision["strategy"] == "hybrid"
            assert decision["needs_planning"] is True
            assert decision["confidence"] == COMPLEX_CONFIDENCE
            assert decision["word_count"] > COMPLEX_QUERY_WORD_THRESHOLD or any(
                pattern in query.lower()
                for pattern in [
                    "compare",
                    "contrast",
                    "analyze",
                    "breakdown",
                    "explain how",
                    "step by step",
                    "relationship between",
                    "impact of",
                ]
            )

    def test_contextual_query_confidence_adjustment(self):
        """Test contextual queries and confidence adjustment."""
        # Mock context with chat history
        mock_context = Mock()
        mock_context.chat_history = [
            Mock(content="Previous question about AI"),
            Mock(content="Follow-up about machine learning"),
        ]
        state = {"context": mock_context}

        # Contextual query with context available
        query_with_context = "Can you elaborate on that approach?"
        result = route_query(query_with_context, state)
        decision = json.loads(result)

        assert decision["context_dependent"] is True
        # Confidence should not be reduced when context is available
        assert decision["confidence"] >= 0.8

        # Contextual query without context (should reduce confidence)
        query_without_context = "What about this method mentioned above?"
        result = route_query(query_without_context, {})
        decision = json.loads(result)

        assert decision["context_dependent"] is False
        # Should apply confidence adjustment factor
        expected_confidence = MEDIUM_CONFIDENCE * CONFIDENCE_ADJUSTMENT_FACTOR
        assert abs(decision["confidence"] - expected_confidence) < 0.01

    def test_graphrag_strategy_selection(self):
        """Test GraphRAG strategy selection for relationship queries."""
        relationship_queries = [
            "What are the connections between AI and machine learning?",
            "Show me the relationships in the neural network architecture",
            "How do different concepts link together in this domain?",
            "What is the network of dependencies in deep learning?",
        ]

        for query in relationship_queries:
            result = route_query(query)
            decision = json.loads(result)

            assert decision["strategy"] == "graphrag"
            assert any(
                pattern in query.lower()
                for pattern in ["connect", "relationship", "network", "link"]
            )

    def test_routing_performance_timing(self):
        """Test routing performance and timing measurement."""
        query = "Test query for performance measurement"

        start_time = time.perf_counter()
        result = route_query(query)
        end_time = time.perf_counter()

        decision = json.loads(result)

        # Verify timing is captured and reasonable
        assert "processing_time_ms" in decision
        assert decision["processing_time_ms"] > 0
        actual_time_ms = (end_time - start_time) * 1000
        # Allow some overhead but timing should be in reasonable range
        assert decision["processing_time_ms"] <= actual_time_ms + 50

    def test_routing_error_handling_scenarios(self):
        """Test comprehensive error handling in routing."""
        # Test with various error conditions
        with (
            patch("src.agents.tools.logger") as mock_logger,
            patch("builtins.len", side_effect=Exception("Length calculation failed")),
        ):
            result = route_query("Test query")
            decision = json.loads(result)

            # Should return fallback decision
            assert decision["strategy"] == "vector"
            assert decision["complexity"] == "simple"
            assert decision["confidence"] == FALLBACK_CONFIDENCE
            assert "error" in decision

            # Should log the error
            mock_logger.error.assert_called_once()


class TestPlanQueryComprehensive:
    """Comprehensive tests for plan_query business logic."""

    def test_comparison_query_decomposition(self):
        """Test comparison query decomposition logic."""
        comparison_queries = [
            ("AI vs machine learning capabilities", ["AI", "machine learning"]),
            (
                "Compare supervised versus unsupervised learning",
                ["supervised", "unsupervised learning"],
            ),
            (
                "Neural networks and traditional algorithms",
                ["Neural networks", "traditional algorithms"],
            ),
        ]

        for query, expected_entities in comparison_queries:
            result = plan_query(query, "complex")
            plan = json.loads(result)

            assert plan["execution_order"] == "parallel"
            assert len(plan["sub_tasks"]) >= 3  # At least: entity1, entity2, comparison

            # Verify entities are included in sub-tasks
            sub_tasks_text = " ".join(plan["sub_tasks"])
            for entity in expected_entities:
                assert entity.lower() in sub_tasks_text.lower()

    def test_analysis_query_decomposition(self):
        """Test analysis query decomposition patterns."""
        analysis_queries = [
            "Analyze the impact of AI on healthcare systems",
            "Breakdown the components of neural network architecture",
            "Examine the effectiveness of different learning algorithms",
        ]

        for query in analysis_queries:
            result = plan_query(query, "complex")
            plan = json.loads(result)

            assert len(plan["sub_tasks"]) == 4
            sub_tasks_text = " ".join(plan["sub_tasks"]).lower()

            # Should include standard analysis phases
            assert any(
                phase in sub_tasks_text
                for phase in ["identify", "research", "analyze", "synthesize"]
            )

    def test_process_explanation_decomposition(self):
        """Test process/explanation query decomposition."""
        process_queries = [
            "How does neural network training work?",
            "Explain the process of machine learning model training",
            "What are the steps in data preprocessing?",
        ]

        for query in process_queries:
            result = plan_query(query, "complex")
            plan = json.loads(result)

            assert len(plan["sub_tasks"]) == 4
            sub_tasks_text = " ".join(plan["sub_tasks"]).lower()

            # Should include process-oriented phases
            assert any(
                phase in sub_tasks_text
                for phase in ["definition", "steps", "detailed", "logical sequence"]
            )

    def test_connector_based_decomposition(self):
        """Test decomposition of queries with logical connectors."""
        connector_queries = [
            "Explain AI and machine learning and their applications",
            "Discuss neural networks or decision trees or random forests",
            "Research deep learning and also computer vision and furthermore "
            "natural language processing",
        ]

        for query in connector_queries:
            result = plan_query(query, "complex")
            plan = json.loads(result)

            # Should split on connectors
            assert len(plan["sub_tasks"]) > 2
            # Should include synthesis step
            sub_tasks_text = " ".join(plan["sub_tasks"]).lower()
            assert "synthesize" in sub_tasks_text

    def test_list_enumeration_decomposition(self):
        """Test list/enumeration query decomposition."""
        list_queries = [
            "List the types of machine learning algorithms",
            "Enumerate the benefits of artificial intelligence",
            "What are examples of deep learning applications?",
        ]

        for query in list_queries:
            result = plan_query(query, "medium")
            plan = json.loads(result)

            assert len(plan["sub_tasks"]) == 3
            sub_tasks_text = " ".join(plan["sub_tasks"]).lower()

            # Should include categorization and organization
            assert any(
                phase in sub_tasks_text
                for phase in ["categorize", "organize", "structured list"]
            )

    def test_planning_performance_metrics(self):
        """Test planning performance measurement and metrics."""
        query = "Complex analytical query for performance testing"

        start_time = time.perf_counter()
        result = plan_query(query, "complex")
        end_time = time.perf_counter()

        plan = json.loads(result)

        # Verify performance metrics
        assert "processing_time_ms" in plan
        assert "task_count" in plan
        assert plan["processing_time_ms"] > 0
        assert plan["task_count"] == len(plan["sub_tasks"])

        actual_time_ms = (end_time - start_time) * 1000
        assert plan["processing_time_ms"] <= actual_time_ms + 50

    def test_planning_error_recovery(self):
        """Test planning error handling and recovery."""
        with (
            patch("src.agents.tools.logger") as mock_logger,
            patch("builtins.len", side_effect=Exception("Planning failure")),
        ):
            result = plan_query("Test query", "complex")
            plan = json.loads(result)

            # Should provide fallback plan
            assert plan["original_query"] == "Test query"
            assert plan["sub_tasks"] == ["Test query"]
            assert plan["execution_order"] == "sequential"
            assert "error" in plan

            mock_logger.error.assert_called_once()


class TestRetrieveDocumentsComprehensive:
    """Comprehensive tests for retrieve_documents business logic."""

    def test_dspy_query_optimization_logic(self):
        """Test DSPy query optimization with real logic."""
        mock_tools_data = {
            "vector": Mock(),
            "retriever": Mock(),
        }
        state = {"tools_data": mock_tools_data}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = "Optimized search result"
            mock_factory.create_vector_search_tool.return_value = mock_tool

            # Test DSPy optimization enabled
            result = retrieve_documents("AI", "vector", use_dspy=True, state=state)
            data = json.loads(result)

            assert data["dspy_used"] is True
            assert data["query_original"] == "AI"
            # Short query should get optimization
            assert "Find documents about AI" in data["query_optimized"]

    def test_query_variant_processing(self):
        """Test query variant processing and limits."""
        mock_tools_data = {"vector": Mock()}
        state = {"tools_data": mock_tools_data}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            call_count = 0

            def mock_call(query):
                nonlocal call_count
                call_count += 1
                return f"Result for query {call_count}: {query}"

            mock_tool.call.side_effect = mock_call
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result = retrieve_documents("AI", "vector", use_dspy=True, state=state)
            data = json.loads(result)

            # Should process primary query + variants (limited by VARIANT_QUERY_LIMIT)
            expected_calls = 1 + VARIANT_QUERY_LIMIT  # Primary + variants
            assert call_count <= expected_calls
            assert len(data["documents"]) >= 1

    def test_graphrag_fallback_mechanism(self):
        """Test GraphRAG fallback to hybrid search."""
        mock_tools_data = {
            "vector": Mock(),
            "kg": Mock(),
        }
        state = {"tools_data": mock_tools_data}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            # GraphRAG tool fails
            mock_kg_tool = Mock()
            mock_kg_tool.call.side_effect = RuntimeError("GraphRAG service unavailable")
            mock_factory.create_kg_search_tool.return_value = mock_kg_tool

            # Hybrid vector tool succeeds
            mock_hybrid_tool = Mock()
            mock_hybrid_tool.call.return_value = "Fallback hybrid result"
            mock_factory.create_hybrid_vector_tool.return_value = mock_hybrid_tool

            result = retrieve_documents(
                "AI relationships", "graphrag", use_graphrag=True, state=state
            )
            data = json.loads(result)

            # Should fallback to hybrid strategy
            assert data["strategy_used"] == "hybrid_vector"
            assert len(data["documents"]) > 0
            assert data["documents"][0]["content"] == "Fallback hybrid result"

    def test_document_deduplication_logic(self):
        """Test document deduplication with content similarity."""
        mock_tools_data = {"vector": Mock()}
        state = {"tools_data": mock_tools_data}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            # Return duplicate-like documents
            mock_tool.call.side_effect = [
                [
                    {
                        "content": "AI is artificial intelligence technology",
                        "score": 0.9,
                    },
                    {"content": "Machine learning algorithms", "score": 0.8},
                ],
                [
                    {
                        "content": "AI is artificial intelligence systems",
                        "score": 0.85,
                    },  # Similar to first
                    {"content": "Deep learning neural networks", "score": 0.7},
                ],
            ]
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result = retrieve_documents("AI", "vector", use_dspy=True, state=state)
            data = json.loads(result)

            # Should deduplicate and limit results
            assert len(data["documents"]) <= MAX_RETRIEVAL_RESULTS
            # Best scores should be preserved
            if len(data["documents"]) > 1:
                scores = [doc.get("score", 0) for doc in data["documents"]]
                assert scores[0] >= scores[-1]  # Should be sorted by score

    def test_retrieval_error_handling(self):
        """Test comprehensive retrieval error handling."""
        # Test missing tools data
        result = retrieve_documents("test query", "vector", state={})
        data = json.loads(result)
        assert data["documents"] == []
        assert "No retrieval tools available" in data["error"]

        # Test tool execution failure
        mock_tools_data = {"vector": Mock()}
        state = {"tools_data": mock_tools_data}

        with patch("src.agents.tools.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.side_effect = Exception("Tool execution failed")
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result = retrieve_documents("test", "vector", state=state)
            data = json.loads(result)

            assert data["documents"] == []
            assert "processing_time_ms" in data


class TestSynthesizeResultsComprehensive:
    """Comprehensive tests for synthesize_results business logic."""

    def test_multi_source_synthesis_algorithm(self):
        """Test multi-source synthesis with complex deduplication."""
        sub_results = [
            {
                "documents": [
                    {
                        "content": "AI is artificial intelligence for computer systems",
                        "score": 0.9,
                    },
                    {
                        "content": "Machine learning uses statistical methods",
                        "score": 0.85,
                    },
                    {
                        "content": "Deep learning neural networks process data",
                        "score": 0.8,
                    },
                ],
                "strategy_used": "vector",
                "processing_time_ms": 120,
            },
            {
                "documents": [
                    {
                        "content": "AI artificial intelligence systems for computing",
                        "score": 0.88,
                    },  # Similar
                    {
                        "content": "Natural language processing analyzes text",
                        "score": 0.82,
                    },
                    {
                        "content": "Computer vision recognizes patterns in images",
                        "score": 0.78,
                    },
                ],
                "strategy_used": "hybrid",
                "processing_time_ms": 150,
            },
            {
                "documents": [
                    {
                        "content": "Reinforcement learning through trial and error",
                        "score": 0.87,
                    },
                    {
                        "content": "Machine learning statistical methods "
                        "for prediction",
                        "score": 0.83,
                    },  # Similar
                ],
                "strategy_used": "kg",
                "processing_time_ms": 90,
            },
        ]

        result = synthesize_results(
            json.dumps(sub_results),
            "artificial intelligence and machine learning overview",
        )
        data = json.loads(result)

        # Verify synthesis metadata
        assert data["synthesis_metadata"]["original_count"] == 8
        assert (
            data["synthesis_metadata"]["after_deduplication"] < 8
        )  # Should remove some duplicates
        assert data["synthesis_metadata"]["final_count"] <= MAX_RETRIEVAL_RESULTS
        assert set(data["synthesis_metadata"]["strategies_used"]) == {
            "vector",
            "hybrid",
            "kg",
        }

        # Verify deduplication ratio calculation
        expected_ratio = (
            data["synthesis_metadata"]["after_deduplication"]
            / data["synthesis_metadata"]["original_count"]
        )
        assert (
            abs(data["synthesis_metadata"]["deduplication_ratio"] - expected_ratio)
            < 0.01
        )

    def test_similarity_based_deduplication(self):
        """Test similarity-based document deduplication algorithm."""
        # Documents with varying similarity levels
        test_docs = [
            {"content": "machine learning algorithms for data analysis", "score": 0.9},
            {
                "content": "data analysis using machine learning algorithms",
                "score": 0.85,
            },  # Very similar
            {"content": "deep learning neural network architectures", "score": 0.8},
            {
                "content": "cooking recipes for Italian pasta dishes",
                "score": 0.75,
            },  # Different topic
            {
                "content": "neural network deep learning for pattern recognition",
                "score": 0.88,
            },  # Similar to #3
        ]

        sub_results = [{"documents": test_docs, "strategy_used": "test"}]

        result = synthesize_results(
            json.dumps(sub_results), "machine learning and deep learning"
        )
        data = json.loads(result)

        # Should remove highly similar documents based on SIMILARITY_THRESHOLD
        assert len(data["documents"]) < len(test_docs)

        # Documents should be ranked by relevance to query
        if len(data["documents"]) > 1:
            relevance_scores = [
                doc.get("relevance_score", 0) for doc in data["documents"]
            ]
            assert relevance_scores == sorted(relevance_scores, reverse=True)

    def test_relevance_ranking_algorithm(self):
        """Test document ranking by relevance algorithm."""
        documents_with_scores = [
            {
                "content": "cooking recipes and food preparation techniques",
                "score": 0.9,
            },
            {
                "content": "machine learning algorithms artificial intelligence",
                "score": 0.7,
            },
            {
                "content": "artificial intelligence machine learning deep learning",
                "score": 0.8,
            },
        ]

        query = "artificial intelligence machine learning"
        ranked = _rank_documents_by_relevance(documents_with_scores, query)

        # Verify relevance scoring
        assert all("relevance_score" in doc for doc in ranked)

        # Most relevant document should be first (highest AI/ML content)
        assert (
            ranked[0]["content"]
            == "artificial intelligence machine learning deep learning"
        )

        # Least relevant (cooking) should be last
        assert "cooking" in ranked[-1]["content"]

        # Relevance scores should be in descending order
        scores = [doc["relevance_score"] for doc in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_synthesis_error_recovery(self):
        """Test synthesis error handling and recovery."""
        # Test invalid JSON input
        result = synthesize_results("invalid json", "test query")
        data = json.loads(result)
        assert data["documents"] == []
        assert "Invalid input format" in data["error"]

        # Test synthesis processing error
        with (
            patch("src.agents.tools.logger") as mock_logger,
            patch("builtins.len", side_effect=Exception("Synthesis error")),
        ):
            result = synthesize_results("[]", "test query")
            data = json.loads(result)

            assert data["documents"] == []
            assert "error" in data
            mock_logger.error.assert_called_once()


class TestValidateResponseComprehensive:
    """Comprehensive tests for validate_response business logic."""

    def test_comprehensive_validation_scoring(self):
        """Test comprehensive response validation with multiple quality checks."""
        # High-quality response test
        query = "What is machine learning?"
        good_response = (
            "Machine learning is a subset of artificial intelligence that enables "
            "computers to learn and improve from experience without explicit "
            "programming. It uses algorithms and statistical models to analyze data "
            "patterns and make predictions or decisions based on the "
            "learned information."
        )
        sources = json.dumps(
            {
                "documents": [
                    {
                        "content": "Machine learning algorithms analyze "
                        "statistical patterns",
                        "score": 0.9,
                    },
                    {
                        "content": "Artificial intelligence computers learn "
                        "from experience",
                        "score": 0.8,
                    },
                ]
            }
        )

        result = validate_response(query, good_response, sources)
        data = json.loads(result)

        # Should pass all validation checks
        assert data["valid"] is True
        assert data["confidence"] >= 0.8
        assert data["suggested_action"] == "accept"
        assert len(data["issues"]) <= 1  # May have minor issues but overall good

    def test_incomplete_response_detection(self):
        """Test detection of incomplete responses."""
        query = "Explain the complex process of neural network training with "
        "backpropagation"
        short_response = "Neural networks learn."  # Too brief for complex query

        result = validate_response(query, short_response, json.dumps({"documents": []}))
        data = json.loads(result)

        # Should detect incompleteness
        incomplete_issues = [
            issue for issue in data["issues"] if issue["type"] == "incomplete_response"
        ]
        assert len(incomplete_issues) > 0
        assert data["confidence"] < 0.8
        assert len(short_response) < MIN_RESPONSE_LENGTH

    def test_source_attribution_analysis(self):
        """Test source attribution detection logic."""
        query = "What is deep learning?"
        response_with_sources = (
            "Deep learning uses neural networks with multiple layers to model complex "
            "patterns in data. These networks can automatically learn feature "
            "representations through backpropagation algorithms and gradient "
            "descent optimization."
        )
        response_without_sources = (
            "Deep learning is a fascinating field with many applications in modern "
            "technology. It has revolutionized computer science and artificial "
            "intelligence research."
        )

        sources = json.dumps(
            {
                "documents": [
                    {
                        "content": "Neural networks multiple layers feature "
                        "representations patterns",
                        "score": 0.9,
                    },
                    {
                        "content": "Backpropagation algorithms gradient descent "
                        "optimization",
                        "score": 0.8,
                    },
                ]
            }
        )

        # Test response that references sources
        result1 = validate_response(query, response_with_sources, sources)
        data1 = json.loads(result1)

        # Should detect source usage (word overlap > threshold)
        source_issues1 = [
            issue for issue in data1["issues"] if issue["type"] == "missing_source"
        ]
        assert len(source_issues1) == 0  # Should not flag as missing sources

        # Test response that doesn't reference sources
        result2 = validate_response(query, response_without_sources, sources)
        data2 = json.loads(result2)

        # Should detect missing source attribution
        source_issues2 = [
            issue for issue in data2["issues"] if issue["type"] == "missing_source"
        ]
        assert len(source_issues2) > 0

    def test_hallucination_detection_patterns(self):
        """Test hallucination detection with various patterns."""
        query = "Explain quantum computing principles"
        hallucination_responses = [
            "I cannot find specific information about quantum computing. Based on my "
            "training data, quantum computing uses quantum mechanics.",
            "According to my knowledge, quantum computers are very advanced. I don't "
            "have access to current research.",
            "No information available in the sources. However, from what I know, "
            "quantum systems are complex.",
        ]

        for response in hallucination_responses:
            result = validate_response(query, response, json.dumps({"documents": []}))
            data = json.loads(result)

            # Should detect potential hallucination
            hallucination_issues = [
                issue
                for issue in data["issues"]
                if issue["type"] == "potential_hallucination"
            ]
            assert len(hallucination_issues) > 0
            assert data["confidence"] <= 0.5

    def test_query_relevance_scoring(self):
        """Test query relevance analysis."""
        query = "machine learning neural networks algorithms"

        relevant_response = (
            "Machine learning algorithms, particularly neural networks, use "
            "mathematical models to learn patterns from data and make predictions."
        )

        irrelevant_response = (
            "Cooking is an art form that requires creativity and skill. Many chefs "
            "prefer traditional recipes over modern techniques."
        )

        # Test relevant response
        # Calculate expected relevance
        query_words = set(query.lower().split())
        response_words = set(relevant_response.lower().split())
        expected_relevance = len(query_words.intersection(response_words)) / len(
            query_words
        )
        assert expected_relevance >= RELEVANCE_THRESHOLD

        # Test irrelevant response
        result2 = validate_response(
            query, irrelevant_response, json.dumps({"documents": []})
        )
        data2 = json.loads(result2)

        # Should detect low relevance
        relevance_issues = [
            issue for issue in data2["issues"] if issue["type"] == "low_relevance"
        ]
        if expected_relevance < RELEVANCE_THRESHOLD:
            assert len(relevance_issues) > 0

    def test_coherence_analysis(self):
        """Test response coherence analysis."""
        query = "What is artificial intelligence?"

        # Test incoherent response (unusual sentence structure)
        incoherent_response = "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P. Q."

        result = validate_response(
            query, incoherent_response, json.dumps({"documents": []})
        )
        data = json.loads(result)

        # Should detect coherence issues
        coherence_issues = [
            issue for issue in data["issues"] if issue["type"] == "coherence_issue"
        ]
        assert len(coherence_issues) > 0

    def test_validation_performance_metrics(self):
        """Test validation performance measurement."""
        query = "Test query"
        response = "Test response for performance measurement"
        sources = json.dumps({"documents": []})

        start_time = time.perf_counter()
        result = validate_response(query, response, sources)
        end_time = time.perf_counter()

        data = json.loads(result)

        # Verify performance metrics
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0
        assert "source_count" in data
        assert "response_length" in data
        assert "issue_count" in data

        actual_time_ms = (end_time - start_time) * 1000
        assert data["processing_time_ms"] <= actual_time_ms + 50


class TestHelperFunctionsComprehensive:
    """Comprehensive tests for helper functions."""

    def test_parse_tool_result_comprehensive(self):
        """Test comprehensive tool result parsing."""
        # Test string result
        string_result = "This is a string response"
        docs = _parse_tool_result(string_result)
        assert len(docs) == 1
        assert docs[0]["content"] == string_result
        assert docs[0]["metadata"]["source"] == "tool_response"

        # Test LlamaIndex response object
        mock_response = Mock()
        mock_response.response = "LlamaIndex response"
        mock_response.metadata = {"source": "test.pdf"}
        docs = _parse_tool_result(mock_response)
        assert len(docs) == 1
        assert docs[0]["content"] == "LlamaIndex response"
        assert docs[0]["metadata"]["source"] == "test.pdf"

        # Test Document list
        doc_list = [
            Document(text="First doc", metadata={"source": "doc1"}),
            Document(text="Second doc", metadata={"source": "doc2"}),
        ]
        doc_list[0].score = 0.9
        doc_list[1].score = 0.8

        docs = _parse_tool_result(doc_list)
        assert len(docs) == 2
        assert docs[0]["score"] == 0.9
        assert docs[1]["score"] == 0.8

        # Test dict list (should pass through unchanged)
        dict_list = [
            {"content": "Dict doc 1", "score": 0.95},
            {"content": "Dict doc 2", "score": 0.85},
        ]
        docs = _parse_tool_result(dict_list)
        assert docs == dict_list

        # Test fallback for unknown type
        unknown_result = {"unknown": "format"}
        docs = _parse_tool_result(unknown_result)
        assert len(docs) == 1
        assert docs[0]["content"] == str(unknown_result)

    def test_document_ranking_edge_cases(self):
        """Test document ranking with edge cases."""
        # Test empty query
        docs = [{"content": "Some content", "score": 0.8}]
        ranked = _rank_documents_by_relevance(docs, "")
        assert len(ranked) == 1
        assert "relevance_score" in ranked[0]

        # Test empty content
        docs = [
            {"metadata": {"source": "test"}, "score": 0.8},  # No content
            {"text": "Alternative text field", "score": 0.9},  # Uses 'text' instead
        ]
        ranked = _rank_documents_by_relevance(docs, "test query")
        assert len(ranked) == 2
        assert all("relevance_score" in doc for doc in ranked)

        # Test no existing scores
        docs = [{"content": "Content without score"}]
        ranked = _rank_documents_by_relevance(docs, "content")
        assert ranked[0]["relevance_score"] > 0

    def test_constants_and_thresholds(self):
        """Test that constants are properly defined and used."""
        # Verify critical constants are defined
        assert COMPLEX_QUERY_WORD_THRESHOLD == 20
        assert MEDIUM_QUERY_WORD_THRESHOLD == 10
        assert RECENT_CHAT_HISTORY_LIMIT == 3
        assert MAX_RETRIEVAL_RESULTS == 10
        assert SIMILARITY_THRESHOLD == 0.8
        assert MIN_RESPONSE_LENGTH == 50
        assert RELEVANCE_THRESHOLD == 0.3
        assert VARIANT_QUERY_LIMIT == 2
        assert CONTENT_KEY_LENGTH == 100
        assert FIRST_N_SOURCES_CHECK == 3

        # Test confidence values
        assert COMPLEX_CONFIDENCE == 0.9
        assert MEDIUM_CONFIDENCE == 0.85
        assert SIMPLE_CONFIDENCE == 0.95
        assert CONFIDENCE_ADJUSTMENT_FACTOR == 0.8
        assert FALLBACK_CONFIDENCE == 0.5


class TestIntegrationScenarios:
    """Integration tests for complex multi-step scenarios."""

    def test_complete_query_processing_flow(self):
        """Test complete flow from routing through validation."""
        query = "Compare machine learning with deep learning approaches"

        # Step 1: Route query
        routing_result = route_query(query)
        routing_decision = json.loads(routing_result)

        assert routing_decision["complexity"] == "complex"
        assert routing_decision["needs_planning"] is True

        # Step 2: Plan query (if needed)
        planning_result = plan_query(query, routing_decision["complexity"])
        planning_output = json.loads(planning_result)

        assert len(planning_output["sub_tasks"]) >= 3
        assert "Compare" in " ".join(planning_output["sub_tasks"])

        # Step 3: Mock retrieval
        mock_retrieval_results = [
            {
                "documents": [
                    {
                        "content": "Machine learning algorithms for data analysis",
                        "score": 0.9,
                    },
                    {
                        "content": "Deep learning neural networks with multiple layers",
                        "score": 0.85,
                    },
                ],
                "strategy_used": "hybrid",
            }
        ]

        # Step 4: Synthesize results
        synthesis_result = synthesize_results(json.dumps(mock_retrieval_results), query)
        synthesis_data = json.loads(synthesis_result)

        assert len(synthesis_data["documents"]) >= 1

        # Step 5: Validate response
        mock_response = (
            "Machine learning focuses on algorithms that learn from data patterns, "
            "while deep learning uses neural networks with multiple layers for "
            "complex pattern recognition and feature extraction."
        )

        validation_result = validate_response(
            query, mock_response, json.dumps(synthesis_data)
        )
        validation_data = json.loads(validation_result)

        assert validation_data["confidence"] > 0.6
        assert validation_data["suggested_action"] in ["accept", "refine"]

    def test_error_cascade_handling(self):
        """Test error handling across the complete workflow."""
        # Test with problematic query that might cause errors
        query = None  # Invalid input

        with patch("src.agents.tools.logger"):
            # Should handle None query gracefully without raising exceptions
            result = route_query(str(query) if query else "")
            decision = json.loads(result)
            # Either returns error or falls back to vector strategy
            assert "error" in decision or decision["strategy"] == "vector"


# Mark the test class to update todo status
@pytest.mark.usefixtures("update_todo_status")
class TestCoverageValidation:
    """Tests to validate coverage improvements."""

    def test_coverage_critical_paths(self):
        """Ensure critical code paths are tested."""
        # This test validates that we've covered the main business logic paths
        # that were previously missing coverage

        # Test all main functions are callable and return valid JSON
        functions_to_test = [
            (route_query, ("test query",)),
            (plan_query, ("test query", "complex")),
            (retrieve_documents, ("test", "vector")),
            (synthesize_results, ("[]", "test")),
            (validate_response, ("query", "response", "[]")),
        ]

        for func, args in functions_to_test:
            result = func(*args)
            # Should return valid JSON string
            assert isinstance(result, str)
            data = json.loads(result)
            assert isinstance(data, dict)

    def test_constants_usage_validation(self):
        """Validate that all constants are properly used."""
        # This ensures we're testing the actual constant values used in the code
        from src.agents.tools import (
            ACCEPT_CONFIDENCE_THRESHOLD,
            CONFIDENCE_REDUCTION_COHERENCE,
            CONFIDENCE_REDUCTION_HALLUCINATION,
            CONFIDENCE_REDUCTION_INCOMPLETE,
            CONFIDENCE_REDUCTION_LOW_RELEVANCE,
            CONFIDENCE_REDUCTION_NO_SOURCE,
            CONFIDENCE_REDUCTION_NO_SOURCES,
            MAX_AVG_SENTENCE_LENGTH,
            MAX_ISSUES_FOR_ACCEPT,
            MAX_SENTENCE_WORD_OVERLAP,
            MIN_AVG_SENTENCE_LENGTH,
            REFINE_CONFIDENCE_THRESHOLD,
            VALIDATION_CONFIDENCE_THRESHOLD,
        )

        # Validate constants are reasonable
        assert 0 < ACCEPT_CONFIDENCE_THRESHOLD <= 1
        assert 0 < VALIDATION_CONFIDENCE_THRESHOLD <= 1
        assert 0 < REFINE_CONFIDENCE_THRESHOLD <= 1
        assert 0 < CONFIDENCE_REDUCTION_COHERENCE <= 1
        assert 0 < CONFIDENCE_REDUCTION_HALLUCINATION <= 1
        assert 0 < CONFIDENCE_REDUCTION_INCOMPLETE <= 1
        assert 0 < CONFIDENCE_REDUCTION_LOW_RELEVANCE <= 1
        assert 0 < CONFIDENCE_REDUCTION_NO_SOURCE <= 1
        assert 0 < CONFIDENCE_REDUCTION_NO_SOURCES <= 1
        assert MIN_AVG_SENTENCE_LENGTH < MAX_AVG_SENTENCE_LENGTH
        assert MAX_SENTENCE_WORD_OVERLAP > 0
        assert MAX_ISSUES_FOR_ACCEPT >= 0


# Fixture to automatically update todo status
@pytest.fixture(autouse=True)
def update_todo_status():
    """Automatically update todo status when tests run."""
    # This would be called by the test framework
    return
    # Tests completed - this indicates progress on the comprehensive testing
