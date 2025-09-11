"""Unit tests for synthesis.synthesize_results (combination/ranking)."""

from __future__ import annotations

import json

import pytest

from src.agents.tools.synthesis import synthesize_results

pytestmark = pytest.mark.unit


class TestSynthesizeResults:
    """Combine and rank results from multiple retrieval batches."""

    def test_synthesize_results_basic(self):
        """Test basic synthesis with multiple sub-results from different strategies."""
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
            {
                "sub_results": json.dumps(sub_results),
                "original_query": "test query",
            }
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
        """Test deduplication with duplicate documents to verify removal."""
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
            },
            {"documents": [{"content": "deep learning uses neural networks"}]},
        ]
        result_json = synthesize_results.invoke(
            {
                "sub_results": json.dumps(sub_results),
                "original_query": "AI query",
            }
        )
        result = json.loads(result_json)

        assert result["synthesis_metadata"]["original_count"] == 3
        assert result["synthesis_metadata"]["after_deduplication"] <= 2
        assert result["synthesis_metadata"]["deduplication_ratio"] <= 1.0

    def test_synthesize_results_ranking(self):
        """Test document ranking with scored documents for proper ordering."""
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

        assert len(result["documents"]) == 2
        assert "relevance_score" in result["documents"][0]

    def test_synthesize_results_invalid_json(self):
        """Test error handling when invalid JSON is provided as input."""
        result_json = synthesize_results.invoke(
            {
                "sub_results": "invalid json",
                "original_query": "test query",
            }
        )
        result = json.loads(result_json)

        assert result["documents"] == []
        assert "error" in result

    def test_synthesize_results_empty_input(self):
        """Test handling of empty input arrays and ensure graceful degradation."""
        result_json = synthesize_results.invoke(
            {
                "sub_results": json.dumps([]),
                "original_query": "test query",
            }
        )
        result = json.loads(result_json)

        assert result["documents"] == []
        assert result["synthesis_metadata"]["original_count"] == 0

    def test_synthesize_results_max_limit(self):
        """Test synthesis respects max document limits with excess results."""
        large_results = [
            {"documents": [{"content": f"doc {i}", "score": 0.8} for i in range(15)]}
        ]
        result_json = synthesize_results.invoke(
            {
                "sub_results": json.dumps(large_results),
                "original_query": "test query",
            }
        )
        result = json.loads(result_json)

        assert len(result["documents"]) <= 10
