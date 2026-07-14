"""Unit tests for synthesis.synthesize_results (combination/ranking)."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from src.agents.tools.synthesis import synthesize_results

pytestmark = pytest.mark.unit


def _invoke_synthesis(
    sub_results: object, *, original_query: str = "test query"
) -> dict[str, Any]:
    turn_id = "synthesis-turn"
    if isinstance(sub_results, list):
        retrieval_results = [
            {"turn_id": turn_id, "retrieval_id": f"batch-{index}", **result}
            if isinstance(result, dict)
            else result
            for index, result in enumerate(sub_results)
        ]
    else:
        retrieval_results = sub_results
    command = synthesize_results.func(  # type: ignore[attr-defined]
        state={
            "messages": [HumanMessage(content=original_query)],
            "retrieval_results": retrieval_results,
            "turn_id": turn_id,
        },
        tool_call_id="synthesis-call",
    )
    assert isinstance(command, Command)
    assert isinstance(command.update, dict)
    payload = cast(dict[str, Any], command.update["synthesis_result"])
    message = command.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert isinstance(message.content, str)
    assert json.loads(message.content) == payload
    return payload


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
        result = _invoke_synthesis(sub_results)

        assert isinstance(result, dict)
        assert "documents" in result
        assert "synthesis_metadata" in result
        assert "original_query" in result
        assert len(result["documents"]) == 2
        assert result["synthesis_metadata"]["original_count"] == 2
        assert result["synthesis_metadata"]["final_count"] == 2
        assert "vector" in result["synthesis_metadata"]["strategies_used"]
        assert "hybrid" in result["synthesis_metadata"]["strategies_used"]

    def test_synthesize_results_deduplicates_stable_node_identity(self):
        """Stable node identities are deduplicated without content heuristics."""
        sub_results = [
            {
                "documents": [
                    {
                        "id": "node-1",
                        "content": (
                            "machine learning is a subset of artificial intelligence"
                        ),
                    }
                ]
            },
            {
                "documents": [
                    {
                        "id": "node-1",
                        "content": (
                            "machine learning is a subset of artificial intelligence"
                        ),
                    }
                ]
            },
            {"documents": [{"content": "deep learning uses neural networks"}]},
        ]
        result = _invoke_synthesis(sub_results, original_query="AI query")

        assert result["synthesis_metadata"]["original_count"] == 3
        assert result["synthesis_metadata"]["after_deduplication"] == 2
        assert result["synthesis_metadata"]["deduplication_ratio"] == 0.67

    def test_synthesize_results_preserves_native_ranking(self):
        """Synthesis leaves the retrieval/reranker order unchanged."""
        sub_results = [
            {
                "documents": [
                    {"content": "machine learning algorithms", "score": 0.7},
                    {"content": "artificial intelligence overview", "score": 0.9},
                ]
            }
        ]
        result = _invoke_synthesis(sub_results, original_query="machine learning")

        assert len(result["documents"]) == 2
        assert [document["score"] for document in result["documents"]] == [0.7, 0.9]
        assert all(
            "relevance_score" not in document for document in result["documents"]
        )

    def test_synthesize_results_invalid_state(self):
        """Malformed persisted retrieval state fails closed."""
        result = _invoke_synthesis("invalid state")

        assert result["documents"] == []
        assert "error" in result

    def test_synthesize_results_empty_input(self):
        """Test handling of empty input arrays and ensure graceful degradation."""
        result = _invoke_synthesis([])

        assert result["documents"] == []
        assert result["synthesis_metadata"]["original_count"] == 0

    def test_synthesize_results_ignores_malformed_historical_batches(self):
        """Only current-turn evidence participates in validation/synthesis."""
        result = _invoke_synthesis(
            [
                {"turn_id": "historical-turn", "legacy_shape": True},
                {
                    "documents": [{"id": "current", "content": "evidence"}],
                },
            ]
        )

        assert [document["id"] for document in result["documents"]] == ["current"]

    def test_synthesize_results_max_limit(self):
        """Test synthesis respects max document limits with excess results."""
        large_results = [
            {"documents": [{"content": f"doc {i}", "score": 0.8} for i in range(15)]}
        ]
        result = _invoke_synthesis(large_results)

        assert len(result["documents"]) <= 10

    def test_synthesize_results_fairly_interleaves_parallel_batches(self):
        """A large first batch cannot erase a later query's top evidence."""
        result = _invoke_synthesis(
            [
                {"documents": [{"id": f"a-{index}"} for index in range(10)]},
                {"documents": [{"id": "b-0"}]},
            ]
        )

        assert [document["id"] for document in result["documents"][:3]] == [
            "a-0",
            "b-0",
            "a-1",
        ]
