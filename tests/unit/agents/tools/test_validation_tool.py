"""Unit tests for validation.validate_response (quality checks)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.agents.tools.validation import validate_response

pytestmark = pytest.mark.unit


class TestValidateResponse:
    """Quality checks for responses including sources and coherence."""

    def test_validate_response_good_quality(self):
        """Test validation of high-quality responses with sources."""
        query = "What is machine learning?"
        response = (
            "Machine learning is a subset of artificial intelligence that focuses on "
            "algorithms that can learn from data without being explicitly programmed."
        )
        sources = json.dumps(
            [
                {
                    "content": (
                        "Machine learning is a method of data analysis that "
                        "automates analytical model building using algorithms "
                        "that iteratively learn from data"
                    ),
                    "score": 0.9,
                }
            ]
        )
        result_json = validate_response.invoke(
            {
                "query": query,
                "response": response,
                "sources": sources,
            }
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
        """Test validation of responses that are too short for query complexity."""
        query = "Explain neural networks in detail"
        response = "Neural networks are good."
        sources = json.dumps([{"content": "detailed neural network info"}])
        result_json = validate_response.invoke(
            {
                "query": query,
                "response": response,
                "sources": sources,
            }
        )
        result = json.loads(result_json)
        assert any(issue["type"] == "incomplete_response" for issue in result["issues"])
        assert result["confidence"] < 1.0

    def test_validate_response_no_sources(self):
        """Test validation behavior when no sources are provided with the response."""
        result_json = validate_response.invoke(
            {
                "query": "test query",
                "response": "test response",
                "sources": "[]",
            }
        )
        result = json.loads(result_json)
        assert any(issue["type"] == "no_sources" for issue in result["issues"])
        assert result["confidence"] < 1.0

    def test_validate_response_missing_source_attribution(self):
        """Test detection of responses that lack proper source attribution."""
        query = "What is AI?"
        response = "Artificial intelligence is about computers thinking like humans."
        sources = json.dumps(
            [
                {
                    "content": (
                        "Machine learning algorithms use statistical techniques to "
                        "learn patterns from data"
                    )
                }
            ]
        )
        result_json = validate_response.invoke(
            {
                "query": query,
                "response": response,
                "sources": sources,
            }
        )
        result = json.loads(result_json)
        assert any(
            issue["type"] in ("missing_source", "no_sources")
            for issue in result["issues"]
        )
        assert result["confidence"] <= 1.0

    def test_validate_response_hallucination_indicator(self):
        """Test detection of potential hallucinations using indicator phrases."""
        result = json.loads(
            validate_response.invoke(
                {
                    "query": "q",
                    "response": "According to my knowledge this is true",
                    "sources": json.dumps(
                        [
                            {
                                "content": (
                                    "some content referencing knowledge systems "
                                    "human intelligence"
                                )
                            }
                        ]
                    ),
                }
            )
        )
        assert any(
            issue["type"] == "potential_hallucination" for issue in result["issues"]
        )
        assert result["confidence"] <= 1.0

    def test_validate_response_relevance_and_actions(self):
        """Test confidence scoring and action suggestions based on response quality."""
        high_conf = json.loads(
            validate_response.invoke(
                {
                    "query": "What is AI?",
                    "response": (
                        "Artificial intelligence uses systems human intelligence"
                    ),
                    "sources": json.dumps(
                        [
                            {
                                "content": (
                                    "Artificial intelligence systems human intelligence"
                                )
                            }
                        ]
                    ),
                }
            )
        )
        low_conf = json.loads(
            validate_response.invoke(
                {
                    "query": "What is AI?",
                    "response": "I don't know.",
                    "sources": "[]",
                }
            )
        )
        assert high_conf["suggested_action"] in ["accept", "refine"]
        assert low_conf["suggested_action"] == "regenerate"

    def test_validate_response_error_handling(self):
        """Test error handling when validation process encounters exceptions."""
        with patch(
            "src.agents.tools.validation.time.perf_counter",
            side_effect=RuntimeError("Timer error"),
        ):
            result_json = validate_response.invoke(
                {
                    "query": "query",
                    "response": "response",
                    "sources": "[]",
                }
            )
            result = json.loads(result_json)
            assert not result["valid"]
        assert result["confidence"] == 0.0
        assert result["suggested_action"] == "regenerate"
        assert "error" in result
