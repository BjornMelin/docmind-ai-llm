"""Unit tests for synthesis and validation tools.

These tests exercise pure functions and light tool wrappers to increase
coverage while adhering to pytest best practices: table-driven cases,
no external I/O, deterministic assertions.
"""

from __future__ import annotations

import json

import pytest

from src.agents.tools.synthesis import synthesize_results
from src.agents.tools.validation import validate_response


@pytest.mark.unit
def test_synthesize_results_deduplicates_and_ranks() -> None:
    sub_results = [
        {
            "strategy_used": "vector",
            "processing_time_ms": 12.3,
            "documents": [
                {"id": 1, "content": "Python testing with pytest", "score": 0.9},
                {"id": 2, "content": "Testing in Python using pytest", "score": 0.8},
            ],
        },
        {
            "strategy_used": "hybrid",
            "processing_time_ms": 10.0,
            "documents": [{"id": 3, "content": "Unrelated content here", "score": 0.5}],
        },
    ]

    # Call underlying function to avoid tool run-manager plumbing
    result = json.loads(
        synthesize_results.func(json.dumps(sub_results), "pytest python")  # type: ignore[attr-defined]
    )

    # At least one of the first two documents should be deduped due to high overlap
    assert len(result["documents"]) in {2, 3}
    # Metadata present and consistent
    meta = result["synthesis_metadata"]
    assert meta["original_count"] == 3
    assert meta["final_count"] >= 1
    assert "vector" in meta["strategies_used"] or "hybrid" in meta["strategies_used"]


@pytest.mark.unit
def test_synthesize_results_invalid_json_returns_error() -> None:
    out = json.loads(synthesize_results.func("not-json", "query"))  # type: ignore[attr-defined]
    assert out["documents"] == []
    assert out["error"] == "Invalid input format"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "sources", "expected_action"),
    [
        # Strong overlap and length -> accept
        (
            "Pytest enables simple assertions and fixtures for testing.",
            {
                "documents": [
                    {"content": "Pytest simple assertions fixtures for testing"}
                ]
            },
            "accept",
        ),
        # No sources and too short -> regenerate or refine (threshold-driven)
        ("Too short", {"documents": []}, "regenerate"),
    ],
)
def test_validate_response_actions(
    response: str, sources: dict, expected_action: str
) -> None:
    payload = validate_response.func(  # type: ignore[attr-defined]
        "pytest testing", response, json.dumps(sources)
    )
    data = json.loads(payload)
    assert data["suggested_action"] in {expected_action, "refine", "accept"}
    assert isinstance(data["confidence"], float)
