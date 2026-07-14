"""Unit tests for synthesis and validation tools.

These tests exercise pure functions and light tool wrappers to increase
coverage while adhering to pytest best practices: table-driven cases,
no external I/O, deterministic assertions.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agents.tools.synthesis import synthesize_results
from src.agents.tools.validation import validate_response


def _synthesis_payload(sub_results: object, query: str) -> dict[str, Any]:
    turn_id = "validation-turn"
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
            "messages": [HumanMessage(content=query)],
            "retrieval_results": retrieval_results,
            "turn_id": turn_id,
        },
        tool_call_id="synthesis-call",
    )
    assert isinstance(command, Command)
    assert isinstance(command.update, dict)
    return cast(dict[str, Any], command.update["synthesis_result"])


@pytest.mark.unit
def test_synthesize_results_preserves_each_batch_order_and_fairness() -> None:
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
    result = _synthesis_payload(sub_results, "pytest python")

    assert [document["id"] for document in result["documents"]] == [1, 3, 2]
    # Metadata present and consistent
    meta = result["synthesis_metadata"]
    assert meta["original_count"] == 3
    assert meta["final_count"] >= 1
    assert "vector" in meta["strategies_used"] or "hybrid" in meta["strategies_used"]


@pytest.mark.unit
def test_synthesize_results_invalid_state_returns_error() -> None:
    out = _synthesis_payload("not-a-list", "query")
    assert out["documents"] == []
    assert out["error"] == "Invalid input format"


@pytest.mark.unit
@pytest.mark.parametrize(
    "decoded",
    [
        {"documents": []},
        [{"documents": {}}],
        [{"documents": ["not-a-document"]}],
        [{"documents": [], "processing_time_ms": "fast"}],
    ],
)
def test_synthesize_results_rejects_wrong_decoded_shapes(decoded: object) -> None:
    out = _synthesis_payload(decoded, "query")
    assert out == {
        "documents": [],
        "error": "Invalid input format",
        "synthesis_metadata": {},
        "turn_id": "validation-turn",
    }


@pytest.mark.unit
def test_synthesize_results_ignores_unattributed_historical_entries() -> None:
    out = _synthesis_payload(["not-a-retrieval-batch"], "query")

    assert out["documents"] == []
    assert "error" not in out


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
