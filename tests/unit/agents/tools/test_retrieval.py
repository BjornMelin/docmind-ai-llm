"""Tests for the canonical router-backed retrieval boundary."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode

from src.agents.tools import retrieval as mod
from src.agents.tools.retrieval import retrieve_documents

pytestmark = pytest.mark.unit


def _router(
    result: Response | None = None,
    *,
    error: Exception | None = None,
) -> MagicMock:
    """Return a typed router test double with a canonical response."""
    router = MagicMock(spec=RouterQueryEngine)
    if error is not None:
        router.aquery.side_effect = error
    else:
        router.aquery.return_value = result or Response(response=None)
    return router


async def _invoke(
    *,
    query: str = "query",
    router: object | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke the tool function with explicit transient runtime context."""
    context = {"router_engine": router} if router is not None else {}
    runtime = cast(ToolRuntime, SimpleNamespace(context=context))
    effective_state = dict(state or {})
    effective_state.setdefault("deadline_ts", time.monotonic() + 1.0)
    effective_state.setdefault("turn_id", "test-turn")
    command = await cast(Any, retrieve_documents).coroutine(
        query=query,
        state=effective_state,
        runtime=runtime,
        tool_call_id="retrieval-call",
    )
    assert isinstance(command, Command)
    assert isinstance(command.update, dict)
    payload = command.update["retrieval_results"][0]
    message = command.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert isinstance(message.content, str)
    assert json.loads(message.content) == payload
    return cast(dict[str, Any], payload)


@pytest.mark.asyncio
async def test_missing_router_returns_stable_error() -> None:
    """Retrieval fails closed when no prebuilt router is available."""
    result = await _invoke(query="q")

    assert result == {
        "documents": [],
        "error": "Document router unavailable",
        "retrieval_id": "retrieval-call",
        "strategy_used": "router",
        "query_optimized": "q",
        "turn_id": "test-turn",
    }


def test_tool_schema_exposes_only_query() -> None:
    """LangGraph runtime and state remain hidden from the model tool schema."""
    schema = cast(Any, retrieve_documents).tool_call_schema.model_json_schema()
    assert schema["properties"].keys() == {"query"}
    assert schema["required"] == ["query"]


@pytest.mark.asyncio
async def test_router_is_queried_once_and_source_nodes_are_normalized() -> None:
    """The canonical router receives the raw query exactly once."""
    node = TextNode(text="document", id_="node-1")
    node.metadata = {
        "source": "/private/report.pdf",
        "image_base64": "secret",
    }
    response = Response(
        response="answer",
        source_nodes=[NodeWithScore(node=node, score=0.75)],
    )
    router = _router(response)

    result = await _invoke(query="find it", router=router)

    router.aquery.assert_awaited_once_with("find it")
    assert result["strategy_used"] == "router"
    assert result["router_used"] is True
    assert result["document_count"] == 1
    assert result["documents"] == [
        {
            "id": "node-1",
            "content": "document",
            "metadata": {"source": "report.pdf"},
            "score": 0.75,
        }
    ]


@pytest.mark.asyncio
async def test_router_results_deduplicate_stable_ids_and_preserve_native_order() -> (
    None
):
    """The first native-ranked occurrence of a stable node identity wins."""
    first = TextNode(text="same", id_="same-node")
    duplicate = TextNode(text="same", id_="same-node")
    different = TextNode(text="different", id_="different-node")
    router = _router(
        Response(
            response="answer",
            source_nodes=[
                NodeWithScore(node=first, score=0.4),
                NodeWithScore(node=duplicate, score=0.9),
                NodeWithScore(node=different, score=0.5),
            ],
        )
    )

    result = await _invoke(router=router)

    assert result["document_count"] == 2
    assert [document["id"] for document in result["documents"]] == [
        "same-node",
        "different-node",
    ]
    assert result["documents"][0]["score"] == 0.4


@pytest.mark.asyncio
async def test_empty_contextual_query_recalls_recent_sanitized_sources() -> None:
    """Contextual follow-ups can reuse sources from persisted graph state."""
    router = _router(Response(response=None))
    state = {
        "synthesis_result": {
            "documents": [
                {
                    "content": "chart",
                    "metadata": {
                        "doc_id": "d1",
                        "source_path": "/private/source.pdf",
                    },
                }
            ]
        }
    }

    result = await _invoke(
        query="what does that chart show?", router=router, state=state
    )

    assert result["documents"] == [{"content": "chart", "metadata": {"doc_id": "d1"}}]


@pytest.mark.asyncio
async def test_router_failure_returns_redacted_diagnostic_contract() -> None:
    """Router exceptions do not expose their message to the caller."""
    router = _router(error=RuntimeError("secret-token=/private/path"))

    result = await _invoke(router=router)

    assert result["error"] == "Document retrieval failed"
    assert result["error_type"] == "RuntimeError"
    assert len(result["error_fingerprint"]) == 12
    assert "secret-token" not in json.dumps(result)


@pytest.mark.asyncio
async def test_router_call_is_cancelled_at_remaining_deadline() -> None:
    """The native async router cannot outlive the coordinator budget."""
    cancelled = asyncio.Event()

    async def _wait_forever(_query: str) -> Response:
        try:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        finally:
            cancelled.set()

    router = _router()
    router.aquery.side_effect = _wait_forever

    result = await _invoke(
        router=router,
        state={
            "deadline_ts": time.monotonic() + 0.01,
            "turn_id": "deadline-turn",
        },
    )

    assert result["error"] == "Document retrieval deadline exceeded"
    assert result["error_type"] == "TimeoutError"
    assert cancelled.is_set()


def test_parse_response_without_sources_returns_no_evidence() -> None:
    """Synthesized response text is never promoted to retrieved evidence."""
    response = Response(response="answer", metadata={"source": "source.txt"})

    assert mod._parse_tool_result(response) == []


def test_parse_dict_list_removes_paths_and_binary_payloads() -> None:
    """Recalled dictionaries are sanitized before entering graph state."""
    result = mod._sanitize_document_dict(
        {
            "content": "doc",
            "image_base64": "AAAA",
            "source": "/private/top.pdf",
            "metadata": {
                "doc_id": "d1",
                "source": "/private/source.pdf",
                "image_path": "/private/image.webp",
            },
        }
    )

    assert result == {
        "content": "doc",
        "source": "top.pdf",
        "metadata": {"doc_id": "d1", "source": "source.pdf"},
    }


def test_parse_source_nodes_uses_native_content_and_identity() -> None:
    """Source parsing uses the canonical LlamaIndex node contract."""
    response = Response(
        response="answer",
        source_nodes=[
            NodeWithScore(node=TextNode(text="content", id_="node-1"), score=0.2)
        ],
    )

    assert mod._parse_tool_result(response) == [
        {"id": "node-1", "content": "content", "metadata": {}, "score": 0.2}
    ]


def test_recall_recent_sources_uses_latest_retrieval_batch() -> None:
    """Fallback recall chooses the most recent retrieval batch."""
    result = mod._recall_recent_sources(
        {
            "retrieval_results": [
                {"documents": [{"content": "old", "metadata": {"id": 1}}]},
                {"documents": [{"content": "new", "metadata": {"id": 2}}]},
            ]
        }
    )

    assert result == [{"content": "new", "metadata": {"id": 2}}]


def test_recall_recent_sources_skips_empty_recent_results() -> None:
    """Empty synthesis/current batches do not hide older usable evidence."""
    result = mod._recall_recent_sources(
        {
            "synthesis_result": {"documents": []},
            "retrieval_results": [
                {"documents": [{"content": "older", "metadata": {"id": 1}}]},
                {"documents": []},
            ],
        }
    )

    assert result == [{"content": "older", "metadata": {"id": 1}}]
