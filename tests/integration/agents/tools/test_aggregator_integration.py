"""Integration smoke tests for src.agents.tools aggregator.

Validates patchability and simple invocation across modules without deep behavior.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.types import Command
from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode

from src.agents.coordinator import MultiAgentCoordinator
from src.agents.models import AgentRuntimeContext, MultiAgentGraphState
from src.agents.tools.retrieval import retrieve_documents
from src.agents.tools.synthesis import synthesize_results

pytestmark = pytest.mark.integration


def _command_payload(command: object, state_key: str) -> dict[str, Any]:
    assert isinstance(command, Command)
    assert isinstance(command.update, dict)
    message = command.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert isinstance(message.content, str)
    value = command.update[state_key]
    if state_key == "retrieval_results":
        value = value[0]
    assert isinstance(value, dict)
    assert json.loads(message.content) == value
    return value


@pytest.mark.asyncio
async def test_retrieve_documents_error_json_when_missing_tools():
    """Missing runtime context returns an error JSON structure."""
    runtime = cast(ToolRuntime, SimpleNamespace(context={}))
    command = await cast(Any, retrieve_documents).coroutine(
        query="q",
        runtime=runtime,
        state={
            "deadline_ts": time.monotonic() + 1.0,
            "turn_id": "missing-router-turn",
        },
        tool_call_id="missing-router",
    )
    out = _command_payload(command, "retrieval_results")
    assert out.get("documents") == []
    assert "error" in out


@pytest.mark.asyncio
async def test_retrieve_documents_with_runtime_router():
    """The runtime router returns normalized documents through the tool boundary."""
    router = MagicMock(spec=RouterQueryEngine)
    router.aquery.return_value = Response(
        response="answer",
        source_nodes=[NodeWithScore(node=TextNode(text="ok", id_="node-1"), score=0.8)],
    )
    runtime = cast(
        ToolRuntime,
        SimpleNamespace(context={"router_engine": router}),
    )
    command = await cast(Any, retrieve_documents).coroutine(
        query="q",
        state={
            "deadline_ts": time.monotonic() + 1.0,
            "turn_id": "runtime-router-turn",
        },
        runtime=runtime,
        tool_call_id="runtime-router",
    )
    out = _command_payload(command, "retrieval_results")
    assert out.get("documents")
    router.aquery.assert_awaited_once_with("q")


@pytest.mark.asyncio
async def test_tool_nodes_persist_parallel_sources_through_final_extraction() -> None:
    """Native Commands preserve parallel retrieval and synthesized final sources."""
    router = MagicMock(spec=RouterQueryEngine)
    router.aquery.side_effect = [
        Response(
            response="first",
            source_nodes=[
                NodeWithScore(
                    node=TextNode(text="first document", id_="node-1"), score=0.9
                )
            ],
        ),
        Response(
            response="second",
            source_nodes=[
                NodeWithScore(
                    node=TextNode(text="second document", id_="node-2"),
                    score=0.8,
                )
            ],
        ),
    ]

    def request_synthesis(_state: dict[str, Any]) -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "synthesize_results",
                            # These fabricated legacy arguments must be ignored;
                            # canonical sources come only from injected state.
                            "args": {
                                "sub_results": json.dumps(
                                    [{"documents": [{"id": "fabricated"}]}]
                                ),
                                "original_query": "fabricated query",
                            },
                            "id": "synthesis-call",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    builder = StateGraph(
        MultiAgentGraphState,
        context_schema=AgentRuntimeContext,
    )
    builder.add_node("retrieve", ToolNode([retrieve_documents]))
    builder.add_node("request_synthesis", request_synthesis)
    builder.add_node("synthesize", ToolNode([synthesize_results]))
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "request_synthesis")
    builder.add_edge("request_synthesis", "synthesize")
    builder.add_edge("synthesize", END)
    graph = builder.compile()

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="compare sources"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "retrieve_documents",
                            "args": {"query": "first"},
                            "id": "retrieval-1",
                            "type": "tool_call",
                        },
                        {
                            "name": "retrieve_documents",
                            "args": {"query": "second"},
                            "id": "retrieval-2",
                            "type": "tool_call",
                        },
                    ],
                ),
            ],
            "deadline_ts": time.monotonic() + 2.0,
            "turn_id": "parallel-turn",
        },
        context={"router_engine": router},
    )

    assert len(result["retrieval_results"]) == 2
    assert {batch["turn_id"] for batch in result["retrieval_results"]} == {
        "parallel-turn"
    }
    assert {batch["documents"][0]["id"] for batch in result["retrieval_results"]} == {
        "node-1",
        "node-2",
    }
    assert [document["id"] for document in result["synthesis_result"]["documents"]] == [
        "node-1",
        "node-2",
    ]
    assert result["synthesis_result"]["original_query"] == "compare sources"
    tool_schema = cast(Any, synthesize_results).tool_call_schema.model_json_schema()
    assert tool_schema["properties"] == {}

    coordinator = MultiAgentCoordinator()
    try:
        response = coordinator._extract_response(
            result,
            "compare sources",
            time.perf_counter(),
            0.0,
        )
    finally:
        coordinator.close()
    assert [source["id"] for source in response.sources] == ["node-1", "node-2"]


@pytest.mark.asyncio
async def test_checkpointed_turn_never_reuses_prior_turn_sources() -> None:
    """Current-turn synthesis/extraction ignores checkpointed prior evidence."""

    def request_synthesis(state: dict[str, Any]) -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "synthesize_results",
                            "args": {},
                            "id": f"synthesize-{state['turn_id']}",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    builder = StateGraph(MultiAgentGraphState)
    builder.add_node("request_synthesis", request_synthesis)
    builder.add_node("synthesize", ToolNode([synthesize_results]))
    builder.add_edge(START, "request_synthesis")
    builder.add_edge("request_synthesis", "synthesize")
    builder.add_edge("synthesize", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "turn-isolation"}}

    first = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="first question")],
            "retrieval_results": [
                {
                    "turn_id": "turn-1",
                    "retrieval_id": "retrieval-1",
                    "documents": [{"id": "prior-source", "content": "old"}],
                }
            ],
            "turn_id": "turn-1",
            "deadline_ts": time.monotonic() + 1.0,
        },
        config=config,
    )
    second = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="unrelated second question")],
            "turn_id": "turn-2",
            "deadline_ts": time.monotonic() + 1.0,
        },
        config=config,
    )

    assert [doc["id"] for doc in first["synthesis_result"]["documents"]] == [
        "prior-source"
    ]
    assert second["synthesis_result"]["turn_id"] == "turn-2"
    assert second["synthesis_result"]["documents"] == []

    coordinator = MultiAgentCoordinator()
    try:
        response = coordinator._extract_response(
            second,
            "unrelated second question",
            time.perf_counter(),
            0.0,
        )
    finally:
        coordinator.close()
    assert response.sources == []


@pytest.mark.asyncio
async def test_checkpointed_retrieval_history_remains_bounded() -> None:
    builder = StateGraph(MultiAgentGraphState)
    builder.add_node("noop", lambda _state: {})
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "bounded-retrieval-history"}}

    result: dict[str, Any] = {}
    for index in range(40):
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=f"turn {index}")],
                "retrieval_results": [
                    {
                        "turn_id": f"turn-{index}",
                        "retrieval_id": "retrieval-call",
                        "documents": [{"id": index}],
                    }
                ],
                "turn_id": f"turn-{index}",
                "deadline_ts": time.monotonic() + 1.0,
            },
            config=config,
        )

    batches = result["retrieval_results"]
    assert len(batches) == 32
    assert [batch["documents"][0]["id"] for batch in batches] == list(range(8, 40))


def test_final_extraction_rejects_stale_same_turn_synthesis() -> None:
    """A later same-turn retrieval batch invalidates the synthesis watermark."""
    batches = [
        {
            "turn_id": "same-turn",
            "retrieval_id": "batch-a",
            "documents": [{"id": "a"}],
        },
        {
            "turn_id": "same-turn",
            "retrieval_id": "batch-b",
            "documents": [{"id": "b"}],
        },
    ]
    final_state = {
        "messages": [AIMessage(content="answer")],
        "turn_id": "same-turn",
        "retrieval_results": batches,
        "synthesis_result": {
            "turn_id": "same-turn",
            "retrieval_watermark": [["same-turn", "batch-a"]],
            "documents": [{"id": "a"}],
        },
    }

    coordinator = MultiAgentCoordinator()
    try:
        response = coordinator._extract_response(
            final_state,
            "question",
            time.perf_counter(),
            0.0,
        )
    finally:
        coordinator.close()

    assert [source["id"] for source in response.sources] == ["a", "b"]
