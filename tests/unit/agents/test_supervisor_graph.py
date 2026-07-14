"""Unit tests for graph-native supervisor utilities.

These tests focus on the repo-local supervisor graph builder and atomic dispatch tool.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command, Send

from src.agents.models import (
    MAX_RETRIEVAL_HISTORY_BATCHES,
    AgentRuntimeContext,
    MultiAgentGraphState,
    merge_retrieval_results,
)
from src.agents.supervisor_graph import (
    SupervisorBuildParams,
    _new_retrieval_results,
    _normalize_agent_name,
    _select_output_messages,
    build_multi_agent_supervisor_graph,
    create_dispatch_tool,
)


@pytest.mark.unit
def test_normalize_agent_name() -> None:
    assert _normalize_agent_name("Retrieval Agent") == "retrieval_agent"
    assert _normalize_agent_name("  Mixed\tWhitespace  ") == "mixed_whitespace"


@pytest.mark.unit
def test_select_output_messages_last_message_includes_tool_pair() -> None:
    msgs = [
        AIMessage(content="a", name="agent"),
        ToolMessage(content="tool", name="t", tool_call_id="1"),
    ]
    selected = _select_output_messages(msgs, output_mode="last_message")
    assert len(selected) == 2
    assert isinstance(selected[-1], ToolMessage)


@pytest.mark.unit
def test_select_output_messages_full_history() -> None:
    msgs = [HumanMessage(content="h"), AIMessage(content="a")]
    selected = _select_output_messages(msgs, output_mode="full_history")
    assert selected == msgs


@pytest.mark.unit
def test_create_dispatch_tool_single_destination_adds_tool_message() -> None:
    dispatch_tool = create_dispatch_tool(
        agent_names=["retrieval_agent"], add_handoff_messages=True
    )

    last = AIMessage(
        content="dispatch",
        tool_calls=[
            {
                "id": "tc1",
                "name": "dispatch_agents",
                "args": {"destinations": ["retrieval_agent"]},
            }
        ],
        name="supervisor",
    )
    state = {"messages": [last]}

    cmd = dispatch_tool.func(  # type: ignore[attr-defined]
        destinations=["retrieval_agent"], state=state, tool_call_id="tc1"
    )
    assert isinstance(cmd, Command)
    assert cmd.graph == Command.PARENT
    assert isinstance(cmd.goto, list)
    assert len(cmd.goto) == 1
    send = cmd.goto[0]
    assert getattr(send, "node", None) == "retrieval_agent"
    assert isinstance(send.arg["messages"][-1], ToolMessage)  # type: ignore[attr-defined]


@pytest.mark.unit
def test_create_dispatch_tool_atomically_sends_parallel_destinations() -> None:
    dispatch_tool = create_dispatch_tool(
        agent_names=["retrieval_agent", "validation_agent"],
        add_handoff_messages=True,
    )

    last = AIMessage(
        content="parallel dispatch",
        tool_calls=[
            {
                "id": "tc1",
                "name": "dispatch_agents",
                "args": {"destinations": ["retrieval_agent", "validation_agent"]},
            }
        ],
        name="supervisor",
    )
    state = {"messages": [last]}

    cmd = dispatch_tool.func(  # type: ignore[attr-defined]
        destinations=["retrieval_agent", "validation_agent"],
        state=state,
        tool_call_id="tc1",
    )
    assert isinstance(cmd, Command)
    assert cmd.graph == Command.PARENT
    assert isinstance(cmd.goto, list)
    assert [send.node for send in cmd.goto] == [  # type: ignore[union-attr]
        "retrieval_agent",
        "validation_agent",
    ]


@pytest.mark.unit
def test_create_dispatch_tool_merges_multiple_calls_and_deduplicates() -> None:
    dispatch_tool = create_dispatch_tool(
        agent_names=["retrieval_agent", "validation_agent"],
        add_handoff_messages=True,
    )
    last = AIMessage(
        content="duplicate dispatch",
        tool_calls=[
            {
                "id": "tc1",
                "name": "dispatch_agents",
                "args": {"destinations": ["retrieval_agent"]},
            },
            {
                "id": "tc2",
                "name": "dispatch_agents",
                "args": {"destinations": ["retrieval_agent", "validation_agent"]},
            },
        ],
        name="supervisor",
    )
    state = {"messages": [last]}

    first = dispatch_tool.func(  # type: ignore[attr-defined]
        destinations=["retrieval_agent"], state=state, tool_call_id="tc1"
    )
    duplicate = dispatch_tool.func(  # type: ignore[attr-defined]
        destinations=["retrieval_agent", "validation_agent"],
        state=state,
        tool_call_id="tc2",
    )

    assert isinstance(first, Command)
    assert isinstance(first.goto, list)
    assert [send.node for send in first.goto] == [  # type: ignore[union-attr]
        "retrieval_agent",
        "validation_agent",
    ]
    assert isinstance(duplicate, Command)
    assert not duplicate.goto
    assert isinstance(duplicate.update, dict)
    duplicate_message = duplicate.update["messages"][0]
    assert isinstance(duplicate_message, ToolMessage)
    assert duplicate_message.tool_call_id == "tc2"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_multi_agent_supervisor_graph_executes_minimal_loop() -> None:
    runtime_router = object()
    worker_contexts: list[object] = []

    class _SubAgent:
        def __init__(self, name: str):
            self.name = name

        async def ainvoke(self, state, config=None, *, context=None):
            del config
            worker_contexts.append(context)
            return {
                "messages": [AIMessage(content=f"{self.name} ok", name=self.name)],
                "retrieval_results": [
                    *state.get("retrieval_results", []),
                    {
                        "retrieval_id": "new-batch",
                        "turn_id": "minimal-turn",
                        "documents": [{"id": "new"}],
                    },
                ],
            }

    retrieval = _SubAgent("retrieval_agent")

    supervisor_calls = {"n": 0}

    class _SupervisorNode:
        async def ainvoke(self, state, config=None, *, context=None):
            del config, context
            supervisor_calls["n"] += 1
            if supervisor_calls["n"] == 1:
                return Command(
                    goto="retrieval_agent",
                    update={"messages": [AIMessage(content="go", name="supervisor")]},
                )
            # A normal create_agent terminal result echoes its complete input
            # state. The outer supervisor wrapper must emit only the new message.
            return {
                **state,
                "messages": [
                    *state["messages"],
                    AIMessage(content="done", name="supervisor"),
                ],
            }

    with patch(
        "src.agents.supervisor_graph.create_agent", return_value=_SupervisorNode()
    ):
        graph = build_multi_agent_supervisor_graph(
            [retrieval],
            model="fake",
            prompt="p",
            state_schema=MultiAgentGraphState,
            context_schema=AgentRuntimeContext,
            params=SupervisorBuildParams(
                supervisor_name="supervisor",
                output_mode="last_message",
                add_handoff_messages=True,
                add_handoff_back_messages=True,
            ),
        )
        compiled = graph.compile()
        out = await compiled.ainvoke(
            {
                "messages": [HumanMessage(content="hi")],
                "retrieval_results": [
                    {
                        "retrieval_id": "prior-batch",
                        "turn_id": "minimal-turn",
                        "documents": [{"id": "prior"}],
                    }
                ],
                "turn_id": "minimal-turn",
                "deadline_ts": 42.0,
            },
            context={"router_engine": runtime_router},
        )

    assert supervisor_calls["n"] == 2
    messages = out["messages"]
    assert any(
        isinstance(m, AIMessage) and getattr(m, "content", "") == "retrieval_agent ok"
        for m in messages
    )
    assert any(
        isinstance(m, AIMessage) and getattr(m, "content", "") == "done"
        for m in messages
    )
    assert out["retrieval_results"] == [
        {
            "retrieval_id": "prior-batch",
            "turn_id": "minimal-turn",
            "documents": [{"id": "prior"}],
        },
        {
            "retrieval_id": "new-batch",
            "turn_id": "minimal-turn",
            "documents": [{"id": "new"}],
        },
    ]
    assert worker_contexts == [{"router_engine": runtime_router}]


@pytest.mark.unit
def test_retrieval_reducer_is_bounded_and_associative() -> None:
    batches = [
        {"retrieval_id": f"batch-{index}", "documents": []}
        for index in range(MAX_RETRIEVAL_HISTORY_BATCHES + 8)
    ]
    left = merge_retrieval_results(
        merge_retrieval_results([], batches[:17]), batches[17:]
    )
    right = merge_retrieval_results(
        [], merge_retrieval_results(batches[:17], batches[17:])
    )

    assert left == right == batches[-MAX_RETRIEVAL_HISTORY_BATCHES:]


@pytest.mark.unit
def test_retrieval_delta_is_scoped_to_current_turn() -> None:
    previous = [
        {"legacy_shape": True},
        {
            "turn_id": "old-turn",
            "retrieval_id": "reused-call",
            "documents": [{"id": "old"}],
        },
    ]
    current = {
        "turn_id": "new-turn",
        "retrieval_id": "reused-call",
        "documents": [{"id": "new"}],
    }

    assert _new_retrieval_results(
        previous,
        [*previous, current],
        turn_id="new-turn",
    ) == [current]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parallel_workers_emit_only_domain_deltas() -> None:
    """Parallel workers must not echo parent LastValue control channels."""

    class _SubAgent:
        def __init__(self, name: str, key: str) -> None:
            self.name = name
            self._key = key

        async def ainvoke(self, state, config=None, *, context=None):
            del config, context
            return {
                **state,
                "messages": [AIMessage(content=self.name, name=self.name)],
                self._key: {"owner": self.name},
                # These nested values must never overwrite parent control state.
                "remaining_steps": 1,
            }

    planner = _SubAgent("planner_agent", "planning_output")
    validator = _SubAgent("validation_agent", "validation_result")
    supervisor_calls = 0

    class _SupervisorNode:
        async def ainvoke(self, state, config=None, *, context=None):
            nonlocal supervisor_calls
            del config, context
            supervisor_calls += 1
            if supervisor_calls == 1:
                return Command(
                    goto=[
                        Send("planner_agent", dict(state)),
                        Send("validation_agent", dict(state)),
                    ],
                    update={"messages": [AIMessage(content="parallel")]},
                )
            return Command(
                goto="__end__",
                update={"messages": [AIMessage(content="done")]},
            )

        async def __call__(self, state, config):
            del config
            return await self.ainvoke(state)

    with patch(
        "src.agents.supervisor_graph.create_agent",
        return_value=_SupervisorNode(),
    ):
        graph = build_multi_agent_supervisor_graph(
            [planner, validator],
            model="fake",
            prompt="p",
            state_schema=MultiAgentGraphState,
            context_schema=AgentRuntimeContext,
        ).compile()
        out = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="hi")],
                "deadline_ts": 42.0,
                "turn_id": "parallel-turn",
                "remaining_steps": 9,
            },
            context={},
        )

    assert out["deadline_ts"] == 42.0
    assert out["turn_id"] == "parallel-turn"
    assert out["remaining_steps"] == 9
    assert out["planning_output"] == {"owner": "planner_agent"}
    assert out["validation_result"] == {"owner": "validation_agent"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_real_create_agent_dispatches_parallel_workers_atomically() -> None:
    """The native create_agent path must preserve every requested destination."""

    class _ToolCallingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            del tools, kwargs
            return self

    model = _ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "dispatch-call",
                        "name": "dispatch_agents",
                        "args": {"destinations": ["planner_agent", "validation_agent"]},
                    }
                ],
            ),
            AIMessage(content="done", name="supervisor"),
        ]
    )
    calls = {"planner_agent": 0, "validation_agent": 0}

    class _SubAgent:
        def __init__(self, name: str, output_key: str) -> None:
            self.name = name
            self._output_key = output_key

        async def ainvoke(self, state, config=None, *, context=None):
            del config, context
            calls[self.name] += 1
            return {
                **state,
                "messages": [
                    *state["messages"],
                    AIMessage(content=f"{self.name} done", name=self.name),
                ],
                self._output_key: {"owner": self.name},
            }

    graph = build_multi_agent_supervisor_graph(
        [
            _SubAgent("planner_agent", "planning_output"),
            _SubAgent("validation_agent", "validation_result"),
        ],
        model=model,
        prompt="Dispatch independent workers together.",
        state_schema=MultiAgentGraphState,
        context_schema=AgentRuntimeContext,
    ).compile()

    out = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="plan and validate")],
            "deadline_ts": 42.0,
            "turn_id": "native-dispatch-turn",
            "remaining_steps": 9,
        },
        context={},
    )

    assert calls == {"planner_agent": 1, "validation_agent": 1}
    assert out["planning_output"] == {"owner": "planner_agent"}
    assert out["validation_result"] == {"owner": "validation_agent"}


@pytest.mark.unit
def test_build_multi_agent_supervisor_graph_requires_named_agents() -> None:
    unnamed = SimpleNamespace(name=None, invoke=lambda *_a, **_k: {"messages": []})
    with (
        patch(
            "src.agents.supervisor_graph.create_agent",
            return_value=lambda *_a, **_k: {},
        ),
        pytest.raises(ValueError, match="non-empty"),
    ):
        build_multi_agent_supervisor_graph([unnamed], model="fake", prompt="p")


@pytest.mark.unit
def test_build_multi_agent_supervisor_graph_requires_unique_agent_names() -> None:
    a1 = SimpleNamespace(
        name="retrieval_agent", invoke=lambda *_a, **_k: {"messages": []}
    )
    a2 = SimpleNamespace(
        name="retrieval_agent", invoke=lambda *_a, **_k: {"messages": []}
    )
    with (
        patch(
            "src.agents.supervisor_graph.create_agent",
            return_value=lambda *_a, **_k: {},
        ),
        pytest.raises(ValueError, match="unique"),
    ):
        build_multi_agent_supervisor_graph([a1, a2], model="fake", prompt="p")
