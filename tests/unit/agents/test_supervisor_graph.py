"""Unit tests for graph-native supervisor utilities.

These tests focus on the repo-local supervisor graph builder and its handoff tools.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from src.agents.supervisor_graph import (
    SupervisorBuildParams,
    _normalize_agent_name,
    _select_output_messages,
    build_multi_agent_supervisor_graph,
    create_forward_message_tool,
    create_handoff_tool,
)


@pytest.mark.unit
def test_normalize_agent_name() -> None:
    assert _normalize_agent_name("Router Agent") == "router_agent"
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
def test_create_handoff_tool_single_handoff_adds_tool_message() -> None:
    handoff_tool = create_handoff_tool(
        agent_name="router_agent", add_handoff_messages=True
    )

    last = AIMessage(
        content="handoff",
        tool_calls=[{"id": "tc1", "name": "transfer_to_router_agent", "args": {}}],
        name="supervisor",
    )
    state = {"messages": [last]}

    cmd = handoff_tool.func(state=state, tool_call_id="tc1")  # type: ignore[attr-defined]
    assert isinstance(cmd, Command)
    assert cmd.goto == "router_agent"
    assert cmd.graph == Command.PARENT
    assert isinstance(cmd.update, dict)
    assert "messages" in cmd.update
    assert isinstance(cmd.update["messages"][-1], ToolMessage)


@pytest.mark.unit
def test_create_handoff_tool_parallel_handoff_filters_calls_and_uses_send() -> None:
    handoff_tool = create_handoff_tool(
        agent_name="router_agent", add_handoff_messages=True
    )

    # Simulate parallel tool calls: only the matching tool_call_id should remain.
    last = AIMessage(
        content=[
            {"type": "tool_use", "id": "tc1"},
            {"type": "tool_use", "id": "tc2"},
            {"type": "text", "text": "hi"},
        ],
        tool_calls=[
            {"id": "tc1", "name": "transfer_to_router_agent", "args": {}},
            {"id": "tc2", "name": "other_tool", "args": {}},
        ],
        name="supervisor",
    )
    state = {"messages": [last]}

    cmd = handoff_tool.func(state=state, tool_call_id="tc1")  # type: ignore[attr-defined]
    assert isinstance(cmd, Command)
    assert cmd.graph == Command.PARENT
    assert isinstance(cmd.goto, list)
    assert cmd.goto
    first_send = cmd.goto[0]
    assert getattr(first_send, "node", None) == "router_agent"
    send_payload = first_send.arg  # type: ignore[attr-defined]
    send_messages = send_payload["messages"]
    assert isinstance(send_messages, list)
    assert isinstance(send_messages[-1], ToolMessage)
    filtered = send_messages[-2]
    assert isinstance(filtered, AIMessage)
    assert len(filtered.tool_calls) == 1
    assert filtered.tool_calls[0]["id"] == "tc1"


@pytest.mark.unit
def test_create_forward_message_tool_forwards_latest_agent_message() -> None:
    forward = create_forward_message_tool(supervisor_name="supervisor")
    state = {
        "messages": [
            AIMessage(content="from router", name="router_agent"),
            AIMessage(content="ignored", name="supervisor"),
        ]
    }
    result = forward.func(from_agent="router_agent", state=state)  # type: ignore[attr-defined]
    assert isinstance(result, Command)
    assert result.graph == Command.PARENT
    assert isinstance(result.update, dict)
    msg = result.update["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.name == "supervisor"
    assert msg.content == "from router"


@pytest.mark.unit
def test_create_forward_message_tool_missing_agent_returns_error_string() -> None:
    forward = create_forward_message_tool(supervisor_name="supervisor")
    state = {"messages": [AIMessage(content="x", name="someone_else")]}
    result = forward.func(from_agent="router_agent", state=state)  # type: ignore[attr-defined]
    assert isinstance(result, str)
    assert "Could not find message" in result


@pytest.mark.unit
def test_build_multi_agent_supervisor_graph_executes_minimal_loop() -> None:
    class _SubAgent:
        def __init__(self, name: str):
            self.name = name

        def invoke(self, _state, config=None):
            del config
            return {"messages": [AIMessage(content=f"{self.name} ok", name=self.name)]}

    router = _SubAgent("router_agent")

    supervisor_calls = {"n": 0}

    class _SupervisorNode:
        def invoke(self, _state, config=None):
            del config
            supervisor_calls["n"] += 1
            if supervisor_calls["n"] == 1:
                return Command(
                    goto="router_agent",
                    update={"messages": [AIMessage(content="go", name="supervisor")]},
                )
            return Command(
                goto="__end__",
                update={"messages": [AIMessage(content="done", name="supervisor")]},
            )

        def __call__(self, state, config):
            return self.invoke(state, config=config)

    with patch(
        "src.agents.supervisor_graph.create_agent", return_value=_SupervisorNode()
    ):
        graph = build_multi_agent_supervisor_graph(
            [router],
            model="fake",
            prompt="p",
            params=SupervisorBuildParams(
                supervisor_name="supervisor",
                output_mode="last_message",
                add_handoff_messages=True,
                add_handoff_back_messages=True,
            ),
        )
        compiled = graph.compile()
        out = compiled.invoke({"messages": [HumanMessage(content="hi")]})

    assert supervisor_calls["n"] == 2
    messages = out["messages"]
    assert any(
        isinstance(m, AIMessage) and getattr(m, "content", "") == "router_agent ok"
        for m in messages
    )
    assert any(
        isinstance(m, AIMessage) and getattr(m, "content", "") == "done"
        for m in messages
    )


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
    a1 = SimpleNamespace(name="router_agent", invoke=lambda *_a, **_k: {"messages": []})
    a2 = SimpleNamespace(name="router_agent", invoke=lambda *_a, **_k: {"messages": []})
    with (
        patch(
            "src.agents.supervisor_graph.create_agent",
            return_value=lambda *_a, **_k: {},
        ),
        pytest.raises(ValueError, match="unique"),
    ):
        build_multi_agent_supervisor_graph([a1, a2], model="fake", prompt="p")
