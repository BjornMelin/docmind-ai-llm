"""Integration coverage for Ollama web tools loop (offline)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from ollama import ChatResponse, Message

from src.agents.tools.ollama_web_tools import run_web_search_agent

pytestmark = pytest.mark.integration


def _tool_call(name: str, **kwargs) -> Message.ToolCall:
    """Create a tool call message."""
    return Message.ToolCall(
        function=Message.ToolCall.Function(name=name, arguments=kwargs)
    )


def test_run_web_search_agent_handles_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent loop should execute tool calls and return final content."""
    calls: list[str] = []

    def fake_web_search(*, query: str, max_results: int = 3) -> dict[str, object]:
        calls.append(f"search:{query}:{max_results}")
        return {"results": []}

    def fake_web_fetch(*, url: str) -> dict[str, object]:
        calls.append(f"fetch:{url}")
        return {"content": ""}

    fake_web_search.__name__ = "web_search"
    fake_web_fetch.__name__ = "web_fetch"

    tool_list: list[Callable[..., object]] = [fake_web_search, fake_web_fetch]

    def fake_get_tools(_cfg):
        return tool_list

    def fake_chat(**_kwargs) -> ChatResponse:
        # First call: request web_search; second call: return final answer.
        step = len([c for c in calls if c.startswith("search:")])
        if step == 0:
            message = Message(
                role="assistant",
                tool_calls=[_tool_call("web_search", query="docmind", max_results=1)],
            )
            return ChatResponse(message=message)
        message = Message(role="assistant", content="done")
        return ChatResponse(message=message)

    monkeypatch.setattr(
        "src.agents.tools.ollama_web_tools.get_ollama_web_tools", fake_get_tools
    )
    monkeypatch.setattr("src.agents.tools.ollama_web_tools.ollama_chat", fake_chat)

    out = run_web_search_agent(model="test", prompt="hi", max_steps=2)
    assert out == "done"
    assert calls == ["search:docmind:1"]
