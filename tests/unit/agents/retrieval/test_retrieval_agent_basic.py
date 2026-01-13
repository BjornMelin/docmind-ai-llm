"""Unit tests for the lean RetrievalAgent wrapper."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any, cast

import pytest


class _DummyTool:
    def __init__(self, payload: Mapping[str, Any]):
        """Expose a langchain-style callable returning the supplied payload."""
        serialisable = dict(payload)
        self.func = lambda **_: json.dumps(serialisable)


class _InvokeOnlyTool:
    """Simulate a LangChain tool exposing only an ``invoke`` method."""

    def __init__(self, payload: Mapping[str, Any]):
        self._payload = dict(payload)
        self.calls: list[dict[str, object]] = []

    def invoke(self, payload: Mapping[str, Any]):
        """Record the payload passed to the invoke-only tool."""
        self.calls.append(dict(payload))
        return json.dumps(self._payload)


def _make_stub_agent(payload: dict[str, object]):
    """Create an agent returning the payload as a JSON message."""

    class _Agent:
        """Stub agent returning the provided payload as JSON."""

        def invoke(self, *_a, **_k):
            """Return the stored payload within a LangGraph message envelope."""
            return {"messages": [SimpleNamespace(content=json.dumps(payload))]}

    return _Agent()


def _bad_payload_agent():
    """Return an agent that emits malformed output to trigger fallback."""

    class _Agent:
        """Stub agent returning malformed content for fallback tests."""

        def invoke(self, *_a, **_k):
            """Return content that cannot be parsed as JSON."""
            return {"messages": [SimpleNamespace(content="not-json")]}

    return _Agent()


def _raising_agent():
    """Return an agent that raises to exercise error fallback."""

    class _Agent:
        """Stub agent raising to exercise error handling."""

        def invoke(self, *_a, **_k):
            """Raise runtime error to mimic agent failure."""
            raise RuntimeError("boom")

    return _Agent()


class _KwargTool:
    """Callable accepting keyword arguments for direct invocation tests."""

    def __init__(self) -> None:
        self.received: dict[str, Any] | None = None

    def __call__(self, **payload: object) -> str:
        self.received = payload
        return json.dumps({"documents": [], "strategy_used": payload["strategy"]})


class _DictOnlyTool:
    """Callable requiring a single mapping payload, raising for kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: object, **kwargs: object) -> str:
        if kwargs:
            raise TypeError("dict payload required")
        if len(args) != 1 or not isinstance(args[0], dict):  # pragma: no cover - guard
            raise TypeError("single mapping argument expected")
        payload = dict(args[0])
        self.calls.append(payload)
        return json.dumps({"documents": [], "strategy_used": payload["strategy"]})


class _PositionalQueryTool:
    """Callable that accepts a positional query and optional state.

    This mirrors the underlying signature of ``@tool`` wrapped functions. If
    called with a single positional mapping, it will *not* raise, but will drop
    ``state`` and break the fallback retrieval path. The direct invocation
    helper must therefore prioritise keyword arguments.
    """

    def __init__(self) -> None:
        self.received_query: object | None = None
        self.received_state: object | None = None

    def __call__(
        self,
        query: str,
        strategy: str = "hybrid",
        use_dspy: bool = True,
        use_graphrag: bool = False,
        state: dict | None = None,
    ) -> str:
        self.received_query = query
        self.received_state = state
        return json.dumps({"documents": [], "strategy_used": strategy})


def _set_tool(agent: Any, tool: Callable[..., object]) -> None:
    """Inject the provided tool into the agent instance for direct invocation tests."""
    agent._tool_callable = tool  # type: ignore[attr-defined]


@pytest.mark.unit
def test_retrieval_agent_success_path(monkeypatch: pytest.MonkeyPatch):
    """Agent invocation returns LangGraph payload and updates metrics."""
    from src.agents import retrieval as mod

    payload = {
        "documents": [
            {"content": "hello", "metadata": {"s": 1}, "score": 0.9},
            {"content": "world", "metadata": {"s": 2}, "score": 0.8},
        ],
        "strategy_used": "hybrid",
        "query_original": "q",
        "query_optimized": "q*",
        "document_count": 2,
        "dspy_used": True,
        "graphrag_used": False,
    }

    monkeypatch.setattr(
        mod, "create_agent", lambda *_, **__: _make_stub_agent(payload)
    )

    agent = mod.RetrievalAgent(llm=None, tools_data={})
    assert agent.total_retrievals == 0

    result = agent.retrieve_documents("q", strategy="hybrid", use_dspy=True)
    assert result.document_count == 2
    assert result.strategy_used == "hybrid"
    assert result.query_original == "q"
    assert result.query_optimized == "q*"
    assert result.dspy_used is True
    assert result.graphrag_used is False

    stats = agent.get_performance_stats()
    assert stats["total_retrievals"] == 1
    assert stats["strategy_usage"]["hybrid"] == 1


@pytest.mark.unit
def test_retrieval_agent_falls_back_on_bad_payload(monkeypatch: pytest.MonkeyPatch):
    """Malformed agent output should fall back to the direct tool call."""
    from src.agents import retrieval as mod

    fallback_payload = {
        "documents": [{"content": "fallback", "metadata": {"k": 1}}],
        "strategy_used": "vector",
        "query_original": "q",
        "query_optimized": "q",
    }

    monkeypatch.setattr(
        mod, "create_agent", lambda *_, **__: _bad_payload_agent()
    )
    monkeypatch.setattr(mod, "retrieve_documents", _DummyTool(fallback_payload))

    agent = mod.RetrievalAgent(llm=None, tools_data={})
    result = agent.retrieve_documents("q", strategy="vector", use_dspy=False)

    assert result.document_count == 1
    assert result.documents[0]["content"] == "fallback"
    assert result.strategy_used == "vector"
    assert agent.strategy_usage["fallback"] == 1


@pytest.mark.unit
def test_retrieval_agent_error_fallback(monkeypatch: pytest.MonkeyPatch):
    """Agent exceptions propagate through the direct tool fallback."""
    from src.agents import retrieval as mod

    fallback_payload = {
        "documents": [],
        "strategy_used": "vector_failed",
        "query_original": "q",
        "query_optimized": "q",
        "error": "tool crash",
    }

    monkeypatch.setattr(mod, "create_agent", lambda *_, **__: _raising_agent())
    monkeypatch.setattr(mod, "retrieve_documents", _DummyTool(fallback_payload))

    agent = mod.RetrievalAgent(llm=None, tools_data={})
    result = agent.retrieve_documents("q", strategy="vector")

    assert result.document_count == 0
    assert result.confidence_score == 0.0
    assert "Retrieval failed" in result.reasoning
    assert agent.strategy_usage["fallback"] == 1

    agent.reset_stats()
    stats = agent.get_performance_stats()
    assert stats["total_retrievals"] == 0
    assert stats["avg_retrieval_time_ms"] == 0.0


@pytest.mark.unit
def test_retrieval_agent_direct_tool_invoke_wrapper(monkeypatch: pytest.MonkeyPatch):
    """Direct fallback should support tools exposing only ``invoke``."""
    from src.agents import retrieval as mod

    fallback_payload = {
        "documents": [{"content": "invoke-only", "metadata": {"k": 2}}],
        "strategy_used": "hybrid",
        "query_original": "q",
        "query_optimized": "q",
    }

    tool = _InvokeOnlyTool(fallback_payload)

    monkeypatch.setattr(mod, "create_agent", lambda *_, **__: _raising_agent())
    monkeypatch.setattr(mod, "retrieve_documents", tool)

    tools_data = {"vector": "vec"}
    agent = mod.RetrievalAgent(llm=None, tools_data=tools_data)
    result = agent.retrieve_documents("q", strategy="hybrid", use_dspy=False)

    assert result.document_count == 1
    assert result.documents[0]["content"] == "invoke-only"
    assert tool.calls
    payload = tool.calls[0]
    assert payload["query"] == "q"
    assert payload["state"] == {"tools_data": tools_data}


@pytest.mark.unit
def test_retrieval_agent_direct_tool_unwrapped_invoke(monkeypatch: pytest.MonkeyPatch):
    """Fallback should pass payload dicts even if tagging fails."""
    from src.agents import retrieval as mod

    payload = {
        "documents": [],
        "strategy_used": "vector",
        "query_original": "q",
        "query_optimized": "q",
    }

    tool = _InvokeOnlyTool(payload)

    monkeypatch.setattr(mod, "create_agent", lambda *_, **__: _raising_agent())
    monkeypatch.setattr(mod, "retrieve_documents", tool)

    agent = mod.RetrievalAgent(llm=None, tools_data={})
    # Simulate an environment where the tagging attribute is missing.
    agent._tool_callable = tool.invoke  # type: ignore[method-assign]

    result = agent.retrieve_documents("q", strategy="vector", use_dspy=False)

    assert result.strategy_used == "vector"
    assert tool.calls


@pytest.mark.unit
def test_retrieval_agent_direct_tool_malformed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback should handle malformed payload emitted by invoke-only tools."""
    from src.agents import retrieval as mod

    class _BadInvokeTool:
        def invoke(self, _payload: Mapping[str, Any]):
            """Return malformed content to trigger payload parsing failure."""
            return "not-json"

    monkeypatch.setattr(mod, "create_agent", lambda *_, **__: _raising_agent())
    monkeypatch.setattr(mod, "retrieve_documents", _BadInvokeTool())

    agent = mod.RetrievalAgent(llm=None, tools_data={})
    result = agent.retrieve_documents("q", strategy="vector", use_dspy=False)

    assert result.document_count == 0
    assert "Malformed tool payload" in result.reasoning
    assert agent.strategy_usage["fallback"] == 1


@pytest.mark.unit
def test_resolve_tool_callable_tags_invoke() -> None:
    """Tools exposing ``invoke`` should be tagged as payload consumers."""
    from src.agents import retrieval as mod

    class _InvokeTool:
        def invoke(self, payload: Mapping[str, Any]):
            """Return the provided payload unmodified for tagging assertions."""
            return dict(payload)

    wrapped = mod._resolve_tool_callable(_InvokeTool())
    wrapped_any = cast(Any, wrapped)
    assert wrapped_any.expects_payload_dict is True
    wrapped({"query": "q"})


@pytest.mark.unit
def test_resolve_tool_callable_tags_callable() -> None:
    """Plain callables should be tagged for kwargs dispatch."""
    from src.agents import retrieval as mod

    def _callable_tool(**_: object) -> str:
        return "ok"

    wrapped = mod._resolve_tool_callable(_callable_tool)
    wrapped_any = cast(Any, wrapped)
    assert wrapped_any.expects_payload_dict is False
    wrapped(query="q")


@pytest.mark.unit
def test_direct_tool_invocation_prioritises_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct invocation should pass keyword payloads when supported."""
    from src.agents import retrieval as mod

    monkeypatch.setattr(mod, "create_agent", lambda *_, **__: _raising_agent())
    agent = mod.RetrievalAgent(llm=None, tools_data={"vector": "v"})

    kw_tool = _KwargTool()
    _set_tool(agent, kw_tool)

    result = agent._call_tool_directly(
        "q", strategy="hybrid", use_dspy=True, use_graphrag=False
    )

    assert kw_tool.received is not None
    assert kw_tool.received["query"] == "q"
    assert result.strategy_used == "hybrid"


@pytest.mark.unit
def test_direct_tool_invocation_supplies_state_for_positional_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct invocation must not drop state when tools accept positional query."""
    from src.agents import retrieval as mod

    monkeypatch.setattr(mod, "create_agent", lambda *_, **__: _raising_agent())
    tools_data = {"vector": "v"}
    agent = mod.RetrievalAgent(llm=None, tools_data=tools_data)

    tool = _PositionalQueryTool()
    _set_tool(agent, tool)

    result = agent._call_tool_directly(
        "q", strategy="hybrid", use_dspy=True, use_graphrag=False
    )

    assert tool.received_query == "q"
    assert tool.received_state == {"tools_data": tools_data}
    assert result.strategy_used == "hybrid"


@pytest.mark.unit
def test_direct_tool_invocation_retries_with_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If kwargs invocation fails, retry with a payload mapping."""
    from src.agents import retrieval as mod

    monkeypatch.setattr(mod, "create_agent", lambda *_, **__: _raising_agent())
    agent = mod.RetrievalAgent(llm=None, tools_data={"vector": "v"})

    dict_tool = _DictOnlyTool()
    _set_tool(agent, dict_tool)

    result = agent._call_tool_directly(
        "q", strategy="vector", use_dspy=False, use_graphrag=False
    )

    assert dict_tool.calls
    assert dict_tool.calls[0]["query"] == "q"
    assert result.strategy_used == "vector"


@pytest.mark.unit
def test_direct_tool_invocation_raises_original_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The initial TypeError should propagate if both attempts fail."""
    from src.agents import retrieval as mod

    monkeypatch.setattr(mod, "create_agent", lambda *_, **__: _raising_agent())
    agent = mod.RetrievalAgent(llm=None, tools_data={})

    class _AlwaysTypeError:
        def __call__(self, *args: object, **kwargs: object) -> str:
            """Always raise to surface the original TypeError."""
            raise TypeError("kwargs not supported")

    _set_tool(agent, _AlwaysTypeError())

    result = agent._call_tool_directly(
        "q", "hybrid", use_dspy=False, use_graphrag=False
    )
    assert result.error == "kwargs not supported"
    assert result.strategy_used == "hybrid_failed"
