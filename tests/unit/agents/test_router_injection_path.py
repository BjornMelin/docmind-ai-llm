"""Unit tests for router injection fast-path in retrieve_documents (SPEC-040)."""

from __future__ import annotations

import json

import pytest

from src.agents.tools.retrieval import retrieve_documents
from src.config import settings

pytestmark = pytest.mark.unit


class _Node:
    def __init__(self, text: str, metadata: dict[str, object]) -> None:
        self.text = text
        self.metadata = metadata


class _NodeWithScore:
    def __init__(self, node: _Node, score: float) -> None:
        self.node = node
        self.score = score


class _Resp:
    def __init__(self) -> None:
        node = _Node("doc-1", {"source": "a.txt"})
        self.source_nodes = [_NodeWithScore(node, 0.42)]


class _Router:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def query(self, query: str) -> _Resp:
        self.calls.append(query)
        return _Resp()


def test_retrieve_documents_uses_router_engine_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.agents, "enable_router_injection", True)
    router = _Router()

    from langchain.tools import ToolRuntime

    runtime = ToolRuntime(
        state={},
        context={"router_engine": router},
        config={},
        stream_writer=lambda _chunk: None,
        tool_call_id=None,
        store=None,
    )

    out = retrieve_documents.func(
        query="q",
        state={},
        runtime=runtime,
        strategy="hybrid",
        use_dspy=True,
        use_graphrag=False,
    )
    payload = json.loads(out)
    assert payload["strategy_used"] == "router_injection"
    assert payload["router_injection"] is True
    assert payload["document_count"] == 1
    assert payload["documents"][0]["content"] == "doc-1"
    assert router.calls == ["q"]
