"""Unit tests for Ollama web tool wrappers."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from src.agents.tools import ollama_web_tools as mod


class _FakeAsyncClient:
    def __init__(self, *, search: Any = None, fetch: Any = None) -> None:
        self.search = search
        self.fetch = fetch
        self.closed = False

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.closed = True

    async def web_search(self, **_kwargs: object) -> Any:
        if isinstance(self.search, Exception):
            raise self.search
        if callable(self.search):
            return await self.search()
        return self.search

    async def web_fetch(self, **_kwargs: object) -> Any:
        if isinstance(self.fetch, Exception):
            raise self.fetch
        return self.fetch


def _config() -> mod.DocMindSettings:
    return mod.DocMindSettings(
        ollama_enable_web_search=True,
        ollama_api_key="key-123",
        security={"allow_remote_endpoints": True},
    )


def _state(*, seconds: float = 1.0) -> dict[str, float]:
    return {"deadline_ts": time.monotonic() + seconds}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ollama_web_search_serializes_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeAsyncClient(
        search={"results": [{"title": "DocMind", "url": "https://example.com"}]}
    )
    monkeypatch.setattr(mod, "build_ollama_async_web_client", lambda *_a, **_k: client)
    tool = mod.get_langchain_web_tools(_config())[0]

    payload = await tool.coroutine(
        query="docmind",
        max_results=2,
        state=_state(),
    )
    data = json.loads(payload)
    assert "results" in data
    assert client.closed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ollama_web_fetch_validates_url_before_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[None] = []
    monkeypatch.setattr(
        mod,
        "build_ollama_async_web_client",
        lambda *_a, **_k: constructed.append(None),
    )
    tool = mod.get_langchain_web_tools(_config())[1]

    payload = await tool.coroutine(url="", state=_state())
    data = json.loads(payload)
    assert "error" in data
    assert constructed == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ollama_web_search_handles_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeAsyncClient(search=ConnectionError("offline"))
    monkeypatch.setattr(mod, "build_ollama_async_web_client", lambda *_a, **_k: client)
    tool = mod.get_langchain_web_tools(_config())[0]

    payload = await tool.coroutine(
        query="docmind",
        max_results=2,
        state=_state(),
    )
    data = json.loads(payload)
    assert "error" in data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_langchain_web_tools_binds_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    client = _FakeAsyncClient(search={"results": [{"query": "docmind"}]})

    def fake_build(cfg: object, **_kwargs: object) -> _FakeAsyncClient:
        calls.append(cfg)
        return client

    monkeypatch.setattr(mod, "build_ollama_async_web_client", fake_build)

    # Build cfg with nested security override to avoid in-place mutation
    cfg = mod.DocMindSettings(
        ollama_enable_web_search=True,
        ollama_api_key="key-123",
        security={"allow_remote_endpoints": True},
    )
    tools = mod.get_langchain_web_tools(cfg)

    out = await tools[0].coroutine(
        query="docmind",
        max_results=1,
        state=_state(),
    )
    assert json.loads(out)["results"][0]["query"] == "docmind"
    assert calls
    assert calls[0] is cfg


@pytest.mark.unit
def test_get_langchain_web_tools_treats_blank_api_key_as_missing() -> None:
    # Build cfg with nested security override to avoid in-place mutation
    cfg = mod.DocMindSettings(
        ollama_enable_web_search=True,
        ollama_api_key="   ",
        security={"allow_remote_endpoints": True},
    )
    assert mod.get_langchain_web_tools(cfg) == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_payload_truncation_keeps_valid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeAsyncClient(
        fetch={"title": "x", "content": "y" * (mod._MAX_TOOL_RESULT_CHARS + 500)}
    )
    monkeypatch.setattr(mod, "build_ollama_async_web_client", lambda *_a, **_k: client)
    tool = mod.get_langchain_web_tools(_config())[1]

    payload = await tool.coroutine(url="https://example.com", state=_state())
    assert len(payload) <= mod._MAX_TOOL_RESULT_CHARS
    data = json.loads(payload)
    assert isinstance(data, dict)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_web_search_cancels_at_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = asyncio.Event()

    async def _wait_forever() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client = _FakeAsyncClient(search=_wait_forever)
    monkeypatch.setattr(mod, "build_ollama_async_web_client", lambda *_a, **_k: client)
    tool = mod.get_langchain_web_tools(_config())[0]

    payload = await tool.coroutine(
        query="docmind",
        max_results=1,
        state=_state(seconds=0.01),
    )

    assert json.loads(payload)["error"] == "Ollama web_search deadline exceeded"
    assert cancelled.is_set()
    assert client.closed is True


@pytest.mark.unit
def test_json_with_limit_tiny_max_chars() -> None:
    """Test _json_with_limit handles tiny max_chars with early floor return."""
    val = "some long string"
    # Minimal token '""' has length 2.
    assert mod._json_with_limit(val, max_chars=0) == '""'
    assert mod._json_with_limit(val, max_chars=-1) == '""'
    assert mod._json_with_limit(val, max_chars=1) == '""'
    # For exactly 2, it still returns minimal because search finds mid=0 -> "" -> '""'
    assert mod._json_with_limit(val, max_chars=2) == '""'
