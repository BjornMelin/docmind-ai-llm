"""Unit tests for Ollama web tool wrappers."""

from __future__ import annotations

import json

import pytest

from src.agents.tools import ollama_web_tools as mod


@pytest.mark.unit
def test_ollama_web_search_serializes_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_web_search(*_, **__) -> dict[str, object]:
        return {"results": [{"title": "DocMind", "url": "https://example.com"}]}

    monkeypatch.setattr(mod, "_resolve_web_tool", lambda _name, *, cfg: fake_web_search)
    payload = mod.ollama_web_search.invoke({"query": "docmind", "max_results": 2})
    data = json.loads(payload)
    assert "results" in data


@pytest.mark.unit
def test_ollama_web_fetch_handles_missing_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_resolve_web_tool", lambda _name, *, cfg: None)
    payload = mod.ollama_web_fetch.invoke({"url": "https://example.com"})
    data = json.loads(payload)
    assert "error" in data


@pytest.mark.unit
def test_ollama_web_search_handles_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_web_search(*_, **__) -> dict[str, object]:
        raise ConnectionError("offline")

    monkeypatch.setattr(mod, "_resolve_web_tool", lambda _name, *, cfg: fake_web_search)
    payload = mod.ollama_web_search.invoke({"query": "docmind", "max_results": 2})
    data = json.loads(payload)
    assert "error" in data


@pytest.mark.unit
def test_get_langchain_web_tools_binds_cfg(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def fake_web_search(*, query: str, max_results: int = 3) -> dict[str, object]:
        return {"results": [{"query": query, "max_results": max_results}]}

    def fake_web_fetch(*, url: str) -> dict[str, object]:
        return {"content": url}

    fake_web_search.__name__ = "web_search"
    fake_web_fetch.__name__ = "web_fetch"

    def fake_get_tools(cfg):
        calls.append(cfg)
        return [fake_web_search, fake_web_fetch]

    monkeypatch.setattr(mod, "get_ollama_web_tools", fake_get_tools)

    # Build cfg with nested security override to avoid in-place mutation
    cfg = mod.DocMindSettings(
        ollama_enable_web_search=True,
        ollama_api_key="key-123",
        security={"allow_remote_endpoints": True},
    )
    tools = mod.get_langchain_web_tools(cfg)

    out = tools[0].invoke({"query": "docmind", "max_results": 1})
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
def test_tool_payload_truncation_keeps_valid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_web_fetch(*, url: str) -> dict[str, object]:
        return {"title": "x", "content": "y" * (mod._MAX_TOOL_RESULT_CHARS + 500)}

    monkeypatch.setattr(mod, "_resolve_web_tool", lambda _name, *, cfg: fake_web_fetch)
    payload = mod.ollama_web_fetch.invoke({"url": "https://example.com"})
    assert len(payload) <= mod._MAX_TOOL_RESULT_CHARS
    data = json.loads(payload)
    assert isinstance(data, dict)


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
