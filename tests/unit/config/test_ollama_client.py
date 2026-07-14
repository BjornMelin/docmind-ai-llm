from __future__ import annotations

from typing import Any

import pytest
from ollama import ChatResponse, Message
from pydantic import SecretStr

from src.config.ollama_client import (
    get_ollama_web_tools,
    ollama_chat,
    resolve_ollama_auth,
)
from src.config.settings import DocMindSettings, SecurityConfig


class _FakeClient:
    def __init__(self) -> None:
        self.chat_calls: list[dict[str, Any]] = []

    def chat(self, *_, **kwargs: Any) -> ChatResponse:
        self.chat_calls.append(kwargs)
        return ChatResponse.model_validate(
            {
                "message": Message(role="assistant", content="ok"),
                "logprobs": [
                    {
                        "token": "ok",
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": "ok", "logprob": -0.1},
                            {"token": "no", "logprob": -1.2},
                        ],
                    }
                ],
            }
        )


@pytest.mark.unit
def test_resolve_ollama_auth_is_host_aware() -> None:
    cloud = DocMindSettings(
        ollama_base_url="https://ollama.com",
        ollama_api_key=SecretStr("key-123"),
        security=SecurityConfig(allow_remote_endpoints=True),
    )
    local = DocMindSettings(ollama_api_key=SecretStr("key-123"))

    assert resolve_ollama_auth(cloud) == (
        "key-123",
        {"authorization": "Bearer key-123"},
    )
    assert resolve_ollama_auth(local) == (None, {})


@pytest.mark.unit
def test_ollama_chat_logprobs_explicit() -> None:
    fake = _FakeClient()
    resp = ollama_chat(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        logprobs=True,
        top_logprobs=2,
        client=fake,  # type: ignore[arg-type]
        cfg=DocMindSettings(),
    )
    assert fake.chat_calls[-1]["logprobs"] is True
    assert fake.chat_calls[-1]["top_logprobs"] == 2
    dumped = resp.model_dump()
    assert dumped.get("logprobs")
    assert dumped["logprobs"][0]["token"] == "ok"  # noqa: S105
    assert dumped["logprobs"][0]["top_logprobs"]


@pytest.mark.unit
def test_ollama_chat_logprobs_from_settings_defaults() -> None:
    fake = _FakeClient()
    cfg = DocMindSettings(ollama_enable_logprobs=True, ollama_top_logprobs=3)
    _ = ollama_chat(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        client=fake,  # type: ignore[arg-type]
        cfg=cfg,
    )
    assert fake.chat_calls[-1]["logprobs"] is True
    assert fake.chat_calls[-1]["top_logprobs"] == 3


@pytest.mark.unit
def test_get_ollama_web_tools_disabled() -> None:
    cfg = DocMindSettings(ollama_enable_web_search=False)
    assert get_ollama_web_tools(cfg) == []


@pytest.mark.unit
def test_get_ollama_web_tools_enabled_requires_api_key() -> None:
    # Use model_construct to bypass validation that requires api_key when web
    # search is enabled.
    cfg = DocMindSettings.model_construct(
        ollama_enable_web_search=True,
        ollama_api_key=None,
        security=SecurityConfig(allow_remote_endpoints=True),
        llm_request_timeout_seconds=120,
    )
    with pytest.raises(ValueError, match=r"API key"):
        _ = get_ollama_web_tools(cfg)


@pytest.mark.unit
def test_get_ollama_web_tools_returns_client_bound_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeWebClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def web_search(self, query: str, max_results: int = 3) -> dict[str, Any]:
            self.calls.append(
                ("web_search", {"query": query, "max_results": max_results})
            )
            return {"results": []}

        def web_fetch(self, url: str) -> dict[str, Any]:
            self.calls.append(("web_fetch", {"url": url}))
            return {"content": ""}

    fake_client = _FakeWebClient()
    monkeypatch.setattr(
        "src.config.ollama_client._cached_client",
        lambda *_, **__: fake_client,
    )
    cfg = DocMindSettings(
        ollama_enable_web_search=True,
        ollama_api_key=SecretStr("key-123"),
        security=SecurityConfig(allow_remote_endpoints=True),
    )
    tools = get_ollama_web_tools(cfg)
    assert {tool.__name__ for tool in tools} == {"web_search", "web_fetch"}

    name_to_tool = {tool.__name__: tool for tool in tools}
    name_to_tool["web_search"](query="docmind", max_results=2)
    name_to_tool["web_fetch"](url="https://example.com")

    assert fake_client.calls == [
        ("web_search", {"query": "docmind", "max_results": 2}),
        ("web_fetch", {"url": "https://example.com"}),
    ]
