from __future__ import annotations

from typing import Any

import pytest
from ollama import (
    ChatResponse,
    EmbedResponse,
    GenerateResponse,
    Message,
)
from pydantic import BaseModel

from src.config.ollama_client import (
    get_ollama_web_tools,
    ollama_chat,
    ollama_chat_structured,
    ollama_embed,
    ollama_generate,
)
from src.config.settings import DocMindSettings, SecurityConfig


class _FakeClient:
    def __init__(self) -> None:
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []
        self.generate_calls: list[dict[str, Any]] = []

    def chat(self, *_, **kwargs: Any) -> ChatResponse:
        self.chat_calls.append(kwargs)
        return ChatResponse(
            message=Message(role="assistant", content="ok"),
            logprobs=[
                {
                    "token": "ok",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": "ok", "logprob": -0.1},
                        {"token": "no", "logprob": -1.2},
                    ],
                }
            ],
        )

    def embed(self, *_, **kwargs: Any) -> EmbedResponse:
        self.embed_calls.append(kwargs)
        return EmbedResponse(embeddings=[[0.0, 1.0]])

    def generate(self, *_, **kwargs: Any) -> GenerateResponse:
        self.generate_calls.append(kwargs)
        return GenerateResponse(response="ok")


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
def test_ollama_embed_dimensions_from_settings() -> None:
    fake = _FakeClient()
    cfg = DocMindSettings(ollama_embed_dimensions=384)
    _ = ollama_embed(
        model="nomic-embed-text",
        inputs="hello",
        client=fake,  # type: ignore[arg-type]
        cfg=cfg,
    )
    assert fake.embed_calls[-1]["dimensions"] == 384


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
def test_ollama_generate_logprobs_from_settings_defaults() -> None:
    fake = _FakeClient()
    cfg = DocMindSettings(ollama_enable_logprobs=True, ollama_top_logprobs=4)
    _ = ollama_generate(
        model="test",
        prompt="hi",
        stream=False,
        client=fake,  # type: ignore[arg-type]
        cfg=cfg,
    )
    assert fake.generate_calls[-1]["logprobs"] is True
    assert fake.generate_calls[-1]["top_logprobs"] == 4


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
        ollama_api_key="key-123",
        security={"allow_remote_endpoints": True},
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


@pytest.mark.unit
def test_ollama_chat_structured_validates_pydantic() -> None:
    class Out(BaseModel):
        answer: str

    class _StructuredFakeClient(_FakeClient):
        def chat(self, *_, **kwargs: Any) -> ChatResponse:
            self.chat_calls.append(kwargs)
            return ChatResponse(
                message=Message(
                    role="assistant",
                    content='{"answer":"ok"}',
                ),
            )

    fake = _StructuredFakeClient()
    cfg = DocMindSettings()
    out = ollama_chat_structured(
        model="test",
        messages=[{"role": "user", "content": "hi"}],
        output_model=Out,
        client=fake,  # type: ignore[arg-type]
        cfg=cfg,
    )
    assert out.answer == "ok"
    assert fake.chat_calls[-1]["format"]
