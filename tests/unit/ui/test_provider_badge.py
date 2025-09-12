"""Unit tests for the provider badge Streamlit component.

Uses a minimal Streamlit stub to capture markdown rendering without
touching real UI, following pytest best practices.
"""

from __future__ import annotations

import types

import pytest

from src.ui.components.provider_badge import provider_badge


class _StreamlitStub:
    def __init__(self) -> None:
        self.markdowns: list[tuple[str, bool]] = []

    def markdown(
        self, text: str, unsafe_allow_html: bool = False
    ) -> None:  # pragma: no cover - trivial
        self.markdowns.append((text, unsafe_allow_html))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "base_url_attr", "has_url"),
    [
        ("ollama", "ollama_base_url", True),
        ("lmstudio", "lmstudio_base_url", True),
        ("vllm", "vllm_base_url", True),
        ("llamacpp", "llamacpp_base_url", False),  # may use model path fallback
    ],
)
def test_provider_badge_renders_markdown(
    monkeypatch: pytest.MonkeyPatch, provider: str, base_url_attr: str, has_url: bool
) -> None:
    stubs = _StreamlitStub()

    # Patch module-level streamlit object inside provider_badge module
    import src.ui.components.provider_badge as mod

    monkeypatch.setattr(mod, "st", stubs, raising=True)

    # Build minimal settings object with required attributes
    vllm_ns = types.SimpleNamespace(model="qwen", vllm_base_url="http://vllm:8000")
    settings_obj = types.SimpleNamespace(
        llm_backend=provider,
        model=None,
        vllm=vllm_ns,
        ollama_base_url="http://ollama:11434",
        lmstudio_base_url="http://localhost:1234/v1",
        vllm_base_url="http://vllm:8000",
        llamacpp_base_url="" if not has_url else "http://llamacpp:8080/v1",
    )

    provider_badge(settings_obj)  # type: ignore[arg-type]

    assert stubs.markdowns, "Expected markdown to be rendered"
    text, allow_html = stubs.markdowns[0]
    assert allow_html is True
    assert f"Provider: <b>{provider}</b>" in text
