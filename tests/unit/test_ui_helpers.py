"""Unit tests for UI helpers using a simple stub for Streamlit API.

We avoid importing real Streamlit components in assertions; instead,
we patch the module-level `st` reference with a minimal stub capturing
calls to increase coverage while keeping tests deterministic.
"""

from __future__ import annotations

import types

import pytest

import src.ui_helpers as ui


class _Sidebar:
    """Record-only sidebar stub replacing Streamlit's sidebar component."""

    def __init__(self) -> None:
        self.markdown_calls: list[str] = []
        self.write_calls: list[str] = []

    def markdown(self, text: str) -> None:  # pragma: no cover - trivial
        """Record markdown text call."""
        self.markdown_calls.append(text)

    def write(self, text: str) -> None:  # pragma: no cover - trivial
        """Record write call."""
        self.write_calls.append(text)


class _StreamlitStub:
    """Simple Streamlit replacement used to capture UI helper output."""

    def __init__(self) -> None:
        self.sidebar = _Sidebar()
        self.infos: list[str] = []

    def info(self, text: str) -> None:  # pragma: no cover - trivial
        """Record info message call."""
        self.infos.append(text)


@pytest.mark.unit
def test_build_reranker_controls_renders_without_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test build reranker controls renders without errors."""
    stub = _StreamlitStub()
    monkeypatch.setattr(ui, "st", stub, raising=True)

    fake_settings = types.SimpleNamespace(
        retrieval=types.SimpleNamespace(
            fusion_mode="rrf",
            fused_top_k=10,
            rrf_k=60,
            reranking_top_k=5,
            reranker_normalize_scores=True,
        )
    )

    ui.build_reranker_controls(fake_settings)  # type: ignore[arg-type]

    # Sanity: stub captured some writes and info
    assert stub.sidebar.markdown_calls
    assert stub.sidebar.write_calls
    # Some informational note rendered
    assert stub.infos
