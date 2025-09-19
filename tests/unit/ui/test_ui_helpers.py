"""Tests for Streamlit helper utilities."""

from __future__ import annotations

from types import SimpleNamespace

from src.ui_helpers import build_reranker_controls


def test_build_reranker_controls_writes_expected_content(monkeypatch):
    captured = {"sidebar": [], "info": []}

    class _Sidebar:
        @staticmethod
        def markdown(message: str) -> None:
            captured["sidebar"].append(("markdown", message))

        @staticmethod
        def write(message: str) -> None:
            captured["sidebar"].append(("write", message))

    monkeypatch.setattr("src.ui_helpers.st.sidebar", _Sidebar)
    monkeypatch.setattr(
        "src.ui_helpers.st.info",
        lambda message: captured["info"].append(message),
    )

    settings = SimpleNamespace(
        retrieval=SimpleNamespace(
            fusion_mode="rrf",
            fused_top_k=60,
            rrf_k=30,
            reranking_top_k=20,
            reranker_normalize_scores=True,
        )
    )

    build_reranker_controls(settings)

    titles = [msg for kind, msg in captured["sidebar"] if kind == "markdown"]
    assert "Retrieval & Reranking" in titles[0]
    assert any("Fusion" in msg for kind, msg in captured["sidebar"] if kind == "write")
    assert captured["info"][0].startswith("Hybrid & Rerank tuning guide")
