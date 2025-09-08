"""UI readonly panel rendering tests (smoke)."""

from types import SimpleNamespace

from src.ui_helpers import build_reranker_controls


class _SB:
    def markdown(self, *_args, **_kwargs):
        return None

    def write(self, *_args, **_kwargs):
        return None


class _ST:
    sidebar = _SB()

    def info(self, *_args, **_kwargs):  # mimic st.info
        return None


def test_ui_readonly_panel_renders(monkeypatch):
    """Panel renders without exceptions when Streamlit is stubbed."""
    monkeypatch.setenv("STREAMLIT_SERVER_HEADLESS", "1")

    # Mock streamlit
    from src import ui_helpers as uih

    uih.st = _ST()  # type: ignore

    s = SimpleNamespace(
        retrieval=SimpleNamespace(
            fusion_mode="rrf",
            fused_top_k=60,
            rrf_k=60,
            reranking_top_k=10,
            reranker_normalize_scores=True,
        )
    )
    build_reranker_controls(s)  # no exception means render OK
