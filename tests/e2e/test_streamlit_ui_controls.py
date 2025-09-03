"""E2E-style test: Streamlit UI controls propagate to settings.

This test provides a minimal fake `streamlit` and `ollama` to allow importing
`src.app` without side effects and verifies that the sidebar control values are
written into `settings.retrieval`.
"""

from __future__ import annotations

import sys
import types

from src.config import settings


class _FakeSidebar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def markdown(self, *_args, **_kwargs):
        """No-op markdown."""

    def radio(self, *_args, **_kwargs):
        # Force a non-default to verify propagation
        return "multimodal"

    def checkbox(self, *_args, **_kwargs):
        return False

    def number_input(self, *_args, **_kwargs):
        return 7


class _FakeStreamlit(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.sidebar = _FakeSidebar()
        self.session_state = {}

    # Front-page UI operations as no-ops
    def set_page_config(self, *args, **kwargs):
        """No-op."""

    def info(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def selectbox(self, *args, **kwargs):
        # Return default index
        return args[1][0] if len(args) > 1 else None

    def markdown(self, *args, **kwargs):
        return None

    def status(self, *args, **kwargs):
        class _Ctxt:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Ctxt()

    def error(self, *args, **kwargs):
        return None

    def spinner(self, *args, **kwargs):
        class _Ctxt:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Ctxt()


def test_streamlit_controls_propagate_to_settings(monkeypatch) -> None:
    """Fake streamlit sidebar sets values that update settings.retrieval."""
    # Insert fake modules before import
    fake_st = _FakeStreamlit()
    sys.modules["streamlit"] = fake_st

    fake_ollama = types.SimpleNamespace(list=lambda: {}, pull=lambda *_: {})
    sys.modules["ollama"] = fake_ollama

    # Import app; sidebar controls should run and write into settings
    import importlib

    importlib.invalidate_caches()
    app = importlib.import_module("src.app")
    assert app is not None

    # Verify settings were updated according to our fake UI choices
    assert settings.retrieval.reranker_mode == "multimodal"
    assert settings.retrieval.reranker_normalize_scores is False
    assert settings.retrieval.reranking_top_k == 7
