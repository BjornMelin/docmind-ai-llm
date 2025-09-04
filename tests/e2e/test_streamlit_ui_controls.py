"""E2E-style test: Streamlit UI controls propagate to settings.

This test provides a minimal fake `streamlit` and `ollama` to allow importing
`src.app` without side effects and verifies that the sidebar control values are
written into `settings.retrieval`.
"""

from __future__ import annotations

import sys
import types

from src.config import settings


class _SessionState(dict):
    def __getattr__(self, name):
        """Attribute-style access to dict items."""
        try:
            return self[name]
        except KeyError as e:  # pragma: no cover - mirrors Streamlit behavior loosely
            raise AttributeError(name) from e

    def __setattr__(self, name, value) -> None:
        """Set items via attribute assignment."""
        self[name] = value


class _FakeSidebar:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def markdown(self, *_args, **_kwargs):
        """No-op markdown."""

    def info(self, *_args, **_kwargs):
        return None

    def header(self, *_args, **_kwargs):
        return None

    def radio(self, *_args, **_kwargs):
        # Force a non-default to verify propagation
        return "multimodal"

    def checkbox(self, *_args, **_kwargs):
        return False

    def number_input(self, *_args, **_kwargs):
        return 7
        return 7

    def __getattr__(self, name):
        """Provide safe fallbacks for unimplemented Streamlit APIs."""
        if name == "fragment":

            def _decorator(fn=None, **_kw):
                if fn is None:

                    def _inner(inner_fn):
                        return inner_fn

                    return _inner
                return fn

            return _decorator

        # Generic no-op function
        def _noop(*_a, **_k):
            return None

        return _noop

    def text_input(self, *_args, **_kwargs):
        return "http://localhost:11434"

    def selectbox(self, *_args, **_kwargs):
        if len(_args) > 1 and isinstance(_args[1], list) and _args[1]:
            return _args[1][0]
        return None

    def expander(self, *_args, **_kwargs):
        class _Ctxt:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Ctxt()

    def status(self, *args, **kwargs):
        class _Ctxt:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Ctxt()

    def success(self, *_args, **_kwargs):
        return None


class _FakeStreamlit(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.sidebar = _FakeSidebar()
        self.session_state = _SessionState()

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

    def title(self, *args, **kwargs):
        """No-op title."""
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

    def stop(self) -> None:  # pragma: no cover - safety
        return None

    # Top-level input controls used inside `with st.sidebar:` blocks
    def radio(self, *_args, **_kwargs):
        return "multimodal"

    def checkbox(self, *_args, **_kwargs):
        return False

    def number_input(self, *_args, **_kwargs):
        return 7
        return 7

    def __getattr__(self, name):
        """Provide safe fallbacks for unimplemented Streamlit APIs."""
        if name == "fragment":

            def _decorator(fn=None, **_kw):
                if fn is None:

                    def _inner(inner_fn):
                        return inner_fn

                    return _inner
                return fn

            return _decorator

        # Generic no-op function
        def _noop(*_a, **_k):
            return None

        return _noop


def test_streamlit_controls_propagate_to_settings(monkeypatch) -> None:
    """Fake streamlit sidebar sets values that update settings.retrieval."""
    # Insert fake modules before import
    fake_st = _FakeStreamlit()
    sys.modules["streamlit"] = fake_st

    # Minimal fake of ollama module symbols used by llama-index.llms.ollama
    fake_ollama = types.SimpleNamespace(
        list=lambda: {}, pull=lambda *_: {}, AsyncClient=object, Client=object
    )
    sys.modules["ollama"] = fake_ollama

    # Import app; sidebar controls should run and write into settings
    import importlib

    # Patch startup validation to avoid external Qdrant checks
    core = importlib.import_module("src.utils.core")
    monkeypatch.setattr(core, "validate_startup_configuration", lambda *_: None)

    importlib.invalidate_caches()
    app = importlib.import_module("src.app")
    assert app is not None

    # Verify settings were updated according to our fake UI choices
    assert settings.retrieval.reranker_mode == "multimodal"
    assert settings.retrieval.reranker_normalize_scores is False
    assert settings.retrieval.reranking_top_k == 7
