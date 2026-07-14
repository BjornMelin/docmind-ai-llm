"""Unit tests for Streamlit cache helpers.

Employs a Streamlit stub and a minimal settings-like object to validate
cache version bumping and safe clearing behavior.
"""

from __future__ import annotations

import types

import pytest

import src.ui.cache as cache_mod


class _CacheAPI:
    def __init__(self) -> None:
        self.cleared_data = 0
        self.cleared_resource = 0

    def clear(self) -> None:  # pragma: no cover - trivial
        # This method will be bound to either resource or data cache in stub
        pass


class _StreamlitStub:
    def __init__(self) -> None:
        self.cache_data = types.SimpleNamespace(clear=self._clear_data)
        self.cache_resource = types.SimpleNamespace(clear=self._clear_resource)
        self.session_state: dict[str, object] = {}
        self._api = _CacheAPI()

    def _clear_data(self) -> None:  # pragma: no cover - trivial
        self._api.cleared_data += 1

    def _clear_resource(self) -> None:  # pragma: no cover - trivial
        self._api.cleared_resource += 1


@pytest.mark.unit
def test_clear_caches_bumps_version_and_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch module-level streamlit object
    stub = _StreamlitStub()
    monkeypatch.setattr(cache_mod, "st", stub, raising=True)

    settings_obj = types.SimpleNamespace(cache_version=0)

    new_version = cache_mod.clear_caches(settings_obj)
    assert new_version == 1
    assert settings_obj.cache_version == 1

    # Ensure best-effort clears invoked
    assert stub._api.cleared_data == 1
    assert stub._api.cleared_resource == 1


@pytest.mark.unit
def test_clear_caches_closes_router_before_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.ui import chat_runtime
    from src.ui.router_session import replace_session_router
    from src.ui.vector_session import (
        VectorIndexResource,
        replace_session_vector_resource,
    )

    events: list[str] = []

    class _Router:
        def close(self) -> None:
            events.append("router")

    class _Client:
        def close(self) -> None:
            events.append("vector")

    stub = _StreamlitStub()
    replace_session_vector_resource(
        stub.session_state,
        VectorIndexResource(object(), client=_Client()),
        runtime_generation=0,
    )
    replace_session_router(
        stub.session_state,
        _Router(),
        runtime_generation=0,
    )
    monkeypatch.setattr(cache_mod, "st", stub, raising=True)
    monkeypatch.setattr(
        chat_runtime,
        "invalidate_coordinator",
        lambda: events.append("coordinator"),
    )

    cache_mod.clear_caches(types.SimpleNamespace(cache_version=0))

    assert events == ["router", "vector", "coordinator"]
    assert stub.session_state["router_engine"] is None
    assert stub.session_state["vector_index"] is None
