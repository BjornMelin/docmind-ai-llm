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
