from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ui.cache import clear_caches

pytestmark = pytest.mark.unit


def test_clear_caches_increments_cache_version(monkeypatch) -> None:
    import streamlit as st  # type: ignore

    class _Cache:
        def __init__(self) -> None:
            self.cleared = 0

        def clear(self) -> None:
            self.cleared += 1

    monkeypatch.setattr(st, "cache_data", _Cache(), raising=False)
    monkeypatch.setattr(st, "cache_resource", _Cache(), raising=False)

    settings_obj = SimpleNamespace(cache_version=1)
    assert clear_caches(settings_obj) == 2
    assert settings_obj.cache_version == 2


@pytest.mark.parametrize(
    "settings_obj",
    [
        SimpleNamespace(cache_version="not-an-int"),
        SimpleNamespace(),
    ],
)
def test_clear_caches_recovers_from_invalid_cache_version(
    monkeypatch, settings_obj
) -> None:
    import streamlit as st  # type: ignore

    monkeypatch.setattr(
        st, "cache_data", SimpleNamespace(clear=lambda: None), raising=False
    )
    monkeypatch.setattr(
        st, "cache_resource", SimpleNamespace(clear=lambda: None), raising=False
    )

    assert clear_caches(settings_obj) == 1
    assert settings_obj.cache_version == 1
