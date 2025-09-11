"""Tests for cache clearing helper in src.ui.cache."""

import types

from src.ui.cache import clear_caches


class _DummyClear:
    """Minimal object exposing a clear() method to record calls."""

    def __init__(self):
        self.cleared = False

    def clear(self):
        """Record the invocation of clear."""
        self.cleared = True


def test_clear_caches_bumps_version_and_clears(monkeypatch):
    """clear_caches increments cache_version and calls clear hooks."""
    # Fake streamlit with cache_data/resource.clear hooks
    fake_st = types.SimpleNamespace()
    fake_st.cache_data = _DummyClear()
    fake_st.cache_resource = _DummyClear()

    # Patch the streamlit module used by src.ui.cache
    import src.ui.cache as cache_mod

    monkeypatch.setattr(cache_mod, "st", fake_st)

    # Prepare a simple settings object with cache_version
    class _S:
        cache_version = 5

    s = _S()
    new_v = clear_caches(s)

    assert new_v == 6
    assert s.cache_version == 6
    assert fake_st.cache_data.cleared is True
    assert fake_st.cache_resource.cleared is True
