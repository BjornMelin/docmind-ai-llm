"""Unit tests for llama_index adapter utilities."""

from __future__ import annotations

from unittest import mock

import pytest

from src.retrieval.llama_index_adapter import (
    MissingLlamaIndexError,
    get_llama_index_adapter,
    llama_index_available,
    set_llama_index_adapter,
)


def test_llama_index_available_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returning ``False`` when ``find_spec`` cannot locate the module."""
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: None if name == "llama_index.core" else mock.DEFAULT,
    )
    assert llama_index_available() is False


def test_get_adapter_missing_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surface a helpful install hint when the dependency is absent."""
    set_llama_index_adapter(None)
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.llama_index_available",
        lambda: False,
    )
    with pytest.raises(MissingLlamaIndexError) as exc_info:
        get_llama_index_adapter(force_reload=True)
    assert "pip install docmind_ai_llm[llama]" in str(exc_info.value)
