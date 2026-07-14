"""Unit tests for LLM factory minimal paths.

Covers unsupported backend error path to improve coverage without invoking
heavy backends.
"""

from __future__ import annotations

import types

import pytest

from src.config import llm_factory


@pytest.mark.unit
def test_build_llm_unsupported_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_settings = types.SimpleNamespace(
        llm_backend="unsupported",
    )
    monkeypatch.setattr(llm_factory, "DocMindSettings", object)  # avoid type import
    with pytest.raises(ValueError, match="Unsupported"):
        llm_factory.build_llm(fake_settings)  # type: ignore[arg-type]
