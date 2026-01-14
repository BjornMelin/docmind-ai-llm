"""Shared utilities for retrieval unit tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_build_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock build_llm to avoid needing full settings in router tests."""
    monkeypatch.setattr(
        "src.config.llm_factory.build_llm",
        lambda _settings: MagicMock(name="mock_llm"),
        raising=False,
    )


def get_router_tool_names(router: Any) -> list[str]:
    """Extract tool names from a router's `query_engine_tools`."""
    return [t.metadata.name for t in getattr(router, "query_engine_tools", [])]
