"""Submodule surface tests for src.agents.tools.* functions (no re-exports)."""

from __future__ import annotations

import time

import pytest
from loguru import logger

from src.agents.tool_factory import ToolFactory
from src.agents.tools.planning import plan_query, route_query
from src.agents.tools.retrieval import retrieve_documents
from src.agents.tools.router_tool import router_tool
from src.agents.tools.synthesis import synthesize_results
from src.agents.tools.validation import validate_response

pytestmark = pytest.mark.unit


def test_submodules_importable_and_invokable() -> None:
    """Smoke check that tool functions are importable and invokable."""

    def _invokable(obj: object) -> bool:
        """Return True when obj is callable or exposes a callable invoke method."""
        return callable(obj) or callable(getattr(obj, "invoke", None))

    assert _invokable(router_tool)
    assert _invokable(route_query)
    assert _invokable(plan_query)
    assert _invokable(retrieve_documents)
    assert _invokable(synthesize_results)
    assert _invokable(validate_response)


def test_patch_points_present() -> None:
    """Patch points available at concrete seams (no aggregator)."""
    assert ToolFactory is not None
    assert logger is not None
    assert hasattr(time, "perf_counter")
