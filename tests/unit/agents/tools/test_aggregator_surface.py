"""Submodule surface tests for src.agents.tools.* functions (no re-exports)."""

from __future__ import annotations

import time

import pytest
from loguru import logger

from src.agents.tool_factory import ToolFactory
from src.agents.tools.planning import ChatMemoryBuffer, plan_query, route_query
from src.agents.tools.retrieval import retrieve_documents
from src.agents.tools.router_tool import router_tool
from src.agents.tools.synthesis import synthesize_results
from src.agents.tools.validation import validate_response

pytestmark = pytest.mark.unit


def test_submodules_importable() -> None:
    """Smoke check that tool functions are importable from submodules."""
    assert callable(router_tool)
    assert callable(route_query)
    assert callable(plan_query)
    assert callable(retrieve_documents)
    assert callable(synthesize_results)
    assert callable(validate_response)


def test_patch_points_present() -> None:
    """Patch points available at concrete seams (no aggregator)."""
    assert ToolFactory is not None
    assert logger is not None
    assert ChatMemoryBuffer is not None
    assert hasattr(time, "perf_counter")
