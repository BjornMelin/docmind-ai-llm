"""Aggregator surface tests for src.agents.tools re-exports and patch points."""

from __future__ import annotations

import pytest

from src.agents.tools import (
    ChatMemoryBuffer,
    ToolFactory,
    logger,
    plan_query,
    retrieve_documents,
    route_query,
    router_tool,
    synthesize_results,
    time,
    validate_response,
)

pytestmark = pytest.mark.unit


def test_reexports_present() -> None:
    """Smoke check that tool functions are re-exported by aggregator."""
    assert callable(router_tool)
    assert callable(route_query)
    assert callable(plan_query)
    assert callable(retrieve_documents)
    assert callable(synthesize_results)
    assert callable(validate_response)


def test_patch_points_present() -> None:
    """Patch points exposed for tests to monkeypatch in aggregator namespace."""
    assert ToolFactory is not None
    assert logger is not None
    assert ChatMemoryBuffer is not None
    assert hasattr(time, "perf_counter")
