"""Tests for retrieval query utility helpers."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_optimize_query_for_strategy_variants():
    from src.agents.retrieval import optimize_query_for_strategy

    assert optimize_query_for_strategy("what is q", "vector").endswith("?")
    assert "relationships" in optimize_query_for_strategy("foo", "graphrag").lower()
    # For hybrid short queries, expect prefix
    s = optimize_query_for_strategy("x y", "hybrid")
    assert s.lower().startswith("find comprehensive")


@pytest.mark.unit
def test_select_optimal_strategy():
    from src.agents.retrieval import select_optimal_strategy

    tools = {"kg": object()}
    assert (
        select_optimal_strategy("graph relationship link network", tools) == "graphrag"
    )
    assert select_optimal_strategy("compare A vs B", {}) == "hybrid"
    assert select_optimal_strategy("short", {}) == "vector"
