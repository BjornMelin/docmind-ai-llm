"""Unit tests for the default tool registry."""

from __future__ import annotations

from types import SimpleNamespace

from src.agents.registry import DefaultToolRegistry
from src.agents.registry import tool_registry as tr


def test_build_tools_data_merges_overrides():
    """The registry should merge overrides into the default payload."""
    registry = DefaultToolRegistry()
    overrides = {"router_engine": "router", "vector": object()}

    tools_data = registry.build_tools_data(overrides)

    assert tools_data["router_engine"] == "router"
    assert "enable_dspy" in tools_data
    assert "vector" not in tools_data


def test_registry_returns_tool_collections():
    """Registry accessors should return non-empty tool collections."""
    registry = DefaultToolRegistry()

    assert registry.get_router_tools(), "router tools expected"
    assert registry.get_planner_tools(), "planner tools expected"
    assert registry.get_retrieval_tools(), "retrieval tools expected"
    assert registry.get_synthesis_tools(), "synthesis tools expected"
    assert registry.get_validation_tools(), "validation tools expected"


def test_is_jsonable_supports_lists_dicts_and_depth_limit() -> None:
    assert tr._is_jsonable([1, "x", True]) is True
    assert tr._is_jsonable({"a": [1, 2], "b": {"c": "d"}}) is True
    value: list[object] = []
    for _ in range(8):
        value = [value]
    nested = value
    assert tr._is_jsonable(nested) is False


def test_resolve_graphrag_flag_callable_raises_fail_closed() -> None:
    class _S:
        def is_graphrag_enabled(self) -> bool:  # pragma: no cover - stub
            raise ValueError("boom")

    reg = DefaultToolRegistry(app_settings=_S())  # type: ignore[arg-type]
    data = reg.build_tools_data()
    assert data["enable_graphrag"] is False


def test_resolve_graphrag_flag_fallback_honors_cfg_enabled_false() -> None:
    s = SimpleNamespace(
        enable_graphrag=True, graphrag_cfg=SimpleNamespace(enabled=False)
    )
    reg = DefaultToolRegistry(app_settings=s)  # type: ignore[arg-type]
    assert reg.build_tools_data()["enable_graphrag"] is False


def test_resolve_graphrag_flag_fallback_handles_bad_cfg() -> None:
    class _Cfg:
        @property
        def enabled(self) -> bool:  # pragma: no cover - stub
            raise ValueError("bad")

    s = SimpleNamespace(enable_graphrag=True, graphrag_cfg=_Cfg())
    reg = DefaultToolRegistry(app_settings=s)  # type: ignore[arg-type]
    assert reg.build_tools_data()["enable_graphrag"] is False


def test_reranking_fields_fail_closed_when_retrieval_missing() -> None:
    s = SimpleNamespace(enable_graphrag=False, retrieval=None)
    reg = DefaultToolRegistry(app_settings=s)  # type: ignore[arg-type]
    data = reg.build_tools_data()
    assert data["reranker_normalize_scores"] is False
    assert data["reranking_top_k"] == 0
