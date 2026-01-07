"""Shared utilities for retrieval unit tests."""

from __future__ import annotations

from typing import Any


def get_router_tool_names(router: Any) -> list[str]:
    """Extract tool names from a router's `query_engine_tools`."""
    return [t.metadata.name for t in getattr(router, "query_engine_tools", [])]
