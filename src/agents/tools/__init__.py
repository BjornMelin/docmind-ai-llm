"""Agents Tools package aggregator.

Exposes public tool functions under `src.agents.tools` while organizing
implementation across cohesive submodules. Maintains stable import surface
for existing tests and downstream code.

Public API:
- router_tool
- route_query
- plan_query
- retrieve_documents
- synthesize_results
- validate_response

Note: Certain dependencies (e.g., `ToolFactory`, `logger`, `ChatMemoryBuffer`,
`time`) are bound here so tests can patch `src.agents.tools.*` directly.
Submodules import these from this package to keep patch paths stable.
"""

from __future__ import annotations

# pylint: disable=cyclic-import
# pylint: disable=useless-import-alias
# Bind patchable dependencies at package level
import time as time  # re-exported for tests to patch perf_counter

from llama_index.core.memory import ChatMemoryBuffer as ChatMemoryBuffer
from loguru import logger as logger  # logging handle

from src.agents.tool_factory import ToolFactory as ToolFactory

from .planning import plan_query, route_query
from .retrieval import retrieve_documents

# Re-export tool functions from submodules (import order matters for cycles)
from .router_tool import router_tool
from .synthesis import synthesize_results
from .validation import validate_response

__all__ = [
    "ChatMemoryBuffer",
    # Patch points
    "ToolFactory",
    "logger",
    "plan_query",
    "retrieve_documents",
    "route_query",
    "router_tool",
    "synthesize_results",
    "time",
    "validate_response",
]
