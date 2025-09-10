"""Router factory for GraphRAG composition (SPEC-006, ADR-038).

Builds a RouterQueryEngine with vector and graph tools when a healthy
PropertyGraphIndex is provided; otherwise, returns a vector-only router.

Selector choice:
- Prefer PydanticSingleSelector when OpenAI LLM is available (function calling
  offers better routing for tool selection).
- Fallback to LLMSingleSelector otherwise.
"""

from __future__ import annotations

from typing import Any

from llama_index.core import Settings
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from loguru import logger


def _has_openai_llm() -> bool:
    """Heuristic check whether Settings.llm is an OpenAI-backed model."""
    llm = getattr(Settings, "llm", None)
    if llm is None:
        return False
    cls = type(llm).__name__.lower()
    return "openai" in cls or "gpt" in getattr(llm, "model", "").lower()


def _choose_selector() -> Any:
    """Choose selector with preference for PydanticSingleSelector when available."""
    if _has_openai_llm():
        try:
            # Available in LlamaIndex when OpenAI installed
            from llama_index.core.selectors import PydanticSingleSelector

            return PydanticSingleSelector.from_defaults()
        except Exception as exc:  # pragma: no cover - optional
            logger.debug("PydanticSingleSelector unavailable, falling back: %s", exc)
    return LLMSingleSelector.from_defaults()


def _graph_healthy(pg_index: Any) -> bool:
    store = getattr(pg_index, "property_graph_store", None)
    if store is None:
        return False
    try:
        # Ensure basic access works; avoid expensive calls
        _ = next(iter(store.get_nodes()))  # type: ignore[attr-defined]
        return True
    except Exception:  # pragma: no cover - best-effort
        return False


def build_router_engine(
    vector_index: Any, pg_index: Any | None, settings: Any
) -> RouterQueryEngine:
    """Build a RouterQueryEngine composed of vector and graph tools.

    Args:
        vector_index: LlamaIndex VectorStoreIndex or compatible with as_query_engine.
        pg_index: LlamaIndex PropertyGraphIndex or None.
        settings: App settings (unused except for future flags; kept for parity).

    Returns:
        RouterQueryEngine: Configured router.
    """
    del settings
    tools: list[QueryEngineTool] = []

    vector_qe = vector_index.as_query_engine()
    tools.append(
        QueryEngineTool(
            query_engine=vector_qe,
            metadata=ToolMetadata(
                name="vector",
                description="Vector search over ingested documents",
            ),
        )
    )

    if pg_index is not None and _graph_healthy(pg_index):
        graph_qe = pg_index.as_query_engine(include_text=True)
        tools.append(
            QueryEngineTool(
                query_engine=graph_qe,
                metadata=ToolMetadata(
                    name="graph",
                    description=(
                        "PropertyGraphIndex query engine for entity relations (depth=1)"
                    ),
                ),
            )
        )
    else:
        logger.info("Graph index missing or unhealthy; using vector-only router")

    selector = _choose_selector()
    router = RouterQueryEngine(selector=selector, query_engine_tools=tools)
    return router


__all__ = ["build_router_engine"]
