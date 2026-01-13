"""Router tool implementation extracted from monolithic tools module."""

from __future__ import annotations

import json
import time
from contextlib import suppress
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolRuntime
from loguru import logger
from opentelemetry import trace

from src.config.settings import settings
from src.utils.telemetry import log_jsonl

_ROUTER_TRACER = trace.get_tracer("docmind.tools.router")


def _get_attr_or_key(obj: Any, key: str) -> Any | None:
    """Return attribute or dict key from an object."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _get_router_engine(obj: Any) -> Any | None:
    """Extract a router_engine attribute from an object."""
    return _get_attr_or_key(obj, "router_engine")


def _resolve_router_engine(
    state: dict | None, runtime_ctx: Any | None, runtime_cfg: Any | None
) -> Any | None:
    """Resolve the router engine from state or runtime context/config."""
    if isinstance(state, dict):
        tools_data = state.get("tools_data")
        router_engine = _get_router_engine(tools_data)
        if router_engine is not None:
            return router_engine
    router_engine = _get_router_engine(runtime_ctx)
    if router_engine is not None:
        return router_engine
    configurable = _get_attr_or_key(runtime_cfg, "configurable")
    runtime_cfg_ctx = _get_attr_or_key(configurable, "runtime")
    return _get_router_engine(runtime_cfg_ctx)


def _extract_response_text(resp: Any) -> str:
    """Extract response text from a router engine response."""
    val = getattr(resp, "response", None)
    if val is not None:
        return str(val)
    val = getattr(resp, "text", None)
    if val is not None:
        return str(val)
    val = getattr(resp, "message", None)
    if val is not None:
        return str(val)
    return str(resp)


def _extract_selected_strategy(resp: Any) -> str | None:
    """Extract the selector strategy name from response metadata."""
    with suppress(Exception):
        metadata = getattr(resp, "metadata", None)
        if isinstance(metadata, dict):
            value = metadata.get("selector_result")
            return str(value) if value is not None else None
    return None


def _build_payload(
    response_text: str, timing_ms: float, selected: str | None
) -> dict[str, Any]:
    """Build the JSON payload returned from the router tool."""
    payload: dict[str, Any] = {
        "response_text": response_text,
        "timing_ms": round(timing_ms, 2),
    }
    if selected:
        payload["selected_strategy"] = selected
        payload["multimodal_used"] = selected == "multimodal_search"
        payload["hybrid_used"] = selected == "hybrid_search"
    return payload


def _log_router_event(selected_strategy: str | None, payload: dict[str, Any]) -> None:
    """Emit structured logging and JSONL telemetry for routing."""
    with suppress(Exception):  # pragma: no cover - logging resilience
        logger.info(
            "router_tool completed: strategy={}, timing_ms={:.2f}",
            selected_strategy,
            payload["timing_ms"],
        )
        evt = {
            "router_selected": True,
            "route": selected_strategy or "unknown",
            "timing_ms": payload["timing_ms"],
        }
        if selected_strategy == "knowledge_graph":
            with suppress(Exception):
                evt["traversal_depth"] = settings.graphrag_cfg.default_path_depth
        log_jsonl(evt)


@tool
def router_tool(
    query: str,
    state: Annotated[dict | None, InjectedState] = None,
    runtime: ToolRuntime | None = None,
) -> str:
    """Route a query through a prebuilt RouterQueryEngine and return JSON."""
    start = time.perf_counter()
    with _ROUTER_TRACER.start_as_current_span("router_tool.query") as span:
        span.set_attribute("router.query.length", len(query))
        try:
            runtime_ctx = runtime.context if runtime is not None else None
            runtime_cfg = runtime.config if runtime is not None else None
            router_engine = _resolve_router_engine(state, runtime_ctx, runtime_cfg)

            span.set_attribute("router.engine.available", router_engine is not None)
            if router_engine is None:
                message = (
                    "router_tool requires 'router_engine' via state.tools_data, "
                    "ToolRuntime.context, or runtime.config.configurable.runtime."
                )
                span.set_attribute("router.success", False)
                span.set_attribute("router.error", message)
                logger.error(message)
                return json.dumps({"error": message})

            try:
                resp = router_engine.query(query)
            except Exception as exc:
                span.set_attribute("router.success", False)
                span.set_attribute("router.error", str(exc))
                logger.error("router_tool query failed: {}", exc)
                return json.dumps({"error": str(exc)})

            response_text = _extract_response_text(resp)
            selected_strategy = _extract_selected_strategy(resp)

            timing_ms = (time.perf_counter() - start) * 1000.0
            payload = _build_payload(response_text, timing_ms, selected_strategy)

            span.set_attribute(
                "router.selected_strategy", selected_strategy or "unknown"
            )
            span.set_attribute("router.success", True)
            span.set_attribute("router.latency_ms", payload["timing_ms"])

            _log_router_event(selected_strategy, payload)

            return json.dumps(payload)

        except Exception as exc:
            span.set_attribute("router.success", False)
            span.set_attribute("router.error", str(exc))
            logger.error("router_tool failed: {}", exc)
            return json.dumps({"error": str(exc)})
