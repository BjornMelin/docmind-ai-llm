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

            def _get_attr_or_key(obj: Any, key: str) -> Any | None:
                if obj is None:
                    return None
                if isinstance(obj, dict):
                    return obj.get(key)
                return getattr(obj, key, None)

            def _get_router_engine(obj: Any) -> Any | None:
                return _get_attr_or_key(obj, "router_engine")

            router_engine = None
            if isinstance(state, dict):
                tools_data = state.get("tools_data")
                router_engine = _get_router_engine(tools_data)

            if router_engine is None:
                router_engine = _get_router_engine(runtime_ctx)

            if router_engine is None:
                configurable = _get_attr_or_key(runtime_cfg, "configurable")
                runtime_cfg_ctx = _get_attr_or_key(configurable, "runtime")
                router_engine = _get_router_engine(runtime_cfg_ctx)

            span.set_attribute("router.engine.available", router_engine is not None)
            if router_engine is None:
                message = (
                    "router_tool requires 'router_engine' via state.tools_data, "
                    "ToolRuntime.context, or runtime.configurable.runtime."
                )
                span.set_attribute("router.success", False)
                span.set_attribute("router.error", message)
                logger.error(message)
                raise RuntimeError(message)

            try:
                resp = router_engine.query(query)
            except Exception as exc:
                span.set_attribute("router.success", False)
                span.set_attribute("router.error", str(exc))
                logger.error("router_tool query failed: {}", exc)
                return json.dumps({"error": str(exc)})

            response_text = (
                getattr(resp, "response", None)
                or getattr(resp, "text", None)
                or getattr(resp, "message", None)
                or str(resp)
            )

            selected_strategy = None
            try:
                metadata = getattr(resp, "metadata", None)
                if isinstance(metadata, dict):
                    selected_strategy = metadata.get("selector_result")
            except Exception:
                selected_strategy = None

            timing_ms = (time.perf_counter() - start) * 1000.0
            multimodal_used = selected_strategy == "multimodal_search"
            hybrid_used = selected_strategy == "hybrid_search"

            payload: dict[str, Any] = {
                "response_text": str(response_text),
                "timing_ms": round(timing_ms, 2),
            }
            if selected_strategy:
                payload["selected_strategy"] = selected_strategy
                payload["multimodal_used"] = multimodal_used
                payload["hybrid_used"] = hybrid_used

            span.set_attribute(
                "router.selected_strategy", selected_strategy or "unknown"
            )
            span.set_attribute("router.success", True)
            span.set_attribute("router.latency_ms", payload["timing_ms"])

            with suppress(Exception):  # pragma: no cover - logging resilience
                logger.info(
                    "router_tool completed: strategy=%s, timing_ms=%.2f",
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
                        evt["traversal_depth"] = (
                            settings.graphrag_cfg.default_path_depth
                        )
                log_jsonl(evt)

            return json.dumps(payload)

        except Exception as exc:
            span.set_attribute("router.success", False)
            span.set_attribute("router.error", str(exc))
            logger.error("router_tool failed: %s", exc)
            return json.dumps({"error": str(exc)})
