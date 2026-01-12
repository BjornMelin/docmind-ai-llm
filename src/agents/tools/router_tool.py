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
    runtime: ToolRuntime = None,  # type: ignore[assignment]
) -> str:
    """Route a query through a prebuilt RouterQueryEngine and return JSON."""
    start = time.perf_counter()
    with _ROUTER_TRACER.start_as_current_span("router_tool.query") as span:
        span.set_attribute("router.query.length", len(query))
        try:
            runtime_ctx = runtime.context if runtime is not None else None
            runtime_cfg = runtime.config if runtime is not None else None

            router_engine = None
            if isinstance(runtime_ctx, dict):
                router_engine = runtime_ctx.get("router_engine")

            if router_engine is None and isinstance(runtime_cfg, dict):
                configurable = runtime_cfg.get("configurable")
                if isinstance(configurable, dict):
                    runtime_cfg_ctx = configurable.get("runtime")
                    if isinstance(runtime_cfg_ctx, dict):
                        router_engine = runtime_cfg_ctx.get("router_engine")

            span.set_attribute("router.engine.available", router_engine is not None)
            if router_engine is None:
                message = (
                    "router_tool requires 'router_engine' via ToolRuntime.context "
                    "or runtime.configurable.runtime."
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
                logger.error("router_tool query failed: %s", exc)
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
