"""Router tool implementation extracted from monolithic tools module."""

from __future__ import annotations

import json
from typing import Annotated, Any, cast

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# Import patch points from package aggregator so tests can monkeypatch
from src.agents import tools as tools_mod


@tool
def router_tool(
    query: str,
    state: Annotated[dict | None, InjectedState] = None,
) -> str:
    """Route a query through a prebuilt RouterQueryEngine and return JSON.

    Contract (never raises across tool boundary):
    - Input: `query: str`, `state: InjectedState` containing a `router_engine` under
      either `state["router_engine"]` or `state["tools_data"]["router_engine"]`.
    - Output JSON on success:
        {"response_text": str,
         "selected_strategy": one of [
             "semantic_search", "hybrid_search", "sub_question_search",
             "knowledge_graph", "multimodal_search"
         ],
         "multimodal_used": bool,
         "hybrid_used": bool,
         "timing_ms": float}
      If unavailable, `selected_strategy` may be omitted.
    - Output JSON on error: {"error": str}
    """
    start = tools_mod.time.perf_counter()
    try:
        st = state if isinstance(state, dict) else {}
        tools_data = cast(dict, st.get("tools_data", {})) if st else {}

        router_engine = st.get("router_engine") or tools_data.get("router_engine")
        if router_engine is None:
            return json.dumps({"error": "router_engine missing in InjectedState"})

        # Execute routed query
        try:
            resp = router_engine.query(query)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            tools_mod.logger.error("router_tool query failed: {}", exc)
            return json.dumps({"error": str(exc)})

        # Extract response text with robust fallbacks
        response_text = (
            getattr(resp, "response", None)
            or getattr(resp, "text", None)
            or getattr(resp, "message", None)
            or str(resp)
        )

        # Extract selected strategy if selector metadata is available
        selected_strategy = None
        try:
            meta = getattr(resp, "metadata", None)
            if isinstance(meta, dict):
                selected_strategy = meta.get("selector_result")
        except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover - metadata shape variance
            selected_strategy = None

        timing_ms = (tools_mod.time.perf_counter() - start) * 1000.0

        # Booleans for quick telemetry
        multimodal_used = selected_strategy == "multimodal_search"
        hybrid_used = selected_strategy == "hybrid_search"

        out: dict[str, Any] = {
            "response_text": str(response_text),
            "timing_ms": round(timing_ms, 2),
        }
        if selected_strategy:
            out["selected_strategy"] = selected_strategy
            out["multimodal_used"] = multimodal_used
            out["hybrid_used"] = hybrid_used

        # Minimal telemetry logs (non-failing)
        from contextlib import suppress

        with suppress(Exception):  # pragma: no cover - logging resilience
            tools_mod.logger.info(
                "router_tool completed: strategy=%s, timing_ms=%.2f",
                selected_strategy,
                out["timing_ms"],
            )

        return json.dumps(out)

    except Exception as e:  # pylint: disable=broad-exception-caught
        tools_mod.logger.error("router_tool failed: {}", e)
        return json.dumps({"error": str(e)})
