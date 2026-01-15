"""Optional Ollama Cloud web search tools and example loop.

This module is feature-flagged via settings. It is not used by default in
DocMind flows, but keeps the codebase ready to adopt Ollama-native tool
calling and web search in a controlled way.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool
from loguru import logger
from ollama import Message, RequestError, ResponseError

from src.config.ollama_client import get_ollama_web_tools, ollama_chat
from src.config.settings import DocMindSettings, settings

_MAX_TOOL_RESULT_CHARS = 8_000
_MAX_ERROR_CHARS = 500
_MAX_QUERY_CHARS = 1_000
_MAX_URL_CHARS = 2_000


def _json_dumps(value: Any) -> str:
    """Serialize a value to a JSON string with standardized settings.

    Args:
        value: The value to serialize.

    Returns:
        A JSON string.
    """
    return json.dumps(
        value,
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _truncate_str(value: str, max_len: int) -> str:
    """Truncate a string to a maximum length with a suffix if needed.

    Args:
        value: The string to truncate.
        max_len: The maximum allowed length.

    Returns:
        The truncated string, possibly with a "…[truncated]" suffix.
    """
    if len(value) <= max_len:
        return value
    suffix = "…[truncated]"
    if max_len <= len(suffix):
        return value[:max_len]
    return value[: max_len - len(suffix)] + suffix


def _to_jsonable(value: Any) -> Any:
    """Convert tool payloads to JSON-serializable Python values.

    Args:
        value: Tool response object or primitive payload.

    Returns:
        JSON-serializable Python value.
    """
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "model_dump_json"):
        raw = str(value.model_dump_json())
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return value


def _shrink_jsonable(
    value: Any,
    *,
    max_str_len: int,
    max_list_items: int,
    depth: int,
) -> Any:
    """Recursively shrink a JSON-serializable object to meet size constraints.

    This function truncates long strings and limits the number of items in lists/dicts
    to prevent extremely large tool results that could exceed model context limits.

    Args:
        value: The JSON-serializable value to shrink.
        max_str_len: Maximum length for string values.
        max_list_items: Maximum number of items in lists, tuples, or sets.
        depth: Maximum recursion depth to prevent stack overflow.

    Returns:
        A shrunk version of the input value.
    """
    if depth <= 0:
        return "...[max depth]"
    if isinstance(value, str):
        return _truncate_str(value, max_str_len)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(k): _shrink_jsonable(
                v,
                max_str_len=max_str_len,
                max_list_items=max_list_items,
                depth=depth - 1,
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        omitted = max(0, len(items) - max_list_items)
        items = items[:max_list_items]
        out = [
            _shrink_jsonable(
                item,
                max_str_len=max_str_len,
                max_list_items=max_list_items,
                depth=depth - 1,
            )
            for item in items
        ]
        if omitted:
            out.append(f"...[+{omitted} items omitted]")
        return out
    return _truncate_str(str(value), max_str_len)


def _json_with_limit(value: Any, *, max_chars: int) -> str:
    """Serialize payloads as valid JSON without truncating mid-token.

    This function attempts to shrink the JSON payload by truncating long strings
    or limiting list items until it fits within max_chars.

    Args:
        value: The value to serialize.
        max_chars: The maximum number of characters allowed in the output string.

    Returns:
        A JSON string that fits within the character limit.
    """
    minimal_token = _json_dumps("")
    if max_chars <= 0 or max_chars < len(minimal_token):
        return minimal_token

    jsonable = _to_jsonable(value)
    payload = _json_dumps(jsonable)
    if len(payload) <= max_chars:
        return payload

    if isinstance(jsonable, str):
        # Find the largest prefix that still yields a JSON string <= max_chars.
        lo, hi = 0, len(jsonable)
        best: str | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = _truncate_str(jsonable, mid)
            dumped = _json_dumps(candidate)
            if len(dumped) <= max_chars:
                best = dumped
                lo = mid + 1
            else:
                hi = mid - 1
        return best or _json_dumps("")

    if isinstance(jsonable, (dict, list, tuple, set)):
        for max_list_items in (10, 5, 3, 1):
            for max_str_len in (2_000, 1_000, 500, 250, 120, 60, 30):
                shrunk = _shrink_jsonable(
                    jsonable,
                    max_str_len=max_str_len,
                    max_list_items=max_list_items,
                    depth=6,
                )
                dumped = _json_dumps(shrunk)
                if len(dumped) <= max_chars:
                    return dumped

    # Last-resort: return a JSON object containing a preview of the JSON payload.
    lo, hi = 0, len(payload)
    best_preview = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = payload[:mid]
        dumped = _json_dumps({"truncated": True, "preview": candidate})
        if len(dumped) <= max_chars:
            best_preview = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return _json_dumps({"truncated": True, "preview": best_preview})


def _resolve_web_tool(name: str, *, cfg: DocMindSettings) -> Callable[..., Any] | None:
    """Return the bound Ollama web tool implementation by name.

    Args:
        name: Tool name to resolve ("web_search" or "web_fetch").
        cfg: DocMind settings instance to bind tool clients/policy.

    Returns:
        Callable tool function when available; otherwise None.
    """
    tools = get_ollama_web_tools(cfg)
    for tool_fn in tools:
        if getattr(tool_fn, "__name__", "") == name:
            return tool_fn
    return None


def _error_payload(*, msg: str, exc: Exception | None = None) -> str:
    """Create a standardized JSON error payload for tool responses.

    Args:
        msg: The error message.
        exc: Optional exception that triggered the error.

    Returns:
        A JSON-serialized error response string.
    """
    reason = exc.__class__.__name__ if exc is not None else None
    detail = str(exc) if exc is not None else None
    payload: dict[str, object] = {"error": msg}
    if reason:
        payload["reason"] = reason
    if detail:
        payload["detail"] = detail[:_MAX_ERROR_CHARS]
    return _json_with_limit(payload, max_chars=_MAX_TOOL_RESULT_CHARS)


def _ollama_web_search_impl(
    *, query: str, max_results: int, cfg: DocMindSettings
) -> str:
    """Implementation of the Ollama web search tool.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        cfg: The DocMind settings configuration to bind tool policies.

    Returns:
        A JSON string containing search results or an error payload.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        return _error_payload(msg="Missing query")
    if len(clean_query) > _MAX_QUERY_CHARS:
        return _error_payload(msg="Query too long")

    try:
        fn = _resolve_web_tool("web_search", cfg=cfg)
        if fn is None:
            return _error_payload(msg="Ollama web_search tool unavailable")
        max_results_int = max(1, min(int(max_results), 10))
        result = fn(query=clean_query, max_results=max_results_int)
        return _json_with_limit(result, max_chars=_MAX_TOOL_RESULT_CHARS)
    except (
        ValueError,
        ConnectionError,
        RequestError,
        ResponseError,
        OSError,
        RuntimeError,
    ) as exc:
        logger.debug("ollama_web_search failed: %s", exc.__class__.__name__)
        return _error_payload(msg="Ollama web_search failed", exc=exc)
    except Exception as exc:  # pragma: no cover - defensive against SDK/runtime errors
        logger.exception("ollama_web_search crashed: %s", exc.__class__.__name__)
        return _error_payload(msg="Ollama web_search failed", exc=exc)


def _ollama_web_fetch_impl(*, url: str, cfg: DocMindSettings) -> str:
    """Implementation of the Ollama web fetch tool.

    Args:
        url: The URL to fetch content from.
        cfg: The DocMind settings configuration to bind tool policies.

    Returns:
        A JSON string containing the fetched content or an error payload.
    """
    clean_url = (url or "").strip()

    # Validate URL before attempting fetch
    validation_error: str | None = None
    if not clean_url:
        validation_error = "Missing url"
    elif len(clean_url) > _MAX_URL_CHARS:
        validation_error = "URL too long"
    elif not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        validation_error = "URL must start with http:// or https://"
    if validation_error:
        return _error_payload(msg=validation_error)

    try:
        fn = _resolve_web_tool("web_fetch", cfg=cfg)
        if fn is None:
            return _error_payload(msg="Ollama web_fetch tool unavailable")
        result = fn(url=clean_url)
        return _json_with_limit(result, max_chars=_MAX_TOOL_RESULT_CHARS)
    except (
        ValueError,
        ConnectionError,
        RequestError,
        ResponseError,
        OSError,
        RuntimeError,
    ) as exc:
        logger.debug("ollama_web_fetch failed: %s", exc.__class__.__name__)
        return _error_payload(msg="Ollama web_fetch failed", exc=exc)
    except Exception as exc:  # pragma: no cover - defensive against SDK/runtime errors
        logger.exception("ollama_web_fetch crashed: %s", exc.__class__.__name__)
        return _error_payload(msg="Ollama web_fetch failed", exc=exc)


def _make_langchain_web_tools(cfg: DocMindSettings) -> list[Any]:
    """Factory function to create LangChain tool wrappers for Ollama web tools.

    Args:
        cfg: The DocMind settings configuration instance.

    Returns:
        A list of LangChain tools.
    """

    @tool("ollama_web_search")
    def _ollama_web_search(query: str, max_results: int = 3) -> str:
        """Run Ollama Cloud web_search (gated by settings)."""
        return _ollama_web_search_impl(query=query, max_results=max_results, cfg=cfg)

    @tool("ollama_web_fetch")
    def _ollama_web_fetch(url: str) -> str:
        """Run Ollama Cloud web_fetch (gated by settings)."""
        return _ollama_web_fetch_impl(url=url, cfg=cfg)

    return [_ollama_web_search, _ollama_web_fetch]


@tool("ollama_web_search")
def ollama_web_search(query: str, max_results: int = 3) -> str:
    """Run Ollama Cloud web_search (gated by settings).

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        JSON string with search results or an error payload.
    """
    return _ollama_web_search_impl(query=query, max_results=max_results, cfg=settings)


@tool("ollama_web_fetch")
def ollama_web_fetch(url: str) -> str:
    """Run Ollama Cloud web_fetch (gated by settings).

    Args:
        url: URL to fetch.

    Returns:
        JSON string with fetched content or an error payload.
    """
    return _ollama_web_fetch_impl(url=url, cfg=settings)


def get_langchain_web_tools(
    cfg: DocMindSettings = settings,
) -> list[Any]:
    """Return LangChain tool wrappers when web tools are enabled.

    Args:
        cfg: DocMind settings.

    Returns:
        List of tool callables to expose to the agent runtime.
    """
    if not cfg.ollama_enable_web_search:
        return []
    if cfg.ollama_api_key is None:
        return []
    if not cfg.ollama_api_key.get_secret_value().strip():
        return []
    try:
        _ = get_ollama_web_tools(cfg)
    except Exception as e:
        logger.debug(
            "failed to get ollama web tools for cfg=%s: %s", cfg, e, exc_info=True
        )
        return []
    return _make_langchain_web_tools(cfg)


def run_web_search_agent(
    *,
    model: str,
    prompt: str,
    max_steps: int = 5,
    think: bool = True,
    deadline_s: float = 0.0,
    cfg: DocMindSettings = settings,
) -> str:
    """Run a minimal Ollama-native web search agent loop.

    This demonstrates the official Ollama tool-calling message contract:
    - the model emits tool_calls
    - the app executes each tool and replies with role='tool' + tool_name

    Args:
        model: Ollama model name (typically a tool-capable model).
        prompt: User question.
        max_steps: Safety cap to avoid infinite loops.
        think: Enable thinking mode for thinking-capable models.
        deadline_s: Absolute deadline timestamp (seconds since epoch). When 0,
            defaults to settings.agents.decision_timeout from "now".
        cfg: DocMind settings.

    Returns:
        Final assistant content (may be empty if the model never returns content).
    """
    tools = get_ollama_web_tools(cfg)
    available: dict[str, Callable[..., Any]] = {}
    for tool_fn in tools:
        name = getattr(tool_fn, "__name__", None)
        if isinstance(name, str) and name:
            available[name] = tool_fn

    messages: list[dict[str, Any] | Message] = [{"role": "user", "content": prompt}]
    effective_deadline = float(deadline_s)
    if effective_deadline == 0.0:
        effective_deadline = time.time() + float(cfg.agents.decision_timeout)

    for _ in range(max_steps):
        remaining_time = max(0.0, effective_deadline - time.time())
        per_call_timeout = min(
            float(cfg.llm_request_timeout_seconds),
            remaining_time,
        )
        per_call_timeout = max(0.1, per_call_timeout)
        timeout_cfg = cfg.model_copy(
            update={"llm_request_timeout_seconds": per_call_timeout}
        )
        response = ollama_chat(
            model=model,
            messages=messages,
            tools=tools,
            stream=False,
            think=think,
            cfg=timeout_cfg,
        )
        messages.append(response.message)

        tool_calls = getattr(response.message, "tool_calls", None)
        if not tool_calls:
            return str(response.message.content or "")

        for call in tool_calls:
            tool_name = str(call.function.name)
            fn = available.get(tool_name)
            if fn is None:
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": f"Tool {tool_name} not found",
                    }
                )
                continue

            # Parse arguments - Ollama may return dict or JSON string
            raw_args = call.function.arguments
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError as exc:
                    logger.debug("Failed to parse tool arguments: %s", exc)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": _error_payload(
                                msg=f"Tool {tool_name} arguments invalid JSON",
                                exc=exc,
                            ),
                        }
                    )
                    continue
                if not isinstance(parsed_args, dict):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": _error_payload(
                                msg=f"Tool {tool_name} arguments must be object",
                            ),
                        }
                    )
                    continue
                args = parsed_args
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                # Fallback for other mapping types
                args = dict(raw_args) if raw_args else {}

            try:
                result = fn(**args)
            except Exception as exc:  # pragma: no cover - defensive example loop
                logger.debug(
                    "Ollama tool call failed: %s", exc.__class__.__name__, exc_info=True
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": _error_payload(
                            msg=f"Tool {tool_name} failed",
                            exc=exc,
                        ),
                    }
                )
                continue
            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": _json_with_limit(
                        result, max_chars=_MAX_TOOL_RESULT_CHARS
                    ),
                }
            )

    logger.warning("Ollama web search agent hit max_steps=%s", max_steps)
    return ""


__all__ = [
    "get_langchain_web_tools",
    "ollama_web_fetch",
    "ollama_web_search",
    "run_web_search_agent",
]
