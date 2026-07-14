"""Optional Ollama Cloud web search tools for LangGraph agents.

This module is feature-flagged via settings and disabled by default.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from loguru import logger
from ollama import RequestError, ResponseError

from src.agents.deadlines import remaining_deadline_seconds
from src.config.ollama_client import build_ollama_async_web_client
from src.config.settings import DocMindSettings, settings
from src.utils.log_safety import build_pii_log_entry

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
        try:
            redaction = build_pii_log_entry(detail, key_id=f"ollama_web_tools:{msg}")
            payload["detail"] = redaction.redacted
        except Exception as exc:  # pragma: no cover - defensive against settings issues
            logger.debug(
                "build_pii_log_entry failed (error_type={} error={})",
                type(exc).__name__,
                str(exc),
            )
            payload["detail"] = ""
    return _json_with_limit(payload, max_chars=_MAX_TOOL_RESULT_CHARS)


async def _ollama_web_search_impl(
    *,
    query: str,
    max_results: int,
    state: dict[str, Any],
    cfg: DocMindSettings,
) -> str:
    """Implementation of the Ollama web search tool.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
        state: Injected graph state containing the absolute deadline.
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
        remaining = remaining_deadline_seconds(
            state,
            operation="Ollama web search",
        )
        max_results_int = max(1, min(int(max_results), 10))
        client = build_ollama_async_web_client(cfg, timeout_s=remaining)
        async with client, asyncio.timeout(remaining):
            result = await client.web_search(
                query=clean_query,
                max_results=max_results_int,
            )
        return _json_with_limit(result, max_chars=_MAX_TOOL_RESULT_CHARS)
    except TimeoutError as exc:
        return _error_payload(msg="Ollama web_search deadline exceeded", exc=exc)
    except (
        ValueError,
        ConnectionError,
        RequestError,
        ResponseError,
        OSError,
        RuntimeError,
    ) as exc:
        logger.debug("ollama_web_search failed (error_type={})", type(exc).__name__)
        return _error_payload(msg="Ollama web_search failed", exc=exc)
    except Exception as exc:  # pragma: no cover - defensive against SDK/runtime errors
        redaction = build_pii_log_entry(str(exc), key_id="ollama_web_search.crashed")
        logger.error(
            "ollama_web_search crashed (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        return _error_payload(msg="Ollama web_search failed", exc=exc)


async def _ollama_web_fetch_impl(
    *,
    url: str,
    state: dict[str, Any],
    cfg: DocMindSettings,
) -> str:
    """Implementation of the Ollama web fetch tool.

    Args:
        url: The URL to fetch content from.
        state: Injected graph state containing the absolute deadline.
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
        remaining = remaining_deadline_seconds(
            state,
            operation="Ollama web fetch",
        )
        client = build_ollama_async_web_client(cfg, timeout_s=remaining)
        async with client, asyncio.timeout(remaining):
            result = await client.web_fetch(url=clean_url)
        return _json_with_limit(result, max_chars=_MAX_TOOL_RESULT_CHARS)
    except TimeoutError as exc:
        return _error_payload(msg="Ollama web_fetch deadline exceeded", exc=exc)
    except (
        ValueError,
        ConnectionError,
        RequestError,
        ResponseError,
        OSError,
        RuntimeError,
    ) as exc:
        logger.debug("ollama_web_fetch failed (error_type={})", type(exc).__name__)
        return _error_payload(msg="Ollama web_fetch failed", exc=exc)
    except Exception as exc:  # pragma: no cover - defensive against SDK/runtime errors
        redaction = build_pii_log_entry(str(exc), key_id="ollama_web_fetch.crashed")
        logger.error(
            "ollama_web_fetch crashed (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        return _error_payload(msg="Ollama web_fetch failed", exc=exc)


def _make_langchain_web_tools(cfg: DocMindSettings) -> list[Any]:
    """Factory function to create LangChain tool wrappers for Ollama web tools.

    Args:
        cfg: The DocMind settings configuration instance.

    Returns:
        A list of LangChain tools.
    """

    @tool("ollama_web_search")
    async def _ollama_web_search(
        query: str,
        state: Annotated[dict[str, Any], InjectedState],
        max_results: int = 3,
    ) -> str:
        """Run Ollama Cloud web_search (gated by settings)."""
        return await _ollama_web_search_impl(
            query=query,
            max_results=max_results,
            state=state,
            cfg=cfg,
        )

    @tool("ollama_web_fetch")
    async def _ollama_web_fetch(
        url: str,
        state: Annotated[dict[str, Any], InjectedState],
    ) -> str:
        """Run Ollama Cloud web_fetch (gated by settings)."""
        return await _ollama_web_fetch_impl(url=url, state=state, cfg=cfg)

    return [_ollama_web_search, _ollama_web_fetch]


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
    return _make_langchain_web_tools(cfg)


__all__ = ["get_langchain_web_tools"]
