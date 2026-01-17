"""Ollama SDK client wiring (official ollama-python).

Centralizes host/auth/timeout configuration and exposes small helpers for
features introduced in ollama-python >= 0.6.x (logprobs, embed dimensions,
web_search/web_fetch, thinking, structured outputs).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal, overload
from urllib.parse import urlparse

import ollama
from ollama import (
    ChatResponse,
    EmbedResponse,
    GenerateResponse,
    Message,
    Options,
    Tool,
)
from pydantic import BaseModel

from src.config.settings import DocMindSettings, settings
from src.config.settings_utils import (
    endpoint_url_allowed,
    ensure_http_scheme,
    parse_endpoint_allowlist_hosts,
)


def _normalize_url(url: str) -> str:
    """Ensure a URL has a scheme.

    Args:
        url: The URL string to normalize.

    Returns:
        URL with http:// prepended if scheme was missing.
    """
    u = (url or "").strip()
    if not u:
        return u
    if "://" not in u:
        return f"http://{u}"
    return u


def _assert_endpoint_allowed(cfg: DocMindSettings, base_url: str) -> None:
    """Validate that the Ollama host is allowed by security policy.

    Args:
        cfg: Application settings.
        base_url: The host URL to check.

    Raises:
        ValueError: If remote endpoints are disabled and host is not local
            or allowlisted.
    """
    if cfg.security.allow_remote_endpoints:
        return

    normalized = ensure_http_scheme(base_url)
    if not normalized:
        raise ValueError("Invalid Ollama host URL")

    allowed_hosts = parse_endpoint_allowlist_hosts(cfg.security.endpoint_allowlist)
    if not endpoint_url_allowed(normalized, allowed_hosts=allowed_hosts):
        raise ValueError(
            "Remote endpoints are disabled. Set allow_remote_endpoints=True "
            "or add the host to security.endpoint_allowlist."
        )


def _resolve_ollama_host(cfg: DocMindSettings) -> str:
    """Resolve Ollama host from settings.

    Args:
        cfg: Application settings.

    Returns:
        The configured Ollama host URL.
    """
    return str(cfg.ollama_base_url or "").strip().rstrip("/")


def _resolve_ollama_api_key(cfg: DocMindSettings) -> str | None:
    """Resolve Ollama API key from settings.

    Args:
        cfg: Application settings.

    Returns:
        The API key string if available, otherwise None.
    """
    if cfg.ollama_api_key is None:
        return None
    candidate = cfg.ollama_api_key.get_secret_value().strip()
    return candidate or None


def is_ollama_web_search_enabled(cfg: DocMindSettings = settings) -> bool:
    """Check if Ollama Cloud web search is enabled."""
    return bool(cfg.ollama_enable_web_search)


def _resolve_ollama_embed_dimensions(cfg: DocMindSettings) -> int | None:
    """Resolve embedding dimensions from settings."""
    return cfg.ollama_embed_dimensions


def _resolve_ollama_logprobs_defaults(cfg: DocMindSettings) -> tuple[bool, int | None]:
    """Resolve logprobs activation and top_logprobs limit."""
    if not cfg.ollama_enable_logprobs:
        return False, None
    return True, int(cfg.ollama_top_logprobs)


def _resolve_headers(cfg: DocMindSettings) -> dict[str, str]:
    """Resolve required HTTP headers for Ollama requests.

    Args:
        cfg: Application settings.

    Returns:
        Dictionary of headers (e.g., Authorization for Ollama Cloud).
    """
    headers: dict[str, str] = {}
    api_key = _resolve_ollama_api_key(cfg)
    host = _resolve_ollama_host(cfg)
    hostname = (
        (urlparse(_normalize_url(host)).hostname or "").strip().lower().rstrip(".")
    )
    if api_key and (hostname == "ollama.com" or hostname.endswith(".ollama.com")):
        headers["authorization"] = f"Bearer {api_key}"
    return headers


@lru_cache(maxsize=4)
def _cached_client(
    host: str, timeout_s: float, headers_items: tuple[tuple[str, str], ...]
) -> ollama.Client:
    """Create and cache an Ollama sync client.

    Args:
        host: Ollama host URL.
        timeout_s: Request timeout in seconds.
        headers_items: Tuple of (key, value) headers for hashing.

    Returns:
        Cached :class:`ollama.Client` instance.
    """
    return ollama.Client(
        host=host,
        timeout=timeout_s,
        headers=dict(headers_items),
    )


def get_ollama_client(cfg: DocMindSettings = settings) -> ollama.Client:
    """Return a cached Ollama sync client configured from settings/env.

    Args:
        cfg: Application settings.

    Returns:
        A shared sync client instance.
    """
    resolved_host = _resolve_ollama_host(cfg).strip()
    if not resolved_host:
        raise ValueError("No Ollama host configured")
    _assert_endpoint_allowed(cfg, resolved_host)
    timeout_s = float(cfg.llm_request_timeout_seconds)
    hdrs = _resolve_headers(cfg)
    return _cached_client(resolved_host, timeout_s, tuple(sorted(hdrs.items())))


@overload
def ollama_chat(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any] | Message],
    tools: Sequence[Mapping[str, Any] | Tool | Callable[..., Any]] | None = None,
    stream: Literal[False],
    think: bool | Literal["low", "medium", "high"] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    response_format: Literal["", "json"] | dict[str, Any] | None = None,
    options: Mapping[str, Any] | Options | None = None,
    keep_alive: float | str | None = None,
    client: ollama.Client | None = None,
    cfg: DocMindSettings = settings,
) -> ChatResponse: ...


@overload
def ollama_chat(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any] | Message],
    tools: Sequence[Mapping[str, Any] | Tool | Callable[..., Any]] | None = None,
    stream: Literal[True],
    think: bool | Literal["low", "medium", "high"] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    response_format: Literal["", "json"] | dict[str, Any] | None = None,
    options: Mapping[str, Any] | Options | None = None,
    keep_alive: float | str | None = None,
    client: ollama.Client | None = None,
    cfg: DocMindSettings = settings,
) -> Iterator[ChatResponse]: ...


def ollama_chat(  # noqa: PLR0913
    *,
    model: str,
    messages: Sequence[Mapping[str, Any] | Message],
    tools: Sequence[Mapping[str, Any] | Tool | Callable[..., Any]] | None = None,
    stream: bool,
    think: bool | Literal["low", "medium", "high"] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    response_format: Literal["", "json"] | dict[str, Any] | None = None,
    options: Mapping[str, Any] | Options | None = None,
    keep_alive: float | str | None = None,
    client: ollama.Client | None = None,
    cfg: DocMindSettings = settings,
) -> ChatResponse | Iterator[ChatResponse]:
    """Call Ollama chat with explicit streaming semantics and optional features.

    Args:
        model: Model name.
        messages: Chat history.
        tools: List of tools/functions for tool calling.
        stream: Whether to stream the response.
        think: Enable thinking features (if supported by model/server).
        logprobs: Whether to return logprobs.
        top_logprobs: Number of top logprobs to return.
        response_format: JSON schema or "json" mode.
        options: Low-level model options (temperature, etc.).
        keep_alive: Model TTL in memory.
        client: Optional client override.
        cfg: Application settings.

    Returns:
        Chat response or iterator if streaming.
    """
    c = client or get_ollama_client(cfg)
    if logprobs is None:
        enabled, top = _resolve_ollama_logprobs_defaults(cfg)
        logprobs = enabled
        top_logprobs = top if top_logprobs is None else top_logprobs
    return c.chat(
        model=model,
        messages=messages,
        tools=tools,
        stream=stream,
        think=think,
        logprobs=logprobs if logprobs else None,
        top_logprobs=top_logprobs,
        format=response_format,
        options=options,
        keep_alive=keep_alive,
    )


@overload
def ollama_generate(
    *,
    model: str,
    prompt: str,
    stream: Literal[False],
    think: bool | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    response_format: Literal["", "json"] | dict[str, Any] | None = None,
    options: Mapping[str, Any] | Options | None = None,
    keep_alive: float | str | None = None,
    client: ollama.Client | None = None,
    cfg: DocMindSettings = settings,
) -> GenerateResponse: ...


@overload
def ollama_generate(
    *,
    model: str,
    prompt: str,
    stream: Literal[True],
    think: bool | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    response_format: Literal["", "json"] | dict[str, Any] | None = None,
    options: Mapping[str, Any] | Options | None = None,
    keep_alive: float | str | None = None,
    client: ollama.Client | None = None,
    cfg: DocMindSettings = settings,
) -> Iterator[GenerateResponse]: ...


def ollama_generate(  # noqa: PLR0913
    *,
    model: str,
    prompt: str,
    stream: bool,
    think: bool | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    response_format: Literal["", "json"] | dict[str, Any] | None = None,
    options: Mapping[str, Any] | Options | None = None,
    keep_alive: float | str | None = None,
    client: ollama.Client | None = None,
    cfg: DocMindSettings = settings,
) -> GenerateResponse | Iterator[GenerateResponse]:
    """Call Ollama generate with explicit streaming semantics and optional features.

    Args:
        model: Model name.
        prompt: Initial prompt.
        stream: Whether to stream the response.
        think: Enable thinking features.
        logprobs: Whether to return logprobs.
        top_logprobs: Number of top logprobs to return.
        response_format: JSON schema or "json" mode.
        options: Low-level model options.
        keep_alive: Model TTL in memory.
        client: Optional client override.
        cfg: Application settings.

    Returns:
        Generate response or iterator if streaming.
    """
    c = client or get_ollama_client(cfg)
    if logprobs is None:
        enabled, top = _resolve_ollama_logprobs_defaults(cfg)
        logprobs = enabled
        top_logprobs = top if top_logprobs is None else top_logprobs
    return c.generate(
        model=model,
        prompt=prompt,
        stream=stream,
        think=think,
        logprobs=logprobs if logprobs else None,
        top_logprobs=top_logprobs,
        format=response_format,
        options=options,
        keep_alive=keep_alive,
    )


def ollama_embed(
    *,
    model: str,
    inputs: str | Sequence[str],
    dimensions: int | None = None,
    truncate: bool | None = None,
    options: Mapping[str, Any] | Options | None = None,
    keep_alive: float | str | None = None,
    client: ollama.Client | None = None,
    cfg: DocMindSettings = settings,
) -> EmbedResponse:
    """Call Ollama embed with optional dimensions support.

    Args:
        model: Model name.
        inputs: Input string or list of strings.
        dimensions: Vector dimensions (if supported by model).
        truncate: Whether to truncate input.
        options: Low-level model options.
        keep_alive: Model TTL in memory.
        client: Optional client override.
        cfg: Application settings.

    Returns:
        Embed response containing vectors.
    """
    c = client or get_ollama_client(cfg)
    resolved_dims = (
        dimensions if dimensions is not None else _resolve_ollama_embed_dimensions(cfg)
    )
    return c.embed(
        model=model,
        input=inputs,
        truncate=truncate,
        options=options,
        keep_alive=keep_alive,
        dimensions=resolved_dims,
    )


def get_ollama_web_tools(cfg: DocMindSettings = settings) -> list[Callable[..., Any]]:
    """Return Ollama Cloud web tools when enabled.

    Args:
        cfg: Application settings.

    Returns:
        A list of web tools (search/fetch) if enabled, otherwise an empty list.
    """
    if not is_ollama_web_search_enabled(cfg):
        return []
    api_key = _resolve_ollama_api_key(cfg)
    if not api_key:
        raise ValueError(
            "Ollama web search is enabled, but no API key is configured. "
            "Set DOCMIND_OLLAMA_API_KEY."
        )
    _assert_endpoint_allowed(cfg, "https://ollama.com")
    timeout_s = float(cfg.llm_request_timeout_seconds)
    headers = {"authorization": f"Bearer {api_key}"}
    # Web tools always connect to Ollama Cloud, not the local Ollama instance
    web_client = _cached_client(
        "https://ollama.com", timeout_s, tuple(sorted(headers.items()))
    )

    return [web_client.web_search, web_client.web_fetch]


def ollama_chat_structured[TModel: BaseModel](
    *,
    model: str,
    messages: Sequence[Mapping[str, Any] | Message],
    output_model: type[TModel],
    think: bool | Literal["low", "medium", "high"] | None = None,
    options: Mapping[str, Any] | Options | None = None,
    keep_alive: float | str | None = None,
    client: ollama.Client | None = None,
    cfg: DocMindSettings = settings,
) -> TModel:
    """Run a non-streaming structured output chat and validate with Pydantic.

    Args:
        model: Model name.
        messages: Chat history.
        output_model: Pydantic model class for validation.
        think: Enable thinking features.
        options: Low-level model options.
        keep_alive: Model TTL in memory.
        client: Optional client override.
        cfg: Application settings.

    Returns:
        Validated Pydantic model instance.
    """
    response = ollama_chat(
        model=model,
        messages=messages,
        stream=False,
        think=think,
        response_format=output_model.model_json_schema(),
        options=options,
        keep_alive=keep_alive,
        client=client,
        cfg=cfg,
    )
    content = getattr(getattr(response, "message", None), "content", None)
    if not content:
        raise ValueError("Structured output response was empty")
    return output_model.model_validate_json(content)


__all__ = [
    "get_ollama_client",
    "get_ollama_web_tools",
    "is_ollama_web_search_enabled",
    "ollama_chat",
    "ollama_chat_structured",
    "ollama_embed",
    "ollama_generate",
]
