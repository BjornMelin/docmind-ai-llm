"""LLM runtime endpoint probes.

Small HTTP probes used by the Settings UI to validate OpenAI-compatible
runtime endpoints without constructing an LLM client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class RuntimeProbeResult:
    """Result from an LLM runtime endpoint probe."""

    ok: bool
    message: str
    models_count: int | None = None
    health_status_code: int | None = None
    models_status_code: int | None = None


def _join_url_path(base: httpx.URL, suffix: str) -> httpx.URL:
    """Return `base` with `suffix` appended to its current path."""
    path = base.path.rstrip("/")
    suffix_path = suffix if suffix.startswith("/") else f"/{suffix}"
    return base.copy_with(path=f"{path}{suffix_path}")


def _llamacpp_health_url(base: httpx.URL) -> httpx.URL:
    """Return the llama.cpp health URL for a `/v1` API base URL."""
    path = base.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return base.copy_with(path=f"{path.rstrip('/')}/health")


def _model_count(payload: Any) -> int | None:
    """Extract OpenAI-compatible model count from a response payload."""
    if not isinstance(payload, dict):
        return None
    models = payload.get("data")
    return len(models) if isinstance(models, list) else None


def probe_openai_compatible_runtime(
    *,
    base_url: str,
    backend: str,
    headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
) -> RuntimeProbeResult:
    """Probe an OpenAI-compatible runtime endpoint.

    Args:
        base_url: Normalized OpenAI-compatible base URL, usually ending `/v1`.
        backend: Runtime backend name.
        headers: Optional request headers.
        timeout_s: HTTP timeout in seconds.

    Returns:
        RuntimeProbeResult with status metadata only.
    """
    base = httpx.URL(str(base_url))
    request_headers = dict(headers or {})
    timeout = httpx.Timeout(float(timeout_s))

    with httpx.Client(timeout=timeout) as client:
        if backend == "llamacpp":
            health_url = _llamacpp_health_url(base)
            health = client.get(health_url, headers=request_headers)
            if health.status_code == 503:
                return RuntimeProbeResult(
                    ok=False,
                    message="llama.cpp server is reachable but still loading a model.",
                    health_status_code=health.status_code,
                )
            if health.status_code >= 400:
                return RuntimeProbeResult(
                    ok=False,
                    message=f"llama.cpp health check failed: HTTP {health.status_code}",
                    health_status_code=health.status_code,
                )
            health_status = health.status_code
        else:
            health_status = None

        models_url = _join_url_path(base, "/models")
        models = client.get(models_url, headers=request_headers)
        if models.status_code >= 400:
            return RuntimeProbeResult(
                ok=False,
                message=f"Model list request failed: HTTP {models.status_code}",
                health_status_code=health_status,
                models_status_code=models.status_code,
            )
        try:
            count = _model_count(models.json())
        except ValueError:
            return RuntimeProbeResult(
                ok=True,
                message="Endpoint reachable; model list returned non-JSON content.",
                health_status_code=health_status,
                models_status_code=models.status_code,
            )
    if count is None:
        return RuntimeProbeResult(
            ok=True,
            message="Endpoint reachable.",
            health_status_code=health_status,
            models_status_code=models.status_code,
        )
    return RuntimeProbeResult(
        ok=True,
        message=f"Endpoint reachable. Models returned: {count}",
        models_count=count,
        health_status_code=health_status,
        models_status_code=models.status_code,
    )


__all__ = ["RuntimeProbeResult", "probe_openai_compatible_runtime"]
