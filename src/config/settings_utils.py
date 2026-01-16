"""Settings helper utilities.

This module contains small, dependency-light helpers used by `src.config.settings`.
It is intentionally free of any dependency on the `DocMindSettings` model to avoid
import cycles.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import cast
from urllib.parse import urlparse

from loguru import logger
from pydantic import AnyHttpUrl, TypeAdapter

_ANY_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)


def parse_any_http_url(value: str) -> AnyHttpUrl:
    """Parse a string as an HTTP/HTTPS URL.

    Args:
        value: URL string to validate and parse.

    Returns:
        A validated `AnyHttpUrl` instance.

    Raises:
        ValueError: If the URL cannot be parsed as an HTTP/HTTPS URL.
    """
    return cast(AnyHttpUrl, _ANY_HTTP_URL_ADAPTER.validate_python(value))


DEFAULT_OLLAMA_BASE_URL = parse_any_http_url("http://localhost:11434")
DEFAULT_LMSTUDIO_BASE_URL = parse_any_http_url("http://localhost:1234/v1")
DEFAULT_VLLM_BASE_URL = parse_any_http_url("http://localhost:8000")
DEFAULT_OPENAI_BASE_URL = DEFAULT_LMSTUDIO_BASE_URL


def ensure_v1(url: object | None) -> str | None:
    """Normalize OpenAI-compatible base URLs to include a single `/v1` segment.

    Args:
        url: URL-like input. `None` and empty strings return `None`.

    Returns:
        Normalized URL string ending in `/v1`, or `None` when the input is empty.
    """
    if url is None:
        return None
    raw = str(url).strip()
    if not raw:
        return None
    try:
        parsed = urlparse(raw.rstrip("/"))
        path = parsed.path or ""
        if not path.endswith("/v1"):
            path = f"{path}/v1"
        return parsed._replace(path=path).geturl()
    except (ValueError, AttributeError, TypeError):
        return raw


def ensure_http_scheme(url: object | None) -> str | None:
    """Ensure a URL has an explicit scheme (defaults to `http://`).

    This is applied in `mode="before"` validators so callers can provide convenient
    `host:port` inputs in env vars/UI without bypassing typed parsing.

    Args:
        url: URL-like input.

    Returns:
        URL string with a scheme, or `None` for empty inputs.
    """
    if url is None:
        return None
    raw = str(url).strip()
    if not raw:
        return None
    if "://" not in raw:
        return f"http://{raw}"
    return raw


def normalize_endpoint_host(host: str) -> str:
    """Canonicalize an endpoint host for comparison.

    Args:
        host: Hostname or IP literal.

    Returns:
        Lowercased host without a trailing dot. IPv6 bracket-literals (`[::1]`)
        are unwrapped to `::1`.
    """
    value = (host or "").strip().lower().rstrip(".")
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value


def is_loopback_endpoint_host(host: str) -> bool:
    """Return True if `host` is a loopback hostname/IP (IPv4 or IPv6).

    Args:
        host: Hostname or IP literal.

    Returns:
        True for `localhost`, `127.0.0.1`, `::1`, or any loopback IP.
    """
    normalized = normalize_endpoint_host(host)
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        parsed_ip = ipaddress.ip_address(normalized.split("%", 1)[0])
    except ValueError:
        return False
    return bool(parsed_ip.is_loopback)


def parse_endpoint_allowlist_hosts(allowlist: list[str]) -> set[str]:
    """Extract canonical hosts from a URL/host allowlist.

    Allowlist entries may be:
    - Full URLs (`https://api.example.com`)
    - Plain hosts (`api.example.com`) or IPs (`127.0.0.1`)

    Args:
        allowlist: Allowlist entries.

    Returns:
        Set of normalized host strings.
    """
    allowed: set[str] = set()
    for entry in allowlist:
        e = (entry or "").strip()
        if not e:
            continue
        if "://" in e:
            try:
                parsed = _ANY_HTTP_URL_ADAPTER.validate_python(e)
            except Exception:
                parsed = None
            if parsed is not None and (host := getattr(parsed, "host", None)):
                allowed.add(normalize_endpoint_host(str(host)))
                continue
        allowed.add(normalize_endpoint_host(e.split("/")[0].split(":")[0]))
    return {h for h in allowed if h}


def resolve_endpoint_host_ips(
    host: str,
) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve a host to concrete IPs for SSRF/DNS-rebinding defense.

    Note: This performs DNS resolution via `socket.getaddrinfo`. In environments
    where DNS resolution is considered network egress, treat allowlisting as an
    explicit opt-in to this behavior.

    Args:
        host: Hostname or IP literal.

    Returns:
        Set of resolved IP addresses. Empty set indicates the host could not be
        resolved.
    """
    normalized = normalize_endpoint_host(host)
    ips: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    try:
        ip = ipaddress.ip_address(normalized.split("%", 1)[0])
    except ValueError:
        ip = None
    if ip is not None:
        ips.add(ip)
        return ips
    try:
        for family, _type, _proto, _canon, sockaddr in socket.getaddrinfo(
            normalized,
            None,
            type=socket.SOCK_STREAM,
        ):
            if family not in {socket.AF_INET, socket.AF_INET6}:
                continue
            ip_s = str(sockaddr[0])
            try:
                ips.add(ipaddress.ip_address(ip_s.split("%", 1)[0]))
            except ValueError:
                continue
    except OSError as exc:
        logger.debug(
            "endpoint DNS resolution failed: host={host} err={err}",
            host=normalized,
            err=exc,
        )
    return ips


def ip_is_blocked_for_endpoints(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Return True if an IP must be blocked for outbound endpoint connections.

    Args:
        ip: Candidate IP address.

    Returns:
        True when `ip` is in a sensitive range that commonly enables SSRF impact:
        private RFC1918, link-local, multicast, reserved, or unspecified ranges.
        Loopback is handled separately.
    """
    return bool(
        ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def endpoint_url_allowed(url: object | None, *, allowed_hosts: set[str]) -> bool:
    """Return True if a parsed endpoint URL is allowed by host + resolved IP policy.

    This is a defense-in-depth policy for SSRF prevention:
    - Accept loopback hosts unconditionally.
    - Require exact hostname/IP match against an allowlist.
    - Resolve the allowlisted host and fail closed if it cannot be resolved.
    - Reject if any resolved IP is in blocked ranges (private/link-local/etc).

    Args:
        url: URL-like input to validate.
        allowed_hosts: Canonical allowlisted host set.

    Returns:
        True if the URL is allowed, otherwise False.
    """
    allowed = True
    if url is not None:
        raw = str(url).strip()
        if raw:
            allowed = False
            try:
                parsed = _ANY_HTTP_URL_ADAPTER.validate_python(raw)
            except Exception:
                parsed = None
            if parsed is not None:
                host_val = getattr(parsed, "host", None)
                if host_val:
                    host = normalize_endpoint_host(str(host_val))
                    if is_loopback_endpoint_host(host):
                        allowed = True
                    elif host in allowed_hosts:
                        resolved = resolve_endpoint_host_ips(host)
                        if resolved:
                            allowed = True
                            for ip in resolved:
                                if ip.is_loopback:
                                    continue
                                if ip_is_blocked_for_endpoints(ip):
                                    allowed = False
                                    break
    return allowed


__all__ = [
    "DEFAULT_LMSTUDIO_BASE_URL",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_VLLM_BASE_URL",
    "endpoint_url_allowed",
    "ensure_http_scheme",
    "ensure_v1",
    "ip_is_blocked_for_endpoints",
    "is_loopback_endpoint_host",
    "normalize_endpoint_host",
    "parse_any_http_url",
    "parse_endpoint_allowlist_hosts",
    "resolve_endpoint_host_ips",
]
