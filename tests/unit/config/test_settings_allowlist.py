"""Tests for DocMindSettings endpoint allowlist validation."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from src.config.settings import DocMindSettings


def _mk(**overrides):
    # Helper to construct settings with strict defaults for tests
    base = {
        "ollama_base_url": "http://localhost:11434",
        "lmstudio_base_url": "http://localhost:1234/v1",
        "vllm_base_url": "http://localhost:8000/v1",
        "llamacpp_base_url": None,
        "security": {
            "allow_remote_endpoints": False,
            "endpoint_allowlist": [
                "http://localhost",
                "http://127.0.0.1",
                "https://localhost",
                "https://127.0.0.1",
            ],
        },
    }
    return DocMindSettings(**(base | overrides))


def test_allow_local_loopback_hosts_ok():
    """Loopback hostnames should be allowed (IPv4 and IPv6)."""
    s = _mk(
        ollama_base_url="http://127.0.0.1:11434",
        lmstudio_base_url="https://localhost:8443/v1",
        llamacpp_base_url="http://[::1]:8080/v1",
    )
    # Should not raise
    s._validate_endpoints_security()


def test_disallow_spoofed_localhost_prefix():
    """Spoofed localhost prefix must be rejected."""
    with pytest.raises(Exception, match="Remote endpoints are disabled"):
        _ = _mk(vllm_base_url="https://localhost.attacker.tld/v1")


def test_allow_explicit_allowlisted_host(monkeypatch):  # type: ignore[no-untyped-def]
    """Exact host in allowlist should be accepted."""
    import socket

    def _fake_getaddrinfo(host, *args, **kwargs):  # type: ignore[no-untyped-def]
        assert str(host) == "api.example.com"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    s = _mk(
        security={
            "allow_remote_endpoints": False,
            "endpoint_allowlist": [
                "https://api.example.com",
                "http://localhost",
                "http://127.0.0.1",
                "https://localhost",
                "https://127.0.0.1",
            ],
        },
        vllm_base_url="https://api.example.com",
    )
    # Should not raise
    s._validate_endpoints_security()


def test_disallow_similar_but_unlisted_host():
    """Similar domains not in allowlist must be rejected."""
    with pytest.raises(Exception, match="Remote endpoints are disabled"):
        _ = _mk(
            security={
                "allow_remote_endpoints": False,
                "endpoint_allowlist": ["https://api.example.com"],
            },
            vllm_base_url="https://api.example.com.evil",
        )


def test_disallow_remote_qdrant_before_client_construction() -> None:
    """Qdrant obeys the same local-first endpoint policy as model services."""
    with pytest.raises(ValidationError, match="localhost Qdrant URL"):
        _mk(database={"qdrant_url": "https://qdrant.example.com"})


@pytest.mark.parametrize(
    "name",
    ["", "   ", ".", "..", "nested/collection", r"nested\collection"],
)
def test_reject_unsafe_qdrant_collection_names(name: str) -> None:
    with pytest.raises(ValidationError, match="safe non-empty names"):
        _mk(database={"qdrant_collection": name})


def test_malformed_url_rejected():
    """Malformed URLs should be rejected."""
    with pytest.raises(ValidationError):
        _ = _mk(vllm_base_url="not a url")


def test_remote_endpoints_allowed_when_policy_true():
    """When allow_remote_endpoints is true, remote hosts should pass validation."""
    s = _mk(
        security={
            "allow_remote_endpoints": True,
            "endpoint_allowlist": [
                "https://api.example.com",
            ],
        },
        vllm_base_url="https://unlisted.remote.example",
    )
    s._validate_endpoints_security()


def test_web_search_accepts_least_privilege_ollama_allowlist(monkeypatch) -> None:
    """Ollama Cloud can be enabled without opening every remote endpoint."""
    import socket

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.32.47", 0))
        ],
    )

    config = _mk(
        ollama_enable_web_search=True,
        ollama_api_key=SecretStr("test-key"),
        security={
            "allow_remote_endpoints": False,
            "endpoint_allowlist": ["https://ollama.com"],
        },
    )

    assert config.ollama_enable_web_search is True
    assert config.security.allow_remote_endpoints is False


def test_web_search_rejects_missing_ollama_allowlist() -> None:
    with pytest.raises(ValidationError, match=r"https://ollama\.com"):
        _mk(
            ollama_enable_web_search=True,
            ollama_api_key=SecretStr("test-key"),
        )
