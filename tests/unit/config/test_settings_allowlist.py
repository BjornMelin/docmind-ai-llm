"""Tests for DocMindSettings endpoint allowlist validation."""

import pytest

from src.config.settings import DocMindSettings


def _mk(**overrides):
    # Helper to construct settings with strict defaults for tests
    base = {
        "allow_remote_endpoints": False,
        "ollama_base_url": "http://localhost:11434",
        "lmstudio_base_url": "http://localhost:1234/v1",
        "vllm_base_url": None,
        "llamacpp_base_url": None,
        "endpoint_allowlist": [
            "http://localhost",
            "http://127.0.0.1",
            "https://localhost",
            "https://127.0.0.1",
        ],
    }
    base.update(overrides)
    return DocMindSettings(**base)


def test_allow_local_loopback_hosts_ok():
    """Loopback hostnames should be allowed (IPv4 and IPv6)."""
    s = _mk(
        ollama_base_url="http://127.0.0.1:11434",
        lmstudio_base_url="https://localhost:8443/v1",
        llamacpp_base_url="http://[::1]:8080/v1",
    )
    # Should not raise
    s._validate_endpoints_security()  # pylint: disable=protected-access


def test_disallow_spoofed_localhost_prefix():
    """Spoofed localhost prefix must be rejected."""
    with pytest.raises(Exception, match="Remote endpoints are disabled"):
        _ = _mk(
            vllm_base_url="https://localhost.attacker.tld/v1",
        )


def test_allow_explicit_allowlisted_host():
    """Exact host in allowlist should be accepted."""
    s = _mk(
        endpoint_allowlist=[
            "https://api.example.com",
            "http://localhost",
            "http://127.0.0.1",
            "https://localhost",
            "https://127.0.0.1",
        ],
        vllm_base_url="https://api.example.com",
    )
    # Should not raise
    s._validate_endpoints_security()  # pylint: disable=protected-access


def test_disallow_similar_but_unlisted_host():
    """Similar domains not in allowlist must be rejected."""
    with pytest.raises(Exception, match="Remote endpoints are disabled"):
        _ = _mk(
            endpoint_allowlist=["https://api.example.com"],
            vllm_base_url="https://api.example.com.evil",
        )


def test_malformed_url_rejected():
    """Malformed URLs should be rejected."""
    with pytest.raises(Exception, match="Remote endpoints are disabled"):
        _ = _mk(
            vllm_base_url="not a url",
        )
