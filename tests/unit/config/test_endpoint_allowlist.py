"""Tests for endpoint allowlist and egress validation.

Validates that non-local endpoints are rejected when remote endpoints
are disabled, and localhost endpoints are accepted.
"""

from __future__ import annotations

import pytest

from src.config.settings import DocMindSettings


def test_endpoint_allowlist_blocks_remote_urls() -> None:
    """When allow_remote_endpoints is False, non-local URLs must raise ValueError."""
    cfg = DocMindSettings()
    cfg.allow_remote_endpoints = False
    cfg.vllm_base_url = "https://api.example.com/v1"
    with pytest.raises(ValueError, match=r".*"):
        cfg._validate_endpoints_security()  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000",
        "http://127.0.0.1:9000",
    ],
)
def test_endpoint_allowlist_allows_localhost(url: str) -> None:
    """Localhost endpoints are allowed when remote endpoints are disabled."""
    cfg = DocMindSettings()
    cfg.allow_remote_endpoints = False
    cfg.vllm_base_url = url
    # Should not raise
    cfg._validate_endpoints_security()  # pylint: disable=protected-access
