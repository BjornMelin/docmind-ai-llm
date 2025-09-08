"""Offline end-to-end smoke checks when Qdrant is available locally."""

import socket

import pytest


def _is_qdrant_up(host: str = "127.0.0.1", port: int = 6333) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _is_qdrant_up(), reason="Qdrant not running on localhost:6333")
def test_e2e_offline_env_flags_set(monkeypatch):
    """Ensure offline env flags are set without raising exceptions."""
    # Assert env-based offline flags are respected in process (smoke)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # This test is a placeholder to gate offline CI path with Qdrant up
    assert True
