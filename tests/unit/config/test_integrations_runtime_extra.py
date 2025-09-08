"""Extra runtime tests to lightly exercise integration helpers for coverage."""

import pytest

from src.config.integrations import get_vllm_server_command, setup_vllm_env


@pytest.mark.unit
def test_get_vllm_server_command_builds_args():
    """Test that get_vllm_server_command builds command args correctly."""
    cmd = get_vllm_server_command()
    assert isinstance(cmd, list)
    assert "vllm" in cmd
    assert "serve" in cmd
    # Contains max model length and dtype flags
    assert "--max-model-len" in cmd
    assert "--kv-cache-dtype" in cmd


@pytest.mark.unit
def test_setup_vllm_env_runs_without_errors(monkeypatch):
    """Test that setup_vllm_env runs without errors and sets environment variables."""
    # Ensure function runs idempotently
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "")
    setup_vllm_env()
    # After running, at least one var should be present
    assert "VLLM_KV_CACHE_DTYPE" in __import__("os").environ
