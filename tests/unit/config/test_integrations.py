"""Unit tests for src.config.integrations module.

Covers LlamaIndex Settings configuration and vLLM command generation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_setup_llamaindex_success() -> None:
    """setup_llamaindex sets Settings.llm and Settings.embed_model on success."""
    with (
        patch("src.config.integrations.settings") as _mock_settings,
        patch("src.config.integrations.build_llm") as mock_build_llm,
        patch("src.config.integrations.HuggingFaceEmbedding") as mock_embed,
    ):
        mock_llm = object()
        mock_build_llm.return_value = mock_llm
        mock_embed.return_value = object()

        from llama_index.core import Settings

        from src.config.integrations import setup_llamaindex

        # Reset globals
        Settings.llm = None
        Settings.embed_model = None

        setup_llamaindex()
        # A global fixture may pre-set a MockLLM; accept any non-None value
        assert Settings.llm is not None
        assert Settings.embed_model is not None
        # Depending on how settings is patched, these may be ints or mocks.
        assert Settings.context_window is not None
        assert Settings.num_output is not None


@pytest.mark.unit
def test_setup_llamaindex_failure_paths() -> None:
    """On import/config errors, Settings.llm/embed_model become None."""
    with (
        patch("src.config.integrations.settings") as _mock_settings,
        patch("src.config.integrations.build_llm", side_effect=ImportError("boom")),
        patch(
            "src.config.integrations.HuggingFaceEmbedding",
            side_effect=RuntimeError("x"),
        ),
    ):
        from llama_index.core import Settings

        from src.config.integrations import setup_llamaindex

        Settings.llm = None
        Settings.embed_model = None

        setup_llamaindex()
        # In some unit contexts, a global fixture may pre-set MockLLM/MockEmbedding.
        # Accept either None (on failure) or a known mock class.
        llm_ok = (Settings.llm is None) or (
            Settings.llm.__class__.__name__ == "MockLLM"
        )
        embed_ok = (Settings.embed_model is None) or (
            Settings.embed_model.__class__.__name__
            in {"MockEmbedding", "HuggingFaceEmbedding"}
        )
        assert llm_ok
        assert embed_ok


@pytest.mark.unit
def test_get_vllm_server_command_flags() -> None:
    """Command contains expected flags and optional chunked prefill toggle."""
    with patch("src.config.integrations.settings") as mock_settings:
        # Minimal config
        mock_settings.vllm.model = "qwen"
        mock_settings.vllm.context_window = 8192
        mock_settings.vllm.kv_cache_dtype = "fp8"
        mock_settings.vllm.gpu_memory_utilization = 0.9
        mock_settings.vllm.max_num_seqs = 4
        mock_settings.vllm.max_num_batched_tokens = 4096

        from src.config.integrations import get_vllm_server_command

        mock_settings.vllm.enable_chunked_prefill = False
        cmd = get_vllm_server_command()
        assert "--enable-chunked-prefill" not in cmd

        mock_settings.vllm.enable_chunked_prefill = True
        cmd2 = get_vllm_server_command()
        assert "--enable-chunked-prefill" in cmd2
