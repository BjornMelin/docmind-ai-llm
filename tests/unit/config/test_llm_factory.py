"""Tests for LLM factory configuration and building functionality."""

from unittest.mock import MagicMock, patch

import pytest

from src.config.llm_factory import build_llm
from src.config.settings import DocMindSettings


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["ollama", "vllm", "lmstudio"])
def test_build_llm_openai_like_and_ollama(backend):
    """Test LLM building for OpenAI-like backends (vllm, lmstudio) and Ollama."""
    cfg = DocMindSettings()
    cfg.llm_backend = backend
    cfg.vllm.model = "qwen2.5-7b-instruct"
    cfg.vllm.context_window = 8192
    cfg.llm_request_timeout_seconds = 123
    cfg.openai_like_api_key = "abc"
    cfg.openai_like_is_chat_model = True
    cfg.openai_like_is_function_calling_model = False

    if backend == "ollama":
        with patch("llama_index.llms.ollama.Ollama", autospec=True) as p:
            inst = MagicMock(name="OllamaInstance")
            p.return_value = inst
            out = build_llm(cfg)
            assert out is inst
            _, kwargs = p.call_args
            assert kwargs["base_url"] == cfg.ollama_base_url
            assert kwargs["model"] == cfg.vllm.model
            assert float(kwargs["request_timeout"]) == float(
                cfg.llm_request_timeout_seconds
            )

    elif backend == "vllm":
        with patch("llama_index.llms.openai_like.OpenAILike", autospec=True) as p:
            inst = MagicMock(name="OpenAILikeVLLM")
            p.return_value = inst
            out = build_llm(cfg)
            assert out is inst
            _, kwargs = p.call_args
            assert kwargs["api_base"] == cfg.vllm.vllm_base_url  # no /v1 by default
            assert kwargs["model"] == cfg.vllm.model
            assert kwargs["api_key"] == cfg.openai_like_api_key
            assert kwargs["is_chat_model"] is True
            assert kwargs["is_function_calling_model"] is False
            assert kwargs["context_window"] == cfg.vllm.context_window
            assert float(kwargs["timeout"]) == float(cfg.llm_request_timeout_seconds)

    elif backend == "lmstudio":
        with patch("llama_index.llms.openai_like.OpenAILike", autospec=True) as p:
            inst = MagicMock(name="OpenAILikeLMStudio")
            p.return_value = inst
            out = build_llm(cfg)
            assert out is inst
            _, kwargs = p.call_args
            assert kwargs["api_base"] == cfg.lmstudio_base_url
            assert kwargs["is_chat_model"] is True
            assert kwargs["is_function_calling_model"] is False
            assert kwargs["context_window"] == cfg.vllm.context_window


@pytest.mark.unit
@pytest.mark.parametrize("gpu", [True, False])
def test_build_llm_llamacpp_offload(gpu):
    """Test LlamaCPP LLM building with GPU acceleration configuration."""
    cfg = DocMindSettings()
    cfg.llm_backend = "llamacpp"
    cfg.enable_gpu_acceleration = gpu

    with patch("llama_index.llms.llama_cpp.LlamaCPP", autospec=True) as p:
        inst = MagicMock(name="LlamaCPP")
        p.return_value = inst
        out = build_llm(cfg)
        assert out is inst
        _, kwargs = p.call_args
        assert kwargs["model_path"] == str(cfg.vllm.llamacpp_model_path)
        assert kwargs["context_window"] == cfg.vllm.context_window
        assert kwargs["model_kwargs"]["n_gpu_layers"] == (-1 if gpu else 0)
