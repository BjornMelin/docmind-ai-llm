"""Unit tests for LLM factory backends (SPEC-001)."""

from src.config.llm_factory import build_llm
from src.config.settings import DocMindSettings


def test_build_llm_vllm_openai_like() -> None:
    """OpenAILike returned for vLLM backend."""
    cfg = DocMindSettings(
        llm_backend="vllm",
        model="Qwen2.5-7B-Instruct",
        vllm_base_url="http://localhost:8000",
        context_window=8192,
        llm_request_timeout_seconds=30,
    )
    llm = build_llm(cfg)
    # Lazy import here to avoid heavy deps in import path for unrelated tests
    from llama_index.llms.openai_like import OpenAILike  # type: ignore

    assert isinstance(llm, OpenAILike)


def test_build_llm_lmstudio_openai_like() -> None:
    """OpenAILike returned for LM Studio backend."""
    cfg = DocMindSettings(
        llm_backend="lmstudio",
        model="Hermes-2-Pro-Llama-3-8B",
        lmstudio_base_url="http://localhost:1234/v1",
        context_window=4096,
    )
    llm = build_llm(cfg)
    from llama_index.llms.openai_like import OpenAILike  # type: ignore

    assert isinstance(llm, OpenAILike)


def test_build_llm_ollama() -> None:
    """Ollama returned for ollama backend."""
    cfg = DocMindSettings(
        llm_backend="ollama",
        model="llama3.1:latest",
        ollama_base_url="http://localhost:11434",
        context_window=4096,
        llm_request_timeout_seconds=10,
    )
    llm = build_llm(cfg)
    from llama_index.llms.ollama import Ollama  # type: ignore

    assert isinstance(llm, Ollama)


def test_build_llm_llamacpp_server_openai_like() -> None:
    """OpenAILike used when llamacpp server base_url provided."""
    cfg = DocMindSettings(
        llm_backend="llamacpp",
        model="qwen3",
        llamacpp_base_url="http://localhost:8080/v1",
        context_window=2048,
    )
    llm = build_llm(cfg)
    from llama_index.llms.openai_like import OpenAILike  # type: ignore

    assert isinstance(llm, OpenAILike)
