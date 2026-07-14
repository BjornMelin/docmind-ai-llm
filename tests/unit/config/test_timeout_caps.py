"""Authoritative timeout-cap tests (SPEC-040)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config.langchain_factory import build_chat_model
from src.config.llm_factory import build_llm
from src.config.settings import DocMindSettings

pytestmark = pytest.mark.unit


def test_build_chat_model_timeout_always_capped_by_decision_timeout() -> None:
    """Always cap ChatOpenAI timeout using the decision timeout.

    Args:
        None.

    Returns:
        None.
    """
    cfg = DocMindSettings.model_validate(
        {
            "llm_backend": "openai_compatible",
            "openai": {"base_url": "http://localhost:8000", "api_key": "abc"},
            "vllm": {"model": "model-1"},
            "llm_request_timeout_seconds": 120,
            "agents": {"decision_timeout": 10},
        }
    )

    with patch("src.config.langchain_factory.ChatOpenAI", autospec=True) as p:
        inst = MagicMock(name="ChatOpenAIInstance")
        p.return_value = inst
        out = build_chat_model(cfg)
        assert out is inst
        _, kwargs = p.call_args
        assert float(kwargs["timeout"]) == 10.0
        assert kwargs["max_retries"] == cfg.agents.max_retries


def test_build_chat_model_timeout_capped_by_coordinator_override() -> None:
    """Cap ChatOpenAI timeout using the coordinator-owned deadline."""
    cfg = DocMindSettings.model_validate(
        {
            "llm_backend": "openai_compatible",
            "openai": {"base_url": "http://localhost:8000", "api_key": "abc"},
            "vllm": {"model": "model-1"},
            "llm_request_timeout_seconds": 120,
        }
    )

    with patch("src.config.langchain_factory.ChatOpenAI", autospec=True) as p:
        inst = MagicMock(name="ChatOpenAIInstance")
        p.return_value = inst

        out = build_chat_model(cfg, timeout_cap=3.5)

        assert out is inst
        _, kwargs = p.call_args
        assert float(kwargs["timeout"]) == 3.5
        assert kwargs["max_retries"] == 0


def test_build_llm_timeout_always_capped_by_decision_timeout() -> None:
    """Always cap LLM timeout using the decision timeout.

    Args:
        None.

    Returns:
        None.
    """
    cfg = DocMindSettings.model_validate(
        {
            "llm_backend": "vllm",
            "vllm_base_url": "http://localhost:8000",
            "vllm": {"model": "model-1"},
            "llm_request_timeout_seconds": 120,
            "openai": {"api_key": "abc"},
            "agents": {"decision_timeout": 10},
        }
    )

    with patch("llama_index.llms.openai_like.OpenAILike", autospec=True) as p:
        inst = MagicMock(name="OpenAILikeInstance")
        p.return_value = inst
        out = build_llm(cfg)
        assert out is inst
        _, kwargs = p.call_args
        assert float(kwargs["timeout"]) == 10.0
        assert kwargs["max_retries"] == cfg.agents.max_retries
