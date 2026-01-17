"""Timeout cap tests when deadline propagation is enabled (SPEC-040)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config.langchain_factory import build_chat_model
from src.config.llm_factory import build_llm
from src.config.settings import DocMindSettings

pytestmark = pytest.mark.unit


def test_build_chat_model_timeout_capped_by_decision_timeout() -> None:
    cfg = DocMindSettings()
    cfg.llm_backend = "openai_compatible"
    cfg.openai.base_url = "http://localhost:8000"
    cfg.openai.api_key = "abc"
    cfg.vllm.model = "model-1"
    cfg.llm_request_timeout_seconds = 120
    cfg.agents.decision_timeout = 10
    cfg.agents.enable_deadline_propagation = True

    with patch("src.config.langchain_factory.ChatOpenAI", autospec=True) as p:
        inst = MagicMock(name="ChatOpenAIInstance")
        p.return_value = inst
        out = build_chat_model(cfg)
        assert out is inst
        _, kwargs = p.call_args
        assert float(kwargs["timeout"]) == 10.0


def test_build_llm_timeout_capped_by_decision_timeout() -> None:
    cfg = DocMindSettings()
    cfg.llm_backend = "vllm"
    cfg.vllm_base_url = "http://localhost:8000"
    cfg.vllm.model = "model-1"
    cfg.llm_request_timeout_seconds = 120
    cfg.openai.api_key = "abc"
    cfg.agents.decision_timeout = 10
    cfg.agents.enable_deadline_propagation = True

    with patch("llama_index.llms.openai_like.OpenAILike", autospec=True) as p:
        inst = MagicMock(name="OpenAILikeInstance")
        p.return_value = inst
        out = build_llm(cfg)
        assert out is inst
        _, kwargs = p.call_args
        assert float(kwargs["timeout"]) == 10.0
