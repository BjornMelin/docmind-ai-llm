"""Unit tests for LangChain chat model factory (LangGraph runtime)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config.langchain_factory import build_chat_model
from src.config.settings import DocMindSettings

pytestmark = pytest.mark.unit


def test_build_chat_model_openai_compatible_responses_sets_flags() -> None:
    """Responses mode enables use_responses_api and responses/v1 output version."""
    cfg = DocMindSettings()
    cfg.llm_backend = "openai_compatible"
    cfg.openai.base_url = "http://localhost:8000"
    cfg.openai.api_key = "abc"
    cfg.openai.api_mode = "responses"
    cfg.openai.default_headers = {"X-Test": "1"}
    cfg.vllm.model = "model-1"

    with patch("src.config.langchain_factory.ChatOpenAI", autospec=True) as p:
        inst = MagicMock(name="ChatOpenAIInstance")
        p.return_value = inst
        out = build_chat_model(cfg)
        assert out is inst
        _, kwargs = p.call_args
        assert str(kwargs["base_url"]).endswith("/v1")
        assert kwargs["model"] == "model-1"
        assert kwargs["default_headers"] == {"X-Test": "1"}
        assert kwargs["use_responses_api"] is True
        assert kwargs["output_version"] == "responses/v1"


def test_build_chat_model_openai_compatible_chat_completions_is_default() -> None:
    """Chat-completions mode keeps responses flags disabled."""
    cfg = DocMindSettings()
    cfg.llm_backend = "openai_compatible"
    cfg.openai.base_url = "http://localhost:8000"
    cfg.openai.api_key = "abc"
    cfg.openai.api_mode = "chat_completions"
    cfg.vllm.model = "model-1"

    with patch("src.config.langchain_factory.ChatOpenAI", autospec=True) as p:
        inst = MagicMock(name="ChatOpenAIInstance")
        p.return_value = inst
        out = build_chat_model(cfg)
        assert out is inst
        _, kwargs = p.call_args
        assert kwargs["use_responses_api"] is False
        assert kwargs["output_version"] is None
