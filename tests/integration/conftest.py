"""Integration-tier pytest fixtures.

Forces LlamaIndex Settings.llm to MockLLM within the integration session to
avoid environment-dependent LLM selection (e.g., Ollama) and any network calls.
This isolation does not affect unit tests which use their own session fixture.
"""

from __future__ import annotations

import pytest
from llama_index.core import Settings
from llama_index.core.llms import MockLLM


@pytest.fixture(scope="session", autouse=True)
def integration_llm_guard() -> None:
    """Force a deterministic, offline LLM for integration tests only."""
    original_llm = Settings.llm
    try:
        Settings.llm = MockLLM(max_tokens=256)
        yield
    finally:
        Settings.llm = original_llm

