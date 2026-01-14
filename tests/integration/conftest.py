"""Integration-tier pytest fixtures.

Forces LlamaIndex Settings.llm to MockLLM within the integration session to
avoid environment-dependent LLM selection (e.g., Ollama) and any network calls.
This isolation does not affect unit tests which use their own session fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM


@pytest.fixture(scope="session", autouse=True)
def integration_llm_guard() -> None:
    """Force a deterministic, offline LLM for integration tests only.

    Avoids accessing the Settings.llm property before assignment to prevent
    auto-resolution of provider-specific defaults which may require API keys.
    """
    original_llm = getattr(Settings, "_llm", None)
    original_embed = getattr(Settings, "_embed_model", None)
    try:
        Settings.llm = MockLLM(max_tokens=256)
        # Ensure embeddings default to a deterministic local mock to avoid
        # provider auto-resolution (e.g., OpenAI) in integration tests.
        Settings.embed_model = MockEmbedding(embed_dim=1024)
        yield
    finally:
        Settings.llm = original_llm
        Settings.embed_model = original_embed


@pytest.fixture(scope="session", autouse=True)
def integration_streamlit_apptest_warmup(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Warm Streamlit AppTest runtime to avoid CI-only cold-start timeouts."""

    def _is_truthy_env(key: str) -> bool:
        return os.getenv(key, "").strip().lower() in {"1", "true", "yes", "on"}

    is_ci = _is_truthy_env("GITHUB_ACTIONS") or _is_truthy_env("CI")
    if not is_ci:
        return

    from streamlit.testing.v1 import AppTest

    page_path = Path(__file__).resolve().parents[2] / "src" / "pages" / "04_settings.py"
    workdir = tmp_path_factory.mktemp("streamlit-apptest-warmup")
    prev_cwd = Path.cwd()
    os.chdir(workdir)
    try:
        app = AppTest.from_file(str(page_path), default_timeout=120).run()
        assert not app.exception
    finally:
        os.chdir(prev_cwd)
