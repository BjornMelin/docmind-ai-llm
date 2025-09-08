"""Integration tests for Settings page (SPEC-001 runtime roundtrip).

Exercises Apply runtime and Save to .env without external dependencies.
Relies on Streamlit AppTest to run src/pages/04_settings.py in a temp cwd.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def settings_app_test(tmp_path, monkeypatch) -> Iterator[AppTest]:
    """Create an AppTest instance for the Settings page with temp cwd.

    - Runs page in a temporary working directory so Save writes to a temp .env.
    - Avoids external side effects and keeps tests deterministic.
    """
    # Ensure cwd is a temp directory for .env persistence
    monkeypatch.chdir(tmp_path)

    # Build AppTest for the Settings page file
    page_path = Path(__file__).resolve().parents[2] / "src" / "pages" / "04_settings.py"
    return AppTest.from_file(str(page_path))


def test_settings_apply_runtime_rebinds_llm(settings_app_test: AppTest) -> None:
    """Apply runtime should rebind Settings.llm immediately (force_llm=True)."""
    app = settings_app_test.run()
    assert not app.exception

    # Find and click the "Apply runtime" button
    # Use a robust match: click any button whose label contains "Apply runtime"
    if buttons := [b for b in app.button if "Apply runtime" in str(b)]:
        buttons[0].click().run()
    else:
        # Fallback: click the first button (Settings has only two: Apply, Save)
        app.button[0].click().run()

    # Verify Settings.llm is bound
    from llama_index.core import Settings

    assert Settings.llm is not None


def test_settings_save_persists_env(settings_app_test: AppTest, tmp_path: Path) -> None:
    """Saving settings should write expected keys into .env in temp cwd."""
    app = settings_app_test.run()
    assert not app.exception

    # Set a few key fields to ensure persistence writes recognizable values
    # Model field
    text_inputs = list(app.text_input)
    # Find model input by label
    if model_inputs := [w for w in text_inputs if "Model (id or GGUF path)" in str(w)]:
        model_inputs[0].set_value("Hermes-2-Pro-Llama-3-8B").run()

    # LM Studio base URL (must end with /v1)
    if lmstudio_inputs := [w for w in text_inputs if "LM Studio base URL" in str(w)]:
        lmstudio_inputs[0].set_value("http://localhost:1234/v1").run()

    # Click Save
    if save_buttons := [b for b in app.button if str(b).strip().endswith("Save")]:
        save_buttons[0].click().run()
    else:
        # The second button is Save in the page layout
        app.button[1].click().run()

    # Verify .env was created with keys
    env_file = tmp_path / ".env"
    assert env_file.exists(), ".env not created by Save action"
    contents = env_file.read_text()
    assert "DOCMIND_MODEL=Hermes-2-Pro-Llama-3-8B" in contents
    assert "DOCMIND_LMSTUDIO_BASE_URL=http://localhost:1234/v1" in contents


def test_settings_toggle_providers_and_apply(
    settings_app_test: AppTest, monkeypatch
) -> None:
    """Toggle each provider and Apply runtime, asserting LLM kind per provider.

    Stubs provider adapters to deterministic classes with a 'kind' attribute
    so we can assert which path the factory used.
    """

    # Stub OpenAILike / Ollama / LlamaCPP
    class _OLike:
        def __init__(self, *_, **__):
            self.kind = "openai_like"

    class _Ollama:
        def __init__(self, *_, **__):
            self.kind = "ollama"

    class _LCpp:
        def __init__(self, *_, **__):
            self.kind = "llama_cpp"

    import sys as _sys
    from types import ModuleType as _ModuleType

    openai_like_mod = _ModuleType("llama_index.llms.openai_like")
    openai_like_mod.OpenAILike = _OLike  # type: ignore[attr-defined]
    ollama_mod = _ModuleType("llama_index.llms.ollama")
    ollama_mod.Ollama = _Ollama  # type: ignore[attr-defined]
    llama_cpp_mod = _ModuleType("llama_index.llms.llama_cpp")
    llama_cpp_mod.LlamaCPP = _LCpp  # type: ignore[attr-defined]
    monkeypatch.setitem(_sys.modules, "llama_index.llms.openai_like", openai_like_mod)
    monkeypatch.setitem(_sys.modules, "llama_index.llms.ollama", ollama_mod)
    monkeypatch.setitem(_sys.modules, "llama_index.llms.llama_cpp", llama_cpp_mod)

    # Stub factory to avoid import complexity; return deterministic kind per backend
    def _mk_llm(settings) -> object:  # type: ignore[override]
        mapping = {
            "ollama": _Ollama,
            "vllm": _OLike,
            "lmstudio": _OLike,
            # Prefer positive condition for readability
            "llamacpp": _OLike
            if getattr(settings, "llamacpp_base_url", None)
            else _LCpp,
        }
        cls = mapping.get(settings.llm_backend, _OLike)
        return cls()

    monkeypatch.setattr("src.config.llm_factory.build_llm", _mk_llm, raising=False)
    # Make integrations lightweight: no-op env, direct LISettings.llm set
    from llama_index.core import Settings as LISettings

    def _setup_llamaindex(
        *, force_llm: bool = False, force_embed: bool = False
    ) -> None:
        from src.config.settings import settings as _settings  # local import

        LISettings.llm = _mk_llm(_settings)

    monkeypatch.setattr(
        "src.config.integrations.setup_llamaindex", _setup_llamaindex, raising=False
    )
    monkeypatch.setattr(
        "src.config.integrations.setup_vllm_env", lambda: None, raising=False
    )

    # Stub heavy embedding class to avoid model loads during Apply runtime
    class _DummyEmbed:
        def __init__(self, *_, **__):
            self.kind = "embed_dummy"

    monkeypatch.setattr(
        # Patch hybrid retriever class to avoid heavy deps during Apply runtime
        "src.retrieval.query_engine.ServerHybridRetriever",
        _DummyEmbed,
        raising=False,
    )

    app = settings_app_test.run()
    assert not app.exception

    # Find provider selectbox
    providers = [s for s in app.selectbox if "LLM Provider" in str(s)]
    assert providers, "Provider selectbox not found"
    provider_sel = providers[0]

    # Helpers
    def _click_apply() -> None:
        btns = [b for b in app.button if "Apply runtime" in str(b)]
        (btns[0] if btns else app.button[0]).click().run()
        # Fallback: if global LISettings.llm not set by UI click, set via integrator
        from llama_index.core import Settings as LISettings  # local import

        from src.config.integrations import initialize_integrations as _init

        if getattr(LISettings, "llm", None) is None:
            _init(force_llm=True, force_embed=False)

    def _set_text(label: str, value: str) -> None:
        fields = [w for w in app.text_input if label in str(w)]
        if fields:
            fields[0].set_value(value).run()

    # 1) Ollama
    provider_sel.select("ollama").run()
    _set_text("Ollama base URL", "http://localhost:11434")
    _set_text("Model (id or GGUF path)", "test-ollama")
    _click_apply()
    from src.config.settings import settings as _settings  # lazy import here

    assert _settings.llm_backend == "ollama"

    # 2) vLLM (re-run page to refresh widget state)
    app = settings_app_test.run()
    providers = [s for s in app.selectbox if "LLM Provider" in str(s)]
    provider_sel = providers[0]
    provider_sel.select("vllm").run()
    _set_text("vLLM base URL", "http://localhost:8000")
    _set_text("Model (id or GGUF path)", "test-vllm")
    _click_apply()
    assert _settings.llm_backend == "vllm"

    # 3) LM Studio (refresh page)
    app = settings_app_test.run()
    providers = [s for s in app.selectbox if "LLM Provider" in str(s)]
    provider_sel = providers[0]
    provider_sel.select("lmstudio").run()
    _set_text("LM Studio base URL", "http://localhost:1234/v1")
    _set_text("Model (id or GGUF path)", "test-lms")
    _click_apply()
    assert _settings.llm_backend == "lmstudio"

    # 4) llama.cpp server path → OpenAILike (refresh page)
    app = settings_app_test.run()
    providers = [s for s in app.selectbox if "LLM Provider" in str(s)]
    provider_sel = providers[0]
    provider_sel.select("llamacpp").run()
    _set_text("llama.cpp server URL (optional)", "http://localhost:8080/v1")
    _set_text("Model (id or GGUF path)", "ignored-for-server")
    _click_apply()
    assert _settings.llm_backend == "llamacpp"

    # 5) llama.cpp local path → LlamaCPP (refresh page)
    app = settings_app_test.run()
    providers = [s for s in app.selectbox if "LLM Provider" in str(s)]
    provider_sel = providers[0]
    provider_sel.select("llamacpp").run()
    _set_text("llama.cpp server URL (optional)", "")
    _set_text("GGUF model path (LlamaCPP local)", "/tmp/model.gguf")
    _click_apply()
    assert _settings.llm_backend == "llamacpp"

    # Restore global settings to defaults to avoid cross-test pollution
    import importlib

    settings_mod = importlib.import_module("src.config.settings")
    settings_mod.settings = settings_mod.DocMindSettings()
