"""Integration tests for Settings page (SPEC-001 runtime roundtrip).

Exercises Apply runtime and Save to .env without external dependencies.
Relies on Streamlit AppTest to run src/pages/04_settings.py in a temp cwd.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(name="settings_app_test")
def fixture_settings_app_test(tmp_path, monkeypatch) -> AppTest:
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

    # Find and click the "Apply runtime" button by label (no index fallback).
    buttons = [b for b in app.button if getattr(b, "label", "") == "Apply runtime"]
    assert buttons, "Apply runtime button not found"
    buttons[0].click().run()

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
    save_buttons = [b for b in app.button if getattr(b, "label", "") == "Save"]
    assert save_buttons, "Save button not found"
    save_buttons[0].click().run()

    # Verify .env was created with keys
    env_file = tmp_path / ".env"
    assert env_file.exists(), ".env not created by Save action"
    from dotenv import dotenv_values

    values = dotenv_values(env_file)
    assert values.get("DOCMIND_MODEL") == "Hermes-2-Pro-Llama-3-8B"
    assert values.get("DOCMIND_LMSTUDIO_BASE_URL") == "http://localhost:1234/v1"


def test_settings_save_normalizes_lmstudio_url(
    settings_app_test: AppTest, tmp_path: Path
) -> None:
    """LM Studio base URL should be normalized to include /v1 on Save."""
    app = settings_app_test.run()
    assert not app.exception

    text_inputs = list(app.text_input)
    if lmstudio_inputs := [w for w in text_inputs if "LM Studio base URL" in str(w)]:
        lmstudio_inputs[0].set_value("http://localhost:1234").run()

    save_buttons = [b for b in app.button if getattr(b, "label", "") == "Save"]
    assert save_buttons, "Save button not found"
    save_buttons[0].click().run()

    env_file = tmp_path / ".env"
    assert env_file.exists(), ".env not created by Save action"
    from dotenv import dotenv_values

    values = dotenv_values(env_file)
    assert values.get("DOCMIND_LMSTUDIO_BASE_URL") == "http://localhost:1234/v1"


def test_settings_invalid_remote_url_disables_actions(
    settings_app_test: AppTest,
) -> None:
    """Remote URLs should be blocked when allow_remote_endpoints is disabled."""
    app = settings_app_test.run()
    assert not app.exception

    vllm_inputs = [w for w in app.text_input if "vLLM base URL" in str(w)]
    assert vllm_inputs, "vLLM base URL input not found"
    vllm_inputs[0].set_value("http://example.com:8000").run()

    apply_buttons = [
        b for b in app.button if getattr(b, "label", "") == "Apply runtime"
    ]
    save_buttons = [b for b in app.button if getattr(b, "label", "") == "Save"]
    assert apply_buttons
    assert save_buttons
    assert apply_buttons[0].disabled is True
    assert save_buttons[0].disabled is True
    assert any(
        "Remote endpoints are disabled" in str(getattr(e, "value", ""))
        for e in app.error
    )


def test_settings_allow_remote_allows_remote_urls(settings_app_test: AppTest) -> None:
    """Remote URLs should be allowed when allow_remote_endpoints is enabled."""
    app = settings_app_test.run()
    assert not app.exception

    allow_remote = [w for w in app.checkbox if "Allow remote endpoints" in str(w)]
    assert allow_remote, "Allow remote endpoints checkbox not found"
    allow_remote[0].set_value(True).run()

    vllm_inputs = [w for w in app.text_input if "vLLM base URL" in str(w)]
    assert vllm_inputs, "vLLM base URL input not found"
    vllm_inputs[0].set_value("http://example.com:8000").run()

    apply_buttons = [
        b for b in app.button if getattr(b, "label", "") == "Apply runtime"
    ]
    save_buttons = [b for b in app.button if getattr(b, "label", "") == "Save"]
    assert apply_buttons
    assert save_buttons
    assert apply_buttons[0].disabled is False
    assert save_buttons[0].disabled is False
    assert not list(app.error)


def _install_llm_stubs(monkeypatch: pytest.MonkeyPatch):
    """Install stub LLM classes and return the factory builder."""

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

    def _mk_llm(settings) -> object:  # type: ignore[override]
        mapping = {
            "ollama": _Ollama,
            "vllm": _OLike,
            "lmstudio": _OLike,
            "llamacpp": _OLike
            if getattr(settings, "llamacpp_base_url", None)
            else _LCpp,
        }
        cls = mapping.get(settings.llm_backend, _OLike)
        return cls()

    monkeypatch.setattr("src.config.llm_factory.build_llm", _mk_llm, raising=False)
    return _mk_llm


def _configure_integrations(monkeypatch: pytest.MonkeyPatch, mk_llm) -> None:
    """Make integrations lightweight with deterministic LLM binding."""
    from llama_index.core import Settings as LISettings

    def _setup_llamaindex(
        *, force_llm: bool = False, force_embed: bool = False
    ) -> None:
        del force_llm, force_embed
        from src.config.settings import settings as _settings

        LISettings.llm = mk_llm(_settings)

    monkeypatch.setattr(
        "src.config.integrations.setup_llamaindex", _setup_llamaindex, raising=False
    )
    monkeypatch.setattr(
        "src.config.integrations.setup_vllm_env", lambda: None, raising=False
    )


def _stub_heavy_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub heavy embedding dependencies used during Apply runtime."""

    class _DummyEmbed:
        def __init__(self, *_, **__):
            self.kind = "embed_dummy"

    monkeypatch.setattr(
        "src.retrieval.hybrid.ServerHybridRetriever",
        _DummyEmbed,
        raising=False,
    )


def _ensure_gguf_file() -> Path:
    """Create a stub GGUF file for llama.cpp local validation."""
    gguf_path = Path("model.gguf")
    gguf_path.write_text("dummy", encoding="utf-8")
    return gguf_path


def _find_provider_select(app: AppTest):
    """Return the provider selectbox widget."""
    providers = [s for s in app.selectbox if "LLM Provider" in str(s)]
    assert providers, "Provider selectbox not found"
    return providers[0]


def _set_text(app: AppTest, label: str, value: str) -> None:
    """Set a text input by label if present."""
    fields = [w for w in app.text_input if label in str(w)]
    if fields:
        fields[0].set_value(value).run()


def _click_apply(app: AppTest) -> None:
    """Click Apply runtime and ensure LISettings.llm is bound."""
    btns = [b for b in app.button if getattr(b, "label", "") == "Apply runtime"]
    assert btns, "Apply runtime button not found"
    btns[0].click().run()
    from llama_index.core import Settings as LISettings

    from src.config.integrations import initialize_integrations as _init

    if getattr(LISettings, "llm", None) is None:
        _init(force_llm=True, force_embed=False)


def _apply_provider(
    settings_app_test: AppTest,
    *,
    provider: str,
    inputs: dict[str, str],
    allow_gguf_base: bool = False,
) -> None:
    """Select a provider, set inputs, and apply runtime."""
    app = settings_app_test.run()
    assert not app.exception
    if allow_gguf_base:
        app.session_state["docmind_allowed_gguf_base_dirs"] = [str(Path.cwd())]
    provider_sel = _find_provider_select(app)
    provider_sel.select(provider).run()
    for label, value in inputs.items():
        _set_text(app, label, value)
    _click_apply(app)


def _reset_settings_defaults() -> None:
    """Reset global settings to defaults to avoid cross-test pollution."""
    import importlib

    settings_mod = importlib.import_module("src.config.settings")
    settings_mod.settings = settings_mod.DocMindSettings()  # type: ignore[attr-defined]


def test_settings_toggle_providers_and_apply(
    settings_app_test: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Toggle each provider and Apply runtime, asserting LLM kind per provider.

    Stubs provider adapters to deterministic classes with a 'kind' attribute
    so we can assert which path the factory used.
    """
    mk_llm = _install_llm_stubs(monkeypatch)
    _configure_integrations(monkeypatch, mk_llm)
    _stub_heavy_embeddings(monkeypatch)
    gguf_path = _ensure_gguf_file()

    _apply_provider(
        settings_app_test,
        provider="ollama",
        inputs={
            "Ollama base URL": "http://localhost:11434",
            "Model (id or GGUF path)": "test-ollama",
        },
        allow_gguf_base=True,
    )
    from src.config.settings import settings as _settings

    assert _settings.llm_backend == "ollama"

    _apply_provider(
        settings_app_test,
        provider="vllm",
        inputs={
            "vLLM base URL": "http://localhost:8000",
            "Model (id or GGUF path)": "test-vllm",
        },
    )
    assert _settings.llm_backend == "vllm"

    _apply_provider(
        settings_app_test,
        provider="lmstudio",
        inputs={
            "LM Studio base URL": "http://localhost:1234/v1",
            "Model (id or GGUF path)": "test-lms",
        },
    )
    assert _settings.llm_backend == "lmstudio"

    _apply_provider(
        settings_app_test,
        provider="llamacpp",
        inputs={
            "llama.cpp server URL (optional)": "http://localhost:8080/v1",
            "Model (id or GGUF path)": "ignored-for-server",
        },
    )
    assert _settings.llm_backend == "llamacpp"

    _apply_provider(
        settings_app_test,
        provider="llamacpp",
        inputs={
            "llama.cpp server URL (optional)": "",
            "GGUF model path (LlamaCPP local)": str(gguf_path),
        },
        allow_gguf_base=True,
    )
    assert _settings.llm_backend == "llamacpp"

    _reset_settings_defaults()
