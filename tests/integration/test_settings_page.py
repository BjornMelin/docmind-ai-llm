"""Integration tests for Settings page (SPEC-001 runtime roundtrip).

Exercises Apply runtime and Save to .env without external dependencies.
Relies on Streamlit AppTest to run src/pages/04_settings.py in a temp cwd.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration


def _install_integrations_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    """Install a lightweight `src.config.integrations` stub for Apply runtime."""
    import sys

    calls: list[dict[str, object]] = []

    def initialize_integrations(
        *, force_llm: bool = False, force_embed: bool = False
    ) -> None:
        from src.config.settings import settings as _settings

        calls.append(
            {
                "force_llm": force_llm,
                "force_embed": force_embed,
                "backend": _settings.llm_backend,
            }
        )
        # Keep integration tests offline and deterministic by relying on the
        # integration-tier Settings.llm guard (MockLLM).
        from llama_index.core import Settings as LISettings

        if getattr(LISettings, "_llm", None) is None:
            from llama_index.core.llms import MockLLM

            LISettings.llm = MockLLM(max_tokens=256)

    stub = ModuleType("src.config.integrations")
    stub.initialize_integrations = initialize_integrations  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.config.integrations", stub)
    return calls


def _assert_hybrid_toggle_is_read_only(app: AppTest) -> None:
    """Assert server-side hybrid toggle is exposed as read-only text."""
    labels = [str(getattr(inp, "label", "")).lower() for inp in app.text_input]
    assert "server-side hybrid enabled" in labels, (
        "Expected server-side hybrid retrieval field to be present in Settings page"
    )

    checkbox_labels = [str(getattr(cb, "label", "")).lower() for cb in app.checkbox]
    assert not any("server-side hybrid" in lbl for lbl in checkbox_labels), (
        "Expected server-side hybrid to be displayed as read-only text, not a checkbox"
    )


@pytest.fixture(name="settings_app_test")
def fixture_settings_app_test(tmp_path, monkeypatch) -> Iterator[AppTest]:
    """Create an AppTest instance for the Settings page with temp cwd.

    - Runs page in a temporary working directory so Save writes to a temp .env.
    - Avoids external side effects and keeps tests deterministic.
    """
    # Ensure cwd is a temp directory for .env persistence
    monkeypatch.chdir(tmp_path)

    from llama_index.core import Settings as LISettings

    original_llm = getattr(LISettings, "_llm", None)
    original_embed = getattr(LISettings, "_embed_model", None)

    # Avoid slow/optional GraphRAG adapter discovery during Settings AppTest reruns.
    # The Settings page only needs the badge health tuple; heavy adapter imports are
    # out of scope for this integration test and can dominate runtime under coverage.
    monkeypatch.setattr(
        "src.retrieval.adapter_registry.get_default_adapter_health",
        lambda *, force_refresh=False: (
            False,
            "unavailable",
            "GraphRAG disabled for Settings AppTest",
        ),
    )

    # Build AppTest for the Settings page file
    page_path = Path(__file__).resolve().parents[2] / "src" / "pages" / "04_settings.py"
    try:
        yield AppTest.from_file(
            str(page_path),
            default_timeout=int(os.environ.get("TEST_TIMEOUT", "30")),
        )
    finally:
        LISettings.llm = original_llm
        LISettings.embed_model = original_embed


def test_settings_apply_runtime_calls_initialize_integrations(
    settings_app_test: AppTest,
    monkeypatch: pytest.MonkeyPatch,
    reset_settings_after_test: None,
) -> None:
    """Apply runtime should call initialize_integrations.

    Expect force_llm=True and force_embed=False.
    """
    app = settings_app_test.run()
    assert not app.exception

    # Install stub after the initial render to avoid impacting import-time
    # behavior in environments with additional optional dependencies (CI llama job).
    calls = _install_integrations_stub(monkeypatch)

    buttons = [b for b in app.button if getattr(b, "label", "") == "Apply runtime"]
    assert buttons, "Apply runtime button not found"
    buttons[0].click().run()

    from src.config.settings import settings as _settings

    assert calls == [
        {"force_llm": True, "force_embed": False, "backend": _settings.llm_backend}
    ], f"Unexpected initialize_integrations calls: {calls}"


def test_settings_save_persists_env(
    settings_app_test: AppTest,
    tmp_path: Path,
    reset_settings_after_test: None,
) -> None:
    """Saving settings should write expected keys into .env in temp cwd."""
    import sys

    before = set(sys.modules)
    app = settings_app_test.run()
    assert not app.exception
    _assert_hybrid_toggle_is_read_only(app)

    # Perf guard: initial render should not trigger heavy integration imports.
    after = set(sys.modules)
    delta = after - before
    assert "src.config.integrations" not in delta

    # Set a few key fields to ensure persistence writes recognizable values
    # Model field
    text_inputs = list(app.text_input)
    # Find model input by label
    if model_inputs := [w for w in text_inputs if "Model (id or GGUF path)" in str(w)]:
        model_inputs[0].set_value("Hermes-2-Pro-Llama-3-8B").run()

    # LM Studio base URL (must end with /v1)
    if lmstudio_inputs := [w for w in text_inputs if "LM Studio base URL" in str(w)]:
        lmstudio_inputs[0].set_value("http://localhost:1234/v1").run()

    # Ollama advanced settings
    if api_key_inputs := [w for w in text_inputs if "Ollama API key" in str(w)]:
        api_key_inputs[0].set_value("key-123").run()

    allow_remote = [w for w in app.checkbox if "Allow remote endpoints" in str(w)]
    assert allow_remote, "Allow remote endpoints checkbox not found"
    allow_remote[0].set_value(True).run()

    web_tools = [w for w in app.checkbox if "Enable Ollama web search tools" in str(w)]
    assert web_tools, "Ollama web tools checkbox not found"
    web_tools[0].set_value(True).run()

    logprobs = [w for w in app.checkbox if "Enable Ollama logprobs" in str(w)]
    if logprobs:
        logprobs[0].set_value(True).run()

    embed_dims = [w for w in app.number_input if "Embed dimensions" in str(w)]
    assert embed_dims, "Embed dimensions input not found"
    embed_dims[0].set_value(384).run()

    top_logprobs = [w for w in app.number_input if "Top logprobs" in str(w)]
    assert top_logprobs, "Top logprobs input not found"
    top_logprobs[0].set_value(2).run()

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
    assert values.get("DOCMIND_OLLAMA_API_KEY") == "key-123"
    assert values.get("DOCMIND_OLLAMA_ENABLE_WEB_SEARCH") == "true"
    assert values.get("DOCMIND_OLLAMA_EMBED_DIMENSIONS") == "384"
    assert values.get("DOCMIND_OLLAMA_ENABLE_LOGPROBS") == "true"
    assert values.get("DOCMIND_OLLAMA_TOP_LOGPROBS") == "2"


def test_settings_save_normalizes_lmstudio_url(
    settings_app_test: AppTest,
    tmp_path: Path,
    reset_settings_after_test: None,
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
    reset_settings_after_test: None,
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


def test_settings_allow_remote_allows_remote_urls(
    settings_app_test: AppTest,
    reset_settings_after_test: None,
) -> None:
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


def test_settings_warns_when_ollama_allowlist_missing(
    settings_app_test: AppTest,
    reset_settings_after_test: None,
) -> None:
    """Enabling Ollama web tools should warn when allowlist lacks ollama.com."""
    app = settings_app_test.run()
    assert not app.exception

    allow_remote = [w for w in app.checkbox if "Allow remote endpoints" in str(w)]
    assert allow_remote, "Allow remote endpoints checkbox not found"
    allow_remote[0].set_value(True).run()

    web_tool_checks = [
        w for w in app.checkbox if "Enable Ollama web search tools" in str(w)
    ]
    assert web_tool_checks, "Ollama web tools checkbox not found"
    web_tool_checks[0].set_value(True).run()

    warnings = [str(getattr(w, "value", "")) for w in app.warning]
    expected_warning = (
        "Ollama web tools require `https://ollama.com` in "
        "`DOCMIND_SECURITY__ENDPOINT_ALLOWLIST`."
    )
    assert any(msg.strip() == expected_warning for msg in warnings), warnings


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
    """Click Apply runtime."""
    btns = [b for b in app.button if getattr(b, "label", "") == "Apply runtime"]
    assert btns, "Apply runtime button not found"
    assert btns[0].disabled is False, (
        f"Apply runtime unexpectedly disabled; errors={list(app.error)}"
    )
    btns[0].click().run()
    assert not app.exception


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


@pytest.fixture
def reset_settings_after_test() -> Iterator[None]:
    """Reset global settings to defaults before and after test to avoid pollution."""
    import importlib

    def _reset_settings() -> None:
        settings_mod = importlib.import_module("src.config.settings")
        current = settings_mod.settings
        if not isinstance(current, settings_mod.DocMindSettings):  # type: ignore[attr-defined]
            raise TypeError(
                "src.config.settings.settings was unexpectedly replaced with a "
                f"{type(current)!r}; expected DocMindSettings"
            )
        current.__init__(_env_file=None)  # type: ignore[arg-type]
        # Clear bootstrap globals so later tests don't inherit dotenv state.
        settings_mod.reset_bootstrap_state()

    # Setup: ensure clean state before test
    _reset_settings()
    yield
    # Teardown: reset settings after test
    _reset_settings()


def test_settings_toggle_providers_and_apply(
    settings_app_test: AppTest,
    monkeypatch: pytest.MonkeyPatch,
    reset_settings_after_test: None,
) -> None:
    """Toggle each provider and Apply runtime (offline)."""
    calls = _install_integrations_stub(monkeypatch)
    gguf_path = _ensure_gguf_file()

    def _assert_last_call(expected_backend: str) -> None:
        assert calls, "Expected initialize_integrations to be called"
        assert calls[-1] == {
            "force_llm": True,
            "force_embed": False,
            "backend": expected_backend,
        }

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
    _assert_last_call("ollama")

    _apply_provider(
        settings_app_test,
        provider="vllm",
        inputs={
            "vLLM base URL": "http://localhost:8000",
            "Model (id or GGUF path)": "test-vllm",
        },
    )
    assert _settings.llm_backend == "vllm"
    _assert_last_call("vllm")

    _apply_provider(
        settings_app_test,
        provider="lmstudio",
        inputs={
            "LM Studio base URL": "http://localhost:1234/v1",
            "Model (id or GGUF path)": "test-lms",
        },
    )
    assert _settings.llm_backend == "lmstudio"
    _assert_last_call("lmstudio")

    _apply_provider(
        settings_app_test,
        provider="llamacpp",
        inputs={
            "llama.cpp server URL (optional)": "http://localhost:8080/v1",
            "Model (id or GGUF path)": "ignored-for-server",
        },
    )
    assert _settings.llm_backend == "llamacpp"
    _assert_last_call("llamacpp")

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
    _assert_last_call("llamacpp")

    assert len(calls) == 5
