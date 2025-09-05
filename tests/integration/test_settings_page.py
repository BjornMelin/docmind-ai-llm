"""Integration tests for Settings page (SPEC-001 runtime roundtrip).

Exercises Apply runtime and Save to .env without external dependencies.
Relies on Streamlit AppTest to run src/pages/04_settings.py in a temp cwd.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture()
def settings_app_test(tmp_path, monkeypatch) -> Iterator[AppTest]:
    """Create an AppTest instance for the Settings page with temp cwd.

    - Runs page in a temporary working directory so Save writes to a temp .env.
    - Avoids external side effects and keeps tests deterministic.
    """
    # Ensure cwd is a temp directory for .env persistence
    monkeypatch.chdir(tmp_path)

    # Build AppTest for the Settings page file
    page_path = (
        Path(__file__).resolve().parents[2] / "src" / "pages" / "04_settings.py"
    )
    yield AppTest.from_file(str(page_path))


def test_settings_apply_runtime_rebinds_llm(settings_app_test: AppTest) -> None:
    """Apply runtime should rebind Settings.llm immediately (force_llm=True)."""
    app = settings_app_test.run()
    assert not app.exception

    # Find and click the "Apply runtime" button
    # Use a robust match: click any button whose label contains "Apply runtime"
    buttons = [b for b in app.button if "Apply runtime" in str(b)]
    if buttons:
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
    model_inputs = [w for w in text_inputs if "Model (id or GGUF path)" in str(w)]
    if model_inputs:
        model_inputs[0].set_value("Hermes-2-Pro-Llama-3-8B").run()

    # LM Studio base URL (must end with /v1)
    lmstudio_inputs = [w for w in text_inputs if "LM Studio base URL" in str(w)]
    if lmstudio_inputs:
        lmstudio_inputs[0].set_value("http://localhost:1234/v1").run()

    # Click Save
    save_buttons = [b for b in app.button if str(b).strip().endswith("Save")]
    if save_buttons:
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

