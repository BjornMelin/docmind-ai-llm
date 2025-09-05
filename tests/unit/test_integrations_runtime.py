"""Smoke test for initialize_integrations rebind behavior (SPEC-001)."""

from llama_index.core import Settings

from src.config.integrations import initialize_integrations
from src.config.settings import settings as global_settings


def test_initialize_integrations_rebinds_settings_llm() -> None:
    """Apply runtime should bind Settings.llm for selected provider."""
    # Save and restore
    original = Settings.llm
    try:
        # Point to a deterministic backend
        global_settings.llm_backend = "lmstudio"  # type: ignore[assignment]
        global_settings.lmstudio_base_url = "http://localhost:1234/v1"  # type: ignore[assignment]
        global_settings.model = "Hermes-2-Pro-Llama-3-8B"  # type: ignore[assignment]

        Settings.llm = None
        initialize_integrations(force_llm=True, force_embed=False)
        assert Settings.llm is not None
    finally:
        Settings.llm = original
