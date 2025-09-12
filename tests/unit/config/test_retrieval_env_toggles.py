import importlib
import os
from types import ModuleType


def _reload_settings() -> ModuleType:
    """Reload settings module to apply current environment.

    Returns:
        The reloaded module object for src.config.settings.
    """
    if "src.config.settings" in list(globals()):
        del globals()["src.config.settings"]
    if "src.config.settings" in list(locals()):
        del locals()["src.config.settings"]
    # Ensure any prior import state is removed
    if "src.config.settings" in importlib.sys.modules:
        importlib.sys.modules.pop("src.config.settings")
    return importlib.import_module("src.config.settings")


def test_dbsf_enabled_overrides_fusion_mode(monkeypatch):
    """DBSF boolean flag should override fusion_mode on startup."""
    monkeypatch.setenv("DOCMIND_RETRIEVAL__FUSION_MODE", "rrf")
    monkeypatch.setenv("DOCMIND_RETRIEVAL__DBSF_ENABLED", "true")
    settings_mod = _reload_settings()
    assert str(settings_mod.settings.retrieval.fusion_mode).lower() == "dbsf"


def test_telemetry_enabled_bridges_disabled_env(monkeypatch):
    """Disabling telemetry maps to DOCMIND_TELEMETRY_DISABLED=true for sinks."""
    # Clear possibly set env
    monkeypatch.delenv("DOCMIND_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("DOCMIND_TELEMETRY_ENABLED", "false")
    _reload_settings()
    assert os.environ.get("DOCMIND_TELEMETRY_DISABLED", "").lower() in {"1", "true"}
