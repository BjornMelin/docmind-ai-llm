"""Models module for DocMind AI.

This module provides backward compatibility by re-exporting configuration
and analysis models from the centralized settings system.

DEPRECATED: Direct imports from this module are deprecated.
Use `from src.config.settings import Settings, AnalysisOutput, settings` instead.
"""  # noqa: N999

# Re-export from centralized settings for backward compatibility
from src.config.settings import AnalysisOutput, AppSettings, Settings, settings

__all__ = ["AnalysisOutput", "AppSettings", "Settings", "settings"]
