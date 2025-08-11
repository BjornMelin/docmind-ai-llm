"""Simplified models module for DocMind AI.

This module exports the essential configuration and analysis models.
"""  # noqa: N999

# Import from the core module
from .core import AnalysisOutput, AppSettings, Settings, settings

__all__ = ["AnalysisOutput", "AppSettings", "Settings", "settings"]
