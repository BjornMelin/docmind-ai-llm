"""Unified configuration interface for DocMind AI.

This module provides the primary configuration interface following ADR-024
implementing Task 2.2.2: LlamaIndex Settings Integration Pattern.

Key Features:
- Single source of truth via unified settings object
- Explicit LlamaIndex integration entrypoints (no import-time side effects)
- Environment variable support for all configuration
- Simplified import patterns across codebase

Usage:
    # Primary pattern (recommended for all code)
    from src.config import settings

    # LlamaIndex integration functions
    from src.config import setup_llamaindex, initialize_integrations
"""

from __future__ import annotations

# IMPORTANT: import order matters.
#
# - Import and bind the settings instance first so `from src.config import settings`
#   always resolves to the settings object (not the `src.config.settings` module).
# - Then import integration entrypoints without triggering initialization.
# ruff: noqa: I001
from .settings import settings
from .integrations import initialize_integrations, setup_llamaindex

__all__ = [
    "initialize_integrations",
    "settings",
    "setup_llamaindex",
]
