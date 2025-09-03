"""Unified configuration interface for DocMind AI.

This module provides the primary configuration interface following ADR-024
implementing Task 2.2.2: LlamaIndex Settings Integration Pattern.

Key Features:
- Single source of truth via unified settings object
- Automatic LlamaIndex Settings integration on import
- Environment variable support for all configuration
- Simplified import patterns across codebase

Usage:
    # Primary pattern (recommended for all code)
    from src.config import settings

    # LlamaIndex integration functions
    from src.config import setup_llamaindex, initialize_integrations
"""

from .integrations import initialize_integrations, setup_llamaindex
from .settings import settings

# IMPORTANT:
# Do NOT auto-initialize LlamaIndex on import. Tests and CLI may import
# src.config for settings without bringing in heavy integration side effects.
# Call initialize_integrations()/setup_llamaindex() explicitly where needed.

__all__ = [
    "initialize_integrations",
    "settings",
    "setup_llamaindex",
]
