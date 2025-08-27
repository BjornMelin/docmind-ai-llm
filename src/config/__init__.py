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

# Initialize LlamaIndex automatically on import (skip in test environment)
import os

from .integrations import initialize_integrations, setup_llamaindex
from .settings import settings

if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("TESTING")):
    try:
        setup_llamaindex()
        import logging

        logging.getLogger(__name__).info(
            "LlamaIndex configuration initialized successfully"
        )
    except (ImportError, AttributeError, ValueError, ConnectionError, OSError) as e:
        import logging

        logging.getLogger(__name__).error(
            "Failed to initialize LlamaIndex configuration: %s", e
        )

__all__ = [
    "settings",
    "initialize_integrations",
    "setup_llamaindex",
]
