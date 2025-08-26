"""Professional configuration interface for DocMind AI.

This module provides a clean, minimal interface to the unified configuration
architecture with convenient access patterns.

Modern Configuration Architecture:
- DocMindSettings: App-specific configuration (src.config.app_settings)
- LlamaIndex Settings: Framework configuration (src.config.llamaindex_setup)
- Data Models: Pydantic schemas (src.models.schemas)

Usage:
    For new code, import directly from the specific modules:
    - from src.config.app_settings import app_settings
    - from src.models.schemas import AnalysisOutput

    For convenience, this module also provides:
    - from src.config.settings import settings, AnalysisOutput
"""

import logging

# Import components from their proper locations
from src.models.schemas import AnalysisOutput

from .app_settings import app_settings
from .llamaindex_setup import setup_llamaindex

# Initialize module-specific logger following professional best practices
logger = logging.getLogger(__name__)

# Initialize LlamaIndex on module import
# This ensures consistent configuration across the application
try:
    setup_llamaindex()
    logger.info("LlamaIndex configuration initialized successfully")
except Exception as e:
    logger.error("Failed to initialize LlamaIndex configuration: %s", e)

# Convenient alias for settings access
# Direct app_settings import is preferred for new code
settings = app_settings

# Professional module exports - minimal and focused
__all__ = [
    "AnalysisOutput",
    "settings",
    "app_settings",
    "setup_llamaindex",
]
