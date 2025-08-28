"""Database utilities for DocMind AI.

This module provides database-related utilities including
persistence and storage management.

For backward compatibility with tests, this module re-exports
functionality from storage.py.
"""

# Import all functionality from storage for backward compatibility
from src.utils.storage import *  # noqa: F401, F403

# Ensure the module can be imported as expected by tests
__all__ = ["storage"]
