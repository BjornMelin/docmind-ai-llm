"""Interfaces for dependency injection.

This module provides abstract interfaces for key components to enable
clean dependency injection and testing.
"""

from .cache import CacheInterface

__all__ = ["CacheInterface"]
