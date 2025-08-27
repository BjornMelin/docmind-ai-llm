"""Cache interface for dependency injection.

Provides abstract base class for cache implementations to enable
clean dependency injection and testing.
"""

from abc import ABC, abstractmethod
from typing import Any


class CacheInterface(ABC):
    """Abstract interface for cache implementations."""

    @abstractmethod
    async def get_document(self, path: str) -> Any | None:
        """Get cached document processing result.

        Args:
            path: Document path to retrieve

        Returns:
            Cached result or None if not found
        """

    @abstractmethod
    async def store_document(self, path: str, result: Any) -> bool:
        """Store document processing result.

        Args:
            path: Document path to store
            result: Processing result to cache

        Returns:
            True if stored successfully
        """

    @abstractmethod
    async def clear_cache(self) -> bool:
        """Clear all cached documents.

        Returns:
            True if cleared successfully
        """

    @abstractmethod
    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache metrics
        """
