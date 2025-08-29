"""LlamaIndex SimpleKVStore cache for document processing.

Single SQLite file, no external services required.
Perfect for single-user Streamlit app with multi-agent coordination.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from llama_index.core.storage.kvstore import SimpleKVStore
from loguru import logger

from src.interfaces import CacheInterface


class SimpleCache(CacheInterface):
    """LlamaIndex SimpleKVStore cache for document processing.

    Single SQLite file, no external services required.
    Perfect for single-user Streamlit app with multi-agent coordination.
    """

    def __init__(self, cache_dir: str = "./cache") -> None:
        """Initialize with SQLite-based cache."""
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Create cache storage - simple KVStore for document caching
        db_path = cache_path / "docmind.db"
        if db_path.exists() and db_path.stat().st_size > 0:
            try:
                self.cache = SimpleKVStore.from_persist_path(str(db_path))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Cache file corrupted, creating new cache: {e}")
                db_path.unlink()  # Remove corrupted file
                self.cache = SimpleKVStore()
        else:
            self.cache = SimpleKVStore()

        # Store persistence path for later saves
        self._persist_path = str(db_path)

        # Hit/miss tracking for real cache metrics
        self._hits = 0
        self._misses = 0
        self._stored_documents = 0  # Track stored documents count

        logger.info(f"SimpleCache initialized at {cache_path}")

    async def get_document(self, path: str) -> Any | None:
        """Get cached document processing result."""
        try:
            key = self._hash(path)
            try:
                result = self.cache.get(key)
                if result is not None:
                    self._hits += 1
                    logger.debug(f"Cache hit for: {Path(path).name}")
                    return result
                else:
                    self._misses += 1
                    return None
            except (KeyError, ValueError):
                # Key not found in cache
                self._misses += 1
                return None
        except (OSError, ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Cache get failed: {e}")
            self._misses += 1
            return None

    async def store_document(self, path: str, result: Any) -> bool:
        """Store document processing result."""
        # Validate input parameters
        if not path or path.isspace():
            raise ValueError("Document path cannot be empty or whitespace")

        if len(path) > 255:  # Reasonable path length limit
            raise ValueError("Document path is too long")

        if "\x00" in path:  # Null character check
            raise ValueError("Document path contains null character")

        try:
            key = self._hash(path)
            # Check if this is a new document
            is_new = self.cache.get(key) is None
            self.cache.put(key, result)
            # Increment count for new documents
            if is_new:
                self._stored_documents += 1
            # Persist to disk
            self.cache.persist(self._persist_path)
            logger.debug(f"Cached result for: {Path(path).name}")
            return True
        except (OSError, ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Cache store failed: {e}")
            return False

    def _hash(self, path: str) -> str:
        """File hash with size+mtime for cache invalidation."""
        p = Path(path)
        if p.exists():
            key = f"{p.name}_{p.stat().st_size}_{p.stat().st_mtime}"
        else:
            key = f"{p.name}_missing"
        return hashlib.sha256(key.encode()).hexdigest()

    async def clear_cache(self) -> bool:
        """Clear all cached documents."""
        try:
            # Clear by creating new cache
            self.cache = SimpleKVStore()
            # Remove the file if it exists
            db_file = Path(self._persist_path)
            if db_file.exists():
                db_file.unlink()
            # Reset hit/miss counters and document count
            self._hits = 0
            self._misses = 0
            self._stored_documents = 0
            logger.info("Cleared document cache and reset metrics")
            return True
        except (OSError, ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Cache clear failed: {e}")
            return False

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get simple cache statistics."""
        try:
            # Calculate hit rate
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

            # Get document count from our tracking
            total_documents = self._stored_documents

            return {
                "cache_type": "simple_sqlite",
                "total_documents": total_documents,
                "hit_rate": round(hit_rate, 3),
                "size_mb": total_documents * 0.1,  # Rough size estimate
                "total_requests": total_requests,
                "hits": self._hits,
                "misses": self._misses,
            }
        except (OSError, ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Cache stats failed: {e}")
            return {"error": str(e), "cache_type": "simple_sqlite"}
