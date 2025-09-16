"""Cache manager utilities for ingestion pipeline.

This module provides a light abstraction over LlamaIndex's :class:`IngestionCache`
so we can honour application configuration without introducing external
infrastructure such as Redis. The manager is intentionally simple:

* DuckDB is retained as the default persistent backend.
* An in-memory backend is provided for unit tests and ephemeral runs.
* Collection names encode pipeline and hashing versions (and tenant when
  provided) to ensure safe cache invalidation without brittle file deletes.
* Clearing the cache rotates the namespace and optionally purges prior
  collections from the embedded database.

The implementation avoids over-engineering while making caches configurable and
observable.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core.ingestion import IngestionCache
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
from llama_index.storage.kvstore.simple_kvstore import SimpleKVStore

from src.models.processing import ProcessingStrategy

_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class CacheSettings:
    """Subset of configuration used by :class:`CacheManager`."""

    backend: str
    dir: Path
    filename: str
    pipeline_version: str
    cache_version: str
    hashing_version: str
    tenant_id: str | None = None


class CacheManager:
    """Factory and lifecycle helper for ingestion caches."""

    def __init__(self, settings: CacheSettings) -> None:
        """Initialize the cache manager.

        Args:
            settings: Consolidated cache configuration.
        """
        self._settings = settings
        self._generation = 0
        self._duckdb_store: DuckDBKVStore | None = None
        self._memory_store: SimpleKVStore | None = None

    @property
    def generation(self) -> int:
        """Return the current namespace generation."""
        return self._generation

    def build_cache(self, strategy: ProcessingStrategy) -> IngestionCache:
        """Instantiate an ingestion cache for the provided strategy.

        Args:
            strategy: Processing strategy driving the namespace selection.

        Returns:
            IngestionCache: Cache instance bound to the current namespace.
        """
        collection = self._collection_name(strategy, self._generation)
        backend = self._settings.backend.lower()

        if backend == "duckdb":
            store = self._get_duckdb_store()
        elif backend == "memory":
            store = self._get_memory_store()
        else:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported cache backend '{backend}'")

        return IngestionCache(cache=store, collection=collection)

    def rotate(self) -> None:
        """Advance the namespace generation for future cache instances."""
        self._generation += 1

    def purge_collections(
        self, strategies: Iterable[ProcessingStrategy], generation: int
    ) -> None:
        """Delete cache rows for a generation across provided strategies.

        Args:
            strategies: Strategies whose cache collections should be purged.
            generation: Namespace generation to remove.
        """
        backend = self._settings.backend.lower()
        if backend == "duckdb":
            store = self._get_duckdb_store()
            client = store.client
            for strategy in strategies:
                collection = self._collection_name(strategy, generation)
                client.execute(
                    f"DELETE FROM {store.table.alias} WHERE collection = ?",  # noqa: S608
                    [collection],
                )
        elif backend == "memory":
            store = self._get_memory_store()
            for strategy in strategies:
                collection = self._collection_name(strategy, generation)
                keys = list(store.get_all(collection).keys())
                for key in keys:
                    store.delete(key, collection)

    def cache_stats(self) -> dict[str, object]:
        """Return simple observability data about the active cache.

        Returns:
            dict[str, object]: Backend-specific statistics, including the
            current namespace generation.
        """
        backend = self._settings.backend.lower()
        if backend == "duckdb":
            store = self._get_duckdb_store()
            db_path = Path(store.persist_dir) / store.database_name
            exists = db_path.exists()
            size = db_path.stat().st_size if exists else 0
            return {
                "backend": "duckdb",
                "path": str(db_path),
                "exists": exists,
                "size_bytes": size,
                "generation": self._generation,
            }

        return {"backend": "memory", "generation": self._generation}

    def _get_duckdb_store(self) -> DuckDBKVStore:
        """Get or create the shared DuckDB-backed KV store."""
        if self._duckdb_store is None:
            self._settings.dir.mkdir(parents=True, exist_ok=True)
            self._duckdb_store = DuckDBKVStore(
                database_name=self._settings.filename,
                persist_dir=str(self._settings.dir),
            )
        return self._duckdb_store

    def _get_memory_store(self) -> SimpleKVStore:
        """Get or create the in-memory KV store used for tests."""
        if self._memory_store is None:
            self._memory_store = SimpleKVStore()
        return self._memory_store

    def _collection_name(self, strategy: ProcessingStrategy, generation: int) -> str:
        """Build a namespace for cache entries.

        Args:
            strategy: Strategy associated with the cache entries.
            generation: Namespace generation identifier.

        Returns:
            str: Sanitized collection name safe for KV backends.
        """
        components = [
            "docmind",
            strategy.value,
            self._settings.pipeline_version,
            self._settings.cache_version,
            self._settings.hashing_version,
            f"gen{generation}",
        ]
        if self._settings.tenant_id:
            components.append(self._settings.tenant_id)

        normalized = [self._sanitize(component) for component in components]
        joined = "-".join(filter(None, normalized))
        if len(joined) <= 60:
            return joined

        digest = hashlib.blake2s(joined.encode("utf-8"), digest_size=6).hexdigest()
        return "-".join([normalized[0], normalized[1], digest])

    @staticmethod
    def _sanitize(value: str) -> str:
        """Normalize component strings for namespace construction."""
        lowered = value.lower()
        cleaned = _SANITIZE_RE.sub("-", lowered)
        return cleaned.strip("-")


def build_cache_settings(settings: Any) -> CacheSettings:
    """Extract cache settings from the runtime configuration.

    Args:
        settings: Application settings object (typically DocMindSettings).

    Returns:
        CacheSettings: Simplified view consumed by :class:`CacheManager`.
    """
    cache_dir = Path(getattr(settings.cache, "dir", Path("./cache")))
    pipeline_version = getattr(
        getattr(settings, "processing", settings), "pipeline_version", "1"
    )
    cache_version = str(getattr(settings, "cache_version", "0"))
    hashing_version = getattr(
        getattr(settings, "hashing", settings), "hmac_secret_version", "1"
    )

    return CacheSettings(
        backend=getattr(settings.cache, "backend", "duckdb"),
        dir=cache_dir,
        filename=getattr(settings.cache, "filename", "docmind.duckdb"),
        pipeline_version=str(pipeline_version),
        cache_version=cache_version,
        hashing_version=str(hashing_version),
        tenant_id=getattr(settings, "tenant_id", None),
    )
