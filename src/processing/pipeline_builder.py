"""Dependency-injected pipeline builder for document ingestion.

The builder encapsulates creation of LlamaIndex `IngestionPipeline` instances so
callers can swap transformation providers, cache/docstore factories, and future
telemetry hooks without modifying the core processor. Although the initial
implementation focuses on wiring the existing unstructured transformation, the
APIs are intentionally generic to support upcoming OCR stages and cache layers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.schema import TransformComponent
from llama_index.core.storage.docstore import BaseDocumentStore

from src.models.processing import ProcessingStrategy

TransformFactory = Callable[[ProcessingStrategy], Iterable[TransformComponent]]
CacheFactory = Callable[[ProcessingStrategy], IngestionCache | None]
DocstoreFactory = Callable[[], BaseDocumentStore | None]


class PipelineBuilder:
    """Construct ingestion pipelines using dependency injection."""

    def __init__(
        self,
        *,
        transform_factories: Sequence[TransformFactory],
        cache_factory: CacheFactory | None,
        docstore_factory: DocstoreFactory,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store factories that assemble ingestion pipelines.

        Args:
            transform_factories: Ordered callables that provide transforms for a
                given processing strategy.
            cache_factory: Factory returning an ingestion cache instance or
                ``None`` when caching is disabled.
            docstore_factory: Factory returning a document store instance or
                ``None`` when persistence is disabled.
            metadata: Optional metadata annotated onto constructed pipelines.
        """
        self._transform_factories = list(transform_factories)
        self._cache_factory = cache_factory
        self._docstore_factory = docstore_factory
        self._metadata = metadata or {}

    def build(self, strategy: ProcessingStrategy) -> IngestionPipeline:
        """Create an :class:`IngestionPipeline` for ``strategy``.

        Args:
            strategy: Processing strategy selected for the document.

        Returns:
            Configured :class:`IngestionPipeline` instance.
        """
        transformations: list[TransformComponent] = []
        for factory in self._transform_factories:
            transformations.extend(list(factory(strategy)))

        cache = self._cache_factory(strategy) if self._cache_factory else None
        docstore = self._docstore_factory()

        pipeline = IngestionPipeline(
            transformations=transformations,
            cache=cache,
            docstore=docstore,
        )
        # Attach pipeline metadata for debugging/telemetry; LlamaIndex does not
        # use it directly but keeping reference simplifies validation.
        pipeline._docmind_metadata = self._metadata | {  # type: ignore[attr-defined]
            "strategy": strategy.value,
        }
        return pipeline
