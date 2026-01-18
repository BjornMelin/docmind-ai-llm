"""Shared test fixtures for vector index and query engine mocking.

Provides reusable fake implementations of LlamaIndex-compatible components
for testing analysis flows that require vector index and query engine interactions.
"""

from __future__ import annotations

from types import SimpleNamespace


class _FakeQueryEngine:
    """Fake query engine for testing vector index interactions.

    Simulates LlamaIndex query engine behavior by returning deterministic
    responses with configurable document IDs embedded in source metadata.
    Supports both positional and keyword argument initialization.
    """

    def __init__(self, doc_id: str | None = None) -> None:
        """Initialize the fake query engine.

        Args:
            doc_id: Document ID to embed in query responses. Defaults to "combined"
                    if not provided.

        Raises:
            ValueError: If doc_id is provided but is an empty string.
        """
        if doc_id is not None and not doc_id:
            raise ValueError("doc_id must be a non-empty string or None")
        self._doc_id = doc_id or "combined"

    def query(self, _query: str) -> object:
        """Execute a fake query returning a deterministic response.

        Args:
            _query: The query string (unused in fake implementation).

        Returns:
            A SimpleNamespace object structured like LlamaIndex QueryEngine response,
            with response and source_nodes attributes containing document metadata.
        """
        node = SimpleNamespace(metadata={"doc_id": self._doc_id})
        src = SimpleNamespace(node=node)
        return SimpleNamespace(response=f"answer:{self._doc_id}", source_nodes=[src])


class _FakeVectorIndex:
    """Fake vector index for testing analysis flows.

    Simulates LlamaIndex VectorIndex behavior by extracting document filters
    from query engine parameters and returning corresponding query engines.
    """

    def as_query_engine(self, **kwargs: object) -> _FakeQueryEngine:
        """Create a query engine configured with the given parameters.

        Extracts a stable doc_id hint from the filters parameter when present,
        falling back to "combined" for multi-document queries.

        Args:
            **kwargs: Keyword arguments, with optional "filters" parameter
                      containing LlamaIndex filter metadata.

        Returns:
            A _FakeQueryEngine configured with an appropriate doc_id.
        """
        doc_id = "combined"
        filters = kwargs.get("filters")
        parts = getattr(filters, "filters", None)
        if isinstance(parts, list) and parts:
            value = getattr(parts[0], "value", None)
            if value is not None:
                doc_id = str(value)
        return _FakeQueryEngine(doc_id=doc_id)


__all__ = ["_FakeQueryEngine", "_FakeVectorIndex"]
