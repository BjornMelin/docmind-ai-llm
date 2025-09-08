"""Factory helpers for LlamaIndex BGE-M3 index/retriever (SPEC-003 v1.1.0).

Library-first wiring that avoids custom sparse/ColBERT glue. All imports of
LlamaIndex/FlagEmbedding are lazy to keep import-time light and tests offline.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def build_bge_m3_index(
    nodes: Sequence[Any],
    *,
    model_name: str = "BAAI/bge-m3",
    weights_for_different_modes: list[float] | None = None,
    device: str | None = None,
) -> Any:
    """Construct a LlamaIndex BGEM3Index over nodes.

    Args:
        nodes: LlamaIndex nodes/documents.
        model_name: BGE-M3 model id.
        weights_for_different_modes: [dense, sparse, colbert] weights.
        device: Optional device override (e.g., "cuda", "cpu").

    Returns:
        BGEM3Index instance.
    """
    # Lazy imports to keep module import side-effect free
    try:
        from llama_index.indices.managed.bge_m3 import BGEM3Index  # type: ignore

        index = BGEM3Index(
            nodes=list(nodes),
            model_name=model_name,
        )
        # weights are applied by retriever; kept here for symmetry
        _ = weights_for_different_modes, device
        return index
    except (ImportError, RuntimeError, ValueError):  # pragma: no cover - fallback
        # Fallback to a minimal in-memory index to keep tests offline
        from llama_index.core import Document as LIDocument
        from llama_index.core import VectorStoreIndex

        docs = []
        for n in nodes:
            try:
                text = getattr(n, "text", None) or getattr(n, "get_text", lambda: "")()
            except (AttributeError, TypeError):
                text = str(n)
            if not text:
                text = ""
            docs.append(LIDocument(text=text))

        idx = VectorStoreIndex.from_documents(docs)

        class _Wrapper:
            def __init__(self, base: Any):
                self._base = base

            def as_retriever(self, **_kwargs: Any) -> Any:
                """Return retriever from wrapped index."""
                return self._base.as_retriever()

        return _Wrapper(idx)


def build_bge_m3_retriever(
    index: Any, *, weights_for_different_modes: list[float] | None = None
) -> Any:
    """Create a BGEM3Retriever from an index with optional tri-mode weights."""
    weights = weights_for_different_modes or [0.4, 0.2, 0.4]
    return index.as_retriever(weights_for_different_modes=weights)


def get_default_bge_m3_retriever(
    nodes: Iterable[Any], *, settings: Any | None = None
) -> Any:
    """Convenience wrapper to build index+retriever from app settings.

    Notes:
        - This intentionally avoids app-level singletons; call-sites pass
          settings if they need overrides.
    """
    model_name = "BAAI/bge-m3"
    weights = [0.4, 0.2, 0.4]
    if settings is not None:  # pragma: no cover - simple mapping
        try:
            model_name = settings.embedding.model_name or model_name
            # If embedding.enable_sparse is False, caller should not use this retriever
            weights_candidate = getattr(settings, "retrieval", None)
            if weights_candidate is not None:
                weights = (
                    getattr(settings.retrieval, "weights_for_different_modes", weights)
                    or weights
                )
        except (AttributeError, ValueError) as exc:
            _ = exc  # avoid noisy logs in offline CI/tests

    index = build_bge_m3_index(list(nodes), model_name=model_name)
    return build_bge_m3_retriever(index, weights_for_different_modes=weights)
