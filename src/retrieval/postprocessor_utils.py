"""Query engine builders for router tools."""

from collections.abc import Sequence
from typing import Any


def build_vector_query_engine(
    index: Any, post: Sequence[Any] | None, **kwargs: Any
) -> Any:
    """Build a vector query engine, optionally applying node postprocessors."""
    if post:
        return index.as_query_engine(node_postprocessors=list(post), **kwargs)
    return index.as_query_engine(**kwargs)


def build_pg_query_engine(
    pg_index: Any, post: Sequence[Any] | None, **kwargs: Any
) -> Any:
    """Build a property graph query engine, optionally applying postprocessors."""
    if post:
        return pg_index.as_query_engine(node_postprocessors=list(post), **kwargs)
    return pg_index.as_query_engine(**kwargs)


def build_retriever_query_engine(
    retriever: Any,
    post: Sequence[Any] | None,
    *,
    llm: Any | None = None,
    engine_cls: Any,
    **kwargs: Any,
) -> Any:
    """Build a retriever query engine, optionally applying postprocessors."""
    engine_kwargs: dict[str, Any] = {**kwargs, "retriever": retriever, "llm": llm}
    if post:
        engine_kwargs["node_postprocessors"] = list(post)
    return engine_cls.from_args(**engine_kwargs)


__all__ = [
    "build_pg_query_engine",
    "build_retriever_query_engine",
    "build_vector_query_engine",
]
