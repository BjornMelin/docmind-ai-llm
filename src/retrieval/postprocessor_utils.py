"""Query engine builders with compatibility fallbacks.

Guards node_postprocessors and legacy kwargs across LlamaIndex versions,
providing tiered fallbacks to maintain tool creation compatibility.
"""

from typing import Any

from llama_index.core.query_engine import RetrieverQueryEngine


def build_vector_query_engine(
    index: Any, post: list | None, **kwargs: dict[str, Any]
) -> Any:
    """Build vector query engine with fallback if postprocessors fail."""
    if post:
        try:
            return index.as_query_engine(node_postprocessors=post, **kwargs)
        except TypeError:
            pass
    try:
        return index.as_query_engine(**kwargs)
    except TypeError:
        return index.as_query_engine()


def build_pg_query_engine(
    pg_index: Any, post: list | None, **kwargs: dict[str, Any]
) -> Any:
    """Build PG query engine with fallback if postprocessors fail."""
    if post:
        try:
            return pg_index.as_query_engine(node_postprocessors=post, **kwargs)
        except TypeError:
            pass
    try:
        return pg_index.as_query_engine(**kwargs)
    except TypeError:
        return pg_index.as_query_engine()


def build_retriever_query_engine(
    retriever: Any,
    post: list | None,
    *,
    llm: Any | None = None,
    engine_cls: Any | None = None,
    **kwargs: dict[str, Any],
) -> Any:
    """Build retriever query engine with fallback if postprocessors fail."""
    engine = engine_cls or RetrieverQueryEngine

    if llm is None:

        class _NoOpLLM:
            def __init__(self):
                md = {"context_window": 2048, "num_output": 256}
                self.metadata = type("_MD", (), md)()

            def predict(self, *_args: Any, **_kwargs: Any) -> str:
                return ""

            def complete(self, *_args: Any, **_kwargs: Any) -> str:
                return ""

        llm = _NoOpLLM()

    last_error = None

    if post:
        try:
            return engine.from_args(
                retriever=retriever, llm=llm, node_postprocessors=post, **kwargs
            )
        except TypeError as e:
            last_error = e

    try:
        return engine.from_args(retriever=retriever, llm=llm, **kwargs)
    except TypeError as e:
        last_error = e

    try:
        return engine.from_args(retriever=retriever, **kwargs)
    except TypeError as e:
        last_error = e

    try:
        return engine.from_args(retriever=retriever)
    except TypeError as e:
        last_error = e

    if last_error:
        raise last_error
    return engine.from_args(retriever=retriever)


__all__ = [
    "build_pg_query_engine",
    "build_retriever_query_engine",
    "build_vector_query_engine",
]
