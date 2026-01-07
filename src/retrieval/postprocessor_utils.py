"""Query engine builders with compatibility fallbacks.

Guards node_postprocessors and legacy kwargs across LlamaIndex versions,
providing tiered fallbacks to maintain tool creation compatibility.
"""

from typing import Any

try:  # pragma: no cover - optional dependency path
    from llama_index.core.query_engine import RetrieverQueryEngine
except ImportError:  # pragma: no cover - optional dependency path
    RetrieverQueryEngine = None

from src.utils.telemetry import log_jsonl


def build_vector_query_engine(
    index: Any, post: list | None, **kwargs: dict[str, Any]
) -> Any:
    """Build vector query engine with fallback if postprocessors fail."""
    if post:
        try:
            return index.as_query_engine(node_postprocessors=post, **kwargs)
        except TypeError:
            log_jsonl(
                {
                    "postproc.fallback": True,
                    "reason": "vector:node_postprocessors_typeerror",
                }
            )
    try:
        return index.as_query_engine(**kwargs)
    except TypeError:
        log_jsonl({"postproc.fallback": True, "reason": "vector:kwargs_typeerror"})
        return index.as_query_engine()


def build_pg_query_engine(
    pg_index: Any, post: list | None, **kwargs: dict[str, Any]
) -> Any:
    """Build PG query engine with fallback if postprocessors fail."""
    if post:
        try:
            return pg_index.as_query_engine(node_postprocessors=post, **kwargs)
        except TypeError:
            log_jsonl(
                {
                    "postproc.fallback": True,
                    "reason": "pg:node_postprocessors_typeerror",
                }
            )
    try:
        return pg_index.as_query_engine(**kwargs)
    except TypeError:
        log_jsonl({"postproc.fallback": True, "reason": "pg:kwargs_typeerror"})
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
    if engine is None:
        raise ImportError(
            "llama_index.core is required to build retriever query engines. "
            "Install it via: pip install docmind_ai_llm[llama]"
        )

    if llm is None:

        class _NoOpLLM:
            def __init__(self):
                md = {"context_window": 2048, "num_output": 256}
                self.metadata = type("_MD", (), md)()

            def predict(self, *_args: Any, **_kwargs: Any) -> str:
                """No-op text prediction to satisfy interface."""
                return ""

            def complete(self, *_args: Any, **_kwargs: Any) -> str:
                """No-op completion to satisfy interface."""
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
            log_jsonl(
                {
                    "postproc.fallback": True,
                    "reason": "retriever:node_postprocessors_typeerror",
                }
            )

    try:
        return engine.from_args(retriever=retriever, llm=llm, **kwargs)
    except TypeError as e:
        last_error = e
        log_jsonl({"postproc.fallback": True, "reason": "retriever:kwargs_typeerror"})

    try:
        return engine.from_args(retriever=retriever, **kwargs)
    except TypeError as e:
        last_error = e

    try:
        return engine.from_args(retriever=retriever)
    except TypeError as e:
        last_error = e
        log_jsonl({"postproc.fallback": True, "reason": "retriever:bare_typeerror"})

    if last_error:
        raise last_error
    return engine.from_args(retriever=retriever)


__all__ = [
    "build_pg_query_engine",
    "build_retriever_query_engine",
    "build_vector_query_engine",
]
