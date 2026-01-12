"""Retrieval tools and helpers extracted from monolithic module."""

from __future__ import annotations

import importlib
import json
import time
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolRuntime
from llama_index.core import Document
from loguru import logger

from .constants import (
    CONTENT_KEY_LENGTH,
    MAX_RETRIEVAL_RESULTS,
    VARIANT_QUERY_LIMIT,
)

_LOG_QUERY_MAX_LEN = 160


def _safe_query_for_log(query: str, *, max_len: int = _LOG_QUERY_MAX_LEN) -> str:
    """Return a normalized, truncated query string for logs.

    Queries can contain sensitive user content. Keep logs useful while reducing
    exposure by collapsing whitespace and truncating to a bounded length.
    """
    collapsed = " ".join(str(query).split())
    if len(collapsed) <= max_len:
        return collapsed
    return f"{collapsed[:max_len]}…"


@tool
def retrieve_documents(
    query: str,
    strategy: str = "hybrid",
    use_dspy: bool = True,
    use_graphrag: bool = False,
    state: Annotated[dict, InjectedState] | None = None,
    runtime: ToolRuntime | None = None,
) -> str:
    """Execute document retrieval using specified strategy and optimizations."""
    try:
        start_time = time.perf_counter()

        vector_index, kg_index, retriever = _extract_indexes(state, runtime)
        if vector_index is None and retriever is None and kg_index is None:
            return json.dumps(
                {
                    "documents": [],
                    "error": "No retrieval tools available",
                    "strategy_used": strategy,
                    "query_optimized": query,
                },
                default=str,
            )

        # Optional aggregator fast-path for resilience testing scenarios
        st = state if isinstance(state, dict) else {}
        tool_count = sum(1 for x in (vector_index, kg_index, retriever) if x)
        if st and (
            any(
                st.get(k)
                for k in (
                    "error_isolation_enabled",
                    "continue_on_tool_failure",
                    "resilience_mode_enabled",
                    "partial_results_acceptable",
                    "auto_cleanup_on_memory_pressure",
                )
            )
            or tool_count >= 2
        ):
            collected = _collect_via_tool_factory(
                query, vector_index, kg_index, retriever
            )
            if collected:
                return json.dumps(
                    {
                        "documents": collected[:MAX_RETRIEVAL_RESULTS],
                        "strategy_used": "mixed",
                        "query_original": query,
                        "query_optimized": query,
                        "document_count": len(collected),
                        "processing_time_ms": round(
                            (time.perf_counter() - start_time) * 1000, 2
                        ),
                        "partial_results": True,
                    },
                    default=str,
                )

        # Use detailed path with query optimization and strategy-specific tools

        # Optimize queries (DSPy or fallback)
        primary_query, query_variants = _optimize_queries(query, use_dspy)
        queries_to_process = [primary_query, *query_variants[:VARIANT_QUERY_LIMIT]]

        # Run GraphRAG if requested
        documents: list[dict] = []
        strategy_used = strategy
        if strategy == "graphrag" and use_graphrag and kg_index:
            docs, fallback = _run_graphrag(kg_index, queries_to_process)
            documents.extend(docs)
            if fallback:
                strategy_used = "hybrid"

        # Run hybrid/vector if needed
        if strategy in ["hybrid", "vector"] or (
            strategy == "graphrag" and not documents
        ):
            docs, last_used, error_json = _run_vector_hybrid(
                strategy, retriever, vector_index, queries_to_process, primary_query
            )
            if error_json:
                return error_json
            documents.extend(docs)
            strategy_used = last_used or strategy_used

        # Contextual recall: if user references prior context and retrieval returned
        # nothing, reuse most recent sources from persisted state. This enables
        # "what does that chart show?" flows across reloads without storing images.
        if not documents and _looks_contextual(query):
            recalled = _recall_recent_sources(st if isinstance(st, dict) else None)
            if recalled:
                documents.extend(recalled)
                strategy_used = f"{strategy_used}+recall"

        # Deduplicate
        documents = _deduplicate_documents(documents)

        processing_time = time.perf_counter() - start_time
        result_data = {
            "documents": documents,
            "strategy_used": strategy_used,
            "query_original": query,
            "query_optimized": primary_query,
            "document_count": len(documents),
            "processing_time_ms": round(processing_time * 1000, 2),
            "dspy_used": use_dspy,
            "graphrag_used": use_graphrag and strategy == "graphrag",
        }
        return json.dumps(result_data, default=str)

    except (OSError, RuntimeError, ValueError, AttributeError) as e:
        logger.error("Document retrieval failed: {}", e)
        return json.dumps(
            {
                "documents": [],
                "error": str(e),
                "strategy_used": strategy,
                "query_optimized": query,
            },
            default=str,
        )


def _extract_indexes(state: dict | None, runtime: ToolRuntime | None):
    # Prefer runtime context (not persisted) for heavy objects like indexes.
    runtime_ctx = runtime.context if runtime is not None else None
    if isinstance(runtime_ctx, dict):
        v = runtime_ctx.get("vector")
        kg = runtime_ctx.get("kg")
        r = runtime_ctx.get("retriever")
        if any(x is not None for x in (v, kg, r)):
            return v, kg, r

    tools_data = state.get("tools_data") if state else None
    if not tools_data:
        logger.warning("No tools data available in state, using fallback")
        return None, None, None
    return tools_data.get("vector"), tools_data.get("kg"), tools_data.get("retriever")


## Fast-path via ToolFactory.create_tools_from_indexes was removed for clarity
## and testability. Use explicit tools for each strategy instead.


def _optimize_queries(query: str, use_dspy: bool) -> tuple[str, list[str]]:
    optimized = {"refined": query, "variants": []}
    if use_dspy:
        try:
            from src.dspy_integration import DSPyLlamaIndexRetriever

            optimized = DSPyLlamaIndexRetriever.optimize_query(query)
            logger.debug(
                "DSPy optimization: '{}' -> '{}'",
                _safe_query_for_log(query),
                _safe_query_for_log(optimized["refined"]),
            )
        except ImportError:
            logger.warning(
                "DSPy integration not available - using fallback optimization"
            )
            if len(query.split()) < 3:
                optimized = {
                    "refined": f"Find documents about {query}",
                    "variants": [f"What is {query}?", f"Explain {query}"],
                }
        else:
            # Ensure short queries still generate variants if DSPy returns none
            if not optimized.get("variants") and len(query.split()) < 3:
                optimized = {
                    "refined": optimized.get("refined", query),
                    "variants": [f"What is {query}?", f"Explain {query}"],
                }
    return optimized["refined"], optimized.get("variants", [])


def _run_graphrag(kg_index: Any, queries: list[str]) -> tuple[list[dict], bool]:
    documents: list[dict] = []
    fallback = False
    for q in queries:
        search_tool = _get_tool_factory().create_kg_search_tool(kg_index)
        if not search_tool:
            continue
        try:
            result = search_tool.call(q)
            new_docs = _parse_tool_result(result)
            documents.extend(new_docs)
            logger.debug(
                "GraphRAG retrieved {} documents for query: {}",
                len(new_docs),
                _safe_query_for_log(q),
            )
        except (OSError, RuntimeError, ValueError, AttributeError) as e:
            logger.warning(
                "GraphRAG failed for query '{}': {}", _safe_query_for_log(q), e
            )
            fallback = True
            break
    return documents, fallback


def _run_vector_hybrid(
    strategy: str,
    retriever: Any,
    vector_index: Any,
    queries: list[str],
    primary_query: str,
) -> tuple[list[dict], str | None, str | None]:
    documents: list[dict] = []
    strategy_used: str | None = None
    for idx, q in enumerate(queries):
        if strategy == "hybrid" and retriever:
            search_tool = _get_tool_factory().create_hybrid_search_tool(retriever)
            strategy_used = "hybrid_fusion"
        elif vector_index:
            if strategy == "hybrid":
                search_tool = _get_tool_factory().create_hybrid_vector_tool(
                    vector_index
                )
                strategy_used = "hybrid_vector"
            else:
                search_tool = _get_tool_factory().create_vector_search_tool(
                    vector_index
                )
                strategy_used = "vector"
        else:
            logger.error("No vector index available for retrieval")
            return (
                [],
                None,
                json.dumps(
                    {
                        "documents": [],
                        "error": "No vector index available",
                        "strategy_used": strategy,
                        "query_optimized": primary_query,
                    },
                    default=str,
                ),
            )
        try:
            result = search_tool.call(q)
            new_docs = _parse_tool_result(result)
            documents.extend(new_docs)
            logger.debug(
                "{} retrieved {} documents for query: {}",
                strategy_used,
                len(new_docs),
                _safe_query_for_log(q),
            )
            # If this is the primary query and hybrid returned empty, fallback to vector
            if (
                strategy in ["hybrid"]
                and idx == 0
                and retriever
                and vector_index
                and not new_docs
            ):
                try:
                    from .telemetry import log_event  # local import
                except ImportError:  # pragma: no cover - defensive
                    log_event = None
                try:
                    vtool = _get_tool_factory().create_vector_search_tool(vector_index)
                    vres = vtool.call(primary_query)
                    vdocs = _parse_tool_result(vres)
                    if vdocs:
                        documents.extend(vdocs)
                        strategy_used = "vector"
                        if log_event:
                            log_event(
                                "hybrid_fallback",
                                reason="empty_results",
                                query=_safe_query_for_log(primary_query),
                            )
                except (OSError, RuntimeError, ValueError, AttributeError) as fe:
                    logger.warning("Vector fallback failed: {}", fe)
        except (OSError, RuntimeError, ValueError, AttributeError) as e:
            logger.error(
                "Retrieval failed for query '{}': {}", _safe_query_for_log(q), e
            )
    return documents, strategy_used, None


def _deduplicate_documents(documents: list[dict]) -> list[dict]:
    if not documents:
        return []
    seen_content: dict[str, dict] = {}
    for doc in documents:
        content = str(doc.get("content", "") or "")
        meta = doc.get("metadata") if isinstance(doc, dict) else None
        meta = meta if isinstance(meta, dict) else {}
        # For multimodal nodes, content may be empty; prefer stable ids.
        stable = meta.get("page_id") or meta.get("chunk_id") or meta.get("doc_id")
        stable = stable or meta.get("document_id")
        stable = stable or meta.get("source")
        content_key = (content[:CONTENT_KEY_LENGTH] if content else str(stable or ""))[
            :CONTENT_KEY_LENGTH
        ]
        if content_key not in seen_content or doc.get("score", 0) > seen_content[
            content_key
        ].get("score", 0):
            seen_content[content_key] = doc
    return list(seen_content.values())[:MAX_RETRIEVAL_RESULTS]


def _looks_contextual(query: str) -> bool:
    import re

    pattern = (
        r"\b(this|that|they|them|above|previous|chart|figure|diagram|table|"
        r"image|photo)\b"
        r"|\bit\b(?=\s+(?:is|was|seems|refers|referring|about|in|on|of))"
    )
    return re.search(pattern, str(query), flags=re.IGNORECASE) is not None


def _recall_recent_sources(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract the most recent persisted sources from prior turns (best-effort)."""
    if not isinstance(state, dict):
        return []
    # Prefer synthesis_result documents (closest to what user saw).
    sr = state.get("synthesis_result")
    if isinstance(sr, dict):
        docs = sr.get("documents")
        if isinstance(docs, list):
            return [_sanitize_document_dict(d) for d in docs if isinstance(d, dict)]

    rr = state.get("retrieval_results")
    if isinstance(rr, list):
        for item in reversed(rr):
            if isinstance(item, dict):
                docs = item.get("documents")
                if isinstance(docs, list):
                    return [
                        _sanitize_document_dict(d) for d in docs if isinstance(d, dict)
                    ]
    return []


def _sanitize_document_dict(doc: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a persisted/recalled document dict for final-release invariants."""

    def _sanitize_metadata(meta: Any) -> dict[str, Any]:
        if not isinstance(meta, dict):
            return {}
        # Never persist raw paths or blobs in agent-visible sources (final-release).
        drop = {
            "image_base64",
            "thumbnail_base64",
            "image_path",
            "thumbnail_path",
            "source_path",
            "file_path",
            "path",
        }
        sanitized = {k: v for k, v in meta.items() if k not in drop}
        # Defensive: drop any future *_base64 keys.
        sanitized = {
            k: v for k, v in sanitized.items() if not str(k).endswith("_base64")
        }
        # `source` is frequently used by upstream libraries to carry a path/URI.
        # Preserve only a safe basename when it looks path-like.
        src = sanitized.get("source")
        if isinstance(src, str) and (
            "/" in src or "\\" in src or src.startswith("file:")
        ):
            from pathlib import Path

            sanitized["source"] = Path(src).name
        return sanitized

    cleaned = dict(doc)
    # Drop forbidden top-level keys too (some tool stacks return flat dicts).
    for key in list(cleaned.keys()):
        if key in {
            "image_base64",
            "thumbnail_base64",
            "image_path",
            "thumbnail_path",
            "source_path",
            "file_path",
            "path",
        } or str(key).endswith("_base64"):
            cleaned.pop(key, None)

    cleaned["metadata"] = _sanitize_metadata(cleaned.get("metadata"))
    # If a tool returns a top-level `source`, apply the same policy.
    src = cleaned.get("source")
    if isinstance(src, str) and ("/" in src or "\\" in src or src.startswith("file:")):
        from pathlib import Path

        cleaned["source"] = Path(src).name
    return cleaned


def _parse_tool_result(result: Any) -> list[dict[str, Any]]:
    """Parse tool result to extract document list."""
    if isinstance(result, str):
        # Tool returned text response — create a minimal document entry
        return [
            {"content": result, "metadata": {"source": "tool_response"}, "score": 1.0}
        ]
    if hasattr(result, "source_nodes"):
        nodes = getattr(result, "source_nodes", None)
        if nodes is None:
            nodes = []
        elif not isinstance(nodes, list):
            try:
                nodes = list(nodes)
            except TypeError:
                nodes = []
        docs: list[dict[str, Any]] = []
        for nws in nodes:
            node = getattr(nws, "node", None) or getattr(nws, "document", None) or nws
            score = float(getattr(nws, "score", 0.0) or 0.0)
            text = ""
            try:
                text_attr = getattr(node, "text", None)
                if text_attr is not None:
                    text = str(text_attr)
                else:
                    get_content = getattr(node, "get_content", None)
                    text = str(get_content()) if callable(get_content) else str(node)
            except Exception:
                text = str(node)
            meta = _sanitize_document_dict(
                {"metadata": getattr(node, "metadata", {}) or {}}
            ).get("metadata", {})
            docs.append({"content": text, "metadata": meta, "score": score})
        if docs:
            return docs
    if hasattr(result, "response"):
        # LlamaIndex response object
        return [
            {
                "content": result.response,
                "metadata": _sanitize_document_dict(
                    {"metadata": getattr(result, "metadata", {})}
                ).get("metadata", {}),
                "score": 1.0,
            }
        ]
    if isinstance(result, list):
        # List of documents
        documents: list[dict[str, Any]] = []
        for item in result:
            if isinstance(item, Document):
                documents.append(
                    {
                        "content": item.text,
                        "metadata": _sanitize_document_dict(
                            {"metadata": item.metadata}
                        ).get("metadata", {}),
                        "score": getattr(item, "score", 1.0),
                    }
                )
            elif isinstance(item, dict):
                documents.append(_sanitize_document_dict(item))
        return documents
    # Fallback - convert to string
    return [{"content": str(result), "metadata": {"source": "unknown"}, "score": 1.0}]


def _collect_via_tool_factory(
    query: str, vector_index: Any, kg_index: Any, retriever: Any
) -> list[dict]:
    try:
        tool_list = _get_tool_factory().create_tools_from_indexes(
            vector_index=vector_index, kg_index=kg_index, retriever=retriever
        )
        collected: list[dict] = []
        for t in tool_list or []:
            try:
                res = t.invoke(query) if hasattr(t, "invoke") else t.call(query)
                collected.extend(_parse_tool_result(res))
            except Exception as te:
                logger.error("Tool execution failed: {}", te)
                logger.warning("Partial failure: continuing with other tools")
                continue
        return collected
    except Exception:
        logger.debug("Tool collection path failed; using detailed path", exc_info=True)
        return []


def _get_tool_factory():
    """Return the canonical ToolFactory class.

    Production-only resolution: always import from src.agents.tool_factory.
    Tests should patch `src.agents.tool_factory.ToolFactory` directly.
    """
    return importlib.import_module("src.agents.tool_factory").ToolFactory
