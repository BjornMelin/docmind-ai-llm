"""Synthesis helpers extracted from monolithic tools module."""

from __future__ import annotations

import json
import time
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from loguru import logger

from .constants import MAX_RETRIEVAL_RESULTS, SIMILARITY_THRESHOLD


@tool
def synthesize_results(
    sub_results: str,
    original_query: str,
    _state: Annotated[dict, InjectedState] | None = None,
) -> str:
    """Combine and synthesize results from multiple retrieval operations."""
    try:
        start_time = time.perf_counter()

        # Parse sub-results
        try:
            results_list = (
                json.loads(sub_results) if isinstance(sub_results, str) else sub_results
            )
        except json.JSONDecodeError:
            logger.error("Invalid JSON in sub_results")
            return json.dumps(
                {
                    "documents": [],
                    "error": "Invalid input format",
                    "synthesis_metadata": {},
                }
            )

        all_documents = []
        strategies_used: set[str] = set()
        total_processing_time = 0

        # Collect all documents from sub-results
        for result in results_list:
            if isinstance(result, dict) and "documents" in result:
                all_documents.extend(result["documents"])
                if "strategy_used" in result:
                    strategies_used.add(result["strategy_used"])
                if "processing_time_ms" in result:
                    total_processing_time += result["processing_time_ms"]

        logger.info(
            "Synthesizing %d documents from %d sources",
            len(all_documents),
            len(results_list),
        )

        # Deduplicate documents by content similarity
        unique_documents = []
        seen_content: set[frozenset[str]] = set()

        for doc in all_documents:
            if isinstance(doc, dict):
                # Create content hash for deduplication
                content = doc.get("content", doc.get("text", ""))
                content_words = set(content.lower().split())

                # Check for substantial overlap with existing documents
                is_duplicate = False
                for seen_words in seen_content:
                    if (
                        len(content_words.intersection(seen_words))
                        / max(len(content_words), 1)
                        > SIMILARITY_THRESHOLD
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_documents.append(doc)
                    seen_content.add(frozenset(content_words))

        # Rank documents by relevance to original query
        ranked_documents = _rank_documents_by_relevance(
            unique_documents, original_query
        )

        # Limit to top results
        max_results = MAX_RETRIEVAL_RESULTS
        final_documents = ranked_documents[:max_results]

        processing_time = time.perf_counter() - start_time

        synthesis_metadata = {
            "original_count": len(all_documents),
            "after_deduplication": len(unique_documents),
            "final_count": len(final_documents),
            "strategies_used": list(strategies_used),
            "deduplication_ratio": round(
                len(unique_documents) / max(len(all_documents), 1), 2
            ),
            "processing_time_ms": round(processing_time * 1000, 2),
            "total_retrieval_time_ms": total_processing_time,
        }

        result_data = {
            "documents": final_documents,
            "synthesis_metadata": synthesis_metadata,
            "original_query": original_query,
        }

        logger.info("Synthesis complete: %d final documents", len(final_documents))
        return json.dumps(result_data, default=str)

    except (RuntimeError, ValueError, AttributeError) as e:
        logger.error("Result synthesis failed: %s", e)
        return json.dumps({"documents": [], "error": str(e), "synthesis_metadata": {}})


def _rank_documents_by_relevance(documents: list[dict], query: str) -> list[dict]:
    """Rank documents by relevance to query using simple scoring."""
    query_words = set(query.lower().split())

    scored_docs = []
    for doc in documents:
        content = doc.get("content", doc.get("text", ""))
        content_words = set(content.lower().split())

        # Simple relevance scoring
        word_overlap = len(query_words.intersection(content_words))
        total_words = len(content_words)

        # Calculate relevance score
        relevance_score = word_overlap / max(len(query_words), 1)
        if total_words > 0:
            relevance_score += (word_overlap / total_words) * 0.5

        # Boost score if existing score is available
        existing_score = doc.get("score", 1.0)
        final_score = relevance_score * existing_score

        doc_copy = doc.copy()
        doc_copy["relevance_score"] = final_score
        scored_docs.append(doc_copy)

    # Sort by relevance score descending
    scored_docs.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return scored_docs
