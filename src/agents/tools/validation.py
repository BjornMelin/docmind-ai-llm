"""Response validation tool extracted from monolithic module."""

from __future__ import annotations

import json
import time
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from loguru import logger

from .constants import (
    ACCEPT_CONFIDENCE_THRESHOLD,
    CONFIDENCE_REDUCTION_COHERENCE,
    CONFIDENCE_REDUCTION_HALLUCINATION,
    CONFIDENCE_REDUCTION_INCOMPLETE,
    CONFIDENCE_REDUCTION_NO_SOURCE,
    CONFIDENCE_REDUCTION_NO_SOURCES,
    FIRST_N_SOURCES_CHECK,
    MAX_AVG_SENTENCE_LENGTH,
    MAX_ISSUES_FOR_ACCEPT,
    MAX_SENTENCE_WORD_OVERLAP,
    MIN_AVG_SENTENCE_LENGTH,
    MIN_RESPONSE_LENGTH,
    REFINE_CONFIDENCE_THRESHOLD,
    RELEVANCE_THRESHOLD,
    VALIDATION_CONFIDENCE_THRESHOLD,
)


@tool
def validate_response(
    query: str,
    response: str,
    sources: str,
    _state: Annotated[dict, InjectedState] | None = None,
) -> str:
    """Validate generated response quality and integrity."""
    try:
        start_time = time.perf_counter()

        source_docs = _parse_sources(sources)

        issues: list[dict] = []
        confidence = 1.0

        # Checks with confidence adjustments
        issue, factor = _check_length(response)
        if issue:
            issues.append(issue)
            confidence *= factor

        issue, factor = _check_source_attribution(source_docs, response)
        if issue:
            issues.append(issue)
            confidence *= factor

        issue, factor = _check_hallucination(response)
        if issue:
            issues.append(issue)
            confidence *= factor

        issue, factor = _check_relevance(query, response)
        if issue:
            issues.append(issue)
            confidence *= factor

        issue, factor = _check_coherence(response)
        if issue:
            issues.append(issue)
            confidence *= factor

        # Determine suggested action
        if (
            confidence >= ACCEPT_CONFIDENCE_THRESHOLD
            and len(issues) <= MAX_ISSUES_FOR_ACCEPT
        ):
            suggested_action = "accept"
        elif confidence >= REFINE_CONFIDENCE_THRESHOLD:
            suggested_action = "refine"
        else:
            suggested_action = "regenerate"

        processing_time = time.perf_counter() - start_time

        validation_result = {
            "valid": confidence >= VALIDATION_CONFIDENCE_THRESHOLD,
            "confidence": round(confidence, 2),
            "issues": issues,
            "suggested_action": suggested_action,
            "processing_time_ms": round(processing_time * 1000, 2),
            "source_count": len(source_docs),
            "response_length": len(response),
            "issue_count": len(issues),
        }

        logger.info(
            "Response validation: {:.2f} confidence, {} action",
            confidence,
            suggested_action,
        )
        return json.dumps(validation_result)

    except (RuntimeError, ValueError, AttributeError) as e:
        logger.error("Response validation failed: {}", e)
        return json.dumps(
            {
                "valid": False,
                "confidence": 0.0,
                "issues": [
                    {
                        "type": "validation_error",
                        "severity": "high",
                        "description": str(e),
                    }
                ],
                "suggested_action": "regenerate",
                "error": str(e),
            }
        )


def _parse_sources(sources: str):
    try:
        src = json.loads(sources) if isinstance(sources, str) else sources
        if isinstance(src, dict) and "documents" in src:
            return src["documents"]
        return src or []
    except json.JSONDecodeError:
        return []


def _check_length(response: str):
    if len(response.strip()) < MIN_RESPONSE_LENGTH:
        return (
            {
                "type": "incomplete_response",
                "severity": "medium",
                "description": "Response appears too brief for the query complexity",
            },
            CONFIDENCE_REDUCTION_INCOMPLETE,
        )
    return None, 1.0


def _check_source_attribution(source_docs: list | None, response: str):
    if not source_docs:
        return (
            {
                "type": "no_sources",
                "severity": "high",
                "description": "No source documents provided for validation",
            },
            CONFIDENCE_REDUCTION_NO_SOURCES,
        )

    response_words = set(response.lower().split())
    for doc in source_docs[:FIRST_N_SOURCES_CHECK]:
        if isinstance(doc, dict):
            doc_content = doc.get("content", doc.get("text", ""))
            doc_words = set(doc_content.lower().split())
            if len(doc_words.intersection(response_words)) >= MAX_SENTENCE_WORD_OVERLAP:
                return None, 1.0

    return (
        {
            "type": "missing_source",
            "severity": "medium",
            "description": "Response does not appear to reference provided sources",
        },
        CONFIDENCE_REDUCTION_NO_SOURCE,
    )


def _check_hallucination(response: str):
    hallucination_indicators = [
        "I cannot find",
        "No information available",
        "According to my knowledge",
        "I don't have access",
        "Based on my training",
    ]
    if any(indicator in response for indicator in hallucination_indicators):
        return (
            {
                "type": "potential_hallucination",
                "severity": "high",
                "description": (
                    "Response contains phrases suggesting use of training data "
                    "over sources"
                ),
            },
            CONFIDENCE_REDUCTION_HALLUCINATION,
        )
    return None, 1.0


def _check_relevance(query: str, response: str):
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    relevance_overlap = len(query_words.intersection(response_words)) / len(query_words)
    if relevance_overlap < RELEVANCE_THRESHOLD:
        return (
            {
                "type": "low_relevance",
                "severity": "medium",
                "description": "Response may not adequately address the query",
            },
            CONFIDENCE_REDUCTION_INCOMPLETE,
        )
    return None, 1.0


def _check_coherence(response: str):
    sentences = response.split(". ")
    if len(sentences) > 1:
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if (
            avg_sentence_length < MIN_AVG_SENTENCE_LENGTH
            or avg_sentence_length > MAX_AVG_SENTENCE_LENGTH
        ):
            return (
                {
                    "type": "coherence_issue",
                    "severity": "low",
                    "description": "Response may have unusual sentence structure",
                },
                CONFIDENCE_REDUCTION_COHERENCE,
            )
    return None, 1.0
