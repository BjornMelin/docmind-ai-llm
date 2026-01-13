"""Unit tests for retrieval tool persistence boundaries (final-release).

Ensures agent-visible sources never persist runtime-local paths and that
contextual queries can recall the most recent sources from persisted state.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.agents.tools import retrieval as retrieval_tool

pytestmark = pytest.mark.unit


def test_parse_tool_result_strips_runtime_paths_from_metadata() -> None:
    node = TextNode(text="x", id_="n1")
    node.metadata.update(
        {
            "doc_id": "doc-1",
            "modality": "pdf_page_image",
            "image_artifact_id": "sha256",
            "image_artifact_suffix": ".webp",
            "image_path": "/abs/path.webp",
            "thumbnail_path": "/abs/thumb.webp",
            "source_path": "/abs/source.pdf",
        }
    )
    resp = SimpleNamespace(source_nodes=[NodeWithScore(node=node, score=1.0)])
    docs = retrieval_tool._parse_tool_result(resp)
    assert docs
    assert isinstance(docs[0], dict)
    meta = docs[0].get("metadata") or {}
    assert "image_path" not in meta
    assert "thumbnail_path" not in meta
    assert "source_path" not in meta
    assert meta.get("image_artifact_id") == "sha256"


def test_contextual_recall_returns_last_sources_when_present() -> None:
    state = {
        "retrieval_results": [
            {
                "documents": [
                    {
                        "content": "",
                        "metadata": {
                            "modality": "pdf_page_image",
                            "image_artifact_id": "a",
                            "thumbnail_artifact_id": "b",
                        },
                        "score": 1.0,
                    }
                ]
            }
        ]
    }
    recalled = retrieval_tool._recall_recent_sources(state)
    assert recalled
    assert recalled[0]["metadata"]["image_artifact_id"] == "a"


def test_contextual_recall_prefers_synthesis_result_documents() -> None:
    state = {
        "synthesis_result": {
            "documents": [
                {
                    "content": "",
                    "metadata": {
                        "modality": "pdf_page_image",
                        "image_artifact_id": "s",
                        "thumbnail_artifact_id": "t",
                    },
                    "score": 1.0,
                }
            ]
        },
        "retrieval_results": [
            {
                "documents": [
                    {
                        "content": "",
                        "metadata": {"image_artifact_id": "r"},
                        "score": 1.0,
                    }
                ]
            }
        ],
    }
    recalled = retrieval_tool._recall_recent_sources(state)
    assert recalled
    assert recalled[0]["metadata"]["image_artifact_id"] == "s"


def test_looks_contextual_matches_simple_pronoun_question() -> None:
    assert retrieval_tool._looks_contextual("What does that chart show?") is True


def test_looks_contextual_returns_false_for_standalone_query() -> None:
    assert retrieval_tool._looks_contextual("What is machine learning?") is False


def test_looks_contextual_returns_false_for_empty_query() -> None:
    assert retrieval_tool._looks_contextual("") is False
