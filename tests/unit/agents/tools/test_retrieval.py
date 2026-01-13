"""Unit tests for retrieval.retrieve_documents and parsing boundaries.

Split and adapted from legacy tests/unit/agents/test_tools.py.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.agents.tools.retrieval import retrieve_documents

pytestmark = pytest.mark.unit


class TestRetrieveDocuments:
    """Retrieval engine behavior across strategies and fallbacks."""

    def test_retrieve_documents_no_tools_data(self):
        """Test retrieval behavior when no tools data is available."""
        result_json = retrieve_documents.invoke({"query": "test query", "state": {}})
        result = json.loads(result_json)

        assert result["documents"] == []
        assert "error" in result
        assert result["strategy_used"] == "hybrid"

    def test_retrieve_documents_no_state(self):
        """Test retrieval behavior when state parameter is omitted."""
        result_json = retrieve_documents.invoke({"query": "test query"})
        result = json.loads(result_json)
        assert result["documents"] == []
        assert "error" in result

    @patch("src.dspy_integration.DSPyLlamaIndexRetriever")
    def test_retrieve_documents_with_dspy_optimization(self, mock_dspy):
        """Test DSPy query optimization integration with retrieval."""
        mock_dspy.optimize_query.return_value = {
            "refined": "optimized test query",
            "variants": ["variant 1", "variant 2"],
        }
        mock_state = {"tools_data": {"vector": MagicMock(), "retriever": None}}

        with patch("src.agents.tool_factory.ToolFactory") as mock_factory:
            mock_tool = MagicMock()
            mock_tool.call.return_value = [{"content": "test doc", "score": 0.9}]
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result_json = retrieve_documents.invoke({
                "query": "test query",
                "strategy": "vector",
                "use_dspy": True,
                "state": mock_state,
            })
            result = json.loads(result_json)

        assert result["query_original"] == "test query"
        assert result["query_optimized"] == "optimized test query"
        assert result["dspy_used"]
        assert len(result["documents"]) > 0

    def test_retrieve_documents_fallback_optimization(self):
        """Test fallback optimization when DSPy import fails."""
        mock_state = {"tools_data": {"vector": MagicMock()}}

        import builtins as _builtins

        real_import = _builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name.startswith("src.dspy_integration"):
                raise ImportError("Module not found")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(_builtins, "__import__", side_effect=_fake_import),
            patch("src.agents.tool_factory.ToolFactory") as mock_factory,
        ):
            mock_tool = MagicMock()
            mock_tool.call.return_value = [{"content": "test doc"}]
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result_json = retrieve_documents.invoke({
                "query": "AI",
                "use_dspy": True,
                "state": mock_state,
            })
            result = json.loads(result_json)

        assert result["query_optimized"].lower().startswith("find documents about")
        assert result["dspy_used"]

    def test_retrieve_documents_deduplication(self):
        """Test deduplication keeps higher-scoring duplicate documents."""
        mock_state = {"tools_data": {"vector": MagicMock()}}
        duplicate_docs = [
            {"content": "Same content here", "score": 0.9},
            {"content": "Same content here", "score": 0.8},
            {"content": "Different content", "score": 0.7},
        ]

        with patch("src.agents.tool_factory.ToolFactory") as mock_factory:
            mock_tool = MagicMock()
            mock_tool.call.return_value = duplicate_docs
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result_json = retrieve_documents.invoke({
                "query": "test query",
                "strategy": "vector",
                "state": mock_state,
                "use_dspy": False,
            })
            result = json.loads(result_json)

        assert len(result["documents"]) == 2
        assert result["documents"][0]["score"] == 0.9

    def test_tool_collection_path_failure_fallbacks(self):
        """Test fallback behavior when tool factory creation fails."""
        mock_state = {"tools_data": {"vector": MagicMock()}}

        with (
            patch(
                "src.agents.tool_factory.ToolFactory.create_tools_from_indexes",
                side_effect=MemoryError("no mem"),
            ),
            patch("src.agents.tool_factory.ToolFactory") as mock_factory,
        ):
            mock_tool = MagicMock()
            mock_tool.call.return_value = []
            mock_factory.create_vector_search_tool.return_value = mock_tool

            result_json = retrieve_documents.invoke({"query": "q", "state": mock_state})
            result = json.loads(result_json)

        assert "documents" in result
        assert result.get("document_count", 0) >= 0


class TestParsingBoundaries:
    """Boundary parsing checks for various tool result shapes."""

    def test_retrieve_parsing_string_result(self):
        """Test parsing of simple string responses into document format."""
        mock_state = {"tools_data": {"vector": Mock()}}
        with patch("src.agents.tool_factory.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = "Simple string response"
            mock_factory.create_vector_search_tool.return_value = mock_tool

            data = json.loads(
                retrieve_documents.invoke({
                    "query": "q",
                    "strategy": "vector",
                    "state": mock_state,
                })
            )

        assert len(data["documents"]) == 1
        assert data["documents"][0]["content"] == "Simple string response"
        assert data["documents"][0]["score"] == 1.0

    def test_retrieve_parsing_llamaindex_like_object(self):
        """Test parsing of LlamaIndex-like objects with response and metadata."""
        mock_state = {"tools_data": {"vector": Mock()}}
        with patch("src.agents.tool_factory.ToolFactory") as mock_factory:
            mock_result = Mock()
            mock_result.response = "LlamaIndex response"
            mock_result.metadata = {"source": "test"}

            mock_tool = Mock()
            mock_tool.call.return_value = mock_result
            mock_factory.create_vector_search_tool.return_value = mock_tool

            data = json.loads(
                retrieve_documents.invoke({
                    "query": "q",
                    "strategy": "vector",
                    "state": mock_state,
                })
            )

        assert len(data["documents"]) == 1
        assert data["documents"][0]["content"] == "LlamaIndex response"
        assert data["documents"][0]["metadata"]["source"] == "test"

    def test_retrieve_parsing_document_list(self):
        """Test parsing of LlamaIndex Document objects list."""
        from llama_index.core import Document

        docs = [
            Document(text="First document", metadata={"id": 1}),
            Document(text="Second document", metadata={"id": 2}),
        ]

        mock_state = {"tools_data": {"vector": Mock()}}
        with patch("src.agents.tool_factory.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = docs
            mock_factory.create_vector_search_tool.return_value = mock_tool

            data = json.loads(
                retrieve_documents.invoke({
                    "query": "q",
                    "strategy": "vector",
                    "state": mock_state,
                })
            )

        assert len(data["documents"]) == 2
        assert data["documents"][0]["content"] == "First document"
        assert data["documents"][1]["content"] == "Second document"

    def test_retrieve_parsing_dict_list_passthrough(self):
        """Test parsing of document dictionaries enforces persistence hygiene."""
        inputs = [
            {
                "content": "First doc",
                "score": 0.9,
                "metadata": {
                    "doc_id": "doc-1",
                    "source": "/abs/source.pdf",
                    "image_path": "/abs/path.webp",
                    "thumbnail_base64": "AAAA",
                },
                "image_base64": "BBBB",
            },
            {
                "content": "Second doc",
                "score": 0.8,
                "metadata": {"doc_id": "doc-2"},
            },
        ]

        mock_state = {"tools_data": {"vector": Mock()}}
        with patch("src.agents.tool_factory.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = inputs
            mock_factory.create_vector_search_tool.return_value = mock_tool

            data = json.loads(
                retrieve_documents.invoke({
                    "query": "q",
                    "strategy": "vector",
                    "state": mock_state,
                })
            )

        assert len(data["documents"]) == 2
        first = data["documents"][0]
        assert first["content"] == "First doc"
        assert "image_base64" not in first
        assert "thumbnail_base64" not in first.get("metadata", {})
        assert "image_path" not in first.get("metadata", {})
        assert first.get("metadata", {}).get("doc_id") == "doc-1"
        assert first.get("metadata", {}).get("source") == "source.pdf"
        assert data["documents"][1].get("metadata", {}).get("doc_id") == "doc-2"

    def test_retrieve_parsing_fallback_conversion(self):
        """Test fallback conversion for unexpected result formats."""
        mock_state = {"tools_data": {"vector": Mock()}}
        with patch("src.agents.tool_factory.ToolFactory") as mock_factory:
            mock_tool = Mock()
            mock_tool.call.return_value = {"unexpected": "format", "data": 123}
            mock_factory.create_vector_search_tool.return_value = mock_tool

            data = json.loads(
                retrieve_documents.invoke({
                    "query": "q",
                    "strategy": "vector",
                    "state": mock_state,
                })
            )

        assert len(data["documents"]) == 1
        assert data["documents"][0]["content"].startswith("{'unexpected': 'format'")


class TestAdditionalCoverage:
    """Targeted branch coverage for helper paths."""

    def test_safe_query_for_log_truncates_long_input(self):
        """_safe_query_for_log truncates and adds an ellipsis."""
        import src.agents.tools.retrieval as mod

        out = mod._safe_query_for_log("x " * 200, max_len=10)
        assert out.endswith("…")
        assert len(out) <= 11

    def test_extract_indexes_prefers_runtime_context(self):
        """_extract_indexes prefers runtime.context over persisted state."""
        import src.agents.tools.retrieval as mod

        runtime = type(
            "R", (), {"context": {"vector": "rv", "kg": "rk", "retriever": "rr"}}
        )()
        v, kg, r = mod._extract_indexes(
            {"tools_data": {"vector": "sv"}}, runtime=runtime
        )
        assert (v, kg, r) == ("rv", "rk", "rr")

    def test_optimize_queries_generates_variants_when_dspy_returns_none(
        self, monkeypatch
    ):
        """_optimize_queries adds variants for short queries when DSPy returns none."""
        import src.agents.tools.retrieval as mod

        fake = type(sys)("src.dspy_integration")

        class DSPyLlamaIndexRetriever:  # pragma: no cover - stub
            @staticmethod
            def optimize_query(_q: str):  # type: ignore[no-untyped-def]
                return {"refined": "ref", "variants": []}

        fake.DSPyLlamaIndexRetriever = DSPyLlamaIndexRetriever  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.dspy_integration", fake)

        refined, variants = mod._optimize_queries("AI", use_dspy=True)
        assert refined == "ref"
        assert len(variants) == 2

    def test_graphrag_fallback_marks_strategy_used_as_hybrid(self, monkeypatch):
        """GraphRAG fallback path reports strategy_used='hybrid'."""
        import src.agents.tools.retrieval as mod

        monkeypatch.setattr(
            mod, "_run_graphrag", lambda *_a, **_k: ([{"content": "x"}], True)
        )
        data = json.loads(
            retrieve_documents.invoke({
                "query": "q",
                "strategy": "graphrag",
                "use_dspy": False,
                "use_graphrag": True,
                "state": {"tools_data": {"kg": object()}},
            })
        )
        assert data["strategy_used"] == "hybrid"
        assert data["documents"]

    def test_contextual_recall_includes_recent_sources_when_retrieval_empty(self):
        """Contextual queries can reuse previously persisted sources."""
        mock_state = {
            "tools_data": {"vector": MagicMock()},
            "synthesis_result": {
                "documents": [{"content": "c", "metadata": {"doc_id": "d1"}}]
            },
        }

        with patch("src.agents.tool_factory.ToolFactory") as mock_factory:
            tool = Mock()
            tool.call.return_value = []
            mock_factory.create_vector_search_tool.return_value = tool

            data = json.loads(
                retrieve_documents.invoke({
                    "query": "that chart",
                    "strategy": "vector",
                    "use_dspy": False,
                    "state": mock_state,
                })
            )

        assert data["documents"]
        assert data["strategy_used"].endswith("+recall")

    def test_vector_strategy_errors_when_vector_index_missing(self):
        """Vector strategy fails with a clear error when vector index is absent."""
        data = json.loads(
            retrieve_documents.invoke({
                "query": "q",
                "strategy": "vector",
                "use_dspy": False,
                "state": {"tools_data": {"kg": object()}},
            })
        )
        assert "No vector index available" in data.get("error", "")

    def test_hybrid_empty_primary_falls_back_to_vector(self):
        """Hybrid can fall back to vector search when primary returns no docs."""
        mock_state = {"tools_data": {"vector": MagicMock(), "retriever": MagicMock()}}

        with patch("src.agents.tool_factory.ToolFactory") as mock_factory:
            hybrid = Mock()
            hybrid.call.return_value = []
            vector = Mock()
            vector.call.return_value = [{"content": "doc", "score": 1.0}]
            mock_factory.create_hybrid_search_tool.return_value = hybrid
            mock_factory.create_vector_search_tool.return_value = vector

            data = json.loads(
                retrieve_documents.invoke({
                    "query": "q",
                    "strategy": "hybrid",
                    "use_dspy": False,
                    "state": mock_state,
                })
            )
        assert data["documents"]
        assert data["strategy_used"] == "vector"

    def test_deduplicate_documents_empty_returns_empty(self):
        """_deduplicate_documents returns empty list for empty input."""
        import src.agents.tools.retrieval as mod

        assert mod._deduplicate_documents([]) == []

    def test_recall_recent_sources_uses_latest_retrieval_results(self):
        """_recall_recent_sources chooses the most recent retrieval batch."""
        import src.agents.tools.retrieval as mod

        out = mod._recall_recent_sources({
            "retrieval_results": [
                {"documents": [{"content": "old", "metadata": {"doc_id": "1"}}]},
                {"documents": [{"content": "new", "metadata": {"doc_id": "2"}}]},
            ]
        })
        assert out
        assert out[0]["content"] == "new"

    def test_sanitize_document_dict_applies_basename_to_source(self):
        """_sanitize_document_dict replaces path-like sources with basenames."""
        import src.agents.tools.retrieval as mod

        cleaned = mod._sanitize_document_dict({"content": "x", "source": "/abs/a.pdf"})
        assert cleaned["source"] == "a.pdf"

    def test_parse_tool_result_handles_source_nodes_none(self):
        """_parse_tool_result handles LlamaIndex-like objects with None nodes."""
        import src.agents.tools.retrieval as mod

        class _R:
            source_nodes = None

        docs = mod._parse_tool_result(_R())
        assert docs
        assert isinstance(docs[0], dict)

    def test_parse_tool_result_get_content_branch_and_text_error(self):
        """_parse_tool_result uses get_content and falls back on text errors."""
        import src.agents.tools.retrieval as mod

        class _NodeNoText:
            def get_content(self):  # type: ignore[no-untyped-def]
                return "gc"

        class _NodeBadText:
            @property
            def text(self):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

            def __str__(self) -> str:
                return "fallback"

        class _Nws:
            def __init__(self, node):  # type: ignore[no-untyped-def]
                self.node = node
                self.score = 0.1

        class _R:
            def __init__(self) -> None:
                self.source_nodes = [_Nws(_NodeNoText()), _Nws(_NodeBadText())]

        docs = mod._parse_tool_result(_R())
        assert [d["content"] for d in docs] == ["gc", "fallback"]
