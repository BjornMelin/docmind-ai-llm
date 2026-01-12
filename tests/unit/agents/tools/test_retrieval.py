"""Unit tests for retrieval.retrieve_documents and parsing boundaries.

Split and adapted from legacy tests/unit/agents/test_tools.py.
"""

from __future__ import annotations

import json
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

            result_json = retrieve_documents.invoke(
                {
                    "query": "test query",
                    "strategy": "vector",
                    "use_dspy": True,
                    "state": mock_state,
                }
            )
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

            result_json = retrieve_documents.invoke(
                {
                    "query": "AI",
                    "use_dspy": True,
                    "state": mock_state,
                }
            )
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

            result_json = retrieve_documents.invoke(
                {
                    "query": "test query",
                    "strategy": "vector",
                    "state": mock_state,
                    "use_dspy": False,
                }
            )
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
                retrieve_documents.invoke(
                    {
                        "query": "q",
                        "strategy": "vector",
                        "state": mock_state,
                    }
                )
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
                retrieve_documents.invoke(
                    {
                        "query": "q",
                        "strategy": "vector",
                        "state": mock_state,
                    }
                )
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
                retrieve_documents.invoke(
                    {
                        "query": "q",
                        "strategy": "vector",
                        "state": mock_state,
                    }
                )
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
                retrieve_documents.invoke(
                    {
                        "query": "q",
                        "strategy": "vector",
                        "state": mock_state,
                    }
                )
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
                retrieve_documents.invoke(
                    {
                        "query": "q",
                        "strategy": "vector",
                        "state": mock_state,
                    }
                )
            )

        assert len(data["documents"]) == 1
        assert data["documents"][0]["content"].startswith("{'unexpected': 'format'")
