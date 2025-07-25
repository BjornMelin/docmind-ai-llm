"""Tests for utility functions and helper modules.

This module tests core utility functions including hardware detection, document
loading, vectorstore creation, document analysis, chat functionality, and
reranking components following 2025 best practices.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from langchain.schema import Document

from models import AnalysisOutput
from utils import (
    JinaRerankCompressor,
    analyze_documents,
    chat_with_context,
    create_vectorstore,
    detect_hardware,
    late_chunking,
    load_documents,
)


@pytest.fixture
def tmp_pdf(tmp_path):
    """Create a temporary PDF file for testing.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path: Path to the temporary PDF file.
    """
    path = tmp_path / "test.pdf"
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4 dummy content")
    return path


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing.

    Returns:
        MagicMock: Mock LLM with invoke and stream methods.
    """
    llm = MagicMock()
    llm.invoke.return_value = "Mock output"
    llm.stream.return_value = iter(["Mock ", "stream"])
    return llm


def test_detect_hardware():
    """Test hardware detection functionality.

    Tests that hardware detection returns valid hardware type
    and appropriate VRAM information.
    """
    hardware, vram = detect_hardware()
    assert hardware in ["GPU detected", "CPU only"]
    if hardware == "CPU only":
        assert vram is None
    else:
        assert isinstance(vram, int)


def test_load_pdf_documents(tmp_pdf):
    """Test PDF document loading functionality.

    Args:
        tmp_pdf: Temporary PDF file fixture.
    """
    docs = load_documents([tmp_pdf])
    assert len(docs) > 0
    assert isinstance(docs[0], Document)
    assert "images" in docs[0].metadata


def test_load_unsupported_format(tmp_path):
    """Test handling of unsupported file formats.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    path = tmp_path / "test.unsupported"
    path.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_documents([path])


def test_late_chunking_functionality():
    """Test late chunking implementation for document processing."""
    text = "Sentence one. Sentence two."
    token_emb = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    chunks = late_chunking(text, token_emb)
    assert len(chunks) == 2
    assert chunks[0].shape == (2,)


@patch("utils.AutoModel.from_pretrained")
def test_load_documents_with_late_chunking(mock_model, tmp_pdf):
    """Test document loading with late chunking enabled.

    Args:
        mock_model: Mock transformer model.
        tmp_pdf: Temporary PDF file fixture.
    """
    mock_model.return_value = MagicMock(last_hidden_state=torch.rand(10, 768))
    docs = load_documents([tmp_pdf], late_chunking=True)
    assert "chunk_embeddings" in docs[0].metadata


@patch("utils.QdrantClient")
def test_create_vectorstore_functionality(mock_client, tmp_pdf):
    """Test vectorstore creation with documents.

    Args:
        mock_client: Mock Qdrant client.
        tmp_pdf: Temporary PDF file fixture.
    """
    docs = load_documents([tmp_pdf])
    # Mock the vectorstore creation since Qdrant may not be available
    with patch("utils.Qdrant") as mock_qdrant:
        mock_qdrant.return_value = MagicMock()
        vs = create_vectorstore(docs, multi_vector=True)
        assert vs is not None
        mock_client.assert_called()


def test_analyze_documents_functionality(mock_llm):
    """Test document analysis with LLM.

    Args:
        mock_llm: Mock LLM fixture.
    """
    texts = ["Test text"]
    result = analyze_documents(
        mock_llm,
        texts,
        "Comprehensive Document Analysis",
        "",
        "Neutral",
        "General Assistant",
        "",
        "Concise",
        4096,
    )
    assert isinstance(result, AnalysisOutput | str)


def test_analyze_documents_chunked_mode(mock_llm):
    """Test document analysis with chunked processing.

    Args:
        mock_llm: Mock LLM fixture.
    """
    texts = ["A" * 10000]  # Large text
    result = analyze_documents(
        mock_llm,
        texts,
        "Comprehensive Document Analysis",
        "",
        "Neutral",
        "General Assistant",
        "",
        "Concise",
        4096,
        chunked=True,
    )
    assert isinstance(result, AnalysisOutput | str)


def test_chat_with_context_functionality(mock_llm, tmp_pdf):
    """Test chat functionality with document context.

    Args:
        mock_llm: Mock LLM fixture.
        tmp_pdf: Temporary PDF file fixture.
    """
    docs = load_documents([tmp_pdf])

    # Mock vectorstore since Qdrant may not be available
    with patch("utils.create_vectorstore") as mock_create_vs:
        mock_vs = MagicMock()
        mock_create_vs.return_value = mock_vs

        vs = create_vectorstore(docs)
        history = [{"user": "Hi", "assistant": "Hello"}]
        response = list(chat_with_context(mock_llm, vs, "Question?", history))
        assert len(response) > 0


def test_jina_rerank_compressor_functionality():
    """Test Jina rerank compressor functionality."""
    compressor = JinaRerankCompressor(top_n=2)
    docs = [Document(page_content="Doc1"), Document(page_content="Doc2")]
    compressed = compressor.compress_documents(docs, "Query")
    assert len(compressed) <= 2
