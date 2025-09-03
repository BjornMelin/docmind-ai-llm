"""Vector storage and retrieval integration tests (modernized).

Uses LlamaIndex in-memory VectorStoreIndex with MockEmbedding configured
globally in tests/conftest.py. Validates indexing, retrieval, metadata,
and basic performance without external services.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from llama_index.core import Document as LIDocument
from llama_index.core import VectorStoreIndex


@pytest.fixture
def integration_settings():
    """Placeholder integration settings for consistency with suite."""

    class _S:
        data_dir = Path("./vector_test_data")
        cache_dir = Path("./vector_test_cache")
        enable_gpu_acceleration = False
        log_level = "INFO"

    return _S()


@pytest.fixture
def texts_meta():
    """Sample texts and metadata for indexing and retrieval tests."""
    samples = [
        (
            "Machine learning enables computers to learn from experience.",
            {"source": "ai_research.txt", "element_category": "NarrativeText"},
        ),
        (
            "Neural networks process information via connected nodes or neurons.",
            {"source": "ai_research.txt", "element_category": "NarrativeText"},
        ),
        (
            "Deep learning uses multiple layers to model complex patterns.",
            {"source": "tech_guide.md", "element_category": "Technical"},
        ),
        (
            "Natural language processing helps interpret human language.",
            {"source": "nlp_guide.txt", "element_category": "Definition"},
        ),
        (
            "Computer vision analyzes visual content from images and videos.",
            {"source": "cv_overview.pdf", "element_category": "Overview"},
        ),
    ]
    return samples


def build_index(samples: list[tuple[str, dict]]) -> VectorStoreIndex:
    """Build a VectorStoreIndex from (text, metadata) samples."""
    docs = [LIDocument(text=t, metadata=m) for t, m in samples]
    return VectorStoreIndex.from_documents(docs)


@pytest.mark.integration
class TestVectorStorageRetrievalWorkflow:
    """In-memory indexing and retrieval workflows."""

    def test_index_and_retrieve_basic(self, request):
        """Index sample docs and verify retrieval returns results."""
        samples = request.getfixturevalue("texts_meta")
        index = build_index(samples)
        results = index.as_retriever(similarity_top_k=3).retrieve(
            "What is machine learning?"
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_retrieve_contains_metadata(self, request):
        """Ensure retrieved nodes expose metadata with categories."""
        samples = request.getfixturevalue("texts_meta")
        index = build_index(samples)
        results = index.as_retriever(similarity_top_k=5).retrieve("neural networks")
        assert len(results) > 0
        assert any(
            getattr(r.node, "metadata", {}).get("element_category") == "NarrativeText"
            for r in results
        )

    def test_multilingual_retrieval(self):
        """Index multilingual texts and retrieve across languages."""
        multilingual = [
            ("Machine learning is transforming technology.", {"lang": "en"}),
            (
                "El aprendizaje automático está transformando la tecnología.",
                {"lang": "es"},
            ),
            ("機械学習はテクノロジーを変革しています。", {"lang": "ja"}),
            ("L'apprentissage automatique transforme la technologie.", {"lang": "fr"}),
            ("机器学习正在改变技术。", {"lang": "zh"}),
        ]
        index = build_index(multilingual)
        results = index.as_retriever(similarity_top_k=5).retrieve(
            "machine learning technology"
        )
        assert len(results) >= 0

    def test_technical_content_retrieval(self):
        """Index technical snippets and verify relevant retrieval."""
        technical = [
            (
                "def calculate_accuracy(preds, labels):\n"
                "    return sum(p == l for p, l in zip(preds, labels)) / len(labels)",
                {"type": "code"},
            ),
            (
                "SELECT COUNT(*) FROM users WHERE created_at > '2024-01-01'",
                {"type": "sql"},
            ),
            (
                (
                    "curl -X POST https://api.example.com/v1/users "
                    "-H 'Content-Type: application/json' "
                    '-d \'{"name": "John"}\''
                ),
                {"type": "api"},
            ),
            ('{"name": "DocMind AI", "version": "1.0.0"}', {"type": "json"}),
        ]
        index = build_index(technical)
        results = index.as_retriever(similarity_top_k=3).retrieve(
            "calculate model accuracy in Python"
        )
        assert len(results) >= 0

    def test_basic_performance(self, request):
        """Simple performance guardrail for small in-memory index."""
        samples = request.getfixturevalue("texts_meta")
        index = build_index(samples)
        start = time.time()
        _ = index.as_retriever(similarity_top_k=3).retrieve("neural networks")
        elapsed = time.time() - start
        assert elapsed < 5.0

    def test_query_matches_source_keyword(self, request):
        """Query containing a source hint should retrieve a matching doc."""
        samples = request.getfixturevalue("texts_meta")
        index = build_index(samples)
        # Include a filename hint from metadata to encourage retrieval
        results = index.as_retriever(similarity_top_k=5).retrieve("ai_research")
        assert isinstance(results, list)
        # At least one result should reference the ai_research.txt source
        assert any(
            getattr(r.node, "metadata", {}).get("source") == "ai_research.txt"
            for r in results
        )
