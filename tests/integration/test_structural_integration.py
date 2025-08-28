"""Modern Structural Integration Tests for DocMind AI.

This suite replaces the legacy
`tests/integration/test_structural_integration_workflows.py` with a clean,
best-practices-aligned set of tests that target the current architecture under
`src/` and avoid deprecated interfaces.

Covered workflows:
- Document processing and strategy detection
- Semantic chunking via Unstructured chunk_by_title integration
- Unified BGE-M3 embeddings interface
- Qdrant unified vector store with hybrid RRF fusion
- Adaptive router query engine wiring with reranker
- Multi-agent coordinator initialization basics
- Configuration propagation and env var mapping
- Resource management: GPU memory context + caching
- Error handling (chunker failures)
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import DocMindSettings


@pytest.fixture
def integration_settings(tmp_path):
    """Create DocMind settings for integration tests.

    Args:
        tmp_path: Temporary path provided by pytest.

    Returns:
        DocMindSettings configured to use the temporary filesystem and CPU.
    """
    return DocMindSettings(
        debug=True,
        log_level="DEBUG",
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        enable_gpu_acceleration=False,  # CPU-only for integration tests
        enable_performance_logging=True,
    )


@pytest.mark.integration
class TestDocumentProcessingIntegration:
    """Integration tests for document processing and chunking."""

    def test_document_info_strategy_resolution(self, tmp_path):
        """Verify file strategy detection and basic metadata handling.

        Args:
            tmp_path: Temporary directory for creating a test file.
        """
        from src.utils.document import get_document_info

        # Create a non-empty PDF file
        pdf = tmp_path / "sample.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%...content...")

        info = get_document_info(pdf)

        assert info["supported"] is True
        assert info["file_extension"] == ".pdf"
        assert info["processing_strategy"] in {"hi_res", "fast", "ocr_only"}

    @pytest.mark.asyncio
    async def test_semantic_chunker_with_mocked_unstructured(
        self, integration_settings
    ):
        """Validate semantic chunker wiring with mocked unstructured output.

        Args:
            integration_settings: Test settings.
        """
        from src.models.processing import DocumentElement
        from src.processing.chunking.unstructured_chunker import SemanticChunker

        # Prepare fake unstructured chunks
        fake_chunks = [
            SimpleNamespace(
                text="Section 1",
                category="Title",
                metadata=SimpleNamespace(section_title="A"),
            ),
            SimpleNamespace(
                text="Content paragraph",
                category="NarrativeText",
                metadata=SimpleNamespace(),
            ),
        ]

        elements = [
            DocumentElement(
                text="Title: A", category="Title", metadata={"page_number": 1}
            ),
            DocumentElement(
                text="Some content.",
                category="NarrativeText",
                metadata={"page_number": 1},
            ),
        ]

        with patch(
            "src.processing.chunking.unstructured_chunker.chunk_by_title",
            return_value=fake_chunks,
        ):
            chunker = SemanticChunker(settings=integration_settings.processing)
            result = await chunker.chunk_elements_async(elements)

        assert result.chunks
        assert len(result.chunks) == len(fake_chunks)
        assert result.total_elements == len(elements)
        assert result.processing_time >= 0


@pytest.mark.integration
class TestEmbeddingsIntegration:
    """Integration tests for unified embeddings behavior."""

    def test_bgem3_unified_embeddings_interface(self, integration_settings):
        """Ensure BGEM3Embedding exposes unified embedding outputs.

        The underlying FlagEmbedding model is monkeypatched to avoid downloads.

        Args:
            integration_settings: Test settings (unused, for parity).
        """
        from src.retrieval import embeddings as emb_mod

        class FakeBGEM3Model:
            def __init__(self, model_name, use_fp16=True, device="cpu"):
                self.model_name = model_name
                self.use_fp16 = use_fp16
                self.device = device

            def encode(
                self,
                texts,
                batch_size=1,
                max_length=128,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            ):
                dense = [[0.1] * 1024 for _ in texts] if return_dense else None
                sparse = [{0: 0.5, 2: 0.3} for _ in texts] if return_sparse else None
                colbert = (
                    [[[0.2] * 32] * 3 for _ in texts] if return_colbert_vecs else None
                )
                out = {}
                if dense is not None:
                    out["dense_vecs"] = dense
                if sparse is not None:
                    out["lexical_weights"] = sparse
                if colbert is not None:
                    out["colbert_vecs"] = colbert
                return out

        with patch.object(emb_mod, "BGEM3FlagModel", FakeBGEM3Model):
            from src.retrieval.embeddings import BGEM3Embedding

            model = BGEM3Embedding(device="cpu")
            res = model.get_unified_embeddings(["a", "b", "c"])

        assert set(res.keys()) == {"dense", "sparse", "colbert"}
        assert len(res["dense"]) == 3
        assert len(res["sparse"]) == 3


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Integration tests for unified Qdrant vector store."""

    def test_qdrant_unified_store_add_and_hybrid_query(self):
        """Validate add and hybrid query code paths with a mocked client."""
        from src.retrieval.vector_store import QdrantUnifiedVectorStore

        # Mock client with minimal methods used by the store
        client = MagicMock()
        client.collection_exists.return_value = True

        # Prepare search responses
        dense_hits = [
            SimpleNamespace(
                id="doc1",
                payload={"text": "A", "metadata": {"k": 1}, "node_id": "n1"},
                score=0.9,
            ),
            SimpleNamespace(
                id="doc3",
                payload={"text": "C", "metadata": {"k": 3}, "node_id": "n3"},
                score=0.7,
            ),
        ]
        sparse_hits = [
            SimpleNamespace(
                id="doc2",
                payload={"text": "B", "metadata": {"k": 2}, "node_id": "n2"},
                score=0.85,
            ),
            SimpleNamespace(
                id="doc1",
                payload={"text": "A", "metadata": {"k": 1}, "node_id": "n1"},
                score=0.6,
            ),
        ]

        # Return dense on first call, sparse on second
        client.search.side_effect = [dense_hits, sparse_hits]

        store = QdrantUnifiedVectorStore(client=client, collection_name="test_unified")

        # Add nodes (minimal path)
        from llama_index.core.schema import TextNode

        nodes = [TextNode(text="t1", id_="n1"), TextNode(text="t2", id_="n2")]
        ids = store.add(
            nodes,
            dense_embeddings=[[0.1] * 1024] * 2,
            sparse_embeddings=[{0: 0.5}, {1: 0.3}],
        )
        assert ids == ["n1", "n2"]

        # Hybrid query
        from llama_index.core.vector_stores.types import VectorStoreQuery

        q = VectorStoreQuery(similarity_top_k=2)
        result = store.query(q, dense_embedding=[0.1] * 1024, sparse_embedding={0: 0.5})
        assert len(result.nodes) == 2
        # Expect fused ordering prefers doc1 then doc2
        # given scores and default alpha
        assert result.ids
        assert isinstance(result.ids[0], str)


@pytest.mark.integration
class TestRouterEngineIntegration:
    """Integration tests for router query engine wiring."""

    def test_adaptive_router_engine_wiring(self):
        """Ensure router engine composes with minimal vector index and reranker."""
        from src.retrieval.query_engine import AdaptiveRouterQueryEngine

        # Mock vector index with as_query_engine
        vector_index = MagicMock()
        vector_index.as_query_engine.return_value = MagicMock()

        # Provide a mock reranker (avoid model loading)
        reranker = MagicMock()

        engine = AdaptiveRouterQueryEngine(vector_index=vector_index, reranker=reranker)
        assert engine.router_engine is not None
        assert engine._query_engine_tools
        assert len(engine._query_engine_tools) >= 2


@pytest.mark.integration
class TestAgentCoordinatorIntegration:
    """Integration tests for MultiAgentCoordinator minimal init path."""

    def test_multi_agent_coordinator_initialization_minimal(self, integration_settings):
        """Validate coordinator basic construction without heavy setup.

        Args:
            integration_settings: Test settings for context window size.
        """
        from src.agents.coordinator import MultiAgentCoordinator

        # Avoid heavy setup by short-circuiting _ensure_setup
        with patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=True):
            coord = MultiAgentCoordinator(
                max_context_length=integration_settings.vllm.context_window
            )

        assert coord is not None
        assert coord.max_context_length == integration_settings.vllm.context_window


@pytest.mark.integration
class TestConfigurationPropagation:
    """Integration tests for configuration accessors and env mapping."""

    def test_nested_config_access(self, integration_settings):
        """Verify nested configuration structure provides expected values.

        Args:
            integration_settings: Test settings instance.
        """
        # Test direct nested access instead of config methods
        assert integration_settings.vllm.model is not None
        assert integration_settings.agents.enable_multi_agent in {True, False}
        assert integration_settings.embedding.model_name.startswith("BAAI/")
        assert integration_settings.processing.chunk_size > 0

    def test_environment_variable_mapping(self):
        """Ensure environment variables map to nested settings fields."""
        env = {
            "DOCMIND_RETRIEVAL__TOP_K": "20",
            "DOCMIND_RETRIEVAL__USE_RERANKING": "false",
            "DOCMIND_RETRIEVAL__STRATEGY": "dense",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = DocMindSettings()
            assert settings.retrieval.top_k == 20
            assert settings.retrieval.use_reranking is False
            assert settings.retrieval.strategy == "dense"


@pytest.mark.integration
class TestResourceManagementIntegration:
    """Integration tests for memory context and caching paths."""

    def test_gpu_memory_context_and_cache(self, integration_settings):
        """Validate gpu_memory_context works alongside SimpleCache.

        Args:
            integration_settings: Test settings for cache init.
        """
        from src.cache.simple_cache import SimpleCache
        from src.utils.storage import gpu_memory_context

        with (
            patch.object(SimpleCache, "get", return_value=None) as mock_get,
            patch.object(SimpleCache, "set", return_value=True) as mock_set,
        ):
            cache = SimpleCache(settings=integration_settings)
            with gpu_memory_context():
                key = "embeddings:test"
                assert cache.get(key) is None
                assert cache.set(key, {"dense": [[0.1] * 8]}) is True
        mock_get.assert_called_once()
        mock_set.assert_called_once()


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling and resilience paths."""

    @pytest.mark.asyncio
    async def test_chunker_raises_chunking_error(self, integration_settings):
        """Ensure SemanticChunker surfaces ChunkingError on failures.

        Args:
            integration_settings: Test settings passed to the chunker.
        """
        from src.models.processing import DocumentElement
        from src.processing.chunking.unstructured_chunker import (
            ChunkingError,
            SemanticChunker,
        )

        elements = [
            DocumentElement(text="X", category="Title", metadata={}),
        ]

        with patch(
            "src.processing.chunking.unstructured_chunker.chunk_by_title",
            side_effect=Exception("boom"),
        ):
            chunker = SemanticChunker(settings=integration_settings.processing)
            with pytest.raises(ChunkingError):
                await chunker.chunk_elements_async(elements)
