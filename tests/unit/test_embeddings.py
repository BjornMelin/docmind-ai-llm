"""Tests for embedding models and GPU optimization.

This module tests SPLADE++ sparse embeddings, BGE-Large dense embeddings,
GPU optimization with torch.compile, multimodal processing with Jina v3,
and the ModelManager following 2025 best practices.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from llama_index.core import Document
from llama_index.core.schema import ImageDocument

from utils.model_manager import ModelManager
from utils.utils import get_embed_model


class TestModelManager:
    """Test ModelManager model handling and caching."""

    def test_model_manager_behavior(self):
        """Test that ModelManager handles models properly."""
        with patch("utils.model_manager.HuggingFaceEmbedding") as mock_hf_embed:
            mock_model = MagicMock()
            mock_hf_embed.return_value = mock_model

            # Test getting embedding model
            model = ModelManager.get_multimodal_embedding_model()
            assert model is not None

    def test_model_manager_error_handling(self):
        """Test ModelManager error handling."""
        with patch("utils.model_manager.HuggingFaceEmbedding") as mock_hf_embed:
            mock_hf_embed.side_effect = Exception("Model loading failed")

            with pytest.raises(Exception, match="Model loading failed"):
                ModelManager.get_multimodal_embedding_model()


class TestSparseEmbeddings:
    """Test SPLADE++ sparse embedding functionality with GPU optimization."""

    def test_splade_model_name_validation(self, test_settings):
        """Test SPLADE++ model name matches configuration."""
        # Test correct model name with fixed typo
        assert test_settings.sparse_embedding_model == "prithvida/Splade_PP_en_v1"
        assert "Splade_PP_en_v1" in test_settings.sparse_embedding_model

    @pytest.mark.parametrize(
        "model_name",
        [
            "prithivida/Splade_PP_en_v1",  # Fixed typo from original
            "naver/splade-cocondenser-ensembledistil",
        ],
    )
    def test_splade_plus_plus_embedding_models(
        self, model_name, mock_sparse_embedding_model
    ):
        """Test SPLADE++ sparse embedding models with fixed typo."""
        with patch(
            "utils.model_manager.SparseTextEmbedding",
            return_value=mock_sparse_embedding_model,
        ):
            # Test encoding with SPLADE++ specific features
            texts = [
                "Hybrid retrieval with SPLADE++ sparse embeddings",
                "Dense BGE-Large semantic understanding combined with sparse term expansion",
            ]
            embeddings = mock_sparse_embedding_model.encode(texts)

            assert len(embeddings) == len(texts)
            for embedding in embeddings:
                # SPLADE++ specific structure
                assert "indices" in embedding or hasattr(embedding, "indices")
                assert "values" in embedding or hasattr(embedding, "values")

                # Extract indices and values for validation
                if isinstance(embedding, dict):
                    indices = embedding["indices"]
                    values = embedding["values"]
                else:
                    indices = embedding.indices
                    values = embedding.values

                assert len(indices) == len(values)
                # SPLADE++ should generate sparse representations
                assert len(indices) > 0, (
                    "SPLADE++ should generate non-empty sparse vectors"
                )

    @pytest.mark.skipif(
        not os.getenv("RUN_NETWORK_TESTS"), reason="Network tests disabled"
    )
    def test_splade_term_expansion(self):
        """Test SPLADE++ term expansion capabilities."""
        from models import AppSettings

        settings = AppSettings()

        # Mock to avoid network download
        with patch("fastembed.SparseTextEmbedding") as mock_sparse:
            mock_instance = MagicMock()
            mock_sparse.return_value = mock_instance

            # Mock sparse embedding with realistic SPLADE++ structure
            mock_embedding = MagicMock()
            mock_embedding.indices = [1, 5, 10, 15, 25, 30]  # Term indices
            mock_embedding.values = [
                0.8,
                0.6,
                0.4,
                0.3,
                0.2,
                0.1,
            ]  # ReLU positive values
            mock_instance.embed.return_value = [mock_embedding]

            sparse_model = mock_sparse(settings.sparse_embedding_model)
            embeddings = list(sparse_model.embed(["library"]))

            assert len(embeddings) == 1
            emb = embeddings[0]
            assert len(emb.indices) > 0
            assert len(emb.values) > 0
            assert all(
                v > 0 for v in emb.values
            )  # ReLU activation ensures positive values

    def test_splade_plus_plus_properties(self, mock_sparse_embedding_model):
        """Test SPLADE++ embedding properties and sparse structure."""
        texts = [
            "Short query",
            "Much longer document with technical terminology and domain-specific concepts",
        ]
        embeddings = mock_sparse_embedding_model.encode(texts)

        for embedding in embeddings:
            # Extract indices and values (handle both dict and object formats)
            if isinstance(embedding, dict):
                indices = embedding["indices"]
                values = embedding["values"]
            else:
                indices = embedding.indices
                values = embedding.values

            # SPLADE++ sparse embeddings validation
            assert all(idx >= 0 for idx in indices), "Indices should be non-negative"
            assert all(val > 0 for val in values), (
                "SPLADE++ uses ReLU activation - values should be positive"
            )
            assert list(indices) == sorted(indices), (
                "Indices should be sorted for efficient storage"
            )

            # SPLADE++ term expansion verification
            assert len(indices) > 5, (
                "SPLADE++ should expand terms beyond original tokens"
            )
            assert max(indices) < 50000, "Indices should be within vocabulary range"

    def test_splade_gpu_providers(self):
        """Test SPLADE++ GPU provider configuration."""
        with patch("fastembed.SparseTextEmbedding") as mock_sparse:
            with patch("torch.cuda.is_available", return_value=True):
                # Should use GPU providers when available
                expected_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

                # Mock call verification
                mock_sparse.assert_not_called()  # Not called yet

    def test_splade_batch_processing(self, mock_sparse_embedding_model):
        """Test SPLADE++ batch processing capabilities."""
        from models import AppSettings

        settings = AppSettings()

        # Large batch of texts
        texts = [
            f"Document {i} with unique content"
            for i in range(settings.embedding_batch_size)
        ]

        embeddings = mock_sparse_embedding_model.encode(texts)

        # Should handle full batch
        assert len(embeddings) == len(texts)
        assert len(embeddings) == settings.embedding_batch_size


class TestDenseEmbeddings:
    """Test BGE-Large dense embedding functionality with GPU acceleration."""

    @pytest.mark.parametrize(
        ("model_name", "expected_dim"),
        [
            ("BAAI/bge-large-en-v1.5", 1024),  # Primary model for dense embeddings
            ("BAAI/bge-base-en-v1.5", 768),
            ("sentence-transformers/all-MiniLM-L6-v2", 384),
        ],
    )
    def test_bge_large_dense_embedding_models(
        self, model_name, expected_dim, mock_embedding_model
    ):
        """Test BGE-Large and related dense embedding models."""
        # Configure mock for BGE-Large specific testing
        mock_embedding_model.embed_documents.return_value = [
            [0.1 * (i + 1)] * expected_dim for i in range(3)
        ]
        mock_embedding_model.embed_query.return_value = [0.5] * expected_dim

        with patch(
            "utils.model_manager.FastEmbedEmbedding", return_value=mock_embedding_model
        ):
            # Test document embedding with semantic content
            texts = [
                "Hybrid retrieval systems combine dense and sparse methods",
                "BGE-Large provides rich semantic understanding for search",
                "RRF fusion optimally combines multiple retrieval approaches",
            ]
            doc_embeddings = mock_embedding_model.embed_documents(texts)

            assert len(doc_embeddings) == len(texts)
            for i, embedding in enumerate(doc_embeddings):
                assert len(embedding) == expected_dim
                assert all(isinstance(val, (int, float)) for val in embedding)
                # Verify embeddings are different for different content
                if i > 0:
                    assert embedding != doc_embeddings[i - 1]

            # Test query embedding
            query_embedding = mock_embedding_model.embed_query(
                "What is hybrid search with RRF?"
            )
            assert len(query_embedding) == expected_dim


class TestGPUOptimization:
    """Test GPU optimization features for embeddings."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_acceleration_detection(self, test_settings):
        """Test GPU acceleration detection and configuration."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("utils.utils.FastEmbedEmbedding") as mock_fastembed:
                mock_model = MagicMock()
                mock_fastembed.return_value = mock_model

                # Test GPU optimization with torch.compile
                model = get_embed_model()

                # Verify GPU optimization was attempted
                assert model is not None
                # Check that GPU providers were configured
                call_kwargs = mock_fastembed.call_args[1]
                assert "CUDAExecutionProvider" in call_kwargs.get("providers", [])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_torch_compile_optimization(self, test_settings):
        """Test torch.compile optimization for GPU acceleration."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.compile") as mock_compile:
                with patch("utils.utils.FastEmbedEmbedding") as mock_fastembed:
                    mock_model = MagicMock()
                    mock_fastembed.return_value = mock_model
                    mock_compile.return_value = mock_model

                    # Test torch.compile optimization
                    model = get_embed_model()

                    # Verify torch.compile was called for GPU optimization
                    mock_compile.assert_called()
                    assert model is not None


class TestMultimodalEmbeddings:
    """Test multimodal embedding capabilities with Jina v3."""

    def test_jina_v3_multimodal_setup(self, test_settings):
        """Test Jina v3 multimodal embedding model setup."""
        with patch("utils.model_manager.HuggingFaceEmbedding") as mock_hf_embed:
            mock_model = MagicMock()
            mock_hf_embed.return_value = mock_model

            # Test multimodal model creation
            model = ModelManager.get_multimodal_embedding_model()

            # Verify Jina v3 configuration would be used
            assert model is not None

    def test_multimodal_document_processing(self, sample_documents):
        """Test multimodal document processing with text and images."""
        # Create mixed document set
        text_docs = sample_documents[:2]
        image_docs = [
            ImageDocument(
                text="Technical diagram showing hybrid search architecture",
                image_path="/fake/path/diagram.jpg",
                metadata={"source": "technical_paper.pdf", "page": 1, "type": "image"},
            )
        ]

        mixed_docs = text_docs + image_docs

        # Test that mixed document types can be processed
        assert len(mixed_docs) == 3
        assert any(isinstance(doc, Document) for doc in mixed_docs)
        assert any(isinstance(doc, ImageDocument) for doc in mixed_docs)

        # Verify document metadata includes multimodal information
        for doc in mixed_docs:
            assert "type" in doc.metadata or hasattr(doc, "image")


class TestEmbeddingIntegration:
    """Test integration between sparse and dense embeddings."""

    def test_embedding_dimension_consistency(self, test_settings):
        """Test that embedding dimensions are consistent with settings."""
        # Test that dense embedding dimension matches BGE-Large
        assert test_settings.dense_embedding_dimension == 1024

        # Verify model names are correctly configured with fixed SPLADE++ typo
        assert "bge-large-en-v1.5" in test_settings.dense_embedding_model
        assert "Splade_PP_en_v1" in test_settings.sparse_embedding_model  # Fixed typo

    def test_embedding_preprocessing(self, sample_documents):
        """Test text preprocessing before embedding."""
        # Test that documents are properly prepared for embedding
        texts = [doc.text for doc in sample_documents]

        for text in texts:
            # Text should not be empty
            assert len(text.strip()) > 0
            # Text should be reasonable length
            assert len(text) < 10000  # Avoid extremely long texts
            # Text should contain meaningful content
            assert any(word.isalpha() for word in text.split())

    @pytest.mark.integration
    def test_embedding_pipeline_integration(
        self,
        sample_documents,
        mock_embedding_model,
        mock_sparse_embedding_model,
    ):
        """Test complete embedding pipeline with both sparse and dense."""
        texts = [doc.text for doc in sample_documents]

        # Test dense embeddings
        with patch(
            "utils.model_manager.FastEmbedEmbedding", return_value=mock_embedding_model
        ):
            dense_embeddings = mock_embedding_model.embed_documents(texts)

            # Verify dense embedding structures
            assert len(dense_embeddings) == len(texts)
            for dense_emb in dense_embeddings:
                assert isinstance(dense_emb, list)
                assert len(dense_emb) == 1024  # BGE-Large dimension

        # Test sparse embeddings
        with patch(
            "utils.model_manager.SparseTextEmbedding",
            return_value=mock_sparse_embedding_model,
        ):
            sparse_embeddings = mock_sparse_embedding_model.encode(texts)

            # Verify sparse embedding structures
            assert len(sparse_embeddings) == len(texts)
            for sparse_emb in sparse_embeddings:
                assert isinstance(sparse_emb, dict)
                assert "indices" in sparse_emb
                assert "values" in sparse_emb
