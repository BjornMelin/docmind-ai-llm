"""Tests for Knowledge Graph creation and spaCy integration.

This module tests:
- SpaCy model management (loading and auto-download)
- Knowledge graph data extraction from text
- Entity and relationship extraction
- Graceful fallback when dependencies missing

Following PyTestQA-Agent standards for comprehensive testing.
NOTE: Legacy src.utils.document functions replaced with ADR-009 architecture.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest


# Mock all legacy document processing functions removed with ADR-009
class LegacyDocumentMocks:
    @staticmethod
    def ensure_spacy_model():
        """Mock for removed ensure_spacy_model function."""
        mock_nlp = MagicMock()
        mock_nlp.vocab = MagicMock()
        return mock_nlp

    @staticmethod
    def extract_entities_with_spacy(text, nlp_model=None):
        """Mock for removed extract_entities_with_spacy function."""
        return [
            {"text": "Sample Entity", "label": "PERSON", "start": 0, "end": 13},
            {"text": "Test Company", "label": "ORG", "start": 20, "end": 32},
        ]

    @staticmethod
    def extract_relationships_with_spacy(text, nlp_model=None):
        """Mock for removed extract_relationships_with_spacy function."""
        return [
            {
                "subject": "Sample Entity",
                "predicate": "works_for",
                "object": "Test Company",
            },
        ]

    @staticmethod
    def create_knowledge_graph_data(text, nlp_model=None):
        """Mock for removed create_knowledge_graph_data function."""
        return {
            "entities": LegacyDocumentMocks.extract_entities_with_spacy(text),
            "relationships": LegacyDocumentMocks.extract_relationships_with_spacy(text),
        }


# Replace all legacy imports with mocks
ensure_spacy_model = LegacyDocumentMocks.ensure_spacy_model
extract_entities_with_spacy = LegacyDocumentMocks.extract_entities_with_spacy
extract_relationships_with_spacy = LegacyDocumentMocks.extract_relationships_with_spacy
create_knowledge_graph_data = LegacyDocumentMocks.create_knowledge_graph_data


class TestSpaCyIntegration:
    """Test spaCy model management and integration."""

    def test_ensure_spacy_model_already_installed(self):
        """Test when spaCy model is already available."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp

            from src.utils.document import ensure_spacy_model

            nlp = ensure_spacy_model()

            assert nlp is not None
            assert nlp == mock_nlp
            mock_load.assert_called_once_with("en_core_web_sm")

    def test_ensure_spacy_model_auto_download_success(self):
        """Test automatic model download when missing."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            # First call fails (not installed), second succeeds
            mock_load.side_effect = [OSError("Model not found"), mock_nlp]

            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)

                from src.utils.document import ensure_spacy_model

                nlp = ensure_spacy_model()

                assert nlp is not None
                assert nlp == mock_nlp
                # Should try to load twice (before and after download)
                assert mock_load.call_count == 2
                # Should call subprocess to download
                mock_subprocess.assert_called_once()

    def test_ensure_spacy_model_auto_download_failure(self):
        """Test fallback when auto-download fails."""
        with patch("spacy.load") as mock_load:
            # Always fail to load model
            mock_load.side_effect = OSError("Model not found")

            with patch("subprocess.run") as mock_subprocess:
                # Download also fails
                mock_subprocess.return_value = MagicMock(returncode=1)

                from src.utils.document import ensure_spacy_model

                nlp = ensure_spacy_model()

                # Should return None when both load and download fail
                assert nlp is None
                # Should try to load twice (before and after failed download)
                assert mock_load.call_count == 2
                mock_subprocess.assert_called_once()

    def test_ensure_spacy_model_no_subprocess_available(self):
        """Test fallback when subprocess is not available."""
        with patch("spacy.load") as mock_load:
            mock_load.side_effect = OSError("Model not found")

            with patch("subprocess.run", side_effect=FileNotFoundError):
                from src.utils.document import ensure_spacy_model

                nlp = ensure_spacy_model()

                # Should return None when subprocess not available
                assert nlp is None

    def test_ensure_spacy_model_subprocess_timeout(self):
        """Test fallback when subprocess times out."""
        with patch("spacy.load") as mock_load:
            mock_load.side_effect = OSError("Model not found")

            with patch(
                "subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)
            ):
                from src.utils.document import ensure_spacy_model

                nlp = ensure_spacy_model()

                # Should return None when download times out
                assert nlp is None

    def test_ensure_spacy_model_generic_subprocess_error(self):
        """Test fallback when subprocess has generic error."""
        with patch("spacy.load") as mock_load:
            mock_load.side_effect = OSError("Model not found")

            with patch("subprocess.run", side_effect=Exception("Generic error")):
                from src.utils.document import ensure_spacy_model

                nlp = ensure_spacy_model()

                # Should return None on generic subprocess error
                assert nlp is None

    def test_ensure_spacy_model_import_error_handling(self):
        """Test graceful handling when spaCy itself is not installed."""
        with patch("spacy.load", side_effect=ImportError("spaCy not installed")):
            from src.utils.document import ensure_spacy_model

            nlp = ensure_spacy_model()

            # Should return None when spaCy not installed
            assert nlp is None


class TestKnowledgeGraphExtraction:
    """Test knowledge graph data extraction functionality."""

    def test_extract_entities_with_spacy_success(self):
        """Test successful entity extraction from text."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()

            # Create mock entities
            mock_entity = MagicMock()
            mock_entity.text = "Apple Inc"
            mock_entity.label_ = "ORG"
            mock_entity.start_char = 0
            mock_entity.end_char = 9

            mock_doc.ents = [mock_entity]
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp

            from src.utils.document import extract_entities_with_spacy

            entities = extract_entities_with_spacy("Apple Inc is a technology company")

            assert len(entities) == 1
            assert entities[0]["text"] == "Apple Inc"
            assert entities[0]["label"] == "ORG"
            assert entities[0]["start"] == 0
            assert entities[0]["end"] == 9
            assert entities[0]["confidence"] == 1.0

    def test_extract_entities_with_spacy_failure(self):
        """Test entity extraction with spaCy failure."""
        with patch("src.utils.document.ensure_spacy_model", return_value=None):
            from src.utils.document import extract_entities_with_spacy

            entities = extract_entities_with_spacy("Some text")

            assert entities == []

    def test_extract_relationships_with_spacy_success(self):
        """Test successful relationship extraction from text."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()

            # Create mock tokens for relationship extraction
            mock_token = MagicMock()
            mock_token.text = "Apple"
            mock_token.dep_ = "nsubj"
            mock_token.idx = 0
            mock_token.head = MagicMock()
            mock_token.head.text = "founded"
            mock_token.head.pos_ = "VERB"

            mock_doc.__iter__ = lambda self: iter([mock_token])
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp

            from src.utils.document import extract_relationships_with_spacy

            relationships = extract_relationships_with_spacy(
                "Apple founded the company"
            )

            assert len(relationships) == 1
            assert relationships[0]["subject"] == "Apple"
            assert relationships[0]["relation"] == "founded"
            assert relationships[0]["dependency"] == "nsubj"
            assert relationships[0]["start"] == 0
            assert relationships[0]["end"] == 5

    def test_extract_relationships_with_spacy_failure(self):
        """Test relationship extraction with spaCy failure."""
        with patch("src.utils.document.ensure_spacy_model", return_value=None):
            from src.utils.document import extract_relationships_with_spacy

            relationships = extract_relationships_with_spacy("Some text")

            assert relationships == []

    def test_create_knowledge_graph_data_success(self):
        """Test successful knowledge graph data creation."""
        with (
            patch("src.utils.document.extract_entities_with_spacy") as mock_entities,
            patch(
                "src.utils.document.extract_relationships_with_spacy"
            ) as mock_relationships,
        ):
            # Mock entity and relationship extraction
            mock_entities.return_value = [
                {
                    "text": "Apple",
                    "label": "ORG",
                    "start": 0,
                    "end": 5,
                    "confidence": 1.0,
                }
            ]
            mock_relationships.return_value = [
                {
                    "subject": "Apple",
                    "relation": "founded",
                    "object": "company",
                    "dependency": "nsubj",
                    "start": 0,
                    "end": 5,
                }
            ]

            from src.utils.document import create_knowledge_graph_data

            kg_data = create_knowledge_graph_data("Apple founded the company")

            assert "entities" in kg_data
            assert "relationships" in kg_data
            assert "text_length" in kg_data
            assert "processed_at" in kg_data

            assert len(kg_data["entities"]) == 1
            assert len(kg_data["relationships"]) == 1
            assert kg_data["text_length"] == 23
            assert isinstance(kg_data["processed_at"], float)

    def test_create_knowledge_graph_data_empty_text(self):
        """Test knowledge graph data creation with empty text."""
        with (
            patch("src.utils.document.extract_entities_with_spacy", return_value=[]),
            patch(
                "src.utils.document.extract_relationships_with_spacy", return_value=[]
            ),
        ):
            from src.utils.document import create_knowledge_graph_data

            kg_data = create_knowledge_graph_data("")

            assert kg_data["entities"] == []
            assert kg_data["relationships"] == []
            assert kg_data["text_length"] == 0
            assert isinstance(kg_data["processed_at"], float)


class TestKnowledgeGraphIntegration:
    """Test knowledge graph integration with document processing."""

    def test_full_pipeline_integration(self):
        """Test full knowledge graph extraction pipeline."""
        sample_text = (
            "Apple Inc, founded by Steve Jobs, revolutionized personal computing. "
            "The company develops innovative products."
        )

        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()

            # Mock entities
            entities = []
            entity1 = MagicMock()
            entity1.text = "Apple Inc"
            entity1.label_ = "ORG"
            entity1.start_char = 0
            entity1.end_char = 9
            entities.append(entity1)

            entity2 = MagicMock()
            entity2.text = "Steve Jobs"
            entity2.label_ = "PERSON"
            entity2.start_char = 23
            entity2.end_char = 33
            entities.append(entity2)

            mock_doc.ents = entities

            # Mock tokens for relationships
            tokens = []
            token1 = MagicMock()
            token1.text = "Apple"
            token1.dep_ = "nsubj"
            token1.idx = 0
            token1.head = MagicMock()
            token1.head.text = "revolutionized"
            token1.head.pos_ = "VERB"
            tokens.append(token1)

            mock_doc.__iter__ = lambda self: iter(tokens)
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp

            from src.utils.document import create_knowledge_graph_data

            kg_data = create_knowledge_graph_data(sample_text)

            # Verify extracted data
            assert len(kg_data["entities"]) == 2
            assert len(kg_data["relationships"]) == 1

            # Check entity data
            apple_entity = next(
                e for e in kg_data["entities"] if e["text"] == "Apple Inc"
            )
            assert apple_entity["label"] == "ORG"

            steve_entity = next(
                e for e in kg_data["entities"] if e["text"] == "Steve Jobs"
            )
            assert steve_entity["label"] == "PERSON"

            # Check relationship data
            relationship = kg_data["relationships"][0]
            assert relationship["subject"] == "Apple"
            assert relationship["relation"] == "revolutionized"

    def test_error_handling_in_pipeline(self):
        """Test error handling in knowledge graph pipeline."""
        with patch("spacy.load", side_effect=Exception("SpaCy error")):
            from src.utils.document import create_knowledge_graph_data

            # Should handle errors gracefully
            kg_data = create_knowledge_graph_data("Some text")

            assert kg_data["entities"] == []
            assert kg_data["relationships"] == []
            assert kg_data["text_length"] == 9
            assert isinstance(kg_data["processed_at"], float)

    @pytest.mark.performance
    def test_knowledge_graph_performance(self, benchmark):
        """Test knowledge graph extraction performance."""
        sample_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This sentence contains various entities and relationships."
        )

        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.ents = []
            mock_doc.__iter__ = lambda self: iter([])
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp

            from src.utils.document import create_knowledge_graph_data

            def extract_kg_data():
                return create_knowledge_graph_data(sample_text)

            result = benchmark(extract_kg_data)

            assert "entities" in result
            assert "relationships" in result
            assert "text_length" in result
            assert "processed_at" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
