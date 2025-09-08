"""Unit tests for src.utils.document helpers (fast, isolated).

Covers metadata extraction, cache helpers, spaCy wrappers, and KG builders
with lightweight mocks to avoid external dependencies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.utils import document as doc_mod


@pytest.mark.unit
def test_get_document_info_supported(tmp_path: Path) -> None:
    """get_document_info returns supported and strategy metadata for files."""
    f = tmp_path / "sample.txt"
    f.write_text("hello")

    fake_strategy = SimpleNamespace(value="fast")
    with patch.object(doc_mod, "DocumentProcessor") as dp:
        dp.return_value.get_strategy_for_file.return_value = fake_strategy
        info = doc_mod.get_document_info(f)

    assert info["supported"] is True
    assert info["processing_strategy"] == "fast"
    assert info["file_extension"] == ".txt"
    assert info["is_readable"] is True
    assert info["file_size_bytes"] == 5


@pytest.mark.unit
def test_get_document_info_missing_file(tmp_path: Path) -> None:
    """Raises FileNotFoundError for missing files."""
    missing = tmp_path / "nope.pdf"
    with pytest.raises(FileNotFoundError):
        _ = doc_mod.get_document_info(missing)


@pytest.mark.unit
def test_cache_helpers_sync(tmp_path: Path) -> None:
    """Sync wrappers delegate to async cache helpers successfully."""
    with (
        patch("src.utils.document.get_cache_stats") as stats_async,
        patch("src.utils.document.clear_document_cache") as clear_async,
    ):
        stats_async.return_value = {"cache_type": "duckdb_kvstore"}
        clear_async.return_value = True

        assert doc_mod.get_cache_stats_sync()["cache_type"] == "duckdb_kvstore"
        assert doc_mod.clear_document_cache_sync() is True


@pytest.mark.unit
def test_clear_document_cache_deletes_duckdb(tmp_path: Path) -> None:
    """clear_document_cache removes duckdb file when present."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    cache_db = cache_dir / "docmind.duckdb"
    cache_db.write_text("db")

    with patch(
        "src.config.settings.settings", new=SimpleNamespace(cache_dir=cache_dir)
    ):
        # Run async clear
        ok = asyncio.run(doc_mod.clear_document_cache())
        assert ok is True
        assert not cache_db.exists()


@pytest.mark.unit
def test_get_cache_stats_returns_path(tmp_path: Path) -> None:
    """get_cache_stats returns duckdb path and type."""
    cache_dir = tmp_path / "cache2"
    cache_dir.mkdir(parents=True)
    with patch(
        "src.config.settings.settings", new=SimpleNamespace(cache_dir=cache_dir)
    ):
        stats = asyncio.run(doc_mod.get_cache_stats())
        assert stats["cache_type"] == "duckdb_kvstore"
        assert str(cache_dir) in stats["db_path"]


@pytest.mark.unit
def test_ensure_spacy_model_uses_manager() -> None:
    """ensure_spacy_model delegates to SpacyManager.ensure_model."""
    fake_nlp = object()
    mgr = SimpleNamespace(ensure_model=Mock(return_value=fake_nlp))
    with patch.object(doc_mod, "get_spacy_manager", return_value=mgr):
        out = doc_mod.ensure_spacy_model("en_core_web_sm")
    assert out is fake_nlp
    mgr.ensure_model.assert_called_once()


@pytest.mark.unit
def test_extract_entities_and_relationships_with_mock_nlp() -> None:
    """Extraction helpers return entities and relationships using mock nlp."""

    class _Ent:
        def __init__(self, text, label_, start_char, end_char):
            self.text = text
            self.label_ = label_
            self.start_char = start_char
            self.end_char = end_char

    class _Token:
        def __init__(self, text, dep_, head_pos):
            self.text = text
            self.dep_ = dep_
            self.head = SimpleNamespace(pos_=head_pos, text="verb")

    class _Doc:
        def __init__(self):
            self.ents = [_Ent("Alice", "PERSON", 0, 5)]

        def __iter__(self):
            yield from [_Token("Alice", "nsubj", "VERB"), _Token("car", "pobj", "VERB")]

    fake_nlp = Mock(return_value=_Doc())

    ents = doc_mod.extract_entities_with_spacy("Alice drives a car", fake_nlp)
    rels = doc_mod.extract_relationships_with_spacy("Alice drives a car", fake_nlp)

    assert ents
    assert ents[0]["label"] == "PERSON"
    assert any(r["type"] in {"NSUBJ", "POBJ"} for r in rels)


@pytest.mark.unit
def test_create_knowledge_graph_data_both_modes() -> None:
    """KG creation supports text+nlp and direct entities+relationships modes."""
    # mode 1: text + patched extractors
    with (
        patch.object(
            doc_mod, "extract_entities_with_spacy", return_value=[{"text": "x"}]
        ),
        patch.object(
            doc_mod,
            "extract_relationships_with_spacy",
            return_value=[{"source": "a", "target": "b"}],
        ),
    ):
        kg = doc_mod.create_knowledge_graph_data("text", Mock())
    assert set(kg.keys()) == {"entities", "relationships", "metadata"}

    # mode 2: entities + relationships input
    kg2 = doc_mod.create_knowledge_graph_data(
        [{"text": "x"}], [{"source": "a", "target": "b"}]
    )
    assert kg2["metadata"]["entity_count"] == 1
    assert kg2["metadata"]["relationship_count"] == 1
