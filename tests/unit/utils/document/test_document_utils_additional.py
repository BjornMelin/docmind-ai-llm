"""Additional property/invariant tests for src.utils.document.

Covers metadata invariants, supported flag, and knowledge graph builders
using lightweight stubs (no real spaCy or processors).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest


def test_get_document_info_supported(monkeypatch, tmp_path):
    from src.utils import document as d

    # Create a small file
    p = tmp_path / "a.txt"
    p.write_text("hello")

    class _Proc:
        def __init__(self, *a, **k):
            pass

        def get_strategy_for_file(self, file_path: Path):
            return SimpleNamespace(value="text")

    monkeypatch.setattr(d, "DocumentProcessor", _Proc)

    info = d.get_document_info(p)
    assert info["file_path"].endswith("a.txt")
    assert info["file_extension"] == ".txt"
    assert info["file_size_bytes"] == 5
    assert info["supported"] is True
    assert info["processing_strategy"] == "text"


def test_get_document_info_unsupported(monkeypatch, tmp_path):
    from src.utils import document as d

    p = tmp_path / "b.xyz"
    p.write_text("abc")

    class _Proc:
        def __init__(self, *a, **k):
            pass

        def get_strategy_for_file(self, file_path: Path):
            raise ValueError("unsupported")

    monkeypatch.setattr(d, "DocumentProcessor", _Proc)

    info = d.get_document_info(p)
    assert info["supported"] is False
    assert info["processing_strategy"] is None


def test_get_document_info_missing_file_raises():
    from src.utils import document as d

    with pytest.raises(FileNotFoundError):
        d.get_document_info("/no/such/file.pdf")


def test_extract_entities_with_spacy_stub():
    from src.utils import document as d

    class _Ent:
        def __init__(self, text, label, start, end):
            self.text = text
            self.label_ = label
            self.start_char = start
            self.end_char = end

    class _Doc:
        ents: ClassVar[list] = [_Ent("X", "TEST", 0, 1)]

    class _NLP:
        def __call__(self, text):
            return _Doc()

    out = d.extract_entities_with_spacy("hello", _NLP())
    assert out
    assert out[0]["label"] == "TEST"


def test_extract_relationships_with_spacy_stub():
    from src.utils import document as d

    class _Head:
        pos_ = "VERB"
        text = "does"

    class _Tok:
        def __init__(self, dep, text):
            self.dep_ = dep
            self.text = text
            self.head = _Head()

    class _Doc(list):
        pass

    class _NLP:
        def __call__(self, text):
            return _Doc([_Tok("nsubj", "Alice"), _Tok("dobj", "work")])

    rel = d.extract_relationships_with_spacy("hello", _NLP())
    assert any(r["type"] in {"NSUBJ", "DOBJ"} for r in rel)


def test_create_knowledge_graph_from_text(monkeypatch):
    from src.utils import document as d

    monkeypatch.setattr(
        d, "extract_entities_with_spacy", lambda text, nlp=None: [{"text": "A"}]
    )
    monkeypatch.setattr(
        d,
        "extract_relationships_with_spacy",
        lambda text, nlp=None: [{"source": "A", "target": "B", "type": "X"}],
    )

    out = d.create_knowledge_graph_data("hello world", nlp_or_relationships=object())
    assert out["metadata"]["entity_count"] == 1
    assert out["metadata"]["relationship_count"] == 1
    assert out["metadata"]["text_length"] == len("hello world")


def test_create_knowledge_graph_from_lists():
    from src.utils import document as d

    ents = [{"text": "A"}, {"text": "B"}]
    rels = [{"source": "A", "target": "B", "type": "X"}]
    out = d.create_knowledge_graph_data(ents, rels)
    assert out["metadata"]["entity_count"] == 2
    assert out["metadata"]["relationship_count"] == 1
    assert out["metadata"]["text_length"] == 0
