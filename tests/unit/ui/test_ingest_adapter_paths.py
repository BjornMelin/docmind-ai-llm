"""Extended tests for ingest adapter behaviors.

Focus: file save, non-empty ingest, GraphRAG export, and analytics logging.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakeUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getbuffer(self):  # streamlit UploadedFile compat
        return self._data


@pytest.mark.unit
def test_save_uploaded_file(monkeypatch, tmp_path):
    from src.config import settings
    from src.ui import ingest_adapter as mod

    monkeypatch.setattr(settings, "data_dir", tmp_path)

    file = _FakeUpload("../unsafe.txt", b"hello")
    p = mod._save_uploaded_file(file)  # pylint: disable=protected-access
    assert p.exists()
    assert p.name == "unsafe.txt"  # sanitized


@pytest.mark.unit
def test_ingest_non_empty_and_graphrag_and_analytics(monkeypatch, tmp_path):
    from src.config import settings
    from src.ui import ingest_adapter as mod

    # Route data dir to tmp
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "analytics_enabled", True)

    # DocumentProcessor async stub
    class _Proc:
        async def process_document_async(self, _p):  # pylint: disable=unused-argument
            el = SimpleNamespace(text="content", metadata={"k": 1})
            return SimpleNamespace(elements=[el])

    monkeypatch.setattr(mod, "DocumentProcessor", lambda settings: _Proc())

    # Vector store + LI stubs
    class _VS:
        pass

    called = {"vs": 0, "vsi": 0, "sc": 0}

    monkeypatch.setattr(
        mod,
        "create_vector_store",
        lambda *_, **__: called.__setitem__("vs", called["vs"] + 1) or _VS(),
    )

    class _SC:
        @staticmethod
        def from_defaults(vector_store):  # pylint: disable=unused-argument
            called["sc"] += 1
            return object()

    monkeypatch.setattr(mod, "StorageContext", _SC)

    class _VSI:
        @staticmethod
        def from_documents(_docs, storage_context):  # pylint: disable=unused-argument
            called["vsi"] += 1
            return object()

    monkeypatch.setattr(mod, "VectorStoreIndex", _VSI)

    # GraphRAG stubs
    class _PG:
        pass

    monkeypatch.setattr(mod, "create_property_graph_index", lambda *_, **__: _PG())
    monkeypatch.setattr(
        mod, "get_export_seed_ids", lambda *_a, **_k: ["a", "b"]
    )  # deterministic

    exports = {"parquet": 0, "jsonl": 0}

    def _exp_parq(*_a, **_k):
        exports["parquet"] += 1

    def _exp_jsonl(*_a, **_k):
        exports["jsonl"] += 1

    monkeypatch.setattr(mod, "export_graph_parquet", _exp_parq)
    monkeypatch.setattr(mod, "export_graph_jsonl", _exp_jsonl)

    # Analytics and telemetry stubs
    class _AMgr:
        def __init__(self):
            self.logged = []

        def log_embedding(self, **kw):  # pylint: disable=unused-argument
            self.logged.append(kw)

        @classmethod
        def instance(cls, *_a, **_k):
            return cls()

    monkeypatch.setattr(mod, "AnalyticsManager", _AMgr)

    telem_calls: list[dict] = []
    monkeypatch.setattr(mod, "log_jsonl", lambda payload: telem_calls.append(payload))

    # Run ingest
    files = [_FakeUpload("a.txt", b"x"), _FakeUpload("b.txt", b"y")]
    out = mod.ingest_files(files, enable_graphrag=True)

    assert out["count"] == 2
    assert out["pg_index"] is not None
    # Vector index and LI wiring invoked
    assert called["vs"] >= 1
    assert called["vsi"] >= 1
    assert called["sc"] >= 1
    # Exports performed and telemetry logged
    assert exports["parquet"] == 1
    assert exports["jsonl"] == 1
    assert any(d.get("export_performed") for d in telem_calls)
