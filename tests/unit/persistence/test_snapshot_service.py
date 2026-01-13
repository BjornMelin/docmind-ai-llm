"""Unit tests for snapshot_service helpers (library-free boundary)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.persistence import snapshot_service as svc

pytestmark = pytest.mark.unit


def test_init_callbacks_defaults_are_noops() -> None:
    log_cb, metric_cb = svc._init_callbacks(None, None)
    assert callable(log_cb)
    assert callable(metric_cb)
    log_cb({"x": 1})
    metric_cb("name", duration_ms=1.0)


def test_persist_indices_requires_vector_index(tmp_path: Path) -> None:
    mgr = SimpleNamespace(persist_vector_index=lambda *_a, **_k: None)
    with pytest.raises(TypeError):
        svc._persist_indices(mgr, tmp_path, None, None)


def test_persist_indices_persists_graph_store_when_present(tmp_path: Path) -> None:
    calls: list[str] = []

    class _Mgr:
        def persist_vector_index(self, *_a, **_k):  # type: ignore[no-untyped-def]
            calls.append("vec")

        def persist_graph_store(self, *_a, **_k):  # type: ignore[no-untyped-def]
            calls.append("graph")

    pg_index = SimpleNamespace(property_graph_store=object(), storage_context=object())
    graph_store, storage_context, pg_out = svc._persist_indices(
        _Mgr(), tmp_path, vector_index=object(), pg_index=pg_index
    )
    assert calls == ["vec", "graph"]
    assert graph_store is pg_index.property_graph_store
    assert storage_context is pg_index.storage_context
    assert pg_out is pg_index


def test_export_graphs_handles_missing_output_and_failures(
    monkeypatch, tmp_path: Path
) -> None:
    graph_config = ModuleType("src.retrieval.graph_config")
    graph_config.get_export_seed_ids = lambda *_a, **_k: ["0"]  # type: ignore[attr-defined]

    def _raise(**_k):  # type: ignore[no-untyped-def]
        raise RuntimeError("fail")

    graph_config.export_graph_jsonl = _raise  # type: ignore[attr-defined]
    graph_config.export_graph_parquet = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.retrieval.graph_config", graph_config)

    seen: list[dict] = []
    out = svc._export_graphs(
        workspace=tmp_path,
        pg_index=object(),
        vector_index=object(),
        graph_store=object(),
        storage_context=object(),
        settings_obj=SimpleNamespace(graphrag_cfg=SimpleNamespace(export_seed_cap=1)),
        log_export_event=lambda payload: seen.append(payload),
        record_graph_export_metric=lambda *_a, **_k: None,
    )
    assert out == []
    assert any(ev.get("export_performed") is False for ev in seen)


def test_build_versions_falls_back_to_unknown_embed_model(
    monkeypatch, tmp_path: Path
) -> None:
    li_core = ModuleType("llama_index.core")
    li_core.Settings = SimpleNamespace(embed_model=None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.core", li_core)

    settings_obj = SimpleNamespace(
        app_version="x",
        data_dir=tmp_path,
        database=SimpleNamespace(client_version="1", vector_store_type="qdrant"),
    )
    vector_index = object()
    versions = svc._build_versions(settings_obj, vector_index, embed_model=None)
    assert versions["app"] == "x"
    assert versions["embed_model"] == "unknown"
    assert versions["vector_client"] == "1"


def test_rebuild_snapshot_cleans_up_on_failure(monkeypatch, tmp_path: Path) -> None:
    cleaned: list[Path] = []

    class _Mgr:
        def __init__(self, _root: Path) -> None:
            self.workspace = tmp_path / "ws"

        def begin_snapshot(self) -> Path:
            return self.workspace

        def cleanup_tmp(self, ws: Path) -> None:
            cleaned.append(ws)

    monkeypatch.setattr(svc, "SnapshotManager", _Mgr)
    monkeypatch.setattr(
        svc,
        "_persist_indices",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    settings_obj = SimpleNamespace(
        app_version="x",
        data_dir=tmp_path,
        database=SimpleNamespace(vector_store_type="qdrant"),
    )
    with pytest.raises(RuntimeError):
        svc.rebuild_snapshot(
            vector_index=object(), pg_index=None, settings_obj=settings_obj
        )
    assert cleaned == [tmp_path / "ws"]
