"""Unit tests for snapshot_service helpers (library-free boundary)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

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
        svc._persist_indices(cast(svc.SnapshotManager, mgr), tmp_path, None, None)


def test_persist_indices_persists_graph_store_when_present(tmp_path: Path) -> None:
    calls: list[str] = []

    class _Mgr:
        def persist_vector_index(self, *_a, **_k):  # type: ignore[no-untyped-def]
            calls.append("vec")

        def persist_graph_storage_context(  # type: ignore[no-untyped-def]
            self, *_a, **_k
        ):
            calls.append("graph")

    pg_index = SimpleNamespace(property_graph_store=object(), storage_context=object())
    graph_store, storage_context, pg_out = svc._persist_indices(
        cast(svc.SnapshotManager, _Mgr()),
        tmp_path,
        vector_index=cast(svc.VectorIndexProtocol, object()),
        pg_index=cast(svc.PgIndexProtocol, pg_index),
    )
    assert calls == ["graph"]
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
        pg_index=cast(svc.PgIndexProtocol, object()),
        vector_index=cast(svc.VectorIndexProtocol, object()),
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
    monkeypatch.setattr(svc, "get_version", lambda: "x")
    monkeypatch.setattr(svc.metadata, "version", lambda _dist: "0.14.21")

    settings_obj = SimpleNamespace(
        data_dir=tmp_path,
        database=SimpleNamespace(client_version="1", vector_store_type="qdrant"),
    )
    vector_index = cast(svc.VectorIndexProtocol, object())
    versions = svc._build_versions(settings_obj, vector_index, embed_model=None)
    assert versions["app"] == "x"
    assert versions["llama_index"] == "0.14.21"
    assert versions["embed_model"] == "unknown"
    assert versions["vector_client"] == "1"


def test_rebuild_snapshot_success_writes_and_finalizes_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    manifest: dict[str, object] = {}
    workspace = tmp_path / "workspace"
    final = tmp_path / "snapshot-final"

    class _Mgr:
        def begin_snapshot(self) -> Path:
            raise AssertionError("caller must open the workspace")

        def write_manifest(self, target: Path, **metadata: object) -> None:
            calls.append("manifest")
            assert target == workspace
            manifest.update(metadata)

        def finalize_snapshot(self, target: Path) -> svc.FinalizedSnapshot:
            calls.append("finalize")
            assert target == workspace
            return svc.FinalizedSnapshot(
                path=final,
                manifest={"corpus_hash": "corpus"},
            )

        def cleanup_tmp(self, _target: Path) -> None:
            raise AssertionError("caller owns workspace cleanup")

    uploads = tmp_path / "uploads"
    document = uploads / "document.txt"
    workspace.mkdir()
    manager = cast(svc.SnapshotManager, _Mgr())
    monkeypatch.setattr(
        svc,
        "_collect_corpus_paths",
        lambda _settings: ([document], uploads),
    )
    monkeypatch.setattr(
        svc,
        "compute_corpus_hash",
        lambda paths, *, base_dir: (
            "corpus" if paths == [document] and base_dir == uploads else "unexpected"
        ),
    )
    monkeypatch.setattr(svc, "compute_config_hash", lambda cfg: f"config-{cfg['x']}")
    monkeypatch.setattr(
        svc,
        "_build_versions",
        lambda *_args: {"app": "test"},
    )
    settings_obj = SimpleNamespace(
        data_dir=tmp_path,
        database=SimpleNamespace(vector_store_type="qdrant"),
        graphrag_cfg=SimpleNamespace(enabled=False),
    )

    result = svc.rebuild_snapshot(
        vector_index=cast(
            svc.VectorIndexProtocol,
            SimpleNamespace(embed_model=None),
        ),
        pg_index=None,
        settings_obj=settings_obj,
        activation=svc.SnapshotActivation(
            manager=manager,
            workspace=workspace,
            text_collection="physical-text-v2",
            image_collection="physical-image-v2",
            expected_corpus_hash="corpus",
            expected_config_hash="config-1",
            activation_config={"x": 1},
            activation_config_hash="config-1",
            collection_metadata={"text": {"schema": "v2"}},
            graph_requested=False,
        ),
    )

    assert result.path == final
    assert result.manifest == {"corpus_hash": "corpus"}
    assert calls == ["manifest", "finalize"]
    assert manifest == {
        "index_id": "docmind",
        "graph_store_type": "none",
        "vector_store_type": "qdrant",
        "text_collection": "physical-text-v2",
        "image_collection": "physical-image-v2",
        "corpus_hash": "corpus",
        "config_hash": "config-1",
        "versions": {"app": "test"},
        "graph_exports": [],
        "collection_metadata": {"text": {"schema": "v2"}},
        "activation_config": {"x": 1},
        "activation_config_hash": "config-1",
    }


def test_rebuild_snapshot_leaves_failure_cleanup_to_caller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    class _Mgr:
        def begin_snapshot(self) -> Path:
            raise AssertionError("service must not open a workspace")

        def cleanup_tmp(self, _workspace: Path) -> None:
            raise AssertionError("service must not clean caller-owned workspace")

        def finalize_snapshot(self, _workspace: Path) -> svc.FinalizedSnapshot:
            raise AssertionError("failed payload must not finalize")

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
            vector_index=cast(svc.VectorIndexProtocol, object()),
            pg_index=None,
            settings_obj=settings_obj,
            activation=svc.SnapshotActivation(
                manager=cast(svc.SnapshotManager, _Mgr()),
                workspace=workspace,
                text_collection="physical-text-v2",
                image_collection="physical-image-v2",
                expected_corpus_hash="c" * 64,
                expected_config_hash=svc.compute_config_hash({"x": 1}),
                activation_config={"x": 1},
                activation_config_hash=svc.compute_config_hash({"x": 1}),
                collection_metadata={},
                graph_requested=False,
            ),
        )
    assert workspace.is_dir()


def test_requested_graph_failure_never_promotes_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing requested graph leaves the prior CURRENT owner untouched."""
    finalized: list[Path] = []
    workspace = tmp_path / "ws"
    workspace.mkdir()

    class _Mgr:
        def begin_snapshot(self) -> Path:
            raise AssertionError("service must not open a workspace")

        def cleanup_tmp(self, _workspace: Path) -> None:
            raise AssertionError("service must not clean caller-owned workspace")

        def finalize_snapshot(self, workspace: Path) -> svc.FinalizedSnapshot:
            finalized.append(workspace)
            return svc.FinalizedSnapshot(path=workspace, manifest={})

    settings_obj = SimpleNamespace(
        data_dir=tmp_path,
        database=SimpleNamespace(vector_store_type="qdrant"),
    )

    with pytest.raises(svc.SnapshotPersistenceError, match="GraphRAG was requested"):
        svc.rebuild_snapshot(
            vector_index=cast(svc.VectorIndexProtocol, object()),
            pg_index=None,
            settings_obj=settings_obj,
            activation=svc.SnapshotActivation(
                manager=cast(svc.SnapshotManager, _Mgr()),
                workspace=workspace,
                text_collection="physical-text-v2",
                image_collection="physical-image-v2",
                expected_corpus_hash="c" * 64,
                expected_config_hash=svc.compute_config_hash({"x": 1}),
                activation_config={"x": 1},
                activation_config_hash=svc.compute_config_hash({"x": 1}),
                collection_metadata={},
                graph_requested=True,
            ),
        )

    assert finalized == []
    assert workspace.is_dir()


def test_requested_graph_requires_persisted_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A non-null graph object cannot commit an empty graph generation."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(
        svc,
        "_persist_indices",
        lambda *_a, **_k: (object(), object(), object()),
    )

    with pytest.raises(svc.SnapshotPersistenceError, match="produced no payload"):
        svc.rebuild_snapshot(
            vector_index=cast(svc.VectorIndexProtocol, object()),
            pg_index=cast(svc.PgIndexProtocol, object()),
            settings_obj=SimpleNamespace(
                data_dir=tmp_path,
                database=SimpleNamespace(vector_store_type="qdrant"),
            ),
            activation=svc.SnapshotActivation(
                manager=cast(svc.SnapshotManager, object()),
                workspace=workspace,
                text_collection="physical-text-v2",
                image_collection="physical-image-v2",
                expected_corpus_hash="c" * 64,
                expected_config_hash=svc.compute_config_hash({"x": 1}),
                activation_config={"x": 1},
                activation_config_hash=svc.compute_config_hash({"x": 1}),
                collection_metadata={},
                graph_requested=True,
            ),
        )
