"""Unit tests for Documents page helper functions (02_documents.py)."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from src.models.processing import IngestionInput
from src.persistence.snapshot_service import SnapshotActivation

pytestmark = pytest.mark.unit

_PHYSICAL_COLLECTIONS = {
    "text": "physical-text-v2",
    "image": "physical-image-v2",
}


@pytest.fixture(autouse=True)
def streamlit_calls(monkeypatch):
    import streamlit as st  # type: ignore

    st.session_state.clear()

    calls: dict[str, list[str]] = {
        "writes": [],
        "infos": [],
        "warnings": [],
        "errors": [],
        "success": [],
        "captions": [],
    }

    class _Col:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_a):  # type: ignore[no-untyped-def]
            return False

        def metric(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return None

        def number_input(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return 1

        def checkbox(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return False

        def button(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return False

    class _Status:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_a):  # type: ignore[no-untyped-def]
            return False

        def update(self, **_kwargs):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr(
        st, "write", lambda s: calls["writes"].append(str(s)), raising=False
    )
    monkeypatch.setattr(
        st, "info", lambda s: calls["infos"].append(str(s)), raising=False
    )
    monkeypatch.setattr(
        st, "warning", lambda s: calls["warnings"].append(str(s)), raising=False
    )
    monkeypatch.setattr(
        st, "error", lambda s: calls["errors"].append(str(s)), raising=False
    )
    monkeypatch.setattr(
        st, "success", lambda s: calls["success"].append(str(s)), raising=False
    )
    monkeypatch.setattr(
        st, "caption", lambda s: calls["captions"].append(str(s)), raising=False
    )
    monkeypatch.setattr(st, "toast", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "json", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "markdown", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "subheader", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "divider", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "slider", lambda *_a, **_k: 1, raising=False)
    monkeypatch.setattr(
        st, "columns", lambda n: [_Col() for _ in range(int(n))], raising=False
    )
    monkeypatch.setattr(st, "button", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(st, "selectbox", lambda *_a, **_k: "", raising=False)
    monkeypatch.setattr(
        st, "form", lambda *_a, **_k: contextlib.nullcontext(), raising=False
    )
    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        st, "form_submit_button", lambda *_a, **_k: False, raising=False
    )
    monkeypatch.setattr(
        st, "expander", lambda *_a, **_k: contextlib.nullcontext(), raising=False
    )
    monkeypatch.setattr(st, "status", lambda *_a, **_k: _Status(), raising=False)
    return calls


def test_filter_group_and_page_parsers() -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")

    exports = [
        {"content_type": "image/png", "metadata": {"doc_id": "d1", "page_no": "2"}},
        {"content_type": "text/plain"},
        {"content_type": "image/jpeg", "metadata": {"document_id": "d2", "page": 1}},
        "bad",
    ]
    imgs = page._filter_image_exports(exports)  # type: ignore[attr-defined]
    assert len(imgs) == 2

    grouped = page._group_exports_by_doc(imgs)  # type: ignore[attr-defined]
    assert set(grouped.keys()) == {"d1", "d2"}
    assert page._page_no({"page_no": "2"}) == 2  # type: ignore[attr-defined]
    assert page._page_no({"page_no": "x"}) == 0  # type: ignore[attr-defined]


def test_handle_ingest_submission_no_files(monkeypatch, streamlit_calls) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    page._handle_ingest_submission(  # type: ignore[attr-defined]
        None,
        use_graphrag=False,
        encrypt_images=False,
        parsing_overrides=page.ParsingOverrides(),
        owner_id="owner",
    )
    assert streamlit_calls["warnings"] == ["No files selected."]


def test_render_ingest_form_smoke(monkeypatch) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")

    files = [object()]
    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: files, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(st, "form_submit_button", lambda *_a, **_k: True, raising=False)

    out_files, use_graphrag, encrypt_images, parsing_overrides, submitted = (
        page._render_ingest_form()  # type: ignore[attr-defined]
    )
    assert out_files == files
    assert use_graphrag is True
    assert encrypt_images is True
    assert isinstance(parsing_overrides, page.ParsingOverrides)
    assert submitted is True


def test_render_parsing_overrides_disables_controls_for_global_defaults(
    monkeypatch,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    calls: list[tuple[str, bool | None]] = []

    def _checkbox(label: str, **kwargs: object) -> bool:
        calls.append((label, kwargs.get("disabled")))  # type: ignore[arg-type]
        return label == "Use global parsing defaults"

    monkeypatch.setattr(st, "checkbox", _checkbox, raising=False)

    result = page._render_parsing_overrides()  # type: ignore[attr-defined]

    assert result == page.ParsingOverrides()
    assert calls == [
        ("Use global parsing defaults", None),
        ("Force RapidOCR", True),
        ("Export searchable PDF", True),
    ]


def test_handle_ingest_submission_starts_job(monkeypatch) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")

    st.session_state.clear()

    calls: dict[str, object] = {}

    class _DummyJobManager:
        def start_job(self, *, owner_id: str, fn):  # type: ignore[no-untyped-def]
            calls["owner_id"] = owner_id
            calls["fn"] = fn
            return "job-1"

    monkeypatch.setattr(page, "get_job_manager", lambda *_a, **_k: _DummyJobManager())
    monkeypatch.setattr(
        page,
        "save_uploaded_file",
        lambda file_obj, **_kwargs: (Path("/tmp/doc.txt"), "a" * 64),
    )
    monkeypatch.setattr(page, "_get_spacy_service", lambda *_a, **_k: None)

    upload = SimpleNamespace(
        name="doc.txt",
        size=3,
        getbuffer=lambda: memoryview(b"doc"),
    )
    page._handle_ingest_submission(  # type: ignore[attr-defined]
        [upload],
        use_graphrag=False,
        encrypt_images=False,
        parsing_overrides={"force_ocr": True},
        owner_id="owner",
    )

    assert st.session_state.get("ingest_job_id") == "job-1"
    assert calls.get("owner_id") == "owner"
    assert callable(calls.get("fn"))


def test_handle_ingest_submission_rejects_duplicate_ids_before_persistence(
    monkeypatch,
    streamlit_calls,
) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    payload = b"same document"
    uploads = [
        SimpleNamespace(
            name=name,
            size=len(payload),
            getbuffer=lambda payload=payload: memoryview(payload),
        )
        for name in ("first.txt", "second.txt")
    ]

    monkeypatch.setattr(
        page,
        "save_uploaded_file",
        lambda _file: pytest.fail("upload persisted before duplicate preflight"),
    )
    monkeypatch.setattr(
        page,
        "_load_optional_spacy_service",
        lambda: pytest.fail("setup ran before duplicate preflight"),
    )

    page._handle_ingest_submission(  # type: ignore[attr-defined]
        uploads,
        use_graphrag=False,
        encrypt_images=False,
        parsing_overrides=page.ParsingOverrides(),
        owner_id="owner",
    )

    assert streamlit_calls["errors"] == ["Failed to start ingestion job (ValueError)."]


def test_render_ingest_results_sets_session_state(monkeypatch, tmp_path: Path) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    monkeypatch.setattr(page, "_render_image_exports", lambda _e: None)
    router_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _build_router(*args, **kwargs):  # type: ignore[no-untyped-def]
        router_calls.append((args, kwargs))
        return "R"

    monkeypatch.setattr(page, "build_router_engine", _build_router)

    st.session_state.clear()
    first_resource = page.VectorIndexResource("V")
    page._render_ingest_results(  # type: ignore[attr-defined]
        {
            "count": 2,
            "vector_index": "V",
            "vector_resource": first_resource,
            "pg_index": "G",
            "collections": dict(_PHYSICAL_COLLECTIONS),
        },
        use_graphrag=True,
    )
    assert st.session_state["vector_index"] == "V"
    assert st.session_state["graphrag_index"] == "G"
    assert st.session_state["router_engine"] == "R"

    # A requested rebuild that produces no graph clears the prior graph owner.
    st.session_state["graphrag_index"] = "G"
    second_resource = page.VectorIndexResource("V2")
    page._render_ingest_results(  # type: ignore[attr-defined]
        {
            "count": 1,
            "vector_index": "V2",
            "vector_resource": second_resource,
            "collections": dict(_PHYSICAL_COLLECTIONS),
        },
        use_graphrag=True,
    )
    assert st.session_state.get("graphrag_index") is None
    assert first_resource.closed
    assert [call[1] for call in router_calls] == [
        {
            "text_collection": "physical-text-v2",
            "image_collection": "physical-image-v2",
        },
        {
            "text_collection": "physical-text-v2",
            "image_collection": "physical-image-v2",
        },
    ]
    assert st.session_state["_snapshot_collections"] == _PHYSICAL_COLLECTIONS


def test_render_image_exports_calls_renderer(monkeypatch) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    called: list[int] = []
    monkeypatch.setattr(
        page, "_render_export_images", lambda items, limit: called.append(len(items))
    )

    page._render_image_exports(  # type: ignore[attr-defined]
        [
            {"content_type": "image/png", "metadata": {"doc_id": "d1", "page_no": 1}},
            {"content_type": "image/png", "metadata": {"doc_id": "d1", "page_no": 2}},
        ]
    )
    assert called == [2]


def test_render_export_images_handles_missing_ref(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")

    class _Store:
        def resolve_path(self, _ref):  # type: ignore[no-untyped-def]
            return tmp_path / "x.png"

    monkeypatch.setattr(page.ArtifactStore, "from_settings", lambda _s: _Store())
    monkeypatch.setattr(st, "image", lambda *_a, **_k: None, raising=False)

    page._render_export_images(  # type: ignore[attr-defined]
        [{"metadata": {"doc_id": "d1", "page_no": 1}}],
        preview_limit=1,
    )
    assert any("no artifact ref" in c for c in streamlit_calls["captions"])


def test_render_maintenance_controls_no_uploads(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    page._render_maintenance_controls(owner_id="owner")  # type: ignore[attr-defined]
    assert "No uploaded files." in streamlit_calls["captions"][-1]


def test_render_maintenance_controls_triggers_handlers(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / "a.pdf"
    target.write_bytes(b"%PDF-1.4\n%fake\n")

    hits: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        page,
        "_start_existing_corpus_rebuild",
        lambda **kwargs: hits.append(("rebuild", kwargs)),
    )
    monkeypatch.setattr(
        page,
        "_start_upload_deletion",
        lambda **kwargs: hits.append(("delete", kwargs)),
    )

    def _columns(_n: int):  # type: ignore[no-untyped-def]
        class _C:
            def __enter__(self):  # type: ignore[no-untyped-def]
                return self

            def __exit__(self, *_a):  # type: ignore[no-untyped-def]
                return False

            def checkbox(self, *_a, **_k):  # type: ignore[no-untyped-def]
                return False

            def button(self, label, **_k):  # type: ignore[no-untyped-def]
                return str(label) == "Rebuild"

        return [_C(), _C()]

    monkeypatch.setattr(st, "columns", _columns, raising=False)
    monkeypatch.setattr(st, "selectbox", lambda *_a, **_k: target.name, raising=False)
    monkeypatch.setattr(
        st,
        "checkbox",
        lambda label, **_k: "cannot be undone" in str(label).lower(),
        raising=False,
    )
    monkeypatch.setattr(
        st, "button", lambda label, **_k: str(label) == "Delete", raising=False
    )

    page._render_maintenance_controls(owner_id="owner")  # type: ignore[attr-defined]
    assert hits == [
        (
            "rebuild",
            {"uploads_dir": uploads, "encrypt": False, "owner_id": "owner"},
        ),
        (
            "delete",
            {"target": target, "encrypt": False, "owner_id": "owner"},
        ),
    ]


def test_doc_id_for_upload_uses_full_file_content(tmp_path: Path) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    doc_id = page._doc_id_for_upload(p)  # type: ignore[attr-defined]
    assert doc_id.startswith("doc-")
    assert len(doc_id) == 4 + 64


def test_doc_id_changes_for_same_size_same_mtime_replacement(tmp_path: Path) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    path = tmp_path / "mutable.txt"
    path.write_text("first", encoding="utf-8")
    original_mtime = path.stat().st_mtime_ns
    first = page._doc_id_for_upload(path)  # type: ignore[attr-defined]

    path.write_text("other", encoding="utf-8")
    os.utime(path, ns=(original_mtime, original_mtime))
    second = page._doc_id_for_upload(path)  # type: ignore[attr-defined]

    assert first != second


def test_start_upload_deletion_refuses_outside_uploads(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    page._start_upload_deletion(  # type: ignore[attr-defined]
        target=outside,
        encrypt=False,
        owner_id="owner",
    )
    assert streamlit_calls["errors"][-1].startswith("Refusing to delete a path")


def test_start_upload_deletion_retains_source_until_generation_commits(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / "a.txt"
    target.write_text("hello", encoding="utf-8")
    retained = uploads / "b.txt"
    retained.write_text("keep", encoding="utf-8")

    nlp_service = object()
    calls: list[tuple[list[IngestionInput], dict[str, object]]] = []
    monkeypatch.setattr(page, "_load_optional_spacy_service", lambda: nlp_service)
    monkeypatch.setattr(
        page,
        "_start_ingestion_job",
        lambda inputs, **kwargs: calls.append((inputs, kwargs)),
    )

    page._start_upload_deletion(  # type: ignore[attr-defined]
        target=target,
        encrypt=True,
        owner_id="owner",
    )

    assert target.is_file()
    assert retained.is_file()
    assert len(calls) == 1
    inputs, kwargs = calls[0]
    assert [item.source_path for item in inputs] == [retained]
    assert all(item.encrypt_images for item in inputs)
    assert kwargs["encrypt_images"] is True
    assert kwargs["nlp_service"] is nlp_service
    assert kwargs["owner_id"] == "owner"
    assert kwargs["quarantine_source"] == target
    assert kwargs["excluded_source_paths"] == (target,)
    assert streamlit_calls["infos"][-1].startswith("Deletion scheduled")


def test_render_latest_snapshot_summary(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(
        page,
        "load_manifest",
        lambda *_, **__: {
            "created_at": "now",
            "corpus_hash": "c" * 20,
            "config_hash": "d" * 20,
        },
    )
    page._render_latest_snapshot_summary()  # type: ignore[attr-defined]
    assert any("Latest snapshot" in c for c in streamlit_calls["captions"])


def test_render_manifest_details_and_log_export_event(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")

    events: list[dict] = []
    monkeypatch.setattr("src.utils.telemetry.log_jsonl", lambda ev: events.append(ev))

    page._render_manifest_details(  # type: ignore[attr-defined]
        {
            "corpus_hash": "c" * 20,
            "config_hash": "d" * 20,
            "versions": {"app": "x"},
            "graph_exports": [
                {"filename": "g.jsonl", "format": "jsonl", "size_bytes": 1}
            ],
        },
        snapshot_dir=tmp_path,
    )
    page._log_export_event(
        {
            "export_performed": True,
            "dest_path": str(tmp_path / "a/b/c.txt"),
        }
    )  # type: ignore[attr-defined]
    assert events
    assert "dest_path" not in events[-1]
    assert events[-1]["dest_basename"] == "c.txt"


def test_handle_manual_export_smoke(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    st.session_state["graphrag_index"] = object()
    st.session_state["vector_index"] = object()

    monkeypatch.setattr(page, "get_export_seed_ids", lambda *_a, **_k: ["0", "1"])
    out_file = tmp_path / "out.jsonl"
    monkeypatch.setattr(page, "timestamped_export_path", lambda _d, _e, **_k: out_file)
    monkeypatch.setattr(
        page,
        "export_graph_jsonl",
        lambda **_k: out_file.write_text("x", encoding="utf-8"),
    )
    monkeypatch.setattr(
        page,
        "export_graph_parquet",
        lambda **_k: out_file.write_text("x", encoding="utf-8"),
    )
    monkeypatch.setattr(page, "record_graph_export_metric", lambda *_a, **_k: None)
    monkeypatch.setattr(page, "_log_export_event", lambda _p: None)

    page._handle_manual_export(tmp_path, "jsonl")  # type: ignore[attr-defined]
    assert streamlit_calls["success"]


def test_ingest_job_closes_vector_resource_when_snapshot_fails(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib
    import threading

    page = importlib.import_module("src.pages.02_documents")

    class _Client:
        closed = False

        def close(self) -> None:
            self.closed = True

    client = _Client()
    resource = page.VectorIndexResource(object(), client=client)
    workspace = tmp_path / "storage" / "_tmp-build123"
    collections = dict(_PHYSICAL_COLLECTIONS)
    collection_metadata = {
        "text": {"snapshot_id": "build123", "role": "text"},
        "image": {"snapshot_id": "build123", "role": "image"},
    }

    class _Manager:
        def __init__(self) -> None:
            self.cleanup_calls: list[Path] = []

        def begin_snapshot(self) -> Path:
            workspace.mkdir(parents=True)
            return workspace

        def cleanup_tmp(self, path: Path) -> None:
            self.cleanup_calls.append(path)

    manager = _Manager()
    ingest_calls: list[dict[str, object]] = []
    rebuild_calls: list[object] = []
    deleted_collections: list[dict[str, str]] = []

    def _collection_names(path: Path) -> dict[str, str]:
        assert path == workspace
        return collections

    monkeypatch.setattr(page, "SnapshotManager", lambda _base_dir: manager)
    monkeypatch.setattr(page, "_physical_collection_names", _collection_names)
    monkeypatch.setattr(
        page,
        "_read_collection_metadata",
        lambda names: collection_metadata if names == collections else {},
    )
    monkeypatch.setattr(
        page,
        "_delete_staged_collections",
        lambda names: deleted_collections.append(dict(names)),
    )

    def _ingest_inputs(*_args, **kwargs):  # type: ignore[no-untyped-def]
        ingest_calls.append(kwargs)
        return {
            "vector_index": resource.index,
            "vector_resource": resource,
            "pg_index": None,
            "activation_corpus_hash": "c" * 64,
            "activation_config": {"x": 1},
            "activation_config_hash": "f" * 64,
            "snapshot_config_hash": "e" * 64,
            "collections": dict(collections),
        }

    monkeypatch.setattr(
        page,
        "ingest_inputs",
        _ingest_inputs,
    )

    def _rebuild_snapshot(*args, **_kwargs):  # type: ignore[no-untyped-def]
        rebuild_calls.append(args[3])
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(
        page,
        "rebuild_snapshot",
        _rebuild_snapshot,
    )

    with pytest.raises(RuntimeError, match="snapshot failed"):
        page._run_ingest_job(  # type: ignore[attr-defined]
            [],
            use_graphrag=False,
            encrypt_images=False,
            nlp_service=None,
            cancel_event=threading.Event(),
            report_progress=lambda _event: None,
        )

    assert client.closed
    assert manager.cleanup_calls == [workspace]
    assert deleted_collections == [collections]
    assert ingest_calls[0]["text_collection_name"] == "physical-text-v2"
    assert ingest_calls[0]["image_collection_name"] == "physical-image-v2"
    activation = cast(SnapshotActivation, rebuild_calls[0])
    assert isinstance(activation, SnapshotActivation)
    assert activation.manager is manager
    assert activation.workspace == workspace
    assert activation.text_collection == "physical-text-v2"
    assert activation.image_collection == "physical-image-v2"
    assert activation.collection_metadata == collection_metadata


def test_render_ingest_results_replaces_and_clears_router(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(page, "_render_image_exports", lambda _e: None)

    class _Router:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    old = _Router()
    new = _Router()
    st.session_state.clear()
    st.session_state["router_engine"] = old
    monkeypatch.setattr(page, "build_router_engine", lambda *_a, **_k: new)
    resource = page.VectorIndexResource(object())

    page._render_ingest_results(
        {
            "count": 1,
            "vector_index": resource.index,
            "vector_resource": resource,
            "collections": dict(_PHYSICAL_COLLECTIONS),
        },
        use_graphrag=False,
    )

    assert old.closed == 1
    assert new.closed == 0
    assert st.session_state["router_engine"] is new


def test_render_ingest_results_build_failure_clears_router(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(page, "_render_image_exports", lambda _e: None)

    class _Router:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    old = _Router()

    class _Client:
        closed = False

        def close(self) -> None:
            self.closed = True

    client = _Client()
    resource = page.VectorIndexResource(object(), client=client)
    st.session_state.clear()
    st.session_state["router_engine"] = old
    monkeypatch.setattr(
        page,
        "build_router_engine",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("router failed")),
    )

    with pytest.raises(RuntimeError, match="router failed"):
        page._render_ingest_results(
            {
                "count": 1,
                "vector_index": resource.index,
                "vector_resource": resource,
                "collections": dict(_PHYSICAL_COLLECTIONS),
            },
            use_graphrag=False,
        )

    assert old.closed == 1
    assert st.session_state["router_engine"] is None
    assert client.closed


def test_render_export_images_handles_encrypted_without_support(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")

    class _Store:
        def resolve_path(self, _ref):  # type: ignore[no-untyped-def]
            return tmp_path / "x.webp.enc"

    monkeypatch.setattr(page.ArtifactStore, "from_settings", lambda _s: _Store())
    monkeypatch.setattr(st, "image", lambda *_a, **_k: None, raising=False)

    images_mod = ModuleType("src.utils.images")
    monkeypatch.setitem(sys.modules, "src.utils.images", images_mod)

    page._render_export_images(  # type: ignore[attr-defined]
        [
            {
                "metadata": {
                    "doc_id": "d1",
                    "page_no": 1,
                    "image_artifact_id": "a" * 64,
                    "image_artifact_suffix": ".webp.enc",
                }
            }
        ],
        preview_limit=1,
    )
    assert any(
        "Encryption support unavailable" in c for c in streamlit_calls["captions"]
    )


def test_start_existing_corpus_rebuild_requires_uploads(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    page._start_existing_corpus_rebuild(  # type: ignore[attr-defined]
        uploads_dir=uploads,
        encrypt=False,
        owner_id="owner",
    )
    assert streamlit_calls["infos"] == [
        "Add a document before rebuilding the search index."
    ]


def test_start_existing_corpus_rebuild_schedules_full_generation(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib

    from src.config.settings import settings
    from src.models.processing import IngestionInput

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    pdf = uploads / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    nlp_service = object()
    calls: list[tuple[list[IngestionInput], dict[str, object]]] = []
    monkeypatch.setattr(page, "_load_optional_spacy_service", lambda: nlp_service)
    monkeypatch.setattr(
        page,
        "_start_ingestion_job",
        lambda inputs, **kwargs: calls.append((inputs, kwargs)),
    )

    page._start_existing_corpus_rebuild(  # type: ignore[attr-defined]
        uploads_dir=uploads,
        encrypt=False,
        owner_id="owner",
    )

    assert len(calls) == 1
    inputs, kwargs = calls[0]
    assert len(inputs) == 1
    assert isinstance(inputs[0], IngestionInput)
    assert inputs[0].source_path == pdf
    assert inputs[0].metadata == {}
    assert kwargs["encrypt_images"] is False
    assert kwargs["nlp_service"] is nlp_service
    assert kwargs["owner_id"] == "owner"


def test_start_upload_deletion_missing_file_warns(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    missing = uploads / "missing.txt"
    page._start_upload_deletion(  # type: ignore[attr-defined]
        target=missing,
        encrypt=False,
        owner_id="owner",
    )
    assert streamlit_calls["warnings"][-1] == "File not found."


def test_log_export_event_without_dest_path(monkeypatch) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    events: list[dict] = []
    monkeypatch.setattr("src.utils.telemetry.log_jsonl", lambda ev: events.append(ev))

    page._log_export_event({"export_performed": True, "context": "manual"})  # type: ignore[attr-defined]
    assert events
    assert "dest_basename" not in events[-1]


def test_handle_manual_export_parquet(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    st.session_state["graphrag_index"] = object()
    st.session_state["vector_index"] = object()

    monkeypatch.setattr(page, "get_export_seed_ids", lambda *_a, **_k: ["0"])
    out_file = tmp_path / "out.parquet"
    monkeypatch.setattr(page, "timestamped_export_path", lambda _d, _e, **_k: out_file)
    monkeypatch.setattr(
        page,
        "export_graph_parquet",
        lambda **_k: out_file.write_text("x", encoding="utf-8"),
    )
    monkeypatch.setattr(page, "record_graph_export_metric", lambda *_a, **_k: None)
    monkeypatch.setattr(page, "_log_export_event", lambda _p: None)

    page._handle_manual_export(tmp_path, "parquet")  # type: ignore[attr-defined]
    assert streamlit_calls["success"]
