"""Unit tests for Documents page helper functions (02_documents.py)."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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


def _valid_manifest() -> dict[str, object]:
    return {
        "corpus_hash": "c" * 64,
        "config_hash": "f" * 64,
        "versions": {},
        "graph_exports": [],
    }


def test_committed_manifest_projects_at_documents_boundary(tmp_path: Path) -> None:
    """Every persistence-accepted manifest satisfies the UI projection contract."""
    import importlib

    from llama_index.core import StorageContext
    from llama_index.core.graph_stores import SimplePropertyGraphStore

    from src.persistence.snapshot import SnapshotManager
    from src.utils.hashing import sha256_file

    page = importlib.import_module("src.pages.02_documents")
    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    manager.persist_graph_storage_context(
        StorageContext.from_defaults(property_graph_store=SimplePropertyGraphStore()),
        workspace,
    )
    export = workspace / "graph" / "graph_export.jsonl"
    export.write_text("{}\n", encoding="utf-8")
    manager.write_manifest(
        workspace,
        index_id="projection-parity",
        graph_store_type="property_graph",
        vector_store_type="qdrant",
        text_collection="physical-text",
        image_collection="physical-image",
        corpus_hash="c" * 64,
        config_hash="f" * 64,
        versions={"app": "2.0", "schema": 2, "enabled": True, "optional": None},
        graph_exports=[
            {
                "filename": export.name,
                "format": "jsonl",
                "size_bytes": export.stat().st_size,
                "sha256": sha256_file(export),
            }
        ],
    )

    finalized = manager.finalize_snapshot(workspace)
    presentation = page._bounded_manifest_presentation(finalized.manifest)

    assert presentation == {
        "corpus_hash": "c" * 64,
        "config_hash": "f" * 64,
        "versions": {
            "app": "2.0",
            "enabled": "True",
            "optional": "None",
            "schema": "2",
        },
        "graph_exports": [
            {
                "filename": "graph_export.jsonl",
                "format": "jsonl",
                "size_bytes": 3,
            }
        ],
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


def test_foreign_corpus_mutation_rejects_ingest_before_prework(
    monkeypatch,
    streamlit_calls,
) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    manager = SimpleNamespace(
        is_exclusivity_active=lambda _key: True,
        activity_snapshot=lambda: SimpleNamespace(
            has_active_jobs=True,
            maintenance_active=False,
        ),
        exclusivity_activity_snapshot=lambda _key: (
            True,
            SimpleNamespace(has_active_jobs=True, maintenance_active=False),
        ),
    )
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        page,
        "_pdf_uploads_are_ready",
        lambda _files: pytest.fail("parser readiness ran during foreign mutation"),
    )
    monkeypatch.setattr(
        page,
        "_load_optional_spacy_service",
        lambda: pytest.fail("NLP setup ran during foreign mutation"),
    )
    monkeypatch.setattr(
        page,
        "_save_ingestion_inputs",
        lambda *_a, **_k: pytest.fail("upload persisted during foreign mutation"),
    )
    upload = SimpleNamespace(name="doc.txt", size=3)

    started = page._handle_ingest_submission(  # type: ignore[attr-defined]
        [upload],
        use_graphrag=False,
        encrypt_images=False,
        parsing_overrides=page.ParsingOverrides(),
        owner_id="owner-two",
    )

    assert started is False
    assert streamlit_calls["warnings"] == [
        "Another session is changing the corpus. Mutation controls are disabled."
    ]


def test_foreign_corpus_mutation_rejects_maintenance_before_hashing(
    monkeypatch,
    tmp_path: Path,
    streamlit_calls,
) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    manager = SimpleNamespace(
        is_exclusivity_active=lambda _key: True,
        activity_snapshot=lambda: SimpleNamespace(
            has_active_jobs=True,
            maintenance_active=False,
        ),
        exclusivity_activity_snapshot=lambda _key: (
            True,
            SimpleNamespace(has_active_jobs=True, maintenance_active=False),
        ),
    )
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        page,
        "_existing_corpus_inputs",
        lambda *_a, **_k: pytest.fail("corpus hashing ran during foreign mutation"),
    )

    started = page._start_existing_corpus_rebuild(  # type: ignore[attr-defined]
        uploads_dir=tmp_path,
        encrypt=False,
        owner_id="owner-two",
    )

    assert started is False
    assert streamlit_calls["warnings"] == [
        "Another session is changing the corpus. Mutation controls are disabled."
    ]


@pytest.mark.parametrize("status", ["queued", "running"])
def test_resolve_active_ingestion_job_tracks_only_active_states(
    monkeypatch,
    status: str,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    state = SimpleNamespace(status=status)
    manager = SimpleNamespace(get=lambda *_a, **_k: state)
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    st.session_state["ingest_job_id"] = "job-1"

    active = page._resolve_active_ingestion_job(  # type: ignore[attr-defined]
        owner_id="owner"
    )

    assert active == ("job-1", state)


def test_resolve_active_ingestion_job_preserves_terminal_state_for_rendering(
    monkeypatch,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    state = SimpleNamespace(status="succeeded")
    manager = SimpleNamespace(get=lambda *_a, **_k: state)
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    st.session_state["ingest_job_id"] = "job-1"

    active = page._resolve_active_ingestion_job(  # type: ignore[attr-defined]
        owner_id="owner"
    )

    assert active is None
    assert st.session_state["ingest_job_id"] == "job-1"


def test_resolve_active_ingestion_job_clears_missing_state(monkeypatch) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    manager = SimpleNamespace(get=lambda *_a, **_k: None)
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    st.session_state["ingest_job_id"] = "missing"
    st.session_state["ingest_job_last_event"] = object()
    st.session_state["ingest_job_cancel_requested_id"] = "missing"

    active = page._resolve_active_ingestion_job(  # type: ignore[attr-defined]
        owner_id="owner"
    )

    assert active is None
    assert "ingest_job_id" not in st.session_state
    assert "ingest_job_last_event" not in st.session_state
    assert "ingest_job_cancel_requested_id" not in st.session_state


def test_terminal_poll_requests_full_rerun_and_renders_notice_once(
    monkeypatch,
    streamlit_calls,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    state = SimpleNamespace(status="canceled", result=None, error=None)
    manager = SimpleNamespace(
        get=lambda *_a, **_k: state,
        drain_progress=lambda *_a, **_k: [],
        consume_terminal=lambda *_a, **_k: True,
        exclusivity_activity_snapshot=lambda _key: (
            False,
            SimpleNamespace(maintenance_active=False),
        ),
    )
    rerun_scopes: list[str] = []
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(st, "progress", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        st,
        "rerun",
        lambda *, scope="app": rerun_scopes.append(scope),
        raising=False,
    )
    st.session_state["ingest_job_id"] = "job-1"

    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )

    assert rerun_scopes == ["app"]
    assert "ingest_job_id" not in st.session_state
    assert st.session_state["ingest_job_completed_id"] == "job-1"
    assert "ingest_job_terminal_notice" in st.session_state
    assert st.session_state["corpus_activity_observed"] == (False, False)

    page._render_ingest_terminal_notice()  # type: ignore[attr-defined]
    assert streamlit_calls["warnings"] == ["Ingestion cancelled."]
    page._render_ingest_terminal_notice()  # type: ignore[attr-defined]
    assert streamlit_calls["warnings"] == ["Ingestion cancelled."]


@pytest.mark.parametrize("status", ["succeeded", "failed", "canceled"])
def test_terminal_capture_consumes_owned_manager_payload(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    consume_calls: list[tuple[str, str]] = []

    class _Manager:
        admission_quiescence = staticmethod(contextlib.nullcontext)

        def consume_terminal(self, job_id: str, *, owner_id: str) -> bool:
            consume_calls.append((job_id, owner_id))
            return True

    manager = _Manager()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        page,
        "_prepare_ingest_success_notice",
        lambda _result: {"status": "succeeded", "message": "done"},
    )
    st.session_state["ingest_job_id"] = "job-1"
    st.session_state["ingest_job_last_event"] = object()
    state = SimpleNamespace(
        status=status,
        result={} if status == "succeeded" else None,
        error="expected failure" if status == "failed" else None,
    )

    assert page._capture_ingest_terminal_state(
        state,
        owner_id="owner",
        job_id="job-1",
        completed_key="ingest_job_completed_id",
    )

    assert consume_calls == [("job-1", "owner")]
    assert st.session_state["ingest_job_completed_id"] == "job-1"
    assert "ingest_job_id" not in st.session_state
    assert "ingest_job_last_event" not in st.session_state
    assert "ingest_job_terminal_notice" in st.session_state
    assert st.session_state["corpus_activity_observed"] == (False, False)


def test_terminal_consume_failure_retains_tracking_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    state = SimpleNamespace(status="canceled", result=None, error=None)
    consume_results = iter((False, True))
    consume_calls: list[tuple[str, str]] = []

    class _Manager:
        def exclusivity_activity_snapshot(self, _key: str):  # type: ignore[no-untyped-def]
            return False, SimpleNamespace(maintenance_active=False)

        def get(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return state

        def drain_progress(self, *_args: object, **_kwargs: object) -> list[object]:
            return []

        def consume_terminal(self, job_id: str, *, owner_id: str) -> bool:
            consume_calls.append((job_id, owner_id))
            return next(consume_results)

    manager = _Manager()
    rerun_scopes: list[str] = []
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(st, "progress", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        st,
        "rerun",
        lambda *, scope="app": rerun_scopes.append(scope),
        raising=False,
    )
    st.session_state["ingest_job_id"] = "job-1"

    page._render_ingest_job_panel.__wrapped__(owner_id="owner")  # type: ignore[attr-defined]

    assert consume_calls == [("job-1", "owner")]
    assert rerun_scopes == ["app"]
    assert st.session_state["ingest_job_id"] == "job-1"
    assert st.session_state["ingest_job_completed_id"] == "job-1"

    page._render_ingest_job_panel.__wrapped__(owner_id="owner")  # type: ignore[attr-defined]

    assert consume_calls == [("job-1", "owner"), ("job-1", "owner")]
    assert "ingest_job_id" not in st.session_state
    assert st.session_state["corpus_activity_observed"] == (False, False)


@pytest.mark.parametrize(
    ("exception_name", "expected_copy"),
    [
        (
            "JobAdmissionPausedError",
            "Runtime maintenance is in progress. Live activation will retry "
            "automatically.",
        ),
        (
            "JobConflictError",
            "Other background work is active. Live activation will retry "
            "automatically.",
        ),
        (
            "ForegroundRuntimeConflictError",
            "The live runtime is in use. Live activation will retry automatically.",
        ),
    ],
)
def test_success_terminal_defers_without_consuming_when_lease_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    streamlit_calls: dict[str, list[str]],
    exception_name: str,
    expected_copy: str,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    exception_type = getattr(page, exception_name)

    class _Manager:
        @contextlib.contextmanager
        def admission_quiescence(self):  # type: ignore[no-untyped-def]
            raise exception_type("lease unavailable")
            yield

        def consume_terminal(self, *_args: object, **_kwargs: object) -> bool:
            pytest.fail("terminal state consumed without a runtime lease")

    monkeypatch.setattr(page, "get_job_manager", _Manager)
    st.session_state["ingest_job_id"] = "job-1"
    st.session_state["ingest_job_last_event"] = object()
    state = SimpleNamespace(status="succeeded", result={"ignored": True})

    consumed = page._capture_ingest_terminal_state(
        state,
        owner_id="owner",
        job_id="job-1",
        completed_key="ingest_job_completed_id",
    )

    assert consumed is False
    assert st.session_state["ingest_job_id"] == "job-1"
    assert "ingest_job_completed_id" not in st.session_state
    assert "ingest_job_terminal_notice" not in st.session_state
    assert streamlit_calls["infos"] == [expected_copy]


def test_success_terminal_lease_blocks_new_job_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.ui.background_jobs import JobManager

    page = importlib.import_module("src.pages.02_documents")
    manager = JobManager(max_workers=1)
    entered = threading.Event()
    release = threading.Event()

    def _prepare(_result: object) -> dict[str, object]:
        entered.set()
        assert release.wait(5)
        return {"status": "succeeded", "message": "done"}

    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(manager, "consume_terminal", lambda *_a, **_k: True)
    monkeypatch.setattr(page, "_prepare_ingest_success_notice", _prepare)
    st.session_state["ingest_job_id"] = "job-1"
    state = SimpleNamespace(status="succeeded", result={})

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                page._capture_ingest_terminal_state,
                state,
                owner_id="owner",
                job_id="job-1",
                completed_key="ingest_job_completed_id",
            )
            assert entered.wait(5)
            with pytest.raises(page.JobAdmissionPausedError):
                manager.start_job(owner_id="other", fn=lambda *_args: None)
            release.set()
            assert future.result(timeout=5) is True

        assert st.session_state["ingest_job_completed_id"] == "job-1"
        assert "ingest_job_id" not in st.session_state
    finally:
        release.set()
        manager._begin_shutdown_for_tests()
        manager._join_shutdown_for_tests()


@pytest.mark.parametrize("exclusivity_key", [None, "corpus-mutation"])
def test_success_terminal_retries_after_active_job_finishes(
    monkeypatch: pytest.MonkeyPatch,
    streamlit_calls: dict[str, list[str]],
    exclusivity_key: str | None,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.ui.background_jobs import JobManager

    page = importlib.import_module("src.pages.02_documents")
    manager = JobManager(max_workers=1)
    job_entered = threading.Event()
    release_job = threading.Event()

    def _job(*_args: object) -> None:
        job_entered.set()
        assert release_job.wait(5)

    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(manager, "consume_terminal", lambda *_a, **_k: True)
    monkeypatch.setattr(
        page,
        "_prepare_ingest_success_notice",
        lambda _result: {"status": "succeeded", "message": "done"},
    )
    st.session_state["ingest_job_id"] = "terminal-job"
    state = SimpleNamespace(status="succeeded", result={})

    try:
        active_job_id = manager.start_job(
            owner_id="other",
            fn=_job,
            exclusivity_key=exclusivity_key,
        )
        assert job_entered.wait(5)
        assert (
            page._capture_ingest_terminal_state(
                state,
                owner_id="owner",
                job_id="terminal-job",
                completed_key="ingest_job_completed_id",
            )
            is False
        )
        assert st.session_state["ingest_job_id"] == "terminal-job"
        release_job.set()
        assert (
            manager.wait_for_completion(
                active_job_id,
                owner_id="other",
                timeout_sec=5,
            )
            == "succeeded"
        )
        assert page._capture_ingest_terminal_state(
            state,
            owner_id="owner",
            job_id="terminal-job",
            completed_key="ingest_job_completed_id",
        )
        assert streamlit_calls["infos"] == [
            "Other background work is active. Live activation will retry automatically."
        ]
    finally:
        release_job.set()
        manager._begin_shutdown_for_tests()
        manager._join_shutdown_for_tests()


def test_success_details_survive_terminal_handoff(
    monkeypatch,
    streamlit_calls,
    tmp_path: Path,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    snapshot_dir = tmp_path / "storage" / "snapshot-v1"
    snapshot_dir.mkdir(parents=True)
    resource = page.VectorIndexResource("vector")
    graph_index = object()
    router = object()
    ingest_result: dict[str, object] = {
        "count": 2,
        "metadata": {
            "nlp.enabled": True,
            "nlp.enriched_nodes": 2,
            "nlp.entity_count": 1,
        },
        "nlp_preview": {
            "entities": [{"label": "ORG", "text": "DocMind"}],
            "sentences": [{"text": "A preview sentence."}],
        },
        "exports": [],
        "vector_resource": resource,
        "pg_index": graph_index,
        "collections": dict(_PHYSICAL_COLLECTIONS),
        "snapshot_id": "snapshot-v1",
        "documents": ["not needed for presentation"],
    }
    manifest = _valid_manifest()
    state = SimpleNamespace(
        status="succeeded",
        result={
            "ingest": ingest_result,
            "snapshot_dir": str(snapshot_dir),
            "manifest": manifest,
            "use_graphrag": True,
            "runtime_generation": settings.cache_version,
        },
        error=None,
    )
    manager = SimpleNamespace(
        get=lambda *_a, **_k: state,
        drain_progress=lambda *_a, **_k: [],
        consume_terminal=lambda *_a, **_k: True,
        admission_quiescence=lambda: contextlib.nullcontext(),
        exclusivity_activity_snapshot=lambda _key: (
            False,
            SimpleNamespace(maintenance_active=False),
        ),
    )
    rerun_scopes: list[str] = []
    rendered_manifests: list[tuple[dict[str, object], Path]] = []
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(st, "progress", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        st,
        "rerun",
        lambda *, scope="app": rerun_scopes.append(scope),
        raising=False,
    )
    monkeypatch.setattr(page, "build_router_engine", lambda *_a, **_k: router)
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda *_a, **_k: snapshot_dir)
    monkeypatch.setattr(
        page,
        "load_manifest",
        lambda _path: pytest.fail("terminal handoff reread the finalized manifest"),
    )
    monkeypatch.setattr(
        page,
        "_render_manifest_details",
        lambda value, path: rendered_manifests.append((value, path)),
    )
    st.session_state["ingest_job_id"] = "job-1"

    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )

    assert rerun_scopes == ["app"]
    assert st.session_state["vector_index"] == "vector"
    assert st.session_state["graphrag_index"] is graph_index
    assert st.session_state["router_engine"] is router
    notice = st.session_state["ingest_job_terminal_notice"]
    presentation_result = notice["presentation"]
    assert presentation_result["count"] == 2
    assert presentation_result["nlp_preview"] == {
        "entities": [{"label": "ORG", "text": "DocMind"}],
        "sentences": [{"text": "A preview sentence."}],
    }
    assert presentation_result["exports"] == []
    assert "documents" not in presentation_result
    assert "vector_resource" not in presentation_result
    assert "pg_index" not in presentation_result
    assert notice["snapshot_dir"] == str(snapshot_dir.resolve())

    page._render_ingest_terminal_notice()  # type: ignore[attr-defined]

    assert "Ingested 2 documents." in streamlit_calls["writes"]
    assert "Router engine is ready for Chat." in streamlit_calls["infos"]
    assert "GraphRAG index is available." in streamlit_calls["infos"]
    assert rendered_manifests == [(manifest, snapshot_dir.resolve())]
    assert st.session_state["latest_manifest"] == manifest
    assert streamlit_calls["success"] == ["Snapshot created: snapshot-v1"]
    assert "ingest_job_terminal_notice" not in st.session_state


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("count", -1),
        ("count", "2"),
        ("vector_resource", object()),
        ("collections", {"text": "physical-text-v2"}),
        ("nlp_preview", {"entities": "not-a-list"}),
    ],
)
def test_malformed_success_payload_is_sanitized_and_releases_resource(
    monkeypatch,
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    snapshot_dir = tmp_path / "storage" / "snapshot-v1"
    snapshot_dir.mkdir(parents=True)
    resource = page.VectorIndexResource("vector")
    monkeypatch.setattr(page, "load_manifest", lambda _path: _valid_manifest())
    ingest_result: dict[str, object] = {
        "count": 1,
        "metadata": {},
        "nlp_preview": {},
        "exports": [],
        "vector_resource": resource,
        "collections": dict(_PHYSICAL_COLLECTIONS),
    }
    ingest_result[field] = invalid_value
    monkeypatch.setattr(
        page,
        "build_router_engine",
        lambda *_a, **_k: pytest.fail("router built for malformed payload"),
    )
    state = SimpleNamespace(
        status="succeeded",
        result={
            "ingest": ingest_result,
            "snapshot_dir": str(snapshot_dir),
            "manifest": _valid_manifest(),
            "use_graphrag": False,
            "runtime_generation": settings.cache_version,
        },
        error=None,
    )

    consumed = page._capture_ingest_terminal_state(  # type: ignore[attr-defined]
        state,
        owner_id="owner",
        job_id="job-1",
        completed_key="ingest_job_completed_id",
    )

    assert consumed is True
    assert st.session_state["ingest_job_terminal_notice"] == {
        "status": "failed",
        "message": "Ingestion finished with an invalid result. Please try again.",
    }
    assert st.session_state.get("vector_index") is None
    if field == "vector_resource":
        resource.close()
    else:
        assert resource.closed


def test_success_handoff_is_bounded_and_contains_no_runtime_objects(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    snapshot_dir = tmp_path / "storage" / "snapshot-v1"
    snapshot_dir.mkdir(parents=True)
    resource = page.VectorIndexResource("vector")
    graph_index = object()
    router = object()
    long_text = "x" * 800
    exports = [
        {
            "content_type": "image/png",
            "metadata": {
                "doc_id": f"doc-{index}-" + long_text,
                "page_no": index,
                "ignored": long_text,
            },
        }
        for index in range(150)
    ]
    monkeypatch.setattr(page, "build_router_engine", lambda *_a, **_k: router)
    monkeypatch.setattr(page, "load_manifest", lambda _path: _valid_manifest())
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda *_a, **_k: snapshot_dir)

    notice = page._prepare_ingest_success_notice(  # type: ignore[attr-defined]
        {
            "ingest": {
                "count": 3,
                "metadata": {
                    "nlp.enabled": True,
                    "nlp.enriched_nodes": 3,
                    "nlp.entity_count": 25,
                    "ignored": long_text,
                },
                "nlp_preview": {
                    "entities": [
                        {"label": long_text, "text": long_text} for _index in range(25)
                    ],
                    "sentences": [{"text": long_text} for _index in range(15)],
                },
                "exports": exports,
                "vector_resource": resource,
                "pg_index": graph_index,
                "collections": dict(_PHYSICAL_COLLECTIONS),
                "snapshot_id": "snapshot-v1",
                "documents": [long_text] * 100,
            },
            "snapshot_dir": str(snapshot_dir),
            "manifest": _valid_manifest(),
            "use_graphrag": True,
            "runtime_generation": settings.cache_version,
        }
    )

    presentation = notice["presentation"]
    assert len(presentation["nlp_preview"]["entities"]) == 20
    assert len(presentation["nlp_preview"]["sentences"]) == 10
    assert len(presentation["exports"]) == 128
    assert presentation["image_preview_summary"] == {
        "selected_artifacts": 128,
        "total_artifacts": 150,
        "total_documents": 150,
        "omitted_artifacts": 22,
        "omitted_documents": 22,
    }
    assert len(presentation["nlp_preview"]["entities"][0]["label"]) == 200
    assert len(presentation["nlp_preview"]["entities"][0]["text"]) == 500
    assert len(presentation["exports"][0]["metadata"]["doc_id"]) == 200
    assert set(presentation["metadata"]) == {
        "nlp.enabled",
        "nlp.enriched_nodes",
        "nlp.entity_count",
    }
    assert set(presentation["exports"][0]["metadata"]) == {"doc_id", "page_no"}

    def _contains_identity(value: object, target: object) -> bool:
        if value is target:
            return True
        if isinstance(value, dict):
            return any(_contains_identity(item, target) for item in value.values())
        if isinstance(value, list):
            return any(_contains_identity(item, target) for item in value)
        return False

    assert not _contains_identity(notice, resource)
    assert not _contains_identity(notice, graph_index)
    assert not _contains_identity(notice, router)
    assert st.session_state["vector_index"] == "vector"
    assert st.session_state["graphrag_index"] is graph_index
    assert st.session_state["router_engine"] is router

    page.replace_session_runtime(
        st.session_state,
        None,
        None,
        runtime_generation=settings.cache_version,
    )


@pytest.mark.parametrize("stale_dimension", ["generation", "current-snapshot"])
def test_stale_success_remains_truthful_without_replacing_live_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stale_dimension: str,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings
    from src.ui.vector_session import clear_session_runtime

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    snapshot_dir = tmp_path / "storage" / "snapshot-v1"
    newer_dir = tmp_path / "storage" / "snapshot-v2"
    snapshot_dir.mkdir(parents=True)
    newer_dir.mkdir()

    class _Client:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class _Router:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    current_generation = int(settings.cache_version)
    old_client = _Client()
    old_resource = page.VectorIndexResource("old-index", client=old_client)
    old_router = _Router()
    page.replace_session_runtime(
        st.session_state,
        old_resource,
        old_router,
        runtime_generation=current_generation,
        state_updates={"_snapshot_loaded_id": "live-snapshot"},
    )
    new_client = _Client()
    new_resource = page.VectorIndexResource("new-index", client=new_client)
    monkeypatch.setattr(page, "load_manifest", lambda _path: _valid_manifest())
    monkeypatch.setattr(
        page,
        "latest_snapshot_dir",
        lambda *_a, **_k: (
            newer_dir if stale_dimension == "current-snapshot" else snapshot_dir
        ),
    )
    monkeypatch.setattr(
        page,
        "build_router_engine",
        lambda *_a, **_k: pytest.fail("router built for superseded terminal state"),
    )
    result_generation = (
        current_generation + 1
        if stale_dimension == "generation"
        else current_generation
    )

    notice = page._prepare_ingest_success_notice(
        {
            "ingest": {
                "count": 1,
                "metadata": {},
                "nlp_preview": {},
                "exports": [],
                "vector_resource": new_resource,
                "collections": dict(_PHYSICAL_COLLECTIONS),
                "snapshot_id": "snapshot-v1",
            },
            "snapshot_dir": str(snapshot_dir),
            "manifest": _valid_manifest(),
            "use_graphrag": False,
            "runtime_generation": result_generation,
        }
    )

    assert notice["status"] == "succeeded"
    assert notice["runtime_publication"] == "superseded"
    assert notice["runtime_identity"] == {
        "generation": result_generation,
        "snapshot_id": "snapshot-v1",
        "graph_enabled": False,
    }
    assert st.session_state["vector_index"] == "old-index"
    assert st.session_state["router_engine"] is old_router
    assert old_client.close_calls == 0
    assert old_router.close_calls == 0
    assert new_client.close_calls == 1

    clear_session_runtime(
        st.session_state,
        runtime_generation=current_generation,
    )


def test_deleted_superseded_snapshot_uses_captured_manifest_dto(
    monkeypatch: pytest.MonkeyPatch,
    streamlit_calls: dict[str, list[str]],
    tmp_path: Path,
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    snapshot_dir = tmp_path / "storage" / "snapshot-v1"
    newer_dir = tmp_path / "storage" / "snapshot-v2"
    snapshot_dir.mkdir(parents=True)
    snapshot_dir.rmdir()
    newer_dir.mkdir(parents=True)
    manifest = _valid_manifest()
    resource = page.VectorIndexResource("new-index")
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda *_a, **_k: newer_dir)
    monkeypatch.setattr(
        page,
        "load_manifest",
        lambda _path: pytest.fail("terminal capture reread a deleted snapshot"),
    )
    monkeypatch.setattr(
        page,
        "build_router_engine",
        lambda *_a, **_k: pytest.fail("router built for superseded snapshot"),
    )
    rendered_manifests: list[tuple[dict[str, object], Path]] = []
    monkeypatch.setattr(page, "_render_ingest_presentation", lambda _value: None)
    monkeypatch.setattr(
        page,
        "_render_manifest_details",
        lambda value, path: rendered_manifests.append((value, path)),
    )

    notice = page._prepare_ingest_success_notice(
        {
            "ingest": {
                "count": 1,
                "metadata": {},
                "nlp_preview": {},
                "exports": [],
                "vector_resource": resource,
                "collections": dict(_PHYSICAL_COLLECTIONS),
                "snapshot_id": "snapshot-v1",
            },
            "snapshot_dir": str(snapshot_dir),
            "manifest": manifest,
            "use_graphrag": False,
            "runtime_generation": settings.cache_version,
        }
    )

    assert notice["status"] == "succeeded"
    assert notice["runtime_publication"] == "superseded"
    assert notice["manifest"] == manifest
    assert resource.closed

    page._render_ingest_success_notice(notice, message=str(notice["message"]))

    assert rendered_manifests == [(manifest, snapshot_dir.resolve())]
    assert streamlit_calls["infos"] == [
        "Snapshot completed, but a newer corpus activation is current."
    ]


@pytest.mark.parametrize("readiness_loss", ["runtime-clear", "new-current"])
def test_terminal_render_rechecks_live_runtime_and_suppresses_stale_exports(
    monkeypatch: pytest.MonkeyPatch,
    streamlit_calls: dict[str, list[str]],
    tmp_path: Path,
    readiness_loss: str,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings
    from src.ui.vector_session import clear_session_runtime

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    snapshot_dir = tmp_path / "storage" / "snapshot-v1"
    newer_dir = tmp_path / "storage" / "snapshot-v2"
    snapshot_dir.mkdir(parents=True)
    newer_dir.mkdir()
    current_generation = int(settings.cache_version)
    resource = page.VectorIndexResource("index")

    class _Router:
        closed = False

        def close(self) -> None:
            self.closed = True

    router = _Router()
    page.replace_session_runtime(
        st.session_state,
        resource,
        router,
        runtime_generation=current_generation,
        state_updates={
            "graphrag_index": object(),
            "_snapshot_loaded_id": "snapshot-v1",
        },
    )
    if readiness_loss == "runtime-clear":
        clear_session_runtime(
            st.session_state,
            runtime_generation=current_generation,
        )
        authoritative = snapshot_dir
    else:
        authoritative = newer_dir
    monkeypatch.setattr(
        page,
        "latest_snapshot_dir",
        lambda *_a, **_k: authoritative,
    )
    rendered_manifests: list[dict[str, object]] = []
    monkeypatch.setattr(
        page,
        "_render_manifest_details",
        lambda manifest, _path: rendered_manifests.append(manifest),
    )
    monkeypatch.setattr(page, "_render_ingest_presentation", lambda _value: None)
    manifest = _valid_manifest()
    manifest["graph_exports"] = [{"path": "stale.graphml"}]
    notice = {
        "status": "succeeded",
        "message": "Snapshot created: snapshot-v1",
        "presentation": {},
        "manifest": manifest,
        "snapshot_dir": str(snapshot_dir),
        "runtime_identity": {
            "generation": current_generation,
            "snapshot_id": "snapshot-v1",
            "graph_enabled": True,
        },
        "runtime_publication": "published",
    }

    page._render_ingest_success_notice(notice, message=str(notice["message"]))

    assert rendered_manifests[0]["graph_exports"] == []
    assert "Router engine is ready for Chat." not in streamlit_calls["infos"]
    assert "GraphRAG index is available." not in streamlit_calls["infos"]
    if readiness_loss == "runtime-clear":
        assert streamlit_calls["infos"] == [
            "Snapshot completed, but the live runtime changed. Chat will reload "
            "the active snapshot."
        ]
        assert st.session_state["latest_manifest"] == manifest
    else:
        assert streamlit_calls["infos"] == [
            "Snapshot completed, but a newer corpus activation is current."
        ]
        assert "latest_manifest" not in st.session_state
        clear_session_runtime(
            st.session_state,
            runtime_generation=current_generation,
        )


def test_router_failure_is_sanitized_and_unwinds_runtime_ownership(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    snapshot_dir = tmp_path / "storage" / "snapshot-v1"
    snapshot_dir.mkdir(parents=True)

    class _Client:
        closed = False

        def close(self) -> None:
            self.closed = True

    client = _Client()
    resource = page.VectorIndexResource("vector", client=client)
    monkeypatch.setattr(page, "load_manifest", lambda _path: _valid_manifest())
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda *_a, **_k: snapshot_dir)
    monkeypatch.setattr(
        page,
        "build_router_engine",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("secret router failure")),
    )
    state = SimpleNamespace(
        status="succeeded",
        result={
            "ingest": {
                "count": 1,
                "metadata": {},
                "nlp_preview": {},
                "exports": [],
                "vector_resource": resource,
                "collections": dict(_PHYSICAL_COLLECTIONS),
            },
            "snapshot_dir": str(snapshot_dir),
            "manifest": _valid_manifest(),
            "use_graphrag": False,
            "runtime_generation": settings.cache_version,
        },
        error=None,
    )

    page._capture_ingest_terminal_state(  # type: ignore[attr-defined]
        state,
        owner_id="owner",
        job_id="job-1",
        completed_key="ingest_job_completed_id",
    )

    notice = st.session_state["ingest_job_terminal_notice"]
    assert notice == {
        "status": "failed",
        "message": "Ingestion finished with an invalid result. Please try again.",
    }
    assert "secret" not in str(notice)
    assert client.closed
    assert st.session_state.get("vector_index") is None
    assert st.session_state.get("router_engine") is None


@pytest.mark.parametrize("manifest_case", ["none", "bad-hash", "bad-exports"])
def test_manifest_failure_preserves_existing_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest_case: str,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    snapshot_dir = tmp_path / "storage" / "snapshot-v1"
    snapshot_dir.mkdir(parents=True)

    class _Client:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class _Router:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    old_client = _Client()
    new_client = _Client()
    old_resource = page.VectorIndexResource("old-index", client=old_client)
    new_resource = page.VectorIndexResource("new-index", client=new_client)
    old_router = _Router()
    page.replace_session_runtime(
        st.session_state,
        old_resource,
        old_router,
        runtime_generation=settings.cache_version,
    )
    if manifest_case == "none":
        manifest = None
    elif manifest_case == "bad-hash":
        manifest = _valid_manifest()
        manifest["corpus_hash"] = "not-a-hash"
    else:
        manifest = _valid_manifest()
        manifest["graph_exports"] = {"not": "a list"}
    monkeypatch.setattr(
        page,
        "load_manifest",
        lambda _path: pytest.fail("terminal handoff reread a snapshot manifest"),
    )
    monkeypatch.setattr(
        page,
        "build_router_engine",
        lambda *_a, **_k: pytest.fail("router built before manifest validation"),
    )
    state = SimpleNamespace(
        status="succeeded",
        result={
            "ingest": {
                "count": 1,
                "metadata": {},
                "nlp_preview": {},
                "exports": [],
                "vector_resource": new_resource,
                "collections": dict(_PHYSICAL_COLLECTIONS),
            },
            "snapshot_dir": str(snapshot_dir),
            "manifest": manifest,
            "use_graphrag": False,
            "runtime_generation": settings.cache_version,
        },
        error=None,
    )

    page._capture_ingest_terminal_state(
        state,
        owner_id="owner",
        job_id="job-1",
        completed_key="ingest_job_completed_id",
    )

    assert st.session_state["ingest_job_terminal_notice"] == {
        "status": "failed",
        "message": "Ingestion finished with an invalid result. Please try again.",
    }
    assert st.session_state["vector_index"] == "old-index"
    assert st.session_state["router_engine"] is old_router
    assert old_client.close_calls == 0
    assert old_router.close_calls == 0
    assert new_client.close_calls == 1


def test_image_preview_round_robin_preserves_document_coverage(monkeypatch) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    exports = [
        {
            "content_type": "image/png",
            "metadata": {"doc_id": "document-a", "page_no": page_number},
        }
        for page_number in range(130)
    ]
    exports.append(
        {
            "content_type": "image/png",
            "metadata": {"doc_id": "document-b", "page_no": 0},
        }
    )

    selected, summary = page._bounded_image_exports(exports)

    assert len(selected) == 128
    assert {item["metadata"]["doc_id"] for item in selected} == {
        "document-a",
        "document-b",
    }
    assert summary == {
        "selected_artifacts": 128,
        "total_artifacts": 131,
        "total_documents": 2,
        "omitted_artifacts": 3,
        "omitted_documents": 0,
    }
    monkeypatch.setattr(
        page.st,
        "slider",
        lambda *_a, **_k: pytest.fail("terminal preview rendered a slider"),
    )
    page._render_ingest_presentation(
        {
            "count": 2,
            "metadata": {
                "nlp.enabled": False,
                "nlp.enriched_nodes": 0,
                "nlp.entity_count": 0,
            },
            "nlp_preview": {"entities": [], "sentences": []},
            "exports": selected,
            "image_preview_summary": summary,
        }
    )


@pytest.mark.parametrize("failure_stage", ["manifest", "render"])
def test_terminal_display_failure_is_sanitized_and_consumed_once(
    monkeypatch,
    streamlit_calls,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    snapshot_dir = tmp_path / "storage" / "snapshot-v1"
    snapshot_dir.mkdir(parents=True)
    st.session_state["ingest_job_terminal_notice"] = {
        "status": "succeeded",
        "message": "Snapshot created: snapshot-v1",
        "snapshot_dir": str(snapshot_dir),
        "presentation": {
            "count": 1,
            "metadata": {
                "nlp.enabled": False,
                "nlp.enriched_nodes": 0,
                "nlp.entity_count": 0,
            },
            "nlp_preview": {"entities": [], "sentences": []},
            "exports": [],
            "image_preview_summary": {
                "selected_artifacts": 0,
                "total_artifacts": 0,
                "total_documents": 0,
                "omitted_artifacts": 0,
                "omitted_documents": 0,
            },
            "router_ready": True,
            "graph_enabled": False,
        },
        "manifest": _valid_manifest(),
    }
    if failure_stage == "manifest":
        st.session_state["ingest_job_terminal_notice"]["manifest"] = None
    else:
        monkeypatch.setattr(
            page,
            "_render_ingest_presentation",
            lambda _value: (_ for _ in ()).throw(RuntimeError("secret render failure")),
        )

    page._render_ingest_terminal_notice()  # type: ignore[attr-defined]
    page._render_ingest_terminal_notice()  # type: ignore[attr-defined]

    assert streamlit_calls["errors"] == [
        "Ingestion results could not be displayed. Please try again."
    ]
    assert "secret" not in str(streamlit_calls["errors"])
    assert "ingest_job_terminal_notice" not in st.session_state


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (
            SimpleNamespace(
                status="succeeded",
                result={"unexpected": "secret raw result"},
                error=None,
            ),
            "Ingestion finished with an invalid result. Please try again.",
        ),
        (
            SimpleNamespace(
                status="mystery",
                result="secret raw result",
                error="secret raw error",
            ),
            "Ingestion ended in an unexpected state. Please try again.",
        ),
    ],
)
def test_malformed_terminal_states_are_consumed_as_sanitized_failures(
    monkeypatch,
    streamlit_calls,
    state: SimpleNamespace,
    message: str,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    manager = SimpleNamespace(
        get=lambda *_a, **_k: state,
        drain_progress=lambda *_a, **_k: [],
        consume_terminal=lambda *_a, **_k: True,
        is_exclusivity_active=lambda _key: False,
        admission_quiescence=lambda: contextlib.nullcontext(),
        exclusivity_activity_snapshot=lambda _key: (
            False,
            SimpleNamespace(maintenance_active=False),
        ),
    )
    rerun_scopes: list[str] = []
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(st, "progress", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        st,
        "rerun",
        lambda *, scope="app": rerun_scopes.append(scope),
        raising=False,
    )
    st.session_state["ingest_job_id"] = "job-1"
    st.session_state["ingest_job_last_event"] = object()

    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )

    assert rerun_scopes == ["app"]
    assert "ingest_job_id" not in st.session_state
    assert "ingest_job_last_event" not in st.session_state
    assert st.session_state["ingest_job_completed_id"] == "job-1"
    notice = st.session_state["ingest_job_terminal_notice"]
    assert notice == {"status": "failed", "message": message}
    assert "secret" not in str(notice)

    page._render_ingest_terminal_notice()  # type: ignore[attr-defined]
    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )
    assert streamlit_calls["errors"] == [message]
    assert rerun_scopes == ["app"]


def test_process_activity_edges_request_exactly_one_full_rerun(monkeypatch) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    manager = SimpleNamespace(mutation=False, maintenance=False)
    manager.exclusivity_activity_snapshot = lambda _key: (
        manager.mutation,
        SimpleNamespace(maintenance_active=manager.maintenance),
    )
    rerun_scopes: list[str] = []
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        st,
        "rerun",
        lambda *, scope="app": rerun_scopes.append(scope),
        raising=False,
    )

    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )
    manager.mutation = True
    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )
    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )
    manager.mutation = False
    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )
    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )
    manager.maintenance = True
    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )
    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )
    manager.maintenance = False
    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )
    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )

    assert rerun_scopes == ["app", "app", "app", "app"]
    assert st.session_state["corpus_activity_observed"] == (False, False)


@pytest.mark.parametrize("active", [False, True])
def test_initial_foreign_occupancy_observation_does_not_rerun(
    monkeypatch,
    active: bool,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    manager = SimpleNamespace(
        exclusivity_activity_snapshot=lambda _key: (
            active,
            SimpleNamespace(maintenance_active=False),
        )
    )
    rerun_scopes: list[str] = []
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        st,
        "rerun",
        lambda *, scope="app": rerun_scopes.append(scope),
        raising=False,
    )

    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )

    assert rerun_scopes == []
    assert st.session_state["corpus_activity_observed"] == (active, False)


def test_missing_tracked_job_requests_one_full_rerun(monkeypatch) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    manager = SimpleNamespace(
        get=lambda *_a, **_k: None,
        is_exclusivity_active=lambda _key: False,
        exclusivity_activity_snapshot=lambda _key: (
            False,
            SimpleNamespace(maintenance_active=False),
        ),
    )
    rerun_scopes: list[str] = []
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        st,
        "rerun",
        lambda *, scope="app": rerun_scopes.append(scope),
        raising=False,
    )
    st.session_state["ingest_job_id"] = "missing"
    st.session_state["ingest_job_cancel_requested_id"] = "missing"

    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )
    page._render_ingest_job_panel.__wrapped__(  # type: ignore[attr-defined]
        owner_id="owner"
    )

    assert rerun_scopes == ["app"]
    assert "ingest_job_id" not in st.session_state
    assert "ingest_job_cancel_requested_id" not in st.session_state


def test_render_ingest_form_smoke(monkeypatch) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")

    files = [object()]
    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: files, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(st, "form_submit_button", lambda *_a, **_k: True, raising=False)

    out_files, use_graphrag, encrypt_images, parsing_overrides, submitted = (
        page._render_ingest_form(  # type: ignore[attr-defined]
            active_job=None,
            mutation_active=False,
            maintenance_active=False,
        )
    )
    assert out_files == files
    assert use_graphrag is True
    assert encrypt_images is True
    assert isinstance(parsing_overrides, page.ParsingOverrides)
    assert submitted is True


def test_render_ingest_form_disables_submission_during_maintenance(
    monkeypatch: pytest.MonkeyPatch,
    streamlit_calls: dict[str, list[str]],
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    disabled: list[bool] = []
    monkeypatch.setattr(
        page,
        "_render_parsing_overrides",
        lambda: page.ParsingOverrides(),
    )
    monkeypatch.setattr(
        st,
        "form_submit_button",
        lambda *_a, **kwargs: disabled.append(bool(kwargs.get("disabled"))) or False,
        raising=False,
    )

    page._render_ingest_form(
        active_job=None,
        mutation_active=False,
        maintenance_active=True,
    )

    assert disabled == [True]
    assert streamlit_calls["infos"] == [
        "Runtime maintenance is in progress. Ingestion is temporarily disabled."
    ]


def test_start_ingestion_handles_maintenance_admission_race_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    streamlit_calls: dict[str, list[str]],
    tmp_path: Path,
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    source = tmp_path / ".pending-uploads" / "transaction" / "document.txt"
    source.parent.mkdir(parents=True)
    source.write_text("document", encoding="utf-8")

    class _Manager:
        activity_snapshot = staticmethod(
            lambda: SimpleNamespace(
                has_active_jobs=False,
                maintenance_active=False,
            )
        )
        is_exclusivity_active = staticmethod(lambda _key: False)
        exclusivity_activity_snapshot = staticmethod(
            lambda _key: (
                False,
                SimpleNamespace(maintenance_active=False),
            )
        )

        def start_job(self, **_kwargs: object) -> str:
            raise page.JobAdmissionPausedError("maintenance won")

    monkeypatch.setattr(page, "get_job_manager", _Manager)
    input_value = IngestionInput(document_id="document", source_path=source)

    started = page._start_ingestion_job(
        [input_value],
        use_graphrag=False,
        encrypt_images=False,
        nlp_service=None,
        owner_id="owner",
        rollback_source_paths=(source,),
    )

    assert started is False
    assert not source.exists()
    assert streamlit_calls["warnings"] == [
        "Runtime maintenance is in progress. Try ingestion again shortly."
    ]


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
        def is_exclusivity_active(self, _key: str) -> bool:
            return False

        def exclusivity_activity_snapshot(
            self, _key: str
        ) -> tuple[bool, SimpleNamespace]:
            return False, SimpleNamespace(maintenance_active=False)

        def activity_snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(
                has_active_jobs=False,
                maintenance_active=False,
            )

        def start_job(  # type: ignore[no-untyped-def]
            self, *, owner_id: str, fn, exclusivity_key: str | None = None
        ):
            calls["owner_id"] = owner_id
            calls["fn"] = fn
            calls["exclusivity_key"] = exclusivity_key
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
    assert calls.get("exclusivity_key") == "corpus-mutation"
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


def test_prepare_and_render_ingest_sets_session_state(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    monkeypatch.setattr(page, "_render_image_exports", lambda _e, **_k: None)
    router_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _build_router(*args, **kwargs):  # type: ignore[no-untyped-def]
        router_calls.append((args, kwargs))
        return "R"

    monkeypatch.setattr(page, "build_router_engine", _build_router)

    st.session_state.clear()
    first_resource = page.VectorIndexResource("V")
    presentation, graph_enabled = page._prepare_ingest_runtime(  # type: ignore[attr-defined]
        {
            "count": 2,
            "vector_index": "V",
            "vector_resource": first_resource,
            "pg_index": "G",
            "collections": dict(_PHYSICAL_COLLECTIONS),
        },
        use_graphrag=True,
    )
    page._render_ingest_presentation(presentation)  # type: ignore[attr-defined]
    assert graph_enabled is True
    assert st.session_state["vector_index"] == "V"
    assert st.session_state["graphrag_index"] == "G"
    assert st.session_state["router_engine"] == "R"

    # A requested rebuild that produces no graph clears the prior graph owner.
    st.session_state["graphrag_index"] = "G"
    second_resource = page.VectorIndexResource("V2")
    presentation, graph_enabled = page._prepare_ingest_runtime(  # type: ignore[attr-defined]
        {
            "count": 1,
            "vector_index": "V2",
            "vector_resource": second_resource,
            "collections": dict(_PHYSICAL_COLLECTIONS),
        },
        use_graphrag=True,
    )
    page._render_ingest_presentation(presentation)  # type: ignore[attr-defined]
    assert graph_enabled is False
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

    page._render_maintenance_controls(  # type: ignore[attr-defined]
        owner_id="owner",
        active_job=None,
        mutation_active=False,
        maintenance_active=False,
    )
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
        st,
        "button",
        lambda label, **_k: str(label) == f'Delete "{target.name}"',
        raising=False,
    )

    page._render_maintenance_controls(  # type: ignore[attr-defined]
        owner_id="owner",
        active_job=None,
        mutation_active=False,
        maintenance_active=False,
    )
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

    def _start_job(inputs, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((inputs, kwargs))
        return True

    monkeypatch.setattr(page, "_load_optional_spacy_service", lambda: nlp_service)
    monkeypatch.setattr(
        page,
        "_start_ingestion_job",
        _start_job,
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

    resource = page.VectorIndexResource(object())
    page.replace_session_runtime(
        st.session_state,
        resource,
        object(),
        runtime_generation=int(settings.cache_version),
        state_updates={"graphrag_index": object()},
    )

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

    try:
        page._handle_manual_export(tmp_path, "jsonl")  # type: ignore[attr-defined]
        assert streamlit_calls["success"]
    finally:
        resource.close()


def test_manual_export_seed_lookup_rejects_runtime_maintenance(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        page,
        "get_export_seed_ids",
        lambda *_a, **_k: pytest.fail("seed lookup ran during maintenance"),
    )
    st.session_state["graphrag_index"] = object()
    try:
        with manager.admission_quiescence():
            page._handle_manual_export(tmp_path, "jsonl")  # type: ignore[attr-defined]
    finally:
        manager.shutdown()
    assert streamlit_calls["warnings"] == [
        "Graph export is unavailable during runtime maintenance."
    ]


def test_manual_export_holds_foreground_lease_through_file_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager(max_workers=1)
    export_entered = threading.Event()
    release_export = threading.Event()
    output = tmp_path / "manual.jsonl"
    resource = page.VectorIndexResource(object())
    page.replace_session_runtime(
        st.session_state,
        resource,
        object(),
        runtime_generation=int(settings.cache_version),
        state_updates={"graphrag_index": object()},
    )
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(page, "get_export_seed_ids", lambda *_a, **_k: ["seed"])
    monkeypatch.setattr(page, "timestamped_export_path", lambda *_a, **_k: output)

    def _export(**_kwargs: object) -> None:
        export_entered.set()
        assert release_export.wait(5)
        output.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(page, "export_graph_jsonl", _export)
    monkeypatch.setattr(page, "record_graph_export_metric", lambda *_a, **_k: None)
    monkeypatch.setattr(page, "_log_export_event", lambda _payload: None)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(page._handle_manual_export, tmp_path, "jsonl")
            assert export_entered.wait(5)
            with (
                pytest.raises(background_jobs.ForegroundRuntimeConflictError),
                manager.admission_quiescence(),
            ):
                pytest.fail("maintenance entered while graph export used live indices")
            release_export.set()
            future.result(timeout=5)
        with manager.admission_quiescence():
            assert manager.activity_snapshot().maintenance_active
    finally:
        release_export.set()
        resource.close()
        manager.shutdown()


def test_manual_export_rejects_resource_retired_after_controls_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    streamlit_calls: dict[str, list[str]],
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    manager = importlib.import_module("src.ui.background_jobs").JobManager()
    graph_index = object()
    resource = page.VectorIndexResource(object())
    page.replace_session_runtime(
        st.session_state,
        resource,
        object(),
        runtime_generation=int(settings.cache_version),
        state_updates={"graphrag_index": graph_index},
    )
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    monkeypatch.setattr(
        page,
        "get_export_seed_ids",
        lambda *_a, **_k: pytest.fail("stale indices reached seed lookup"),
    )
    monkeypatch.setattr(
        page,
        "export_graph_jsonl",
        lambda **_k: pytest.fail("stale graph reached export"),
    )

    def _button(label: str, **_kwargs: object) -> bool:
        if label == "Export JSONL":
            assert st.session_state["graphrag_index"] is graph_index
            resource.close()
            return True
        return False

    monkeypatch.setattr(st, "button", _button)

    try:
        page._render_export_controls()  # type: ignore[attr-defined]
    finally:
        manager.shutdown()

    assert streamlit_calls["warnings"] == [
        "Graph export deferred because the runtime changed. Retry."
    ]
    assert streamlit_calls["success"] == []


def test_stale_runtime_cleanup_defers_during_active_analysis(
    monkeypatch, streamlit_calls
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager(max_workers=1)
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)
    started = threading.Event()
    release = threading.Event()

    def _analysis(_cancel, _progress):  # type: ignore[no-untyped-def]
        started.set()
        release.wait(timeout=5)

    job_id = manager.start_job(owner_id="owner", fn=_analysis)
    assert started.wait(timeout=2)

    class _Client:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class _Router:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    client = _Client()
    router = _Router()
    resource = page.VectorIndexResource("index", client=client)
    page.replace_session_runtime(
        st.session_state,
        resource,
        router,
        runtime_generation=page.settings.cache_version - 1,
    )
    try:
        assert page._clear_stale_session_runtime() is False
        assert st.session_state["router_engine"] is router
        assert st.session_state["vector_index"] == "index"
        assert client.close_calls == 0
        assert router.close_calls == 0
        assert streamlit_calls["captions"] == [
            "Stale runtime cleanup deferred while background work is active."
        ]
    finally:
        release.set()
        assert manager.wait_for_completion(job_id, owner_id="owner") == "succeeded"
        manager.shutdown()


def test_stale_runtime_cleanup_clears_malformed_router_only_state(monkeypatch) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")
    background_jobs = importlib.import_module("src.ui.background_jobs")
    manager = background_jobs.JobManager()
    monkeypatch.setattr(page, "get_job_manager", lambda: manager)

    class _Router:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    router = _Router()
    st.session_state["router_engine"] = router
    st.session_state["_snapshot_loaded_id"] = "orphan"
    st.session_state["_snapshot_collections"] = {"text": "orphan"}
    try:
        assert page._clear_stale_session_runtime() is True
        assert router.close_calls == 1
        assert st.session_state["router_engine"] is None
        assert st.session_state["vector_index"] is None
        assert "_snapshot_loaded_id" not in st.session_state
        assert "_snapshot_collections" not in st.session_state
    finally:
        manager.shutdown()


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
            runtime_generation=page.settings.cache_version,
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


def test_prepare_ingest_runtime_replaces_and_clears_router(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(page, "_render_image_exports", lambda _e, **_k: None)

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

    presentation, graph_enabled = page._prepare_ingest_runtime(
        {
            "count": 1,
            "vector_index": resource.index,
            "vector_resource": resource,
            "collections": dict(_PHYSICAL_COLLECTIONS),
        },
        use_graphrag=False,
    )
    page._render_ingest_presentation(presentation)

    assert graph_enabled is False
    assert old.closed == 1
    assert new.closed == 0
    assert st.session_state["router_engine"] is new


def test_prepare_ingest_runtime_build_failure_preserves_router(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(page, "_render_image_exports", lambda _e, **_k: None)

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
        page._prepare_ingest_runtime(
            {
                "count": 1,
                "vector_index": resource.index,
                "vector_resource": resource,
                "collections": dict(_PHYSICAL_COLLECTIONS),
            },
            use_graphrag=False,
        )

    assert old.closed == 0
    assert st.session_state["router_engine"] is old
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

    resource = page.VectorIndexResource(object())
    page.replace_session_runtime(
        st.session_state,
        resource,
        object(),
        runtime_generation=int(settings.cache_version),
        state_updates={"graphrag_index": object()},
    )

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

    try:
        page._handle_manual_export(tmp_path, "parquet")  # type: ignore[attr-defined]
        assert streamlit_calls["success"]
    finally:
        resource.close()
