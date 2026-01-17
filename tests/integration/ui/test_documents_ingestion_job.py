"""Integration test validating the Documents ingestion background job flow."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.ui.background_jobs import ProgressEvent


@pytest.fixture
def documents_ingest_app_test(tmp_path: Path, monkeypatch) -> AppTest:
    """Create an AppTest instance for the Documents ingestion job path."""
    from src.config.settings import settings as app_settings

    monkeypatch.setenv("DOCMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCMIND_CACHE_DIR", str(tmp_path / "cache"))
    app_settings.data_dir = tmp_path
    app_settings.chat.sqlite_path = tmp_path / "chat.db"
    app_settings.database.sqlite_db_path = tmp_path / "docmind.db"

    # Provide deterministic "uploaded files" for Streamlit's file_uploader.
    class _FakeUpload:
        name = "doc.txt"

    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: [_FakeUpload()])

    # Stub ingestion adapter functions that the page imports.
    import src.ui.ingest_adapter as ingest_adapter

    monkeypatch.setattr(
        ingest_adapter,
        "save_uploaded_file",
        lambda *_a, **_k: (tmp_path / "doc.txt", "a" * 64),
    )
    monkeypatch.setattr(
        ingest_adapter,
        "ingest_inputs",
        lambda inputs, **_k: {
            "count": len(inputs),
            "vector_index": object(),
            "pg_index": None,
            "exports": [],
            "metadata": {},
            "nlp_preview": None,
        },
    )

    # Stub router engine build to avoid heavy retrieval imports.
    import src.retrieval.router_factory as router_factory

    monkeypatch.setattr(
        router_factory, "build_router_engine", lambda *_a, **_k: object()
    )

    # Stub multimodal retriever module import (best effort).
    monkeypatch.setitem(
        sys.modules,
        "src.retrieval.multimodal_fusion",
        SimpleNamespace(MultimodalFusionRetriever=lambda *_a, **_k: object()),
    )

    # Stub snapshot rebuild + manifest load.
    import src.persistence.snapshot as snapshot_mod
    import src.persistence.snapshot_service as snapshot_service

    snapshot_dir = tmp_path / "storage" / "backup"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        snapshot_service, "rebuild_snapshot", lambda *_a, **_k: snapshot_dir
    )
    monkeypatch.setattr(
        snapshot_mod,
        "load_manifest",
        lambda *_a, **_k: {
            "corpus_hash": "c" * 64,
            "config_hash": "f" * 64,
            "versions": {},
        },
    )

    # Replace background job manager with a synchronous fake so success renders
    # deterministically in the same AppTest run.
    import src.ui.background_jobs as bg

    class _State:
        def __init__(self, *, owner_id: str, status: str, result, error: str | None):
            self.owner_id = owner_id
            self.status = status
            self.result = result
            self.error = error

    class _FakeJobManager:
        def __init__(self) -> None:
            self._events: dict[str, list[ProgressEvent]] = {}
            self._states: dict[str, _State] = {}

        def start_job(self, *, owner_id: str, fn):  # type: ignore[no-untyped-def]
            job_id = "job-1"
            events: list[ProgressEvent] = []

            def _report(evt: ProgressEvent) -> None:
                events.append(evt)

            try:
                res = fn(threading.Event(), _report)
                state = _State(
                    owner_id=owner_id, status="succeeded", result=res, error=None
                )
            except Exception as exc:  # pragma: no cover - defensive
                state = _State(
                    owner_id=owner_id, status="failed", result=None, error=str(exc)
                )
            self._events[job_id] = events
            self._states[job_id] = state
            return job_id

        def get(self, job_id: str, *, owner_id: str):  # type: ignore[no-untyped-def]
            state = self._states.get(job_id)
            if state is None or state.owner_id != owner_id:
                return None
            return state

        def drain_progress(self, job_id: str, *, owner_id: str, max_events: int = 100):  # type: ignore[no-untyped-def]
            _ = max_events
            state = self._states.get(job_id)
            if state is None or state.owner_id != owner_id:
                return []
            events = self._events.get(job_id, [])
            self._events[job_id] = []
            return events

        def cancel(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return True

    fake_mgr = _FakeJobManager()
    monkeypatch.setattr(bg, "get_job_manager", lambda *_a, **_k: fake_mgr)
    monkeypatch.setattr(bg, "get_or_create_owner_id", lambda: "owner")

    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "02_documents.py"
    at = AppTest.from_file(str(page_path))
    at.default_timeout = 6
    return at


@pytest.mark.integration
def test_documents_ingestion_job_renders_success(
    documents_ingest_app_test: AppTest,
) -> None:
    app = documents_ingest_app_test.run()
    assert not app.exception

    ingest_buttons = [b for b in app.button if b.label == "Ingest"]
    assert ingest_buttons, "Ingest button not found"

    result = ingest_buttons[0].click().run()
    assert not result.exception

    success_messages = [msg.value for msg in result.success]
    assert any("Snapshot created" in value for value in success_messages)
