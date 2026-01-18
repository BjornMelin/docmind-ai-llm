"""Integration test validating the Documents ingestion background job flow."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.ui.background_jobs as bg
from tests.helpers.apptest_utils import apptest_timeout_sec


@pytest.fixture
def documents_ingest_app_test(
    tmp_path: Path,
    monkeypatch,
    fake_job_manager,
    fake_job_owner_id,
) -> Generator[AppTest]:
    """Create an AppTest instance for the Documents ingestion job path.

    Args:
        tmp_path: Temporary directory for test data.
        monkeypatch: Pytest monkeypatch fixture.
        fake_job_manager: Synchronous fake job manager fixture.
        fake_job_owner_id: Deterministic owner identifier fixture.

    Returns:
        Generator yielding an AppTest instance for the documents page.
    """
    from src.config.settings import settings as app_settings

    # Save original values
    orig_data_dir = app_settings.data_dir
    orig_chat_sqlite = app_settings.chat.sqlite_path
    orig_db_sqlite = app_settings.database.sqlite_db_path
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
            "manifest": {
                "corpus_hash": "c" * 64,
                "config_hash": "f" * 64,
                "versions": {},
            },
            "exports": [],
            "duration_ms": 0.0,
            "metadata": {},
            "nlp_preview": None,
            "documents": [],
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

    monkeypatch.setattr(bg, "get_job_manager", lambda *_a, **_k: fake_job_manager)
    monkeypatch.setattr(bg, "get_or_create_owner_id", lambda: fake_job_owner_id)

    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "02_documents.py"
    at = AppTest.from_file(str(page_path), default_timeout=apptest_timeout_sec())

    try:
        yield at
    finally:
        # Restore original values
        app_settings.data_dir = orig_data_dir
        app_settings.chat.sqlite_path = orig_chat_sqlite
        app_settings.database.sqlite_db_path = orig_db_sqlite
        # Note: monkeypatch.setenv auto-restores env vars on teardown.


@pytest.mark.integration
def test_documents_ingestion_job_renders_success(
    documents_ingest_app_test: AppTest,
) -> None:
    """Run the ingestion flow and assert success rendering.

    Args:
        documents_ingest_app_test: Prepared AppTest fixture.

    Returns:
        None.
    """
    app = documents_ingest_app_test.run()
    assert not app.exception

    ingest_buttons = [b for b in app.button if b.label == "Ingest"]
    assert ingest_buttons, "Ingest button not found"

    result = ingest_buttons[0].click().run()
    assert not result.exception

    success_messages = [msg.value for msg in result.success]
    assert any("Snapshot created" in value for value in success_messages)
