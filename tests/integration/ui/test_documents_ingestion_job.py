"""Integration test validating the Documents ingestion background job flow."""

from __future__ import annotations

import hashlib
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.ui.background_jobs as bg
from tests.helpers.apptest_utils import apptest_timeout_sec


@pytest.fixture
def documents_ingest_app_test(  # noqa: PLR0915 - integration harness owns page seams
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
    monkeypatch.setenv("DOCMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCMIND_CACHE__DIR", str(tmp_path / "cache"))
    app_settings.data_dir = tmp_path
    app_settings.chat.sqlite_path = tmp_path / "chat.db"
    physical_collections = {
        "text": f"{app_settings.database.qdrant_collection}__build123",
        "image": f"{app_settings.database.qdrant_image_collection}__build123",
    }

    # Provide deterministic "uploaded files" for Streamlit's file_uploader.
    class _FakeUpload:
        name = "doc.txt"
        size = 8

        def getbuffer(self) -> memoryview:
            return memoryview(b"document")

    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: [_FakeUpload()])

    # Stub ingestion adapter functions that the page imports.
    import src.ui.ingest_adapter as ingest_adapter

    def _save_upload(_file, *, destination_dir=None):  # type: ignore[no-untyped-def]
        root = (
            Path(destination_dir)
            if destination_dir is not None
            else tmp_path / "uploads"
        )
        root.mkdir(parents=True, exist_ok=True)
        upload_path = root / "doc.txt"
        upload_path.write_bytes(b"document")
        return upload_path, hashlib.sha256(b"document").hexdigest()

    monkeypatch.setattr(
        ingest_adapter,
        "save_uploaded_file",
        _save_upload,
    )
    from src.ui.vector_session import VectorIndexResource

    def _ingest_inputs(inputs, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["text_collection_name"] == physical_collections["text"]
        assert kwargs["image_collection_name"] == physical_collections["image"]
        resource = VectorIndexResource(object())
        return {
            "count": len(inputs),
            "vector_index": resource.index,
            "vector_resource": resource,
            "pg_index": None,
            "activation_corpus_hash": "c" * 64,
            "activation_config": {"test": True},
            "activation_config_hash": "f" * 64,
            "snapshot_config_hash": "e" * 64,
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
            "collections": dict(physical_collections),
        }

    monkeypatch.setattr(ingest_adapter, "ingest_inputs", _ingest_inputs)

    # Stub router engine build to avoid heavy retrieval imports.
    import src.retrieval.router_factory as router_factory

    monkeypatch.setattr(
        router_factory, "build_router_engine", lambda *_a, **_k: object()
    )

    # Stub snapshot rebuild + manifest load.
    import qdrant_client

    import src.persistence.snapshot as snapshot_mod
    import src.persistence.snapshot_service as snapshot_service

    snapshot_dir = tmp_path / "storage" / "backup"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    workspace = tmp_path / "storage" / "_tmp-build123"

    class _SnapshotManager:
        def __init__(self, base_dir: Path) -> None:
            assert base_dir == tmp_path / "storage"
            self.cleanup_calls: list[Path] = []

        def begin_snapshot(self) -> Path:
            workspace.mkdir(parents=True, exist_ok=True)
            return workspace

        def cleanup_tmp(self, path: Path) -> None:
            self.cleanup_calls.append(path)

    class _QdrantClient:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            self.closed = False

        def get_collection(self, collection_name: str) -> SimpleNamespace:
            owner = (
                "text" if collection_name == physical_collections["text"] else "image"
            )
            assert collection_name == physical_collections[owner]
            return SimpleNamespace(
                config=SimpleNamespace(
                    metadata={"snapshot_id": "build123", "role": owner}
                )
            )

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(snapshot_mod, "SnapshotManager", _SnapshotManager)
    monkeypatch.setattr(qdrant_client, "QdrantClient", _QdrantClient)

    def _rebuild_snapshot(*args, **kwargs) -> Path:  # type: ignore[no-untyped-def]
        activation = args[3]
        assert activation.workspace == workspace
        assert activation.text_collection == physical_collections["text"]
        assert activation.image_collection == physical_collections["image"]
        assert activation.collection_metadata == {
            "text": {"snapshot_id": "build123", "role": "text"},
            "image": {"snapshot_id": "build123", "role": "image"},
        }
        kwargs["commit_source_changes"]()
        return snapshot_dir

    monkeypatch.setattr(
        snapshot_service,
        "rebuild_snapshot",
        _rebuild_snapshot,
    )
    monkeypatch.setattr(
        snapshot_mod,
        "load_manifest",
        lambda *_a, **_k: {
            "corpus_hash": "c" * 64,
            "config_hash": "f" * 64,
            "versions": {},
            "collections": dict(physical_collections),
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
        # Note: monkeypatch.setenv auto-restores env vars on teardown.


@pytest.mark.integration
def test_parsing_overrides_are_reachable_before_submit(
    documents_ingest_app_test: AppTest,
) -> None:
    """Enable per-ingestion parser controls without submitting the form."""
    app = documents_ingest_app_test.run()
    assert not app.exception

    checkboxes = {checkbox.label: checkbox for checkbox in app.checkbox}
    use_global = checkboxes["Use global parsing defaults"]
    assert use_global.value is True
    assert checkboxes["Force RapidOCR"].disabled is True
    assert checkboxes["Export searchable PDF"].disabled is True

    result = use_global.uncheck().run()
    assert not result.exception

    updated = {checkbox.label: checkbox for checkbox in result.checkbox}
    assert updated["Use global parsing defaults"].value is False
    assert updated["Force RapidOCR"].disabled is False
    assert updated["Export searchable PDF"].disabled is False

    configured = updated["Force RapidOCR"].check().run()
    assert not configured.exception
    configured_checkboxes = {
        checkbox.label: checkbox for checkbox in configured.checkbox
    }
    assert configured_checkboxes["Force RapidOCR"].value is True
    assert any(button.label == "Ingest" for button in configured.button)
    assert not any(
        "Snapshot created" in message.value for message in configured.success
    )


@pytest.mark.integration
def test_documents_ingestion_job_renders_success(
    documents_ingest_app_test: AppTest,
    fake_job_manager,
    fake_job_owner_id: str,
) -> None:
    """Run the ingestion flow and assert success rendering.

    Args:
        documents_ingest_app_test: Prepared AppTest fixture.
        fake_job_manager: Synchronous job manager used by the harness.
        fake_job_owner_id: Deterministic owner identity used by the harness.

    Returns:
        None.
    """
    app = documents_ingest_app_test.run()
    assert not app.exception

    ingest_buttons = [b for b in app.button if b.label == "Ingest"]
    assert ingest_buttons, "Ingest button not found"

    result = ingest_buttons[0].click().run()
    assert not result.exception
    state = fake_job_manager.get("job-1", owner_id=fake_job_owner_id)
    assert state is not None
    assert state.status == "succeeded", state.error

    success_messages = [msg.value for msg in result.success]
    assert any("Snapshot created" in value for value in success_messages)
