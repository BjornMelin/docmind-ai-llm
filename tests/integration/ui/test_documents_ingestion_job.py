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
                "graph_exports": [],
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

    def _rebuild_snapshot(*args, **kwargs):  # type: ignore[no-untyped-def]
        activation = args[3]
        assert activation.workspace == workspace
        assert activation.text_collection == physical_collections["text"]
        assert activation.image_collection == physical_collections["image"]
        assert activation.collection_metadata == {
            "text": {"snapshot_id": "build123", "role": "text"},
            "image": {"snapshot_id": "build123", "role": "image"},
        }
        kwargs["commit_source_changes"]()
        return snapshot_mod.FinalizedSnapshot(
            path=snapshot_dir,
            manifest={
                "corpus_hash": "c" * 64,
                "config_hash": "f" * 64,
                "versions": {},
                "graph_exports": [],
                "collections": dict(physical_collections),
            },
        )

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
            "graph_exports": [],
            "collections": dict(physical_collections),
        },
    )
    monkeypatch.setattr(
        snapshot_mod,
        "latest_snapshot_dir",
        lambda *_a, **_k: snapshot_dir,
    )

    # Replace background job manager with a synchronous fake so success renders
    # deterministically in the same AppTest run.

    class _CorpusMutationJobManager:
        def is_exclusivity_active(self, _key: str) -> bool:
            return False

        def exclusivity_activity_snapshot(
            self, key: str
        ) -> tuple[bool, bg.JobActivitySnapshot]:
            return self.is_exclusivity_active(key), fake_job_manager.activity_snapshot()

        def start_job(  # type: ignore[no-untyped-def]
            self, *, owner_id: str, fn, exclusivity_key: str | None = None
        ):
            assert exclusivity_key == "corpus-mutation"
            return fake_job_manager.start_job(owner_id=owner_id, fn=fn)

        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(fake_job_manager, name)

    manager = _CorpusMutationJobManager()
    monkeypatch.setattr(bg, "get_job_manager", lambda: manager)
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
    assert state is None

    success_messages = [msg.value for msg in result.success]
    assert any("Snapshot created" in value for value in success_messages)
    info_messages = [message.value for message in result.info]
    router_ready = any(
        "Router engine is ready for Chat." in value for value in info_messages
    )
    assert router_ready, info_messages
    assert result.session_state["latest_manifest"]["corpus_hash"] == "c" * 64
    assert (
        next(button for button in result.button if button.label == "Ingest").disabled
        is False
    )
    assert (
        next(button for button in result.button if button.label == "Rebuild").disabled
        is False
    )

    followup = result.run()
    assert not followup.exception
    assert not any("Snapshot created" in msg.value for msg in followup.success)
    confirm = next(
        checkbox
        for checkbox in followup.checkbox
        if checkbox.label == 'I understand deleting "doc.txt" cannot be undone'
    )
    confirmed = confirm.check().run()
    delete = next(
        button for button in confirmed.button if button.label == 'Delete "doc.txt"'
    )
    assert delete.disabled is False


@pytest.mark.integration
def test_active_ingestion_blocks_a_second_mutation(
    documents_ingest_app_test: AppTest,
    monkeypatch,
) -> None:
    """Keep a second corpus worker from starting while ingestion is active."""

    class _RunningState:
        status = "running"
        result = None
        error = None

    class _RunningJobManager:
        def __init__(self) -> None:
            self.submissions = 0
            self.owner_id: str | None = None

        def is_exclusivity_active(self, _key: str) -> bool:
            return self.submissions > 0

        def activity_snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(
                has_active_jobs=self.submissions > 0,
                maintenance_active=False,
            )

        def exclusivity_activity_snapshot(
            self, key: str
        ) -> tuple[bool, SimpleNamespace]:
            return self.is_exclusivity_active(key), self.activity_snapshot()

        def start_job(  # type: ignore[no-untyped-def]
            self, *, owner_id: str, fn, exclusivity_key: str | None = None
        ) -> str:
            del fn
            assert exclusivity_key == "corpus-mutation"
            self.submissions += 1
            self.owner_id = owner_id
            return "running-job"

        def get(self, job_id: str, *, owner_id: str):  # type: ignore[no-untyped-def]
            if job_id == "running-job" and owner_id == self.owner_id:
                return _RunningState()
            return None

        def drain_progress(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return []

        def cancel(self, *_args, **_kwargs) -> bool:  # type: ignore[no-untyped-def]
            return True

    manager = _RunningJobManager()
    monkeypatch.setattr(bg, "get_job_manager", lambda: manager)
    from src.config.settings import settings as app_settings

    uploads = app_settings.data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "existing.txt").write_text("existing", encoding="utf-8")

    app = documents_ingest_app_test.run()
    first_ingest = next(button for button in app.button if button.label == "Ingest")
    active = first_ingest.click().run()
    assert not active.exception
    assert manager.submissions == 1

    second_ingest = next(button for button in active.button if button.label == "Ingest")
    rebuild = next(button for button in active.button if button.label == "Rebuild")
    delete = next(
        button for button in active.button if button.label == 'Delete "existing.txt"'
    )
    assert second_ingest.disabled is True
    assert rebuild.disabled is True
    assert delete.disabled is True

    attempted = second_ingest.click().run()
    assert not attempted.exception
    assert manager.submissions == 1


@pytest.mark.integration
def test_foreign_corpus_mutation_disables_all_mutation_controls(
    documents_ingest_app_test: AppTest,
    monkeypatch,
) -> None:
    """Expose only foreign occupancy while disabling every corpus mutation."""

    class _ForeignMutationManager:
        def __init__(self) -> None:
            self.submissions = 0
            self.active = False

        def is_exclusivity_active(self, key: str) -> bool:
            assert key == "corpus-mutation"
            return self.active

        def activity_snapshot(self) -> SimpleNamespace:
            return SimpleNamespace(
                has_active_jobs=self.active,
                maintenance_active=False,
            )

        def exclusivity_activity_snapshot(
            self, key: str
        ) -> tuple[bool, SimpleNamespace]:
            return self.is_exclusivity_active(key), self.activity_snapshot()

        def start_job(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.submissions += 1
            raise AssertionError("foreign occupancy reached job submission")

    manager = _ForeignMutationManager()
    monkeypatch.setattr(bg, "get_job_manager", lambda: manager)
    from src.config.settings import settings as app_settings

    uploads = app_settings.data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "existing.txt").write_text("existing", encoding="utf-8")

    app = documents_ingest_app_test.run()
    assert not app.exception
    assert (
        next(button for button in app.button if button.label == "Ingest").disabled
        is False
    )
    assert (
        next(button for button in app.button if button.label == "Rebuild").disabled
        is False
    )

    manager.active = True
    acquired = app.run()
    assert not acquired.exception
    assert any(
        "Another session is changing the corpus" in message.value
        for message in acquired.info
    )
    ingest = next(button for button in acquired.button if button.label == "Ingest")
    rebuild = next(button for button in acquired.button if button.label == "Rebuild")
    assert ingest.disabled is True
    assert rebuild.disabled is True

    confirm = next(
        checkbox
        for checkbox in acquired.checkbox
        if checkbox.label == 'I understand deleting "existing.txt" cannot be undone'
    )
    confirmed = confirm.check().run()
    delete = next(
        button for button in confirmed.button if button.label == 'Delete "existing.txt"'
    )
    assert delete.disabled is True
    ingest.click().run()
    assert manager.submissions == 0

    steady = confirmed.run()
    assert not steady.exception
    assert (
        next(button for button in steady.button if button.label == "Ingest").disabled
        is True
    )

    manager.active = False
    released = steady.run()
    assert not released.exception
    assert (
        next(button for button in released.button if button.label == "Ingest").disabled
        is False
    )
    assert (
        next(button for button in released.button if button.label == "Rebuild").disabled
        is False
    )
    released_confirm = next(
        checkbox
        for checkbox in released.checkbox
        if checkbox.label == 'I understand deleting "existing.txt" cannot be undone'
    )
    if not released_confirm.value:
        released = released_confirm.check().run()
    assert (
        next(
            button
            for button in released.button
            if button.label == 'Delete "existing.txt"'
        ).disabled
        is False
    )


@pytest.mark.integration
def test_delete_confirmation_is_bound_to_selected_file(
    documents_ingest_app_test: AppTest,
) -> None:
    """Require independent confirmation after the deletion target changes."""
    from src.config.settings import settings as app_settings

    uploads = app_settings.data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "a.txt").write_text("a", encoding="utf-8")
    (uploads / "b.txt").write_text("b", encoding="utf-8")

    app = documents_ingest_app_test.run()
    assert not app.exception
    selection = next(box for box in app.selectbox if box.label == "Select file")
    selected_a = selection.select("a.txt").run()
    confirm_a = next(
        box
        for box in selected_a.checkbox
        if box.label == 'I understand deleting "a.txt" cannot be undone'
    )
    confirmed_a = confirm_a.check().run()
    delete_a = next(
        button for button in confirmed_a.button if button.label == 'Delete "a.txt"'
    )
    assert delete_a.disabled is False

    selection = next(box for box in confirmed_a.selectbox if box.label == "Select file")
    selected_b = selection.select("b.txt").run()
    confirm_b = next(
        box
        for box in selected_b.checkbox
        if box.label == 'I understand deleting "b.txt" cannot be undone'
    )
    delete_b = next(
        button for button in selected_b.button if button.label == 'Delete "b.txt"'
    )
    assert confirm_b.value is False
    assert delete_b.disabled is True
