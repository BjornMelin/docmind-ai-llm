"""Focused tests for Documents-page upload transactions."""

from __future__ import annotations

import hashlib
import importlib
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from src.models.processing import IngestionConfig, IngestionInput, ParsingOverrides
from src.persistence.snapshot_service import SnapshotActivation
from src.persistence.upload_journal import recover_upload_quarantines
from src.ui.vector_session import VectorIndexResource
from src.utils.hashing import document_id_from_sha256

pytestmark = pytest.mark.unit

_COLLECTIONS = {"text": "text__build", "image": "image__build"}


def _page():  # type: ignore[no-untyped-def]
    return importlib.import_module("src.pages.02_documents")


def _input(path: Path) -> IngestionInput:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return IngestionInput(
        document_id=document_id_from_sha256(digest),
        source_path=path,
    )


class _Manager:
    def __init__(self, workspace: Path, events: list[str] | None = None) -> None:
        self.workspace = workspace
        self.events = events if events is not None else []
        self.cleanup_calls: list[Path] = []
        self.locked = False

    def begin_snapshot(self) -> Path:
        self.locked = True
        self.events.append("begin")
        self.workspace.mkdir(parents=True)
        return self.workspace

    def cleanup_tmp(self, workspace: Path) -> None:
        self.events.append("cleanup")
        self.cleanup_calls.append(workspace)
        self.locked = False


def _patch_job_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    page: object,
    manager: _Manager,
) -> None:
    monkeypatch.setattr(page, "SnapshotManager", lambda _storage: manager)
    monkeypatch.setattr(
        page,
        "_physical_collection_names",
        lambda _workspace: dict(_COLLECTIONS),
    )
    monkeypatch.setattr(
        page,
        "_read_collection_metadata",
        lambda _collections: {
            "text": {"docmind_owner": "text"},
            "image": {"docmind_owner": "image"},
        },
    )


def test_saved_upload_stays_pending_until_worker_starts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The Streamlit rerun persists bytes outside the authoritative corpus."""
    from src.config.settings import settings

    page = _page()
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    upload = SimpleNamespace(
        name="report.txt",
        size=7,
        getbuffer=lambda: memoryview(b"content"),
        seek=lambda _offset: None,
    )

    inputs = page._save_ingestion_inputs(  # type: ignore[attr-defined]
        [upload],
        encrypt_images=False,
        parsing_overrides=ParsingOverrides(),
    )

    assert len(inputs) == 1
    source = Path(inputs[0].source_path)
    assert source.is_relative_to(tmp_path / ".pending-uploads")
    assert source.read_bytes() == b"content"
    assert not (tmp_path / "uploads").exists()


def test_worker_promotes_pending_upload_at_activation_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Promotion occurs only after the worker owns the snapshot transaction."""
    from src.config.settings import settings

    page = _page()
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    pending = tmp_path / ".pending-uploads" / "tx" / "report.txt"
    pending.parent.mkdir(parents=True)
    pending.write_text("content", encoding="utf-8")
    item = _input(pending)
    events: list[str] = []
    manager = _Manager(tmp_path / "storage" / "_tmp-build", events)
    _patch_job_boundaries(monkeypatch, page, manager)
    resource = VectorIndexResource(object())
    monkeypatch.setattr(page, "_delete_staged_collections", lambda _names: None)

    def _ingest(inputs, **_kwargs):  # type: ignore[no-untyped-def]
        assert manager.locked
        events.append("ingest")
        staged = Path(inputs[0].source_path)
        assert staged == pending
        assert staged.read_text(encoding="utf-8") == "content"
        assert not (tmp_path / "uploads").exists() or not any(
            (tmp_path / "uploads").iterdir()
        )
        return {
            "count": 1,
            "documents": [object()],
            "activation_corpus_hash": "c" * 64,
            "activation_config": {"x": 1},
            "activation_config_hash": "f" * 64,
            "snapshot_config_hash": "e" * 64,
            "manifest": {"corpus_hash": "c" * 64},
            "vector_resource": resource,
            "pg_index": None,
            "collections": dict(_COLLECTIONS),
        }

    monkeypatch.setattr(page, "ingest_inputs", _ingest)

    def _rebuild(*_args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["commit_source_changes"]()
        assert manager.locked is True
        recover_upload_quarantines(
            data_dir=tmp_path,
            active_collections=_COLLECTIONS,
        )
        events.append("commit")
        return page.FinalizedSnapshot(
            path=tmp_path / "storage" / "active",
            manifest={
                "corpus_hash": "c" * 64,
                "config_hash": "f" * 64,
                "versions": {},
                "graph_exports": [],
            },
        )

    monkeypatch.setattr(page, "rebuild_snapshot", _rebuild)

    result = page._run_ingest_job(  # type: ignore[attr-defined]
        [item],
        use_graphrag=False,
        encrypt_images=False,
        nlp_service=None,
        cancel_event=threading.Event(),
        report_progress=lambda _event: None,
        rollback_source_paths=(pending,),
        runtime_generation=settings.cache_version,
    )

    assert events == ["begin", "ingest", "commit"]
    assert not pending.exists()
    assert result["snapshot_dir"].endswith("/active")
    assert result["manifest"] == {
        "corpus_hash": "c" * 64,
        "config_hash": "f" * 64,
        "versions": {},
        "graph_exports": [],
    }
    promoted = next((tmp_path / "uploads").iterdir())
    assert promoted.read_text(encoding="utf-8") == "content"
    assert not (tmp_path / ".upload-transactions").exists()
    resource.close()


def test_corpus_exclusivity_covers_bounded_terminal_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The corpus lease survives activation until the public DTO is complete."""
    from src.config.settings import settings
    from src.ui.background_jobs import JobConflictError, JobManager

    page = _page()
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    pending = tmp_path / ".pending-uploads" / "tx" / "report.txt"
    pending.parent.mkdir(parents=True)
    pending.write_text("content", encoding="utf-8")
    manager = _Manager(tmp_path / "storage" / "_tmp-build")
    _patch_job_boundaries(monkeypatch, page, manager)
    resource = VectorIndexResource(object())
    job_manager = JobManager(max_workers=1)
    presentation_started = threading.Event()
    release_presentation = threading.Event()

    monkeypatch.setattr(page, "_delete_staged_collections", lambda _names: None)

    def _ingest(_inputs, **_kwargs):  # type: ignore[no-untyped-def]
        return {
            "count": 1,
            "documents": [object()],
            "activation_corpus_hash": "c" * 64,
            "activation_config": {"x": 1},
            "activation_config_hash": "f" * 64,
            "snapshot_config_hash": "e" * 64,
            "vector_resource": resource,
            "pg_index": None,
            "collections": dict(_COLLECTIONS),
        }

    def _rebuild(*_args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["commit_source_changes"]()
        return page.FinalizedSnapshot(
            path=tmp_path / "storage" / "active",
            manifest={
                "corpus_hash": "c" * 64,
                "config_hash": "f" * 64,
                "versions": {},
                "graph_exports": [],
            },
        )

    bounded_presentation = page._bounded_manifest_presentation

    def _block_presentation(manifest):  # type: ignore[no-untyped-def]
        presentation_started.set()
        assert release_presentation.wait(timeout=5)
        return bounded_presentation(manifest)

    monkeypatch.setattr(page, "ingest_inputs", _ingest)
    monkeypatch.setattr(page, "rebuild_snapshot", _rebuild)
    monkeypatch.setattr(page, "_bounded_manifest_presentation", _block_presentation)

    def _work(cancel_event, report_progress):  # type: ignore[no-untyped-def]
        return page._run_ingest_job(
            [_input(pending)],
            use_graphrag=False,
            encrypt_images=False,
            nlp_service=None,
            cancel_event=cancel_event,
            report_progress=report_progress,
            rollback_source_paths=(pending,),
            runtime_generation=settings.cache_version,
        )

    job_id = job_manager.start_job(
        owner_id="owner",
        fn=_work,
        exclusivity_key="corpus-mutation",
    )
    try:
        assert presentation_started.wait(timeout=2)
        assert job_manager.is_exclusivity_active("corpus-mutation")
        with pytest.raises(JobConflictError):
            job_manager.start_job(
                owner_id="other",
                fn=lambda _cancel, _report: None,
                exclusivity_key="corpus-mutation",
            )

        release_presentation.set()
        assert job_manager.wait_for_completion(job_id, owner_id="owner") == "succeeded"
        state = job_manager.get(job_id, owner_id="owner")
        assert state is not None
        assert state.result["manifest"] == {
            "corpus_hash": "c" * 64,
            "config_hash": "f" * 64,
            "versions": {},
            "graph_exports": [],
        }
        assert not job_manager.is_exclusivity_active("corpus-mutation")
        assert job_manager.consume_terminal(job_id, owner_id="owner")
    finally:
        release_presentation.set()
        resource.close()
        job_manager.shutdown()


def test_duplicate_content_is_rejected_and_pending_bytes_are_removed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Content identity, not the uploaded filename, owns corpus uniqueness."""
    from src.config.settings import settings

    page = _page()
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    existing = tmp_path / "uploads" / "existing.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("same", encoding="utf-8")
    pending = tmp_path / ".pending-uploads" / "tx" / "duplicate.txt"
    pending.parent.mkdir(parents=True)
    pending.write_text("same", encoding="utf-8")
    manager = _Manager(tmp_path / "storage" / "_tmp-build")
    _patch_job_boundaries(monkeypatch, page, manager)
    monkeypatch.setattr(page, "_delete_staged_collections", lambda _names: None)

    with pytest.raises(ValueError, match="already exists"):
        page._run_ingest_job(  # type: ignore[attr-defined]
            [_input(pending)],
            use_graphrag=False,
            encrypt_images=False,
            nlp_service=None,
            cancel_event=threading.Event(),
            report_progress=lambda _event: None,
            rollback_source_paths=(pending,),
            runtime_generation=settings.cache_version,
        )

    assert existing.read_text(encoding="utf-8") == "same"
    assert not pending.exists()
    assert manager.cleanup_calls == [manager.workspace]
    assert [path.name for path in (tmp_path / "uploads").iterdir()] == ["existing.txt"]


def test_failed_build_removes_promoted_source_before_releasing_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed generation cannot leak its newly authoritative upload."""
    from src.config.settings import settings

    page = _page()
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    pending = tmp_path / ".pending-uploads" / "tx" / "report.txt"
    pending.parent.mkdir(parents=True)
    pending.write_text("content", encoding="utf-8")
    manager = _Manager(tmp_path / "storage" / "_tmp-build")
    _patch_job_boundaries(monkeypatch, page, manager)
    staged_cleanup: list[dict[str, str]] = []

    def _fail_ingest(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        assert manager.locked
        assert pending.is_file()
        assert not (tmp_path / "uploads").exists() or not any(
            (tmp_path / "uploads").iterdir()
        )
        raise RuntimeError("build failed")

    def _delete_staged(collections: dict[str, str]) -> None:
        assert manager.locked
        assert not any((tmp_path / "uploads").iterdir())
        staged_cleanup.append(dict(collections))

    monkeypatch.setattr(page, "ingest_inputs", _fail_ingest)
    monkeypatch.setattr(page, "_delete_staged_collections", _delete_staged)

    with pytest.raises(RuntimeError, match="build failed"):
        page._run_ingest_job(  # type: ignore[attr-defined]
            [_input(pending)],
            use_graphrag=False,
            encrypt_images=False,
            nlp_service=None,
            cancel_event=threading.Event(),
            report_progress=lambda _event: None,
            rollback_source_paths=(pending,),
            runtime_generation=settings.cache_version,
        )

    assert staged_cleanup == [_COLLECTIONS]
    assert manager.cleanup_calls == [manager.workspace]
    assert not manager.locked
    assert not any((tmp_path / "uploads").iterdir())
    assert not (tmp_path / ".upload-transactions").exists()


def test_path_exclusion_keeps_same_content_at_a_different_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Deleting one path does not accidentally exclude its content twin."""
    from src.config.settings import settings
    from src.ui import ingest_adapter

    page = _page()
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True)
    target = uploads / "target.txt"
    remaining = uploads / "remaining.txt"
    target.write_text("same", encoding="utf-8")
    remaining.write_text("same", encoding="utf-8")
    current = [_input(remaining)]

    additional, corpus_ids = ingest_adapter._additional_corpus_inputs(
        current,
        IngestionConfig(enable_image_encryption=False),
        excluded_source_paths=(target,),
    )

    assert additional == []
    assert corpus_ids == {current[0].document_id}
    assert (
        page._existing_corpus_inputs(  # type: ignore[attr-defined]
            uploads,
            encrypt=False,
            excluded_path=target,
        )[0].source_path
        == remaining
    )


def test_final_document_deletion_commits_empty_generation_with_graphrag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GraphRAG-enabled deletion accepts a verified empty replacement corpus."""
    from src.config.settings import settings

    page = _page()
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    target = tmp_path / "uploads" / "last.txt"
    target.parent.mkdir(parents=True)
    target.write_text("last document", encoding="utf-8")
    manager = _Manager(tmp_path / "storage" / "_tmp-build")
    _patch_job_boundaries(monkeypatch, page, manager)
    resource = VectorIndexResource(object())
    ingest_calls: list[dict[str, object]] = []
    rebuild_calls: list[object] = []

    def _ingest(inputs, **kwargs):  # type: ignore[no-untyped-def]
        assert inputs == []
        ingest_calls.append(kwargs)
        return {
            "count": 0,
            "documents": [],
            "activation_corpus_hash": "c" * 64,
            "activation_config": {"x": 1},
            "activation_config_hash": "f" * 64,
            "snapshot_config_hash": "e" * 64,
            "vector_resource": resource,
            "pg_index": None,
            "collections": dict(_COLLECTIONS),
        }

    def _rebuild(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["commit_source_changes"]()
        assert not target.exists()
        assert manager.locked is True
        recover_upload_quarantines(
            data_dir=tmp_path,
            active_collections=_COLLECTIONS,
        )
        rebuild_calls.append(args[3])
        return page.FinalizedSnapshot(
            path=tmp_path / "storage" / "active",
            manifest={
                "corpus_hash": "c" * 64,
                "config_hash": "f" * 64,
                "versions": {},
                "graph_exports": [],
            },
        )

    monkeypatch.setattr(page, "ingest_inputs", _ingest)
    monkeypatch.setattr(page, "rebuild_snapshot", _rebuild)

    result = page._run_ingest_job(  # type: ignore[attr-defined]
        [],
        use_graphrag=True,
        encrypt_images=False,
        nlp_service=None,
        cancel_event=threading.Event(),
        report_progress=lambda _event: None,
        excluded_source_paths=(target,),
        quarantine_source=target,
        runtime_generation=settings.cache_version,
    )

    assert ingest_calls[0]["enable_graphrag"] is True
    assert ingest_calls[0]["excluded_source_paths"] == (target,)
    activation = cast(SnapshotActivation, rebuild_calls[0])
    assert isinstance(activation, SnapshotActivation)
    assert activation.graph_requested is False
    assert result["ingest"]["count"] == 0
    assert not target.exists()
    assert not (tmp_path / ".quarantine").exists()
    resource.close()
