"""Streamlit Documents page wiring the ingestion pipeline and snapshots."""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import streamlit as st
from loguru import logger

from src.config import settings
from src.models.processing import (
    IngestionInput,
    ParsingOverrides,
)
from src.persistence.artifacts import ArtifactRef, ArtifactStore
from src.persistence.upload_journal import (
    promote_pending_uploads,
    quarantine_upload,
    restore_quarantined_upload,
    rollback_upload_promotion,
)
from src.processing.ingestion_api import require_unique_document_ids
from src.processing.parsing.health import parser_health
from src.telemetry.opentelemetry import (
    configure_observability,
    record_graph_export_metric,
)
from src.ui.artifacts import render_artifact_image
from src.ui.background_jobs import (
    ForegroundRuntimeConflictError,
    JobAdmissionPausedError,
    JobCanceledError,
    JobConflictError,
    JobManager,
    JobStateView,
    ProgressEvent,
    get_job_manager,
    get_or_create_owner_id,
)
from src.ui.router_session import session_router_is_current
from src.ui.vector_session import (
    VectorIndexResource,
    clear_session_runtime,
    replace_session_runtime,
    session_vector_resource_is_current,
)
from src.utils.hashing import document_id_from_sha256, sha256_file

ProgressReporter = Callable[[ProgressEvent], None]
_PHYSICAL_COLLECTION_MAX_LENGTH = 200
_CORPUS_MUTATION_KEY = "corpus-mutation"
_CORPUS_ACTIVITY_OBSERVED_KEY = "corpus_activity_observed"
_INGEST_TERMINAL_PRESENTATION_KEY = "ingest_job_terminal_notice"
_INVALID_INGEST_RESULT_MESSAGE = (
    "Ingestion finished with an invalid result. Please try again."
)
_INGEST_DISPLAY_FAILURE_MESSAGE = (
    "Ingestion results could not be displayed. Please try again."
)
_TERMINAL_MAINTENANCE_WAIT_COPY = (
    "Runtime maintenance is in progress. Live activation will retry automatically."
)
_TERMINAL_JOB_WAIT_COPY = (
    "Other background work is active. Live activation will retry automatically."
)
_TERMINAL_FOREGROUND_WAIT_COPY = (
    "The live runtime is in use. Live activation will retry automatically."
)
_NLP_ENTITY_PRESENTATION_LIMIT = 20
_NLP_SENTENCE_PRESENTATION_LIMIT = 10
_IMAGE_EXPORT_PRESENTATION_LIMIT = 128
_PRESENTATION_TEXT_MAX_LENGTH = 500
_PRESENTATION_ID_MAX_LENGTH = 200
_ARTIFACT_SUFFIX_CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789.-_"
_MANIFEST_VERSION_PRESENTATION_LIMIT = 32
_MANIFEST_EXPORT_PRESENTATION_LIMIT = 32
ActiveIngestionJob = tuple[str, JobStateView]


if TYPE_CHECKING:
    from src.nlp.spacy_service import SpacyNlpService
    from src.persistence.snapshot import FinalizedSnapshot, SnapshotManager


def _emit_ingest_progress(
    report_progress: ProgressReporter,
    percent: int,
    phase: str,
    message: str,
) -> None:
    """Emit one bounded background-job progress event without failing work."""
    event = ProgressEvent(
        percent=max(0, min(100, int(percent))),
        phase=phase,  # type: ignore[arg-type]
        message=str(message)[:200],
        timestamp=datetime.now(UTC),
    )
    with contextlib.suppress(Exception):
        report_progress(event)


@st.cache_resource(show_spinner=False)
def _get_spacy_service(
    cache_version: int, cfg_dump: dict[str, Any]
) -> SpacyNlpService | None:
    """Initialize and return the Spacy NLP service.

    Args:
        cache_version: Integer used to bust the Streamlit cache.
        cfg_dump: Serialized configuration dictionary.

    Returns:
        SpacyNlpService | None: Initialized service or None if disabled/failed.
    """
    from src.nlp.settings import SpacyNlpSettings
    from src.nlp.spacy_service import SpacyNlpService

    _ = cache_version  # cache bust
    cfg = SpacyNlpSettings.model_validate(cfg_dump)
    return SpacyNlpService(cfg)


def main() -> None:  # pragma: no cover - Streamlit page
    """Render the Documents page and handle ingestion form submissions."""
    configure_observability(settings)
    _clear_stale_session_runtime()
    st.title("Documents")
    _render_ingest_terminal_notice()

    owner_id = get_or_create_owner_id()
    _render_latest_snapshot_summary()
    _render_ingest_job_panel(owner_id=owner_id)

    active_job, mutation_active, maintenance_active = _resolve_corpus_mutation_activity(
        owner_id=owner_id
    )
    files, use_graphrag, encrypt_images, parsing_overrides, submitted = (
        _render_ingest_form(
            active_job=active_job,
            mutation_active=mutation_active,
            maintenance_active=maintenance_active,
        )
    )
    if submitted:
        started = _handle_ingest_submission(
            files,
            use_graphrag,
            encrypt_images,
            parsing_overrides,
            owner_id=owner_id,
        )
        if started:
            st.rerun()

    active_job, mutation_active, maintenance_active = _resolve_corpus_mutation_activity(
        owner_id=owner_id
    )
    _render_maintenance_controls(
        owner_id=owner_id,
        active_job=active_job,
        mutation_active=mutation_active,
        maintenance_active=maintenance_active,
    )

    # Snapshot utilities and manual exports are driven by session_state indices.
    _render_export_controls()


def _clear_stale_session_runtime() -> bool:
    """Clear an obsolete session runtime only while mutation is quiesced."""
    runtime_keys = (
        "_vector_index_resource",
        "_vector_runtime_generation",
        "vector_index",
        "router_engine",
        "_router_runtime_generation",
        "graphrag_index",
        "_snapshot_collections",
        "_snapshot_loaded_id",
    )

    def _mutation_needed() -> bool:
        runtime_present = any(key in st.session_state for key in runtime_keys)
        return runtime_present and not (
            session_vector_resource_is_current(
                st.session_state,
                runtime_generation=settings.cache_version,
            )
            and session_router_is_current(
                st.session_state,
                runtime_generation=settings.cache_version,
            )
        )

    if not _mutation_needed():
        return True
    try:
        with get_job_manager().admission_quiescence():
            if _mutation_needed():
                clear_session_runtime(
                    st.session_state,
                    runtime_generation=settings.cache_version,
                )
    except JobAdmissionPausedError:
        st.caption("Stale runtime cleanup deferred during runtime maintenance.")
        return False
    except ForegroundRuntimeConflictError:
        st.caption("Stale runtime cleanup deferred while the live runtime is in use.")
        return False
    except JobConflictError:
        st.caption("Stale runtime cleanup deferred while background work is active.")
        return False
    return True


def _render_ingest_form(
    *,
    active_job: ActiveIngestionJob | None,
    mutation_active: bool,
    maintenance_active: bool,
) -> tuple[list[Any] | None, bool, bool, ParsingOverrides, bool]:
    """Render the ingestion form and return submitted values."""
    # Keep dependent parser controls outside the form so changing the global
    # defaults toggle reruns immediately and enables overrides before submit.
    parsing_overrides = _render_parsing_overrides()
    if maintenance_active:
        st.info(
            "Runtime maintenance is in progress. Ingestion is temporarily disabled."
        )
    elif mutation_active:
        st.info(_corpus_mutation_copy(active_job))
    with st.form("ingest_form", clear_on_submit=False):
        files = st.file_uploader("Add files", type=None, accept_multiple_files=True)
        col_opts = st.columns(2)
        with col_opts[0]:
            use_graphrag = st.checkbox(
                "Build GraphRAG (beta)",
                value=bool(settings.graphrag_cfg.enabled),
            )
        with col_opts[1]:
            encrypt_images = st.checkbox(
                "Encrypt page images (AES-GCM)",
                value=bool(settings.processing.encrypt_page_images),
            )
        with st.expander("About snapshots", expanded=False):
            st.markdown(
                "- Snapshots are created atomically in data/storage/<timestamp>.\n"
                "- manifest.meta.json stores corpus/config hashes for "
                "staleness detection.\n"
                "- Rebuild a snapshot anytime; for new content, re-ingest first."
            )
        with st.expander("Snapshot parsing metadata", expanded=False):
            st.json(
                {
                    "profile": settings.parsing.profile,
                    "ocr_engine": settings.ocr.engine,
                },
                expanded=False,
            )
        submitted = st.form_submit_button(
            "Ingest",
            disabled=mutation_active or maintenance_active,
        )
    return files, use_graphrag, encrypt_images, parsing_overrides, submitted


def _render_parsing_overrides() -> ParsingOverrides:
    """Render per-ingestion parser overrides."""
    with st.expander("Parsing overrides", expanded=False):
        use_global = st.checkbox("Use global parsing defaults", value=True)
        force_ocr = st.checkbox(
            "Force RapidOCR",
            value=False,
            disabled=use_global,
        )
        export_searchable_pdf = st.checkbox(
            "Export searchable PDF",
            value=False,
            disabled=use_global,
        )
        if use_global:
            return ParsingOverrides()
    return ParsingOverrides(
        force_ocr=force_ocr,
        export_searchable_pdf=export_searchable_pdf,
    )


def _handle_ingest_submission(
    files: list[Any] | None,
    use_graphrag: bool,
    encrypt_images: bool,
    parsing_overrides: ParsingOverrides,
    *,
    owner_id: str,
) -> bool:
    """Handle ingestion submission by starting a background job.

    Args:
        files: List of uploaded file objects from Streamlit.
        use_graphrag: Whether to trigger GraphRAG snapshot rebuild.
        encrypt_images: Whether to encrypt extracted images.
        parsing_overrides: Per-ingestion parser override values.
        owner_id: Unique identifier for the job owner.
    """
    if not files:
        st.warning("No files selected.")
        return False
    if _reject_active_corpus_mutation(owner_id=owner_id):
        return False
    if not _pdf_uploads_are_ready(files):
        return False
    try:
        _require_unique_upload_document_ids(files)
        nlp_service = _load_optional_spacy_service()
        saved_inputs = _save_ingestion_inputs(
            files,
            encrypt_images=encrypt_images,
            parsing_overrides=parsing_overrides,
        )
        if not saved_inputs:
            st.warning("No valid files to ingest.")
            return False
        return _start_ingestion_job(
            saved_inputs,
            use_graphrag=use_graphrag,
            encrypt_images=encrypt_images,
            nlp_service=nlp_service,
            owner_id=owner_id,
            rollback_source_paths=tuple(
                Path(item.source_path)
                for item in saved_inputs
                if item.source_path is not None
            ),
        )
    except Exception as exc:  # pragma: no cover - UX best effort
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="documents.ingest_start")
        logger.warning(
            "Failed to start ingestion job (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        st.error(f"Failed to start ingestion job ({type(exc).__name__}).")
        st.caption(f"Error reference: {redaction.redacted}")
        return False


def _require_unique_upload_document_ids(files: list[Any]) -> None:
    """Reject duplicate upload content before any file is persisted."""
    seen: set[str] = set()
    for file_obj in files:
        if file_obj is None or getattr(file_obj, "name", None) is None:
            continue
        file_size = getattr(file_obj, "size", None)
        if isinstance(file_size, int) and file_size <= 0:
            continue
        try:
            payload = file_obj.getbuffer()
        except AttributeError as exc:
            raise TypeError("Uploaded file must expose getbuffer()") from exc
        document_id = document_id_from_sha256(hashlib.sha256(payload).hexdigest())
        if document_id in seen:
            raise ValueError(f"Duplicate document_id in ingestion batch: {document_id}")
        seen.add(document_id)


def _pdf_uploads_are_ready(files: list[Any]) -> bool:
    """Return whether selected PDF uploads can be parsed offline."""
    has_pdf = any(
        str(getattr(file_obj, "name", "")).lower().endswith(".pdf")
        for file_obj in files
    )
    if not has_pdf:
        return True
    readiness = parser_health(settings)
    if readiness["pdf_dependencies_ready"]:
        return True
    st.error("PDF parser dependencies or Docling models are unavailable.")
    st.code(str(readiness["prefetch_command"]))
    return False


def _load_optional_spacy_service() -> SpacyNlpService | None:
    """Load optional spaCy enrichment without blocking document ingestion."""
    try:
        cfg_dump = settings.spacy.model_dump()  # type: ignore[attr-defined]
        return _get_spacy_service(settings.cache_version, cfg_dump)
    except Exception as exc:
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="documents.spacy_init")
        logger.warning(
            "Failed to initialize SpacyNlpService (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        return None


def _save_ingestion_inputs(
    files: list[Any],
    *,
    encrypt_images: bool,
    parsing_overrides: ParsingOverrides,
) -> list[IngestionInput]:
    """Persist valid uploads and return canonical ingestion inputs."""
    pending_dir = settings.data_dir / ".pending-uploads" / uuid4().hex
    pending_dir.mkdir(parents=True, exist_ok=False)
    saved_inputs: list[IngestionInput] = []
    for file_obj in files:
        if file_obj is None:
            logger.warning("Skipping file (missing upload object).")
            continue
        file_name = getattr(file_obj, "name", None)
        file_size = getattr(file_obj, "size", None)
        if file_name is None:
            logger.warning("Skipping file (missing name attribute).")
            continue
        if isinstance(file_size, int) and file_size <= 0:
            logger.warning("Skipping file (empty upload).")
            continue
        saved_input = _save_ingestion_input(
            file_obj,
            file_name=str(file_name),
            encrypt_images=encrypt_images,
            parsing_overrides=parsing_overrides,
            destination_dir=pending_dir,
        )
        if saved_input is not None:
            saved_inputs.append(saved_input)
    if not saved_inputs:
        with contextlib.suppress(OSError):
            pending_dir.rmdir()
            pending_dir.parent.rmdir()
    return saved_inputs


def _save_ingestion_input(
    file_obj: Any,
    *,
    file_name: str,
    encrypt_images: bool,
    parsing_overrides: ParsingOverrides,
    destination_dir: Path,
) -> IngestionInput | None:
    """Persist one upload and return its ingestion contract when successful."""
    from src.ui.ingest_adapter import save_uploaded_file

    try:
        stored_path, digest = save_uploaded_file(
            file_obj,
            destination_dir=destination_dir,
        )
    except Exception as exc:
        from src.utils.log_safety import build_pii_log_entry

        error_redaction = build_pii_log_entry(str(exc), key_id="documents.save_file")
        name_redaction = build_pii_log_entry(file_name, key_id="documents.upload_name")
        logger.warning(
            "Skipping file (name={} error_type={} error={})",
            name_redaction.redacted,
            type(exc).__name__,
            error_redaction.redacted,
        )
        return None
    return IngestionInput(
        document_id=document_id_from_sha256(digest),
        source_path=stored_path,
        metadata={
            "uploaded_at": datetime.now(UTC).isoformat(),
            "sha256": digest,
        },
        encrypt_images=bool(encrypt_images),
        parsing_overrides=parsing_overrides,
    )


def _start_ingestion_job(
    saved_inputs: list[IngestionInput],
    *,
    use_graphrag: bool,
    encrypt_images: bool,
    nlp_service: SpacyNlpService | None,
    owner_id: str,
    rollback_source_paths: tuple[Path, ...] = (),
    excluded_source_paths: tuple[Path, ...] = (),
    quarantine_source: Path | None = None,
) -> bool:
    """Start one background ingestion job and persist its UI state."""
    try:
        require_unique_document_ids(saved_inputs)
        if _reject_active_corpus_mutation(owner_id=owner_id):
            _delete_rollback_sources(rollback_source_paths)
            return False
        job_manager = get_job_manager()
        submission_runtime_generation = int(settings.cache_version)

        def _work(
            cancel_event: threading.Event, report: ProgressReporter
        ) -> dict[str, Any]:
            return _run_ingest_job(
                saved_inputs,
                use_graphrag=use_graphrag,
                encrypt_images=encrypt_images,
                nlp_service=nlp_service,
                cancel_event=cancel_event,
                report_progress=report,
                rollback_source_paths=rollback_source_paths,
                excluded_source_paths=excluded_source_paths,
                quarantine_source=quarantine_source,
                runtime_generation=submission_runtime_generation,
            )

        job_id = job_manager.start_job(
            owner_id=owner_id,
            fn=_work,
            exclusivity_key=_CORPUS_MUTATION_KEY,
        )
    except JobAdmissionPausedError:
        _delete_rollback_sources(rollback_source_paths)
        st.warning("Runtime maintenance is in progress. Try ingestion again shortly.")
        return False
    except JobConflictError:
        _delete_rollback_sources(rollback_source_paths)
        st.warning("Another corpus change is already running.")
        return False
    except Exception:
        _delete_rollback_sources(rollback_source_paths)
        raise
    st.session_state["ingest_job_id"] = job_id
    st.session_state.pop("ingest_job_last_event", None)
    st.session_state.pop("ingest_job_cancel_requested_id", None)
    st.session_state["ingest_job_use_graphrag"] = bool(use_graphrag)
    st.session_state["ingest_job_encrypt_images"] = bool(encrypt_images)
    st.toast("Ingestion started", icon="⏳")
    return True


def _run_ingest_job(
    inputs: list[IngestionInput],
    *,
    use_graphrag: bool,
    encrypt_images: bool,
    nlp_service: SpacyNlpService | None,
    cancel_event: threading.Event,
    report_progress: ProgressReporter,
    rollback_source_paths: tuple[Path, ...] = (),
    excluded_source_paths: tuple[Path, ...] = (),
    quarantine_source: Path | None = None,
    runtime_generation: int,
) -> dict[str, Any]:
    """Worker entrypoint for ingestion + snapshot rebuild (no Streamlit APIs).

    Args:
        inputs: List of ingestion input objects.
        use_graphrag: Whether to perform GraphRAG indexing.
        encrypt_images: Whether to encrypt image artifacts.
        nlp_service: Optional NLP service for entity enrichment.
        cancel_event: Event to monitor for cooperative cancellation.
        report_progress: Function to emit progress events.
        rollback_source_paths: Newly persisted uploads removed if activation fails.
        excluded_source_paths: Exact upload paths omitted from the new generation.
        quarantine_source: Existing upload moved out of the corpus before commit.
        runtime_generation: Live runtime generation captured before job submission.

    Returns:
        dict[str, Any]: Mapping of document IDs to their respective results.

    Raises:
        JobCanceledError: If cancellation is requested.
    """
    from src.persistence.snapshot import SnapshotManager
    from src.ui.ingest_adapter import ingest_inputs

    transaction = _IngestTransaction(
        manager=SnapshotManager(settings.data_dir / "storage"),
        owned_source_paths=rollback_source_paths,
    )
    try:
        _emit_ingest_progress(
            report_progress, 0, "save", f"Prepared {len(inputs)} file(s)"
        )
        if cancel_event.is_set():
            raise JobCanceledError()

        workspace = transaction.manager.begin_snapshot()
        transaction.workspace = workspace
        collections = _physical_collection_names(workspace)
        transaction.collections = collections
        inputs, promotion_moves = _plan_pending_inputs(
            inputs,
        )
        transaction.promotion_moves = promotion_moves
        _emit_ingest_progress(
            report_progress, 10, "ingest", "Building an isolated corpus generation"
        )
        ingest_result = ingest_inputs(
            inputs,
            text_collection_name=collections["text"],
            image_collection_name=collections["image"],
            excluded_source_paths=excluded_source_paths,
            activation_path_aliases={
                source: destination for source, destination, _digest in promotion_moves
            },
            enable_graphrag=use_graphrag,
            encrypt_images=encrypt_images,
            nlp_service=nlp_service,
        )
        resource = ingest_result.get("vector_resource")
        if not isinstance(resource, VectorIndexResource):
            raise RuntimeError("Ingestion did not return an owned vector resource")
        transaction.resource = resource
        _emit_ingest_progress(report_progress, 70, "index", "Finalizing indices")
        if cancel_event.is_set():
            raise JobCanceledError()

        _emit_ingest_progress(
            report_progress,
            85,
            "snapshot",
            "Activating the verified corpus generation",
        )
        finalized = _activate_ingest_generation(
            transaction,
            ingest_result,
            use_graphrag=use_graphrag,
            quarantine_source=quarantine_source,
        )
        final = finalized.path
        ingest_result["snapshot_id"] = final.name
        manifest = _bounded_manifest_presentation(finalized.manifest)
        payload = {
            "ingest": ingest_result,
            "snapshot_dir": str(final),
            "manifest": manifest,
            "use_graphrag": bool(use_graphrag),
            "runtime_generation": int(runtime_generation),
        }
        _emit_ingest_progress(report_progress, 100, "done", "Done")
        transaction.transferred = True
        return payload
    finally:
        transaction.cleanup()


@dataclass(slots=True)
class _IngestTransaction:
    """Own resources and crash journals for one corpus activation attempt."""

    manager: SnapshotManager
    owned_source_paths: tuple[Path, ...]
    workspace: Path | None = None
    collections: dict[str, str] | None = None
    resource: VectorIndexResource | None = None
    quarantined_source: tuple[Path, Path] | None = None
    promotion_transaction_id: str | None = None
    promotion_moves: tuple[tuple[Path, Path, str], ...] = ()
    committed: bool = False
    transferred: bool = False

    def cleanup(self) -> None:
        """Release transient resources and roll back every uncommitted owner."""
        if not self.transferred and self.resource is not None:
            try:
                self.resource.close()
            except Exception as exc:
                logger.warning(
                    "Failed to close staged vector resource (error_type={})",
                    type(exc).__name__,
                )
        if self.committed:
            return
        try:
            if self.quarantined_source is not None:
                try:
                    restore_quarantined_upload(*self.quarantined_source)
                except (OSError, ValueError) as exc:
                    logger.error(
                        "Failed to restore quarantined upload (error_type={})",
                        type(exc).__name__,
                    )
            if self.promotion_transaction_id is not None:
                try:
                    rollback_upload_promotion(
                        data_dir=settings.data_dir,
                        transaction_id=self.promotion_transaction_id,
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    logger.error(
                        "Upload promotion rollback deferred (error_type={})",
                        type(exc).__name__,
                    )
            else:
                _delete_rollback_sources(self.owned_source_paths)
            if self.collections is not None:
                try:
                    _delete_staged_collections(self.collections)
                except Exception as exc:
                    logger.warning(
                        "Staged collection cleanup failed (error_type={})",
                        type(exc).__name__,
                    )
        finally:
            if self.workspace is not None:
                self.manager.cleanup_tmp(self.workspace)


def _activate_ingest_generation(
    transaction: _IngestTransaction,
    ingest_result: dict[str, Any],
    *,
    use_graphrag: bool,
    quarantine_source: Path | None,
) -> FinalizedSnapshot:
    """Validate and atomically activate one completed ingestion generation."""
    from src.persistence.snapshot_service import SnapshotActivation, rebuild_snapshot

    resource = transaction.resource
    if resource is None:
        raise RuntimeError("Ingestion transaction has no owned vector resource")
    workspace = transaction.workspace
    collections = transaction.collections
    if workspace is None or collections is None:
        raise RuntimeError("Ingestion transaction was not initialized")

    ingest_result["vector_index"] = resource.index
    ingest_result["vector_resource"] = resource
    pg_index = ingest_result.get("pg_index") if use_graphrag else None
    expected_corpus_hash = ingest_result.get("activation_corpus_hash")
    if not isinstance(expected_corpus_hash, str) or len(expected_corpus_hash) != 64:
        raise RuntimeError("Ingestion did not return a corpus identity")
    expected_config_hash = ingest_result.get("snapshot_config_hash")
    activation_config_hash = ingest_result.get("activation_config_hash")
    activation_config = ingest_result.get("activation_config")
    if (
        not isinstance(expected_config_hash, str)
        or len(expected_config_hash) != 64
        or not isinstance(activation_config_hash, str)
        or len(activation_config_hash) != 64
        or not isinstance(activation_config, dict)
    ):
        raise RuntimeError("Ingestion did not return a configuration identity")

    def _commit_source_changes() -> None:
        if transaction.promotion_moves:
            promote_pending_uploads(
                data_dir=settings.data_dir,
                transaction_id=workspace.name,
                collections=collections,
                moves=list(transaction.promotion_moves),
            )
            transaction.promotion_transaction_id = workspace.name
        if quarantine_source is not None:
            transaction.quarantined_source = quarantine_upload(
                data_dir=settings.data_dir,
                source_path=quarantine_source,
                transaction_id=workspace.name,
                collections=collections,
            )

    final = rebuild_snapshot(
        resource.index,
        pg_index,
        settings,
        SnapshotActivation(
            manager=transaction.manager,
            workspace=workspace,
            text_collection=collections["text"],
            image_collection=collections["image"],
            expected_corpus_hash=expected_corpus_hash,
            expected_config_hash=expected_config_hash,
            activation_config=activation_config,
            activation_config_hash=activation_config_hash,
            collection_metadata=_read_collection_metadata(collections),
            graph_requested=bool(use_graphrag and ingest_result.get("documents")),
        ),
        commit_source_changes=_commit_source_changes,
        log_export_event=_log_export_event,
        record_graph_export_metric=record_graph_export_metric,
    )
    transaction.committed = True
    return final


def _collection_base_name(base_name: str) -> str:
    """Return the canonical safe prefix for an owned physical collection."""
    canonical = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in base_name
    ).strip("_")
    if not canonical:
        raise ValueError("Qdrant collection base name is empty")
    return canonical


def _physical_collection_prefix(base_name: str, *, build_id_length: int) -> str:
    """Return the length-bounded prefix shared by one physical generation."""
    suffix_length = 2 + build_id_length
    available = _PHYSICAL_COLLECTION_MAX_LENGTH - suffix_length
    if available < 1:
        raise ValueError("Physical collection build identity is too long")
    return f"{_collection_base_name(base_name)[:available]}__"


def _physical_collection_names(workspace: Path) -> dict[str, str]:
    """Return unique immutable collection names owned by one snapshot workspace."""
    build_id = "".join(
        character
        for character in workspace.name.removeprefix("_tmp-")
        if character.isalnum()
    )
    if not build_id:
        raise ValueError("Snapshot workspace does not expose a valid build identity")

    def _name(base_name: str) -> str:
        prefix = _physical_collection_prefix(
            base_name,
            build_id_length=len(build_id),
        )
        return f"{prefix}{build_id}"

    collections = {
        "text": _name(settings.database.qdrant_collection),
        "image": _name(settings.database.qdrant_image_collection),
    }
    if collections["text"] == collections["image"]:
        raise ValueError("Text and image collection identities must be distinct")
    return collections


def _read_collection_metadata(collections: dict[str, str]) -> dict[str, Any]:
    """Read the exact Qdrant metadata verified by the ingestion boundary."""
    from qdrant_client import QdrantClient

    from src.utils.storage import get_client_config

    client = QdrantClient(**get_client_config())
    try:
        metadata: dict[str, Any] = {}
        for owner, collection_name in collections.items():
            info = client.get_collection(collection_name)
            value = getattr(getattr(info, "config", None), "metadata", None)
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"Staged {owner} collection has no immutable metadata"
                )
            metadata[owner] = dict(value)
        return metadata
    finally:
        with contextlib.suppress(Exception):
            client.close()


def _delete_staged_collections(collections: dict[str, str]) -> None:
    """Best-effort removal of an uncommitted physical collection generation."""
    from src.persistence.snapshot import is_snapshot_version_name, load_manifest

    storage_dir = settings.data_dir / "storage"
    try:
        retained_manifests = [
            manifest
            for candidate in storage_dir.iterdir()
            if candidate.is_dir()
            and not candidate.is_symlink()
            and is_snapshot_version_name(candidate.name)
            and (manifest := load_manifest(candidate)) is not None
        ]
    except FileNotFoundError:
        retained_manifests = []
    except Exception as exc:
        logger.warning(
            "Staged collection cleanup skipped because snapshots are unreadable "
            "(error_type={})",
            type(exc).__name__,
        )
        return

    retained_names = {
        str(value)
        for manifest in retained_manifests
        for manifest_collections in [manifest.get("collections")]
        if isinstance(manifest_collections, dict)
        for value in manifest_collections.values()
    }
    if retained_names.intersection(collections.values()):
        logger.error("Refusing to delete a collection referenced by a snapshot")
        return

    from qdrant_client import QdrantClient

    from src.utils.storage import get_client_config

    client = QdrantClient(**get_client_config())
    try:
        for collection_name in collections.values():
            try:
                if client.collection_exists(collection_name):
                    client.delete_collection(collection_name=collection_name)
            except Exception as exc:
                logger.warning(
                    "Failed to remove uncommitted Qdrant collection (error_type={})",
                    type(exc).__name__,
                )
    finally:
        with contextlib.suppress(Exception):
            client.close()


def _delete_rollback_sources(paths: tuple[Path, ...]) -> None:
    """Remove only newly persisted uploads owned by a failed ingestion job."""
    uploads_root = (settings.data_dir / "uploads").resolve()
    pending_root = (settings.data_dir / ".pending-uploads").resolve()
    for source_path in paths:
        try:
            candidate = source_path.resolve()
            if not (
                candidate.is_relative_to(uploads_root)
                or candidate.is_relative_to(pending_root)
            ):
                logger.error("Refusing to roll back a source outside owned roots")
                continue
            candidate.unlink(missing_ok=True)
            if candidate.is_relative_to(pending_root):
                with contextlib.suppress(OSError):
                    candidate.parent.rmdir()
                    pending_root.rmdir()
        except OSError as exc:
            logger.warning(
                "Failed to roll back an uncommitted upload (error_type={})",
                type(exc).__name__,
            )


def _plan_pending_inputs(
    inputs: list[IngestionInput],
) -> tuple[list[IngestionInput], tuple[tuple[Path, Path, str], ...]]:
    """Plan final upload paths without mutating the authoritative corpus."""
    pending_root = (settings.data_dir / ".pending-uploads").resolve()
    uploads_root = (settings.data_dir / "uploads").resolve()
    uploads_root.mkdir(parents=True, exist_ok=True)
    existing_ids = {
        document_id_from_sha256(sha256_file(path))
        for path in uploads_root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    planned_inputs: list[IngestionInput] = []
    planned_paths: list[Path] = []
    moves: list[tuple[Path, Path, str]] = []
    for item in inputs:
        if item.source_path is None:
            planned_inputs.append(item)
            continue
        source = Path(item.source_path)
        resolved = source.resolve(strict=True)
        if not resolved.is_relative_to(pending_root):
            planned_inputs.append(item)
            continue
        if source.is_symlink() or resolved.parent.parent != pending_root:
            raise ValueError("Pending upload path is outside its transaction")
        digest = sha256_file(resolved)
        current_document_id = document_id_from_sha256(digest)
        if current_document_id in existing_ids:
            raise ValueError("This document content already exists in the corpus")

        destination = uploads_root / resolved.name
        counter = 1
        while destination.exists() or destination in planned_paths:
            destination = uploads_root / f"{resolved.stem}-{counter}{resolved.suffix}"
            counter += 1
        moves.append((resolved, destination, digest))
        planned_paths.append(destination)
        planned_inputs.append(item)
        existing_ids.add(current_document_id)
    return planned_inputs, tuple(moves)


def _resolve_active_ingestion_job(*, owner_id: str) -> ActiveIngestionJob | None:
    """Return the current owner's queued or running corpus mutation."""
    job_id = st.session_state.get("ingest_job_id")
    if not isinstance(job_id, str) or not job_id:
        _clear_ingest_job_tracking()
        return None

    state = get_job_manager().get(job_id, owner_id=owner_id)
    if state is None:
        _clear_ingest_job_tracking()
        return None
    if state.status in ("queued", "running"):
        return job_id, state
    return None


def _resolve_corpus_mutation_activity(
    *, owner_id: str
) -> tuple[ActiveIngestionJob | None, bool, bool]:
    """Return current-session state and process-wide corpus occupancy."""
    active_job = _resolve_active_ingestion_job(owner_id=owner_id)
    manager = get_job_manager()
    process_mutation_active, activity = manager.exclusivity_activity_snapshot(
        _CORPUS_MUTATION_KEY
    )
    mutation_active = active_job is not None or process_mutation_active
    st.session_state[_CORPUS_ACTIVITY_OBSERVED_KEY] = (
        mutation_active,
        activity.maintenance_active,
    )
    return active_job, mutation_active, activity.maintenance_active


def _clear_ingest_job_tracking() -> None:
    """Clear session-local state for a job that is no longer pollable."""
    st.session_state.pop("ingest_job_id", None)
    st.session_state.pop("ingest_job_last_event", None)
    st.session_state.pop("ingest_job_cancel_requested_id", None)


def _active_ingestion_copy(active_job: ActiveIngestionJob) -> str:
    """Return concise progress copy for an active corpus mutation."""
    _, state = active_job
    last = st.session_state.get("ingest_job_last_event")
    if isinstance(last, ProgressEvent):
        return f"Corpus change in progress: {last.phase} · {last.percent}%."
    if state.status == "queued":
        return "Corpus change queued. Mutation controls are temporarily disabled."
    return "Corpus change in progress. Mutation controls are temporarily disabled."


def _corpus_mutation_copy(active_job: ActiveIngestionJob | None) -> str:
    """Return activity copy without exposing another session's job details."""
    if active_job is not None:
        return _active_ingestion_copy(active_job)
    return "Another session is changing the corpus. Mutation controls are disabled."


def _reject_active_corpus_mutation(*, owner_id: str) -> bool:
    """Warn and reject prework when any corpus mutation is already active."""
    active_job, mutation_active, maintenance_active = _resolve_corpus_mutation_activity(
        owner_id=owner_id
    )
    if maintenance_active:
        st.warning("Runtime maintenance is in progress. Try this action again shortly.")
        return True
    if not mutation_active:
        return False
    st.warning(_corpus_mutation_copy(active_job))
    return True


@st.fragment(run_every=float(settings.ui.progress_poll_interval_sec))
def _render_ingest_job_panel(*, owner_id: str) -> None:
    """Render background ingestion job progress (auto-refresh).

    Args:
        owner_id: Unique identifier for the job owner.
    """
    job_id = st.session_state.get("ingest_job_id")
    job_manager = get_job_manager()
    mutation_active, activity = job_manager.exclusivity_activity_snapshot(
        _CORPUS_MUTATION_KEY
    )
    observed = (mutation_active, activity.maintenance_active)
    previously_observed = st.session_state.get(_CORPUS_ACTIVITY_OBSERVED_KEY)
    st.session_state[_CORPUS_ACTIVITY_OBSERVED_KEY] = observed
    if (
        isinstance(previously_observed, tuple)
        and len(previously_observed) == 2
        and previously_observed != observed
    ):
        st.rerun(scope="app")
        return
    if not isinstance(job_id, str) or not job_id:
        return

    state = job_manager.get(job_id, owner_id=owner_id)
    if state is None:
        _clear_ingest_job_tracking()
        st.rerun(scope="app")
        return

    events = job_manager.drain_progress(job_id, owner_id=owner_id)
    if events:
        st.session_state["ingest_job_last_event"] = events[-1]

    last: ProgressEvent | None = st.session_state.get("ingest_job_last_event")
    pct = int(last.percent) if isinstance(last, ProgressEvent) else 0
    phase = last.phase if isinstance(last, ProgressEvent) else "ingest"
    message = last.message if isinstance(last, ProgressEvent) else ""

    st.subheader("Ingestion job")
    st.progress(max(0, min(100, pct)) / 100.0)
    st.caption(f"{phase} · {pct}%")
    if message:
        st.write(message)

    if state.status in ("queued", "running"):
        cancellation_requested = (
            st.session_state.get("ingest_job_cancel_requested_id") == job_id
        )
        if st.button(
            "Cancel ingestion",
            type="secondary",
            disabled=cancellation_requested,
        ) and job_manager.cancel(job_id, owner_id=owner_id):
            st.session_state["ingest_job_cancel_requested_id"] = job_id
            cancellation_requested = True
        if cancellation_requested:
            st.warning("Cancellation requested. Waiting for a safe stopping point.")

    else:
        # Terminal states are captured once, then presented on a full-app run.
        completed_key = "ingest_job_completed_id"
        if st.session_state.get(completed_key) == job_id:
            if job_manager.consume_terminal(job_id, owner_id=owner_id):
                _clear_ingest_job_tracking()
                st.session_state[_CORPUS_ACTIVITY_OBSERVED_KEY] = (False, False)
            else:
                logger.warning(
                    "Terminal ingestion job consumption deferred; retaining job state"
                )
        elif _capture_ingest_terminal_state(
            state,
            owner_id=owner_id,
            job_id=job_id,
            completed_key=completed_key,
        ):
            st.rerun(scope="app")


def _render_ingest_terminal_notice() -> None:
    """Render and consume one terminal outcome after a full-app rerun."""
    if _INGEST_TERMINAL_PRESENTATION_KEY not in st.session_state:
        return
    notice = st.session_state.get(_INGEST_TERMINAL_PRESENTATION_KEY)
    try:
        if not isinstance(notice, dict):
            raise TypeError("Terminal presentation must be a dictionary")
        status = notice.get("status")
        message = notice.get("message")
        if not isinstance(message, str):
            raise TypeError("Terminal presentation message must be a string")
        if status == "succeeded":
            _render_ingest_success_notice(notice, message=message)
        elif status == "failed":
            st.error(message)
        elif status == "canceled":
            st.warning(message)
        else:
            raise ValueError("Unknown terminal presentation status")
        detail = notice.get("detail")
        if isinstance(detail, str) and detail:
            st.caption(detail)
    except Exception as exc:
        _log_ingest_boundary_failure("display", exc)
        st.error(_INGEST_DISPLAY_FAILURE_MESSAGE)
    finally:
        if st.session_state.get(_INGEST_TERMINAL_PRESENTATION_KEY) is notice:
            st.session_state.pop(_INGEST_TERMINAL_PRESENTATION_KEY, None)


def _render_ingest_success_notice(notice: dict[str, Any], *, message: str) -> None:
    """Render durable success with readiness checked against live ownership."""
    presentation = notice.get("presentation")
    manifest = notice.get("manifest")
    if not isinstance(presentation, dict) or not isinstance(manifest, dict):
        raise TypeError("Success presentation must be a dictionary")
    final = _validated_snapshot_path(notice.get("snapshot_dir"))
    runtime_ready, graph_ready, snapshot_authoritative = _terminal_runtime_readiness(
        notice, final
    )
    _render_ingest_presentation(presentation)
    displayed_manifest = (
        manifest if runtime_ready else {**manifest, "graph_exports": []}
    )
    _render_manifest_details(displayed_manifest, final)
    if snapshot_authoritative:
        st.session_state["latest_manifest"] = manifest
    st.success(message)
    if runtime_ready:
        st.info("Router engine is ready for Chat.")
        if graph_ready:
            st.info("GraphRAG index is available.")
    elif snapshot_authoritative:
        st.info(
            "Snapshot completed, but the live runtime changed. Chat will reload the "
            "active snapshot."
        )
    else:
        st.info("Snapshot completed, but a newer corpus activation is current.")
    st.toast("Ingestion complete", icon="✅")


def _capture_ingest_terminal_state(
    state: Any, *, owner_id: str, job_id: str, completed_key: str
) -> bool:
    """Capture one terminal result for presentation on a full-app rerun.

    Args:
        state: Terminal job state.
        owner_id: Owner identifier authorized to consume the job.
        job_id: Unique identifier for the job.
        completed_key: Session state key to track job completion.
    """
    job_manager = get_job_manager()
    status = getattr(state, "status", None)
    result = getattr(state, "result", None)
    if status == "succeeded":
        try:
            with job_manager.admission_quiescence():
                try:
                    notice = _prepare_ingest_success_notice(result)
                except Exception as exc:
                    _log_ingest_boundary_failure("prepare", exc)
                    notice = {
                        "status": "failed",
                        "message": _INVALID_INGEST_RESULT_MESSAGE,
                    }
                _store_ingest_terminal_notice(
                    notice,
                    job_manager=job_manager,
                    owner_id=owner_id,
                    job_id=job_id,
                    completed_key=completed_key,
                )
            return True
        except JobAdmissionPausedError:
            st.info(_TERMINAL_MAINTENANCE_WAIT_COPY)
            return False
        except ForegroundRuntimeConflictError:
            st.info(_TERMINAL_FOREGROUND_WAIT_COPY)
            return False
        except JobConflictError:
            st.info(_TERMINAL_JOB_WAIT_COPY)
            return False
    elif status == "failed":
        from src.utils.log_safety import build_pii_log_entry

        error = getattr(state, "error", None)
        err_msg = str(error) if error else "unknown error"
        redaction = build_pii_log_entry(err_msg, key_id="documents.ingest_failed")
        logger.error(
            "Ingestion failed (error_type={} error={})",
            type(error).__name__ if error else "Error",
            redaction.redacted,
        )
        notice = {
            "status": "failed",
            "message": "Ingestion failed. Please try again.",
            "detail": f"Error reference: {redaction.redacted}",
        }
    elif status == "canceled":
        notice = {
            "status": "canceled",
            "message": "Ingestion cancelled.",
        }
    else:
        notice = {
            "status": "failed",
            "message": "Ingestion ended in an unexpected state. Please try again.",
        }

    _store_ingest_terminal_notice(
        notice,
        job_manager=job_manager,
        owner_id=owner_id,
        job_id=job_id,
        completed_key=completed_key,
    )
    return True


def _store_ingest_terminal_notice(
    notice: dict[str, Any],
    *,
    job_manager: JobManager,
    owner_id: str,
    job_id: str,
    completed_key: str,
) -> None:
    """Consume one terminal job after its notice is ready for a full rerun."""
    st.session_state[_INGEST_TERMINAL_PRESENTATION_KEY] = notice
    st.session_state[completed_key] = job_id
    if job_manager.consume_terminal(job_id, owner_id=owner_id):
        _clear_ingest_job_tracking()
        st.session_state[_CORPUS_ACTIVITY_OBSERVED_KEY] = (False, False)
    else:
        logger.warning(
            "Terminal ingestion job consumption deferred; retaining job state"
        )


def _prepare_ingest_success_notice(result: Any) -> dict[str, Any]:
    """Validate a worker result, transfer runtime ownership, and build its DTO."""
    from src.persistence.snapshot import latest_snapshot_dir

    resource: VectorIndexResource | None = None
    try:
        if not isinstance(result, dict):
            raise TypeError("Completed ingestion result must be a dictionary")
        ingest_result = result.get("ingest")
        if not isinstance(ingest_result, dict):
            raise TypeError("Completed ingestion payload must be a dictionary")
        candidate = ingest_result.get("vector_resource")
        resource = candidate if isinstance(candidate, VectorIndexResource) else None
        final = _validated_snapshot_path(result.get("snapshot_dir"))
        manifest = _bounded_manifest_presentation(result.get("manifest"))
        runtime_result = {**ingest_result, "snapshot_id": final.name}
        runtime_generation = _require_nonnegative_int(
            result.get("runtime_generation"),
            "runtime_generation",
        )
        use_graphrag = result.get("use_graphrag", False)
        if not isinstance(use_graphrag, bool):
            raise TypeError("GraphRAG presentation flag must be a boolean")
        authoritative = latest_snapshot_dir(Path(settings.data_dir) / "storage")
        generation_current = runtime_generation == int(settings.cache_version)
        snapshot_current = (
            authoritative is not None and authoritative.resolve() == final
        )
        if generation_current and snapshot_current:
            presentation, graph_enabled = _prepare_ingest_runtime(
                runtime_result,
                use_graphrag,
            )
            publication = "published"
        else:
            (
                resource,
                pg_index,
                _text_collection,
                _image_collection,
                _snapshot_id,
                presentation,
            ) = _validated_ingest_runtime(runtime_result, use_graphrag)
            graph_enabled = pg_index is not None
            resource.close()
            publication = "superseded"
        return {
            "status": "succeeded",
            "message": f"Snapshot created: {final.name}",
            "presentation": presentation,
            "manifest": manifest,
            "snapshot_dir": str(final),
            "runtime_identity": {
                "generation": runtime_generation,
                "snapshot_id": final.name,
                "graph_enabled": graph_enabled,
            },
            "runtime_publication": publication,
        }
    except Exception:
        if resource is not None:
            resource.close()
        raise


def _validated_snapshot_path(value: Any) -> Path:
    """Return a normalized snapshot path confined to configured storage."""
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise TypeError("Completed ingestion has an invalid snapshot path")
    root = (Path(settings.data_dir) / "storage").resolve()
    snapshot = Path(value).resolve()
    if snapshot == root or not snapshot.is_relative_to(root):
        raise ValueError("Completed ingestion snapshot escaped storage")
    return snapshot


def _terminal_runtime_readiness(
    notice: dict[str, Any], snapshot_dir: Path
) -> tuple[bool, bool, bool]:
    """Revalidate terminal runtime ownership at the moment it is rendered."""
    from src.persistence.snapshot import latest_snapshot_dir

    identity = notice.get("runtime_identity")
    publication = notice.get("runtime_publication")
    if not isinstance(identity, dict) or publication not in {
        "published",
        "superseded",
    }:
        raise TypeError("Terminal runtime identity is invalid")
    generation = _require_nonnegative_int(
        identity.get("generation"),
        "terminal runtime generation",
    )
    snapshot_id = identity.get("snapshot_id")
    graph_enabled = identity.get("graph_enabled")
    if (
        not isinstance(snapshot_id, str)
        or not snapshot_id
        or Path(snapshot_id).name != snapshot_id
        or not isinstance(graph_enabled, bool)
    ):
        raise TypeError("Terminal snapshot identity is invalid")
    authoritative = latest_snapshot_dir(Path(settings.data_dir) / "storage")
    snapshot_authoritative = (
        authoritative is not None and authoritative.resolve() == snapshot_dir
    )
    graph_ready = graph_enabled and st.session_state.get("graphrag_index") is not None
    runtime_ready = (
        publication == "published"
        and generation == int(settings.cache_version)
        and snapshot_authoritative
        and st.session_state.get("_snapshot_loaded_id") == snapshot_id
        and session_vector_resource_is_current(
            st.session_state,
            runtime_generation=generation,
        )
        and session_router_is_current(
            st.session_state,
            runtime_generation=generation,
        )
        and (not graph_enabled or graph_ready)
    )
    return runtime_ready, graph_ready, snapshot_authoritative


def _bounded_manifest_presentation(value: Any) -> dict[str, Any]:
    """Validate a finalized manifest and return only bounded display fields."""
    if not isinstance(value, dict):
        raise TypeError("Completed ingestion has no finalized manifest")
    corpus_hash = _validated_sha256(value.get("corpus_hash"), "corpus_hash")
    config_hash = _validated_sha256(value.get("config_hash"), "config_hash")
    versions = value.get("versions")
    if not isinstance(versions, dict):
        raise TypeError("Finalized manifest versions must be a dictionary")
    bounded_versions: dict[str, str] = {}
    for key in sorted(versions):
        if not isinstance(key, str):
            raise TypeError("Finalized manifest version keys must be strings")
        version = versions[key]
        if not isinstance(version, str | int | float | bool) and version is not None:
            raise TypeError("Finalized manifest version values must be scalar")
        bounded_versions[key[:_PRESENTATION_ID_MAX_LENGTH]] = str(version)[
            :_PRESENTATION_ID_MAX_LENGTH
        ]
        if len(bounded_versions) >= _MANIFEST_VERSION_PRESENTATION_LIMIT:
            break
    graph_exports = value.get("graph_exports")
    if not isinstance(graph_exports, list):
        raise TypeError("Finalized manifest graph exports must be a list")
    bounded_exports = [
        _bounded_manifest_export(item)
        for item in graph_exports[:_MANIFEST_EXPORT_PRESENTATION_LIMIT]
    ]
    return {
        "corpus_hash": corpus_hash,
        "config_hash": config_hash,
        "versions": bounded_versions,
        "graph_exports": bounded_exports,
    }


def _validated_sha256(value: Any, field: str) -> str:
    """Return one lowercase SHA-256 manifest identity."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"Finalized manifest {field} is invalid")
    return value


def _bounded_manifest_export(value: Any) -> dict[str, Any]:
    """Validate one graph-export row needed by snapshot presentation."""
    from src.persistence.snapshot import (
        MANIFEST_EXPORT_FILENAME_MAX_LENGTH,
        MANIFEST_EXPORT_FORMAT_MAX_LENGTH,
    )

    if not isinstance(value, dict):
        raise TypeError("Finalized manifest export entries must be dictionaries")
    filename = value.get("filename") or value.get("path")
    format_name = value.get("format")
    size_bytes = value.get("size_bytes")
    if (
        not isinstance(filename, str)
        or not filename
        or len(filename) > MANIFEST_EXPORT_FILENAME_MAX_LENGTH
        or Path(filename).name != filename
        or not isinstance(format_name, str)
        or not format_name
        or len(format_name) > MANIFEST_EXPORT_FORMAT_MAX_LENGTH
    ):
        raise TypeError("Finalized manifest export identity is invalid")
    return {
        "filename": filename,
        "format": format_name,
        "size_bytes": _require_nonnegative_int(size_bytes, "graph export size"),
    }


def _prepare_ingest_runtime(
    result: dict[str, Any], use_graphrag: bool
) -> tuple[dict[str, Any], bool]:
    """Validate ingestion output and transfer its live runtime resources."""
    from src.retrieval.router_factory import build_router_engine

    (
        resource,
        pg_index,
        text_collection,
        image_collection,
        snapshot_id,
        presentation,
    ) = _validated_ingest_runtime(result, use_graphrag)
    runtime_generation = settings.cache_version
    try:
        router = build_router_engine(
            resource.index,
            pg_index,
            settings,
            text_collection=text_collection,
            image_collection=image_collection,
        )
    except Exception:
        resource.close()
        raise
    state_updates: dict[str, Any] = {
        "_snapshot_collections": {
            "text": text_collection,
            "image": image_collection,
        },
    }
    state_removals: list[str] = []
    if pg_index is not None:
        state_updates["graphrag_index"] = pg_index
    else:
        state_removals.append("graphrag_index")
    if snapshot_id is not None:
        state_updates["_snapshot_loaded_id"] = snapshot_id
    else:
        state_removals.append("_snapshot_loaded_id")
    replace_session_runtime(
        st.session_state,
        resource,
        router,
        runtime_generation=runtime_generation,
        state_updates=state_updates,
        state_removals=state_removals,
    )
    return presentation, pg_index is not None


def _validated_ingest_runtime(
    result: dict[str, Any], use_graphrag: bool
) -> tuple[VectorIndexResource, Any | None, str, str, str | None, dict[str, Any]]:
    """Validate live runtime inputs and build their object-free presentation."""
    candidate = result.get("vector_resource")
    resource = candidate if isinstance(candidate, VectorIndexResource) else None
    try:
        if resource is None or resource.closed:
            raise TypeError("Completed ingestion has no live owned vector resource")
        count = _require_nonnegative_int(result.get("count"), "count")
        metadata = _bounded_nlp_metadata(result.get("metadata"))
        preview = _bounded_nlp_preview(result.get("nlp_preview"))
        exports, image_preview_summary = _bounded_image_exports(result.get("exports"))
        collections = result.get("collections")
        if not isinstance(collections, dict):
            raise TypeError("Completed ingestion has no physical collection identity")
        text_collection = collections.get("text")
        image_collection = collections.get("image")
        if (
            not isinstance(text_collection, str)
            or not text_collection
            or not isinstance(image_collection, str)
            or not image_collection
        ):
            raise TypeError("Completed ingestion has invalid collection identities")
        snapshot_id = result.get("snapshot_id")
        if snapshot_id is not None and (
            not isinstance(snapshot_id, str)
            or not snapshot_id
            or len(snapshot_id) > _PRESENTATION_ID_MAX_LENGTH
            or Path(snapshot_id).name != snapshot_id
        ):
            raise TypeError("Completed ingestion has an invalid snapshot identity")
        pg_index = result.get("pg_index") if use_graphrag else None
        presentation = {
            "count": count,
            "metadata": metadata,
            "nlp_preview": preview,
            "exports": exports,
            "image_preview_summary": image_preview_summary,
        }
        return (
            resource,
            pg_index,
            text_collection,
            image_collection,
            snapshot_id,
            presentation,
        )
    except Exception:
        if resource is not None:
            resource.close()
        raise


def _render_ingest_presentation(presentation: dict[str, Any]) -> None:
    """Render a validated, object-free ingestion presentation DTO."""
    count = _require_nonnegative_int(presentation.get("count"), "count")
    metadata = presentation.get("metadata")
    preview = presentation.get("nlp_preview")
    exports = presentation.get("exports")
    image_preview_summary = presentation.get("image_preview_summary")
    if (
        not isinstance(metadata, dict)
        or not isinstance(preview, dict)
        or not isinstance(exports, list)
        or not isinstance(image_preview_summary, dict)
    ):
        raise TypeError("Ingestion presentation has an invalid shape")
    selected_artifacts = _require_nonnegative_int(
        image_preview_summary.get("selected_artifacts"),
        "selected image artifacts",
    )
    total_artifacts = _require_nonnegative_int(
        image_preview_summary.get("total_artifacts"),
        "total image artifacts",
    )
    total_documents = _require_nonnegative_int(
        image_preview_summary.get("total_documents"),
        "total image documents",
    )
    omitted_artifacts = _require_nonnegative_int(
        image_preview_summary.get("omitted_artifacts"),
        "omitted image artifacts",
    )
    omitted_documents = _require_nonnegative_int(
        image_preview_summary.get("omitted_documents"),
        "omitted image documents",
    )
    if (
        selected_artifacts != len(exports)
        or total_artifacts != selected_artifacts + omitted_artifacts
        or omitted_documents > total_documents
    ):
        raise TypeError("Image preview summary does not match its exports")
    st.write(f"Ingested {count} documents.")
    _render_nlp_preview(preview, metadata)
    _render_image_exports(exports)
    if omitted_documents:
        st.caption(
            f"Preview omits {omitted_artifacts} images, including all images "
            f"from {omitted_documents} documents."
        )
    elif omitted_artifacts:
        st.caption(
            f"Preview omits {omitted_artifacts} images; every document remains "
            "represented."
        )


def _require_nonnegative_int(value: Any, field: str) -> int:
    """Return an exact nonnegative integer or reject the worker payload."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{field} must be a nonnegative integer")
    return value


def _bounded_nlp_metadata(value: Any) -> dict[str, Any]:
    """Return only validated NLP counters rendered by the Documents page."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("NLP metadata must be a dictionary")
    enabled = value.get("nlp.enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("NLP enabled metadata must be a boolean")
    return {
        "nlp.enabled": enabled,
        "nlp.enriched_nodes": _require_nonnegative_int(
            value.get("nlp.enriched_nodes", 0), "nlp.enriched_nodes"
        ),
        "nlp.entity_count": _require_nonnegative_int(
            value.get("nlp.entity_count", 0), "nlp.entity_count"
        ),
    }


def _bounded_nlp_preview(value: Any) -> dict[str, list[dict[str, str]]]:
    """Return capped, string-bounded NLP entities and sentences."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("NLP preview must be a dictionary")
    raw_entities = value.get("entities") or []
    raw_sentences = value.get("sentences") or []
    if not isinstance(raw_entities, list) or not isinstance(raw_sentences, list):
        raise TypeError("NLP preview collections must be lists")
    entities: list[dict[str, str]] = []
    for entity in raw_entities[:_NLP_ENTITY_PRESENTATION_LIMIT]:
        if not isinstance(entity, dict):
            continue
        label = entity.get("label")
        text = entity.get("text")
        if isinstance(label, str) and isinstance(text, str):
            entities.append(
                {
                    "label": label[:_PRESENTATION_ID_MAX_LENGTH],
                    "text": text[:_PRESENTATION_TEXT_MAX_LENGTH],
                }
            )
    sentences: list[dict[str, str]] = []
    for sentence in raw_sentences[:_NLP_SENTENCE_PRESENTATION_LIMIT]:
        if not isinstance(sentence, dict):
            continue
        text = sentence.get("text")
        if isinstance(text, str):
            sentences.append({"text": text[:_PRESENTATION_TEXT_MAX_LENGTH]})
    return {"entities": entities, "sentences": sentences}


def _bounded_image_exports(
    value: Any,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return a fair, capped image preview and its omission summary."""
    if value is None:
        value = []
    if not isinstance(value, list):
        raise TypeError("Image exports must be a list")
    exports_by_document: dict[str, list[dict[str, Any]]] = {}
    for export in value:
        if not isinstance(export, dict):
            raise TypeError("Image export entries must be dictionaries")
        content_type = export.get("content_type")
        if not isinstance(content_type, str):
            raise TypeError("Image export content type must be a string")
        if not content_type.startswith("image/"):
            continue
        metadata = export.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise TypeError("Image export metadata must be a dictionary")
        raw_document_id = metadata.get("doc_id") or metadata.get("document_id")
        bounded_metadata = _bounded_image_export_metadata(metadata)
        document_id = (
            raw_document_id
            if isinstance(raw_document_id, str) and raw_document_id
            else "-"
        )
        exports_by_document.setdefault(document_id, []).append(
            {
                "content_type": content_type[:100],
                "metadata": bounded_metadata,
            }
        )

    document_ids = sorted(exports_by_document)
    for document_id in document_ids:
        exports_by_document[document_id].sort(
            key=lambda item: (
                _page_no(item["metadata"]),
                str(item["metadata"].get("thumbnail_artifact_id", "")),
                str(item["metadata"].get("image_artifact_id", "")),
            )
        )
    offsets = dict.fromkeys(document_ids, 0)
    selected: list[dict[str, Any]] = []
    selected_documents: set[str] = set()
    while len(selected) < _IMAGE_EXPORT_PRESENTATION_LIMIT:
        added = False
        for document_id in document_ids:
            items = exports_by_document[document_id]
            offset = offsets[document_id]
            if offset >= len(items):
                continue
            selected.append(items[offset])
            selected_documents.add(document_id)
            offsets[document_id] = offset + 1
            added = True
            if len(selected) >= _IMAGE_EXPORT_PRESENTATION_LIMIT:
                break
        if not added:
            break

    total_artifacts = sum(len(items) for items in exports_by_document.values())
    return selected, {
        "selected_artifacts": len(selected),
        "total_artifacts": total_artifacts,
        "total_documents": len(document_ids),
        "omitted_artifacts": total_artifacts - len(selected),
        "omitted_documents": len(set(document_ids) - selected_documents),
    }


def _bounded_image_export_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return only bounded metadata fields consumed by image preview rendering."""
    bounded: dict[str, Any] = {}
    document_id = metadata.get("doc_id") or metadata.get("document_id")
    if document_id is not None:
        if not isinstance(document_id, str):
            raise TypeError("Image export document id must be a string")
        bounded["doc_id"] = document_id[:_PRESENTATION_ID_MAX_LENGTH]
    page = metadata.get("page_no")
    if page is None:
        page = metadata.get("page")
    if page is None:
        page = metadata.get("page_number")
    if page is not None:
        bounded["page_no"] = _require_nonnegative_int(page, "image export page")
    for prefix in ("thumbnail", "image"):
        bounded.update(_bounded_artifact_metadata(metadata, prefix))
    return bounded


def _bounded_artifact_metadata(metadata: dict[str, Any], prefix: str) -> dict[str, str]:
    """Validate one optional content-addressed artifact reference."""
    artifact_id = metadata.get(f"{prefix}_artifact_id")
    if artifact_id is None:
        return {}
    if (
        not isinstance(artifact_id, str)
        or len(artifact_id) != 64
        or any(character not in "0123456789abcdef" for character in artifact_id)
    ):
        raise TypeError("Image export artifact id is invalid")
    suffix = metadata.get(f"{prefix}_artifact_suffix") or ""
    if (
        not isinstance(suffix, str)
        or len(suffix) > 16
        or (suffix and not suffix.startswith("."))
        or any(separator in suffix for separator in ("/", "\\"))
        or any(
            character.lower() not in _ARTIFACT_SUFFIX_CHARACTERS for character in suffix
        )
    ):
        raise TypeError("Image export artifact suffix is invalid")
    return {
        f"{prefix}_artifact_id": artifact_id,
        f"{prefix}_artifact_suffix": suffix,
    }


def _log_ingest_boundary_failure(stage: str, exc: Exception) -> None:
    """Log a redacted ingestion-boundary diagnostic without exposing payloads."""
    from src.utils.log_safety import build_pii_log_entry

    redaction = build_pii_log_entry(str(exc), key_id=f"documents.ingest_{stage}")
    logger.error(
        "Ingestion terminal boundary failed (stage={} error_type={} error={})",
        stage,
        type(exc).__name__,
        redaction.redacted,
    )


def _render_nlp_preview(preview: dict[str, Any], meta: dict[str, Any]) -> None:
    """Render a visual preview of NLP enrichment results.

    Args:
        preview: NLP preview dictionary containing entities and sentences.
        meta: Document metadata dictionary.
    """
    enabled = bool(meta.get("nlp.enabled", False))
    if not enabled:
        st.caption("NLP enrichment: disabled")
        return
    enriched = int(meta.get("nlp.enriched_nodes", 0) or 0)
    entity_count = int(meta.get("nlp.entity_count", 0) or 0)
    with st.expander("NLP enrichment (spaCy)", expanded=False):
        st.write(f"Enriched nodes: {enriched}")
        st.write(f"Entities extracted: {entity_count}")

        ents = preview.get("entities")
        if isinstance(ents, list) and ents:
            st.subheader("Sample entities")
            for ent in ents[:20]:
                if not isinstance(ent, dict):
                    continue
                label = ent.get("label")
                text = ent.get("text")
                if isinstance(label, str) and isinstance(text, str):
                    st.write(f"- {label}: {text}")

        sents = preview.get("sentences")
        if isinstance(sents, list) and sents:
            st.subheader("Sample sentences")
            for sent in sents[:10]:
                if not isinstance(sent, dict):
                    continue
                text = sent.get("text")
                if isinstance(text, str) and text.strip():
                    st.write(f"- {text}")


def _render_image_exports(exports: list[dict[str, Any]]) -> None:
    """Render a lightweight preview of PDF page-image exports."""
    images = _filter_image_exports(exports)
    if not images:
        return
    by_doc = _group_exports_by_doc(images)
    with st.expander("Page images (preview)", expanded=False):
        st.caption(
            "Previews are loaded from local artifacts (no base64 stored in Qdrant)."
        )
        for doc_id, items in sorted(by_doc.items(), key=lambda t: t[0]):
            st.subheader(f"Document: {doc_id}")
            ordered = sorted(items, key=lambda x: _page_no(x.get("metadata") or {}))
            _render_export_images(ordered, _IMAGE_EXPORT_PRESENTATION_LIMIT)


def _filter_image_exports(exports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter export entries down to image artifacts.

    Extracts all entries with image/* content types (e.g., image/png, image/jpeg).
    Skips non-dict entries and non-image content types.

    Args:
        exports: List of export artifact dicts, each with optional 'content_type'.

    Returns:
        Filtered list containing only image artifacts.
    """
    images: list[dict[str, Any]] = []
    for e in exports:
        if not isinstance(e, dict):
            continue
        ct = str(e.get("content_type") or "")
        if ct.startswith("image/"):
            images.append(e)
    return images


def _doc_id(meta: dict[str, Any]) -> str:
    """Extract a display document id from export metadata.

    Checks for 'doc_id' or 'document_id' fields in the metadata.
    Falls back to '-' if neither field is present.

    Args:
        meta: Export metadata dict.

    Returns:
        Document ID string (e.g., 'doc-a1b2c3d4'), or '-' if not found.
    """
    return str(meta.get("doc_id") or meta.get("document_id") or "-")


def _page_no(meta: dict[str, Any]) -> int:
    """Parse the page number from export metadata.

    Checks multiple field names (page_no, page, page_number) to accommodate
    different export sources. Returns 0 if parsing fails or field is missing.

    Args:
        meta: Export metadata dict.

    Returns:
        Page number as int (1-indexed), or 0 if not found or invalid.
    """
    raw = meta.get("page_no") or meta.get("page") or meta.get("page_number")
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _group_exports_by_doc(
    images: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group image exports by document id for efficient rendering.

    Organizes exports by document, extracting metadata and using _doc_id
    to determine the grouping key. Non-dict metadata is treated as empty dict.

    Args:
        images: List of image export dicts, each with optional 'metadata'.

    Returns:
        Dict mapping document IDs to lists of export artifacts for that document.
    """
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for e in images:
        meta = e.get("metadata")
        meta = meta if isinstance(meta, dict) else {}
        by_doc.setdefault(_doc_id(meta), []).append(e)
    return by_doc


def _render_export_images(items: list[dict[str, Any]], preview_limit: int) -> None:
    """Render page image thumbnails for a document in a 4-column grid.

    Attempts to load and display thumbnail or full images from the artifact store.
    Handles encrypted images when the encryption utility is available.
    Limits display to preview_limit to avoid UI slowdown.

    Args:
        items: List of image export dicts, each with artifact metadata.
        preview_limit: Max number of images to render per document.
    """
    store = ArtifactStore.from_settings(settings)
    cols = st.columns(4)
    for i, e in enumerate(items[: int(preview_limit)]):
        meta = e.get("metadata")
        meta = meta if isinstance(meta, dict) else {}
        page_no = _page_no(meta)
        thumb_id = meta.get("thumbnail_artifact_id")
        thumb_sfx = meta.get("thumbnail_artifact_suffix") or ""
        img_id = meta.get("image_artifact_id")
        img_sfx = meta.get("image_artifact_suffix") or ""

        ref = None
        if thumb_id:
            ref = ArtifactRef(sha256=str(thumb_id), suffix=str(thumb_sfx))
        elif img_id:
            ref = ArtifactRef(sha256=str(img_id), suffix=str(img_sfx))

        col = cols[i % 4]
        with col:
            if ref is None:
                st.caption(f"p{page_no or '?'} (no artifact ref)")
                continue
            render_artifact_image(
                ref,
                store=store,
                caption=f"p{page_no}",
                use_container_width=True,
                missing_caption=f"p{page_no or '?'} (missing)",
                encrypted_caption="Encryption support unavailable.",
            )


def _render_maintenance_controls(
    *,
    owner_id: str,
    active_job: ActiveIngestionJob | None,
    mutation_active: bool,
    maintenance_active: bool,
) -> None:
    """Render generation-safe corpus rebuild and deletion controls."""
    uploads_dir = settings.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    with st.expander("Maintenance", expanded=False):
        st.caption(
            "Every maintenance action builds and verifies a complete isolated search "
            "generation before it becomes active."
        )
        if maintenance_active:
            st.info(
                "Runtime maintenance is in progress. Corpus actions are temporarily "
                "disabled."
            )
        elif mutation_active:
            st.info(_corpus_mutation_copy(active_job))

        st.subheader("Rebuild search index")
        st.caption(
            "Reparse the complete corpus and rebuild text, image, and optional graph "
            "indexes as one generation. The active generation stays available until "
            "the replacement passes verification."
        )
        cols = st.columns(2)
        encrypt = cols[0].checkbox(
            "Encrypt images",
            value=bool(settings.processing.encrypt_page_images),
            key="rebuild_encrypt_images",
        )
        if cols[1].button(
            "Rebuild",
            use_container_width=True,
            disabled=mutation_active or maintenance_active,
        ) and _start_existing_corpus_rebuild(
            uploads_dir=uploads_dir,
            encrypt=bool(encrypt),
            owner_id=owner_id,
        ):
            st.rerun()

        st.subheader("Delete uploaded document")
        files = sorted(
            [p for p in uploads_dir.iterdir() if p.is_file()],
            key=lambda p: p.name,
        )
        if not files:
            st.caption("No uploaded files.")
            return

        selection = st.selectbox(
            "Select file",
            options=[p.name for p in files],
            index=0,
            key="delete_upload_select",
        )
        st.caption(
            "Content-addressed image artifacts are retained so historical chat "
            "evidence remains renderable."
        )
        confirm = st.checkbox(
            f'I understand deleting "{selection}" cannot be undone',
            value=False,
            key=f"delete_upload_confirm::{selection}",
        )
        if st.button(
            f'Delete "{selection}"',
            disabled=not confirm or mutation_active or maintenance_active,
            type="secondary",
        ):
            target = uploads_dir / str(selection)
            if _start_upload_deletion(
                target=target,
                encrypt=bool(encrypt),
                owner_id=owner_id,
            ):
                st.rerun()


def _doc_id_for_upload(path: Path) -> str:
    """Return canonical document id for an uploaded file."""
    digest = sha256_file(path)
    return document_id_from_sha256(digest)


def _existing_corpus_inputs(
    uploads_dir: Path,
    *,
    encrypt: bool,
    excluded_path: Path | None = None,
) -> list[IngestionInput]:
    """Build canonical inputs for every direct, regular upload except one target."""
    excluded = excluded_path.resolve() if excluded_path is not None else None
    inputs: list[IngestionInput] = []
    for path in sorted(uploads_dir.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            continue
        if excluded is not None and path.resolve() == excluded:
            continue
        inputs.append(
            IngestionInput(
                document_id=_doc_id_for_upload(path),
                source_path=path,
                encrypt_images=encrypt,
            )
        )
    require_unique_document_ids(inputs)
    return inputs


def _start_existing_corpus_rebuild(
    *,
    uploads_dir: Path,
    encrypt: bool,
    owner_id: str,
) -> bool:
    """Start a complete generation rebuild without owning existing sources."""
    if _reject_active_corpus_mutation(owner_id=owner_id):
        return False
    inputs = _existing_corpus_inputs(uploads_dir, encrypt=encrypt)
    if not inputs:
        st.info("Add a document before rebuilding the search index.")
        return False
    return _start_ingestion_job(
        inputs,
        use_graphrag=bool(settings.graphrag_cfg.enabled),
        encrypt_images=encrypt,
        nlp_service=_load_optional_spacy_service(),
        owner_id=owner_id,
    )


def _start_upload_deletion(
    *,
    target: Path,
    encrypt: bool,
    owner_id: str,
) -> bool:
    """Schedule source removal as a full staged corpus generation."""
    if _reject_active_corpus_mutation(owner_id=owner_id):
        return False
    uploads_dir = (settings.data_dir / "uploads").resolve()
    if target.is_symlink() or target.parent.resolve() != uploads_dir:
        st.error("Refusing to delete a path outside the uploads directory.")
        return False
    if not target.is_file():
        st.warning("File not found.")
        return False
    inputs = _existing_corpus_inputs(
        uploads_dir,
        encrypt=encrypt,
        excluded_path=target,
    )
    started = _start_ingestion_job(
        inputs,
        use_graphrag=bool(settings.graphrag_cfg.enabled),
        encrypt_images=encrypt,
        nlp_service=_load_optional_spacy_service(),
        owner_id=owner_id,
        excluded_source_paths=(target,),
        quarantine_source=target,
    )
    if started:
        st.info(
            "Deletion scheduled. The active corpus remains unchanged until "
            "verification."
        )
    return started


def _render_latest_snapshot_summary() -> None:
    """Show a summary of the latest snapshot manifest when available."""
    from src.persistence.snapshot import load_manifest

    base_dir = settings.data_dir / "storage"
    with contextlib.suppress(Exception):
        manifest = load_manifest(base_dir=base_dir)
        if manifest:
            created_at = manifest.get("created_at", "unknown")
            corpus_hash = manifest.get("corpus_hash", "-")
            config_hash = manifest.get("config_hash", "-")
            st.caption(
                f"Latest snapshot: created {created_at} | corpus={corpus_hash[:12]} "
                f"config={config_hash[:12]}"
            )


def _render_manifest_details(
    manifest: dict[str, Any] | None, snapshot_dir: Path
) -> None:
    """Display manifest metadata after a rebuild."""
    if not manifest:
        return
    st.markdown("### Snapshot Metadata")
    cols = st.columns(3)
    cols[0].metric("Snapshot", snapshot_dir.name)
    cols[1].metric("Corpus Hash", manifest.get("corpus_hash", "-")[:16])
    cols[2].metric("Config Hash", manifest.get("config_hash", "-")[:16])
    versions = manifest.get("versions", {})
    if versions:
        st.json({"versions": versions}, expanded=False)
    exports = manifest.get("graph_exports") or []
    if exports:
        st.write("Packaged graph exports:")
        for item in exports:
            rel_path = item.get("filename") or item.get("path")
            format_name = item.get("format")
            size_bytes = item.get("size_bytes", 0)
            st.caption(f"• {rel_path} ({format_name}, {size_bytes} bytes)")


def _render_export_controls() -> None:
    """Render manual GraphRAG export controls when a graph index exists."""
    if st.session_state.get("graphrag_index") is None:
        return

    st.subheader("GraphRAG Exports")
    out_dir = settings.data_dir / "exports" / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export JSONL"):
            _handle_manual_export(out_dir, "jsonl")
    with col2:
        if st.button("Export Parquet"):
            _handle_manual_export(out_dir, "parquet")


def _handle_manual_export(out_dir: Path, extension: str) -> None:
    """Handle manual graph export actions."""
    from src.persistence.snapshot_utils import timestamped_export_path
    from src.retrieval.graph_config import (
        export_graph_jsonl,
        export_graph_parquet,
        get_export_seed_ids,
    )

    try:
        with get_job_manager().foreground_runtime_activity():
            if not session_vector_resource_is_current(
                st.session_state,
                runtime_generation=int(settings.cache_version),
            ):
                st.warning("Graph export deferred because the runtime changed. Retry.")
                return
            pg_index = st.session_state["graphrag_index"]
            vector_index = st.session_state.get("vector_index")
            cap = int(getattr(settings.graphrag_cfg, "export_seed_cap", 32))
            seeds = get_export_seed_ids(pg_index, vector_index, cap=cap)
            out = timestamped_export_path(
                out_dir,
                extension,
                prefix="graph_export-manual",
            )
            start = time.perf_counter()
            if extension == "jsonl":
                export_graph_jsonl(
                    property_graph_index=pg_index,
                    output_path=out,
                    seed_node_ids=seeds,
                )
            else:
                export_graph_parquet(
                    property_graph_index=pg_index,
                    output_path=out,
                    seed_node_ids=seeds,
                )
            duration_ms = (time.perf_counter() - start) * 1000.0
        st.success(f"Exported {extension.upper()} to {out}")
        _log_export_event(
            {
                "export_performed": True,
                "export_type": f"graph_{extension}",
                "seed_count": len(seeds),
                "capped": len(seeds) >= cap,
                "dest_path": str(out),
                "context": "manual",
            }
        )
        size_bytes = out.stat().st_size if out.exists() else None
        record_graph_export_metric(
            f"graph_{extension}",
            duration_ms=duration_ms,
            seed_count=len(seeds),
            size_bytes=size_bytes,
            context="manual",
        )
    except JobAdmissionPausedError:
        st.warning("Graph export is unavailable during runtime maintenance.")
    except Exception as exc:  # pragma: no cover - UX best effort
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="documents.graph_export")
        logger.warning(
            "Graph export failed (ext={} error_type={} error={})",
            str(extension),
            type(exc).__name__,
            redaction.redacted,
        )
        st.warning(f"{extension.upper()} export failed ({type(exc).__name__}).")
        st.caption(f"Error reference: {redaction.redacted}")


# ---- Page helpers ----


def _log_export_event(payload: dict[str, Any]) -> None:
    """Emit a telemetry event for graph exports (best effort)."""
    event = {**payload, "timestamp": datetime.now(UTC).isoformat()}
    dest_path = event.get("dest_path")
    if dest_path:
        # Never persist filesystem paths (absolute or relative) in telemetry.
        # Keep only a safe filename for operator diagnostics.
        with contextlib.suppress(Exception):
            event["dest_basename"] = Path(str(dest_path)).name
        event.pop("dest_path", None)
        event.pop("dest_relpath", None)
    with contextlib.suppress(Exception):
        from src.utils.telemetry import log_jsonl

        log_jsonl(event)


if __name__ == "__main__":  # pragma: no cover
    main()
