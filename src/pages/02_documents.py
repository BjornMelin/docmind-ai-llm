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
from src.persistence.snapshot import (
    SnapshotManager,
    is_snapshot_version_name,
    load_manifest,
)
from src.persistence.snapshot_service import SnapshotActivation, rebuild_snapshot
from src.persistence.snapshot_utils import timestamped_export_path
from src.persistence.upload_journal import (
    promote_pending_uploads,
    quarantine_upload,
    restore_quarantined_upload,
    rollback_upload_promotion,
)
from src.processing.ingestion_api import require_unique_document_ids
from src.processing.parsing.health import parser_health
from src.retrieval.graph_config import (
    export_graph_jsonl,
    export_graph_parquet,
    get_export_seed_ids,
)
from src.retrieval.router_factory import build_router_engine
from src.telemetry.opentelemetry import (
    configure_observability,
    record_graph_export_metric,
)
from src.ui.artifacts import render_artifact_image
from src.ui.background_jobs import (
    JobCanceledError,
    ProgressEvent,
    get_job_manager,
    get_or_create_owner_id,
)
from src.ui.ingest_adapter import ingest_inputs, save_uploaded_file
from src.ui.router_session import replace_session_router
from src.ui.vector_session import (
    VectorIndexResource,
    clear_stale_session_vector_resource,
    replace_session_vector_resource,
)
from src.utils.hashing import document_id_from_sha256, sha256_file

ProgressReporter = Callable[[ProgressEvent], None]
_PHYSICAL_COLLECTION_MAX_LENGTH = 200


if TYPE_CHECKING:
    from src.nlp.spacy_service import SpacyNlpService


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
    clear_stale_session_vector_resource(
        st.session_state,
        runtime_generation=settings.cache_version,
    )
    st.title("Documents")

    owner_id = get_or_create_owner_id()
    _render_latest_snapshot_summary()

    files, use_graphrag, encrypt_images, parsing_overrides, submitted = (
        _render_ingest_form()
    )
    if submitted:
        _handle_ingest_submission(
            files,
            use_graphrag,
            encrypt_images,
            parsing_overrides,
            owner_id=owner_id,
        )
    _render_ingest_job_panel(owner_id=owner_id)

    _render_maintenance_controls(owner_id=owner_id)

    # Snapshot utilities and manual exports are driven by session_state indices.
    _render_export_controls()


def _render_ingest_form() -> tuple[
    list[Any] | None, bool, bool, ParsingOverrides, bool
]:
    """Render the ingestion form and return submitted values."""
    # Keep dependent parser controls outside the form so changing the global
    # defaults toggle reruns immediately and enables overrides before submit.
    parsing_overrides = _render_parsing_overrides()
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
        submitted = st.form_submit_button("Ingest")
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
) -> None:
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
        return
    if not _pdf_uploads_are_ready(files):
        return
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
            return
        _start_ingestion_job(
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
) -> None:
    """Start one background ingestion job and persist its UI state."""
    try:
        require_unique_document_ids(saved_inputs)
        job_manager = get_job_manager(settings.cache_version)

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
            )

        job_id = job_manager.start_job(owner_id=owner_id, fn=_work)
    except Exception:
        _delete_rollback_sources(rollback_source_paths)
        raise
    st.session_state["ingest_job_id"] = job_id
    st.session_state["ingest_job_use_graphrag"] = bool(use_graphrag)
    st.session_state["ingest_job_encrypt_images"] = bool(encrypt_images)
    st.toast("Ingestion started", icon="⏳")


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

    Returns:
        dict[str, Any]: Mapping of document IDs to their respective results.

    Raises:
        JobCanceledError: If cancellation is requested.
    """
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
        final = _activate_ingest_generation(
            transaction,
            ingest_result,
            use_graphrag=use_graphrag,
            quarantine_source=quarantine_source,
        )
        _emit_ingest_progress(report_progress, 100, "done", "Done")

        ingest_result["snapshot_id"] = final.name
        payload = {
            "ingest": ingest_result,
            "snapshot_dir": str(final),
            "use_graphrag": bool(use_graphrag),
        }
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
) -> Path:
    """Validate and atomically activate one completed ingestion generation."""
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


@st.fragment(run_every=float(settings.ui.progress_poll_interval_sec))
def _render_ingest_job_panel(*, owner_id: str) -> None:
    """Render background ingestion job progress (auto-refresh).

    Args:
        owner_id: Unique identifier for the job owner.
    """
    job_id = st.session_state.get("ingest_job_id")
    if not isinstance(job_id, str) or not job_id:
        return

    job_manager = get_job_manager(settings.cache_version)
    state = job_manager.get(job_id, owner_id=owner_id)
    if state is None:
        st.session_state.pop("ingest_job_id", None)
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
        # Terminal states: render result once, then clear job id.
        completed_key = "ingest_job_completed_id"
        if st.session_state.get(completed_key) != job_id:
            _render_ingest_terminal_state(
                state, job_id=job_id, completed_key=completed_key
            )


def _render_ingest_terminal_state(
    state: Any, *, job_id: str, completed_key: str
) -> None:
    """Render the results of a terminal ingestion job and clear session state.

    Args:
        state: Terminal job state.
        job_id: Unique identifier for the job.
        completed_key: Session state key to track job completion.
    """
    st.session_state.pop("ingest_job_cancel_requested_id", None)
    if state.status == "succeeded" and isinstance(state.result, dict):
        st.session_state[completed_key] = job_id
        st.session_state.pop("ingest_job_id", None)
        payload = state.result
        ingest_result = payload.get("ingest")
        snapshot_dir = payload.get("snapshot_dir")
        use_graphrag = bool(payload.get("use_graphrag", False))

        if isinstance(ingest_result, dict):
            _render_ingest_results(ingest_result, use_graphrag)
        if isinstance(snapshot_dir, str) and snapshot_dir:
            final = Path(snapshot_dir)
            manifest = load_manifest(final)
            _render_manifest_details(manifest, final)
            st.session_state["latest_manifest"] = manifest
            st.success(f"Snapshot created: {final.name}")
        st.toast("Ingestion complete", icon="✅")

    elif state.status == "failed":
        st.session_state[completed_key] = job_id
        st.session_state.pop("ingest_job_id", None)
        from src.utils.log_safety import build_pii_log_entry

        err_msg = str(state.error) if state.error else "unknown error"
        redaction = build_pii_log_entry(err_msg, key_id="documents.ingest_failed")
        logger.error(
            "Ingestion failed (error_type={} error={})",
            type(state.error).__name__ if state.error else "Error",
            redaction.redacted,
        )
        st.error("Ingestion failed. Please try again.")
        st.caption(f"Error reference: {redaction.redacted}")

    elif state.status == "canceled":
        st.session_state[completed_key] = job_id
        st.session_state.pop("ingest_job_id", None)
        st.warning("Ingestion cancelled.")


def _render_ingest_results(result: dict[str, Any], use_graphrag: bool) -> None:
    """Render ingestion results and hydrate session state indices.

    Args:
        result: Dictionary of document ingestion results.
        use_graphrag: Whether GraphRAG was enabled for this run.
    """
    count = int(result.get("count", 0))
    st.write(f"Ingested {count} documents.")
    _render_nlp_preview(result.get("nlp_preview") or {}, result.get("metadata") or {})
    _render_image_exports(result.get("exports") or [])

    resource = result.get("vector_resource")
    if not isinstance(resource, VectorIndexResource):
        raise RuntimeError("Completed ingestion has no owned vector resource")
    pg_index = result.get("pg_index") if use_graphrag else None
    vector_index = resource.index
    collections = result.get("collections")
    if not isinstance(collections, dict):
        resource.close()
        raise RuntimeError("Completed ingestion has no physical collection identity")
    text_collection = collections.get("text")
    image_collection = collections.get("image")
    if not isinstance(text_collection, str) or not isinstance(image_collection, str):
        resource.close()
        raise RuntimeError("Completed ingestion has invalid collection identities")

    replace_session_vector_resource(
        st.session_state,
        resource,
        runtime_generation=settings.cache_version,
    )
    if pg_index is not None:
        st.session_state["graphrag_index"] = pg_index
    else:
        st.session_state.pop("graphrag_index", None)

    try:
        router = build_router_engine(
            vector_index,
            pg_index,
            settings,
            text_collection=text_collection,
            image_collection=image_collection,
        )
    except Exception:
        replace_session_vector_resource(
            st.session_state,
            None,
            runtime_generation=settings.cache_version,
        )
        raise
    replace_session_router(
        st.session_state,
        router,
        runtime_generation=settings.cache_version,
    )
    st.session_state["_snapshot_collections"] = dict(collections)
    snapshot_id = result.get("snapshot_id")
    if isinstance(snapshot_id, str) and snapshot_id:
        st.session_state["_snapshot_loaded_id"] = snapshot_id
    st.info("Router engine is ready for Chat.")
    if pg_index is not None:
        st.info("GraphRAG index is available.")


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
        preview_limit = _preview_limit(by_doc)
        for doc_id, items in sorted(by_doc.items(), key=lambda t: t[0]):
            st.subheader(f"Document: {doc_id}")
            ordered = sorted(items, key=lambda x: _page_no(x.get("metadata") or {}))
            _render_export_images(ordered, preview_limit)


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


def _preview_limit(by_doc: dict[str, list[dict[str, Any]]]) -> int:
    """Render slider widget for preview limit and return selected value.

    Displays an interactive slider to control how many images are shown per document.
    Max value is capped at 128 to prevent UI slowdowns with large exports.
    Defaults to 32 images when no documents present.

    Args:
        by_doc: Dict mapping document IDs to lists of exports.

    Returns:
        Selected preview limit as int, or default 32 if no exports.
    """
    max_preview = min(128, max((len(items) for items in by_doc.values()), default=0))
    if not max_preview:
        return 32
    return int(
        st.slider(
            "Preview per document",
            min_value=1,
            max_value=max_preview,
            value=min(32, max_preview),
            key="doc_preview_limit",
        )
    )


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


def _render_maintenance_controls(*, owner_id: str) -> None:
    """Render generation-safe corpus rebuild and deletion controls."""
    uploads_dir = settings.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    with st.expander("Maintenance", expanded=False):
        st.caption(
            "Every maintenance action builds and verifies a complete isolated search "
            "generation before it becomes active."
        )

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
        if cols[1].button("Rebuild", use_container_width=True):
            _start_existing_corpus_rebuild(
                uploads_dir=uploads_dir,
                encrypt=bool(encrypt),
                owner_id=owner_id,
            )

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
            "I understand this cannot be undone",
            value=False,
            key="delete_upload_confirm",
        )
        if st.button("Delete", disabled=not confirm, type="secondary"):
            target = uploads_dir / str(selection)
            _start_upload_deletion(
                target=target,
                encrypt=bool(encrypt),
                owner_id=owner_id,
            )


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
) -> None:
    """Start a complete generation rebuild without owning existing sources."""
    inputs = _existing_corpus_inputs(uploads_dir, encrypt=encrypt)
    if not inputs:
        st.info("Add a document before rebuilding the search index.")
        return
    _start_ingestion_job(
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
) -> None:
    """Schedule source removal as a full staged corpus generation."""
    uploads_dir = (settings.data_dir / "uploads").resolve()
    if target.is_symlink() or target.parent.resolve() != uploads_dir:
        st.error("Refusing to delete a path outside the uploads directory.")
        return
    if not target.is_file():
        st.warning("File not found.")
        return
    inputs = _existing_corpus_inputs(
        uploads_dir,
        encrypt=encrypt,
        excluded_path=target,
    )
    _start_ingestion_job(
        inputs,
        use_graphrag=bool(settings.graphrag_cfg.enabled),
        encrypt_images=encrypt,
        nlp_service=_load_optional_spacy_service(),
        owner_id=owner_id,
        excluded_source_paths=(target,),
        quarantine_source=target,
    )
    st.info(
        "Deletion scheduled. The active corpus remains unchanged until verification."
    )


def _render_latest_snapshot_summary() -> None:
    """Show a summary of the latest snapshot manifest when available."""
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
    try:
        pg_index = st.session_state["graphrag_index"]
        vector_index = st.session_state.get("vector_index")
        cap = int(getattr(settings.graphrag_cfg, "export_seed_cap", 32))
        seeds = get_export_seed_ids(pg_index, vector_index, cap=cap)
        out = timestamped_export_path(out_dir, extension, prefix="graph_export-manual")
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
