"""Streamlit Documents page wiring the ingestion pipeline and snapshots."""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st
from loguru import logger

from src.config import settings
from src.models.processing import IngestionInput
from src.persistence.artifacts import ArtifactRef, ArtifactStore
from src.persistence.snapshot import SnapshotLockTimeout, load_manifest
from src.persistence.snapshot_utils import timestamped_export_path
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
from src.utils.storage import create_vector_store

ProgressReporter = Callable[[ProgressEvent], None]


if TYPE_CHECKING:
    from src.nlp.spacy_service import SpacyNlpService


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
    st.title("Documents")

    owner_id = get_or_create_owner_id()
    _render_latest_snapshot_summary()

    files, use_graphrag, encrypt_images, submitted = _render_ingest_form()
    if submitted:
        _handle_ingest_submission(
            files, use_graphrag, encrypt_images, owner_id=owner_id
        )
    _render_ingest_job_panel(owner_id=owner_id)

    _render_maintenance_controls()

    # Snapshot utilities and manual exports are driven by session_state indices.
    _render_export_controls()


def _render_ingest_form() -> tuple[list[Any] | None, bool, bool, bool]:
    """Render the ingestion form and return submitted values."""
    with st.form("ingest_form", clear_on_submit=False):
        files = st.file_uploader("Add files", type=None, accept_multiple_files=True)
        col_opts = st.columns(2)
        with col_opts[0]:
            use_graphrag = st.checkbox(
                "Build GraphRAG (beta)",
                value=bool(
                    settings.is_graphrag_enabled()
                    if hasattr(settings, "is_graphrag_enabled")
                    else getattr(settings, "enable_graphrag", True)
                ),
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
        submitted = st.form_submit_button("Ingest")
    return files, use_graphrag, encrypt_images, submitted


def _handle_ingest_submission(
    files: list[Any] | None,
    use_graphrag: bool,
    encrypt_images: bool,
    *,
    owner_id: str,
) -> None:
    """Handle ingestion submission by starting a background job.

    Args:
        files: List of uploaded file objects from Streamlit.
        use_graphrag: Whether to trigger GraphRAG snapshot rebuild.
        encrypt_images: Whether to encrypt extracted images.
        owner_id: Unique identifier for the job owner.
    """
    if not files:
        st.warning("No files selected.")
        return
    try:
        nlp_service = None
        try:
            cfg_dump = settings.spacy.model_dump()  # type: ignore[attr-defined]
            nlp_service = _get_spacy_service(settings.cache_version, cfg_dump)
        except Exception as exc:
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(str(exc), key_id="documents.spacy_init")
            logger.warning(
                "Failed to initialize SpacyNlpService (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            nlp_service = None

        saved_inputs: list[IngestionInput] = []
        for file_obj in files:
            stored_path, digest = save_uploaded_file(file_obj)
            saved_inputs.append(
                IngestionInput(
                    document_id=f"doc-{digest[:16]}",
                    source_path=stored_path,
                    metadata={
                        "source_filename": getattr(file_obj, "name", stored_path.name),
                        "uploaded_at": datetime.now(UTC).isoformat(),
                        "sha256": digest,
                    },
                    encrypt_images=bool(encrypt_images),
                )
            )

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
            )

        job_id = job_manager.start_job(owner_id=owner_id, fn=_work)
        st.session_state["ingest_job_id"] = job_id
        st.session_state["ingest_job_use_graphrag"] = bool(use_graphrag)
        st.session_state["ingest_job_encrypt_images"] = bool(encrypt_images)
        st.toast("Ingestion started", icon="⏳")
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


def _run_ingest_job(
    inputs: list[IngestionInput],
    *,
    use_graphrag: bool,
    encrypt_images: bool,
    nlp_service: SpacyNlpService | None,
    cancel_event: threading.Event,
    report_progress: ProgressReporter,
) -> dict[str, Any]:
    """Worker entrypoint for ingestion + snapshot rebuild (no Streamlit APIs).

    Args:
        inputs: List of ingestion input objects.
        use_graphrag: Whether to perform GraphRAG indexing.
        encrypt_images: Whether to encrypt image artifacts.
        nlp_service: Optional NLP service for entity enrichment.
        cancel_event: Event to monitor for cooperative cancellation.
        report_progress: Function to emit progress events.

    Returns:
        dict[str, Any]: Mapping of document IDs to their respective results.

    Raises:
        JobCanceledError: If cancellation is requested.
    """

    def _emit(percent: int, phase: str, message: str) -> None:
        evt = ProgressEvent(
            percent=max(0, min(100, int(percent))),
            phase=phase,  # type: ignore[arg-type]
            message=str(message)[:200],
            timestamp=datetime.now(UTC),
        )
        with contextlib.suppress(Exception):
            report_progress(evt)

    _emit(0, "save", f"Prepared {len(inputs)} file(s)")
    if cancel_event.is_set():
        raise JobCanceledError()

    _emit(10, "ingest", "Ingesting documents")
    ingest_result = ingest_inputs(
        inputs,
        enable_graphrag=use_graphrag,
        encrypt_images=encrypt_images,
        nlp_service=nlp_service,
    )

    _emit(70, "index", "Finalizing indices")
    if cancel_event.is_set():
        raise JobCanceledError()

    vector_index = ingest_result.get("vector_index") or _create_vector_index_fallback()
    if vector_index is None:
        raise RuntimeError("Vector index unavailable after ingestion")
    pg_index = ingest_result.get("pg_index") if use_graphrag else None

    _emit(85, "snapshot", "Rebuilding snapshot")
    final = rebuild_snapshot(vector_index, pg_index, settings)
    _emit(100, "done", "Done")

    return {
        "ingest": ingest_result,
        "snapshot_dir": str(final),
        "use_graphrag": bool(use_graphrag),
    }


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
        if st.button("Cancel ingestion", type="secondary"):
            job_manager.cancel(job_id, owner_id=owner_id)
            st.session_state.pop("ingest_job_id", None)
            st.warning("Ingestion cancelled.")

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
        st.error(f"Ingestion failed: {state.error or 'unknown error'}")

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

    vector_index = result.get("vector_index")
    pg_index = result.get("pg_index") if use_graphrag else None
    if vector_index is None:
        vector_index = _create_vector_index_fallback()

    if vector_index is not None:
        st.session_state["vector_index"] = vector_index
        _set_multimodal_retriever()
    if pg_index is not None:
        st.session_state["graphrag_index"] = pg_index
    elif not use_graphrag:
        st.session_state.pop("graphrag_index", None)

    router = None
    if vector_index is not None:
        router = build_router_engine(vector_index, pg_index, settings)
        st.session_state["router_engine"] = router
    if router is not None:
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


def _set_multimodal_retriever() -> None:
    """Initialize a multimodal retriever for downstream tools if available."""
    try:
        from src.retrieval.multimodal_fusion import MultimodalFusionRetriever

        st.session_state["hybrid_retriever"] = MultimodalFusionRetriever()
    except Exception as exc:  # pragma: no cover - UI best-effort
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="documents.multimodal_init")
        logger.debug(
            "Multimodal retriever init failed (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        st.session_state.pop("hybrid_retriever", None)


def _handle_snapshot_rebuild(vector_index: Any, pg_index: Any | None) -> None:
    """Rebuild snapshot and render status feedback.

    Args:
        vector_index: Processed vector index.
        pg_index: Optional processed property graph index.
    """
    try:
        final = rebuild_snapshot(vector_index, pg_index, settings)
        manifest = load_manifest(final)
        _render_manifest_details(manifest, final)
        st.session_state["latest_manifest"] = manifest
        st.success(f"Snapshot created: {final.name}")
    except SnapshotLockTimeout:
        st.warning(
            "Snapshot rebuild already in progress. Please wait and try again shortly."
        )
    except Exception as exc:  # pragma: no cover - UX best effort
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="documents.snapshot_rebuild")
        logger.warning(
            "Snapshot rebuild failed (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        st.warning(f"Snapshot failed ({type(exc).__name__}). Check logs for details.")


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


def _render_maintenance_controls() -> None:
    """Render operational maintenance controls (reindex/delete)."""
    uploads_dir = settings.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    with st.expander("Maintenance", expanded=False):
        st.caption(
            "Operational tools: reindex PDF page images and delete uploaded documents."
        )

        st.subheader("Reindex PDF page images")
        st.caption(
            "Re-renders PDF pages to images and reindexes SigLIP vectors in Qdrant. "
            "This can repair missing artifacts or restore the image collection after "
            "an outage. Text/vector/graph indices are not rebuilt."
        )
        cols = st.columns(3)
        limit = cols[0].number_input(
            "PDF limit",
            min_value=1,
            max_value=200,
            value=25,
            step=1,
            key="reindex_pdf_limit",
        )
        encrypt = cols[1].checkbox(
            "Encrypt images",
            value=bool(settings.processing.encrypt_page_images),
            key="reindex_encrypt_images",
        )
        do_reindex = cols[2].button("Reindex", use_container_width=True)
        if do_reindex:
            _handle_reindex_page_images(
                uploads_dir=uploads_dir, limit=int(limit), encrypt=encrypt
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
        purge = st.checkbox(
            "Also delete local artifacts (may break old chats)",
            value=False,
            key="delete_upload_purge_artifacts",
        )
        confirm = st.checkbox(
            "I understand this cannot be undone",
            value=False,
            key="delete_upload_confirm",
        )
        if st.button("Delete", disabled=not confirm, type="secondary"):
            target = uploads_dir / str(selection)
            _handle_delete_upload(target=target, purge_artifacts=bool(purge))


@st.cache_data(show_spinner=False)
def _sha256_for_file(path_str: str, _mtime_ns: int, _size: int) -> str:
    """Compute file sha256 with cache invalidated by stat tuple.

    Args:
        path_str: Path to the target file.
        _mtime_ns: Modification time in nanoseconds (for cache busting).
        _size: File size in bytes (for cache busting).

    Returns:
        str: Hexadecimal SHA-256 hash of the file content.
    """
    p = Path(path_str)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _doc_id_for_upload(path: Path) -> str:
    """Return canonical document id for an uploaded file."""
    stt = path.stat()
    digest = _sha256_for_file(str(path), int(stt.st_mtime_ns), int(stt.st_size))
    return f"doc-{digest[:16]}"


def _handle_reindex_page_images(
    *, uploads_dir: Path, limit: int, encrypt: bool
) -> None:
    from src.models.processing import IngestionConfig, IngestionInput
    from src.processing.ingestion_pipeline import reindex_page_images_sync

    pdfs = sorted(uploads_dir.glob("*.pdf"), key=lambda p: p.name)[: int(limit)]
    if not pdfs:
        st.info("No PDFs found under uploads.")
        return

    cache_dir = settings.cache_dir / "ingestion"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = IngestionConfig(
        chunk_size=int(settings.processing.chunk_size),
        chunk_overlap=int(settings.processing.chunk_overlap),
        enable_image_encryption=bool(encrypt),
        enable_image_indexing=True,
        cache_dir=cache_dir,
        cache_collection=f"{settings.database.qdrant_image_collection}_image_reindex",
        docstore_path=None,
    )

    inputs: list[IngestionInput] = []
    for p in pdfs:
        try:
            doc_id = _doc_id_for_upload(p)
        except Exception as exc:
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(str(exc), key_id="documents.sha256")
            logger.debug(
                "sha256 compute skipped for {} (error_type={} error={})",
                p.name,
                type(exc).__name__,
                redaction.redacted,
            )
            continue
        inputs.append(
            IngestionInput(
                document_id=doc_id,
                source_path=p,
                metadata={"source_filename": p.name},
                encrypt_images=bool(encrypt),
            )
        )

    with st.status("Reindexing page images…", expanded=True) as status:
        result = reindex_page_images_sync(cfg, inputs)
        meta_raw = result.get("metadata") if isinstance(result, dict) else None
        meta = meta_raw if isinstance(meta_raw, dict) else {}
        st.json(meta or {}, expanded=False)
        st.success(
            f"Reindexed {meta.get('image_index.indexed', 0)} page images "
            f"({meta.get('image_index.skipped', 0)} skipped)."
        )
        status.update(label="Done", state="complete")


def _handle_delete_upload(*, target: Path, purge_artifacts: bool) -> None:  # noqa: PLR0915
    """Delete an uploaded file and best-effort clean Qdrant points."""
    uploads_dir = settings.data_dir / "uploads"
    uploads_dir_resolved = uploads_dir.resolve()

    candidate = target.expanduser()
    # Reject symlinks: avoid deleting the link target outside uploads.
    try:
        if candidate.is_symlink():
            st.error("Refusing to delete symlink.")
            return
    except OSError:
        st.error("Invalid path.")
        return

    # Containment: only allow direct children of uploads_dir.
    try:
        if candidate.parent.resolve() != uploads_dir_resolved:
            st.error("Refusing to delete path outside uploads directory.")
            return
    except OSError:
        st.error("Invalid path.")
        return

    if not candidate.exists():
        st.warning("File not found.")
        return

    doc_id = _doc_id_for_upload(candidate)

    deleted_image_points = 0
    deleted_text_points = 0

    try:
        from qdrant_client import QdrantClient
        from qdrant_client import models as qmodels

        from src.retrieval.image_index import (
            collect_artifact_refs_for_doc_id,
            count_artifact_references_in_image_collection,
            delete_page_images_for_doc_id,
        )
        from src.utils.storage import get_client_config

        client = QdrantClient(**get_client_config())
        try:
            artifacts_to_consider: list[ArtifactRef] = (
                list(
                    collect_artifact_refs_for_doc_id(
                        client,
                        settings.database.qdrant_image_collection,
                        doc_id=doc_id,
                    )
                )
                if purge_artifacts
                else []
            )

            deleted_image_points = delete_page_images_for_doc_id(
                client,
                settings.database.qdrant_image_collection,
                doc_id=doc_id,
            )

            # Best-effort: delete text points for this doc id (key varies by indexer).
            text_filter = qmodels.Filter(
                should=[
                    qmodels.FieldCondition(
                        key="doc_id", match=qmodels.MatchValue(value=str(doc_id))
                    ),
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=str(doc_id)),
                    ),
                    qmodels.FieldCondition(
                        key="ref_doc_id", match=qmodels.MatchValue(value=str(doc_id))
                    ),
                ]
            )
            try:
                before = client.count(
                    collection_name=settings.database.qdrant_collection,
                    count_filter=text_filter,
                    exact=True,
                )
                text_count = int(getattr(before, "count", 0) or 0)
                client.delete(
                    collection_name=settings.database.qdrant_collection,
                    points_selector=qmodels.FilterSelector(filter=text_filter),
                    wait=True,
                )
                deleted_text_points = text_count
            except Exception:
                deleted_text_points = 0

            if purge_artifacts and artifacts_to_consider:
                store = ArtifactStore.from_settings(settings)
                removed = 0
                for ref in artifacts_to_consider:
                    # Safety: only delete if not referenced anywhere in the image
                    # collection. This does not check chat persistence; old chats may
                    # lose images if artifacts are purged.
                    ref_count = count_artifact_references_in_image_collection(
                        client,
                        settings.database.qdrant_image_collection,
                        artifact_id=ref.sha256,
                    )
                    if ref_count == 0:
                        with contextlib.suppress(Exception):
                            store.delete(ref)
                            removed += 1
                st.caption(f"Deleted local artifacts: {removed}")
        finally:
            with contextlib.suppress(Exception):
                client.close()
    except Exception as exc:
        st.caption(f"Qdrant cleanup skipped: {type(exc).__name__}")

    with contextlib.suppress(Exception):
        candidate.unlink()

    st.success(
        f"Deleted {candidate.name}. "
        "Qdrant: image_points="
        f"{deleted_image_points} text_points≈{deleted_text_points}."
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

    st.subheader("Snapshot Utilities")
    st.caption(
        "Rebuild persisted files and manifest for the current graph/vector"
        " indices. If your documents changed, re-ingest first to rebuild"
        " the graph."
    )
    if st.button("Rebuild GraphRAG Snapshot"):
        try:
            pg_index = st.session_state.get("graphrag_index")
            vector_index = st.session_state.get("vector_index")
            if vector_index is None:
                st.warning("Vector index missing; ingest documents first.")
            else:
                final = rebuild_snapshot(vector_index, pg_index, settings)
                manifest = load_manifest(final)
                _render_manifest_details(manifest, final)
                st.success(f"Snapshot rebuilt: {final.name}")
        except SnapshotLockTimeout:
            st.warning(
                "Snapshot rebuild already in progress. Please wait and try again."
            )
        except Exception as exc:  # pragma: no cover - UX best effort
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(
                str(exc), key_id="documents.snapshot_manager"
            )
            logger.warning(
                "Snapshot manager error (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            st.error(f"Snapshot manager error ({type(exc).__name__}).")
            st.caption(f"Error reference: {redaction.redacted}")


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


def _create_vector_index_fallback() -> Any | None:
    """Create a vector index from persisted store when ingestion skipped it."""
    try:
        from llama_index.core import VectorStoreIndex

        store = create_vector_store(
            settings.database.qdrant_collection,
            enable_hybrid=getattr(settings.retrieval, "enable_server_hybrid", True),
        )
        return VectorStoreIndex.from_vector_store(store)
    except Exception:  # pragma: no cover - defensive
        return None


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


def rebuild_snapshot(
    vector_index: Any, pg_index: Any | None, settings_obj: Any
) -> Path:
    """Rebuild snapshot for current indices and return final path."""
    from src.persistence.snapshot_service import rebuild_snapshot as _rebuild

    return _rebuild(
        vector_index,
        pg_index,
        settings_obj,
        log_export_event=_log_export_event,
        record_graph_export_metric=record_graph_export_metric,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
