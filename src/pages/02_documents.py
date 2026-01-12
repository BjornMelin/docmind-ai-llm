"""Streamlit Documents page wiring the ingestion pipeline and snapshots."""

from __future__ import annotations

import contextlib
import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger

from src.config.settings import settings
from src.persistence.artifacts import ArtifactRef, ArtifactStore
from src.persistence.snapshot import SnapshotLockTimeout, load_manifest
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
from src.ui.ingest_adapter import ingest_files
from src.utils.storage import create_vector_store


def main() -> None:  # pragma: no cover - Streamlit page
    """Render the Documents page and handle ingestion form submissions."""
    configure_observability(settings)
    st.title("Documents")

    _render_latest_snapshot_summary()

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

    if submitted:
        if not files:
            st.warning("No files selected.")
        else:
            with st.status("Ingesting…", expanded=True) as status:
                try:
                    result = ingest_files(
                        files,
                        enable_graphrag=use_graphrag,
                        encrypt_images=encrypt_images,
                    )
                    count = int(result.get("count", 0))
                    st.write(f"Ingested {count} documents.")
                    _render_image_exports(result.get("exports") or [])

                    vector_index = result.get("vector_index")
                    pg_index = result.get("pg_index") if use_graphrag else None

                    if vector_index is None:
                        vector_index = _create_vector_index_fallback()

                    if vector_index is not None:
                        st.session_state["vector_index"] = vector_index
                        # Best-effort: provide a multimodal retriever for Chat/agents.
                        try:
                            from src.retrieval.multimodal_fusion import (
                                MultimodalFusionRetriever,
                            )

                            st.session_state["hybrid_retriever"] = (
                                MultimodalFusionRetriever()
                            )
                        except Exception:  # pragma: no cover - UI best-effort
                            logger.debug(
                                "Multimodal retriever init failed", exc_info=True
                            )
                            st.session_state.pop("hybrid_retriever", None)
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

                    try:
                        final = rebuild_snapshot(vector_index, pg_index, settings)
                        manifest = load_manifest(final)
                        _render_manifest_details(manifest, final)
                        st.session_state["latest_manifest"] = manifest
                        st.success(f"Snapshot created: {final.name}")
                    except SnapshotLockTimeout:
                        st.warning(
                            "Snapshot rebuild already in progress. Please wait "
                            "and try again shortly."
                        )
                    except Exception as exc:  # pragma: no cover - UX best effort
                        st.warning(f"Snapshot failed: {exc}")

                    status.update(label="Done", state="complete")
                    st.toast("Ingestion complete", icon="✅")
                except SnapshotLockTimeout:
                    status.update(label="Locked", state="error")
                    st.warning(
                        "Snapshot rebuild already in progress. Please wait "
                        "and try again shortly."
                    )
                except Exception as exc:  # pragma: no cover - UX best effort
                    status.update(label="Failed", state="error")
                    st.error(f"Ingestion failed: {exc}")

    _render_maintenance_controls()

    # Snapshot utilities and manual exports are driven by session_state indices.
    _render_export_controls()


def _render_image_exports(exports: list[dict[str, Any]]) -> None:
    """Render a lightweight preview of PDF page-image exports."""
    images = []
    for e in exports:
        if not isinstance(e, dict):
            continue
        ct = str(e.get("content_type") or "")
        if ct.startswith("image/"):
            images.append(e)
    if not images:
        return

    store = ArtifactStore.from_settings(settings)

    def _doc_id(meta: dict[str, Any]) -> str:
        return str(meta.get("doc_id") or meta.get("document_id") or "-")

    def _page_no(meta: dict[str, Any]) -> int:
        raw = meta.get("page_no") or meta.get("page") or meta.get("page_number")
        try:
            return int(raw) if raw is not None else 0
        except (TypeError, ValueError):
            return 0

    by_doc: dict[str, list[dict[str, Any]]] = {}
    for e in images:
        meta = e.get("metadata")
        meta = meta if isinstance(meta, dict) else {}
        by_doc.setdefault(_doc_id(meta), []).append(e)

    with st.expander("Page images (preview)", expanded=False):
        st.caption(
            "Previews are loaded from local artifacts (no base64 stored in Qdrant)."
        )
        max_preview = min(
            128,
            max((len(items) for items in by_doc.values()), default=0),
        )
        if max_preview:
            preview_limit = st.slider(
                "Preview per document",
                min_value=1,
                max_value=max_preview,
                value=min(32, max_preview),
                key="doc_preview_limit",
            )
        else:
            preview_limit = 32
        for doc_id, items in sorted(by_doc.items(), key=lambda t: t[0]):
            st.subheader(f"Document: {doc_id}")
            ordered = sorted(
                items,
                key=lambda x: _page_no(x.get("metadata") or {}),
            )
            cols = st.columns(4)
            for i, e in enumerate(ordered[: int(preview_limit)]):
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
                    try:
                        img_path = store.resolve_path(ref)
                        if str(img_path).endswith(".enc"):
                            from src.utils.images import open_image_encrypted

                            with open_image_encrypted(str(img_path)) as im:
                                st.image(
                                    im, caption=f"p{page_no}", use_container_width=True
                                )
                        else:
                            st.image(
                                str(img_path),
                                caption=f"p{page_no}",
                                use_container_width=True,
                            )
                    except Exception:
                        st.caption(f"p{page_no or '?'} (missing)")


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
def _sha256_for_file(path_str: str, mtime_ns: int, size: int) -> str:
    """Compute file sha256 with cache invalidated by stat tuple."""
    del mtime_ns, size
    p = Path(path_str)
    data = p.read_bytes()
    return hashlib.sha256(data).hexdigest()


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
        cache_collection="docmind_image_reindex",
        docstore_path=None,
    )

    inputs: list[IngestionInput] = []
    for p in pdfs:
        try:
            doc_id = _doc_id_for_upload(p)
        except Exception as exc:
            logger.debug("sha256 compute skipped for %s: %s", p, exc)
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


def _handle_delete_upload(*, target: Path, purge_artifacts: bool) -> None:
    """Delete an uploaded file and best-effort clean Qdrant points."""
    uploads_dir = (settings.data_dir / "uploads").resolve()
    try:
        resolved = target.resolve()
    except OSError:
        st.error("Invalid path.")
        return
    if not resolved.is_relative_to(uploads_dir):
        st.error("Refusing to delete path outside uploads directory.")
        return
    if not resolved.exists():
        st.warning("File not found.")
        return

    doc_id = _doc_id_for_upload(resolved)

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

        with QdrantClient(**get_client_config()) as client:
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
                deleted_text_points = int(getattr(before, "count", 0) or 0)
                client.delete(
                    collection_name=settings.database.qdrant_collection,
                    points_selector=text_filter,
                    wait=True,
                )
            except Exception:
                deleted_text_points = 0

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
    except Exception as exc:
        st.caption(f"Qdrant cleanup skipped: {type(exc).__name__}")

    with contextlib.suppress(Exception):
        resolved.unlink()

    st.success(
        f"Deleted {resolved.name}. "
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
            st.error(f"Snapshot manager error: {exc}")


def _handle_manual_export(out_dir: Path, extension: str) -> None:
    """Handle manual graph export actions."""
    try:
        pg_index = st.session_state["graphrag_index"]
        vector_index = st.session_state.get("vector_index")
        cap = int(getattr(settings.graphrag_cfg, "export_seed_cap", 32))
        seeds = get_export_seed_ids(pg_index, vector_index, cap=cap)
        out = _timestamped_export_path(out_dir, extension)
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
        st.warning(f"{extension.upper()} export failed: {exc}")


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


def _timestamped_export_path(out_dir: Path, extension: str) -> Path:
    """Return a timestamped export path within ``out_dir``."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = out_dir / f"graph_export-{ts}.{extension}"
    counter = 1
    while candidate.exists():
        candidate = out_dir / f"graph_export-{ts}-{counter}.{extension}"
        counter += 1
    return candidate


def _log_export_event(payload: dict[str, Any]) -> None:
    """Emit a telemetry event for graph exports (best effort)."""
    event = {**payload, "timestamp": datetime.now(UTC).isoformat()}
    dest_path = event.get("dest_path")
    if dest_path:
        # Final-release: never persist filesystem paths (absolute or relative) in
        # telemetry. Keep only a safe filename for operator diagnostics.
        with contextlib.suppress(Exception):
            event["dest_basename"] = Path(str(dest_path)).name
        event.pop("dest_path", None)
        event.pop("dest_relpath", None)
    with contextlib.suppress(Exception):
        from src.utils.telemetry import log_jsonl

        log_jsonl(event)


def rebuild_snapshot(vector_index: Any, pg_index: Any, settings_obj: Any) -> Path:
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
