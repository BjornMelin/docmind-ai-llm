# pylint: disable=invalid-name,broad-exception-caught
"""Streamlit Documents page.

Allows users to upload and ingest files into the system. The function delegates
the ingestion to a thin adapter that wraps the document processing pipeline
while reporting progress in the UI.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import streamlit as st
from llama_index.core import VectorStoreIndex
from loguru import logger

from src.config.settings import settings
from src.persistence.snapshot import (
    SnapshotManager,
    compute_config_hash,
    compute_corpus_hash,
)
from src.retrieval.graph_config import (
    export_graph_jsonl,
    export_graph_parquet,
    get_export_seed_ids,
)
from src.retrieval.router_factory import build_router_engine
from src.ui.ingest_adapter import ingest_files
from src.utils.storage import create_vector_store


def main() -> None:  # pragma: no cover - Streamlit page
    """Render the Documents page and handle ingestion form submissions."""
    st.title("Documents")

    with st.form("ingest_form", clear_on_submit=False):
        files = st.file_uploader("Add files", type=None, accept_multiple_files=True)
        use_graphrag = st.checkbox("Build GraphRAG (beta)", value=False)
        with st.expander("About snapshots", expanded=False):
            st.markdown(
                "- Snapshots are created atomically in data/storage/<timestamp>.\n"
                "- manifest.json stores corpus/config hashes for staleness detection.\n"
                "- Rebuild a snapshot anytime; for new content, re-ingest first."
            )
        submitted = st.form_submit_button("Ingest")

    if submitted:
        if not files:
            st.warning("No files selected.")
        else:
            with st.status("Ingesting…", expanded=True) as status:
                try:
                    result = ingest_files(files, enable_graphrag=use_graphrag)
                    count = int(result.get("count", 0))
                    st.write(f"Ingested {count} documents.")
                    # Build/update router engine in session after indexing
                    with contextlib.suppress(Exception):
                        vs = create_vector_store(
                            settings.database.qdrant_collection, enable_hybrid=True
                        )
                        vector_index = VectorStoreIndex.from_vector_store(vs)
                        # Store for Chat page overrides (tools_data)
                        st.session_state["vector_index"] = vector_index
                        # Default router (vector-only)
                        st.session_state["router_engine"] = build_router_engine(
                            vector_index, None, settings
                        )
                        st.info("Router engine is ready for Chat.")
                        # If GraphRAG was requested, keep the PG index for tools
                        if use_graphrag and result.get("pg_index") is not None:
                            pg_index = result["pg_index"]
                            st.session_state["graphrag_index"] = pg_index
                            st.info("GraphRAG index is available.")
                            # Build GraphRAG router (vector + graph tools)
                            st.session_state["router_engine"] = build_router_engine(
                                vector_index, pg_index, settings
                            )
                            # Persist snapshot (vector + graph)
                            storage_dir = settings.data_dir / "storage"
                            mgr = SnapshotManager(storage_dir)
                            paths = mgr.begin_snapshot()
                            try:
                                mgr.persist_vector_index(vector_index, paths)
                                mgr.persist_graph_store(
                                    pg_index.property_graph_store, paths
                                )
                                # Compute hashes
                                uploads_dir = settings.data_dir / "uploads"
                                corpus_paths = (
                                    list(uploads_dir.glob("**/*"))
                                    if uploads_dir.exists()
                                    else []
                                )
                                chash = compute_corpus_hash(
                                    [p for p in corpus_paths if p.is_file()],
                                    base_dir=uploads_dir,
                                )
                                cfg = {
                                    "router": settings.retrieval.router,
                                    "hybrid": settings.retrieval.enable_server_hybrid,
                                    "graph_enabled": True,
                                    "chunk_size": settings.processing.chunk_size,
                                    "chunk_overlap": settings.processing.chunk_overlap,
                                }
                                cfg_hash = compute_config_hash(cfg)
                                vtype = settings.database.vector_store_type
                                mgr.write_manifest(
                                    paths,
                                    index_id="docmind",
                                    graph_store_type="property_graph",
                                    vector_store_type=vtype,
                                    corpus_hash=chash,
                                    config_hash=cfg_hash,
                                    versions={"app": settings.app_version},
                                )
                                final = mgr.finalize_snapshot(paths)
                                st.success(f"Snapshot created: {final.name}")
                            except Exception as e:  # pragma: no cover - UX best effort
                                mgr.cleanup_tmp(paths)
                                st.warning(f"Snapshot failed: {e}")
                            # Export helpers with simple seeds
                            try:
                                seeds: list[str] = get_export_seed_ids(
                                    pg_index, vector_index, cap=32
                                )
                                out_dir = settings.data_dir / "graph"
                                export_graph_jsonl(
                                    pg_index, out_dir / "graph.jsonl", seeds
                                )
                                export_graph_parquet(
                                    pg_index, out_dir / "graph.parquet", seeds
                                )
                                out_resolved = Path(out_dir).resolve()
                                st.info(
                                    f"Exports written to: {out_resolved} (best effort)"
                                )
                            except Exception as e:  # pragma: no cover - UX best effort
                                st.warning(f"Graph exports failed: {e}")
                    status.update(label="Done", state="complete")
                    st.toast("Ingestion complete", icon="✅")
                except Exception as e:  # pragma: no cover - UX best-effort
                    status.update(label="Failed", state="error")
                    st.error(f"Ingestion failed: {e}")

    # Post-ingest utilities: export buttons when a graph index is available
    if st.session_state.get("graphrag_index") is not None:
        st.subheader("GraphRAG Exports")
        out_dir = settings.data_dir / "graph"
        out_dir.mkdir(parents=True, exist_ok=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export JSONL"):
                try:
                    pg_index = st.session_state["graphrag_index"]
                    vector_index = st.session_state.get("vector_index")
                    cap = int(getattr(settings.graphrag_cfg, "export_seed_cap", 32))
                    seeds: list[str] = get_export_seed_ids(
                        pg_index, vector_index, cap=cap
                    )
                    out = out_dir / "graph.jsonl"
                    export_graph_jsonl(pg_index, out, seeds)
                    st.success(f"Exported JSONL to {out}")
                    try:
                        from src.utils.telemetry import log_jsonl

                        log_jsonl(
                            {
                                "export_performed": True,
                                "export_type": "graph_jsonl",
                                "seed_count": len(seeds),
                                "capped": len(seeds) >= 32,
                                "dest_relpath": str(
                                    out_dir.relative_to(settings.data_dir)
                                ),
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Telemetry logging failed (best-effort): {e}")
                except Exception as e:  # pragma: no cover - UX best effort
                    st.warning(f"JSONL export failed: {e}")
        with col2:
            if st.button("Export Parquet"):
                try:
                    pg_index = st.session_state["graphrag_index"]
                    vector_index = st.session_state.get("vector_index")
                    cap = int(getattr(settings.graphrag_cfg, "export_seed_cap", 32))
                    seeds = get_export_seed_ids(pg_index, vector_index, cap=cap)
                    out = out_dir / "graph.parquet"
                    export_graph_parquet(pg_index, out, seeds)
                    st.success(f"Exported Parquet to {out}")
                    try:
                        from src.utils.telemetry import log_jsonl

                        log_jsonl(
                            {
                                "export_performed": True,
                                "export_type": "graph_parquet",
                                "seed_count": len(seeds),
                                "capped": len(seeds) >= 32,
                                "dest_relpath": str(
                                    out_dir.relative_to(settings.data_dir)
                                ),
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Telemetry logging failed (best-effort): {e}")
                except Exception as e:  # pragma: no cover - UX best effort
                    st.warning(f"Parquet export failed: {e}")

        st.subheader("Snapshot Utilities")
        st.caption(
            "Rebuild persisted files and manifest for the current graph/vector"
            " indices. If your documents changed, re-ingest first to rebuild"
            " the graph."
        )
        if st.button("Rebuild GraphRAG Snapshot"):
            try:
                pg_index = st.session_state["graphrag_index"]
                vector_index = st.session_state.get("vector_index")
                if vector_index is None:
                    st.warning("Vector index missing; ingest documents first.")
                else:
                    final = rebuild_snapshot(vector_index, pg_index, settings)
                    st.success(f"Snapshot rebuilt: {final.name}")
            except Exception as e:  # pragma: no cover - UX best effort
                st.error(f"Snapshot manager error: {e}")


# ---- Page helpers ----


def rebuild_snapshot(vector_index: Any, pg_index: Any, settings_obj: Any) -> Path:
    """Rebuild snapshot for current indices and return final path."""
    storage_dir = settings_obj.data_dir / "storage"
    mgr = SnapshotManager(storage_dir)
    paths = mgr.begin_snapshot()
    try:
        mgr.persist_vector_index(vector_index, paths)
        mgr.persist_graph_store(pg_index.property_graph_store, paths)
        uploads_dir = settings_obj.data_dir / "uploads"
        corpus_paths = (
            [p for p in uploads_dir.glob("**/*") if p.is_file()]
            if uploads_dir.exists()
            else []
        )
        chash = compute_corpus_hash(corpus_paths, base_dir=uploads_dir)
        cfg = {
            "router": settings_obj.retrieval.router,
            "hybrid": settings_obj.retrieval.enable_server_hybrid,
            "graph_enabled": True,
            "chunk_size": settings_obj.processing.chunk_size,
            "chunk_overlap": settings_obj.processing.chunk_overlap,
        }
        cfg_hash = compute_config_hash(cfg)
        mgr.write_manifest(
            paths,
            index_id="docmind",
            graph_store_type="property_graph",
            vector_store_type=settings_obj.database.vector_store_type,
            corpus_hash=chash,
            config_hash=cfg_hash,
            versions={"app": settings_obj.app_version},
        )
        final = mgr.finalize_snapshot(paths)
        return final
    except Exception:
        mgr.cleanup_tmp(paths)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
