"""Streamlit Documents page.

Allows users to upload and ingest files into the system. The function delegates
the ingestion to a thin adapter that wraps the document processing pipeline
while reporting progress in the UI.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import streamlit as st
from llama_index.core import VectorStoreIndex

from src.config.settings import settings
from src.persistence.snapshot import (
    SnapshotManager,
    compute_config_hash,
    compute_corpus_hash,
)
from src.retrieval.graph_config import export_graph_jsonl, export_graph_parquet
from src.retrieval.query_engine import (
    ServerHybridRetriever,
    _HybridParams,
    create_adaptive_router_engine,
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
                        retriever = ServerHybridRetriever(
                            _HybridParams(
                                collection=settings.database.qdrant_collection
                            )
                        )
                        # Store for Chat page overrides (tools_data)
                        st.session_state.vector_index = vector_index
                        st.session_state.hybrid_retriever = retriever
                        # Default router (adaptive, non-graph)
                        st.session_state.router_engine = create_adaptive_router_engine(
                            vector_index=vector_index, hybrid_retriever=retriever
                        )
                        st.info("Router engine is ready for Chat.")
                        # If GraphRAG was requested, keep the PG index for tools
                        if use_graphrag and result.get("pg_index") is not None:
                            pg_index = result["pg_index"]
                            st.session_state.graphrag_index = pg_index
                            st.info("GraphRAG index is available.")
                            # Build GraphRAG router (vector + graph tools)
                            st.session_state.router_engine = build_router_engine(
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
                                    [p for p in corpus_paths if p.is_file()]
                                )
                                cfg = {
                                    "router": settings.retrieval.router,
                                    "hybrid": settings.retrieval.hybrid_enabled,
                                    "graph_enabled": True,
                                    "chunk_size": settings.processing.chunk_size,
                                    "chunk_overlap": settings.processing.chunk_overlap,
                                }
                                cfg_hash = compute_config_hash(cfg)
                                mgr.write_manifest(
                                    paths,
                                    index_id="docmind",
                                    graph_store_type="property_graph",
                                    vector_store_type=settings.database.vector_store_type,
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
                                store = pg_index.property_graph_store
                                # Seed with up to 50 nodes from store
                                seeds = []
                                for idx, n in enumerate(store.get_nodes()):  # type: ignore[attr-defined]
                                    if idx >= 50:
                                        break
                                    node_id = getattr(n, "id", None)
                                    if node_id is not None:
                                        seeds.append(str(node_id))
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
                    store = pg_index.property_graph_store
                    seeds: list[str] = []
                    for idx, n in enumerate(store.get_nodes()):  # type: ignore[attr-defined]
                        if idx >= 50:
                            break
                        nid = getattr(n, "id", None)
                        if nid is not None:
                            seeds.append(str(nid))
                    out = out_dir / "graph.jsonl"
                    export_graph_jsonl(pg_index, out, seeds)
                    st.success(f"Exported JSONL to {out}")
                except Exception as e:  # pragma: no cover - UX best effort
                    st.warning(f"JSONL export failed: {e}")
        with col2:
            if st.button("Export Parquet"):
                try:
                    pg_index = st.session_state["graphrag_index"]
                    store = pg_index.property_graph_store
                    seeds = []
                    for idx, n in enumerate(store.get_nodes()):  # type: ignore[attr-defined]
                        if idx >= 50:
                            break
                        nid = getattr(n, "id", None)
                        if nid is not None:
                            seeds.append(str(nid))
                    out = out_dir / "graph.parquet"
                    export_graph_parquet(pg_index, out, seeds)
                    st.success(f"Exported Parquet to {out}")
                except Exception as e:  # pragma: no cover - UX best effort
                    st.warning(f"Parquet export failed: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
