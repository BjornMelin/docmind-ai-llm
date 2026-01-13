"""Streamlit Chat page.

This page renders a simple chat UI backed by the multi-agent coordinator.

The coordinator does not expose a streaming interface; we simulate streaming by
writing the response in small chunks to the UI for better perceived latency.
"""

from __future__ import annotations

import atexit
import contextlib
import sqlite3
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from loguru import logger
from opentelemetry import trace

from src.agents.coordinator import MultiAgentCoordinator
from src.config.settings import settings
from src.persistence.artifacts import ArtifactRef, ArtifactStore
from src.persistence.chat_db import open_chat_db, touch_session
from src.persistence.langchain_embeddings import LlamaIndexEmbeddingsAdapter
from src.persistence.memory_store import DocMindSqliteStore
from src.persistence.snapshot import (
    latest_snapshot_dir,
    load_manifest,
    load_property_graph_index,
    load_vector_index,
)
from src.persistence.snapshot_utils import (
    collect_corpus_paths as _collect_corpus_paths,
)
from src.persistence.snapshot_utils import (
    compute_staleness,
)
from src.persistence.snapshot_utils import (
    current_config_dict as _current_config_dict,
)
from src.retrieval.router_factory import build_router_engine
from src.telemetry.opentelemetry import configure_observability
from src.ui.chat_sessions import (
    ChatSelection,
    get_chat_db_conn,
    render_session_sidebar,
    render_time_travel_sidebar,
)
from src.ui.components.provider_badge import provider_badge
from src.utils.telemetry import log_jsonl

# Exact UI copy required by SPEC-014 acceptance
STALE_TOOLTIP = (
    "Snapshot is stale (content/config changed). Rebuild in Documents → "
    "Rebuild GraphRAG Snapshot."
)


_TRACER = trace.get_tracer("docmind.chat")


@st.cache_resource(show_spinner=False)
def _get_checkpointer() -> SqliteSaver:
    conn = open_chat_db(settings.chat.sqlite_path, cfg=settings)
    saver = SqliteSaver(conn)
    saver.setup()
    atexit.register(_close_checkpointer, saver, conn)
    return saver


@st.cache_resource(show_spinner=False)
def _get_memory_store() -> DocMindSqliteStore:
    store = DocMindSqliteStore(
        settings.chat.sqlite_path,
        index={
            "dims": int(settings.embedding.dimension),
            "embed": LlamaIndexEmbeddingsAdapter(),
            "fields": ["content"],
        },
        filter_fetch_cap=int(settings.chat.memory_store_filter_fetch_cap),
        cfg=settings,
    )
    atexit.register(_close_memory_store, store)
    return store


@st.cache_resource(show_spinner=False)
def _get_coordinator() -> MultiAgentCoordinator:
    return MultiAgentCoordinator(
        checkpointer=_get_checkpointer(), store=_get_memory_store()
    )


def _close_checkpointer(saver: SqliteSaver, conn: Any) -> None:
    with contextlib.suppress(Exception):
        close = getattr(saver, "close", None)
        if callable(close):
            close()
    with contextlib.suppress(Exception):
        conn.close()


def _close_memory_store(store: DocMindSqliteStore) -> None:
    with contextlib.suppress(Exception):
        store.close()


def _chunked_stream(text: str, chunk_size: int = 48) -> Iterable[str]:
    """Yield text in fixed-size chunks.

    Args:
        text: Full text to emit.
        chunk_size: Maximum size of each emitted chunk.

    Yields:
        Chunks of ``text`` up to ``chunk_size`` characters.
    """
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


def _get_settings_override() -> dict[str, Any] | None:
    """Return settings_override for the coordinator, if available.

    Uses a prebuilt router engine stored in Streamlit session state by the
    Documents page. When present, this enables intelligent routing in Chat.
    Also forwards optional retrieval components when available:
    - vector_index -> tools_data['vector'] for vector/hybrid flows
    - hybrid_retriever -> tools_data['retriever'] for server-side fusion
    - graphrag_index -> tools_data['kg'] for GraphRAG flows
    """
    overrides: dict[str, Any] = {}
    router = st.session_state.get("router_engine")
    if router is not None:
        overrides["router_engine"] = router
    # Optional retrieval components
    if st.session_state.get("vector_index") is not None:
        overrides["vector"] = st.session_state["vector_index"]
    if st.session_state.get("hybrid_retriever") is not None:
        overrides["retriever"] = st.session_state["hybrid_retriever"]
    if st.session_state.get("graphrag_index") is not None:
        overrides["kg"] = st.session_state["graphrag_index"]
    return overrides or None


def main() -> None:  # pragma: no cover - Streamlit page
    """Render the Chat page and handle interactions."""
    configure_observability(settings)
    st.title("Chat")
    provider_badge(settings)

    conn = get_chat_db_conn()
    selection = render_session_sidebar(conn)
    coord = _get_coordinator()
    render_time_travel_sidebar(
        checkpoints=coord.list_checkpoints(
            thread_id=selection.thread_id, user_id=selection.user_id, limit=20
        )
    )
    _render_memory_sidebar(selection.user_id, selection.thread_id)
    _render_visual_search_sidebar()
    _ensure_router_engine()
    _render_staleness_badge()
    messages = _load_chat_messages(coord, selection)
    _render_chat_history(messages)
    _handle_chat_prompt(coord, selection, conn)


# ---- Page helpers ----


def _set_last_sources_for_render(
    sources: list[dict[str, Any]], *, thread_id: str
) -> None:
    """Persist last sources in session_state for fragment rendering."""
    st.session_state["active_thread_id"] = str(thread_id)
    st.session_state["last_sources"] = sources


@st.fragment
def _render_sources_fragment() -> None:
    """Render retrieved sources, including page images when available.

    Uses a fragment so pagination controls do not force a full-page rerun.
    """
    sources = st.session_state.get("last_sources") or []
    if not isinstance(sources, list) or not sources:
        return

    store = ArtifactStore.from_settings(settings)
    try:
        from src.utils.images import open_image_encrypted
    except Exception:
        open_image_encrypted = None
    thread_id = str(st.session_state.get("active_thread_id") or "default")
    max_show = min(50, len(sources))
    default_show = min(10, max_show)

    with st.expander("Sources", expanded=False):
        show_n = st.slider(
            "Show sources",
            min_value=1,
            max_value=max_show,
            value=default_show,
            key=f"chat_sources_show_n__{thread_id}",
        )
        for idx, src in enumerate(sources[: int(show_n)], start=1):
            meta = src.get("metadata") if isinstance(src, dict) else None
            meta = meta if isinstance(meta, dict) else {}
            modality = str(meta.get("modality") or "")
            doc_id = meta.get("doc_id") or meta.get("document_id") or "-"
            page_no = meta.get("page_no") or meta.get("page") or meta.get("page_number")
            st.caption(f"{idx}. doc={doc_id} page={page_no or '-'} modality={modality}")

            if modality == "pdf_page_image":
                # Prefer thumbnail, fall back to full image.
                thumb_id = meta.get("thumbnail_artifact_id")
                thumb_sfx = meta.get("thumbnail_artifact_suffix") or ""
                img_id = meta.get("image_artifact_id")
                img_sfx = meta.get("image_artifact_suffix") or ""
                ref = None
                if thumb_id:
                    ref = ArtifactRef(sha256=str(thumb_id), suffix=str(thumb_sfx))
                elif img_id:
                    ref = ArtifactRef(sha256=str(img_id), suffix=str(img_sfx))
                if ref is not None:
                    try:
                        img_path = store.resolve_path(ref)
                        if str(img_path).endswith(".enc"):
                            if open_image_encrypted is not None:
                                with open_image_encrypted(str(img_path)) as im:
                                    st.image(im, use_container_width=True)
                            else:
                                st.caption("Encryption support unavailable.")
                        else:
                            st.image(str(img_path), use_container_width=True)
                    except Exception:
                        st.caption("Image artifact unavailable (reindex to restore).")
                continue

            content = src.get("content") if isinstance(src, dict) else ""
            if content:
                st.markdown(str(content)[:800])


def _memory_sidebar_namespace(
    user_id: str, thread_id: str, scope: str
) -> tuple[str, ...]:
    if scope == "session":
        return ("memories", str(user_id), str(thread_id))
    return ("memories", str(user_id))


def _render_memory_add(store: DocMindSqliteStore, ns: tuple[str, ...]) -> None:
    add = st.text_input("Add memory", key="memory_add")
    last_saved = st.session_state.get("_memory_last_saved")
    if st.button("Save memory", key="memory_save") and add.strip():
        content = add.strip()
        if content != last_saved:
            store.put(
                ns,
                str(uuid.uuid4()),
                {"content": content, "kind": "fact", "importance": 0.7},
                index=["content"],
            )
            st.session_state["_memory_last_saved"] = content
            st.session_state["memory_add"] = ""
            st.rerun()


def _render_memory_results(store: DocMindSqliteStore, ns: tuple[str, ...]) -> None:
    q = st.text_input("Search", key="memory_search").strip()
    results = store.search(ns, query=q or None, limit=10)
    if not results:
        st.caption("No memories stored.")
        return

    for item in results:
        value = getattr(item, "value", None)
        content = value.get("content") if isinstance(value, dict) else None
        st.caption(f"{item.key} · score={getattr(item, 'score', None)}")
        if content:
            st.write(str(content))

        confirm_key = f"mem_del_confirm__{item.key}"
        confirm = st.checkbox("Confirm", key=confirm_key)
        if st.button(
            "Delete",
            key=f"mem_del_{item.key}",
            disabled=not confirm,
        ):
            item_namespace = getattr(item, "namespace", ns)
            store.delete(item_namespace, str(item.key))
            st.session_state.pop(confirm_key, None)
            st.rerun()


def _purge_memory_namespace(store: DocMindSqliteStore, ns: tuple[str, ...]) -> int:
    purged = 0
    while True:
        batch = store.search(ns, query=None, limit=5000)
        if not batch:
            break
        for item in batch:
            store.delete(getattr(item, "namespace", ns), str(item.key))
        purged += len(batch)
        if len(batch) < 5000:
            break
    return purged


def _render_memory_purge(
    store: DocMindSqliteStore,
    ns: tuple[str, ...],
    scope: str,
    user_id: str,
    thread_id: str,
) -> None:
    st.divider()
    st.caption("Danger zone")
    purge_confirm_key = f"mem_purge_confirm__{scope}__{user_id}__{thread_id}"
    purge_confirm = st.checkbox(
        "Confirm purge (irreversible)",
        key=purge_confirm_key,
    )
    if st.button(
        "Purge scope memories",
        type="primary",
        key=f"mem_purge__{scope}__{user_id}__{thread_id}",
        disabled=not purge_confirm,
    ):
        purged = _purge_memory_namespace(store=store, ns=ns)
        logger.info(
            "Purged {} memory items (scope={} user_id={} thread_id={})",
            purged,
            scope,
            user_id,
            thread_id,
        )
        st.session_state.pop(purge_confirm_key, None)
        st.rerun()


def _render_memory_sidebar(user_id: str, thread_id: str) -> None:
    """Render a simple memory review UI (ADR-058/SPEC-041)."""
    store = _get_memory_store()

    with st.sidebar:
        st.subheader("Memories")
        scope = st.selectbox(
            "Scope",
            options=["session", "user"],
            index=0,
            key="memory_scope",
        )
        ns = _memory_sidebar_namespace(
            user_id=user_id, thread_id=thread_id, scope=scope
        )
        _render_memory_add(store=store, ns=ns)
        _render_memory_results(store=store, ns=ns)
        _render_memory_purge(
            store=store, ns=ns, scope=scope, user_id=user_id, thread_id=thread_id
        )


def _ensure_router_engine() -> None:
    """Ensure a router engine is available in session state."""
    try:
        _load_latest_snapshot_into_session()
    except Exception as exc:
        st.caption(f"Autoload skipped: {exc}")
    if "router_engine" not in st.session_state:
        try:
            snap = latest_snapshot_dir()
            if snap is not None:
                _hydrate_router_from_snapshot(snap)
        except Exception as exc:
            logger.debug("Hydration from snapshot failed: %s", exc)
    if "router_engine" not in st.session_state:
        st.session_state["router_engine"] = None


def _render_staleness_badge() -> None:
    """Render snapshot staleness status in the UI."""
    try:
        storage_dir = settings.data_dir / "storage"
        if not storage_dir.exists():
            return
        latest = latest_snapshot_dir(storage_dir)
        if latest is None:
            return
        with _TRACER.start_as_current_span("chat.staleness_check") as span:
            span.set_attribute("snapshot.id", latest.name)
            manifest_data = load_manifest(latest)
            if not manifest_data:
                return
            uploads_dir = settings.data_dir / "uploads"
            corpus_paths = _collect_corpus_paths(uploads_dir)
            cfg = _current_config_dict()
            is_stale = compute_staleness(manifest_data, corpus_paths, cfg)
            span.set_attribute("snapshot.is_stale", bool(is_stale))
            if is_stale:
                st.warning(STALE_TOOLTIP)
                with st.sidebar:
                    st.caption("Snapshot stale: content or config changed.")
                with contextlib.suppress(Exception):
                    log_jsonl({
                        "snapshot_stale_detected": True,
                        "snapshot_id": latest.name,
                        "reason": "digest_mismatch",
                    })
            else:
                st.caption(f"Snapshot up-to-date: {latest.name}")
    except Exception as exc:
        st.caption(f"Staleness check skipped: {exc}")


def _load_chat_messages(coord: Any, selection: ChatSelection) -> list[Any]:
    """Load persisted chat messages from the coordinator."""
    try:
        state = coord.get_state_values(
            thread_id=selection.thread_id, user_id=selection.user_id
        )
        return state.get("messages", []) if isinstance(state, dict) else []
    except Exception as exc:
        logger.debug(
            "Failed to load chat history for thread_id={} user_id={}: {}",
            selection.thread_id,
            selection.user_id,
            exc,
        )
        return []


def _render_chat_history(messages: list[Any]) -> None:
    """Render historical chat messages."""
    for msg in messages:
        role = None
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        if role is None:
            continue
        with st.chat_message(role):
            content = getattr(msg, "content", "")
            st.markdown(str(content))


def _handle_chat_prompt(
    coord: Any, selection: ChatSelection, conn: sqlite3.Connection
) -> None:
    """Handle user input and render assistant response."""
    prompt = st.chat_input("Ask something…")
    if not prompt:
        return
    with st.chat_message("user"):
        st.markdown(prompt)

    overrides = _get_settings_override()
    resp = coord.process_query(
        query=prompt,
        context=None,
        settings_override=overrides,
        thread_id=selection.thread_id,
        user_id=selection.user_id,
        checkpoint_id=selection.resume_checkpoint_id,
    )
    st.session_state.pop("chat_resume_checkpoint_id", None)
    answer = getattr(resp, "content", str(resp))
    sources = getattr(resp, "sources", []) if resp is not None else []

    with st.chat_message("assistant"):
        st.write_stream(_chunked_stream(answer))
        if isinstance(sources, list) and sources:
            _set_last_sources_for_render(sources, thread_id=selection.thread_id)
            _render_sources_fragment()
    with contextlib.suppress(Exception):
        cps = coord.list_checkpoints(
            thread_id=selection.thread_id, user_id=selection.user_id, limit=1
        )
        last = cps[0].get("checkpoint_id") if cps else None
        touch_session(
            conn,
            thread_id=selection.thread_id,
            last_checkpoint_id=str(last) if last else None,
        )


def _visual_search_inputs() -> tuple[Any | None, int, bool]:
    """Collect visual search inputs from the sidebar UI."""
    with st.sidebar:
        st.subheader("Visual search")
        up = st.file_uploader(
            "Query image",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            key="chat_query_image",
        )
        if up is None:
            st.caption("Upload an image to search visually indexed PDF pages.")
            return None, 0, False
        cols = st.columns(2)
        top_k = cols[0].number_input("Top K", min_value=1, max_value=30, value=8)
        run = cols[1].button("Search", use_container_width=True)
        return up, int(top_k), bool(run)


def _query_visual_search(upload: Any, top_k: int) -> list[Any]:
    """Execute a SigLIP-powered visual search against Qdrant."""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        st.warning("Pillow is required for visual search.")
        return []

    from src.retrieval.multimodal_fusion import ImageSearchParams, ImageSiglipRetriever

    img = Image.open(upload)  # type: ignore[arg-type]
    retriever = ImageSiglipRetriever(
        ImageSearchParams(
            collection=settings.database.qdrant_image_collection, top_k=int(top_k)
        )
    )
    try:
        return retriever.retrieve_by_image(img, top_k=int(top_k))
    finally:
        retriever.close()


def _render_visual_results(nodes: list[Any], top_k: int) -> None:
    """Render visual search results in the sidebar."""
    if not nodes:
        st.caption("No matches.")
        return

    try:
        from src.utils.images import open_image_encrypted
    except Exception:
        open_image_encrypted = None

    store = ArtifactStore.from_settings(settings)
    st.caption(f"Matches: {len(nodes)}")
    for item in nodes[: int(top_k)]:
        node = getattr(item, "node", None)
        payload = getattr(node, "metadata", {}) or {}
        doc_id = payload.get("doc_id") or payload.get("document_id") or "-"
        page_no = (
            payload.get("page_no")
            or payload.get("page")
            or payload.get("page_number")
            or "-"
        )
        st.caption(f"doc={doc_id} page={page_no}")

        thumb_id = payload.get("thumbnail_artifact_id")
        thumb_sfx = payload.get("thumbnail_artifact_suffix") or ""
        img_id = payload.get("image_artifact_id")
        img_sfx = payload.get("image_artifact_suffix") or ""
        ref = None
        if thumb_id:
            ref = ArtifactRef(sha256=str(thumb_id), suffix=str(thumb_sfx))
        elif img_id:
            ref = ArtifactRef(sha256=str(img_id), suffix=str(img_sfx))
        if ref is None:
            continue

        try:
            img_path = store.resolve_path(ref)
            if str(img_path).endswith(".enc"):
                if open_image_encrypted is not None:
                    with open_image_encrypted(str(img_path)) as im:
                        st.image(im, use_container_width=True)
                else:
                    st.caption("Encryption support unavailable.")
            else:
                st.image(str(img_path), use_container_width=True)
        except Exception:
            st.caption("Image artifact unavailable (reindex to restore).")


def _render_visual_search_sidebar() -> None:
    """Render a lightweight query-by-image UI (SigLIP → Qdrant image collection)."""
    upload, top_k, run = _visual_search_inputs()
    if not upload or not run:
        return
    try:
        points = _query_visual_search(upload, top_k)
        _render_visual_results(points, top_k)
    except Exception as exc:  # pragma: no cover - UI best-effort
        st.caption(f"Visual search unavailable: {type(exc).__name__}")


def _load_latest_snapshot_into_session() -> None:
    """Autoload the latest snapshot into session_state per policy.

    Policies:
    - latest_non_stale (default): Load when manifest not stale.
    - pinned: Load pinned snapshot id when configured.
    - ignore: Do nothing.
    """
    policy = getattr(settings.graphrag_cfg, "autoload_policy", "latest_non_stale")
    if policy == "ignore":
        return

    snap_dir: Path | None = None
    if policy == "pinned":
        sid = getattr(settings.graphrag_cfg, "pinned_snapshot_id", None)
        if sid:
            storage_root = (settings.data_dir / "storage").resolve()
            candidate = (storage_root / sid).resolve()
            if candidate.exists() and candidate.is_relative_to(storage_root):
                snap_dir = candidate
    else:  # latest_non_stale
        snap_dir = latest_snapshot_dir()

    if not snap_dir:
        return

    man = load_manifest(snap_dir)
    if policy == "latest_non_stale":
        # Prefer non-stale snapshots; if stale, still hydrate to enable chat
        # while the UI shows a stale warning.
        try:
            uploads_dir = settings.data_dir / "uploads"
            corpus_paths = _collect_corpus_paths(uploads_dir)
            cfg = _current_config_dict()
            if man:
                compute_staleness(man, corpus_paths, cfg)
                _hydrate_router_from_snapshot(snap_dir)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ):  # pragma: no cover - defensive
            return
    else:  # pinned (skip staleness)
        _hydrate_router_from_snapshot(snap_dir)


def _hydrate_router_from_snapshot(snap_dir: Path) -> None:
    # Avoid repeated hydration on every rerun; snapshot load can be expensive.
    current_id = str(getattr(snap_dir, "name", snap_dir))
    if (
        st.session_state.get("_snapshot_loaded_id") == current_id
        and "router_engine" in st.session_state
    ):
        return

    vec = load_vector_index(snap_dir)
    kg = load_property_graph_index(snap_dir)
    # Store in session for downstream tools (keep None if not available)
    st.session_state["vector_index"] = vec
    if kg is not None:
        st.session_state["graphrag_index"] = kg
    # Best-effort: provide a multimodal fusion retriever for agent tools.
    try:
        from src.retrieval.multimodal_fusion import MultimodalFusionRetriever

        old_retriever = st.session_state.get("hybrid_retriever")
        if old_retriever is not None:
            with contextlib.suppress(Exception):
                old_retriever.close()
        st.session_state["hybrid_retriever"] = MultimodalFusionRetriever()
    except Exception:  # pragma: no cover - fail open in UI
        st.session_state.pop("hybrid_retriever", None)
    # Build router with fail-open to vector-only
    try:
        router = build_router_engine(vec, kg, settings)
        st.session_state["router_engine"] = router
        st.session_state["_snapshot_loaded_id"] = current_id
        st.caption(
            f"Autoloaded snapshot: {snap_dir.name} (graph={'yes' if kg else 'no'})"
        )
    except (
        ValueError,
        RuntimeError,
        OSError,
        TypeError,
    ):  # pragma: no cover - defensive
        # No router created; keep vector/graph in session for manual wiring
        import logging

        logging.getLogger(__name__).debug(
            "Failed to build router from snapshot; continuing without wiring",
            exc_info=True,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
