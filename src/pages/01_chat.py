"""Streamlit Chat page.

This page renders a simple chat UI backed by the multi-agent coordinator.

The coordinator does not expose a streaming interface; we simulate streaming by
writing the response in small chunks to the UI for better perceived latency.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from loguru import logger
from opentelemetry import trace

from src.agents.coordinator import MultiAgentCoordinator
from src.analysis.models import AnalysisResult, DocumentRef
from src.analysis.service import (
    AnalysisCancelledError,
    discover_uploaded_documents,
    run_analysis,
)
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
from src.ui.artifacts import render_artifact_image
from src.ui.background_jobs import (
    JobCanceledError,
    ProgressEvent,
    get_job_manager,
    get_or_create_owner_id,
)
from src.ui.chat_sessions import (
    ChatSelection,
    get_chat_db_conn,
    render_session_sidebar,
    render_time_travel_sidebar,
)
from src.ui.components.provider_badge import provider_badge
from src.utils.log_safety import build_pii_log_entry
from src.utils.telemetry import log_jsonl

# Exact UI copy required by SPEC-014 acceptance
STALE_TOOLTIP = (
    "Snapshot is stale (content/config changed). Rebuild in Documents → "
    "Rebuild GraphRAG Snapshot."
)


_TRACER = trace.get_tracer("docmind.chat")


@st.cache_resource(show_spinner=False)
def _get_checkpointer() -> SqliteSaver:
    """Initialize and return the SQLite checkpointer for LangGraph.

    Returns:
        SqliteSaver: Configured checkpointer instance.
    """
    conn = open_chat_db(settings.chat.sqlite_path, cfg=settings)
    saver = SqliteSaver(conn)
    saver.setup()
    atexit.register(_close_checkpointer, saver, conn)
    return saver


@st.cache_resource(show_spinner=False)
def _get_memory_store() -> DocMindSqliteStore:
    """Initialize and return the SQLite-backed memory store.

    Returns:
        DocMindSqliteStore: Configured memory store instance.
    """
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
    """Initialize and return the MultiAgentCoordinator.

    Returns:
        MultiAgentCoordinator: Configured coordinator instance.
    """
    return MultiAgentCoordinator(
        checkpointer=_get_checkpointer(), store=_get_memory_store()
    )


def _close_checkpointer(saver: SqliteSaver, conn: Any) -> None:
    """Close the checkpointer and its underlying database connection.

    Args:
        saver: The SqliteSaver instance to close.
        conn: The SQLite connection to close.
    """
    with contextlib.suppress(Exception):
        close = getattr(saver, "close", None)
        if callable(close):
            close()
    with contextlib.suppress(Exception):
        conn.close()


def _close_memory_store(store: DocMindSqliteStore) -> None:
    """Close the memory store.

    Args:
        store: The DocMindSqliteStore instance to close.
    """
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

    Returns:
        dict[str, Any] | None: Settings override dictionary or None.
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

    owner_id = get_or_create_owner_id()
    conn = get_chat_db_conn()
    selection = render_session_sidebar(conn)
    coord = _get_coordinator()
    render_time_travel_sidebar(
        coord=coord,
        conn=conn,
        thread_id=selection.thread_id,
        user_id=selection.user_id,
        checkpoints=coord.list_checkpoints(
            thread_id=selection.thread_id, user_id=selection.user_id, limit=20
        ),
    )
    _render_memory_sidebar(selection.user_id, selection.thread_id)
    _render_visual_search_sidebar()
    _ensure_router_engine()
    _render_analysis_sidebar(owner_id=owner_id)
    _render_analysis_job_panel(owner_id=owner_id)
    _render_analysis_results()
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
                    render_artifact_image(
                        ref,
                        store=store,
                        use_container_width=True,
                        missing_caption=(
                            "Image artifact unavailable (reindex to restore)."
                        ),
                        encrypted_caption="Encryption support unavailable.",
                    )
                continue

            content = src.get("content") if isinstance(src, dict) else ""
            if content:
                st.markdown(str(content)[:800])


def _memory_sidebar_namespace(
    user_id: str, thread_id: str, scope: str
) -> tuple[str, ...]:
    """Return the memory namespace for the given scope.

    Args:
        user_id: The unique identifier for the user.
        thread_id: The unique identifier for the chat thread.
        scope: The storage scope ('session' or 'user').

    Returns:
        tuple[str, ...]: The hierarchical namespace for memory storage.
    """
    if scope == "session":
        return ("memories", str(user_id), str(thread_id))
    return ("memories", str(user_id))


def _render_memory_add(store: DocMindSqliteStore, ns: tuple[str, ...]) -> None:
    """Render the UI to add a new memory.

    Args:
        store: The memory store instance.
        ns: The storage namespace.
    """
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
    """Render memory search results and deletion controls.

    Args:
        store: The memory store instance.
        ns: The storage namespace.
    """
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
    """Delete all memory items in a given namespace.

    Args:
        store: The memory store instance.
        ns: The storage namespace to purge.

    Returns:
        int: Total number of items purged.
    """
    purged = 0
    max_batches = 100
    total_failures = 0
    max_failures = 50
    for _ in range(max_batches):
        batch = store.search(ns, query=None, limit=5000)
        if not batch:
            break

        batch_deleted = 0
        for item in batch:
            try:
                store.delete(getattr(item, "namespace", ns), str(item.key))
                batch_deleted += 1
            except Exception as exc:
                redaction = build_pii_log_entry(str(exc), key_id="chat.memory_delete")
                key_redacted = build_pii_log_entry(
                    str(item.key), key_id="chat.memory_item_key"
                ).redacted
                logger.debug(
                    "Failed to delete memory item (key={}, error_type={}, error={})",
                    key_redacted,
                    type(exc).__name__,
                    redaction.redacted,
                )
                total_failures += 1

        if batch_deleted == 0:
            logger.warning("Purge stuck: failed to delete any items in batch")
            break
        if total_failures >= max_failures:
            logger.warning("Purge aborted: too many failures ({})", total_failures)
            break

        purged += batch_deleted
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
    """Render the memory purge control with a confirmation lock.

    Args:
        store: The memory store instance.
        ns: The storage namespace.
        scope: The storage scope name.
        user_id: The unique identifier for the user.
        thread_id: The unique identifier for the chat thread.
    """
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
        redaction = build_pii_log_entry(str(exc), key_id="chat.autoload_snapshot")
        logger.debug(
            "Autoload skipped (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        st.caption("Autoload skipped.")
    if "router_engine" not in st.session_state:
        try:
            snap = latest_snapshot_dir()
            if snap is not None:
                _hydrate_router_from_snapshot(snap)
        except Exception as exc:
            redaction = build_pii_log_entry(str(exc), key_id="chat.hydrate_router")
            logger.debug(
                "Hydration from snapshot failed (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
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
                    log_jsonl(
                        {
                            "snapshot_stale_detected": True,
                            "snapshot_id": latest.name,
                            "reason": "digest_mismatch",
                        }
                    )
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
        thread_redacted = build_pii_log_entry(
            str(selection.thread_id), key_id="chat.thread_id"
        ).redacted
        user_redacted = build_pii_log_entry(
            str(selection.user_id), key_id="chat.user_id"
        ).redacted
        err_redaction = build_pii_log_entry(str(exc), key_id="chat.load_history")
        logger.debug(
            "Failed to load chat history (thread_id={}, user_id={}, "
            "error_type={}, error={})",
            thread_redacted,
            user_redacted,
            type(exc).__name__,
            err_redaction.redacted,
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
    try:
        resp = coord.process_query(
            query=prompt,
            context=None,
            settings_override=overrides,
            thread_id=selection.thread_id,
            user_id=selection.user_id,
        )
    except Exception as exc:
        err_redaction = build_pii_log_entry(str(exc), key_id="chat.process_query")
        logger.error(
            "Chat request failed (error_type={} error={})",
            type(exc).__name__,
            err_redaction.redacted,
        )
        with st.chat_message("assistant"):
            st.error(
                "Request failed. Verify provider settings and endpoint connectivity."
            )
            st.caption(f"Error id: {err_redaction.fingerprint[:12]}")
        return
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


@st.cache_data(show_spinner=False)
def _cached_uploaded_documents(
    uploads_dir_str: str, uploads_mtime_ns: int
) -> list[DocumentRef]:
    """Cache uploaded document discovery for sidebar selectors."""
    _ = uploads_mtime_ns  # cache bust when uploads change
    return discover_uploaded_documents(Path(uploads_dir_str))


def _analysis_selected_documents(
    all_docs: list[DocumentRef], selected_labels: list[str]
) -> list[DocumentRef]:
    """Return selected documents by label."""
    by_label = {f"{d.doc_name} ({d.doc_id})": d for d in all_docs}
    selected: list[DocumentRef] = []
    for label in selected_labels:
        doc = by_label.get(str(label))
        if doc is not None:
            selected.append(doc)
    return selected


def _get_analysis_sidebar_inputs() -> tuple[str, list[DocumentRef], str]:
    """Gather user inputs for analysis from the sidebar.

    Returns:
        tuple[str, list[DocumentRef], str]: Selected mode, list of documents,
            and the user analysis query.
    """
    mode_options = ["auto", "combined", "separate"]
    default_mode = str(settings.analysis.mode)
    try:
        default_index = mode_options.index(default_mode)
    except ValueError:
        default_index = 0
    requested_mode = st.selectbox(
        "Mode",
        options=mode_options,
        index=default_index,
        key="analysis_mode",
    )

    uploads_dir = settings.data_dir / "uploads"
    try:
        uploads_mtime_ns = (
            int(uploads_dir.stat().st_mtime_ns) if uploads_dir.exists() else 0
        )
    except OSError:
        uploads_mtime_ns = 0

    docs = _cached_uploaded_documents(str(uploads_dir), uploads_mtime_ns)
    labels = [f"{d.doc_name} ({d.doc_id})" for d in docs]
    selected_labels = st.multiselect(
        "Documents (optional)",
        options=labels,
        default=[],
        key="analysis_docs",
        help="Select documents to scope analysis; leave empty to use the whole corpus.",
    )
    selected_docs = _analysis_selected_documents(docs, selected_labels)

    query = st.text_area(
        "Prompt",
        key="analysis_query",
        height=120,
        placeholder="Ask a question to analyze across selected documents…",
    ).strip()

    return requested_mode, selected_docs, query


def _run_analysis_job_work(
    cancel_event: threading.Event,
    report: Any,
    *,
    query: str,
    mode: str,
    vector_index: Any,
    documents: list[DocumentRef],
) -> AnalysisResult:
    """Worker function for background analysis jobs (SPEC-036).

    Args:
        cancel_event: Threading event to signal cancellation.
        report: Function to report progress events.
        query: The user prompt for analysis.
        mode: The analysis mode ('auto', 'combined', 'separate').
        vector_index: The vector index to search against.
        documents: Optional list of documents to scope the analysis.

    Returns:
        AnalysisResult: The consolidated analysis results.
    """

    def _emit(percent: int, phase: str, message: str) -> None:
        evt = ProgressEvent(
            percent=max(0, min(100, int(percent))),
            phase=phase,  # type: ignore[arg-type]
            message=str(message)[:200],
            timestamp=datetime.now(UTC),
        )
        with contextlib.suppress(Exception):
            report(evt)

    _emit(0, "analysis", "Starting")

    def _progress(pct: int, message: str) -> None:
        _emit(pct, "analysis", message)

    try:
        result = run_analysis(
            query=query,
            mode=mode,  # type: ignore[arg-type]
            vector_index=vector_index,
            documents=documents,
            cfg=settings,
            cancel_event=cancel_event,
            report_progress=_progress,
        )
    except AnalysisCancelledError as exc:
        raise JobCanceledError() from exc

    _emit(100, "done", "Done")
    return result


def _render_analysis_sidebar(*, owner_id: str) -> None:
    """Render analysis controls in the sidebar (SPEC-036)."""
    with st.sidebar:
        st.subheader("Analysis")
        requested_mode, selected_docs, query = _get_analysis_sidebar_inputs()

        job_id = st.session_state.get("analysis_job_id")
        has_job = isinstance(job_id, str) and bool(job_id)
        cols = st.columns(2)
        run_clicked = cols[0].button(
            "Run analysis",
            key="analysis_run",
            type="primary",
            disabled=has_job,
            use_container_width=True,
        )
        cancel_clicked = cols[1].button(
            "Cancel",
            key="analysis_cancel",
            disabled=not has_job,
            use_container_width=True,
        )

        if cancel_clicked and has_job:
            get_job_manager(settings.cache_version).cancel(
                str(job_id), owner_id=owner_id
            )
            return

        if not run_clicked:
            return
        if not query:
            st.error("Analysis prompt is required.")
            return
        vector_index = st.session_state.get("vector_index")
        if vector_index is None:
            st.error("No snapshot loaded. Build a snapshot in Documents first.")
            return
        if requested_mode == "separate" and not selected_docs:
            st.error("Separate mode requires selecting at least one document.")
            return

        job_manager = get_job_manager(settings.cache_version)
        work_fn = functools.partial(
            _run_analysis_job_work,
            query=query,
            mode=requested_mode,
            vector_index=vector_index,
            documents=selected_docs,
        )
        job_id = job_manager.start_job(owner_id=owner_id, fn=work_fn)
        st.session_state["analysis_job_id"] = job_id
        st.session_state.pop("analysis_last_result", None)
        st.toast("Analysis started", icon="⏳")
        return


@st.fragment(run_every=float(settings.ui.progress_poll_interval_sec))
def _render_analysis_job_panel(*, owner_id: str) -> None:
    """Render analysis job progress and stash results when complete.

    Args:
        owner_id: The unique identifier for the job owner.
    """
    job_id = st.session_state.get("analysis_job_id")
    if not isinstance(job_id, str) or not job_id:
        return

    job_manager = get_job_manager(settings.cache_version)
    state = job_manager.get(job_id, owner_id=owner_id)
    if state is None:
        st.session_state.pop("analysis_job_id", None)
        return

    events = job_manager.drain_progress(job_id, owner_id=owner_id)
    if events:
        st.session_state["analysis_last_event"] = events[-1]

    last_event = st.session_state.get("analysis_last_event")
    last: ProgressEvent | None = (
        last_event if isinstance(last_event, ProgressEvent) else None
    )
    pct = int(last.percent) if isinstance(last, ProgressEvent) else 0
    phase = last.phase if isinstance(last, ProgressEvent) else "analysis"
    message = last.message if isinstance(last, ProgressEvent) else ""

    st.subheader("Analysis job")
    st.progress(max(0, min(100, pct)) / 100.0)
    st.caption(f"{phase} · {pct}%")
    if message:
        st.caption(message)

    if state.status == "running":
        if st.button("Cancel analysis", key="analysis_cancel_inline"):
            job_manager.cancel(job_id, owner_id=owner_id)
        return

    st.session_state.pop("analysis_job_id", None)
    succeeded = state.status == "succeeded"
    result_is_analysis = isinstance(state.result, AnalysisResult)
    if succeeded and result_is_analysis:
        st.session_state["analysis_last_result"] = state.result
        st.success("Analysis completed.")
    elif state.status == "canceled":
        st.warning("Analysis canceled.")
    elif state.status == "failed":
        st.error("Analysis failed.")


def _render_analysis_results() -> None:
    """Render completed analysis results including summary and per-doc tabs."""
    result = st.session_state.get("analysis_last_result")
    if not isinstance(result, AnalysisResult):
        return

    st.subheader(f"Analysis ({result.mode})")
    for warning in result.warnings:
        st.warning(str(warning)[:200])

    if result.mode == "combined":
        if result.combined:
            st.markdown(result.combined)
        return

    if result.reduce:
        with st.expander("Summary", expanded=True):
            st.markdown(result.reduce)

    if not result.per_doc:
        st.caption("No per-document results available.")
        return

    tabs = st.tabs([r.doc_name for r in result.per_doc])
    for tab, doc_res in zip(tabs, result.per_doc, strict=False):
        with tab:
            st.markdown(doc_res.answer)
            if doc_res.citations:
                st.caption(f"Citations: {len(doc_res.citations)}")


def _visual_search_inputs() -> tuple[Any | None, int, bool]:
    """Collect visual search inputs from the sidebar UI.

    Returns:
        tuple[Any | None, int, bool]: Uploaded file, top_k count, and search
            button click status.
    """
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


@st.cache_resource
def _get_image_siglip_retriever() -> Any:
    """Get cached ImageSiglipRetriever to avoid reconnection overhead."""
    from src.retrieval.multimodal_fusion import ImageSearchParams, ImageSiglipRetriever

    return ImageSiglipRetriever(
        ImageSearchParams(
            collection=settings.database.qdrant_image_collection, top_k=10
        )
    )


def _query_visual_search(upload: Any, top_k: int) -> list[Any]:
    """Execute a SigLIP-powered visual search against Qdrant.

    Args:
        upload: The uploaded image file.
        top_k: Number of results to retrieve.

    Returns:
        list[Any]: List of retrieved image nodes from Qdrant.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        st.warning("Pillow is required for visual search.")
        return []

    img = Image.open(upload)  # type: ignore[arg-type]
    retriever = _get_image_siglip_retriever()
    return retriever.retrieve_by_image(img, top_k=int(top_k))


def _render_visual_results(nodes: list[Any], top_k: int) -> None:
    """Render visual search results in the sidebar.

    Args:
        nodes: List of results from the visual retriever.
        top_k: Maximum number of results to display.
    """
    if not nodes:
        st.caption("No matches.")
        return

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

        render_artifact_image(
            ref,
            store=store,
            use_container_width=True,
            missing_caption="Image artifact unavailable (reindex to restore).",
            encrypted_caption="Encryption support unavailable.",
        )


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
    """Hydrate the router engine and indices from a snapshot directory.

    Args:
        snap_dir: Path to the snapshot storage directory.
    """
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
    if vec is None:
        logger.debug(
            "Snapshot autoload skipped: missing vector index (snapshot_id={})",
            current_id,
        )
        st.session_state["router_engine"] = None
        st.session_state["_snapshot_loaded_id"] = current_id
        return
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
        AttributeError,
        ValueError,
        RuntimeError,
        OSError,
        TypeError,
    ) as exc:  # pragma: no cover - defensive
        # No router created; keep vector/graph in session for manual wiring
        redaction = build_pii_log_entry(str(exc), key_id="chat.router_from_snapshot")
        logger.debug(
            "Failed to build router from snapshot; continuing without wiring "
            "(error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
