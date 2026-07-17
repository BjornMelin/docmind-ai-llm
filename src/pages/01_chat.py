"""Streamlit Chat page backed by the multi-agent coordinator."""

from __future__ import annotations

import contextlib
import functools
import sqlite3
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.sqlite import SqliteStore
from loguru import logger
from opentelemetry import trace

from src.analysis.models import AnalysisResult, DocumentRef
from src.analysis.service import (
    AnalysisCancelledError,
    discover_uploaded_documents,
    run_analysis,
)
from src.config.settings import settings
from src.persistence.artifacts import ArtifactRef, ArtifactStore
from src.persistence.chat_db import touch_session
from src.persistence.checkpoint_identity import memory_namespace
from src.telemetry.opentelemetry import configure_observability
from src.ui.artifacts import render_artifact_image
from src.ui.background_jobs import (
    ForegroundRuntimeConflictError,
    JobAdmissionPausedError,
    JobCanceledError,
    JobConflictError,
    ProgressEvent,
    get_job_manager,
    get_or_create_owner_id,
)
from src.ui.chat_runtime import (
    ChatModelReadiness,
    ChatModelUnavailableError,
)
from src.ui.chat_sessions import (
    ChatSelection,
    get_chat_db_conn,
    render_session_sidebar,
    render_time_travel_sidebar,
)
from src.ui.components.provider_badge import provider_badge
from src.ui.router_session import (
    session_router_is_current,
)
from src.ui.vector_session import (
    VectorIndexResource,
    clear_session_runtime,
    replace_session_runtime,
    session_vector_resource_is_current,
)
from src.utils.log_safety import build_pii_log_entry
from src.utils.telemetry import log_jsonl

# Exact UI copy required by SPEC-014 acceptance
STALE_TOOLTIP = (
    "Snapshot is stale (content/config changed). Rebuild in Documents → "
    "Rebuild search index."
)


_TRACER = trace.get_tracer("docmind.chat")

_MEMORY_CAPACITY_ERROR = "Memory limit reached. Delete a memory before adding another."
_MEMORY_SAVE_ERROR = "Memory could not be saved. Please retry."
_MEMORY_SAVE_REJECTED_ERROR = "This memory scope was purged and is no longer writable."
_MEMORY_SEARCH_ERROR = "Memories could not be loaded. Please retry."
_MEMORY_DELETE_ERROR = "Memory could not be deleted. Please retry."
_CHAT_PERSISTENCE_ERROR = "Chat persistence is unavailable. Please retry."
_SNAPSHOT_REFRESH_BUSY = "Snapshot refresh deferred while background work is active."
_SNAPSHOT_REFRESH_FOREGROUND = (
    "Snapshot refresh deferred while the live runtime is in use."
)
_SNAPSHOT_REFRESH_MAINTENANCE = (
    "Snapshot refresh deferred while runtime maintenance is in progress."
)
_ANALYSIS_TERMINAL_NOTICE_KEY = "analysis_terminal_notice"
_ANALYSIS_COMPLETED_JOB_KEY = "analysis_job_completed_id"

if TYPE_CHECKING:
    from src.agents.coordinator import MultiAgentCoordinator
    from src.retrieval.multimodal_fusion import ImageSiglipRetriever


@dataclass(frozen=True, slots=True)
class _MemoryPurgeResult:
    """Outcome of one verified namespace purge."""

    deleted: int
    failures: int
    complete: bool


@dataclass(frozen=True, slots=True)
class _MemoryOperationResult[ValueT]:
    """Sanitized result of one sidebar memory-store operation."""

    value: ValueT | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _ChatHistoryLoadResult:
    """Immutable, UI-safe outcome of loading one Chat history."""

    messages: tuple[Any, ...]
    status: Literal["ready", "maintenance", "failed"]
    fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class _SnapshotStatus:
    """Immutable local snapshot state captured for one Chat rerun."""

    snapshot_id: str | None
    snapshot_dir: Path | None
    manifest_present: bool
    is_stale: bool | None
    error: bool = False


def _run_memory_operation[ValueT](
    operation: Callable[[], ValueT],
    *,
    name: str,
    error_message: str,
    accepted: Callable[[ValueT], bool] | None = None,
    rejected_message: str | None = None,
) -> _MemoryOperationResult[ValueT]:
    """Run one memory operation behind the sidebar's error boundary."""
    from src.agents.tools.memory import MemoryCapacityError

    try:
        value = operation()
    except MemoryCapacityError:
        return _MemoryOperationResult(value=None, error=_MEMORY_CAPACITY_ERROR)
    except Exception as exc:
        redaction = build_pii_log_entry(str(exc), key_id=f"chat.memory_{name}")
        logger.debug(
            "Memory sidebar operation failed (operation={} error_type={} error={})",
            name,
            type(exc).__name__,
            redaction.redacted,
        )
        return _MemoryOperationResult(value=None, error=error_message)

    if accepted is not None and not accepted(value):
        logger.warning("Memory sidebar operation rejected (operation={})", name)
        return _MemoryOperationResult(
            value=None,
            error=rejected_message or error_message,
        )
    return _MemoryOperationResult(value=value, error=None)


def _close_memory_store(store: SqliteStore) -> None:
    """Close the memory store."""
    from src.persistence.memory_store import close_memory_store

    with contextlib.suppress(Exception):
        close_memory_store(store)


@st.cache_resource(show_spinner=False, on_release=_close_memory_store)
def _get_memory_store() -> SqliteStore:
    """Initialize and return the SQLite-backed memory store.

    Returns:
        Configured native LangGraph SQLite store.
    """
    from src.config import setup_llamaindex
    from src.config.integrations import (
        EmbeddingModelInitializationError,
        EmbeddingModelUnavailableError,
    )
    from src.persistence.langchain_embeddings import LlamaIndexEmbeddingsAdapter
    from src.persistence.memory_store import open_memory_store

    try:
        setup_llamaindex()
    except EmbeddingModelUnavailableError as exc:
        status: Literal["cache_missing", "local_path_incomplete"] = (
            "local_path_incomplete"
            if settings.embedding.local_model_path is not None
            else "cache_missing"
        )
        raise ChatModelUnavailableError(status) from exc
    except EmbeddingModelInitializationError as exc:
        raise ChatModelUnavailableError("initialization_failed") from exc
    return open_memory_store(
        settings.chat.sqlite_path,
        index={
            "dims": int(settings.embedding.dimension),
            "embed": LlamaIndexEmbeddingsAdapter(),
            "fields": ["content"],
        },
        cfg=settings,
    )


def _get_coordinator() -> MultiAgentCoordinator:
    """Initialize and return the MultiAgentCoordinator.

    Returns:
        MultiAgentCoordinator: Configured coordinator instance.
    """
    from src.ui.chat_runtime import get_coordinator

    return get_coordinator(
        cache_version=settings.cache_version,
        checkpointer_path=settings.chat.sqlite_path,
        store=_get_memory_store(),
    )


def _check_chat_model_artifacts() -> ChatModelReadiness:
    """Check configured embedding artifacts without loading the model stack."""
    from src.ui.chat_runtime import check_model_artifacts

    configured_local_path = settings.embedding.local_model_path
    return check_model_artifacts(
        model_name=str(settings.embedding.model_name),
        model_revision=settings.embedding.model_revision,
        cache_folder=Path(settings.embedding.cache_folder),
        local_model_path=(
            Path(configured_local_path) if configured_local_path is not None else None
        ),
    )


@contextlib.contextmanager
def _coordinator_activity() -> Iterator[MultiAgentCoordinator]:
    """Lease the live runtime before acquiring its current coordinator."""
    with get_job_manager().foreground_runtime_activity():
        yield _get_coordinator()


@contextlib.contextmanager
def _chat_session_activity() -> Iterator[sqlite3.Connection]:
    """Lease and acquire the current cached Chat session connection."""
    with get_job_manager().foreground_runtime_activity():
        yield get_chat_db_conn()


@contextlib.contextmanager
def _memory_store_activity() -> Iterator[SqliteStore]:
    """Lease and acquire the current cached memory store."""
    with get_job_manager().foreground_runtime_activity():
        yield _get_memory_store()


def _purge_chat_session(
    *,
    thread_id: str,
    user_id: str,
) -> bool:
    """Purge one session through a freshly leased coordinator."""
    with _chat_session_activity() as conn, _coordinator_activity() as coord:
        return bool(
            coord.purge_session(
                conn=conn,
                thread_id=thread_id,
                user_id=user_id,
            )
        )


def _list_chat_checkpoints(
    selection: ChatSelection, *, limit: int
) -> list[dict[str, object]]:
    """List checkpoints through a freshly leased coordinator."""
    with _coordinator_activity() as coord:
        return coord.list_checkpoints(
            thread_id=selection.thread_id,
            user_id=selection.user_id,
            limit=limit,
        )


def _fork_chat_checkpoint(
    *, thread_id: str, user_id: str, checkpoint_id: str
) -> str | None:
    """Fork a checkpoint through a freshly leased coordinator."""
    with _coordinator_activity() as coord:
        return coord.fork_from_checkpoint(
            thread_id=thread_id,
            user_id=user_id,
            checkpoint_id=checkpoint_id,
        )


def _get_settings_override() -> dict[str, Any] | None:
    """Return settings_override for the coordinator, if available.

    Uses a prebuilt router engine stored in Streamlit session state by the
    Documents page. When present, this enables intelligent routing in Chat.
    The router is forwarded through LangGraph's transient
    ``ToolRuntime.context`` and is never persisted in checkpoints.

    Returns:
        dict[str, Any] | None: Settings override dictionary or None.
    """
    overrides: dict[str, Any] = {}
    router = st.session_state.get("router_engine")
    if session_router_is_current(
        st.session_state,
        runtime_generation=settings.cache_version,
    ):
        overrides["router_engine"] = router
    return overrides or None


def main() -> None:  # pragma: no cover - Streamlit page
    """Render the Chat page and handle interactions."""
    configure_observability(settings)
    st.title("Chat")
    provider_badge(settings)
    snapshot_status = _compute_snapshot_status()
    model_readiness = _check_chat_model_artifacts()

    owner_id = get_or_create_owner_id()
    try:
        with _chat_session_activity() as conn:
            selection = render_session_sidebar(
                conn,
                hard_purge=(
                    (
                        lambda thread_id, user_id: _purge_chat_session(
                            thread_id=thread_id,
                            user_id=user_id,
                        )
                    )
                    if model_readiness.status == "ready"
                    else None
                ),
            )
            _render_staleness_badge(snapshot_status)
            if model_readiness.status == "ready":
                checkpoints: list[dict[str, object]] = []
                try:
                    checkpoints = _list_chat_checkpoints(selection, limit=20)
                except ChatModelUnavailableError as exc:
                    model_readiness = ChatModelReadiness(status=exc.status)
                except JobAdmissionPausedError:
                    checkpoints = []
                    st.caption("Time travel is unavailable during runtime maintenance.")
                if model_readiness.status == "ready":
                    render_time_travel_sidebar(
                        fork_from_checkpoint=_fork_chat_checkpoint,
                        conn=conn,
                        thread_id=selection.thread_id,
                        user_id=selection.user_id,
                        checkpoints=checkpoints,
                    )
    except JobAdmissionPausedError:
        st.info("Chat sessions are unavailable during runtime maintenance.")
        return
    except sqlite3.DatabaseError as exc:
        logger.bind(
            event="chat_persistence_unavailable",
            error_type=type(exc).__name__,
        ).error("Chat persistence unavailable")
        st.error(_CHAT_PERSISTENCE_ERROR)
        return
    if model_readiness.status != "ready":
        if _snapshot_runtime_present():
            _clear_snapshot_runtime()
        _render_chat_runtime_unavailable(model_readiness)
        return
    _render_memory_sidebar(selection.user_id, selection.thread_id)
    _render_visual_search_sidebar()
    _ensure_router_engine(snapshot_status)
    _render_analysis_sidebar(owner_id=owner_id)
    _render_analysis_job_panel(owner_id=owner_id)
    _render_analysis_terminal_notice()
    _render_analysis_results()
    history = _load_chat_messages(selection)
    if not _render_chat_history_result(history):
        return
    _handle_chat_prompt(selection)


def _render_chat_runtime_unavailable(readiness: ChatModelReadiness) -> None:
    """Render sanitized setup guidance for unavailable local model artifacts."""
    if readiness.status == "local_path_incomplete":
        st.warning(
            "Chat is unavailable because the configured local model is incomplete."
        )
        st.caption(
            "Install a complete SentenceTransformers snapshot at the configured "
            "location, or remove DOCMIND_EMBEDDING__LOCAL_MODEL_PATH from your "
            "environment or .env."
        )
    elif readiness.status == "cache_missing":
        st.warning(
            "Chat is unavailable because its local model artifacts are not installed."
        )
        st.code("uv run python tools/models/pull.py --all --parser-defaults")
    else:
        st.warning("Chat is unavailable because its embedding could not initialize.")
        st.caption(
            "Verify the configured embedding model and device settings, then restart."
        )
    st.caption("Sessions and local snapshot status remain available.")


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
        return memory_namespace(user_id=user_id, thread_id=thread_id)
    return memory_namespace(user_id=user_id)


def _render_memory_add(store: SqliteStore, ns: tuple[str, ...]) -> None:
    """Render the UI to add a new memory.

    Args:
        store: The memory store instance.
        ns: The storage namespace.
    """
    from src.agents.tools.memory import MemoryWrite, save_memory

    add = st.text_input("Add memory", key="memory_add")
    st.session_state.pop("_memory_last_saved", None)
    if st.button("Save memory", key="memory_save") and add.strip():
        content = add.strip()
        result = _run_memory_operation(
            lambda: save_memory(
                store,
                ns,
                write=MemoryWrite(
                    content=content,
                    kind="fact",
                    importance=0.7,
                    tags=None,
                    origin="explicit",
                ),
            ),
            name="save",
            error_message=_MEMORY_SAVE_ERROR,
            accepted=lambda memory_id_value: memory_id_value is not None,
            rejected_message=_MEMORY_SAVE_REJECTED_ERROR,
        )
        if result.error is not None:
            st.error(result.error)
            return
        st.session_state["memory_add"] = ""
        st.rerun()


def _render_memory_results(store: SqliteStore, ns: tuple[str, ...]) -> None:
    """Render memory search results and deletion controls.

    Args:
        store: The memory store instance.
        ns: The storage namespace.
    """
    from src.agents.tools.memory import delete_memory

    q = st.text_input("Search", key="memory_search").strip()
    search_result = _run_memory_operation(
        lambda: store.search(ns, query=q or None, limit=10),
        name="search",
        error_message=_MEMORY_SEARCH_ERROR,
        accepted=lambda results: results is not None,
    )
    if search_result.error is not None:
        st.error(search_result.error)
        return
    results = search_result.value
    if results is None:
        st.error(_MEMORY_SEARCH_ERROR)
        return
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
            delete_result = _run_memory_operation(
                lambda item=item, item_namespace=item_namespace: delete_memory(
                    store,
                    item_namespace,
                    str(item.key),
                ),
                name="delete",
                error_message=_MEMORY_DELETE_ERROR,
                accepted=bool,
            )
            if delete_result.error is not None:
                st.error(delete_result.error)
                return
            st.session_state.pop(confirm_key, None)
            st.rerun()


def _purge_memory_namespace(
    store: SqliteStore, ns: tuple[str, ...]
) -> _MemoryPurgeResult:
    """Delete all memory items in a given namespace.

    Args:
        store: The memory store instance.
        ns: The storage namespace to purge.

    Returns:
        Verified deletion count, failure count, and completion state.
    """
    from src.agents.tools.memory import (
        advance_memory_namespace_generation,
        delete_memory,
        memory_namespace_lock,
    )

    purged = 0
    max_batches = 100
    total_failures = 0
    max_failures = 50
    with memory_namespace_lock(ns):
        advance_memory_namespace_generation(ns)
        search_failed = False
        for _ in range(max_batches):
            search_result = _run_memory_operation(
                lambda: store.search(ns, query=None, limit=5000),
                name="purge_search",
                error_message=_MEMORY_SEARCH_ERROR,
                accepted=lambda results: results is not None,
            )
            if search_result.error is not None:
                total_failures += 1
                search_failed = True
                break
            batch = search_result.value
            if batch is None:
                total_failures += 1
                search_failed = True
                break
            if not batch:
                break

            batch_deleted = 0
            for item in batch:
                delete_result = _run_memory_operation(
                    lambda item=item: delete_memory(
                        store,
                        getattr(item, "namespace", ns),
                        str(item.key),
                    ),
                    name="purge_delete",
                    error_message=_MEMORY_DELETE_ERROR,
                    accepted=bool,
                )
                if delete_result.error is not None:
                    total_failures += 1
                else:
                    batch_deleted += 1

            purged += batch_deleted
            if batch_deleted == 0:
                logger.warning("Purge stuck: failed to delete any items in batch")
                break
            if total_failures >= max_failures:
                logger.warning("Purge aborted: too many failures ({})", total_failures)
                break

            if len(batch) < 5000:
                break

        if search_failed:
            complete = False
        else:
            verify_result = _run_memory_operation(
                lambda: store.search(ns, query=None, limit=1),
                name="purge_verify",
                error_message=_MEMORY_SEARCH_ERROR,
                accepted=lambda results: results is not None,
            )
            if verify_result.error is not None:
                total_failures += 1
                complete = False
            else:
                complete = not verify_result.value

    return _MemoryPurgeResult(
        deleted=purged,
        failures=total_failures,
        complete=complete,
    )


def _render_memory_purge(
    store: SqliteStore,
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
        result = _purge_memory_namespace(store=store, ns=ns)
        logger.info(
            "Purged memory items (count={} failures={} complete={} scope={})",
            result.deleted,
            result.failures,
            result.complete,
            scope,
        )
        if result.complete:
            st.session_state.pop(purge_confirm_key, None)
            st.rerun()
        else:
            st.error(
                "Memory purge is incomplete. No confirmation was cleared; retry "
                "after checking the local store."
            )


def _render_memory_sidebar(user_id: str, thread_id: str) -> None:
    """Render a simple memory review UI (ADR-058/SPEC-041)."""
    try:
        with _memory_store_activity() as store, st.sidebar:
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
                store=store,
                ns=ns,
                scope=scope,
                user_id=user_id,
                thread_id=thread_id,
            )
    except JobAdmissionPausedError:
        with st.sidebar:
            st.info("Memories are unavailable during runtime maintenance.")


def _ensure_router_engine(status: _SnapshotStatus) -> None:
    """Ensure a router engine is available in session state."""
    try:
        if status.error:
            st.caption("Autoload skipped.")
        _load_latest_snapshot_into_session(status)
    except Exception as exc:
        redaction = build_pii_log_entry(str(exc), key_id="chat.autoload_snapshot")
        logger.debug(
            "Autoload skipped (error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        st.caption("Autoload skipped.")
        _clear_snapshot_runtime()


def _report_snapshot_refresh_conflict(exc: JobConflictError) -> None:
    """Render sanitized, truthful feedback for a deferred runtime mutation."""
    if isinstance(exc, JobAdmissionPausedError):
        st.caption(_SNAPSHOT_REFRESH_MAINTENANCE)
    elif isinstance(exc, ForegroundRuntimeConflictError):
        st.caption(_SNAPSHOT_REFRESH_FOREGROUND)
    else:
        st.caption(_SNAPSHOT_REFRESH_BUSY)


def _clear_snapshot_runtime() -> bool:
    """Close snapshot resources only after acquiring runtime quiescence."""
    try:
        with get_job_manager().admission_quiescence():
            if _snapshot_runtime_present():
                _clear_snapshot_runtime_quiesced()
    except (JobAdmissionPausedError, JobConflictError) as exc:
        _report_snapshot_refresh_conflict(exc)
        return False
    return True


def _clear_snapshot_runtime_quiesced() -> None:
    """Close snapshot resources while the caller owns runtime quiescence."""
    clear_session_runtime(
        st.session_state,
        runtime_generation=settings.cache_version,
    )


def _compute_snapshot_status() -> _SnapshotStatus:
    """Capture local snapshot identity and freshness for this rerun."""
    latest: Path | None = None
    try:
        from src.persistence.snapshot import latest_snapshot_dir, load_manifest
        from src.persistence.snapshot_utils import (
            collect_corpus_paths,
            compute_staleness,
            current_config_dict,
        )

        storage_dir = settings.data_dir / "storage"
        if not storage_dir.exists():
            return _SnapshotStatus(None, None, False, None)
        latest = latest_snapshot_dir(storage_dir)
        if latest is None:
            return _SnapshotStatus(None, None, False, None)
        with _TRACER.start_as_current_span("chat.staleness_check") as span:
            span.set_attribute("snapshot.id", latest.name)
            manifest_data = load_manifest(latest)
            if not manifest_data:
                return _SnapshotStatus(latest.name, latest, False, None)
            uploads_dir = settings.data_dir / "uploads"
            corpus_paths = collect_corpus_paths(uploads_dir)
            cfg = current_config_dict()
            is_stale = compute_staleness(manifest_data, corpus_paths, cfg)
            span.set_attribute("snapshot.is_stale", bool(is_stale))
            return _SnapshotStatus(latest.name, latest, True, bool(is_stale))
    except Exception as exc:
        redaction = build_pii_log_entry(str(exc), key_id="chat.staleness_check")
        logger.debug(
            "Staleness check skipped (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        return _SnapshotStatus(
            latest.name if latest is not None else None,
            latest,
            False,
            None,
            error=True,
        )


def _render_staleness_badge(status: _SnapshotStatus) -> None:
    """Render one already-computed snapshot freshness result."""
    if status.error:
        st.caption("Staleness check unavailable.")
        return
    if not status.manifest_present or status.snapshot_id is None:
        return
    if status.is_stale:
        st.warning(STALE_TOOLTIP)
        with st.sidebar:
            st.caption("Snapshot stale: content or config changed.")
        with contextlib.suppress(Exception):
            log_jsonl(
                {
                    "snapshot_stale_detected": True,
                    "snapshot_id": status.snapshot_id,
                    "reason": "digest_mismatch",
                }
            )
        return
    st.caption(f"Snapshot up-to-date: {status.snapshot_id}")


def _load_chat_messages(selection: ChatSelection) -> _ChatHistoryLoadResult:
    """Return a sanitized, render-free result for persisted Chat messages."""
    try:
        with _coordinator_activity() as coord:
            state = coord.get_state_values(
                thread_id=selection.thread_id, user_id=selection.user_id
            )
        if not isinstance(state, dict):
            raise TypeError("invalid Chat history state")
        messages = state.get("messages", [])
        if not isinstance(messages, list | tuple):
            raise TypeError("invalid Chat history messages")
        return _ChatHistoryLoadResult(
            messages=tuple(messages),
            status="ready",
        )
    except JobAdmissionPausedError:
        return _ChatHistoryLoadResult(messages=(), status="maintenance")
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
        return _ChatHistoryLoadResult(
            messages=(),
            status="failed",
            fingerprint=err_redaction.fingerprint[:12],
        )


def _render_chat_history_result(result: _ChatHistoryLoadResult) -> bool:
    """Render one history result and report whether Chat input is safe to show."""
    if result.status == "ready":
        if not result.messages:
            st.caption("No messages yet. Ask a question to start this chat.")
        _render_chat_history(result.messages)
        return True

    if result.status == "maintenance":
        st.info("Chat history is unavailable during runtime maintenance.")
    else:
        st.error("Chat history could not be loaded. Please retry.")
        if result.fingerprint:
            st.caption(f"Error id: {result.fingerprint}")

    if st.button("Retry chat history", key="chat_history_retry"):
        st.rerun()
    return False


def _render_chat_history(messages: tuple[Any, ...] | list[Any]) -> None:
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


def _handle_chat_prompt(selection: ChatSelection) -> None:
    """Handle user input and render assistant response."""
    prompt = st.chat_input("Ask something…")
    if not prompt:
        return
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with _coordinator_activity() as coord:
            conn = get_chat_db_conn()
            overrides = _get_settings_override()
            with st.spinner("Generating response…", show_time=True):
                resp = coord.process_query(
                    query=prompt,
                    settings_override=overrides,
                    thread_id=selection.thread_id,
                    user_id=selection.user_id,
                )
            try:
                checkpoints = coord.list_checkpoints(
                    thread_id=selection.thread_id,
                    user_id=selection.user_id,
                    limit=1,
                )
            except Exception:
                checkpoints = []
            with contextlib.suppress(Exception):
                last = checkpoints[0].get("checkpoint_id") if checkpoints else None
                touch_session(
                    conn,
                    thread_id=selection.thread_id,
                    last_checkpoint_id=str(last) if last else None,
                )
    except JobAdmissionPausedError:
        with st.chat_message("assistant"):
            st.warning(
                "Runtime maintenance is in progress. Try your request again shortly."
            )
        return
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
        st.markdown(answer)
        if isinstance(sources, list) and sources:
            _set_last_sources_for_render(sources, thread_id=selection.thread_id)
            _render_sources_fragment()


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
        job_manager = get_job_manager()
        activity = job_manager.activity_snapshot()
        if activity.maintenance_active:
            st.info("Runtime maintenance is in progress. Analysis is unavailable.")
        cols = st.columns(2)
        run_clicked = cols[0].button(
            "Run analysis",
            key="analysis_run",
            type="primary",
            disabled=has_job or activity.maintenance_active,
            use_container_width=True,
        )
        cancel_clicked = cols[1].button(
            "Cancel",
            key="analysis_cancel",
            disabled=not has_job,
            use_container_width=True,
        )

        if cancel_clicked and has_job:
            if get_job_manager().cancel(str(job_id), owner_id=owner_id):
                st.session_state["analysis_cancel_requested_id"] = str(job_id)
                st.warning("Cancellation requested. Waiting for a safe stopping point.")
            return

        if not run_clicked:
            return
        if not query:
            st.error("Analysis prompt is required.")
            return
        if requested_mode == "separate" and not selected_docs:
            st.error("Separate mode requires selecting at least one document.")
            return

        try:
            with job_manager.foreground_runtime_activity():
                vector_index = st.session_state.get("vector_index")
                if vector_index is None:
                    st.error("No snapshot loaded. Build a snapshot in Documents first.")
                    return
                work_fn = functools.partial(
                    _run_analysis_job_work,
                    query=query,
                    mode=requested_mode,
                    vector_index=vector_index,
                    documents=selected_docs,
                )
                job_id = job_manager.start_job(owner_id=owner_id, fn=work_fn)
        except JobAdmissionPausedError:
            st.warning(
                "Runtime maintenance is in progress. Try analysis again shortly."
            )
        else:
            st.session_state["analysis_job_id"] = job_id
            st.session_state.pop("analysis_last_result", None)
            st.session_state.pop("analysis_last_event", None)
            st.session_state.pop(_ANALYSIS_COMPLETED_JOB_KEY, None)
            st.session_state.pop(_ANALYSIS_TERMINAL_NOTICE_KEY, None)
            st.toast("Analysis started", icon="⏳")


@st.fragment(run_every=float(settings.ui.progress_poll_interval_sec))
def _render_analysis_job_panel(*, owner_id: str) -> None:
    """Render analysis job progress and stash results when complete.

    Args:
        owner_id: The unique identifier for the job owner.
    """
    job_id = st.session_state.get("analysis_job_id")
    if not isinstance(job_id, str) or not job_id:
        return

    job_manager = get_job_manager()
    state = job_manager.get(job_id, owner_id=owner_id)
    if state is None:
        _clear_analysis_job_tracking()
        st.rerun(scope="app")
        return

    if st.session_state.get(_ANALYSIS_COMPLETED_JOB_KEY) == job_id:
        if job_manager.consume_terminal(job_id, owner_id=owner_id):
            _clear_analysis_job_tracking()
            st.rerun(scope="app")
        else:
            logger.warning(
                "Terminal analysis job consumption deferred; retaining job state"
            )
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

    if state.status in ("queued", "running"):
        cancellation_requested = (
            st.session_state.get("analysis_cancel_requested_id") == job_id
        )
        if st.button(
            "Cancel analysis",
            key="analysis_cancel_inline",
            disabled=cancellation_requested,
        ) and job_manager.cancel(job_id, owner_id=owner_id):
            st.session_state["analysis_cancel_requested_id"] = job_id
            cancellation_requested = True
        if cancellation_requested:
            st.warning("Cancellation requested. Waiting for a safe stopping point.")
        return

    succeeded = state.status == "succeeded"
    result_is_analysis = isinstance(state.result, AnalysisResult)
    if succeeded and result_is_analysis:
        st.session_state["analysis_last_result"] = state.result
        notice = {"status": "succeeded", "message": "Analysis completed."}
    elif state.status == "canceled":
        notice = {"status": "canceled", "message": "Analysis canceled."}
    else:
        notice = {"status": "failed", "message": "Analysis failed."}
    st.session_state[_ANALYSIS_TERMINAL_NOTICE_KEY] = notice
    st.session_state[_ANALYSIS_COMPLETED_JOB_KEY] = job_id
    if job_manager.consume_terminal(job_id, owner_id=owner_id):
        _clear_analysis_job_tracking()
    else:
        logger.warning(
            "Terminal analysis job consumption deferred; retaining job state"
        )
    st.rerun(scope="app")


def _clear_analysis_job_tracking() -> None:
    """Clear session-local polling state after manager ownership is released."""
    st.session_state.pop("analysis_job_id", None)
    st.session_state.pop("analysis_cancel_requested_id", None)
    st.session_state.pop("analysis_last_event", None)


def _render_analysis_terminal_notice() -> None:
    """Render one durable analysis outcome exactly once."""
    notice = st.session_state.pop(_ANALYSIS_TERMINAL_NOTICE_KEY, None)
    if not isinstance(notice, dict):
        return
    status = notice.get("status")
    message = notice.get("message")
    if not isinstance(message, str):
        return
    if status == "succeeded":
        st.success(message)
    elif status == "canceled":
        st.warning(message)
    elif status == "failed":
        st.error(message)


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


def _close_image_siglip_retriever(retriever: ImageSiglipRetriever) -> None:
    """Close one cached visual retriever and its owned workers/clients."""
    with contextlib.suppress(Exception):
        retriever.close()


@st.cache_resource(on_release=_close_image_siglip_retriever)
def _get_image_siglip_retriever(
    collection_name: str, cache_version: int
) -> ImageSiglipRetriever:
    """Get cached ImageSiglipRetriever to avoid reconnection overhead."""
    from src.retrieval.multimodal_fusion import ImageSearchParams, ImageSiglipRetriever

    del cache_version
    return ImageSiglipRetriever(ImageSearchParams(collection=collection_name, top_k=10))


def _query_visual_search(upload: Any, top_k: int) -> list[Any]:
    """Execute a SigLIP-powered visual search against Qdrant.

    Args:
        upload: The uploaded image file.
        top_k: Number of results to retrieve.

    Returns:
        list[Any]: List of retrieved image nodes from Qdrant.
    """
    try:
        from src.utils.images import open_untrusted_image
    except ImportError:
        st.warning("Pillow is required for visual search.")
        return []

    try:
        img = open_untrusted_image(upload)
    except ImportError:
        st.warning("Pillow is required for visual search.")
        return []
    except Exception:
        st.warning("Uploaded image could not be opened safely.")
        return []

    with get_job_manager().foreground_runtime_activity():
        collections = st.session_state.get("_snapshot_collections")
        image_collection = (
            collections.get("image") if isinstance(collections, dict) else None
        )
        if not isinstance(image_collection, str) or not image_collection:
            raise RuntimeError("No activated image collection is available")
        retriever = _get_image_siglip_retriever(
            image_collection,
            settings.cache_version,
        )
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
    except JobAdmissionPausedError:
        st.warning("Visual search is unavailable during runtime maintenance.")
    except Exception as exc:  # pragma: no cover - UI best-effort
        st.caption(f"Visual search unavailable: {type(exc).__name__}")


def _snapshot_runtime_present() -> bool:
    """Return whether session state owns any snapshot runtime state."""
    return any(
        st.session_state.get(key) is not None
        for key in (
            "_vector_index_resource",
            "vector_index",
            "router_engine",
            "graphrag_index",
            "_snapshot_loaded_id",
            "_snapshot_collections",
        )
    )


def _snapshot_runtime_is_current(snapshot_id: str) -> bool:
    """Return whether the full snapshot runtime is current and queryable."""
    return (
        st.session_state.get("_snapshot_loaded_id") == snapshot_id
        and session_router_is_current(
            st.session_state,
            runtime_generation=settings.cache_version,
        )
        and session_vector_resource_is_current(
            st.session_state,
            runtime_generation=settings.cache_version,
        )
    )


def _snapshot_autoload_action(status: _SnapshotStatus) -> tuple[str, Path | None]:
    """Plan a no-op, clear, or hydrate action without mutating runtime state."""
    policy = getattr(settings.graphrag_cfg, "autoload_policy", "latest_non_stale")
    snap_dir = status.snapshot_dir
    if status.error or snap_dir is None or not status.manifest_present:
        return ("clear", None) if _snapshot_runtime_present() else ("none", None)
    if status.is_stale:
        return ("clear", None) if _snapshot_runtime_present() else ("none", None)

    if policy == "ignore":
        loaded_id = st.session_state.get("_snapshot_loaded_id")
        should_clear = (
            _snapshot_runtime_present()
            and not session_router_is_current(
                st.session_state,
                runtime_generation=settings.cache_version,
            )
        ) or (loaded_id is not None and not _snapshot_runtime_is_current(snap_dir.name))
        return ("clear" if should_clear else "none"), None

    if _snapshot_runtime_is_current(snap_dir.name):
        return "none", None
    return "hydrate", snap_dir


def _load_latest_snapshot_into_session(status: _SnapshotStatus) -> bool:
    """Autoload the latest snapshot into session_state per policy.

    Policies:
    - latest_non_stale (default): Load when manifest not stale.
    - ignore: Do not autohydrate, but invalidate stale snapshot-owned runtime.
    """
    action, snap_dir = _snapshot_autoload_action(status)
    if action == "none":
        return True
    try:
        with get_job_manager().admission_quiescence():
            # The first read keeps the common current-runtime path lease-free. The
            # second makes the check and mutation one quiesced transaction.
            try:
                fresh_status = _compute_snapshot_status()
                action, snap_dir = _snapshot_autoload_action(fresh_status)
                if action == "clear":
                    _clear_snapshot_runtime_quiesced()
                elif action == "hydrate" and snap_dir is not None:
                    _hydrate_router_from_snapshot_quiesced(snap_dir)
            except (
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
            ):  # pragma: no cover - defensive
                _clear_snapshot_runtime_quiesced()
    except (JobAdmissionPausedError, JobConflictError) as exc:
        _report_snapshot_refresh_conflict(exc)
        return False
    return True


def _hydrate_router_from_snapshot(snap_dir: Path) -> bool:
    """Hydrate the router engine and indices from a snapshot directory.

    Args:
        snap_dir: Path to the snapshot storage directory.
    """
    current_id = str(getattr(snap_dir, "name", snap_dir))
    if _snapshot_runtime_is_current(current_id):
        return True
    try:
        with get_job_manager().admission_quiescence():
            if not _snapshot_runtime_is_current(current_id):
                _hydrate_router_from_snapshot_quiesced(snap_dir)
    except (JobAdmissionPausedError, JobConflictError) as exc:
        _report_snapshot_refresh_conflict(exc)
        return False
    return True


def _hydrate_router_from_snapshot_quiesced(snap_dir: Path) -> None:
    """Hydrate and publish a snapshot while runtime quiescence is held."""
    from src.persistence.snapshot import (
        load_manifest,
        load_property_graph_index,
        load_vector_index,
    )
    from src.retrieval.router_factory import build_router_engine

    current_id = str(getattr(snap_dir, "name", snap_dir))
    if _snapshot_runtime_is_current(current_id):
        return

    resource: VectorIndexResource | None = None
    try:
        manifest = load_manifest(snap_dir)
        if not manifest:
            raise RuntimeError("Activated snapshot manifest is unavailable")
        collections = manifest.get("collections")
        if not isinstance(collections, dict):
            raise RuntimeError("Activated snapshot has no collection identities")
        text_collection = collections.get("text")
        image_collection = collections.get("image")
        if not isinstance(text_collection, str) or not isinstance(
            image_collection, str
        ):
            raise RuntimeError("Activated snapshot collection identities are invalid")
        vec = load_vector_index(snap_dir)
        if vec is None:
            raise RuntimeError("Activated snapshot has no vector index")
        vector_store = getattr(vec, "vector_store", None)
        if vector_store is None:
            raise TypeError("Activated vector index does not expose its backing store")
        resource = VectorIndexResource.from_vector_store(vec, vector_store)
        kg = load_property_graph_index(snap_dir)
        if manifest.get("graph_store_type") == "property_graph" and kg is None:
            raise RuntimeError("Activated snapshot property graph is unavailable")
    except Exception:
        if resource is not None:
            resource.close()
        _clear_snapshot_runtime_quiesced()
        raise
    try:
        router = build_router_engine(
            vec,
            kg,
            settings,
            text_collection=text_collection,
            image_collection=image_collection,
        )
        state_updates: dict[str, Any] = {
            "_snapshot_loaded_id": current_id,
            "_snapshot_collections": dict(collections),
        }
        state_removals: list[str] = []
        if kg is not None:
            state_updates["graphrag_index"] = kg
        else:
            state_removals.append("graphrag_index")
        replace_session_runtime(
            st.session_state,
            resource,
            router,
            runtime_generation=settings.cache_version,
            state_updates=state_updates,
            state_removals=state_removals,
        )
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
        if resource is not None:
            resource.close()
        redaction = build_pii_log_entry(str(exc), key_id="chat.router_from_snapshot")
        logger.debug(
            "Failed to build router from snapshot; clearing snapshot runtime "
            "(error_type={}, error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        _clear_snapshot_runtime_quiesced()
        raise RuntimeError("Activated snapshot router construction failed") from exc


if __name__ == "__main__":  # pragma: no cover
    main()
