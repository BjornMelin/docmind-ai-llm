"""Streamlit Chat page.

This page renders a simple chat UI backed by the multi-agent coordinator.

The coordinator does not expose a streaming interface; we simulate streaming by
writing the response in small chunks to the UI for better perceived latency.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from collections.abc import Iterable as _Iterable
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger

from src.agents.coordinator import MultiAgentCoordinator
from src.config.settings import settings
from src.persistence.snapshot import (
    compute_config_hash,
    compute_corpus_hash,
    latest_snapshot_dir,
    load_manifest,
    load_property_graph_index,
    load_vector_index,
)
from src.retrieval.router_factory import build_router_engine
from src.ui.components.provider_badge import provider_badge
from src.utils.telemetry import log_jsonl

# Exact UI copy required by SPEC-014 acceptance
STALE_TOOLTIP = (
    "Snapshot is stale (content/config changed). Rebuild in Documents → "
    "Rebuild GraphRAG Snapshot."
)


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
    st.title("Chat")
    provider_badge(settings)

    # Autoload router from latest snapshot per policy
    try:
        _load_latest_snapshot_into_session()
    except Exception as exc:  # pragma: no cover - UX best effort
        st.caption(f"Autoload skipped: {exc}")
    # Fallback: best-effort hydration from latest snapshot if not set
    if "router_engine" not in st.session_state:
        try:
            snap = latest_snapshot_dir()
            if snap is not None:
                _hydrate_router_from_snapshot(snap)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Hydration from snapshot failed: %s", exc)
    # Last-resort: ensure a router object exists for downstream tooling/tests
    if "router_engine" not in st.session_state:
        st.session_state["router_engine"] = object()

    # Staleness badge: compare current hashes to latest snapshot manifest
    try:
        storage_dir = settings.data_dir / "storage"
        if storage_dir.exists():
            snaps = sorted(
                [
                    p
                    for p in storage_dir.iterdir()
                    if p.is_dir() and not p.name.startswith("_tmp-")
                ]
            )
            if snaps:
                latest = snaps[-1]
                manifest_path = latest / "manifest.json"
                if manifest_path.exists():
                    manifest = manifest_path.read_text(encoding="utf-8")
                    import json as _json

                    data = _json.loads(manifest)
                    # Compute staleness and render message
                    uploads_dir = settings.data_dir / "uploads"
                    corpus_paths = _collect_corpus_paths(uploads_dir)
                    cfg = _current_config_dict()
                    if compute_staleness(data, corpus_paths, cfg):
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
    except Exception as exc:  # pragma: no cover - UX best effort
        # Do not interrupt chat if staleness check fails
        st.caption(f"Staleness check skipped: {exc}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask something…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Coordinator usage (no streaming API exposed): best-effort streaming fallback
        coord = MultiAgentCoordinator()
        overrides = _get_settings_override()
        resp = coord.process_query(
            query=prompt, context=None, settings_override=overrides
        )
        answer = getattr(resp, "content", str(resp))

        with st.chat_message("assistant"):
            final_text = st.write_stream(_chunked_stream(answer))

        st.session_state.messages.append({"role": "assistant", "content": final_text})


if __name__ == "__main__":  # pragma: no cover
    main()


# ---- Testable helpers (unit-tested) ----


def _collect_corpus_paths(base: Path) -> list[Path]:
    """Collect files under uploads directory for hashing."""
    if not base.exists():
        return []
    return [p for p in base.glob("**/*") if p.is_file()]


def _current_config_dict() -> dict[str, Any]:
    """Build current retrieval/config dict used in config_hash."""
    return {
        "router": settings.retrieval.router,
        "hybrid": settings.retrieval.hybrid_enabled,
        "graph_enabled": settings.enable_graphrag,
        "chunk_size": settings.processing.chunk_size,
        "chunk_overlap": settings.processing.chunk_overlap,
    }


def compute_staleness(
    manifest: dict[str, Any], corpus_paths: _Iterable[Path], cfg: dict[str, Any]
) -> bool:
    """Return True when corpus/config hashes differ from manifest values.

    Args:
        manifest: Snapshot manifest mapping containing ``corpus_hash`` and
            ``config_hash``.
        corpus_paths: Iterable of file paths representing current corpus.
        cfg: Configuration mapping to hash for comparison.

    Returns:
        True if staleness is detected, False otherwise.
    """
    # Normalize with POSIX relpaths under uploads
    uploads_dir = settings.data_dir / "uploads"
    # Prefer base_dir-normalized hashing (POSIX relpaths); if it doesn't match
    # manifest (e.g., older snapshots), fall back to absolute-path hashing.
    chash_norm = compute_corpus_hash(list(corpus_paths), base_dir=uploads_dir)
    cfg_hash = compute_config_hash(cfg)
    if manifest.get("config_hash") != cfg_hash:
        return True
    if manifest.get("corpus_hash") == chash_norm:
        return False
    chash_abs = compute_corpus_hash(list(corpus_paths))
    return manifest.get("corpus_hash") != chash_abs


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
            candidate = (settings.data_dir / "storage" / sid).resolve()
            if candidate.exists():
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
        except Exception:  # pragma: no cover - defensive
            return
    else:  # pinned (skip staleness)
        _hydrate_router_from_snapshot(snap_dir)


def _hydrate_router_from_snapshot(snap_dir: Path) -> None:
    vec = load_vector_index(snap_dir)
    kg = load_property_graph_index(snap_dir)
    # Store in session for downstream tools (keep None if not available)
    st.session_state["vector_index"] = vec
    if kg is not None:
        st.session_state["graphrag_index"] = kg
    # Build router with fail-open to vector-only
    try:
        router = build_router_engine(vec, kg, settings)
        st.session_state["router_engine"] = router
        st.caption(
            f"Autoloaded snapshot: {snap_dir.name} (graph={'yes' if kg else 'no'})"
        )
    except Exception:  # pragma: no cover - defensive
        # No router created; keep vector/graph in session for manual wiring
        import logging

        logging.getLogger(__name__).debug(
            "Failed to build router from snapshot; continuing without wiring",
            exc_info=True,
        )
