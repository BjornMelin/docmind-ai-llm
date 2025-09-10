"""Streamlit Chat page.

This page renders a simple chat UI backed by the multi-agent coordinator.

The coordinator does not expose a streaming interface; we simulate streaming by
writing the response in small chunks to the UI for better perceived latency.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterable as _Iterable
from pathlib import Path
from typing import Any

import streamlit as st

from src.agents.coordinator import MultiAgentCoordinator
from src.config.settings import settings
from src.persistence.snapshot import compute_config_hash, compute_corpus_hash
from src.ui.components.provider_badge import provider_badge


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
                        st.warning(
                            "Snapshot is stale (content/config changed). "
                            "Open Documents -> 'Rebuild GraphRAG Snapshot' to refresh."
                        )
                    else:
                        st.caption(f"Snapshot up-to-date: {latest.name}")
    except Exception as exc:  # pragma: no cover - UX best effort
        # Do not interrupt chat if staleness check fails
        st.debug(f"Staleness check skipped: {exc}")

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
    """Return True when corpus/config hashes differ from manifest values."""
    chash = compute_corpus_hash(list(corpus_paths))
    cfg_hash = compute_config_hash(cfg)
    return (
        manifest.get("corpus_hash") != chash or manifest.get("config_hash") != cfg_hash
    )
