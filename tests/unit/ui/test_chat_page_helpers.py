"""Unit tests for Chat page helpers (01_chat.py)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_streamlit_session(monkeypatch):
    # Ensure streamlit import has expected attributes for testing
    import streamlit as st  # type: ignore

    # Reset session_state and capture captions per test
    st.session_state.clear()
    captures = {"captions": []}
    monkeypatch.setattr(st, "caption", lambda msg: captures["captions"].append(msg))
    return captures


@pytest.mark.unit
def test_chunked_stream_edges():
    import importlib

    page = importlib.import_module("src.pages.01_chat")

    # Empty
    assert list(page._chunked_stream("")) == []
    # Exact multiple
    s = "a" * 96
    out = list(page._chunked_stream(s, chunk_size=48))
    assert len(out) == 2
    assert "".join(out) == s
    # Remainder
    s = "abcde"
    out = list(page._chunked_stream(s, chunk_size=2))
    assert out == ["ab", "cd", "e"]


@pytest.mark.unit
def test_get_settings_override_forwarding(monkeypatch):
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    import streamlit as st  # type: ignore

    st.session_state.clear()
    assert page._get_settings_override() is None

    st.session_state["router_engine"] = object()
    ov = page._get_settings_override()
    assert ov == {"router_engine": st.session_state["router_engine"]}

    st.session_state["vector_index"] = object()
    st.session_state["hybrid_retriever"] = object()
    st.session_state["graphrag_index"] = object()
    ov = page._get_settings_override()
    assert set(ov.keys()) == {"router_engine", "vector", "retriever", "kg"}


@pytest.mark.unit
def test_hydrate_router_from_snapshot(monkeypatch):
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    import streamlit as st  # type: ignore

    # Patch loader helpers to deterministic stubs
    monkeypatch.setattr(page, "load_vector_index", lambda _p: "VEC")
    monkeypatch.setattr(page, "load_property_graph_index", lambda _p: "KG")
    monkeypatch.setattr(page, "build_router_engine", lambda *_, **__: "ROUTER")

    st.session_state.clear()
    page._hydrate_router_from_snapshot(Path("/tmp/snap"))
    assert st.session_state["vector_index"] == "VEC"
    assert st.session_state["graphrag_index"] == "KG"
    assert st.session_state["router_engine"] == "ROUTER"

    # Error path: build_router_engine raises → keep vec/kg only
    monkeypatch.setattr(
        page,
        "build_router_engine",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("x")),
    )
    st.session_state.clear()
    page._hydrate_router_from_snapshot(Path("/tmp/snap"))
    assert (
        st.session_state.get("router_engine") is None
        or "router_engine" not in st.session_state
    )
    assert st.session_state["vector_index"] == "VEC"


@pytest.mark.unit
def test_load_latest_snapshot_policies(monkeypatch, tmp_path):
    import importlib

    page = importlib.import_module("src.pages.01_chat")
    import streamlit as st  # type: ignore

    from src.config import settings

    st.session_state.clear()

    # ignore policy → no action
    monkeypatch.setattr(settings.graphrag_cfg, "autoload_policy", "ignore")
    page._load_latest_snapshot_into_session()
    assert st.session_state == {}

    # pinned policy with existing dir
    d = tmp_path / "storage" / "SID"
    d.mkdir(parents=True)
    monkeypatch.setattr(settings.graphrag_cfg, "autoload_policy", "pinned")
    monkeypatch.setattr(settings.graphrag_cfg, "pinned_snapshot_id", "SID")
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    # Ensure hydrate path callable but side effects minimal
    monkeypatch.setattr(
        page,
        "_hydrate_router_from_snapshot",
        lambda p: st.session_state.__setitem__("router_engine", "R"),
    )
    page._load_latest_snapshot_into_session()
    assert st.session_state.get("router_engine") == "R"

    # latest_non_stale path with latest_snapshot_dir returning a path
    st.session_state.clear()
    monkeypatch.setattr(settings.graphrag_cfg, "autoload_policy", "latest_non_stale")
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda: d)
    monkeypatch.setattr(page, "load_manifest", lambda _p: {"ok": True})
    monkeypatch.setattr(page, "compute_staleness", lambda *_a, **_k: False)
    monkeypatch.setattr(
        page,
        "_hydrate_router_from_snapshot",
        lambda p: st.session_state.__setitem__("router_engine", "R2"),
    )
    page._load_latest_snapshot_into_session()
    assert st.session_state.get("router_engine") == "R2"
