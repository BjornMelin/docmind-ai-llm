"""Unit tests for Chat page helpers (01_chat.py)."""

from __future__ import annotations

import contextlib
import sqlite3
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def clean_streamlit_session(monkeypatch):
    # Ensure streamlit import has expected attributes for testing
    import streamlit as st  # type: ignore

    # Reset session_state and capture captions per test
    st.session_state.clear()
    captures: dict[str, list[str] | int] = {
        "captions": [],
        "warnings": [],
        "markdown": [],
        "images": [],
        "chat_roles": [],
        "write_stream": [],
        "reruns": 0,
    }

    def _chat_message(role: str):  # type: ignore[no-untyped-def]
        captures["chat_roles"].append(str(role))  # type: ignore[index]
        return contextlib.nullcontext()

    monkeypatch.setattr(
        st,
        "caption",
        lambda msg: captures["captions"].append(str(msg)),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(
        st,
        "warning",
        lambda msg: captures["warnings"].append(str(msg)),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(
        st,
        "markdown",
        lambda msg: captures["markdown"].append(str(msg)),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(
        st,
        "image",
        lambda img, **_k: captures["images"].append(str(img)),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(
        st, "expander", lambda *_a, **_k: contextlib.nullcontext(), raising=False
    )
    monkeypatch.setattr(st, "sidebar", contextlib.nullcontext(), raising=False)
    monkeypatch.setattr(st, "subheader", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "slider", lambda *_a, **_k: 1, raising=False)
    monkeypatch.setattr(
        st,
        "columns",
        lambda n: [SimpleNamespace() for _ in range(int(n))],
        raising=False,
    )
    monkeypatch.setattr(st, "chat_message", _chat_message, raising=False)
    monkeypatch.setattr(st, "chat_input", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        st,
        "write_stream",
        lambda gen: captures["write_stream"].append("".join(list(gen))),  # type: ignore[index]
        raising=False,
    )
    monkeypatch.setattr(st, "write", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "selectbox", lambda *_a, **_k: "session", raising=False)
    monkeypatch.setattr(st, "text_input", lambda *_a, **_k: "", raising=False)
    monkeypatch.setattr(st, "button", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        st,
        "rerun",
        lambda: captures.__setitem__("reruns", int(captures["reruns"]) + 1),
        raising=False,
    )
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


@pytest.mark.unit
def test_close_helpers_cover_suppress_paths():
    import importlib

    page = importlib.import_module("src.pages.01_chat")

    class _Saver:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    class _Conn:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    saver = _Saver()
    conn = _Conn()
    page._close_checkpointer(saver, conn)
    assert saver.closed == 1
    assert conn.closed == 1

    class _Store:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    store = _Store()
    page._close_memory_store(store)
    assert store.closed == 1


@pytest.mark.unit
def test_set_last_sources_for_render_sets_session_state():
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")
    st.session_state.clear()
    sources = [{"content": "c", "metadata": {"doc_id": "d"}}]
    page._set_last_sources_for_render(sources, thread_id="t1")
    assert st.session_state["active_thread_id"] == "t1"
    assert st.session_state["last_sources"] == sources


@pytest.mark.unit
def test_render_sources_fragment_renders_text_and_image(
    monkeypatch, clean_streamlit_session
):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    class _Store:
        def resolve_path(self, _ref):  # type: ignore[no-untyped-def]
            return Path("img.png")

    monkeypatch.setattr(page.ArtifactStore, "from_settings", lambda _s: _Store())
    monkeypatch.setattr(st, "slider", lambda *_a, **_k: 2, raising=False)

    st.session_state["active_thread_id"] = "t1"
    st.session_state["last_sources"] = [
        {
            "metadata": {
                "modality": "pdf_page_image",
                "doc_id": "d1",
                "page_no": 1,
                "thumbnail_artifact_id": "a",
                "thumbnail_artifact_suffix": ".png",
            }
        },
        {"content": "hello", "metadata": {"doc_id": "d1"}},
    ]

    page._render_sources_fragment.__wrapped__()  # type: ignore[attr-defined]
    assert clean_streamlit_session["images"]
    assert clean_streamlit_session["markdown"]


@pytest.mark.unit
def test_render_memory_sidebar_puts_and_reruns(monkeypatch, clean_streamlit_session):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    seen: dict[str, object] = {}

    class _Store:
        def put(self, ns, key, value, index=None):  # type: ignore[no-untyped-def]
            seen["put"] = (ns, key, value, index)

        def search(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return []

        def delete(self, *_a, **_k):  # type: ignore[no-untyped-def]
            raise AssertionError("delete should not be called")

    monkeypatch.setattr(page, "_get_memory_store", lambda: _Store())

    def _text_input(label, key=None, **_k):  # type: ignore[no-untyped-def]
        if key == "memory_add":
            return "remember me"
        if key == "memory_search":
            return ""
        return ""

    def _button(label, key=None, **_k):  # type: ignore[no-untyped-def]
        return key == "memory_save"

    monkeypatch.setattr(st, "text_input", _text_input, raising=False)
    monkeypatch.setattr(st, "button", _button, raising=False)

    page._render_memory_sidebar("u1", "t1")
    assert "put" in seen
    assert st.session_state["_memory_last_saved"] == "remember me"
    assert st.session_state["memory_add"] == ""
    assert int(clean_streamlit_session["reruns"]) == 1


@pytest.mark.unit
def test_ensure_router_engine_sets_none_and_hydrates(
    monkeypatch, clean_streamlit_session
):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    st.session_state.clear()
    monkeypatch.setattr(
        page,
        "_load_latest_snapshot_into_session",
        lambda: (_ for _ in ()).throw(RuntimeError("x")),
    )
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda *_a, **_k: None)
    page._ensure_router_engine()
    assert st.session_state.get("router_engine") is None
    assert any("Autoload skipped" in c for c in clean_streamlit_session["captions"])

    st.session_state.clear()
    snap = Path("/tmp/snap")
    monkeypatch.setattr(page, "_load_latest_snapshot_into_session", lambda: None)
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda *_a, **_k: snap)
    monkeypatch.setattr(
        page,
        "_hydrate_router_from_snapshot",
        lambda _p: st.session_state.__setitem__("router_engine", "R"),
    )
    page._ensure_router_engine()
    assert st.session_state.get("router_engine") == "R"


@pytest.mark.unit
def test_render_staleness_badge_warns_and_logs(
    monkeypatch, tmp_path: Path, clean_streamlit_session
):
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.01_chat")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    storage = tmp_path / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    snap = storage / "sid"
    snap.mkdir()

    events: list[dict] = []
    monkeypatch.setattr(page, "latest_snapshot_dir", lambda *_a, **_k: snap)
    monkeypatch.setattr(
        page, "load_manifest", lambda _p: {"corpus_hash": "c", "config_hash": "d"}
    )
    monkeypatch.setattr(page, "_collect_corpus_paths", lambda _p: [])
    monkeypatch.setattr(page, "_current_config_dict", lambda: {})
    monkeypatch.setattr(page, "compute_staleness", lambda *_a, **_k: True)
    monkeypatch.setattr(page, "log_jsonl", lambda ev: events.append(ev))

    page._render_staleness_badge()
    assert page.STALE_TOOLTIP in clean_streamlit_session["warnings"]
    assert events
    assert events[-1].get("snapshot_stale_detected") is True


@pytest.mark.unit
def test_render_chat_history_and_handle_prompt(monkeypatch):
    import importlib

    import streamlit as st  # type: ignore
    from langchain_core.messages import AIMessage, HumanMessage

    page = importlib.import_module("src.pages.01_chat")

    roles: list[str] = []

    def _chat_message(role: str):  # type: ignore[no-untyped-def]
        roles.append(str(role))
        return contextlib.nullcontext()

    monkeypatch.setattr(st, "chat_message", _chat_message, raising=False)
    monkeypatch.setattr(st, "markdown", lambda *_a, **_k: None, raising=False)
    page._render_chat_history([
        HumanMessage(content="hi"),
        AIMessage(content="yo"),
        object(),
    ])
    assert roles == ["user", "assistant"]

    # Prompt flow
    monkeypatch.setattr(st, "chat_input", lambda *_a, **_k: "Q", raising=False)

    touched: list[tuple] = []
    monkeypatch.setattr(
        page, "touch_session", lambda *_a, **_k: touched.append((_a, _k))
    )
    monkeypatch.setattr(page, "_render_sources_fragment", lambda: None)

    class _Coord:
        def process_query(self, **_k):  # type: ignore[no-untyped-def]
            return SimpleNamespace(content="A", sources=[{"content": "src"}])

        def list_checkpoints(self, **_k):  # type: ignore[no-untyped-def]
            return [{"checkpoint_id": "cp"}]

    conn = sqlite3.connect(":memory:")
    try:
        selection = SimpleNamespace(
            thread_id="t", user_id="u", resume_checkpoint_id=None
        )
        page._handle_chat_prompt(_Coord(), selection, conn)
    finally:
        conn.close()
    assert touched


@pytest.mark.unit
def test_visual_search_helpers(monkeypatch, clean_streamlit_session):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    # No upload → returns early
    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: None, raising=False)
    up, top_k, run = page._visual_search_inputs()
    assert up is None
    assert top_k == 0
    assert run is False

    # Upload path
    upload = object()

    class _Col:
        def number_input(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return 5

        def button(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return True

    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: upload, raising=False)
    monkeypatch.setattr(st, "columns", lambda _n: [_Col(), _Col()], raising=False)
    up, top_k, run = page._visual_search_inputs()
    assert up is upload
    assert top_k == 5
    assert run is True

    # Query visual search with stubbed dependencies
    pil = ModuleType("PIL")
    pil_image = ModuleType("PIL.Image")

    class _Image:
        @staticmethod
        def open(_f):  # type: ignore[no-untyped-def]
            return object()

    pil.Image = _Image  # type: ignore[attr-defined]
    pil_image.open = _Image.open  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", pil_image)

    mm = importlib.import_module("src.retrieval.multimodal_fusion")

    class _Retriever:
        def __init__(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return None

        def retrieve_by_image(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return [
                SimpleNamespace(
                    node=SimpleNamespace(
                        metadata={
                            "doc_id": "d",
                            "page_no": 1,
                            "thumbnail_artifact_id": "a",
                            "thumbnail_artifact_suffix": ".png",
                        }
                    )
                )
            ]

        def close(self) -> None:
            return None

    monkeypatch.setattr(mm, "ImageSiglipRetriever", _Retriever)

    pts = page._query_visual_search(object(), top_k=1)
    assert isinstance(pts, list)
    assert pts

    class _Store:
        def resolve_path(self, _ref):  # type: ignore[no-untyped-def]
            return Path("img.png")

    monkeypatch.setattr(page.ArtifactStore, "from_settings", lambda _s: _Store())
    page._render_visual_results(pts, top_k=1)
    assert clean_streamlit_session["images"]

    called: list[tuple] = []
    monkeypatch.setattr(page, "_visual_search_inputs", lambda: (upload, 1, True))
    monkeypatch.setattr(page, "_query_visual_search", lambda *_a, **_k: pts)
    monkeypatch.setattr(
        page, "_render_visual_results", lambda *_a, **_k: called.append((_a, _k))
    )
    page._render_visual_search_sidebar()
    assert called


@pytest.mark.unit
def test_load_chat_messages_handles_exception() -> None:
    import importlib

    page = importlib.import_module("src.pages.01_chat")

    class _Coord:
        def get_state_values(self, **_k):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    selection = SimpleNamespace(thread_id="t", user_id="u")
    assert page._load_chat_messages(_Coord(), selection) == []


@pytest.mark.unit
def test_render_memory_sidebar_delete_flow(monkeypatch, clean_streamlit_session):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    class _Item:
        def __init__(self) -> None:
            self.key = "k1"
            self.value = {"content": "hello"}
            self.score = 0.5

    deleted: list[tuple] = []

    class _Store:
        def search(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return [_Item()]

        def delete(self, ns, key):  # type: ignore[no-untyped-def]
            deleted.append((ns, key))

    monkeypatch.setattr(page, "_get_memory_store", lambda: _Store())
    monkeypatch.setattr(st, "text_input", lambda *_a, **_k: "", raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(
        st,
        "button",
        lambda label, **_k: str(label) == "Delete",
        raising=False,
    )

    page._render_memory_sidebar("u1", "t1")
    assert deleted == [(("memories", "u1", "t1"), "k1")]
    assert int(clean_streamlit_session["reruns"]) == 1


@pytest.mark.unit
def test_hydrate_router_from_snapshot_skips_when_already_loaded(monkeypatch):
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    st.session_state.clear()
    st.session_state["_snapshot_loaded_id"] = "sid"
    st.session_state["router_engine"] = object()

    monkeypatch.setattr(
        page,
        "load_vector_index",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not load")),
    )
    page._hydrate_router_from_snapshot(Path("sid"))


@pytest.mark.unit
def test_hydrate_router_from_snapshot_closes_old_retriever(monkeypatch):
    import importlib
    import sys
    from types import ModuleType

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.01_chat")

    class _Old:
        def __init__(self) -> None:
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    old = _Old()
    st.session_state.clear()
    st.session_state["hybrid_retriever"] = old

    monkeypatch.setattr(page, "load_vector_index", lambda *_a, **_k: "V")
    monkeypatch.setattr(page, "load_property_graph_index", lambda *_a, **_k: None)
    monkeypatch.setattr(page, "build_router_engine", lambda *_a, **_k: "R")

    mm = ModuleType("src.retrieval.multimodal_fusion")

    class _MM:
        def close(self) -> None:
            return None

    mm.MultimodalFusionRetriever = _MM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.retrieval.multimodal_fusion", mm)

    page._hydrate_router_from_snapshot(Path("sid"))
    assert old.closed == 1
    assert "graphrag_index" not in st.session_state


@pytest.mark.unit
def test_load_latest_snapshot_pinned_without_sid_returns(monkeypatch):
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.01_chat")
    monkeypatch.setattr(
        settings.graphrag_cfg, "autoload_policy", "pinned", raising=False
    )
    monkeypatch.setattr(
        settings.graphrag_cfg, "pinned_snapshot_id", None, raising=False
    )
    page._load_latest_snapshot_into_session()
