"""Unit tests for Documents page helper functions (02_documents.py)."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def streamlit_calls(monkeypatch):
    import streamlit as st  # type: ignore

    st.session_state.clear()

    calls: dict[str, list[str]] = {
        "writes": [],
        "infos": [],
        "warnings": [],
        "errors": [],
        "success": [],
        "captions": [],
    }

    class _Col:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_a):  # type: ignore[no-untyped-def]
            return False

        def metric(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return None

        def number_input(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return 1

        def checkbox(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return False

        def button(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return False

    class _Status:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_a):  # type: ignore[no-untyped-def]
            return False

        def update(self, **_kwargs):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr(
        st, "write", lambda s: calls["writes"].append(str(s)), raising=False
    )
    monkeypatch.setattr(
        st, "info", lambda s: calls["infos"].append(str(s)), raising=False
    )
    monkeypatch.setattr(
        st, "warning", lambda s: calls["warnings"].append(str(s)), raising=False
    )
    monkeypatch.setattr(
        st, "error", lambda s: calls["errors"].append(str(s)), raising=False
    )
    monkeypatch.setattr(
        st, "success", lambda s: calls["success"].append(str(s)), raising=False
    )
    monkeypatch.setattr(
        st, "caption", lambda s: calls["captions"].append(str(s)), raising=False
    )
    monkeypatch.setattr(st, "toast", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "json", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "markdown", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "subheader", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "divider", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "slider", lambda *_a, **_k: 1, raising=False)
    monkeypatch.setattr(
        st, "columns", lambda n: [_Col() for _ in range(int(n))], raising=False
    )
    monkeypatch.setattr(st, "button", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: False, raising=False)
    monkeypatch.setattr(st, "selectbox", lambda *_a, **_k: "", raising=False)
    monkeypatch.setattr(
        st, "form", lambda *_a, **_k: contextlib.nullcontext(), raising=False
    )
    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        st, "form_submit_button", lambda *_a, **_k: False, raising=False
    )
    monkeypatch.setattr(
        st, "expander", lambda *_a, **_k: contextlib.nullcontext(), raising=False
    )
    monkeypatch.setattr(st, "status", lambda *_a, **_k: _Status(), raising=False)
    return calls


def test_filter_group_and_page_parsers() -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")

    exports = [
        {"content_type": "image/png", "metadata": {"doc_id": "d1", "page_no": "2"}},
        {"content_type": "text/plain"},
        {"content_type": "image/jpeg", "metadata": {"document_id": "d2", "page": 1}},
        "bad",
    ]
    imgs = page._filter_image_exports(exports)  # type: ignore[attr-defined]
    assert len(imgs) == 2

    grouped = page._group_exports_by_doc(imgs)  # type: ignore[attr-defined]
    assert set(grouped.keys()) == {"d1", "d2"}
    assert page._page_no({"page_no": "2"}) == 2  # type: ignore[attr-defined]
    assert page._page_no({"page_no": "x"}) == 0  # type: ignore[attr-defined]


def test_handle_ingest_submission_no_files(monkeypatch, streamlit_calls) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    page._handle_ingest_submission(None, use_graphrag=False, encrypt_images=False)  # type: ignore[attr-defined]
    assert streamlit_calls["warnings"] == ["No files selected."]


def test_render_ingest_form_smoke(monkeypatch) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")

    files = [object()]
    monkeypatch.setattr(st, "file_uploader", lambda *_a, **_k: files, raising=False)
    monkeypatch.setattr(st, "checkbox", lambda *_a, **_k: True, raising=False)
    monkeypatch.setattr(st, "form_submit_button", lambda *_a, **_k: True, raising=False)

    out_files, use_graphrag, encrypt_images, submitted = page._render_ingest_form()  # type: ignore[attr-defined]
    assert out_files == files
    assert use_graphrag is True
    assert encrypt_images is True
    assert submitted is True


def test_handle_ingest_submission_success_calls_helpers(monkeypatch) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        page,
        "ingest_files",
        lambda *_a, **_k: {"count": 1, "vector_index": "V", "pg_index": None},
    )
    monkeypatch.setattr(
        page,
        "_render_ingest_results",
        lambda result, use_graphrag: seen.__setitem__("render", (result, use_graphrag)),
    )
    monkeypatch.setattr(
        page,
        "_handle_snapshot_rebuild",
        lambda vector_index, pg_index: seen.__setitem__(
            "snapshot", (vector_index, pg_index)
        ),
    )

    page._handle_ingest_submission([object()], use_graphrag=False, encrypt_images=False)  # type: ignore[attr-defined]
    assert "render" in seen
    assert "snapshot" in seen


def test_handle_ingest_submission_snapshot_lock(monkeypatch, streamlit_calls) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")

    monkeypatch.setattr(
        page,
        "ingest_files",
        lambda *_a, **_k: {"count": 1, "vector_index": "V", "pg_index": None},
    )
    monkeypatch.setattr(page, "_render_ingest_results", lambda *_a, **_k: None)
    monkeypatch.setattr(
        page,
        "_handle_snapshot_rebuild",
        lambda *_a, **_k: (_ for _ in ()).throw(page.SnapshotLockTimeout()),
    )

    page._handle_ingest_submission([object()], use_graphrag=False, encrypt_images=False)  # type: ignore[attr-defined]
    assert any(
        "Snapshot rebuild already in progress" in w for w in streamlit_calls["warnings"]
    )


def test_render_ingest_results_sets_session_state(monkeypatch, tmp_path: Path) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    monkeypatch.setattr(page, "_render_image_exports", lambda _e: None)
    monkeypatch.setattr(
        page,
        "_set_multimodal_retriever",
        lambda: st.session_state.__setitem__("hybrid_retriever", object()),
    )
    monkeypatch.setattr(page, "build_router_engine", lambda *_a, **_k: "R")

    st.session_state.clear()
    page._render_ingest_results(  # type: ignore[attr-defined]
        {"count": 2, "vector_index": "V", "pg_index": "G"},
        use_graphrag=True,
    )
    assert st.session_state["vector_index"] == "V"
    assert st.session_state["graphrag_index"] == "G"
    assert st.session_state["router_engine"] == "R"
    assert "hybrid_retriever" in st.session_state

    # Without GraphRAG, graphrag_index is removed
    st.session_state["graphrag_index"] = "G"
    page._render_ingest_results({"count": 1, "vector_index": "V2"}, use_graphrag=False)  # type: ignore[attr-defined]
    assert st.session_state.get("graphrag_index") is None


def test_render_image_exports_calls_renderer(monkeypatch) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    called: list[int] = []
    monkeypatch.setattr(
        page, "_render_export_images", lambda items, limit: called.append(len(items))
    )

    page._render_image_exports(  # type: ignore[attr-defined]
        [
            {"content_type": "image/png", "metadata": {"doc_id": "d1", "page_no": 1}},
            {"content_type": "image/png", "metadata": {"doc_id": "d1", "page_no": 2}},
        ]
    )
    assert called == [2]


def test_render_export_images_handles_missing_ref(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")

    class _Store:
        def resolve_path(self, _ref):  # type: ignore[no-untyped-def]
            return tmp_path / "x.png"

    monkeypatch.setattr(page.ArtifactStore, "from_settings", lambda _s: _Store())
    monkeypatch.setattr(st, "image", lambda *_a, **_k: None, raising=False)

    page._render_export_images(  # type: ignore[attr-defined]
        [{"metadata": {"doc_id": "d1", "page_no": 1}}],
        preview_limit=1,
    )
    assert any("no artifact ref" in c for c in streamlit_calls["captions"])


def test_render_maintenance_controls_no_uploads(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    page._render_maintenance_controls()  # type: ignore[attr-defined]
    assert "No uploaded files." in streamlit_calls["captions"][-1]


def test_render_maintenance_controls_triggers_handlers(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / "a.pdf"
    target.write_bytes(b"%PDF-1.4\n%fake\n")

    hits: list[str] = []
    monkeypatch.setattr(
        page, "_handle_reindex_page_images", lambda **_k: hits.append("reindex")
    )
    monkeypatch.setattr(
        page, "_handle_delete_upload", lambda **_k: hits.append("delete")
    )

    def _columns(_n: int):  # type: ignore[no-untyped-def]
        class _C:
            def __enter__(self):  # type: ignore[no-untyped-def]
                return self

            def __exit__(self, *_a):  # type: ignore[no-untyped-def]
                return False

            def number_input(self, *_a, **_k):  # type: ignore[no-untyped-def]
                return 1

            def checkbox(self, *_a, **_k):  # type: ignore[no-untyped-def]
                return False

            def button(self, label, **_k):  # type: ignore[no-untyped-def]
                return str(label) == "Reindex"

        return [_C(), _C(), _C()]

    monkeypatch.setattr(st, "columns", _columns, raising=False)
    monkeypatch.setattr(st, "selectbox", lambda *_a, **_k: target.name, raising=False)
    monkeypatch.setattr(
        st,
        "checkbox",
        lambda label, **_k: "cannot be undone" in str(label).lower(),
        raising=False,
    )
    monkeypatch.setattr(
        st, "button", lambda label, **_k: str(label) == "Delete", raising=False
    )

    page._render_maintenance_controls()  # type: ignore[attr-defined]
    assert hits == ["reindex", "delete"]


def test_sha256_for_file_and_doc_id_for_upload(monkeypatch, tmp_path: Path) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")
    stt = p.stat()
    digest = page._sha256_for_file.__wrapped__(  # type: ignore[attr-defined]
        str(p), int(stt.st_mtime_ns), int(stt.st_size)
    )
    assert isinstance(digest, str)
    assert len(digest) == 64

    # Avoid Streamlit caching wrapper in tests by calling the wrapped function.
    monkeypatch.setattr(page, "_sha256_for_file", page._sha256_for_file.__wrapped__)  # type: ignore[attr-defined]
    doc_id = page._doc_id_for_upload(p)  # type: ignore[attr-defined]
    assert doc_id.startswith("doc-")
    assert len(doc_id) == 4 + 16


def test_handle_delete_upload_refuses_outside_uploads(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    page._handle_delete_upload(target=outside, purge_artifacts=False)  # type: ignore[attr-defined]
    assert streamlit_calls["errors"][-1].startswith("Refusing to delete path")


def test_handle_delete_upload_deletes_and_reports(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / "a.txt"
    target.write_text("hello", encoding="utf-8")

    # Disable Streamlit cache wrapper inside _doc_id_for_upload
    monkeypatch.setattr(page, "_sha256_for_file", page._sha256_for_file.__wrapped__)  # type: ignore[attr-defined]

    # Stub qdrant_client module to avoid network.
    qdrant = ModuleType("qdrant_client")
    qmodels = ModuleType("qdrant_client.models")

    class _Client:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.deleted = 0

        def count(self, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(count=4)

        def delete(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.deleted += 1
            return None

        def close(self) -> None:
            return None

    class _Filter:  # minimal shape
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            return None

    class _FieldCondition:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            return None

    class _MatchValue:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            return None

    class _FilterSelector:
        def __init__(self, **_kwargs):  # type: ignore[no-untyped-def]
            return None

    qdrant.QdrantClient = _Client  # type: ignore[attr-defined]
    qmodels.Filter = _Filter  # type: ignore[attr-defined]
    qmodels.FieldCondition = _FieldCondition  # type: ignore[attr-defined]
    qmodels.MatchValue = _MatchValue  # type: ignore[attr-defined]
    qmodels.FilterSelector = _FilterSelector  # type: ignore[attr-defined]
    qdrant.models = qmodels  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", qmodels)

    image_index = ModuleType("src.retrieval.image_index")
    image_index.collect_artifact_refs_for_doc_id = lambda *_a, **_k: []  # type: ignore[attr-defined]
    image_index.count_artifact_references_in_image_collection = (  # type: ignore[attr-defined]
        lambda *_a, **_k: 0
    )
    image_index.delete_page_images_for_doc_id = lambda *_a, **_k: 2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.retrieval.image_index", image_index)

    monkeypatch.setattr("src.utils.storage.get_client_config", lambda: {})

    page._handle_delete_upload(target=target, purge_artifacts=False)  # type: ignore[attr-defined]
    assert not target.exists()
    assert streamlit_calls["success"]
    assert "image_points=2" in streamlit_calls["success"][-1]
    assert "text_points≈4" in streamlit_calls["success"][-1]


def test_render_latest_snapshot_summary(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(
        page,
        "load_manifest",
        lambda *_, **__: {
            "created_at": "now",
            "corpus_hash": "c" * 20,
            "config_hash": "d" * 20,
        },
    )
    page._render_latest_snapshot_summary()  # type: ignore[attr-defined]
    assert any("Latest snapshot" in c for c in streamlit_calls["captions"])


def test_render_manifest_details_and_log_export_event(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")

    events: list[dict] = []
    monkeypatch.setattr("src.utils.telemetry.log_jsonl", lambda ev: events.append(ev))

    page._render_manifest_details(  # type: ignore[attr-defined]
        {
            "corpus_hash": "c" * 20,
            "config_hash": "d" * 20,
            "versions": {"app": "x"},
            "graph_exports": [
                {"filename": "g.jsonl", "format": "jsonl", "size_bytes": 1}
            ],
        },
        snapshot_dir=tmp_path,
    )
    page._log_export_event({
        "export_performed": True,
        "dest_path": str(tmp_path / "a/b/c.txt"),
    })  # type: ignore[attr-defined]
    assert events
    assert "dest_path" not in events[-1]
    assert events[-1]["dest_basename"] == "c.txt"


def test_handle_manual_export_smoke(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    st.session_state["graphrag_index"] = object()
    st.session_state["vector_index"] = object()

    monkeypatch.setattr(page, "get_export_seed_ids", lambda *_a, **_k: ["0", "1"])
    out_file = tmp_path / "out.jsonl"
    monkeypatch.setattr(page, "timestamped_export_path", lambda _d, _e: out_file)
    monkeypatch.setattr(
        page,
        "export_graph_jsonl",
        lambda **_k: out_file.write_text("x", encoding="utf-8"),
    )
    monkeypatch.setattr(
        page,
        "export_graph_parquet",
        lambda **_k: out_file.write_text("x", encoding="utf-8"),
    )
    monkeypatch.setattr(page, "record_graph_export_metric", lambda *_a, **_k: None)
    monkeypatch.setattr(page, "_log_export_event", lambda _p: None)

    page._handle_manual_export(tmp_path, "jsonl")  # type: ignore[attr-defined]
    assert streamlit_calls["success"]


def test_create_vector_index_fallback(monkeypatch) -> None:
    import importlib
    import sys
    from types import ModuleType

    page = importlib.import_module("src.pages.02_documents")

    li_core = ModuleType("llama_index.core")

    class _VectorStoreIndex:
        @classmethod
        def from_vector_store(cls, store):  # type: ignore[no-untyped-def]
            return ("IDX", store)

    li_core.VectorStoreIndex = _VectorStoreIndex  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_index.core", li_core)
    monkeypatch.setattr(page, "create_vector_store", lambda *_a, **_k: "STORE")

    out = page._create_vector_index_fallback()  # type: ignore[attr-defined]
    assert out == ("IDX", "STORE")


def test_render_ingest_results_uses_vector_index_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(page, "_render_image_exports", lambda _e: None)
    monkeypatch.setattr(page, "_create_vector_index_fallback", lambda: "V")
    monkeypatch.setattr(page, "_set_multimodal_retriever", lambda: None)
    monkeypatch.setattr(page, "build_router_engine", lambda *_a, **_k: "R")

    st.session_state.clear()
    page._render_ingest_results({"count": 1, "vector_index": None}, use_graphrag=False)  # type: ignore[attr-defined]
    assert st.session_state["vector_index"] == "V"
    assert st.session_state["router_engine"] == "R"


def test_set_multimodal_retriever_sets_session_state(monkeypatch) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")

    mm = ModuleType("src.retrieval.multimodal_fusion")

    class _MM:
        pass

    mm.MultimodalFusionRetriever = _MM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.retrieval.multimodal_fusion", mm)

    st.session_state.clear()
    page._set_multimodal_retriever()  # type: ignore[attr-defined]
    assert isinstance(st.session_state.get("hybrid_retriever"), _MM)


def test_handle_snapshot_rebuild_snapshot_lock(monkeypatch, streamlit_calls) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(
        page,
        "rebuild_snapshot",
        lambda *_a, **_k: (_ for _ in ()).throw(page.SnapshotLockTimeout()),
    )
    page._handle_snapshot_rebuild(vector_index=object(), pg_index=None)  # type: ignore[attr-defined]
    assert any("already in progress" in w for w in streamlit_calls["warnings"])


def test_render_export_images_handles_encrypted_without_support(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    page = importlib.import_module("src.pages.02_documents")

    class _Store:
        def resolve_path(self, _ref):  # type: ignore[no-untyped-def]
            return tmp_path / "x.webp.enc"

    monkeypatch.setattr(page.ArtifactStore, "from_settings", lambda _s: _Store())
    monkeypatch.setattr(st, "image", lambda *_a, **_k: None, raising=False)

    images_mod = ModuleType("src.utils.images")
    monkeypatch.setitem(sys.modules, "src.utils.images", images_mod)

    page._render_export_images(  # type: ignore[attr-defined]
        [
            {
                "metadata": {
                    "doc_id": "d1",
                    "page_no": 1,
                    "image_artifact_id": "a" * 64,
                    "image_artifact_suffix": ".webp.enc",
                }
            }
        ],
        preview_limit=1,
    )
    assert any("(missing)" in c for c in streamlit_calls["captions"])


def test_handle_reindex_page_images_no_pdfs(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    page._handle_reindex_page_images(uploads_dir=uploads, limit=1, encrypt=False)  # type: ignore[attr-defined]
    assert streamlit_calls["infos"] == ["No PDFs found under uploads."]


def test_handle_reindex_page_images_smoke(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)
    monkeypatch.setattr(page, "_sha256_for_file", page._sha256_for_file.__wrapped__)  # type: ignore[attr-defined]

    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    pdf = uploads / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")

    models_processing = ModuleType("src.models.processing")

    class _Cfg:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

    class _Input(SimpleNamespace):
        pass

    models_processing.IngestionConfig = _Cfg  # type: ignore[attr-defined]
    models_processing.IngestionInput = _Input  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.models.processing", models_processing)

    ingestion_pipeline = ModuleType("src.processing.ingestion_pipeline")
    ingestion_pipeline.reindex_page_images_sync = lambda _cfg, _inputs: {  # type: ignore[attr-defined]
        "metadata": {"image_index.indexed": 1, "image_index.skipped": 0}
    }
    monkeypatch.setitem(
        sys.modules, "src.processing.ingestion_pipeline", ingestion_pipeline
    )

    page._handle_reindex_page_images(uploads_dir=uploads, limit=1, encrypt=False)  # type: ignore[attr-defined]
    assert any("Reindexed" in s for s in streamlit_calls["success"])


def test_handle_delete_upload_missing_file_warns(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    missing = uploads / "missing.txt"
    page._handle_delete_upload(target=missing, purge_artifacts=False)  # type: ignore[attr-defined]
    assert streamlit_calls["warnings"][-1] == "File not found."


def test_handle_delete_upload_purges_unreferenced_artifacts(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    uploads = tmp_path / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    target = uploads / "a.txt"
    target.write_text("hello", encoding="utf-8")

    monkeypatch.setattr(page, "_sha256_for_file", page._sha256_for_file.__wrapped__)  # type: ignore[attr-defined]

    qdrant = ModuleType("qdrant_client")
    qmodels = ModuleType("qdrant_client.models")

    class _Client:
        def count(self, **_kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(count=0)

        def delete(self, **_kwargs):  # type: ignore[no-untyped-def]
            return None

        def close(self) -> None:
            return None

    qdrant.QdrantClient = lambda **_k: _Client()  # type: ignore[attr-defined]
    qmodels.Filter = lambda **_k: None  # type: ignore[attr-defined]
    qmodels.FieldCondition = lambda **_k: None  # type: ignore[attr-defined]
    qmodels.MatchValue = lambda **_k: None  # type: ignore[attr-defined]
    qmodels.FilterSelector = lambda **_k: None  # type: ignore[attr-defined]
    qdrant.models = qmodels  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", qmodels)

    ref = page.ArtifactRef(sha256="a" * 64, suffix=".png")  # type: ignore[attr-defined]
    image_index = ModuleType("src.retrieval.image_index")
    image_index.collect_artifact_refs_for_doc_id = lambda *_a, **_k: [ref]  # type: ignore[attr-defined]
    image_index.delete_page_images_for_doc_id = lambda *_a, **_k: 1  # type: ignore[attr-defined]
    image_index.count_artifact_references_in_image_collection = lambda *_a, **_k: 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.retrieval.image_index", image_index)

    deleted: list[page.ArtifactRef] = []  # type: ignore[attr-defined]

    class _Store:
        def delete(self, r):  # type: ignore[no-untyped-def]
            deleted.append(r)

    monkeypatch.setattr(page.ArtifactStore, "from_settings", lambda _s: _Store())
    monkeypatch.setattr("src.utils.storage.get_client_config", lambda: {})

    page._handle_delete_upload(target=target, purge_artifacts=True)  # type: ignore[attr-defined]
    assert deleted == [ref]
    assert any("Deleted local artifacts: 1" in c for c in streamlit_calls["captions"])


def test_log_export_event_without_dest_path(monkeypatch) -> None:
    import importlib

    page = importlib.import_module("src.pages.02_documents")
    events: list[dict] = []
    monkeypatch.setattr("src.utils.telemetry.log_jsonl", lambda ev: events.append(ev))

    page._log_export_event({"export_performed": True, "context": "manual"})  # type: ignore[attr-defined]
    assert events
    assert "dest_basename" not in events[-1]


def test_handle_manual_export_parquet(
    monkeypatch, tmp_path: Path, streamlit_calls
) -> None:
    import importlib

    import streamlit as st  # type: ignore

    from src.config.settings import settings

    page = importlib.import_module("src.pages.02_documents")
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    st.session_state["graphrag_index"] = object()
    st.session_state["vector_index"] = object()

    monkeypatch.setattr(page, "get_export_seed_ids", lambda *_a, **_k: ["0"])
    out_file = tmp_path / "out.parquet"
    monkeypatch.setattr(page, "timestamped_export_path", lambda _d, _e: out_file)
    monkeypatch.setattr(
        page,
        "export_graph_parquet",
        lambda **_k: out_file.write_text("x", encoding="utf-8"),
    )
    monkeypatch.setattr(page, "record_graph_export_metric", lambda *_a, **_k: None)
    monkeypatch.setattr(page, "_log_export_event", lambda _p: None)

    page._handle_manual_export(tmp_path, "parquet")  # type: ignore[attr-defined]
    assert streamlit_calls["success"]
