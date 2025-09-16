"""Additional lightweight pipeline tests for DocumentProcessor/Unstructured.

Focused on metadata round-trip and identity behavior when elements are not
"unstructured-like". Uses minimal fakes and monkeypatching to avoid heavy
dependencies and external I/O.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.unit
def test_round_trip_elements_to_nodes_and_back(monkeypatch, tmp_path) -> None:
    """Elements -> nodes -> elements round-trip retains basic metadata.

    Uses a fake Document class to avoid importing heavy LlamaIndex modules.
    """
    # Import module under test
    import importlib

    dmod = importlib.import_module("src.processing.document_processor")
    processing_strategy_cls = importlib.import_module(
        "src.models.processing"
    ).ProcessingStrategy

    # Fake Document compatible with code paths using get_content/metadata
    class FakeDoc:
        def __init__(
            self, *, text: str = "", doc_id: str | None = None, metadata=None, **_: Any
        ):
            self.text = text
            self.doc_id = doc_id or "parent"
            self.metadata = metadata or {}
            # Attributes forwarded during element-node creation
            self.excluded_embed_metadata_keys = None
            self.excluded_llm_metadata_keys = None
            self.metadata_separator = " "
            self.metadata_template = ""
            self.text_template = ""

        def get_content(self) -> str:
            return self.text

    # Patch _document to return our fake class so isinstance checks pass
    monkeypatch.setattr(dmod, "_document", lambda: FakeDoc)

    # Avoid creating real DuckDB-backed cache/docstore during DocumentProcessor init
    monkeypatch.setattr(dmod, "DuckDBKVStore", lambda **_: object())
    monkeypatch.setattr(dmod, "IngestionCache", lambda **_: object())
    monkeypatch.setattr(dmod, "SimpleDocumentStore", lambda *_, **__: object())

    # Build minimal fake element with some metadata
    element = SimpleNamespace(
        text="Hello World",
        category="NarrativeText",
        metadata=SimpleNamespace(
            page_number=3, filename="f.txt", coordinates=[1, 2, 3, 4]
        ),
    )

    # Original node with file path metadata
    src_file = tmp_path / "f.txt"
    src_file.write_text("x")
    original_node = FakeDoc(
        text="", metadata={"file_path": str(src_file), "source": str(src_file)}
    )

    # Convert elements -> nodes
    tfm = dmod.UnstructuredTransformation(processing_strategy_cls.FAST)
    nodes = tfm._convert_elements_to_nodes([element], original_node, src_file)  # pylint: disable=protected-access

    assert isinstance(nodes, list)
    assert len(nodes) == 1
    node = nodes[0]
    assert isinstance(node, FakeDoc)
    # Key metadata propagated onto node
    assert node.metadata.get("element_index") == 0
    assert node.metadata.get("page_number") == 3
    assert node.metadata.get("processing_strategy") == processing_strategy_cls.FAST

    # Convert nodes -> elements
    simple_hashing = SimpleNamespace(
        canonicalization_version="1",
        hmac_secret="unit-secret",
        hmac_secret_version="1",
        metadata_keys=[
            "content_type",
            "language",
            "source",
            "source_path",
            "tenant_id",
            "size_bytes",
        ],
    )
    proc = dmod.DocumentProcessor(
        settings=SimpleNamespace(cache_dir=str(tmp_path), hashing=simple_hashing)
    )
    out_elements: list[Any] = proc._convert_nodes_to_elements(nodes)  # pylint: disable=protected-access

    assert len(out_elements) == 1
    el = out_elements[0]
    assert el.text == "Hello World"
    # Round-trip retains metadata keys
    assert el.metadata.get("page_number") == 3
    assert el.metadata.get("element_index") == 0


@pytest.mark.unit
def test_unstructured_identity_when_not_unstructured_like(
    monkeypatch, tmp_path
) -> None:
    """When elements aren't unstructured-like, chunkers are not invoked.

    Patches is_unstructured_like to False, verifying the transformation
    bypasses chunk_by_title/basic and treats elements as already chunked.
    """
    import importlib

    dmod = importlib.import_module("src.processing.document_processor")
    processing_strategy_cls = importlib.import_module(
        "src.models.processing"
    ).ProcessingStrategy

    calls: dict[str, int] = {"title": 0, "basic": 0}

    # Fake Document for isinstance and node creation
    class FakeDoc:
        def __init__(self, *, text: str = "", metadata=None, **_: Any):
            self.text = text
            self.metadata = metadata or {}
            self.excluded_embed_metadata_keys = None
            self.excluded_llm_metadata_keys = None
            self.metadata_separator = " "
            self.metadata_template = ""
            self.text_template = ""

    monkeypatch.setattr(dmod, "_document", lambda: FakeDoc)

    # Patch helpers: partition returns one element; chunkers increment counters
    element = SimpleNamespace(
        text="T", category="NarrativeText", metadata=SimpleNamespace()
    )
    monkeypatch.setattr(dmod, "partition", lambda *_, **__: [element])

    def _count_title(**_: Any) -> list[Any]:
        calls["title"] += 1
        return [element]

    def _count_basic(**_: Any) -> list[Any]:
        calls["basic"] += 1
        return [element]

    monkeypatch.setattr(dmod, "chunk_by_title", _count_title)
    monkeypatch.setattr(dmod, "chunk_by_basic", _count_basic)
    # Force detection to treat elements as not unstructured-like
    monkeypatch.setattr(dmod, "is_unstructured_like", lambda _e: False)

    # Prepare a FakeDoc with a valid path in metadata
    f = tmp_path / "x.txt"
    f.write_text("hello")
    doc = FakeDoc(text="", metadata={"file_path": str(f)})

    tfm = dmod.UnstructuredTransformation(processing_strategy_cls.FAST)
    out = tfm([doc])
    assert isinstance(out, list)
    # Ensure chunkers were not called when identity path is taken
    assert calls["title"] == 0
    assert calls["basic"] == 0
