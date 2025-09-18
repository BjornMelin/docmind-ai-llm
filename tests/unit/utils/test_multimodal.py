"""Tests for multimodal utilities."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils import multimodal


class _FakeClip:
    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector

    def get_image_embedding(self, image):
        return self._vector


class _TorchLikeTensor:
    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._vector


@pytest.mark.asyncio
async def test_generate_image_embeddings_normalizes_vector() -> None:
    clip = _FakeClip(np.array([3.0, 4.0], dtype=np.float32))
    embedding = await multimodal.generate_image_embeddings(clip, image="img")
    assert pytest.approx(np.linalg.norm(embedding), rel=1e-6) == 1.0


@pytest.mark.asyncio
async def test_generate_image_embeddings_handles_torch_tensor() -> None:
    clip = _FakeClip(_TorchLikeTensor(np.array([1.0, 0.0, 0.0], dtype=np.float32)))
    embedding = await multimodal.generate_image_embeddings(clip, image="img")
    assert embedding.shape == (3,)


def test_validate_vram_usage_without_torch_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    monkeypatch.delitem(sys.modules, "torch", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch not available")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert multimodal.validate_vram_usage(_FakeClip(np.zeros(1))) == 0.0


def test_validate_vram_usage_with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCuda:
        def __init__(self) -> None:
            self.calls = 0

        def is_available(self) -> bool:
            return True

        def memory_allocated(self) -> int:
            value = self.calls * (1024**3)
            self.calls += 1
            return value

        def empty_cache(self) -> None:
            self.calls = 0

    fake_torch = SimpleNamespace(cuda=_FakeCuda())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    clip = _FakeClip(np.zeros(4))
    delta = multimodal.validate_vram_usage(clip, images=[object()])
    assert delta == pytest.approx(1.0, rel=1e-6)


def test_batch_process_images_validates_dimension() -> None:
    clip = _FakeClip(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    embeddings = multimodal.batch_process_images(
        clip,
        images=[object(), object()],
        batch_size=1,
        output_dim=4,
    )
    assert embeddings.shape == (2, 4)


def test_batch_process_images_converts_mismatched_dimension_to_zero() -> None:
    clip = _FakeClip(np.array([1.0, 2.0], dtype=np.float32))
    embeddings = multimodal.batch_process_images(
        clip,
        images=[object()],
        output_dim=4,
    )
    assert embeddings.shape == (1, 4)
    assert np.array_equal(embeddings[0], np.zeros(4))


def test_batch_process_images_converts_failures_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FlakyClip:
        def __init__(self) -> None:
            self.calls = 0

        def get_image_embedding(self, _img):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom")
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    clip = _FlakyClip()
    embeddings = multimodal.batch_process_images(
        clip, images=[object(), object()], output_dim=3
    )
    assert embeddings.shape == (2, 3)
    assert np.array_equal(embeddings[1], np.zeros(3))


@pytest.mark.asyncio
async def test_cross_modal_search_text_to_image_truncates_output() -> None:
    class _Node:
        def __init__(self, text: str) -> None:
            self.node = SimpleNamespace(text=text, metadata={"image_path": "img.png"})
            self.score = 0.8

    class _Response:
        def __init__(self) -> None:
            self.source_nodes = [_Node("x" * 300)]

    class _Engine:
        def query(self, _query: str) -> _Response:
            return _Response()

    class _Index:
        def as_query_engine(self) -> _Engine:
            return _Engine()

    results = await multimodal.cross_modal_search(
        _Index(), query="demo", search_type="text_to_image", top_k=1
    )
    assert len(results) == 1
    assert len(results[0]["text"]) == multimodal.TEXT_TRUNCATION_LIMIT


@pytest.mark.asyncio
async def test_cross_modal_search_image_to_image() -> None:
    class _Node:
        def __init__(self) -> None:
            self.node = SimpleNamespace(text="desc", metadata={"image_path": "img.png"})
            self.score = 0.5

    class _Retriever:
        def retrieve(self, _query_image):
            return [_Node()]

    class _Index:
        def as_retriever(self) -> _Retriever:
            return _Retriever()

    results = await multimodal.cross_modal_search(
        _Index(), query_image="image", search_type="image_to_image", top_k=1
    )
    assert results[0]["image_path"] == "img.png"


def test_create_image_documents_handles_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = multimodal.create_image_documents(["img.png"])
    assert docs == [
        multimodal.ImageDocument(
            image_path="img.png", metadata={"source": "multimodal"}
        )
    ]


def test_create_image_documents_skips_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExplodingPath:
        def __str__(self) -> str:
            raise ValueError("boom")

    docs = multimodal.create_image_documents([_ExplodingPath()])
    assert docs == []


@pytest.mark.asyncio
async def test_validate_end_to_end_pipeline() -> None:
    clip = _FakeClip(np.array([1.0, 0.0], dtype=np.float32))
    result = await multimodal.validate_end_to_end_pipeline(
        query="Explain SigLIP and OpenCLIP integration",
        query_image="image",
        clip=clip,
        property_graph=object(),
        llm=object(),
    )
    assert "final_response" in result
    assert result["visual_similarity"]["norm"] == pytest.approx(1.0, rel=1e-6)
