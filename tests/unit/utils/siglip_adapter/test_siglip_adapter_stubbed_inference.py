"""Unit tests for SiglipEmbedding with a stubbed model/processor (no HF downloads)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_siglip_embedding_text_and_image_with_stubbed_model() -> None:
    torch = pytest.importorskip("torch")  # type: ignore
    pil_image = pytest.importorskip("PIL.Image")  # type: ignore
    import numpy as np

    from src.utils.siglip_adapter import SiglipEmbedding

    class _Proc:
        def __call__(self, *, images=None, text=None, return_tensors="pt", **_kw):  # type: ignore[no-untyped-def]
            if images is not None:
                return {"pixel_values": torch.zeros((len(images), 3, 224, 224))}
            if text is not None:
                return {
                    "input_ids": torch.zeros((len(text), 4), dtype=torch.long),
                    "attention_mask": torch.ones((len(text), 4), dtype=torch.long),
                }
            return {}

    class _Model:
        def get_image_features(self, *, pixel_values):  # type: ignore[no-untyped-def]
            batch = int(pixel_values.shape[0])
            return torch.ones((batch, 4), dtype=torch.float32)

        def get_text_features(self, *, input_ids, attention_mask):  # type: ignore[no-untyped-def]
            batch = int(input_ids.shape[0])
            return torch.arange(1, 1 + (batch * 4), dtype=torch.float32).reshape(
                batch, 4
            )

    emb = SiglipEmbedding(model_id="stub", device="cpu")
    # Inject stubs to bypass lazy HuggingFace model loading
    emb._model = _Model()  # type: ignore[attr-defined]
    emb._proc = _Proc()  # type: ignore[attr-defined]
    emb._dim = 4  # type: ignore[attr-defined]

    q = emb.get_text_embedding("hello")
    assert isinstance(q, np.ndarray)
    assert q.shape == (4,)
    assert float(np.linalg.norm(q)) == pytest.approx(1.0, rel=1e-4)

    text_batch = emb.get_text_embeddings(["hello", "world"], batch_size=1)
    assert text_batch.shape == (2, 4)
    for row in text_batch:
        assert float(np.linalg.norm(row)) == pytest.approx(1.0, rel=1e-4)

    img = pil_image.new("RGB", (32, 32), color=(255, 0, 0))
    v = emb.get_image_embedding(img)
    assert v.shape == (4,)
    assert float(np.linalg.norm(v)) == pytest.approx(1.0, rel=1e-4)

    batch = emb.get_image_embeddings([img, img], batch_size=1)
    assert batch.shape == (2, 4)
    for row in batch:
        assert float(np.linalg.norm(row)) == pytest.approx(1.0, rel=1e-4)
