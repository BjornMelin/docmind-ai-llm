"""Coverage-focused tests for ImageEmbedder backend loading paths."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_image_embedder_openclip_backend_load_and_encode(monkeypatch) -> None:
    import torch

    class _Model:
        embed_dim = 768

        def eval(self):  # type: ignore[no-untyped-def]
            return self

        def to(self, _device: str):  # type: ignore[no-untyped-def]
            return self

        def encode_image(self, x):  # type: ignore[no-untyped-def]
            # x: [B, C, H, W]
            return torch.ones((x.shape[0], self.embed_dim), dtype=torch.float32)

    def _preprocess(_img):  # type: ignore[no-untyped-def]
        return torch.zeros((3, 2, 2), dtype=torch.float32)

    open_clip = ModuleType("open_clip")
    open_clip.create_model_and_transforms = (  # type: ignore[attr-defined]
        lambda _name, pretrained=None: (_Model(), None, _preprocess)
    )
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)

    from src.models.embeddings import ImageEmbedder

    ie = ImageEmbedder(backbone="openclip_vitl14", device="cpu", default_batch_size=2)
    out = ie.encode_image([object(), object()], normalize=False)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 768)


def test_image_embedder_siglip_via_vision_siglip(monkeypatch) -> None:
    import torch

    class _Backend:
        def get_image_features(self, *, pixel_values):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                pooler_output=torch.ones(
                    (pixel_values.shape[0], 768), dtype=torch.float32
                )
            )

    class _Processor:
        def __call__(self, *, images, return_tensors="pt"):  # type: ignore[no-untyped-def]
            assert return_tensors == "pt"
            return {"pixel_values": torch.zeros((len(images), 3, 224, 224))}

    vision = ModuleType("src.utils.vision_siglip")
    vision.load_siglip = (  # type: ignore[attr-defined]
        lambda model_id, device, revision=None: (_Backend(), _Processor(), device)
    )
    vision.DEFAULT_SIGLIP_MODEL_REVISION = "test-revision"  # type: ignore[attr-defined]
    vision.siglip_features = (  # type: ignore[attr-defined]
        lambda output, normalize=True: (
            output.pooler_output / output.pooler_output.norm(dim=-1, keepdim=True)
            if normalize
            else output.pooler_output
        )
    )
    monkeypatch.setitem(sys.modules, "src.utils.vision_siglip", vision)

    from src.models.embeddings import ImageEmbedder

    ie = ImageEmbedder(backbone="siglip_base", device="cpu")
    out = ie.encode_image([object()])
    assert out.shape == (1, 768)


def test_image_embedder_device_override_raises(monkeypatch) -> None:
    from src.models.embeddings import ImageEmbedder

    ie = ImageEmbedder(backbone="auto", device="cpu")
    with pytest.raises(ValueError, match="Per-call device override is not supported"):
        ie.encode_image([object()], device="cuda")


def test_image_embedder_bge_visualized_backbone_raises() -> None:
    from src.models.embeddings import ImageEmbedder

    ie = ImageEmbedder(backbone="bge_visualized", device="cpu")
    with pytest.raises(RuntimeError, match="bge_visualized"):
        ie.encode_image([object()])


def test_image_embedder_siglip_falls_back_to_transformers(monkeypatch) -> None:
    import torch

    # Force vision_siglip import path to fail
    vision = ModuleType("src.utils.vision_siglip")
    vision.load_siglip = (  # type: ignore[attr-defined]
        lambda *_a, **_k: (_ for _ in ()).throw(ImportError("x"))
    )
    vision.DEFAULT_SIGLIP_MODEL_REVISION = "test-revision"  # type: ignore[attr-defined]
    vision.siglip_features = (  # type: ignore[attr-defined]
        lambda output, normalize=True: (
            output / output.norm(dim=-1, keepdim=True) if normalize else output
        )
    )
    monkeypatch.setitem(sys.modules, "src.utils.vision_siglip", vision)

    class _Backend:
        def get_image_features(self, *, pixel_values):  # type: ignore[no-untyped-def]
            return torch.ones((pixel_values.shape[0], 768), dtype=torch.float32)

    class _Processor:
        def __call__(self, *, images, return_tensors="pt"):  # type: ignore[no-untyped-def]
            return {"pixel_values": torch.zeros((len(images), 3, 224, 224))}

    class _SiglipModel:
        @staticmethod
        def from_pretrained(  # type: ignore[no-untyped-def]
            _model_id: str, device_map=None, revision=None
        ):
            _ = device_map
            assert revision == "test-revision"
            return _Backend()

    class _SiglipProcessor:
        @staticmethod
        def from_pretrained(_model_id: str, revision=None):  # type: ignore[no-untyped-def]
            assert revision == "test-revision"
            return _Processor()

    transformers = ModuleType("transformers")
    transformers.SiglipModel = _SiglipModel  # type: ignore[attr-defined]
    transformers.SiglipProcessor = _SiglipProcessor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    from src.models.embeddings import ImageEmbedder

    ie = ImageEmbedder(backbone="siglip_base", device="cpu")
    out = ie.encode_image([object()])
    assert out.shape == (1, 768)
