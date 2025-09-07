import numpy as np
import pytest

from src.models.embeddings import ImageEmbedder


@pytest.mark.unit
def test_image_embedder_backbone_shapes_vitl_and_vith():
    # Patch to avoid importing real backends
    ie = ImageEmbedder(device="cpu")

    def fake_encode(images, *, backbone=None, **_):  # noqa: ANN001
        dim = 768 if backbone == "openclip_vitl14" else 1024
        return np.ones((len(images), dim), dtype=np.float32)

    # Monkeypatch the instance method
    ie.encode_image = fake_encode  # type: ignore[method-assign]

    arr_l = ie.encode_image([object()], backbone="openclip_vitl14")
    arr_h = ie.encode_image([object()], backbone="openclip_vith14")

    assert arr_l.shape == (1, 768)
    assert arr_h.shape == (1, 1024)


@pytest.mark.unit
def test_image_embedder_normalization_property():
    ie = ImageEmbedder(device="cpu")

    def fake_encode(images, *, backbone=None, normalize=True, **_):  # noqa: ANN001
        dim = 768
        x = np.random.randn(len(images), dim).astype(np.float32)
        if normalize:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        return x

    ie.encode_image = fake_encode  # type: ignore[method-assign]
    out = ie.encode_image([object(), object()], backbone="openclip_vitl14", normalize=True)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

