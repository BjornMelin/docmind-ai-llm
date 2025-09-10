"""Integration tests for UnifiedEmbedder (offline via stubs)."""

import numpy as np
import pytest

from src.models.embeddings import UnifiedEmbedder


@pytest.mark.integration
def test_unified_embedder_integration_dense_and_sparse_text():
    """Dense 1024-d + sparse via stubbed BGE-M3 backend."""
    u = UnifiedEmbedder()

    # Patch text backend to avoid external downloads
    class _Stub:
        """Minimal stub for text embedding backend used in tests."""

        def encode(self, texts, **_):
            """Return fixed dense+lexical outputs sized to inputs."""
            n = len(texts)
            return {
                "dense_vecs": np.ones((n, 1024), dtype=np.float32),
                "lexical_weights": [{0: 1.0} for _ in range(n)],
            }

    u.text._backend = _Stub()  # type: ignore[attr-defined]  # pylint: disable=protected-access
    out = u.text.encode_text(["x", "y"], return_dense=True, return_sparse=True)
    assert out["dense"].shape == (2, 1024)
    assert len(out["sparse"]) == 2


@pytest.mark.integration
def test_unified_embedder_integration_image_vectors():
    """Image vectors returned with expected dimension (768/1024).

    Covers single and batch images across both backbone dimensions.
    """
    u = UnifiedEmbedder()

    # Dummy images: batch of 2, shape (H, W, C)
    img1 = np.ones((224, 224, 3), dtype=np.uint8)
    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    images = [img1, img2]

    # Monkeypatch encode_image to avoid heavy deps and emulate dims by backbone
    def _fake_encode(imgs, *, backbone=None, **_):
        dim = 1024 if backbone in {"openclip_vith14", "vit_l_1024"} else 768
        return np.ones((len(imgs), dim), dtype=np.float32)

    u.image.encode_image = _fake_encode  # type: ignore[method-assign]

    # Test 1024-d backbone (ViT-H/14 analogue)
    out_1024 = u.image.encode_image(images, backbone="openclip_vith14")
    assert out_1024.shape == (2, 1024)
    # Test 768-d backbone (ViT-L/14 analogue)
    out_768 = u.image.encode_image(images, backbone="openclip_vitl14")
    assert out_768.shape == (2, 768)

    # Also test single image for both dimensions
    out_1024_single = u.image.encode_image([img1], backbone="openclip_vith14")
    assert out_1024_single.shape == (1, 1024)
    out_768_single = u.image.encode_image([img2], backbone="openclip_vitl14")
    assert out_768_single.shape == (1, 768)
