import numpy as np
import pytest

from src.models.embeddings import UnifiedEmbedder


@pytest.mark.unit
def test_unified_embedder_routes_text_and_images():
    # Fake sub-embedders so we don't import heavy libs
    class _T:
        def encode_text(self, texts, **_):  # noqa: ANN001
            return {"dense": np.ones((len(texts), 1024), dtype=np.float32), "sparse": [{0: 1.0} for _ in texts]}

    class _I:
        def encode_image(self, images, **_):  # noqa: ANN001
            return np.ones((len(images), 768), dtype=np.float32)

    u = UnifiedEmbedder(text=_T(), image=_I())
    out = u.encode(["a", "b", object()])
    assert out["dense"].shape == (2, 1024)
    assert isinstance(out["sparse"], list)
    assert out["image_dense"].shape == (1, 768)

