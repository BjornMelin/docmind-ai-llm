"""SigLIP consumption of encrypted images: decrypt-to-temp and cleanup.

This test verifies that given an `.enc` image path, the SigLIP rescoring path
decrypts to a temporary file before opening and cleans up the temporary file,
without requiring actual model inference.
"""

import os
import tempfile
from types import SimpleNamespace

from llama_index.core.schema import ImageNode, NodeWithScore

from src.retrieval import reranking as rr


def test_siglip_decrypts_and_cleans_temp(monkeypatch, tmp_path):
    # Create a fake encrypted path (we won't actually encrypt to keep this unit-only)
    enc_path = tmp_path / "page.webp.enc"
    enc_path.write_bytes(b"ciphertext")
    monkeypatch.setattr(
        rr,
        "ArtifactStore",
        SimpleNamespace(
            from_settings=lambda _settings: SimpleNamespace(
                resolve_path=lambda _ref: enc_path
            )
        ),
        raising=False,
    )

    nodes = [
        NodeWithScore(
            node=ImageNode(
                metadata={
                    "modality": "pdf_page_image",
                    "image_artifact_id": "deadbeef",
                    "image_artifact_suffix": ".webp.enc",
                }
            ),
            score=0.0,
        )
    ]

    # Track calls
    calls = {"dec": [], "open": [], "rm": []}

    # Monkeypatch decrypt_file to return a temp plaintext path
    def fake_decrypt(path: str) -> str:
        assert path.endswith(".enc")
        fd, name = tempfile.mkstemp(suffix=".webp")
        os.close(fd)
        calls["dec"].append(path)
        # Write dummy bytes so PIL open would succeed if called
        with open(name, "wb") as f:
            f.write(b"img")
        return name

    # Patch decrypt function where it resides
    monkeypatch.setattr("src.utils.security.decrypt_file", fake_decrypt)

    # Patch PIL.Image.open used in module to avoid real image parsing
    class _Img:
        def convert(self, *_: str):
            return self

    def fake_open(path: str):
        calls["open"].append(path)
        return _Img()

    # Patch PIL.Image.open used by reranking
    monkeypatch.setattr("PIL.Image.open", staticmethod(fake_open))

    # Patch os.remove to observe cleanup
    def fake_remove(path: str):
        calls["rm"].append(path)

    # Cleanup happens inside images helper; patch its os.remove
    monkeypatch.setattr("src.utils.images.os.remove", fake_remove, raising=False)

    # Make budget 0 so _siglip_rescore aborts after loading images and triggers cleanup
    out = rr._siglip_rescore("q", nodes, budget_ms=0)
    # Should fail-open to input nodes (same object identity)
    assert out[0] is nodes[0]

    # Assert decrypt invoked and opened decrypted path
    assert calls["dec"] == [str(enc_path)]
    assert calls["open"]
    assert calls["open"][0].endswith(".webp")
    # Cleanup called on temp file path
    assert calls["rm"]
    assert calls["rm"][0].endswith(".webp")
