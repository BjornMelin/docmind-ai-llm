"""Predownload core models for offline operation.

Usage:
  uv run python scripts/model_prep/predownload_models.py

Downloads (to HF cache):
  - BAAI/bge-m3 (text embeddings)
  - BAAI/bge-reranker-v2-m3 (text reranking)
  - google/siglip-base-patch16-224 (visual rerank cosine)

Notes:
  - Set HF_HOME to control cache path. This script does not contact any
    external endpoints after models are present.
"""

from __future__ import annotations

import os


def _download_bge_m3() -> None:
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    AutoTokenizer.from_pretrained("BAAI/bge-m3")
    AutoModel.from_pretrained("BAAI/bge-m3")


def _download_bge_reranker() -> None:
    from sentence_transformers import CrossEncoder  # type: ignore

    CrossEncoder("BAAI/bge-reranker-v2-m3")


def _download_siglip() -> None:
    from transformers import SiglipModel, SiglipProcessor  # type: ignore

    SiglipModel.from_pretrained("google/siglip-base-patch16-224")
    SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")


def main() -> None:
    """Download required models into the local HF cache."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    _download_bge_m3()
    _download_bge_reranker()
    _download_siglip()
    print("Predownload complete.")


if __name__ == "__main__":
    main()
