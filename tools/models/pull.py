"""Utility to pre-download model artifacts from Hugging Face Hub.

This helps make local/offline runs more reliable by fetching model files into a
configured cache directory in advance.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_MODELS: list[tuple[str, str]] = [
    ("BAAI/bge-m3", "model.safetensors"),
    ("BAAI/bge-reranker-v2-m3", "model.safetensors"),
    ("google/siglip-base-patch16-224", "open_clip_pytorch_model.bin"),
    ("Qdrant/bm42-all-minilm-l6-v2-attentions", "model.onnx"),
]


def pull(pairs: Iterable[tuple[str, str]], cache_dir: Path) -> None:
    """Download a list of (repo_id, filename) pairs into the cache directory.

    Args:
        pairs: Sequence of tuple pairs specifying the file to download.
        cache_dir: Destination cache directory on local disk.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    for repo_id, filename in pairs:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )
        print(f"ok: {repo_id}/{filename} -> {path}")


def main() -> None:
    """Parse CLI arguments and pull requested model artifacts."""
    ap = argparse.ArgumentParser(description="Pre-download models for offline use")
    ap.add_argument("--all", action="store_true", help="Download default set")
    ap.add_argument(
        "--add",
        nargs=2,
        action="append",
        metavar=("REPO_ID", "FILENAME"),
        help="Additional model file(s) to fetch",
    )
    ap.add_argument("--cache_dir", default="~/.cache/huggingface/hub")
    args = ap.parse_args()

    pairs: list[tuple[str, str]] = []
    if args.all:
        pairs.extend(DEFAULT_MODELS)
    if args.add:
        pairs.extend([tuple(x) for x in args.add])

    if not pairs:
        ap.error("nothing to download. use --all or --add REPO_ID FILENAME")

    pull(pairs, Path(args.cache_dir).expanduser())
    print("Hint: export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 for offline runtime.")


if __name__ == "__main__":
    main()
