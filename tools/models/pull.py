"""Utility to pre-download model artifacts from Hugging Face Hub.

This helps make local/offline runs more reliable by fetching model files into a
configured cache directory in advance.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BGE_M3_IGNORE_PATTERNS = (
    ".gitattributes",
    "README.md",
    "imgs/*",
    "*.jpg",
    "*.webp",
    "onnx/*",
)

_SIGLIP_TRANSFORMERS_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


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


def pull_siglip_snapshot(cache_dir: Path) -> None:
    """Download the pinned Transformers SigLIP snapshot used by the app."""
    from src.utils.vision_siglip import (
        DEFAULT_SIGLIP_MODEL_ID,
        DEFAULT_SIGLIP_MODEL_REVISION,
    )

    path = snapshot_download(
        repo_id=DEFAULT_SIGLIP_MODEL_ID,
        revision=DEFAULT_SIGLIP_MODEL_REVISION,
        allow_patterns=list(_SIGLIP_TRANSFORMERS_FILES),
        cache_dir=str(cache_dir),
        local_files_only=False,
    )
    print(f"ok: {DEFAULT_SIGLIP_MODEL_ID}@{DEFAULT_SIGLIP_MODEL_REVISION} -> {path}")


def pull_bge_m3_snapshot(cache_dir: Path) -> None:
    """Download the pinned SentenceTransformers BGE-M3 snapshot."""
    from src.config.embedding_defaults import (
        DEFAULT_BGE_M3_MODEL_ID,
        DEFAULT_BGE_M3_MODEL_REVISION,
    )

    path = snapshot_download(
        repo_id=DEFAULT_BGE_M3_MODEL_ID,
        revision=DEFAULT_BGE_M3_MODEL_REVISION,
        ignore_patterns=list(_BGE_M3_IGNORE_PATTERNS),
        cache_dir=str(cache_dir),
        local_files_only=False,
    )
    print(f"ok: {DEFAULT_BGE_M3_MODEL_ID}@{DEFAULT_BGE_M3_MODEL_REVISION} -> {path}")


def pull_rapidocr(cache_dir: Path, *, force: bool = False) -> None:
    """Download RapidOCR ONNX artifacts into DocMind's local model cache."""
    from src.processing.parsing.backends.rapidocr_backend import (
        missing_rapidocr_onnx_models,
        prefetch_rapidocr_models,
        verify_rapidocr_models,
    )

    target = prefetch_rapidocr_models(cache_dir, force=force)
    verify_rapidocr_models(cache_dir)
    missing = missing_rapidocr_onnx_models(cache_dir)
    if missing:
        raise SystemExit(f"RapidOCR prefetch incomplete: {', '.join(missing)}")
    print(f"ok: RapidOCR ONNX models -> {target}")


def pull_docling_layout(cache_dir: Path, *, force: bool = False) -> None:
    """Download Docling layout artifacts into DocMind's local model cache."""
    from src.processing.parsing.backends.docling_backend import (
        missing_docling_layout_models,
        prefetch_docling_layout_models,
        verify_docling_layout_models,
    )

    target = prefetch_docling_layout_models(cache_dir, force=force)
    verify_docling_layout_models(cache_dir)
    missing = missing_docling_layout_models(cache_dir)
    if missing:
        raise SystemExit(f"Docling layout prefetch incomplete: {', '.join(missing)}")
    print(f"ok: Docling layout model -> {target}")


def main() -> None:
    """Parse CLI arguments and pull requested model artifacts."""
    ap = argparse.ArgumentParser(description="Pre-download models for offline use")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Download the pinned BGE-M3 and SigLIP snapshots",
    )
    ap.add_argument(
        "--bge-m3",
        action="store_true",
        help="Download the pinned canonical BGE-M3 SentenceTransformers snapshot",
    )
    ap.add_argument(
        "--rapidocr",
        action="store_true",
        help="Download verified RapidOCR ONNX files for local OCR",
    )
    ap.add_argument(
        "--docling-layout",
        action="store_true",
        help="Download verified Docling layout files for local PDF parsing",
    )
    ap.add_argument(
        "--parser-defaults",
        action="store_true",
        help="Download Docling layout and RapidOCR files for default parsing",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download parser model files even when they already exist",
    )
    ap.add_argument(
        "--rapidocr-cache-dir",
        default="./cache/models",
        help="Destination for Docling and RapidOCR parser files",
    )
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
    if args.add:
        pairs.extend([tuple(x) for x in args.add])

    if args.parser_defaults or args.docling_layout:
        pull_docling_layout(
            Path(args.rapidocr_cache_dir).expanduser(),
            force=bool(args.force),
        )

    if args.parser_defaults or args.rapidocr:
        pull_rapidocr(
            Path(args.rapidocr_cache_dir).expanduser(),
            force=bool(args.force),
        )

    if args.all:
        pull_bge_m3_snapshot(Path(args.cache_dir).expanduser())
        pull_siglip_snapshot(Path(args.cache_dir).expanduser())
    elif args.bge_m3:
        pull_bge_m3_snapshot(Path(args.cache_dir).expanduser())

    if not pairs and not (
        args.all
        or args.bge_m3
        or args.rapidocr
        or args.docling_layout
        or args.parser_defaults
    ):
        ap.error(
            "nothing to download. use --all, --bge-m3, --parser-defaults, "
            "--rapidocr, --docling-layout, or --add REPO_ID FILENAME"
        )

    if pairs:
        pull(pairs, Path(args.cache_dir).expanduser())
    print("Hint: export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 for offline runtime.")


if __name__ == "__main__":
    main()
