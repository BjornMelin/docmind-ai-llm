"""Unit tests for model pull CLI (tools/models/pull.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.config.embedding_defaults import (
    DEFAULT_BGE_M3_MODEL_ID,
    DEFAULT_BGE_M3_MODEL_REVISION,
    DEFAULT_BGE_RERANKER_MODEL_ID,
    DEFAULT_BGE_RERANKER_MODEL_REVISION,
    DEFAULT_BM42_FILES,
    DEFAULT_BM42_MODEL_ID,
    DEFAULT_BM42_SOURCE_REPO,
    DEFAULT_BM42_SOURCE_REVISION,
)
from tools.models.pull import (
    _BGE_M3_IGNORE_PATTERNS,
    _BGE_RERANKER_TRANSFORMERS_FILES,
    _SIGLIP_TRANSFORMERS_FILES,
    main,
    pull,
    pull_bge_m3_snapshot,
    pull_bge_reranker_snapshot,
    pull_bm42_snapshot,
    pull_siglip_snapshot,
    resolve_bm42_snapshot,
)


def test_all_prefetches_complete_pinned_snapshots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Keep ``--all`` free of incomplete single-file model downloads."""
    snapshots: list[tuple[str, Path]] = []
    raw_pairs: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "sys.argv",
        ["pull.py", "--all", "--cache_dir", str(tmp_path)],
    )
    monkeypatch.setattr(
        "tools.models.pull.pull_bge_m3_snapshot",
        lambda path: snapshots.append(("bge", path)),
    )
    monkeypatch.setattr(
        "tools.models.pull.pull_siglip_snapshot",
        lambda path: snapshots.append(("siglip", path)),
    )
    monkeypatch.setattr(
        "tools.models.pull.pull_bge_reranker_snapshot",
        lambda path: snapshots.append(("reranker", path)),
    )
    monkeypatch.setattr(
        "tools.models.pull.pull_bm42_snapshot",
        lambda path: snapshots.append(("bm42", path)),
    )
    monkeypatch.setattr(
        "tools.models.pull.pull",
        lambda pairs, _path: raw_pairs.extend(pairs),
    )

    main()

    assert snapshots == [
        ("bge", tmp_path),
        ("bm42", tmp_path),
        ("reranker", tmp_path),
        ("siglip", tmp_path),
    ]
    assert raw_pairs == []


def test_pull_mocks_hf(tmp_path: Path) -> None:
    """Test that model pull CLI invokes HuggingFace download correctly.

    Args:
        tmp_path: Temporary directory path provided by pytest fixture.

    This test mocks the HuggingFace hub download function to verify that
    the pull command correctly invokes downloads for model files and
    returns appropriate file paths.
    """
    calls: list[tuple[str, str, str]] = []

    def fake_download(
        repo_id: str, filename: str, cache_dir: str, local_files_only: bool
    ) -> str:
        calls.append((repo_id, filename, cache_dir))
        return str(tmp_path / f"{repo_id.replace('/', '__')}__{filename}")

    with patch("tools.models.pull.hf_hub_download", side_effect=fake_download):
        pull([("BAAI/bge-m3", "model.safetensors")], tmp_path)

    assert calls
    assert calls[0][0] == "BAAI/bge-m3"


def test_pull_siglip_snapshot_uses_pinned_transformers_files(tmp_path: Path) -> None:
    """Prefetch the runtime SigLIP format at its canonical revision."""
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    with patch(
        "tools.models.pull.snapshot_download",
        side_effect=fake_snapshot_download,
    ):
        pull_siglip_snapshot(tmp_path)

    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == "google/siglip-base-patch16-224"
    assert call["revision"] == "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
    assert set(call["allow_patterns"]) == set(_SIGLIP_TRANSFORMERS_FILES)  # type: ignore[arg-type]


def test_pull_bge_m3_snapshot_excludes_duplicate_onnx_weights(tmp_path: Path) -> None:
    """Fetch one pinned PyTorch snapshot without the duplicate ONNX export."""
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    with patch(
        "tools.models.pull.snapshot_download",
        side_effect=fake_snapshot_download,
    ):
        pull_bge_m3_snapshot(tmp_path)

    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == DEFAULT_BGE_M3_MODEL_ID
    assert call["revision"] == DEFAULT_BGE_M3_MODEL_REVISION
    assert set(call["ignore_patterns"]) == set(_BGE_M3_IGNORE_PATTERNS)  # type: ignore[arg-type]


def test_pull_bge_reranker_uses_complete_pinned_transformers_files(
    tmp_path: Path,
) -> None:
    """Fetch the runtime CrossEncoder snapshot at its immutable revision."""
    calls: list[dict[str, object]] = []

    with patch(
        "tools.models.pull.snapshot_download",
        side_effect=lambda **kwargs: calls.append(kwargs) or str(tmp_path / "snapshot"),
    ):
        pull_bge_reranker_snapshot(tmp_path)

    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == DEFAULT_BGE_RERANKER_MODEL_ID
    assert call["revision"] == DEFAULT_BGE_RERANKER_MODEL_REVISION
    assert set(call["allow_patterns"]) == set(  # type: ignore[arg-type]
        _BGE_RERANKER_TRANSFORMERS_FILES
    )


def test_pull_bm42_uses_complete_pinned_fastembed_files(tmp_path: Path) -> None:
    """Fetch FastEmbed's logical BM42 model from its immutable source repo."""
    calls: list[dict[str, object]] = []

    with patch(
        "tools.models.pull.snapshot_download",
        side_effect=lambda **kwargs: calls.append(kwargs) or str(tmp_path / "snapshot"),
    ):
        pull_bm42_snapshot(tmp_path)

    assert len(calls) == 1
    call = calls[0]
    assert DEFAULT_BM42_MODEL_ID == "Qdrant/bm42-all-minilm-l6-v2-attentions"
    assert call["repo_id"] == DEFAULT_BM42_SOURCE_REPO
    assert call["revision"] == DEFAULT_BM42_SOURCE_REVISION
    assert set(call["allow_patterns"]) == set(DEFAULT_BM42_FILES)  # type: ignore[arg-type]
    assert call["local_files_only"] is False


def test_resolve_bm42_reuses_manifest_without_network(tmp_path: Path) -> None:
    """Resolve the same partial snapshot for offline FastEmbed construction."""
    calls: list[dict[str, object]] = []

    with patch(
        "tools.models.pull.snapshot_download",
        side_effect=lambda **kwargs: calls.append(kwargs) or str(tmp_path / "snapshot"),
    ):
        path = resolve_bm42_snapshot(tmp_path, local_files_only=True)

    assert path == str(tmp_path / "snapshot")
    assert calls[0]["local_files_only"] is True
    assert set(calls[0]["allow_patterns"]) == set(DEFAULT_BM42_FILES)  # type: ignore[arg-type]


def test_bge_reranker_flag_selects_only_the_reranker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Keep the dedicated reranker selector independent from ``--all``."""
    selected: list[Path] = []
    monkeypatch.setattr(
        "sys.argv",
        ["pull.py", "--bge-reranker", "--cache_dir", str(tmp_path)],
    )
    monkeypatch.setattr(
        "tools.models.pull.pull_bge_reranker_snapshot",
        selected.append,
    )

    main()

    assert selected == [tmp_path]
