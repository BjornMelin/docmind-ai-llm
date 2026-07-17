"""Unit tests for model pull CLI (tools/models/pull.py)."""

from __future__ import annotations

import argparse
import builtins
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
    _SIGLIP_TRANSFORMERS_FILES,
    _resolve_cache_dirs,
    main,
    pull,
    pull_bge_m3_snapshot,
    pull_bge_reranker_snapshot,
    pull_bm42_snapshot,
    pull_siglip_snapshot,
    resolve_bm42_snapshot,
)


def test_explicit_active_cache_does_not_import_settings_for_unused_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Keep Docker's partial-source prefetch layers independent of settings."""
    real_import = builtins.__import__

    def _reject_settings_import(
        name: str,
        globals_arg=None,
        locals_arg=None,
        fromlist=(),
        level: int = 0,
    ):
        if name == "src.config.settings":
            raise AssertionError("unused cache imported application settings")
        return real_import(name, globals_arg, locals_arg, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _reject_settings_import)
    model_cache = tmp_path / "models"
    parser_cache = tmp_path / "parser"
    common = {
        "bge_m3": False,
        "bge_reranker": False,
        "bm42": False,
        "add": None,
        "docling_layout": False,
    }

    resolved_model, _ = _resolve_cache_dirs(
        argparse.Namespace(
            **common,
            all=True,
            parser_defaults=False,
            cache_dir=str(model_cache),
            parser_cache_dir=None,
        )
    )
    _, resolved_parser = _resolve_cache_dirs(
        argparse.Namespace(
            **common,
            all=False,
            parser_defaults=True,
            cache_dir=None,
            parser_cache_dir=str(parser_cache),
        )
    )

    assert resolved_model == model_cache.resolve()
    assert resolved_parser == parser_cache.resolve()


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


def test_cli_default_uses_configured_embedding_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Omitted cache overrides follow bootstrapped canonical settings."""
    from importlib import import_module

    settings_module = import_module("src.config.settings")

    configured_models = tmp_path / "configured-models"
    configured_parser = tmp_path / "configured-parser"
    (tmp_path / ".env").write_text(
        "\n".join(
            (
                f"DOCMIND_EMBEDDING__CACHE_FOLDER={configured_models}",
                f"DOCMIND_PARSING__MODEL_CACHE_DIR={configured_parser}",
            )
        ),
        encoding="utf-8",
    )
    snapshots: list[Path] = []
    parser_caches: list[Path] = []
    monkeypatch.chdir(tmp_path)
    settings_module.reset_bootstrap_state()
    monkeypatch.setattr("sys.argv", ["pull.py", "--bge-m3", "--parser-defaults"])
    monkeypatch.setattr("tools.models.pull.pull_bge_m3_snapshot", snapshots.append)
    monkeypatch.setattr(
        "tools.models.pull.pull_docling_layout",
        lambda path, *, force=False: parser_caches.append(path),
    )

    try:
        main()
    finally:
        settings_module.settings.__init__(_env_file=None)  # type: ignore[arg-type]
        settings_module.reset_bootstrap_state()

    assert snapshots == [configured_models.resolve()]
    assert parser_caches == [configured_parser.resolve()]


def test_explicit_cache_overrides_skip_settings_bootstrap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Explicit model and parser destinations remain authoritative."""
    from importlib import import_module
    from unittest.mock import Mock

    settings_module = import_module("src.config.settings")

    bootstrap = Mock()
    model_cache = tmp_path / "models"
    parser_cache = tmp_path / "parser"
    monkeypatch.setattr(settings_module, "bootstrap_settings", bootstrap)
    monkeypatch.setattr(
        "sys.argv",
        [
            "pull.py",
            "--bge-m3",
            "--cache_dir",
            str(model_cache),
            "--parser-defaults",
            "--parser-cache-dir",
            str(parser_cache),
        ],
    )
    snapshots: list[Path] = []
    parser_caches: list[Path] = []
    monkeypatch.setattr("tools.models.pull.pull_bge_m3_snapshot", snapshots.append)
    monkeypatch.setattr(
        "tools.models.pull.pull_docling_layout",
        lambda path, *, force=False: parser_caches.append(path),
    )

    main()

    bootstrap.assert_not_called()
    assert snapshots == [model_cache.resolve()]
    assert parser_caches == [parser_cache.resolve()]


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


def test_pull_bge_reranker_fetches_the_complete_pinned_snapshot(
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
    assert "allow_patterns" not in call
    assert call["local_files_only"] is False


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
