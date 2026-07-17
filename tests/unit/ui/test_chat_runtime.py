"""Lifecycle tests for the process-wide Chat coordinator owner."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.agents import coordinator as coordinator_module
from src.ui import chat_runtime
from src.ui.router_session import replace_session_router, session_router_is_current

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_runtime() -> None:
    chat_runtime.invalidate_coordinator()
    yield
    chat_runtime.invalidate_coordinator()


def _write_tokenizer_assets(
    model_root: Path,
    payload_names: tuple[str, ...] = ("tokenizer.json",),
) -> None:
    (model_root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    for payload_name in payload_names:
        (model_root / payload_name).write_bytes(b"tokenizer")


def test_invalidate_without_resource_does_not_construct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[object] = []
    monkeypatch.setattr(
        coordinator_module,
        "MultiAgentCoordinator",
        lambda **_kwargs: constructed.append(object()),
    )

    chat_runtime.invalidate_coordinator()

    assert constructed == []


def test_local_model_readiness_requires_config_weights_and_tokenizer(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()

    assert chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    ) == chat_runtime.ChatModelReadiness(status="local_path_incomplete")

    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"model")

    assert chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    ) == chat_runtime.ChatModelReadiness(status="local_path_incomplete")

    _write_tokenizer_assets(model_path)

    assert chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    ) == chat_runtime.ChatModelReadiness(status="ready")


@pytest.mark.parametrize(
    "payload_names",
    [
        ("tokenizer.json",),
        ("sentencepiece.bpe.model",),
        ("spiece.model",),
        ("tokenizer.model",),
        ("sentencepiece.model",),
        ("spm.model",),
        ("prophetnet.tokenizer",),
        ("vocab.txt",),
        ("vocab.json", "merges.txt"),
        ("source.spm", "target.spm", "vocab.json"),
    ],
    ids=[
        "fast-tokenizer",
        "sentencepiece-bpe",
        "sentencepiece-short",
        "sentencepiece-generic",
        "sentencepiece-rembert",
        "sentencepiece-deberta-v2",
        "prophetnet",
        "wordpiece",
        "byte-bpe-pair",
        "marian-shared-vocab",
    ],
)
def test_local_model_readiness_accepts_supported_tokenizer_families(
    tmp_path: Path,
    payload_names: tuple[str, ...],
) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"model")
    _write_tokenizer_assets(model_path, payload_names)

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="ready")


@pytest.mark.parametrize("empty_name", ["tokenizer_config.json", "tokenizer.json"])
def test_local_model_readiness_rejects_empty_tokenizer_assets(
    tmp_path: Path,
    empty_name: str,
) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"model")
    _write_tokenizer_assets(model_path)
    (model_path / empty_name).write_bytes(b"")

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="local_path_incomplete")


@pytest.mark.parametrize(
    ("nonempty_names", "empty_names"),
    [
        (("vocab.json",), ()),
        (("merges.txt",), ()),
        (("vocab.json",), ("merges.txt",)),
        (("merges.txt",), ("vocab.json",)),
    ],
)
def test_local_model_readiness_rejects_incomplete_vocab_merges_pair(
    tmp_path: Path,
    nonempty_names: tuple[str, ...],
    empty_names: tuple[str, ...],
) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"model")
    (model_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    for name in nonempty_names:
        (model_path / name).write_bytes(b"tokenizer")
    for name in empty_names:
        (model_path / name).touch()

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="local_path_incomplete")


@pytest.mark.parametrize("missing_name", ["source.spm", "target.spm", "vocab.json"])
def test_local_model_readiness_rejects_incomplete_marian_payload(
    tmp_path: Path,
    missing_name: str,
) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"model_type": "marian"}',
        encoding="utf-8",
    )
    (model_path / "model.safetensors").write_bytes(b"model")
    marian_payload = {"source.spm", "target.spm", "vocab.json"} - {missing_name}
    _write_tokenizer_assets(model_path, tuple(marian_payload))

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="local_path_incomplete")


def test_local_model_readiness_requires_separate_marian_target_vocab(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"model_type": "marian"}',
        encoding="utf-8",
    )
    (model_path / "model.safetensors").write_bytes(b"model")
    _write_tokenizer_assets(model_path, ("source.spm", "target.spm", "vocab.json"))
    (model_path / "tokenizer_config.json").write_text(
        '{"separate_vocabs": true}',
        encoding="utf-8",
    )

    assert chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    ) == chat_runtime.ChatModelReadiness(status="local_path_incomplete")

    (model_path / "target_vocab.json").write_bytes(b"tokenizer")

    assert chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    ) == chat_runtime.ChatModelReadiness(status="ready")


def test_local_model_readiness_requires_assets_in_one_model_root(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "embedding"
    transformer = model_path / "0_Transformer"
    transformer.mkdir(parents=True)
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"model")
    _write_tokenizer_assets(transformer)

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="local_path_incomplete")


def test_local_model_readiness_rejects_tokenizer_symlink_escape(tmp_path: Path) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"model")
    (model_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside-tokenizer.json"
    outside.write_bytes(b"tokenizer")
    try:
        (model_path / "tokenizer.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {type(exc).__name__}")

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="local_path_incomplete")


def test_local_model_readiness_rejects_config_symlink_escape(tmp_path: Path) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    outside = tmp_path / "outside-config.json"
    outside.write_text('{"model_type": "bert"}', encoding="utf-8")
    try:
        (model_path / "config.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {type(exc).__name__}")
    (model_path / "model.safetensors").write_bytes(b"model")
    _write_tokenizer_assets(model_path)

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="local_path_incomplete")


@pytest.mark.parametrize(
    "index_name",
    ["model.safetensors.index.json", "pytorch_model.bin.index.json"],
)
def test_local_model_readiness_accepts_complete_sharded_weights(
    tmp_path: Path,
    index_name: str,
) -> None:
    model_path = tmp_path / "embedding"
    shard_dir = model_path / "weights"
    shard_dir.mkdir(parents=True)
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    _write_tokenizer_assets(model_path)
    (shard_dir / "shard-01.bin").write_bytes(b"first")
    (shard_dir / "shard-02.bin").write_bytes(b"second")
    (model_path / index_name).write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "weights/shard-01.bin",
                    "layer.1": "weights/shard-02.bin",
                    "layer.2": "weights/shard-01.bin",
                }
            }
        ),
        encoding="utf-8",
    )

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="ready")


def test_local_model_readiness_accepts_shard_symlink_within_model_root(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "embedding"
    transformer = model_path / "0_Transformer"
    shared = model_path / "shared"
    transformer.mkdir(parents=True)
    shared.mkdir()
    (transformer / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    _write_tokenizer_assets(transformer)
    target = shared / "shard.safetensors"
    target.write_bytes(b"model shard")
    shard = transformer / "model-00001-of-00001.safetensors"
    try:
        shard.symlink_to(Path(os.path.relpath(target, start=transformer)))
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {type(exc).__name__}")
    (transformer / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": shard.name}}),
        encoding="utf-8",
    )

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="ready")


@pytest.mark.parametrize("shard_state", ["missing", "empty"])
def test_local_model_readiness_rejects_incomplete_sharded_weights(
    tmp_path: Path,
    shard_state: str,
) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    _write_tokenizer_assets(model_path)
    if shard_state == "empty":
        (model_path / "shard.bin").touch()
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "shard.bin"}}),
        encoding="utf-8",
    )

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="local_path_incomplete")


@pytest.mark.parametrize(
    "manifest",
    [
        "not-json",
        "[]",
        '{"weight_map": []}',
        '{"weight_map": {}}',
        '{"weight_map": {"layer": ""}}',
        '{"weight_map": {"layer": 1}}',
    ],
)
def test_local_model_readiness_rejects_malformed_weight_index(
    tmp_path: Path,
    manifest: str,
) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    _write_tokenizer_assets(model_path)
    (model_path / "model.safetensors.index.json").write_text(manifest, encoding="utf-8")

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="local_path_incomplete")


@pytest.mark.parametrize(
    "path_kind",
    ["parent-traversal", "absolute"],
)
def test_local_model_readiness_rejects_non_relative_shard_paths(
    tmp_path: Path,
    path_kind: str,
) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    _write_tokenizer_assets(model_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    shard_name = "../outside.bin" if path_kind == "parent-traversal" else str(outside)
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": shard_name}}),
        encoding="utf-8",
    )

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="local_path_incomplete")


def test_local_model_readiness_rejects_shard_symlink_escape(tmp_path: Path) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    _write_tokenizer_assets(model_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    shard = model_path / "shard.bin"
    try:
        shard.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {type(exc).__name__}")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": shard.name}}),
        encoding="utf-8",
    )

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=model_path,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="local_path_incomplete")


def test_local_model_readiness_expands_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"model")
    _write_tokenizer_assets(model_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    readiness = chat_runtime.check_model_artifacts(
        model_name="unused",
        model_revision=None,
        cache_folder=Path("~/cache"),
        local_model_path=Path("~/embedding"),
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="ready")


@pytest.mark.parametrize("complete", [False, True])
def test_cached_model_readiness_validates_returned_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    complete: bool,
) -> None:
    from huggingface_hub import snapshot_download

    cache_root = tmp_path / "cache"
    snapshot = cache_root / "snapshot"
    snapshot.mkdir(parents=True)
    if complete:
        (snapshot / "config.json").write_text(
            '{"model_type": "bert"}', encoding="utf-8"
        )
        (snapshot / "model.safetensors").write_bytes(b"model")
        _write_tokenizer_assets(snapshot)
    calls: list[dict[str, object]] = []

    def _download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _download)

    readiness = chat_runtime.check_model_artifacts(
        model_name="org/model",
        model_revision="revision",
        cache_folder=cache_root,
        local_model_path=None,
    )

    assert readiness.status == ("ready" if complete else "cache_missing")
    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "revision",
            "cache_dir": cache_root,
            "local_files_only": True,
        }
    ]
    assert snapshot_download is not _download


@pytest.mark.parametrize("weight_location", ["cache", "outside"])
def test_cached_model_readiness_contains_direct_weight_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    weight_location: str,
) -> None:
    cache_root = tmp_path / "cache"
    model_cache = cache_root / "models--org--model"
    snapshot = model_cache / "snapshots" / "revision"
    blobs = model_cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (snapshot / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    _write_tokenizer_assets(snapshot)
    inside_weight = blobs / "weight-hash"
    inside_weight.write_bytes(b"cached weight")
    outside_weight = tmp_path / "outside-weight"
    outside_weight.write_bytes(b"outside weight")
    target = inside_weight if weight_location == "cache" else outside_weight
    try:
        (snapshot / "model.safetensors").symlink_to(
            Path(os.path.relpath(target, start=snapshot))
        )
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {type(exc).__name__}")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: str(snapshot),
    )

    readiness = chat_runtime.check_model_artifacts(
        model_name="org/model",
        model_revision="revision",
        cache_folder=cache_root,
        local_model_path=None,
    )

    expected = "ready" if weight_location == "cache" else "cache_missing"
    assert readiness == chat_runtime.ChatModelReadiness(status=expected)


@pytest.mark.parametrize("shard_location", ["cache", "outside"])
def test_cached_model_readiness_contains_snapshot_shard_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shard_location: str,
) -> None:
    cache_root = tmp_path / "cache"
    model_cache = cache_root / "models--org--model"
    snapshot = model_cache / "snapshots" / "revision"
    blobs = model_cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    config_blob = blobs / "config-hash"
    config_blob.write_text('{"model_type": "bert"}', encoding="utf-8")
    tokenizer_config_blob = blobs / "tokenizer-config-hash"
    tokenizer_config_blob.write_text("{}", encoding="utf-8")
    inside_blob = blobs / "shard-hash"
    inside_blob.write_bytes(b"cached shard")
    tokenizer_blob = blobs / "tokenizer-hash"
    tokenizer_blob.write_bytes(b"cached tokenizer")
    outside_blob = tmp_path / "outside-shard"
    outside_blob.write_bytes(b"outside shard")
    target = inside_blob if shard_location == "cache" else outside_blob
    shard = snapshot / "model-00001-of-00001.safetensors"
    try:
        (snapshot / "config.json").symlink_to(
            Path(os.path.relpath(config_blob, start=snapshot))
        )
        (snapshot / "tokenizer_config.json").symlink_to(
            Path(os.path.relpath(tokenizer_config_blob, start=snapshot))
        )
        shard.symlink_to(Path(os.path.relpath(target, start=snapshot)))
        (snapshot / "tokenizer.json").symlink_to(
            Path(os.path.relpath(tokenizer_blob, start=snapshot))
        )
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {type(exc).__name__}")
    index_blob = blobs / "weight-index-hash"
    index_blob.write_text(
        json.dumps({"weight_map": {"layer": shard.name}}),
        encoding="utf-8",
    )
    try:
        (snapshot / "model.safetensors.index.json").symlink_to(
            Path(os.path.relpath(index_blob, start=snapshot))
        )
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {type(exc).__name__}")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: str(snapshot),
    )

    readiness = chat_runtime.check_model_artifacts(
        model_name="org/model",
        model_revision="revision",
        cache_folder=cache_root,
        local_model_path=None,
    )

    expected = "ready" if shard_location == "cache" else "cache_missing"
    assert readiness == chat_runtime.ChatModelReadiness(status=expected)


def test_cached_model_readiness_rejects_weight_index_symlink_escape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    snapshot = cache_root / "models--org--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    _write_tokenizer_assets(snapshot)
    (snapshot / "shard.safetensors").write_bytes(b"model shard")
    outside_index = tmp_path / "outside-index.json"
    outside_index.write_text(
        json.dumps({"weight_map": {"layer": "shard.safetensors"}}),
        encoding="utf-8",
    )
    try:
        (snapshot / "model.safetensors.index.json").symlink_to(outside_index)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {type(exc).__name__}")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: str(snapshot),
    )

    readiness = chat_runtime.check_model_artifacts(
        model_name="org/model",
        model_revision="revision",
        cache_folder=cache_root,
        local_model_path=None,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="cache_missing")


def test_cached_model_readiness_sanitizes_invalid_model_id(tmp_path: Path) -> None:
    readiness = chat_runtime.check_model_artifacts(
        model_name="private invalid model id",
        model_revision=None,
        cache_folder=tmp_path / "cache",
        local_model_path=None,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="initialization_failed")


def test_cached_model_readiness_sanitizes_cache_access_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fail_cache_access(**_kwargs: object) -> str:
        raise PermissionError("private cache path")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fail_cache_access)

    readiness = chat_runtime.check_model_artifacts(
        model_name="org/model",
        model_revision=None,
        cache_folder=tmp_path / "private-cache",
        local_model_path=None,
    )

    assert readiness == chat_runtime.ChatModelReadiness(status="initialization_failed")


def test_cached_model_readiness_propagates_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fail_unexpectedly(**_kwargs: object) -> str:
        raise RuntimeError("programmer failure")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fail_unexpectedly)

    with pytest.raises(RuntimeError, match="programmer failure"):
        chat_runtime.check_model_artifacts(
            model_name="org/model",
            model_revision=None,
            cache_folder=tmp_path / "cache",
            local_model_path=None,
        )


def test_same_generation_reuses_one_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Coordinator:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    monkeypatch.setattr(
        coordinator_module,
        "MultiAgentCoordinator",
        _Coordinator,
    )
    store = object()

    first = chat_runtime.get_coordinator(
        cache_version=1,
        checkpointer_path=Path("chat.db"),
        store=store,
    )
    second = chat_runtime.get_coordinator(
        cache_version=1,
        checkpointer_path=Path("chat.db"),
        store=store,
    )

    assert second is first
    assert first.close_count == 0


def test_new_generation_closes_previous_coordinator_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Coordinator:
        def __init__(self, **_kwargs: object) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    monkeypatch.setattr(
        coordinator_module,
        "MultiAgentCoordinator",
        _Coordinator,
    )
    store = object()
    first = chat_runtime.get_coordinator(
        cache_version=1,
        checkpointer_path=Path("chat.db"),
        store=store,
    )

    second = chat_runtime.get_coordinator(
        cache_version=2,
        checkpointer_path=Path("chat.db"),
        store=store,
    )

    assert second is not first
    assert first.close_count == 1
    chat_runtime.invalidate_coordinator()
    assert second.close_count == 1


def test_new_generation_retires_two_session_routers_before_old_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _Coordinator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            events.append("coordinator")

    class _Router:
        def __init__(self, name: str) -> None:
            self.name = name
            self.closed = False

        def close(self) -> None:
            if self.closed:
                return
            self.closed = True
            events.append(self.name)

    monkeypatch.setattr(
        coordinator_module,
        "MultiAgentCoordinator",
        _Coordinator,
    )
    store = object()
    chat_runtime.get_coordinator(
        cache_version=1,
        checkpointer_path=Path("chat.db"),
        store=store,
    )
    first_state: dict[str, object] = {}
    second_state: dict[str, object] = {}
    replace_session_router(
        first_state,
        _Router("router-a"),
        runtime_generation=1,
    )  # type: ignore[arg-type]
    replace_session_router(
        second_state,
        _Router("router-b"),
        runtime_generation=1,
    )  # type: ignore[arg-type]

    chat_runtime.get_coordinator(
        cache_version=2,
        checkpointer_path=Path("chat.db"),
        store=store,
    )

    assert events[:3] == ["router-a", "router-b", "coordinator"]
    assert not session_router_is_current(first_state, runtime_generation=2)
    assert not session_router_is_current(second_state, runtime_generation=2)


@pytest.mark.parametrize("failure_stage", ["retirement", "coordinator"])
def test_invalidate_detaches_coordinator_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    class _Coordinator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            if failure_stage == "coordinator":
                raise RuntimeError("close failed")

    monkeypatch.setattr(
        coordinator_module,
        "MultiAgentCoordinator",
        _Coordinator,
    )
    coordinator = chat_runtime.get_coordinator(
        cache_version=1,
        checkpointer_path=Path("chat.db"),
        store=object(),
    )
    if failure_stage == "retirement":
        monkeypatch.setattr(
            chat_runtime,
            "_retire_session_resources",
            lambda: (_ for _ in ()).throw(RuntimeError("retirement failed")),
        )

    chat_runtime.invalidate_coordinator()

    assert chat_runtime._COORDINATOR is None
    assert chat_runtime._RESOURCE_KEY is None
    assert coordinator is not chat_runtime._COORDINATOR
