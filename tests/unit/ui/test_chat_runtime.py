"""Lifecycle tests for the process-wide Chat coordinator owner."""

from __future__ import annotations

import json
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


def test_local_model_readiness_requires_config_and_weights(tmp_path: Path) -> None:
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
    ) == chat_runtime.ChatModelReadiness(status="ready")


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


@pytest.mark.parametrize("shard_state", ["missing", "empty"])
def test_local_model_readiness_rejects_incomplete_sharded_weights(
    tmp_path: Path,
    shard_state: str,
) -> None:
    model_path = tmp_path / "embedding"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
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

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    if complete:
        (snapshot / "config.json").write_text(
            '{"model_type": "bert"}', encoding="utf-8"
        )
        (snapshot / "model.safetensors").write_bytes(b"model")
    calls: list[dict[str, object]] = []

    def _download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _download)

    readiness = chat_runtime.check_model_artifacts(
        model_name="org/model",
        model_revision="revision",
        cache_folder=Path("~/cache"),
        local_model_path=None,
    )

    assert readiness.status == ("ready" if complete else "cache_missing")
    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "revision",
            "cache_dir": Path("~/cache").expanduser(),
            "local_files_only": True,
        }
    ]
    assert snapshot_download is not _download


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
