"""Qdrant schema operator safety tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Literal

import pytest

from scripts import qdrant_schema
from src.utils.storage import CollectionCompatibilityResult

pytestmark = pytest.mark.unit


def _result(
    compatible: bool,
    action: Literal["unchanged", "created", "recreated", "blocked", "error"],
    reason: str,
    point_count: int | None,
) -> CollectionCompatibilityResult:
    return CollectionCompatibilityResult(
        compatible,
        action,
        reason,
        point_count,
    )


def test_check_never_calls_mutating_ensure(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = _result(False, "blocked", "text_dense_head_missing", 7)
    monkeypatch.setattr(qdrant_schema, "check_hybrid_collection", lambda *_: blocked)

    def _mutating_call(*_args: object) -> CollectionCompatibilityResult:
        raise AssertionError("check must not call ensure_hybrid_collection")

    monkeypatch.setattr(
        qdrant_schema,
        "rebuild_empty_hybrid_collection",
        _mutating_call,
    )

    result = qdrant_schema.inspect_or_rebuild(
        object(),  # type: ignore[arg-type]
        collection="documents",
        dense_dim=1024,
        rebuild_empty=False,
    )

    assert result is blocked


def test_rebuild_delegates_to_canonical_operator_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rebuilt = _result(True, "created", "compatible", 0)
    calls: list[tuple[str, int]] = []

    def _rebuild(
        _client: object,
        collection: str,
        dense_dim: int,
    ) -> CollectionCompatibilityResult:
        calls.append((collection, dense_dim))
        return rebuilt

    monkeypatch.setattr(qdrant_schema, "rebuild_empty_hybrid_collection", _rebuild)

    result = qdrant_schema.inspect_or_rebuild(
        object(),  # type: ignore[arg-type]
        collection="documents",
        dense_dim=1024,
        rebuild_empty=True,
    )

    assert result is rebuilt
    assert calls == [("documents", 1024)]


def test_rebuild_command_states_quiescence_requirement(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The destructive command explicitly warns that writers must be stopped."""

    class _Client:
        def close(self) -> None:
            return None

    monkeypatch.setattr(sys, "argv", ["qdrant_schema.py", "rebuild-empty"])
    monkeypatch.setattr(qdrant_schema, "QdrantClient", lambda **_: _Client())
    monkeypatch.setattr(
        qdrant_schema,
        "inspect_or_rebuild",
        lambda *_args, **_kwargs: _result(True, "recreated", "compatible", 0),
    )

    qdrant_schema.main()

    assert qdrant_schema.QUIESCENCE_NOTICE in capsys.readouterr().err


def _run_local_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: str,
    rest_binding: str = "127.0.0.1:7333",
    grpc_binding: str = "127.0.0.1:7334",
) -> tuple[CompletedProcess[str], str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$DOCKER_LOG"
case "${1:-}" in
  ps)
    if [[ "${2:-}" == "-a" ]]; then
      [[ "$DOCKER_STATE" != "missing" ]] && printf '%s\\n' "$DOCKER_NAME"
    else
      [[ "$DOCKER_STATE" == "running" ]] && printf '%s\\n' "$DOCKER_NAME"
    fi
    ;;
  port)
    if [[ "${3:-}" == "6333/tcp" ]]; then
      [[ "$DOCKER_REST_BINDING" != "missing" ]] || exit 1
      printf '%s\\n' "$DOCKER_REST_BINDING"
    else
      [[ "$DOCKER_GRPC_BINDING" != "missing" ]] || exit 1
      printf '%s\\n' "$DOCKER_GRPC_BINDING"
    fi
    ;;
esac
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    log = tmp_path / "docker.log"
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("DOCKER_LOG", str(log))
    monkeypatch.setenv("DOCKER_STATE", state)
    monkeypatch.setenv("DOCKER_NAME", "docmind-qdrant-local")
    monkeypatch.setenv("DOCKER_REST_BINDING", rest_binding)
    monkeypatch.setenv("DOCKER_GRPC_BINDING", grpc_binding)
    monkeypatch.setenv("DOCMIND_QDRANT_PORT", "7333")
    monkeypatch.setenv("DOCMIND_QDRANT_GRPC_PORT", "7334")
    monkeypatch.setenv("DOCMIND_QDRANT_STORAGE", str(tmp_path / "storage"))
    script = Path(__file__).resolve().parents[3] / "scripts" / "start_qdrant_local.sh"

    completed = run(
        ["bash", str(script)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, log.read_text(encoding="utf-8")


@pytest.mark.parametrize("state", ["running", "stopped"])
def test_local_launcher_reuses_only_exact_loopback_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    """Exact REST and gRPC mappings permit reuse of a known container name."""
    completed, docker_log = _run_local_launcher(
        tmp_path,
        monkeypatch,
        state=state,
    )

    assert completed.returncode == 0
    assert "port docmind-qdrant-local 6333/tcp" in docker_log
    assert "port docmind-qdrant-local 6334/tcp" in docker_log
    assert ("start docmind-qdrant-local" in docker_log) is (state == "stopped")
    assert not any(
        command.startswith(("stop ", "rm ")) for command in docker_log.splitlines()
    )


@pytest.mark.parametrize("state", ["running", "stopped"])
@pytest.mark.parametrize(
    ("rest_binding", "grpc_binding"),
    [
        ("0.0.0.0:7333", "127.0.0.1:7334"),
        ("[::]:7333", "127.0.0.1:7334"),
        ("127.0.0.1:7333\n0.0.0.0:7333", "127.0.0.1:7334"),
        ("127.0.0.1:7333", "missing"),
        ("127.0.0.1:7333", "0.0.0.0:7334"),
        ("127.0.0.1:6333", "127.0.0.1:7334"),
    ],
)
def test_local_launcher_refuses_unsafe_existing_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    rest_binding: str,
    grpc_binding: str,
) -> None:
    """Missing, public, or wrong-port mappings fail without container mutation."""
    completed, docker_log = _run_local_launcher(
        tmp_path,
        monkeypatch,
        state=state,
        rest_binding=rest_binding,
        grpc_binding=grpc_binding,
    )

    assert completed.returncode != 0
    assert "Refusing to reuse existing container" in completed.stderr
    assert "docker port docmind-qdrant-local" in completed.stderr
    assert not any(
        command.startswith(("stop ", "rm ", "start "))
        for command in docker_log.splitlines()
    )


def test_local_launcher_creates_loopback_only_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New containers publish configured REST and gRPC ports only on loopback."""
    completed, docker_log = _run_local_launcher(
        tmp_path,
        monkeypatch,
        state="missing",
    )

    assert completed.returncode == 0
    assert "-p 127.0.0.1:7333:6333" in docker_log
    assert "-p 127.0.0.1:7334:6334" in docker_log
