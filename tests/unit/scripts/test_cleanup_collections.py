"""Tests for the offline collection cleanup CLI."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import cleanup_collections as command


class _Client:
    def __init__(self, **_kwargs: object) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_cli_requires_explicit_quiescence_acknowledgement() -> None:
    with pytest.raises(SystemExit) as exc_info:
        command.main([])

    assert exc_info.value.code == 2


def test_cli_defaults_to_dry_run_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _Client()
    monkeypatch.setattr(command, "QdrantClient", lambda **_kwargs: client)
    monkeypatch.setattr(command, "get_client_config", lambda _cfg: {})
    cleanup_calls: list[bool] = []

    def _cleanup(_client: object, *, delete: bool, cfg: object) -> SimpleNamespace:
        del cfg
        assert _client is client
        cleanup_calls.append(delete)
        return SimpleNamespace(as_dict=lambda: {"status": "ok", "mode": "dry-run"})

    monkeypatch.setattr(command, "cleanup_orphan_collections", _cleanup)

    assert command.main(["--confirm-app-stopped"]) == 0

    assert cleanup_calls == [False]
    assert client.closed is True
    assert json.loads(capsys.readouterr().out) == {
        "mode": "dry-run",
        "status": "ok",
    }


def test_cli_closes_client_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _Client()
    monkeypatch.setattr(command, "QdrantClient", lambda **_kwargs: client)
    monkeypatch.setattr(command, "get_client_config", lambda _cfg: {})
    monkeypatch.setattr(
        command,
        "cleanup_orphan_collections",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("blocked")),
    )

    assert command.main(["--confirm-app-stopped", "--delete"]) == 2

    assert client.closed is True
    assert json.loads(capsys.readouterr().err) == {
        "error": "blocked",
        "error_type": "RuntimeError",
        "status": "error",
    }
