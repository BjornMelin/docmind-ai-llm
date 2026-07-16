"""Tests for the offline collection cleanup CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest

from scripts import cleanup_collections as command


class _Client:
    def __init__(
        self,
        *,
        close_error: Exception | None = None,
        **_kwargs: object,
    ) -> None:
        self.closed = False
        self.close_error = close_error

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture(autouse=True)
def _stub_bootstrap_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(command, "bootstrap_settings", lambda: None)


def _stub_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    client: _Client,
    cleanup: Callable[..., object],
) -> None:
    monkeypatch.setattr(command, "QdrantClient", lambda **_kwargs: client)
    monkeypatch.setattr(command, "get_client_config", lambda _cfg: {})
    monkeypatch.setattr(command, "cleanup_orphan_collections", cleanup)


def _read_error(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    captured = capsys.readouterr()
    assert captured.out == ""
    return cast(dict[str, object], json.loads(captured.err))


def _record_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure_stage: str | None = None,
) -> list[str]:
    events: list[str] = []

    def _record(stage: str) -> None:
        events.append(stage)
        if failure_stage == stage:
            raise RuntimeError(f"{stage} failed")

    def _client_config(_cfg: object) -> dict[str, object]:
        _record("config")
        return {}

    def _client(**_kwargs: object) -> _Client:
        _record("client")
        return _Client()

    def _cleanup(*_args: object, **_kwargs: object) -> SimpleNamespace:
        _record("cleanup")
        return SimpleNamespace(as_dict=lambda: {"status": "ok"})

    monkeypatch.setattr(command, "bootstrap_settings", lambda: _record("bootstrap"))
    monkeypatch.setattr(command, "get_client_config", _client_config)
    monkeypatch.setattr(command, "QdrantClient", _client)
    monkeypatch.setattr(command, "cleanup_orphan_collections", _cleanup)
    return events


def test_cli_requires_explicit_quiescence_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    side_effects = _record_lifecycle(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        command.main([])

    assert exc_info.value.code == 2
    assert side_effects == []


def test_cli_bootstraps_before_reading_client_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _record_lifecycle(monkeypatch)

    assert command.main(["--confirm-app-stopped"]) == 0

    assert events == ["bootstrap", "config", "client", "cleanup"]


@pytest.mark.parametrize(
    ("failure_stage", "expected_events"),
    [
        ("bootstrap", ["bootstrap"]),
        ("config", ["bootstrap", "config"]),
        ("client", ["bootstrap", "config", "client"]),
    ],
)
def test_cli_reports_setup_failures_without_tracebacks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_stage: str,
    expected_events: list[str],
) -> None:
    events = _record_lifecycle(monkeypatch, failure_stage=failure_stage)

    assert command.main(["--confirm-app-stopped"]) == 2

    assert _read_error(capsys) == {
        "error": f"{failure_stage} failed",
        "error_type": "RuntimeError",
        "status": "error",
    }
    assert events == expected_events


def test_cli_defaults_to_dry_run_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _Client()
    cleanup_calls: list[bool] = []

    def _cleanup(_client: object, *, delete: bool, cfg: object) -> SimpleNamespace:
        del cfg
        assert _client is client
        cleanup_calls.append(delete)
        return SimpleNamespace(as_dict=lambda: {"status": "ok", "mode": "dry-run"})

    _stub_runtime(monkeypatch, client=client, cleanup=_cleanup)

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
    client = _Client(close_error=RuntimeError("close failed"))

    def _cleanup(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("blocked")

    _stub_runtime(monkeypatch, client=client, cleanup=_cleanup)

    assert command.main(["--confirm-app-stopped", "--delete"]) == 2

    assert client.closed is True
    assert _read_error(capsys) == {
        "error": "blocked",
        "error_type": "RuntimeError",
        "status": "error",
    }


def test_cli_reports_close_failure_without_success_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _Client(close_error=RuntimeError("close failed"))

    completed_result = {
        "status": "ok",
        "mode": "delete",
        "deleted_collections": ["old-generation"],
    }

    def _cleanup(
        *_args: object,
        delete: bool,
        **_kwargs: object,
    ) -> SimpleNamespace:
        assert delete is True
        return SimpleNamespace(as_dict=lambda: completed_result)

    _stub_runtime(monkeypatch, client=client, cleanup=_cleanup)

    assert command.main(["--confirm-app-stopped", "--delete"]) == 2

    assert client.closed is True
    assert _read_error(capsys) == {
        "error": "close failed",
        "error_type": "RuntimeError",
        "result": completed_result,
        "status": "error",
    }
