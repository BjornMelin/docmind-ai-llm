"""Container health gate tests."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest

from scripts import container_health

pytestmark = pytest.mark.unit


def test_container_health_checks_streamlit_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected: list[tuple[tuple[str, int], int]] = []

    def _connect(address: tuple[str, int], timeout: int):
        connected.append((address, timeout))
        return nullcontext()

    monkeypatch.setattr(container_health.socket, "create_connection", _connect)

    container_health.main()

    assert connected == [(("127.0.0.1", 8501), 3)]


def test_production_compose_uses_the_canonical_container_health_script() -> None:
    repository = Path(__file__).parents[3]
    compose = (repository / "docker-compose.prod.yml").read_text(encoding="utf-8")

    assert 'test: ["CMD", "python", "scripts/container_health.py"]' in compose
    assert "socket.create_connection" not in compose


def test_container_entrypoint_runs_parser_preflight_before_app() -> None:
    repository = Path(__file__).parents[3]
    lines = (
        (repository / "scripts/container_entrypoint.sh")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    assert lines.index("python scripts/parser_health.py --check") < lines.index(
        'exec "$@"'
    )
