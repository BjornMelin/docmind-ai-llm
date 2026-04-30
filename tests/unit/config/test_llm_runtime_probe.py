"""Tests for LLM runtime endpoint probes."""

from __future__ import annotations

import pytest

from src.config import llm_runtime_probe


class _Response:
    """Small httpx.Response stand-in for probe tests."""

    def __init__(self, status_code: int, payload: object | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        """Return configured JSON payload."""
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.mark.unit
def test_llamacpp_probe_checks_health_before_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llama.cpp probe checks `/health` before `/v1/models`."""
    requests: list[tuple[str, dict[str, str] | None]] = []

    class _Client:
        def __init__(self, *, timeout: object) -> None:
            self.timeout = timeout

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(
            self,
            url: object,
            *,
            headers: dict[str, str] | None = None,
        ) -> _Response:
            raw = str(url)
            requests.append((raw, headers))
            if raw.endswith("/health"):
                return _Response(200, {"status": "ok"})
            return _Response(200, {"data": [{"id": "local-gguf"}]})

    monkeypatch.setattr(llm_runtime_probe.httpx, "Client", _Client)

    result = llm_runtime_probe.probe_openai_compatible_runtime(
        base_url="http://localhost:8080/v1",
        backend="llamacpp",
        headers={"Authorization": "Bearer local"},
        timeout_s=5.0,
    )

    assert result.ok is True
    assert result.models_count == 1
    assert requests == [
        ("http://localhost:8080/health", {"Authorization": "Bearer local"}),
        ("http://localhost:8080/v1/models", {"Authorization": "Bearer local"}),
    ]


@pytest.mark.unit
def test_llamacpp_probe_reports_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """llama.cpp loading status should not fall through to `/models`."""
    calls = 0

    class _Client:
        def __init__(self, *, timeout: object) -> None:
            self.timeout = timeout

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(
            self,
            url: object,
            *,
            headers: dict[str, str] | None = None,
        ) -> _Response:
            nonlocal calls
            calls += 1
            return _Response(503, {"error": {"message": "Loading model"}})

    monkeypatch.setattr(llm_runtime_probe.httpx, "Client", _Client)

    result = llm_runtime_probe.probe_openai_compatible_runtime(
        base_url="http://localhost:8080/v1",
        backend="llamacpp",
    )

    assert result.ok is False
    assert "loading" in result.message
    assert calls == 1


@pytest.mark.unit
def test_openai_compatible_probe_uses_models_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic OpenAI-compatible probe should only call `/models`."""
    requests: list[str] = []

    class _Client:
        def __init__(self, *, timeout: object) -> None:
            self.timeout = timeout

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def get(
            self,
            url: object,
            *,
            headers: dict[str, str] | None = None,
        ) -> _Response:
            requests.append(str(url))
            return _Response(200, {"data": []})

    monkeypatch.setattr(llm_runtime_probe.httpx, "Client", _Client)

    result = llm_runtime_probe.probe_openai_compatible_runtime(
        base_url="http://localhost:1234/v1",
        backend="lmstudio",
    )

    assert result.ok is True
    assert result.models_count == 0
    assert requests == ["http://localhost:1234/v1/models"]
