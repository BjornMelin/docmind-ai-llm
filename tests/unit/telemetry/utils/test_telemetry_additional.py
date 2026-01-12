"""Additional telemetry tests: disabled, sampling, rotation, and errors."""

from __future__ import annotations

import json
from pathlib import Path


def test_disabled_env_no_write(monkeypatch, tmp_path):
    from src.utils import telemetry as t

    tp = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(t, "_TELEM_PATH", tp)
    monkeypatch.setenv("DOCMIND_TELEMETRY_DISABLED", "true")
    t.log_jsonl({"k": 1})
    assert not tp.exists()


def test_sampling_skips_when_rate_zero(monkeypatch, tmp_path):
    from src.utils import telemetry as t

    tp = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(t, "_TELEM_PATH", tp)
    monkeypatch.setenv("DOCMIND_TELEMETRY_SAMPLE", "0.0")
    t.log_jsonl({"k": 2})
    assert not tp.exists()


def test_rotation_and_write(monkeypatch, tmp_path):
    from src.utils import telemetry as t

    tp = tmp_path / "telemetry.jsonl"
    rot = tmp_path / "telemetry.jsonl.1"
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_bytes(b"x" * 10)
    # Set low rotate threshold
    monkeypatch.setenv("DOCMIND_TELEMETRY_ROTATE_BYTES", "4")
    monkeypatch.setattr(t, "_TELEM_PATH", tp)
    # Ensure enabled and full sample
    monkeypatch.delenv("DOCMIND_TELEMETRY_DISABLED", raising=False)
    monkeypatch.setenv("DOCMIND_TELEMETRY_SAMPLE", "1.0")

    t.log_jsonl({"k": 3})
    # Rotated file exists
    assert rot.exists()
    # New file has one JSON line
    data = tp.read_text(encoding="utf-8").strip().splitlines()
    assert len(data) == 1
    rec = json.loads(data[0])
    assert rec["k"] == 3


def test_rotation_rename_error_is_debug_logged(monkeypatch, tmp_path):
    from src.utils import telemetry as t

    class _Stat:
        st_size = 999

    class _P(Path):
        # Path subclass to override methods used
        _flavour = Path(".")._flavour  # type: ignore[attr-defined]

        def exists(self):
            return True

        def stat(self):
            return _Stat()

        def with_suffix(self, sfx: str):
            return _P(str(self) + sfx)

        def rename(self, _dst):
            raise OSError("nope")

    tp = _P(str(tmp_path / "telemetry.jsonl"))
    monkeypatch.setattr(t, "_TELEM_PATH", tp)
    monkeypatch.setenv("DOCMIND_TELEMETRY_ROTATE_BYTES", "1")
    monkeypatch.setenv("DOCMIND_TELEMETRY_SAMPLE", "1.0")

    calls: list[str] = []

    def _dbg(msg: str) -> None:
        calls.append(str(msg))

    monkeypatch.setattr(t.logger, "debug", _dbg, raising=False)
    t.log_jsonl({"k": 4})
    # Should not raise; debug log emitted about rotation skip
    assert any("telemetry rotation skipped" in msg for msg in calls)
