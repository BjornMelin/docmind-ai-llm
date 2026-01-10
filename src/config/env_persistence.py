"""Helpers for persisting environment variables to .env."""

from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path

from dotenv import set_key, unset_key

_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def _validate_env_key(key: str) -> None:
    if not key or not _KEY_RE.fullmatch(key):
        raise ValueError("Invalid env var key; expected [A-Z][A-Z0-9_]*")


def _validate_env_value(value: str) -> None:
    if _CONTROL_CHARS_RE.search(value):
        raise ValueError("Env var value contains control characters")


def persist_env(vars_to_set: dict[str, str], *, env_path: Path | None = None) -> None:
    """Persist key=value pairs into the project's .env file.

    Creates the file if it does not exist. Empty values remove the key.
    """
    target = env_path or Path(".env")
    if not target.exists():
        with suppress(FileExistsError):  # pragma: no cover - race-safe
            target.open("x", encoding="utf-8").close()

    for key, value in vars_to_set.items():
        _validate_env_key(key)
        if value == "":
            unset_key(str(target), key)
        else:
            _validate_env_value(value)
            set_key(str(target), key, value, quote_mode="auto")
