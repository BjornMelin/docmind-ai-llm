"""Helpers for persisting environment variables to `.env`.

This helper is intended for DocMind-owned environment variables. Keys are
validated as `[A-Z][A-Z0-9_]*`.
"""

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
    """Reject values containing ASCII control characters.

    This intentionally focuses on ASCII control bytes (0x00-0x1F plus 0x7F),
    which are the most common source of broken `.env` parsing. Unicode control
    characters (e.g. U+2028/U+2029) are not currently rejected.
    """
    if _CONTROL_CHARS_RE.search(value):
        raise ValueError("Env var value contains control characters")


class EnvPersistError(RuntimeError):
    """Raised when persisting a specific env key fails."""

    def __init__(self, key: str, message: str):
        super().__init__(message)
        self.key = key


def persist_env(vars_to_set: dict[str, str], *, env_path: Path | None = None) -> None:
    """Persist key=value pairs into the project's .env file.

    Creates the file if it does not exist. Empty values remove the key.

    Raises:
        EnvPersistError: When a specific key cannot be written/removed.
    """
    target = env_path or Path(".env")
    if not target.exists():
        with suppress(FileExistsError):  # pragma: no cover - race-safe
            target.open("x", encoding="utf-8").close()

    for key, value in vars_to_set.items():
        _validate_env_key(key)
        try:
            if value == "":
                unset_key(str(target), key)
            else:
                _validate_env_value(value)
                set_key(str(target), key, value, quote_mode="auto")
        except (OSError, RuntimeError, ValueError) as exc:
            raise EnvPersistError(key=key, message=f"key={key}: {exc}") from exc
