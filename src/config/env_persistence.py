"""Helpers for persisting environment variables to .env."""

from __future__ import annotations

from pathlib import Path

from dotenv import set_key, unset_key


def persist_env(vars_to_set: dict[str, str], *, env_path: Path | None = None) -> None:
    """Persist key=value pairs into the project's .env file.

    Creates the file if it does not exist. Empty values remove the key.
    """
    target = env_path or Path(".env")
    if not target.exists():
        target.write_text("", encoding="utf-8")

    for key, value in vars_to_set.items():
        if value == "":
            unset_key(str(target), key)
        else:
            set_key(str(target), key, value, quote_mode="auto")
