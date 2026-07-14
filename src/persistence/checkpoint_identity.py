"""Canonical user-scoped identity for LangGraph checkpoints."""

from __future__ import annotations

import hashlib

_MEMORY_USER_DOMAIN = b"docmind:memory-namespace:user:v1\0"
_MEMORY_THREAD_DOMAIN = b"docmind:memory-namespace:thread:v1\0"


def _memory_namespace_component(*, domain: bytes, prefix: str, value: str) -> str:
    digest = hashlib.sha256(domain + str(value).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def checkpoint_thread_id(*, thread_id: str, user_id: str) -> str:
    """Return the opaque LangGraph key for one public user/thread pair."""
    user = str(user_id)
    payload = f"{len(user)}:{user}{thread_id}".encode()
    return f"docmind:{hashlib.sha256(payload).hexdigest()}"


def memory_namespace(*, user_id: str, thread_id: str | None = None) -> tuple[str, ...]:
    """Return the opaque canonical LangGraph memory namespace."""
    user_component = _memory_namespace_component(
        domain=_MEMORY_USER_DOMAIN,
        prefix="u",
        value=user_id,
    )
    if thread_id is None:
        return ("memories", "user", user_component)
    thread_component = _memory_namespace_component(
        domain=_MEMORY_THREAD_DOMAIN,
        prefix="t",
        value=thread_id,
    )
    return ("memories", "session", user_component, thread_component)


def memory_namespace_prefix(*, user_id: str, thread_id: str) -> str:
    """Encode one exact session namespace as native SqliteStore does."""
    return ".".join(memory_namespace(user_id=user_id, thread_id=thread_id))


def memory_id(content: str, kind: str) -> str:
    """Return the canonical content-addressed key for one memory."""
    identity = f"{str(kind).strip().casefold()}\0{str(content).strip().casefold()}"
    return f"mem-{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"
