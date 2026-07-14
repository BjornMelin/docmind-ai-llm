"""Unit tests for opaque persistence identity derivation."""

from __future__ import annotations

import re

import pytest

from src.persistence.checkpoint_identity import memory_namespace

pytestmark = pytest.mark.unit


def test_memory_namespace_components_are_fixed_width_and_domain_separated() -> None:
    session_namespace = memory_namespace(user_id="same%_user", thread_id="same%_user")
    user_namespace = memory_namespace(user_id="same%_user")

    assert session_namespace[:2] == ("memories", "session")
    assert user_namespace[:2] == ("memories", "user")
    assert session_namespace[: len(user_namespace)] != user_namespace
    assert re.fullmatch(r"u-[0-9a-f]{64}", session_namespace[2])
    assert re.fullmatch(r"t-[0-9a-f]{64}", session_namespace[3])
    assert session_namespace[2][2:] != session_namespace[3][2:]
    assert "same%_user" not in ".".join(session_namespace)


def test_memory_namespace_never_prefix_collides_for_raw_identifiers() -> None:
    short_user = memory_namespace(user_id="u", thread_id="thread")
    longer_user = memory_namespace(user_id="u2", thread_id="thread")
    short_thread = memory_namespace(user_id="u", thread_id="thread")
    longer_thread = memory_namespace(user_id="u", thread_id="thread-2")

    assert short_user[2] != longer_user[2]
    assert not longer_user[2].startswith(short_user[2])
    assert short_thread[3] != longer_thread[3]
    assert not longer_thread[3].startswith(short_thread[3])
