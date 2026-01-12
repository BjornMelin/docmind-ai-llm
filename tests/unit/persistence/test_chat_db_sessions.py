"""Unit tests for chat_db session registry + purge helpers (ADR-057/SPEC-041)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from src.config.settings import DocMindSettings
from src.persistence.chat_db import (
    create_session,
    ensure_session_registry,
    list_sessions,
    open_chat_db,
    purge_session,
    rename_session,
    soft_delete_session,
    touch_session,
)

pytestmark = pytest.mark.unit


def test_open_chat_db_anchors_relative_paths_under_data_dir(tmp_path: Path) -> None:
    cfg = cast("DocMindSettings", type("_Cfg", (), {"data_dir": tmp_path})())
    conn = open_chat_db(Path("data/chat.db"), cfg=cfg)  # legacy default shape
    try:
        assert Path(conn.execute("PRAGMA database_list;").fetchone()[2]).exists()
        ensure_session_registry(conn)
        created = create_session(title="A", conn=conn)
        assert created.thread_id
        assert list_sessions(conn)[0].title == "A"
    finally:
        conn.close()


def test_session_crud_and_purge(tmp_path: Path) -> None:
    cfg = cast("DocMindSettings", type("_Cfg", (), {"data_dir": tmp_path})())
    conn = open_chat_db(Path("chat.db"), cfg=cfg)
    try:
        ensure_session_registry(conn)
        created = create_session(title="First", conn=conn)

        rename_session(conn, thread_id=created.thread_id, title="Renamed")
        sessions = list_sessions(conn)
        assert sessions[0].title == "Renamed"

        touch_session(conn, thread_id=created.thread_id, last_checkpoint_id="c1")
        sessions = list_sessions(conn)
        assert sessions[0].last_checkpoint_id == "c1"

        soft_delete_session(conn, thread_id=created.thread_id)
        assert list_sessions(conn) == []
        deleted_sessions = list_sessions(conn, include_deleted=True)
        assert len(deleted_sessions) == 1
        assert deleted_sessions[0].thread_id == created.thread_id
        assert deleted_sessions[0].deleted_at_ms is not None

        # Create minimal LangGraph tables expected by purge_session.
        conn.execute("CREATE TABLE IF NOT EXISTS writes (thread_id TEXT);")
        conn.execute("CREATE TABLE IF NOT EXISTS checkpoints (thread_id TEXT);")
        conn.commit()

        purge_session(conn, thread_id=created.thread_id)
        assert list_sessions(conn, include_deleted=True) == []
    finally:
        conn.close()
