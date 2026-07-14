"""Tests for stable deployment identity persistence."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest

from src.persistence.deployment_identity import (
    DEPLOYMENT_ID_FILENAME,
    DeploymentIdentityError,
    get_or_create_deployment_id,
    read_deployment_id,
)


def test_get_or_create_deployment_id_is_stable(tmp_path: Path) -> None:
    deployment_id = get_or_create_deployment_id(tmp_path)

    assert str(UUID(deployment_id)) == deployment_id
    assert get_or_create_deployment_id(tmp_path) == deployment_id
    assert read_deployment_id(tmp_path) == deployment_id
    assert (tmp_path / DEPLOYMENT_ID_FILENAME).read_text(encoding="ascii") == (
        f"{deployment_id}\n"
    )


def test_get_or_create_rejects_invalid_existing_identity(tmp_path: Path) -> None:
    (tmp_path / DEPLOYMENT_ID_FILENAME).write_text("not-a-uuid\n", encoding="ascii")

    with pytest.raises(DeploymentIdentityError, match="unreadable or invalid"):
        get_or_create_deployment_id(tmp_path)


def test_get_or_create_refuses_rotation_over_retained_snapshot(tmp_path: Path) -> None:
    """Identity loss with durable state requires explicit operator recovery."""
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "CURRENT").write_text(
        "20260714T010203-a1b2c3d4\n",
        encoding="utf-8",
    )

    with pytest.raises(DeploymentIdentityError, match="durable snapshots"):
        get_or_create_deployment_id(tmp_path)

    assert not (tmp_path / DEPLOYMENT_ID_FILENAME).exists()


def test_read_rejects_symlinked_identity(tmp_path: Path) -> None:
    target = tmp_path / "identity-target"
    target.write_text(f"{UUID(int=1)}\n", encoding="ascii")
    try:
        os.symlink(target, tmp_path / DEPLOYMENT_ID_FILENAME)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(DeploymentIdentityError, match="non-symlink"):
        read_deployment_id(tmp_path)
