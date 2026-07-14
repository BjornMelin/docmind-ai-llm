"""Crash-recovery tests for upload quarantine journals."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

from src.persistence import upload_journal
from src.persistence.hashing import compute_corpus_hash
from src.persistence.snapshot import recover_snapshot_transactions
from src.persistence.upload_journal import (
    UploadJournalRecoveryError,
    promote_pending_uploads,
    quarantine_upload,
    reconstruct_precommit_corpus_hash,
    recover_upload_quarantines,
)
from src.utils.hashing import sha256_file

pytestmark = pytest.mark.unit

_OLD_COLLECTIONS = {"text": "text__old", "image": "image__old"}
_NEW_COLLECTIONS = {"text": "text__new", "image": "image__new"}


def _upload(data_dir: Path, name: str = "document.txt") -> Path:
    path = data_dir / "uploads" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("authoritative corpus bytes", encoding="utf-8")
    return path


def _journal_payload(*, source_name: str = "document.txt") -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "quarantine",
        "source_name": source_name,
        "sha256": sha256(b"authoritative corpus bytes").hexdigest(),
        "collections": dict(_NEW_COLLECTIONS),
    }


def _pending_upload(
    data_dir: Path,
    name: str = "document.txt",
    content: str = "pending corpus bytes",
) -> tuple[Path, Path, str]:
    pending = data_dir / ".pending-uploads" / "submission" / name
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text(content, encoding="utf-8")
    destination = data_dir / "uploads" / name
    return pending, destination, sha256_file(pending)


def test_recovery_cleans_journal_when_crash_precedes_source_move(
    tmp_path: Path,
) -> None:
    """A journal-only transaction leaves the still-authoritative source intact."""
    source = _upload(tmp_path)
    transaction_dir = tmp_path / ".quarantine" / "build"
    transaction_dir.mkdir(parents=True)
    (transaction_dir / "transaction.json").write_text(
        json.dumps(_journal_payload()),
        encoding="utf-8",
    )

    recover_upload_quarantines(
        data_dir=tmp_path,
        active_collections=_OLD_COLLECTIONS,
    )

    assert source.read_text(encoding="utf-8") == "authoritative corpus bytes"
    assert not transaction_dir.exists()


def test_recovery_restores_quarantine_when_old_generation_is_current(
    tmp_path: Path,
) -> None:
    """A pre-commit source move is rolled back after a process crash."""
    source = _upload(tmp_path)
    _, quarantined = quarantine_upload(
        data_dir=tmp_path,
        source_path=source,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
    )
    assert not source.exists()
    assert quarantined.is_file()

    recover_upload_quarantines(
        data_dir=tmp_path,
        active_collections=_OLD_COLLECTIONS,
    )

    assert source.read_text(encoding="utf-8") == "authoritative corpus bytes"
    assert not quarantined.parent.exists()


def test_startup_recovery_resolves_upload_journal_under_writer_lock(
    tmp_path: Path,
) -> None:
    """The public startup boundary restores a pre-commit source transaction."""
    source = _upload(tmp_path)
    _, quarantined = quarantine_upload(
        data_dir=tmp_path,
        source_path=source,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
    )

    recover_snapshot_transactions(tmp_path / "storage")

    assert source.read_text(encoding="utf-8") == "authoritative corpus bytes"
    assert not quarantined.parent.exists()
    assert (tmp_path / "storage" / ".lock").is_file()


def test_startup_recovery_removes_unjournaled_pending_uploads(tmp_path: Path) -> None:
    """A restart retires private upload bytes saved before journal creation."""
    pending, _destination, _digest = _pending_upload(tmp_path)

    recover_snapshot_transactions(tmp_path / "storage")

    assert not pending.exists()
    assert not (tmp_path / ".pending-uploads").exists()


def test_begin_snapshot_does_not_remove_live_pending_uploads(tmp_path: Path) -> None:
    """Per-job recovery leaves another submission's pre-journal bytes intact."""
    from src.persistence.snapshot import SnapshotManager

    pending, _destination, _digest = _pending_upload(tmp_path)
    manager = SnapshotManager(tmp_path / "storage")
    workspace = manager.begin_snapshot()
    try:
        assert pending.is_file()
    finally:
        manager.cleanup_tmp(workspace)


def test_recovery_discards_quarantine_when_matching_generation_committed(
    tmp_path: Path,
) -> None:
    """A committed source deletion is completed after a process crash."""
    source = _upload(tmp_path)
    _, quarantined = quarantine_upload(
        data_dir=tmp_path,
        source_path=source,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
    )

    recover_upload_quarantines(
        data_dir=tmp_path,
        active_collections=_NEW_COLLECTIONS,
    )

    assert not source.exists()
    assert not quarantined.parent.exists()


def test_recovery_blocks_on_malformed_journal_without_touching_sources(
    tmp_path: Path,
) -> None:
    """Unreadable transaction state is retained for operator recovery."""
    source = _upload(tmp_path)
    transaction_dir = tmp_path / ".quarantine" / "build"
    transaction_dir.mkdir(parents=True)
    payload_dir = transaction_dir / "payload"
    payload_dir.mkdir()
    quarantined = payload_dir / "upload"
    source.replace(quarantined)
    (transaction_dir / "transaction.json").write_text("{", encoding="utf-8")

    with pytest.raises(UploadJournalRecoveryError, match="unreadable"):
        recover_upload_quarantines(
            data_dir=tmp_path,
            active_collections=_NEW_COLLECTIONS,
        )

    assert not source.exists()
    assert quarantined.read_text(encoding="utf-8") == "authoritative corpus bytes"
    assert transaction_dir.is_dir()


def test_recovery_rejects_symlinked_journal_without_touching_sources(
    tmp_path: Path,
) -> None:
    """Transaction authority cannot be redirected outside the quarantine root."""
    source = _upload(tmp_path)
    transaction_dir = tmp_path / ".quarantine" / "build"
    transaction_dir.mkdir(parents=True)
    payload_dir = transaction_dir / "payload"
    payload_dir.mkdir()
    quarantined = payload_dir / "upload"
    source.replace(quarantined)
    external = tmp_path / "attacker-controlled.json"
    external.write_text(json.dumps(_journal_payload()), encoding="utf-8")
    (transaction_dir / "transaction.json").symlink_to(external)

    with pytest.raises(UploadJournalRecoveryError, match="trusted regular file"):
        recover_upload_quarantines(
            data_dir=tmp_path,
            active_collections=_NEW_COLLECTIONS,
        )

    assert not source.exists()
    assert quarantined.read_text(encoding="utf-8") == "authoritative corpus bytes"
    assert transaction_dir.is_dir()


def test_recovery_never_overwrites_an_existing_upload(tmp_path: Path) -> None:
    """Ambiguous duplicate paths remain quarantined instead of losing bytes."""
    source = _upload(tmp_path)
    _, quarantined = quarantine_upload(
        data_dir=tmp_path,
        source_path=source,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
    )
    source.write_text("newer bytes", encoding="utf-8")

    with pytest.raises(UploadJournalRecoveryError, match="resolve"):
        recover_upload_quarantines(
            data_dir=tmp_path,
            active_collections=_OLD_COLLECTIONS,
        )

    assert source.read_text(encoding="utf-8") == "newer bytes"
    assert quarantined.read_text(encoding="utf-8") == "authoritative corpus bytes"
    assert (quarantined.parent.parent / "transaction.json").is_file()


def test_reserved_journal_filename_is_quarantined_without_collision(
    tmp_path: Path,
) -> None:
    """An upload named like the journal cannot overwrite transaction authority."""
    source = _upload(tmp_path, "transaction.json")

    _, quarantined = quarantine_upload(
        data_dir=tmp_path,
        source_path=source,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
    )

    transaction_dir = quarantined.parent.parent
    journal = transaction_dir / "transaction.json"
    payload = json.loads(journal.read_text(encoding="utf-8"))
    assert quarantined == transaction_dir / "payload" / "upload"
    assert payload["source_name"] == "transaction.json"
    assert quarantined.read_text(encoding="utf-8") == "authoritative corpus bytes"

    recover_upload_quarantines(
        data_dir=tmp_path,
        active_collections=_OLD_COLLECTIONS,
    )

    assert source.read_text(encoding="utf-8") == "authoritative corpus bytes"
    assert not transaction_dir.exists()


@pytest.mark.parametrize("root_name", [".upload-transactions", ".quarantine"])
@pytest.mark.parametrize("debris", ["empty", "temporary"])
def test_recovery_durably_retires_pre_mutation_debris(
    tmp_path: Path,
    root_name: str,
    debris: str,
) -> None:
    """Only empty and sole temporary-journal crash states are auto-retired."""
    transaction_dir = tmp_path / root_name / "build"
    transaction_dir.mkdir(parents=True)
    if debris == "temporary":
        (transaction_dir / "transaction.tmp").write_text("{", encoding="utf-8")

    recover_upload_quarantines(
        data_dir=tmp_path,
        active_collections=_OLD_COLLECTIONS,
    )

    assert not (tmp_path / root_name).exists()


@pytest.mark.parametrize("root_name", [".upload-transactions", ".quarantine"])
def test_recovery_fails_closed_on_unexplained_transaction_debris(
    tmp_path: Path,
    root_name: str,
) -> None:
    """Any non-journal entry preserves the transaction for operator recovery."""
    transaction_dir = tmp_path / root_name / "build"
    transaction_dir.mkdir(parents=True)
    unexplained = transaction_dir / "unknown"
    unexplained.write_text("bytes", encoding="utf-8")

    with pytest.raises(UploadJournalRecoveryError, match="resolve"):
        recover_upload_quarantines(
            data_dir=tmp_path,
            active_collections=_OLD_COLLECTIONS,
        )

    assert unexplained.read_text(encoding="utf-8") == "bytes"


def test_committed_missing_quarantine_retires_journal_idempotently(
    tmp_path: Path,
) -> None:
    """A crash after committed deletion but before retirement is recoverable."""
    source = _upload(tmp_path)
    _, quarantined = quarantine_upload(
        data_dir=tmp_path,
        source_path=source,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
    )
    quarantined.unlink()

    recover_upload_quarantines(
        data_dir=tmp_path,
        active_collections=_NEW_COLLECTIONS,
    )

    assert not source.exists()
    assert not (tmp_path / ".quarantine").exists()
    recover_upload_quarantines(
        data_dir=tmp_path,
        active_collections=_NEW_COLLECTIONS,
    )


def test_uncommitted_missing_quarantine_fails_closed(tmp_path: Path) -> None:
    """Missing deletion bytes cannot be retired against the old CURRENT."""
    source = _upload(tmp_path)
    _, quarantined = quarantine_upload(
        data_dir=tmp_path,
        source_path=source,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
    )
    quarantined.unlink()

    with pytest.raises(UploadJournalRecoveryError, match="no recoverable source"):
        recover_upload_quarantines(
            data_dir=tmp_path,
            active_collections=_OLD_COLLECTIONS,
        )

    assert (quarantined.parent.parent / "transaction.json").is_file()


def test_transaction_retirement_fsyncs_every_removed_directory_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Journal, transaction, and root removals all persist their parent entry."""
    source = _upload(tmp_path)
    _, quarantined = quarantine_upload(
        data_dir=tmp_path,
        source_path=source,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
    )
    transaction_dir = quarantined.parent.parent
    observed: list[Path] = []
    monkeypatch.setattr(upload_journal, "_fsync_dir", observed.append)

    recover_upload_quarantines(
        data_dir=tmp_path,
        active_collections=_NEW_COLLECTIONS,
    )

    assert transaction_dir in observed
    assert transaction_dir.parent in observed
    assert tmp_path in observed


def test_precommit_hash_reconstructs_only_one_bound_activation(tmp_path: Path) -> None:
    """Bound promotion and deletion journals reproduce the old corpus identity."""
    retained = _upload(tmp_path, "retained.txt")
    removed = _upload(tmp_path, "removed.txt")
    uploads_root = tmp_path / "uploads"
    old_hash = compute_corpus_hash(
        [retained, removed],
        base_dir=uploads_root,
    )
    pending, destination, digest = _pending_upload(
        tmp_path,
        "added.txt",
        "new corpus bytes",
    )
    promote_pending_uploads(
        data_dir=tmp_path,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
        moves=[(pending, destination, digest)],
    )
    quarantine_upload(
        data_dir=tmp_path,
        source_path=removed,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
    )
    active_manifest = {
        "collections": dict(_OLD_COLLECTIONS),
        "corpus_hash": old_hash,
    }

    assert reconstruct_precommit_corpus_hash(tmp_path, active_manifest) == old_hash
    assert reconstruct_precommit_corpus_hash(tmp_path, {}) is None
    assert (
        reconstruct_precommit_corpus_hash(
            tmp_path,
            {
                "collections": dict(_NEW_COLLECTIONS),
                "corpus_hash": old_hash,
            },
        )
        is None
    )


def test_promotion_journal_is_durable_before_source_move(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every authoritative move observes its already-persisted transaction."""
    pending, destination, digest = _pending_upload(tmp_path)
    journal = tmp_path / ".upload-transactions" / "build" / "transaction.json"
    real_replace = os.replace
    move_observed = False

    def _replace(source: Path, target: Path) -> None:
        nonlocal move_observed
        if Path(source) == pending:
            payload = json.loads(journal.read_text(encoding="utf-8"))
            assert payload["operation"] == "promote"
            assert payload["collections"] == _NEW_COLLECTIONS
            move_observed = True
        real_replace(source, target)

    monkeypatch.setattr(upload_journal.os, "replace", _replace)

    promote_pending_uploads(
        data_dir=tmp_path,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
        moves=[(pending, destination, digest)],
    )

    assert move_observed
    assert destination.read_text(encoding="utf-8") == "pending corpus bytes"


def test_old_current_removes_promoted_and_pending_copies(tmp_path: Path) -> None:
    """An uncommitted generation retains no transaction-owned corpus bytes."""
    pending, destination, digest = _pending_upload(tmp_path)
    promote_pending_uploads(
        data_dir=tmp_path,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
        moves=[(pending, destination, digest)],
    )
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text("pending corpus bytes", encoding="utf-8")

    recover_upload_quarantines(
        data_dir=tmp_path,
        active_collections=_OLD_COLLECTIONS,
    )

    assert not pending.exists()
    assert not destination.exists()
    assert not (tmp_path / ".upload-transactions").exists()


def test_matching_current_keeps_destination_and_retires_promotion_journal(
    tmp_path: Path,
) -> None:
    """Committed corpus bytes survive post-activation journal recovery."""
    pending, destination, digest = _pending_upload(tmp_path)
    promote_pending_uploads(
        data_dir=tmp_path,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
        moves=[(pending, destination, digest)],
    )

    recover_upload_quarantines(
        data_dir=tmp_path,
        active_collections=_NEW_COLLECTIONS,
    )

    assert not pending.exists()
    assert destination.read_text(encoding="utf-8") == "pending corpus bytes"
    assert not (tmp_path / ".upload-transactions").exists()


def test_old_current_never_deletes_changed_destination(tmp_path: Path) -> None:
    """Recovery leaves changed bytes and their journal for manual resolution."""
    pending, destination, digest = _pending_upload(tmp_path)
    promote_pending_uploads(
        data_dir=tmp_path,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
        moves=[(pending, destination, digest)],
    )
    destination.write_text("new owner bytes", encoding="utf-8")

    with pytest.raises(UploadJournalRecoveryError, match="resolve"):
        recover_upload_quarantines(
            data_dir=tmp_path,
            active_collections=_OLD_COLLECTIONS,
        )

    assert destination.read_text(encoding="utf-8") == "new owner bytes"
    assert (tmp_path / ".upload-transactions" / "build" / "transaction.json").is_file()


def test_rollback_rejects_symlinked_pending_parent_without_external_unlink(
    tmp_path: Path,
) -> None:
    """Recovery never follows a replaced pending parent outside its owned root."""
    pending, destination, digest = _pending_upload(tmp_path)
    promote_pending_uploads(
        data_dir=tmp_path,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
        moves=[(pending, destination, digest)],
    )
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external = external_dir / pending.name
    external.write_text("pending corpus bytes", encoding="utf-8")
    submission = tmp_path / ".pending-uploads" / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    submission.rmdir()
    submission.symlink_to(external_dir, target_is_directory=True)

    with pytest.raises(UploadJournalRecoveryError, match="resolve"):
        recover_upload_quarantines(
            data_dir=tmp_path,
            active_collections=_OLD_COLLECTIONS,
        )

    assert external.read_text(encoding="utf-8") == "pending corpus bytes"


def test_old_current_recovers_partial_multi_move_crash(tmp_path: Path) -> None:
    """Rollback handles a transaction split across pending and uploads roots."""
    first = _pending_upload(tmp_path, "first.txt", "first bytes")
    second = _pending_upload(tmp_path, "second.txt", "second bytes")
    promote_pending_uploads(
        data_dir=tmp_path,
        transaction_id="build",
        collections=_NEW_COLLECTIONS,
        moves=[first, second],
    )
    second_pending, second_destination, _digest = second
    second_destination.replace(second_pending)

    recover_upload_quarantines(
        data_dir=tmp_path,
        active_collections=_OLD_COLLECTIONS,
    )

    assert not first[0].exists()
    assert not first[1].exists()
    assert not second_pending.exists()
    assert not second_destination.exists()
    assert not (tmp_path / ".upload-transactions").exists()


def test_promotion_rejects_duplicate_destinations_before_any_move(
    tmp_path: Path,
) -> None:
    """One transaction cannot overwrite an earlier move at the same path."""
    first = _pending_upload(tmp_path, "first.txt", "first bytes")
    second_pending, _second_destination, second_digest = _pending_upload(
        tmp_path,
        "second.txt",
        "second bytes",
    )
    shared_destination = first[1]

    with pytest.raises(ValueError, match=r"unsafe|duplicate"):
        promote_pending_uploads(
            data_dir=tmp_path,
            transaction_id="build",
            collections=_NEW_COLLECTIONS,
            moves=[
                first,
                (second_pending, shared_destination, second_digest),
            ],
        )

    assert first[0].read_text(encoding="utf-8") == "first bytes"
    assert second_pending.read_text(encoding="utf-8") == "second bytes"
    assert not shared_destination.exists()
