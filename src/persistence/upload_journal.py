"""Crash-recoverable source transactions for corpus activation."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any

from src.persistence.hashing import compute_corpus_hash_entries
from src.utils.hashing import sha256_file

_JOURNAL_NAME = "transaction.json"
_PAYLOAD_DIR_NAME = "payload"
_PAYLOAD_NAME = "upload"
_SCHEMA_VERSION = 1


class UploadJournalRecoveryError(RuntimeError):
    """Raised when a durable upload transaction cannot be resolved safely."""


def _fsync_dir(path: Path) -> None:
    """Persist a transaction directory entry or raise."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    """Persist transaction-owned file bytes before pointer activation."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _journal_temporary(journal: Path) -> Path:
    """Return the sole temporary path used for an atomic journal write."""
    return journal.with_suffix(".tmp")


def _write_journal(path: Path, payload: dict[str, Any]) -> None:
    """Write one journal atomically and durably."""
    temporary = _journal_temporary(path)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _safe_name(value: object) -> str | None:
    """Return a direct path name or ``None`` for an unsafe value."""
    if not isinstance(value, str) or not value or value in {".", ".."}:
        return None
    return value if Path(value).name == value else None


def _safe_digest(value: object) -> str | None:
    """Return one canonical lowercase SHA-256 digest."""
    if not isinstance(value, str) or len(value) != 64:
        return None
    return (
        value if all(character in "0123456789abcdef" for character in value) else None
    )


def _collections(payload: dict[str, Any]) -> dict[str, str] | None:
    """Return the exact physical collection generation in a journal/manifest."""
    value = payload.get("collections")
    if not isinstance(value, dict):
        return None
    text = value.get("text")
    image = value.get("image")
    if not isinstance(text, str) or not text or not isinstance(image, str) or not image:
        return None
    return {"text": text, "image": image}


def _journal_collections(payload: dict[str, Any]) -> dict[str, str] | None:
    """Return collections only from the exact journal-owned mapping shape."""
    value = payload.get("collections")
    if not isinstance(value, dict) or set(value) != {"text", "image"}:
        return None
    return _collections(payload)


def _trusted_root(data_dir: Path, name: str, *, create: bool = False) -> Path:
    """Resolve one transaction root without accepting a symlink boundary."""
    candidate = data_dir / name
    if candidate.is_symlink():
        raise ValueError(f"{name} root must not be a symlink")
    existed = candidate.exists()
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
        if not existed:
            _fsync_dir(candidate.parent)
    if candidate.exists() and not candidate.is_dir():
        raise ValueError(f"{name} root must be a directory")
    return candidate.resolve()


def _validate_contained_path(path: Path, root: Path) -> None:
    """Reject escapes and symlinks in every component below a trusted root."""
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Upload transaction path escapes its owned root") from exc
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("Upload transaction path contains a symlink")
    if not path.resolve(strict=False).is_relative_to(root):
        raise ValueError("Upload transaction path resolves outside its owned root")


def _unlink_durable(path: Path) -> None:
    """Unlink one directory entry and persist its removal."""
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_dir(path.parent)


def _rmdir_if_empty(path: Path) -> bool:
    """Durably remove an empty directory, returning false only when nonempty."""
    try:
        path.rmdir()
    except FileNotFoundError:
        return True
    except OSError as exc:
        if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
            return False
        raise
    _fsync_dir(path.parent)
    return True


def _clean_transaction_dir(
    transaction_dir: Path,
    *,
    payload_dir: Path | None = None,
) -> None:
    """Durably retire a resolved journal, transaction directory, and empty root."""
    journal = transaction_dir / _JOURNAL_NAME
    allowed = {_JOURNAL_NAME, _journal_temporary(journal).name}
    if payload_dir is not None:
        allowed.add(payload_dir.name)
        if payload_dir.exists():
            if payload_dir.is_symlink() or not payload_dir.is_dir():
                raise ValueError("Upload transaction payload is not a safe directory")
            _validate_exact_entries(payload_dir, set())
    _validate_exact_entries(transaction_dir, allowed)
    _unlink_durable(journal)
    _unlink_durable(_journal_temporary(journal))
    if payload_dir is not None and not _rmdir_if_empty(payload_dir):
        raise OSError(errno.ENOTEMPTY, "Upload transaction payload is not empty")
    if not _rmdir_if_empty(transaction_dir):
        raise OSError(errno.ENOTEMPTY, "Upload transaction directory is not empty")
    _rmdir_if_empty(transaction_dir.parent)


def _retire_pre_mutation_debris(transaction_dir: Path) -> bool:
    """Retire only crash debris that proves no authoritative mutation occurred."""
    entries = list(transaction_dir.iterdir())
    if not entries:
        _clean_transaction_dir(transaction_dir)
        return True
    temporary = _journal_temporary(transaction_dir / _JOURNAL_NAME)
    if (
        len(entries) == 1
        and entries[0] == temporary
        and not temporary.is_symlink()
        and temporary.is_file()
    ):
        _clean_transaction_dir(transaction_dir)
        return True
    return False


def _validate_exact_entries(directory: Path, allowed: set[str]) -> None:
    """Reject every unexplained transaction-owned directory entry."""
    if {entry.name for entry in directory.iterdir()} - allowed:
        raise ValueError("transaction contains unexplained entries")


def cleanup_pending_uploads_after_restart(data_dir: Path) -> None:
    """Durably remove unjournaled upload submissions during startup recovery."""
    pending_root = _trusted_root(data_dir, ".pending-uploads")
    if not pending_root.exists():
        return
    try:
        transactions = list(pending_root.iterdir())
        for transaction in transactions:
            if transaction.is_symlink() or not transaction.is_dir():
                raise ValueError("Pending upload root contains an unsafe entry")
            _validate_contained_path(transaction, pending_root)
            descendants = list(transaction.rglob("*"))
            for path in descendants:
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise ValueError("Pending upload transaction contains unsafe data")
                _validate_contained_path(path, pending_root)
            for path in descendants:
                if path.is_file():
                    _unlink_durable(path)
            directories = [path for path in descendants if path.is_dir()]
            for directory in sorted(
                directories,
                key=lambda item: len(item.relative_to(transaction).parts),
                reverse=True,
            ):
                if not _rmdir_if_empty(directory):
                    raise OSError(
                        errno.ENOTEMPTY, "Pending upload directory is not empty"
                    )
            if not _rmdir_if_empty(transaction):
                raise OSError(
                    errno.ENOTEMPTY, "Pending upload transaction is not empty"
                )
        if not _rmdir_if_empty(pending_root):
            raise OSError(errno.ENOTEMPTY, "Pending upload root is not empty")
    except (OSError, ValueError) as exc:
        raise UploadJournalRecoveryError(
            "Pending upload restart cleanup failed closed"
        ) from exc


def quarantine_upload(
    *,
    data_dir: Path,
    source_path: Path,
    transaction_id: str,
    collections: dict[str, str],
) -> tuple[Path, Path]:
    """Journal and move one direct upload out of the authoritative corpus."""
    uploads_root = _trusted_root(data_dir, "uploads")
    source = source_path.resolve(strict=True)
    if (
        source_path.is_symlink()
        or source.parent != uploads_root
        or not source.is_file()
    ):
        raise ValueError("Deletion target must be a direct regular upload")
    source_digest = sha256_file(source)
    transaction_name = _safe_name(transaction_id)
    if transaction_name is None:
        raise ValueError("Quarantine transaction identity is invalid")
    generation = _collections({"collections": collections})
    if generation is None:
        raise ValueError("Quarantine transaction requires physical collections")

    quarantine_root = _trusted_root(data_dir, ".quarantine", create=True)
    transaction_dir = quarantine_root / transaction_name
    transaction_dir.mkdir(exist_ok=False)
    _fsync_dir(quarantine_root)
    payload_dir = transaction_dir / _PAYLOAD_DIR_NAME
    destination = payload_dir / _PAYLOAD_NAME
    journal = transaction_dir / _JOURNAL_NAME
    try:
        _write_journal(
            journal,
            {
                "schema_version": _SCHEMA_VERSION,
                "operation": "quarantine",
                "source_name": source.name,
                "sha256": source_digest,
                "collections": generation,
            },
        )
        payload_dir.mkdir()
        _fsync_dir(transaction_dir)
        os.replace(source, destination)
        _fsync_file(destination)
        _fsync_dir(uploads_root)
        _fsync_dir(payload_dir)
    except Exception:
        if source.exists() and not destination.exists():
            _clean_transaction_dir(transaction_dir, payload_dir=payload_dir)
        raise
    return source, destination


def _quarantine_transaction_dir(quarantine: Path) -> Path:
    """Return the transaction directory for the fixed quarantine payload path."""
    if quarantine.name != _PAYLOAD_NAME or quarantine.parent.name != _PAYLOAD_DIR_NAME:
        raise ValueError("Quarantined upload path is not canonical")
    return quarantine.parent.parent


def _trusted_quarantine_context(
    quarantine: Path,
) -> tuple[Path, Path, Path, str]:
    """Bind a public quarantine path back to its durable journal."""
    transaction_dir = _quarantine_transaction_dir(quarantine)
    data_dir = transaction_dir.parent.parent
    parsed_dir, _payload, source, canonical_quarantine, digest = (
        _read_quarantine_transaction(
            data_dir=data_dir,
            transaction_id=transaction_dir.name,
        )
    )
    if parsed_dir != transaction_dir or canonical_quarantine != quarantine:
        raise ValueError("Quarantined upload path does not match its journal")
    return parsed_dir, source, canonical_quarantine, digest


def restore_quarantined_upload(source: Path, quarantine: Path) -> None:
    """Restore one pre-commit source and retire its journal."""
    transaction_dir, canonical_source, quarantine, digest = _trusted_quarantine_context(
        quarantine
    )
    if source != canonical_source:
        raise ValueError("Restore destination does not match the quarantine journal")
    if not quarantine.exists():
        if _owned_file_exists(source, digest, root=source.parent):
            _clean_transaction_dir(transaction_dir, payload_dir=quarantine.parent)
        return
    if not _owned_file_exists(quarantine, digest, root=transaction_dir):
        raise ValueError("Quarantined upload is unavailable")
    if source.exists() or source.is_symlink():
        raise ValueError("Cannot restore quarantined upload over an existing path")
    os.replace(quarantine, source)
    _fsync_file(source)
    _fsync_dir(source.parent)
    _fsync_dir(quarantine.parent)
    _clean_transaction_dir(transaction_dir, payload_dir=quarantine.parent)


def discard_quarantined_upload(quarantine: Path) -> None:
    """Delete one source after its physical collection generation commits."""
    transaction_dir, _source, quarantine, digest = _trusted_quarantine_context(
        quarantine
    )
    _remove_if_owned(quarantine, digest, root=transaction_dir)
    _clean_transaction_dir(transaction_dir, payload_dir=quarantine.parent)


def _safe_pending_relative(value: object) -> Path | None:
    """Return a contained relative pending path from journal data."""
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or candidate == Path("."):
        return None
    return candidate


def _promotion_moves(
    payload: dict[str, Any],
) -> list[tuple[Path, str, str]] | None:
    """Validate and normalize promotion move records."""
    raw_moves = payload.get("moves")
    if not isinstance(raw_moves, list) or not raw_moves:
        return None
    moves: list[tuple[Path, str, str]] = []
    pending_paths: set[Path] = set()
    destinations: set[str] = set()
    for item in raw_moves:
        if not isinstance(item, dict) or set(item) != {
            "pending",
            "destination",
            "sha256",
        }:
            return None
        pending = _safe_pending_relative(item.get("pending"))
        destination = _safe_name(item.get("destination"))
        digest = _safe_digest(item.get("sha256"))
        if (
            pending is None
            or destination is None
            or digest is None
            or pending in pending_paths
            or destination in destinations
        ):
            return None
        pending_paths.add(pending)
        destinations.add(destination)
        moves.append((pending, destination, digest))
    return moves


def promote_pending_uploads(
    *,
    data_dir: Path,
    transaction_id: str,
    collections: dict[str, str],
    moves: list[tuple[Path, Path, str]],
) -> None:
    """Journal and promote new uploads while the snapshot writer lock is held."""
    transaction_name = _safe_name(transaction_id)
    if transaction_name is None or not moves:
        raise ValueError("Upload promotion transaction is invalid")
    generation = _collections({"collections": collections})
    if generation is None:
        raise ValueError("Upload promotion requires physical collections")

    pending_root = _trusted_root(data_dir, ".pending-uploads")
    uploads_root = _trusted_root(data_dir, "uploads", create=True)
    journal_moves: list[dict[str, str]] = []
    normalized: list[tuple[Path, Path, str]] = []
    seen_sources: set[Path] = set()
    seen_destinations: set[Path] = set()
    for pending, destination, digest in moves:
        source = pending.resolve(strict=True)
        target = destination.resolve()
        if (
            source in seen_sources
            or target in seen_destinations
            or pending.is_symlink()
            or not source.is_relative_to(pending_root)
            or target.parent != uploads_root
            or destination.is_symlink()
            or destination.exists()
            or _safe_digest(digest) is None
            or sha256_file(source) != digest
        ):
            raise ValueError("Upload promotion move is unsafe or inconsistent")
        seen_sources.add(source)
        seen_destinations.add(target)
        relative = source.relative_to(pending_root)
        journal_moves.append(
            {
                "pending": relative.as_posix(),
                "destination": target.name,
                "sha256": digest,
            }
        )
        normalized.append((source, target, digest))

    transaction_root = _trusted_root(data_dir, ".upload-transactions", create=True)
    transaction_dir = transaction_root / transaction_name
    transaction_dir.mkdir(exist_ok=False)
    _fsync_dir(transaction_root)
    try:
        _write_journal(
            transaction_dir / _JOURNAL_NAME,
            {
                "schema_version": _SCHEMA_VERSION,
                "operation": "promote",
                "collections": generation,
                "moves": journal_moves,
            },
        )
        for source, destination, _digest in normalized:
            os.replace(source, destination)
            _fsync_file(destination)
            _fsync_dir(source.parent)
            _fsync_dir(uploads_root)
    except Exception:
        journal = transaction_dir / _JOURNAL_NAME
        if journal.is_file() and not journal.is_symlink():
            rollback_upload_promotion(
                data_dir=data_dir,
                transaction_id=transaction_name,
            )
        else:
            _clean_transaction_dir(transaction_dir)
        raise


def _read_json_journal(journal: Path) -> dict[str, Any]:
    """Read one trusted JSON object journal."""
    if journal.is_symlink() or not journal.is_file():
        raise ValueError("journal is not a trusted regular file")
    try:
        payload: object = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("journal is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("journal has an invalid schema")
    return payload


def _read_promotion_transaction(
    *, data_dir: Path, transaction_id: str
) -> tuple[Path, dict[str, Any], list[tuple[Path, str, str]]]:
    """Read one trusted promotion journal."""
    transaction_name = _safe_name(transaction_id)
    if transaction_name is None:
        raise ValueError("transaction identity is invalid")
    transaction_root = _trusted_root(data_dir, ".upload-transactions")
    transaction_dir = transaction_root / transaction_name
    _validate_contained_path(transaction_dir, transaction_root)
    if transaction_dir.is_symlink() or not transaction_dir.is_dir():
        raise ValueError("transaction is not a safe directory")
    _validate_exact_entries(transaction_dir, {_JOURNAL_NAME})
    payload = _read_json_journal(transaction_dir / _JOURNAL_NAME)
    if (
        set(payload) != {"schema_version", "operation", "collections", "moves"}
        or payload.get("schema_version") != _SCHEMA_VERSION
        or payload.get("operation") != "promote"
        or _journal_collections(payload) is None
    ):
        raise ValueError("journal is invalid")
    moves = _promotion_moves(payload)
    if moves is None:
        raise ValueError("moves are invalid")
    return transaction_dir, payload, moves


def _read_quarantine_transaction(
    *, data_dir: Path, transaction_id: str
) -> tuple[Path, dict[str, Any], Path, Path, str]:
    """Read one trusted source-quarantine journal and its canonical paths."""
    transaction_name = _safe_name(transaction_id)
    if transaction_name is None:
        raise ValueError("transaction identity is invalid")
    quarantine_root = _trusted_root(data_dir, ".quarantine")
    transaction_dir = quarantine_root / transaction_name
    _validate_contained_path(transaction_dir, quarantine_root)
    if transaction_dir.is_symlink() or not transaction_dir.is_dir():
        raise ValueError("transaction is not a safe directory")
    _validate_exact_entries(transaction_dir, {_JOURNAL_NAME, _PAYLOAD_DIR_NAME})
    payload = _read_json_journal(transaction_dir / _JOURNAL_NAME)
    source_name = _safe_name(payload.get("source_name"))
    digest = _safe_digest(payload.get("sha256"))
    if (
        set(payload)
        != {"schema_version", "operation", "source_name", "sha256", "collections"}
        or payload.get("schema_version") != _SCHEMA_VERSION
        or payload.get("operation") != "quarantine"
        or source_name is None
        or digest is None
        or _journal_collections(payload) is None
    ):
        raise ValueError("journal has invalid transaction fields")

    uploads_root = _trusted_root(data_dir, "uploads", create=False)
    source = uploads_root / source_name
    _validate_contained_path(source, uploads_root)
    payload_dir = transaction_dir / _PAYLOAD_DIR_NAME
    quarantine = payload_dir / _PAYLOAD_NAME
    if payload_dir.exists():
        if payload_dir.is_symlink() or not payload_dir.is_dir():
            raise ValueError("payload is not a trusted directory")
        _validate_exact_entries(payload_dir, {_PAYLOAD_NAME})
        _validate_contained_path(quarantine, transaction_dir)
    return transaction_dir, payload, source, quarantine, digest


def _owned_file_exists(path: Path, digest: str, *, root: Path) -> bool:
    """Verify one optional transaction-owned file against path and digest."""
    _validate_contained_path(path, root)
    if path.is_symlink():
        raise ValueError("transaction-owned file is a symlink")
    if not path.exists():
        return False
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError("transaction-owned bytes do not match their journal")
    return True


def _remove_if_owned(path: Path, digest: str, *, root: Path) -> bool:
    """Delete a transaction-owned file only when its digest still matches."""
    if not _owned_file_exists(path, digest, root=root):
        return False
    _unlink_durable(path)
    return True


def rollback_upload_promotion(*, data_dir: Path, transaction_id: str) -> None:
    """Delete every uncommitted pending or promoted upload in one transaction."""
    transaction_dir, _payload, moves = _read_promotion_transaction(
        data_dir=data_dir,
        transaction_id=transaction_id,
    )
    pending_root = _trusted_root(data_dir, ".pending-uploads")
    uploads_root = _trusted_root(data_dir, "uploads", create=True)
    for pending_relative, destination_name, digest in moves:
        pending = pending_root / pending_relative
        destination = uploads_root / destination_name
        _remove_if_owned(pending, digest, root=pending_root)
        _remove_if_owned(destination, digest, root=uploads_root)
        _rmdir_if_empty(pending.parent)
    _clean_transaction_dir(transaction_dir)
    _rmdir_if_empty(pending_root)


def commit_upload_promotion(*, data_dir: Path, transaction_id: str) -> None:
    """Retire a committed promotion journal while keeping corpus destinations."""
    transaction_dir, _payload, moves = _read_promotion_transaction(
        data_dir=data_dir,
        transaction_id=transaction_id,
    )
    pending_root = _trusted_root(data_dir, ".pending-uploads")
    uploads_root = _trusted_root(data_dir, "uploads")
    for pending_relative, destination_name, digest in moves:
        pending = pending_root / pending_relative
        destination = uploads_root / destination_name
        if not _owned_file_exists(destination, digest, root=uploads_root):
            raise ValueError("Committed upload promotion destination is unavailable")
        _remove_if_owned(pending, digest, root=pending_root)
        _rmdir_if_empty(pending.parent)
    _clean_transaction_dir(transaction_dir)
    _rmdir_if_empty(pending_root)


def _transaction_directories(root: Path, *, label: str) -> list[Path]:
    """Return trusted direct transaction directories under one existing root."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root is not a safe directory")
    directories = sorted(root.iterdir())
    if any(path.is_symlink() or not path.is_dir() for path in directories):
        raise ValueError(f"{label} transaction is not a safe directory")
    return directories


def _recover_upload_promotions(
    *,
    data_dir: Path,
    active_collections: dict[str, str] | None,
) -> None:
    """Resolve durable pending-upload promotions against active collections."""
    transaction_root = data_dir / ".upload-transactions"
    if not transaction_root.exists():
        return
    try:
        directories = _transaction_directories(
            transaction_root,
            label="Upload promotion",
        )
        for transaction_dir in directories:
            if _retire_pre_mutation_debris(transaction_dir):
                continue
            _directory, payload, _moves = _read_promotion_transaction(
                data_dir=data_dir,
                transaction_id=transaction_dir.name,
            )
            if active_collections == _collections(payload):
                commit_upload_promotion(
                    data_dir=data_dir,
                    transaction_id=transaction_dir.name,
                )
            else:
                rollback_upload_promotion(
                    data_dir=data_dir,
                    transaction_id=transaction_dir.name,
                )
        _rmdir_if_empty(transaction_root)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise UploadJournalRecoveryError(
            "Upload promotion recovery could not resolve every transaction"
        ) from exc


def recover_upload_quarantines(
    *,
    data_dir: Path,
    active_collections: dict[str, str] | None,
) -> None:
    """Recover journaled source moves while the snapshot writer lock is held."""
    _recover_upload_promotions(
        data_dir=data_dir,
        active_collections=active_collections,
    )
    quarantine_root = data_dir / ".quarantine"
    if not quarantine_root.exists():
        return
    try:
        directories = _transaction_directories(
            quarantine_root,
            label="Upload quarantine",
        )
        uploads_root = _trusted_root(data_dir, "uploads", create=True)
        for transaction_dir in directories:
            if _retire_pre_mutation_debris(transaction_dir):
                continue
            directory, payload, source, quarantine, digest = (
                _read_quarantine_transaction(
                    data_dir=data_dir,
                    transaction_id=transaction_dir.name,
                )
            )
            source_exists = _owned_file_exists(source, digest, root=uploads_root)
            quarantine_exists = _owned_file_exists(
                quarantine,
                digest,
                root=directory,
            )
            if source_exists and quarantine_exists:
                raise ValueError("transaction has two owned source copies")
            committed = active_collections == _collections(payload)
            if committed:
                if quarantine_exists:
                    discard_quarantined_upload(quarantine)
                elif source_exists:
                    _remove_if_owned(source, digest, root=uploads_root)
                    _clean_transaction_dir(
                        directory,
                        payload_dir=quarantine.parent,
                    )
                else:
                    _clean_transaction_dir(
                        directory,
                        payload_dir=quarantine.parent,
                    )
            elif quarantine_exists:
                restore_quarantined_upload(source, quarantine)
            elif source_exists:
                _clean_transaction_dir(directory, payload_dir=quarantine.parent)
            else:
                raise ValueError("uncommitted deletion has no recoverable source bytes")
        _rmdir_if_empty(quarantine_root)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise UploadJournalRecoveryError(
            f"Upload quarantine recovery could not resolve every transaction: {exc}"
        ) from exc


def _strict_upload_entries(uploads_root: Path) -> dict[str, Path]:
    """Return the exact regular-file corpus without following any symlink."""
    entries: dict[str, Path] = {}
    if not uploads_root.exists():
        return entries
    for path in uploads_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Uploads corpus contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Uploads corpus contains an unsupported entry")
        logical = path.relative_to(uploads_root).as_posix()
        entries[logical] = path
    return entries


def _live_transaction_directories(data_dir: Path) -> list[tuple[str, Path]]:
    """Return the single live activation's strict transaction directories."""
    transactions: list[tuple[str, Path]] = []
    roots = [
        (data_dir / ".upload-transactions", "promotion"),
        (data_dir / ".quarantine", "quarantine"),
    ]
    for root, operation in roots:
        if root.exists():
            transactions.extend(
                (operation, directory)
                for directory in _transaction_directories(
                    root,
                    label=f"Upload {operation}",
                )
            )
    if not transactions or len({path.name for _, path in transactions}) != 1:
        raise ValueError("Journals do not identify one live activation")
    return transactions


def _bind_journal_generation(
    payload: dict[str, Any],
    *,
    active: dict[str, str],
    expected: dict[str, str] | None,
) -> dict[str, str]:
    """Bind one journal to one wholly different physical generation."""
    generation = _collections(payload)
    if (
        generation is None
        or generation["text"] == active["text"]
        or generation["image"] == active["image"]
        or (expected is not None and generation != expected)
    ):
        raise ValueError("Journal collection generation is unrelated")
    return generation


def _load_bound_journal_changes(
    data_dir: Path,
    active_collections: dict[str, str],
) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, Path, str]]]:
    """Strictly parse all journals belonging to one non-active generation."""
    generation: dict[str, str] | None = None
    promotions: list[tuple[Path, str, str]] = []
    quarantines: list[tuple[Path, Path, str]] = []
    for operation, directory in _live_transaction_directories(data_dir):
        if operation == "promotion":
            _transaction, payload, moves = _read_promotion_transaction(
                data_dir=data_dir,
                transaction_id=directory.name,
            )
            promotions.extend(moves)
        else:
            transaction, payload, source, quarantine, digest = (
                _read_quarantine_transaction(
                    data_dir=data_dir,
                    transaction_id=directory.name,
                )
            )
            if transaction != directory:
                raise ValueError("Quarantine transaction path changed")
            quarantines.append((source, quarantine, digest))
        generation = _bind_journal_generation(
            payload,
            active=active_collections,
            expected=generation,
        )
    return promotions, quarantines


def _remove_promoted_entries(
    entries: dict[str, Path],
    promotions: list[tuple[Path, str, str]],
    *,
    pending_root: Path,
    uploads_root: Path,
) -> None:
    """Remove only digest-bound promoted files from the pre-commit identity."""
    for pending_relative, destination_name, digest in promotions:
        pending = pending_root / pending_relative
        destination = uploads_root / destination_name
        pending_exists = _owned_file_exists(pending, digest, root=pending_root)
        destination_exists = _owned_file_exists(
            destination,
            digest,
            root=uploads_root,
        )
        if pending_exists == destination_exists:
            raise ValueError("Promotion has an unexplained byte state")
        if destination_exists:
            if entries.pop(destination_name, None) != destination:
                raise ValueError("Promoted destination is not in the live corpus")
        elif destination_name in entries:
            raise ValueError("Promotion destination has unrelated corpus bytes")


def _restore_quarantined_entries(
    entries: dict[str, Path],
    quarantines: list[tuple[Path, Path, str]],
    *,
    uploads_root: Path,
) -> None:
    """Substitute digest-bound quarantined bytes at their original logical path."""
    for source, quarantine, digest in quarantines:
        source_exists = _owned_file_exists(source, digest, root=uploads_root)
        transaction_dir = _quarantine_transaction_dir(quarantine)
        quarantine_exists = _owned_file_exists(
            quarantine,
            digest,
            root=transaction_dir,
        )
        if source_exists == quarantine_exists:
            raise ValueError("Quarantine has an unexplained byte state")
        logical = source.name
        if source_exists and entries.get(logical) != source:
            raise ValueError("Quarantine source is not in the live corpus")
        if not source_exists:
            if logical in entries:
                raise ValueError("Quarantine logical path is already occupied")
            entries[logical] = quarantine


def reconstruct_precommit_corpus_hash(
    data_dir: Path,
    active_manifest: dict[str, Any],
) -> str | None:
    """Reconstruct the active corpus only from strictly bound live journals."""
    try:
        active_collections = _collections(active_manifest)
        expected_hash = _safe_digest(active_manifest.get("corpus_hash"))
        if active_collections is None or expected_hash is None:
            return None
        promotions, quarantines = _load_bound_journal_changes(
            data_dir,
            active_collections,
        )
        uploads_root = _trusted_root(data_dir, "uploads")
        entries = _strict_upload_entries(uploads_root)
        pending_root = _trusted_root(data_dir, ".pending-uploads")
        _remove_promoted_entries(
            entries,
            promotions,
            pending_root=pending_root,
            uploads_root=uploads_root,
        )
        _restore_quarantined_entries(
            entries,
            quarantines,
            uploads_root=uploads_root,
        )
        reconstructed = compute_corpus_hash_entries(list(entries.items()))
        return reconstructed if reconstructed == expected_hash else None
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None


__all__ = [
    "UploadJournalRecoveryError",
    "cleanup_pending_uploads_after_restart",
    "commit_upload_promotion",
    "discard_quarantined_upload",
    "promote_pending_uploads",
    "quarantine_upload",
    "reconstruct_precommit_corpus_hash",
    "recover_upload_quarantines",
    "restore_quarantined_upload",
    "rollback_upload_promotion",
]
