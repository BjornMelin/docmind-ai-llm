#!/usr/bin/env python3
"""Manual local backups with rotation (ADR-033, SPEC-037).

This script creates timestamped backup directories under `data/backups/` by
default and prunes older backups beyond a retention window.

Usage:
    uv run python scripts/backup.py create --help
    uv run python scripts/backup.py prune --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from src.config.settings import bootstrap_settings, settings
from src.persistence.backup_service import create_backup, prune_backups


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DocMind local backup utility")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a new backup")
    create.add_argument(
        "--dest",
        type=str,
        default=None,
        help=(
            "Backup root directory. If omitted, defaults to "
            "`settings.data_dir/backups`."
        ),
    )
    create.add_argument("--include-uploads", action="store_true")
    create.add_argument("--include-analytics", action="store_true")
    create.add_argument("--include-logs", action="store_true")
    create.add_argument(
        "--keep-last",
        type=int,
        default=None,
        help="How many backups to retain (defaults to settings.backup_keep_last).",
    )
    create.add_argument(
        "--no-qdrant-snapshot",
        action="store_true",
        help="Disable best-effort Qdrant collection snapshots.",
    )
    create.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON summary to stdout.",
    )

    prune = sub.add_parser("prune", help="Prune old backups beyond keep-last")
    prune.add_argument(
        "--root",
        type=str,
        default=None,
        help="Backup root directory (defaults to settings.data_dir/backups).",
    )
    prune.add_argument(
        "--keep-last",
        type=int,
        default=None,
        help="How many backups to retain (defaults to settings.backup_keep_last).",
    )

    return parser


def _cmd_create(args: argparse.Namespace) -> int:
    bootstrap_settings()

    dest_root = Path(args.dest).expanduser() if args.dest else None
    result = create_backup(
        dest_root=dest_root,
        include_uploads=bool(args.include_uploads),
        include_analytics=bool(args.include_analytics),
        include_logs=bool(args.include_logs),
        keep_last=args.keep_last,
        qdrant_snapshot=not bool(args.no_qdrant_snapshot),
        cfg=settings,
    )

    payload = {
        "backup_dir": str(result.backup_dir),
        "included": result.included,
        "bytes_written": result.bytes_written,
        "duration_ms": round(result.duration_ms, 2),
        "warnings": result.warnings,
        "qdrant_snapshots": [s.__dict__ for s in result.qdrant_snapshots],
    }

    if args.json:
        sys.stdout.write(f"{json.dumps(payload, indent=2, sort_keys=True)}\n")
        return 0

    logger.info("Backup created: {}", result.backup_dir)
    logger.info("Included: {}", ", ".join(result.included) if result.included else "-")
    logger.info("Bytes written: {}", result.bytes_written)
    if result.warnings:
        logger.warning("Warnings: {}", "; ".join(result.warnings))
    return 0


def _cmd_prune(args: argparse.Namespace) -> int:
    bootstrap_settings()

    root = (
        Path(args.root).expanduser() if args.root else (settings.data_dir / "backups")
    )
    keep_last = int(
        args.keep_last if args.keep_last is not None else settings.backup_keep_last
    )
    deleted = prune_backups(root, keep_last=keep_last)
    logger.info("Pruned {} backup(s) under {}", len(deleted), root)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for backup operations.

    Args:
        argv: Optional argv list (excluding program name).

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "create":
        return _cmd_create(args)
    if args.command == "prune":
        return _cmd_prune(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
