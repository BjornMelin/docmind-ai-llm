"""Content-addressed artifact storage (local-first).

Stores binary artifacts (e.g., rendered PDF page images + thumbnails) under a
directory rooted at ``settings.data_dir`` by default.

Final-release constraints:
- Persist **references** (sha256 + suffix), not raw filesystem paths, in durable
  stores (Qdrant payloads, LangGraph SQLite state, etc.).
- Enforce a directory jail: resolved artifact paths must live under the root.
"""

from __future__ import annotations

import contextlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from loguru import logger

from src.utils.hashing import sha256_file


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Stable reference to an artifact stored in an ArtifactStore."""

    sha256: str
    suffix: str

    def filename(self) -> str:
        """Return the storage filename for this artifact."""
        return f"{self.sha256}{self.suffix}"


def _suffix_for_artifact(path: Path) -> str:
    # Preserve compound suffixes used by encrypted images: ".webp.enc".
    suffixes = path.suffixes
    if not suffixes:
        return ""
    if len(suffixes) >= 2 and suffixes[-1] == ".enc":
        return "".join(suffixes[-2:])
    return suffixes[-1]


class ArtifactStore:
    """Local directory-backed content-addressed artifact store."""

    def __init__(self, root: Path) -> None:
        """Create an artifact store rooted at ``root``."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: object) -> ArtifactStore:
        """Build an ArtifactStore from ``DocMindSettings``-like settings."""
        base = getattr(settings, "data_dir", Path("./data"))
        artifacts_cfg = getattr(settings, "artifacts", None)
        override = getattr(artifacts_cfg, "dir", None) if artifacts_cfg else None
        root = Path(override) if override is not None else (Path(base) / "artifacts")
        return cls(root=root)

    def resolve_path(self, ref: ArtifactRef) -> Path:
        """Resolve an artifact ref to a filesystem path (directory-jail enforced)."""
        if len(ref.sha256) != 64 or any(
            c not in "0123456789abcdef" for c in ref.sha256
        ):
            raise ValueError("Invalid artifact sha256")
        if ref.suffix:
            if not ref.suffix.startswith("."):
                raise ValueError("Invalid artifact suffix")
            if any(sep in ref.suffix for sep in ("/", "\\")):
                raise ValueError("Invalid artifact suffix")
            if len(ref.suffix) > 16:
                raise ValueError("Invalid artifact suffix")
            allowed = set("abcdefghijklmnopqrstuvwxyz0123456789.-_")
            if any(c.lower() not in allowed for c in ref.suffix):
                raise ValueError("Invalid artifact suffix")

        candidate = (self.root / ref.filename()).resolve()
        root = self.root.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Artifact path escaped root")
        return candidate

    def exists(self, ref: ArtifactRef) -> bool:
        """Return True if the referenced artifact exists."""
        return self.resolve_path(ref).exists()

    def put_file(self, src: Path) -> ArtifactRef:
        """Store an existing file; returns its content-addressed reference."""
        src = Path(src)
        if not src.exists():
            raise FileNotFoundError(str(src))
        sha = sha256_file(src)
        suffix = _suffix_for_artifact(src)
        ref = ArtifactRef(sha256=sha, suffix=suffix)
        dest = self.resolve_path(ref)
        if dest.exists():
            return ref
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".{ref.sha256}.{uuid4().hex}.tmp"
        shutil.copyfile(src, tmp)
        try:
            tmp.rename(dest)
        except FileExistsError:
            with contextlib.suppress(Exception):
                tmp.unlink()
        return ref

    def delete(self, ref: ArtifactRef) -> None:
        """Delete an artifact by reference (best-effort)."""
        path = self.resolve_path(ref)
        if not path.exists():
            return
        path.unlink()

    def prune(self, *, max_total_bytes: int, min_age_seconds: int = 0) -> int:
        """Best-effort GC: delete oldest artifacts until under budget.

        Returns number of deleted files. Concurrent modifications may make the
        byte totals approximate; failures are handled best-effort.
        """
        root = self.root
        files: list[tuple[float, int, Path]] = []
        total = 0
        for p in root.glob("*"):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            total += int(st.st_size)
            files.append((float(st.st_mtime), int(st.st_size), p))

        if total <= max_total_bytes:
            return 0

        cutoff = time.time() - max(0, int(min_age_seconds))
        # Oldest-first eviction.
        files.sort(key=lambda t: (t[0], str(t[2])))
        deleted = 0
        for mtime, size, path in files:
            if total <= max_total_bytes:
                break
            if mtime > cutoff:
                continue
            try:
                path.unlink()
                total -= int(size)
                deleted += 1
            except OSError:
                continue
        if deleted:
            logger.info("ArtifactStore GC deleted {} file(s)", deleted)
        return deleted


__all__ = ["ArtifactRef", "ArtifactStore"]
