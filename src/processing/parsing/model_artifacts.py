"""Immutable parser model manifest and integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import file_digest
from pathlib import Path
from shutil import copy2, rmtree
from time import monotonic


@dataclass(frozen=True)
class ModelArtifact:
    """One immutable model artifact."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelBundle:
    """A versioned collection of production model artifacts."""

    name: str
    root: str
    repo_id: str
    revision: str
    artifacts: tuple[ModelArtifact, ...]


class ModelIntegrityError(RuntimeError):
    """Raised when a parser model bundle differs from its manifest."""


DOCLING_LAYOUT_BUNDLE = ModelBundle(
    name="Docling layout Heron",
    root="docling-project--docling-layout-heron",
    repo_id="docling-project/docling-layout-heron",
    revision="8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
    artifacts=(
        ModelArtifact(
            path="config.json",
            size=3268,
            sha256="fdea30805ce2f5666b147fca941dcdd27ad468e27d6ed21902207d3da056a97d",
        ),
        ModelArtifact(
            path="model.safetensors",
            size=171658996,
            sha256="00333a43451945aaf89db8ca9c0a17e75d1537c17db60fdb91aa95f4c7929e0c",
        ),
        ModelArtifact(
            path="preprocessor_config.json",
            size=444,
            sha256="cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
        ),
    ),
)


def model_files(cache_dir: Path, bundle: ModelBundle) -> dict[str, Path]:
    """Return manifest paths for a model bundle."""
    root = Path(cache_dir) / bundle.root
    return {artifact.path: root / artifact.path for artifact in bundle.artifacts}


def model_directory_issues(root: Path, bundle: ModelBundle) -> dict[str, str]:
    """Return missing, unexpected, or mismatched files for a bundle directory."""
    return _uncached_model_directory_issues(root, bundle)


def cached_model_directory_issues(
    root: Path,
    bundle: ModelBundle,
    *,
    ttl_seconds: float = 30.0,
) -> dict[str, str]:
    """Return UI health issues with a stat-aware short-lived digest cache."""
    if ttl_seconds <= 0:
        return model_directory_issues(root, bundle)
    signature = _directory_signature(root)
    ttl_bucket = int(monotonic() / ttl_seconds)
    return dict(
        _cached_model_directory_issues(
            str(root),
            bundle,
            signature,
            ttl_bucket,
        )
    )


def _directory_signature(root: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    """Return metadata that invalidates cached integrity results on file changes."""
    if not root.exists() and not root.is_symlink():
        return ((".", 0, 0, 0, 0),)
    paths = [root]
    if root.is_dir():
        paths.extend(root.rglob("*"))
    signature: list[tuple[str, int, int, int, int]] = []
    for path in paths:
        try:
            stat = path.lstat()
        except OSError:
            signature.append((str(path), 0, 0, 0, 0))
            continue
        relative = "." if path == root else path.relative_to(root).as_posix()
        signature.append(
            (
                relative,
                int(stat.st_ino),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
            )
        )
    return tuple(sorted(signature))


@lru_cache(maxsize=32)
def _cached_model_directory_issues(
    root_value: str,
    bundle: ModelBundle,
    signature: tuple[tuple[str, int, int, int, int], ...],
    ttl_bucket: int,
) -> tuple[tuple[str, str], ...]:
    """Hash one unchanged model directory at most once per process."""
    del signature, ttl_bucket
    return tuple(
        sorted(_uncached_model_directory_issues(Path(root_value), bundle).items())
    )


def _uncached_model_directory_issues(
    root: Path,
    bundle: ModelBundle,
) -> dict[str, str]:
    """Perform a full manifest and digest verification."""
    expected = {artifact.path: artifact for artifact in bundle.artifacts}
    issues: dict[str, str] = {}
    if root.is_symlink():
        issues["."] = "model root must be a regular directory"
    for relative_path, artifact in expected.items():
        path = root / relative_path
        if _path_has_symlink(root, Path(relative_path)):
            issues[relative_path] = "symlinked path"
            continue
        if not path.is_file():
            issues[relative_path] = "missing regular file"
            continue
        if path.stat().st_size != artifact.size:
            issues[relative_path] = "size mismatch"
            continue
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        if digest != artifact.sha256:
            issues[relative_path] = "SHA-256 mismatch"

    if root.is_dir():
        for path in root.rglob("*"):
            if not (path.is_file() or path.is_symlink()):
                continue
            relative_path = path.relative_to(root).as_posix()
            if relative_path not in expected:
                issues[relative_path] = "unexpected file"
    return issues


def _path_has_symlink(root: Path, relative_path: Path) -> bool:
    """Return whether any component below ``root`` is a symbolic link."""
    current = root
    for part in relative_path.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def verify_model_directory(root: Path, bundle: ModelBundle) -> Path:
    """Verify one exact model directory and return it."""
    issues = model_directory_issues(root, bundle)
    if issues:
        detail = "; ".join(f"{path}: {reason}" for path, reason in issues.items())
        raise ModelIntegrityError(f"{bundle.name} integrity check failed: {detail}")
    return root


def verify_model_bundle(cache_dir: Path, bundle: ModelBundle) -> Path:
    """Verify one model bundle inside the application cache."""
    return verify_model_directory(Path(cache_dir) / bundle.root, bundle)


def install_downloaded_model_bundle(
    download_dir: Path,
    cache_dir: Path,
    bundle: ModelBundle,
) -> Path:
    """Install only verified manifest files from a dependency-native download."""
    prepared = Path(download_dir).parent / "verified"
    for artifact in bundle.artifacts:
        source = Path(download_dir) / artifact.path
        if source.is_symlink() or not source.is_file():
            raise ModelIntegrityError(
                f"{bundle.name} download missing regular file: {artifact.path}"
            )
        destination = prepared / artifact.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination)
    verify_model_directory(prepared, bundle)

    target = Path(cache_dir) / bundle.root
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        rmtree(target)
    prepared.replace(target)
    return verify_model_directory(target, bundle)


__all__ = [
    "DOCLING_LAYOUT_BUNDLE",
    "ModelArtifact",
    "ModelBundle",
    "ModelIntegrityError",
    "cached_model_directory_issues",
    "install_downloaded_model_bundle",
    "model_directory_issues",
    "model_files",
    "verify_model_bundle",
    "verify_model_directory",
]
