"""Fetch the PyTorch wheel during Docker builds with retry and checksum safety."""

import hashlib
import json
import os
import platform
import socket
import sys
import time
import urllib.request
from html.parser import HTMLParser
from http.client import HTTPResponse
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse


class _TorchWheelIndexParser(HTMLParser):
    def __init__(self, needle_prefix: str) -> None:
        """Create an index parser that captures the first matching wheel link.

        Args:
            needle_prefix: Substring that must appear in an anchor tag's href for it
                to be considered a match.
        """
        super().__init__()
        self._needle_prefix = needle_prefix
        self.result: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle an HTML start tag and record the first matching anchor href.

        Args:
            tag: The HTML tag name.
            attrs: Tag attributes as (name, value) pairs.
        """
        if self.result is not None or tag.lower() != "a":
            return
        for attr, value in attrs:
            if attr.lower() == "href" and value and self._needle_prefix in value:
                self.result = value
                return


def _resolve_torch_wheel_url(
    version: str, py_tag: str, plat: str
) -> tuple[str, str, str | None]:
    """Resolve a wheel URL/filename/checksum for the requested torch build.

    Resolution order:
      1) Explicit env override via TORCH_WHEEL_URL (+ optional TORCH_WHEEL_SHA256).
      2) PyTorch CPU wheel index lookup.
      3) Fallback to PyPI JSON API for torch.

    Args:
        version: Torch version (e.g. "2.7.1").
        py_tag: Python compatibility tag (e.g. "cp311").
        plat: Platform tag (e.g. "manylinux_2_28_x86_64").

    Returns:
        A tuple of (url, filename, sha256). sha256 may be None when unavailable.

    Raises:
        SystemExit: If metadata cannot be fetched/parsed or no suitable wheel exists.
    """
    explicit_url = os.environ.get("TORCH_WHEEL_URL", "").strip()
    if explicit_url:
        explicit_sha = os.environ.get("TORCH_WHEEL_SHA256", "").strip() or None
        explicit_filename = Path(urlparse(explicit_url).path).name
        return explicit_url, explicit_filename, explicit_sha

    cpu_index = "https://download.pytorch.org/whl/cpu/torch/"
    try:
        html = _open_https(cpu_index, timeout=30).read().decode("utf-8", "ignore")
        needle_prefix = (
            f"/whl/cpu/torch-{version}%2Bcpu-{py_tag}-{py_tag}-{plat}.whl#sha256="
        )
        parser = _TorchWheelIndexParser(needle_prefix=needle_prefix)
        parser.feed(html)
        href = parser.result
        if href:
            path, sha = href.split("#sha256=", 1)
            filename = Path(unquote(path)).name
            print(f"Using PyTorch CPU wheel: {filename}", flush=True)
            return f"https://download.pytorch.org{path}", filename, sha
        print("CPU wheel not found in index; falling back to PyPI.", flush=True)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"CPU wheel lookup failed ({exc}); falling back to PyPI.", flush=True)

    index_url = f"https://pypi.org/pypi/torch/{version}/json"
    print(f"Fetching torch metadata: {index_url}", flush=True)
    try:
        with _open_https(index_url, timeout=60) as resp:
            data = json.load(resp)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise SystemExit(
            f"Failed to fetch or parse torch metadata from {index_url}: {exc}"
        ) from exc
    candidates = [
        u
        for u in data.get("urls", [])
        if u.get("filename", "").endswith(f"{plat}.whl")
        and py_tag in u.get("filename", "")
    ]
    if not candidates:
        raise SystemExit(f"No torch wheel found for {version} {py_tag} {plat}")
    chosen = candidates[0]
    return (
        chosen["url"],
        chosen["filename"],
        chosen.get("digests", {}).get("sha256"),
    )


def _detect_platform() -> str:
    """Map the current machine architecture to a supported manylinux platform tag.

    Returns:
        A manylinux platform tag compatible with PyTorch CPU wheels.

    Raises:
        SystemExit: If the current architecture is not supported.
    """
    arch = platform.machine().lower()
    if arch in {"x86_64", "amd64"}:
        return "manylinux_2_28_x86_64"
    if arch in {"aarch64", "arm64"}:
        return "manylinux_2_28_aarch64"
    raise SystemExit(f"Unsupported architecture for torch wheel: {arch}")


def _sha256_digest(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Args:
        path: File path to hash.

    Returns:
        Lowercase hex SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_matches(path: Path, expected: str) -> bool:
    """Check whether a file's SHA-256 matches an expected digest.

    Args:
        path: File path to hash.
        expected: Expected hex digest.

    Returns:
        True if the digests match (case-insensitive), otherwise False.
    """
    digest = _sha256_digest(path)
    return digest.lower() == expected.lower()


def _ensure_https(url: str) -> None:
    """Validate that a URL uses HTTPS.

    Args:
        url: URL to validate.

    Raises:
        SystemExit: If the URL scheme is not HTTPS.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SystemExit(f"Refusing non-HTTPS URL: {url}")


def _open_https(url: str, timeout: int) -> HTTPResponse:
    """Open an HTTPS URL with a timeout.

    Args:
        url: HTTPS URL to open.
        timeout: Timeout in seconds.

    Returns:
        A file-like HTTP response.
    """
    _ensure_https(url)
    return urllib.request.urlopen(url, timeout=timeout)  # type: ignore[return-value]  # noqa: S310


def _open_https_request(req: urllib.request.Request, timeout: int) -> HTTPResponse:
    """Open an HTTPS request object with a timeout.

    Args:
        req: Prepared urllib request.
        timeout: Timeout in seconds.

    Returns:
        A file-like HTTP response.
    """
    _ensure_https(req.full_url)
    return urllib.request.urlopen(req, timeout=timeout)  # type: ignore[return-value]  # noqa: S310


def _cache_dir() -> Path:
    """Return the on-disk cache directory for downloaded wheels.

    Uses TORCH_CACHE_DIR when set; otherwise defaults to ~/.cache/torch.

    Returns:
        The cache directory path (created if missing).
    """
    cache_root = os.environ.get("TORCH_CACHE_DIR", "").strip()
    cache_dir = Path(cache_root) if cache_root else Path.home() / ".cache" / "torch"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _report_progress(downloaded: int, total: int | None) -> None:
    """Print coarse download progress.

    Args:
        downloaded: Bytes downloaded so far.
        total: Total bytes expected, if known.
    """
    if total:
        pct = (downloaded / total) * 100
        print(f"torch wheel download: {pct:.1f}%", flush=True)
        return
    mib = downloaded / (1024 * 1024)
    print(f"torch wheel download: {mib:.0f} MiB", flush=True)


def _download_once(url: str, dest: Path) -> None:
    """Download a URL to disk once, resuming if a partial file exists.

    If the server ignores the Range request (responds 200), the partial file is
    discarded and the download restarts from byte 0.

    Args:
        url: HTTPS URL to download.
        dest: Destination file path.
    """
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "docmind-docker"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)  # noqa: S310
    file_mode = "r+b" if dest.exists() else "wb"
    with _open_https_request(req, timeout=60) as resp, dest.open(file_mode) as fh:
        if existing and resp.status == 200:
            fh.seek(0)
            fh.truncate()
            existing = 0
        else:
            fh.seek(existing)
        content_range = resp.headers.get("Content-Range")
        if content_range and "/" in content_range:
            total_str = content_range.split("/")[-1].strip()
        else:
            total_str = resp.headers.get("Content-Length")
        total_int = int(total_str) if total_str and total_str.isdigit() else None
        downloaded = existing
        last_report = time.time()
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_report >= 15:
                _report_progress(downloaded, total_int)
                last_report = time.time()


def _handle_416(dest: Path, sha256: str | None) -> bool:
    """Handle HTTP 416 (Range Not Satisfiable) for a resume attempt.

    When a checksum is available, validates the existing file. Without a checksum,
    uses a conservative minimum-size heuristic to decide whether to keep the file.

    Args:
        dest: Existing destination file path.
        sha256: Expected sha256 digest, if known.

    Returns:
        True if the existing file should be accepted as complete; False if it
        should be deleted and re-downloaded.
    """
    if not dest.exists():
        return False
    if sha256:
        if _sha256_matches(dest, sha256):
            return True
        dest.unlink()
        return False
    file_size = dest.stat().st_size
    min_size = 50 * 1024 * 1024
    if file_size < min_size:
        print(f"File too small ({file_size} bytes); re-downloading.", flush=True)
        dest.unlink()
        return False
    print(
        "Received HTTP 416 for existing torch wheel with no checksum; "
        "assuming file is complete and proceeding without verification.",
        flush=True,
    )
    return True


def _download_with_retries(url: str, dest: Path, sha256: str | None) -> None:
    """Download a URL to disk with retries and basic error recovery.

    Retries transient network errors and handles HTTP 416 for resume cases.

    Args:
        url: HTTPS URL to download.
        dest: Destination file path.
        sha256: Expected sha256 digest, if known.
    """
    for attempt in range(1, 6):
        try:
            _download_once(url, dest)
            return
        except HTTPError as exc:
            if exc.code == 416 and _handle_416(dest, sha256):
                return
            print(f"torch wheel download failed (attempt {attempt}): {exc}", flush=True)
            if attempt == 5:
                raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(
                f"torch wheel download failed (attempt {attempt}): "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt == 5:
                raise
        time.sleep(min(2**attempt, 30))


def _verify_checksum(dest: Path, sha256: str | None) -> None:
    """Verify a file checksum when an expected digest is provided.

    Args:
        dest: File path to verify.
        sha256: Expected sha256 digest, or None to skip verification.

    Raises:
        SystemExit: If the checksum does not match.
    """
    if not sha256:
        return
    if not _sha256_matches(dest, sha256):
        raise SystemExit(
            f"Torch wheel checksum mismatch: expected {sha256}, got {_sha256_digest(dest)}"
        )


def main() -> None:
    """Resolve and download a torch wheel into the Docker build cache.

    Writes the final wheel path to torch-wheel.txt within the cache directory so
    Dockerfiles can copy/use it deterministically.
    """
    socket.setdefaulttimeout(60)
    version = os.environ.get("TORCH_VERSION", "2.7.1")
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat = _detect_platform()
    url, filename, sha256 = _resolve_torch_wheel_url(version, py_tag, plat)
    print(f"Downloading torch wheel: {filename}", flush=True)

    cache_dir = _cache_dir()
    dest = cache_dir / filename
    _download_with_retries(url, dest, sha256)
    _verify_checksum(dest, sha256)

    (cache_dir / "torch-wheel.txt").write_text(str(dest), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
