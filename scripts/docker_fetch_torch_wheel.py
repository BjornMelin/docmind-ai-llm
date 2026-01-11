"""Fetch the PyTorch wheel during Docker builds with retry and checksum safety."""

import hashlib
import json
import os
import platform
import socket
import sys
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse


def _resolve_torch_wheel_url(
    version: str, py_tag: str, plat: str
) -> tuple[str, str, str | None]:
    explicit_url = os.environ.get("TORCH_WHEEL_URL", "").strip()
    if explicit_url:
        explicit_sha = os.environ.get("TORCH_WHEEL_SHA256", "").strip() or None
        explicit_filename = Path(urlparse(explicit_url).path).name
        return explicit_url, explicit_filename, explicit_sha

    cpu_index = "https://download.pytorch.org/whl/cpu/torch/"
    try:
        html = _open_https(cpu_index, timeout=30).read().decode("utf-8", "ignore")
        needle = f"/whl/cpu/torch-{version}%2Bcpu-{py_tag}-{py_tag}-{plat}.whl#sha256="
        idx = html.find(needle)
        if idx != -1:
            end = html.find('"', idx)
            href = html[idx:end]
            path, sha = href.split("#sha256=", 1)
            filename = Path(unquote(path)).name
            print(f"Using PyTorch CPU wheel: {filename}", flush=True)
            return f"https://download.pytorch.org{path}", filename, sha
        print("CPU wheel not found in index; falling back to PyPI.", flush=True)
    except Exception as exc:
        print(f"CPU wheel lookup failed ({exc}); falling back to PyPI.", flush=True)

    index_url = f"https://pypi.org/pypi/torch/{version}/json"
    print(f"Fetching torch metadata: {index_url}", flush=True)
    try:
        with _open_https(index_url, timeout=60) as resp:
            data = json.load(resp)
    except Exception as exc:
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
    arch = platform.machine().lower()
    if arch in {"x86_64", "amd64"}:
        return "manylinux_2_28_x86_64"
    if arch in {"aarch64", "arm64"}:
        return "manylinux_2_28_aarch64"
    raise SystemExit(f"Unsupported architecture for torch wheel: {arch}")


def _sha256_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_matches(path: Path, expected: str) -> bool:
    digest = _sha256_digest(path)
    return digest.lower() == expected.lower()


def _ensure_https(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SystemExit(f"Refusing non-HTTPS URL: {url}")


def _open_https(url: str, timeout: int) -> urllib.request.addinfourl:
    _ensure_https(url)
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310


def _open_https_request(
    req: urllib.request.Request, timeout: int
) -> urllib.request.addinfourl:
    _ensure_https(req.full_url)
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310


def main() -> None:
    """Resolve and download the torch wheel into the Docker build cache."""
    socket.setdefaulttimeout(60)
    version = os.environ.get("TORCH_VERSION", "2.7.1")
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat = _detect_platform()
    url, filename, sha256 = _resolve_torch_wheel_url(version, py_tag, plat)
    print(f"Downloading torch wheel: {filename}", flush=True)

    cache_root = os.environ.get("TORCH_CACHE_DIR", "").strip()
    cache_dir = Path(cache_root) if cache_root else Path.home() / ".cache" / "torch"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / filename

    for attempt in range(1, 6):
        try:
            existing = dest.stat().st_size if dest.exists() else 0
            headers = {"User-Agent": "docmind-docker"}
            if existing:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)  # noqa: S310
            with _open_https_request(req, timeout=60) as resp, dest.open("ab") as fh:
                total = resp.headers.get("Content-Range")
                if total and "/" in total:
                    total = total.split("/")[-1].strip()
                else:
                    total = resp.headers.get("Content-Length")
                total = int(total) if total and total.isdigit() else None
                downloaded = existing
                last_report = time.time()
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - last_report >= 15:
                        if total:
                            pct = (downloaded / total) * 100
                            print(f"torch wheel download: {pct:.1f}%", flush=True)
                        else:
                            mib = downloaded / (1024 * 1024)
                            print(f"torch wheel download: {mib:.0f} MiB", flush=True)
                        last_report = time.time()
            break
        except HTTPError as exc:
            if exc.code == 416 and dest.exists():
                if sha256:
                    if _sha256_matches(dest, sha256):
                        break
                    dest.unlink()
                else:
                    # Without a checksum we can only assume the file is complete.
                    print(
                        "Received HTTP 416 for existing torch wheel with no checksum; "
                        "assuming file is complete and proceeding "
                        "without verification.",
                        flush=True,
                    )
                    break
            print(f"torch wheel download failed (attempt {attempt}): {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))
        except Exception as exc:
            print(
                f"torch wheel download failed (attempt {attempt}): "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))

    if sha256:
        digest = _sha256_digest(dest)
        if digest.lower() != sha256.lower():
            raise SystemExit(
                f"Torch wheel checksum mismatch: expected {sha256}, got {digest}"
            )

    (cache_dir / "torch-wheel.txt").write_text(str(dest), encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
