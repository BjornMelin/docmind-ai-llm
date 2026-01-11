"""Fetch the PyTorch wheel during Docker builds with retry and checksum safety."""

import hashlib
import json
import os
import platform
import socket
import sys
import time
import urllib.request
from urllib.error import HTTPError
from urllib.parse import unquote, urlparse


def _resolve_torch_wheel_url(
    version: str, py_tag: str, plat: str
) -> tuple[str, str, str | None]:
    explicit_url = os.environ.get("TORCH_WHEEL_URL", "").strip()
    if explicit_url:
        explicit_sha = os.environ.get("TORCH_WHEEL_SHA256", "").strip() or None
        return explicit_url, os.path.basename(explicit_url), explicit_sha

    cpu_index = "https://download.pytorch.org/whl/cpu/torch/"
    try:
        html = _open_https(cpu_index, timeout=30).read().decode("utf-8", "ignore")
        needle = f"/whl/cpu/torch-{version}%2Bcpu-{py_tag}-{py_tag}-{plat}.whl#sha256="
        idx = html.find(needle)
        if idx != -1:
            end = html.find('"', idx)
            href = html[idx:end]
            path, sha = href.split("#sha256=", 1)
            filename = unquote(os.path.basename(path))
            print(f"Using PyTorch CPU wheel: {filename}", flush=True)
            return f"https://download.pytorch.org{path}", filename, sha
        print("CPU wheel not found in index; falling back to PyPI.", flush=True)
    except Exception as exc:
        print(f"CPU wheel lookup failed ({exc}); falling back to PyPI.", flush=True)

    index_url = f"https://pypi.org/pypi/torch/{version}/json"
    print(f"Fetching torch metadata: {index_url}", flush=True)
    with _open_https(index_url, timeout=60) as resp:
        data = json.load(resp)
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


def _sha256_digest(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_matches(path: str, expected: str) -> bool:
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

    cache_dir = "/root/.cache/torch"
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, filename)

    for attempt in range(1, 6):
        try:
            existing = os.path.getsize(dest) if os.path.exists(dest) else 0
            headers = {"User-Agent": "docmind-docker"}
            if existing:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)  # noqa: S310
            with _open_https_request(req, timeout=60) as resp, open(dest, "ab") as fh:
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
            if exc.code == 416 and os.path.exists(dest):
                if sha256 and _sha256_matches(dest, sha256):
                    break
                os.remove(dest)
            print(f"torch wheel download failed (attempt {attempt}): {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(5)
        except Exception as exc:
            print(f"torch wheel download failed (attempt {attempt}): {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(5)

    if sha256:
        digest = _sha256_digest(dest)
        if digest.lower() != sha256.lower():
            raise SystemExit(
                f"Torch wheel checksum mismatch: expected {sha256}, got {digest}"
            )

    with open(os.path.join(cache_dir, "torch-wheel.txt"), "w", encoding="utf-8") as fh:
        fh.write(dest)
    print(dest)


if __name__ == "__main__":
    main()
