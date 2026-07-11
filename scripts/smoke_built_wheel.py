"""Verify a built wheel's package contents and published dependency contract."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import zipfile
from email import policy
from email.message import Message
from email.parser import BytesParser
from pathlib import Path

_REQUIRED_WHEEL_FILES = {
    "src/processing/parsing/service.py",
    "src/prompting/templates/prompts/comprehensive-analysis.prompt.md",
    "src/version.py",
}
_FORBIDDEN_WHEEL_FILES = {
    "src/models/embedding_constants.py",
    "src/models/embeddings.py",
    "src/processing/ocr_controller.py",
    "src/processing/parsing/validators.py",
    "src/processing/pipeline_builder.py",
    "src/processing/utils.py",
    "src/retrieval/adapter_registry.py",
    "src/retrieval/adapters/__init__.py",
    "src/retrieval/adapters/protocols.py",
}
_REQUIRED_WHEEL_DEPENDENCIES = {
    "docling",
    "fastembed",
    "llama-index-core",
    "llama-index-embeddings-huggingface",
    "llama-index-llms-ollama",
    "llama-index-llms-openai",
    "llama-index-llms-openai-like",
    "onnxruntime",
    "pypdfium2",
    "rapidocr",
    "torch",
}
_FORBIDDEN_WHEEL_DEPENDENCIES = {
    "fastembed-gpu",
    "kuzu",
    "llama-index",
    "llama-index-embeddings-clip",
    "llama-index-embeddings-fastembed",
    "llama-index-embeddings-openai",
    "llama-index-graph-stores-kuzu",
    "openai-whisper",
    "onnxruntime-gpu",
}
_FORBIDDEN_WHEEL_EXTRAS = {"graph", "llama"}


def _normalize_distribution_name(value: str) -> str:
    """Return the canonical distribution name at the start of a metadata value."""
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)
    if match is None:
        raise SystemExit(f"Invalid wheel metadata value: {value!r}")
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _read_wheel_metadata(archive: zipfile.ZipFile) -> Message:
    """Read the wheel's single core metadata document."""
    metadata_files = [
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_files) != 1:
        raise SystemExit(
            "Wheel must contain exactly one .dist-info/METADATA file; "
            f"found {len(metadata_files)}"
        )
    return BytesParser(policy=policy.default).parsebytes(
        archive.read(metadata_files[0])
    )


def _validate_wheel_metadata(metadata: Message) -> None:
    """Enforce the published dependency and extras contract."""
    dependencies = {
        _normalize_distribution_name(value)
        for value in metadata.get_all("Requires-Dist", [])
    }
    extras = {
        _normalize_distribution_name(value)
        for value in metadata.get_all("Provides-Extra", [])
    }

    missing = sorted(_REQUIRED_WHEEL_DEPENDENCIES.difference(dependencies))
    forbidden = sorted(_FORBIDDEN_WHEEL_DEPENDENCIES.intersection(dependencies))
    forbidden_extras = sorted(_FORBIDDEN_WHEEL_EXTRAS.intersection(extras))
    if missing:
        raise SystemExit(
            f"Wheel metadata is missing required dependencies: {', '.join(missing)}"
        )
    if forbidden:
        raise SystemExit(
            f"Wheel metadata contains forbidden dependencies: {', '.join(forbidden)}"
        )
    if forbidden_extras:
        raise SystemExit(
            f"Wheel metadata contains forbidden extras: {', '.join(forbidden_extras)}"
        )


def validate_wheel_contents(wheel: Path) -> None:
    """Reject incomplete wheels and stale package or dependency contracts."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata = _read_wheel_metadata(archive)
    missing = sorted(_REQUIRED_WHEEL_FILES.difference(names))
    forbidden = sorted(_FORBIDDEN_WHEEL_FILES.intersection(names))
    absent_from_source = sorted(
        name for name in names if name.startswith("src/") and not Path(name).is_file()
    )
    if missing:
        raise SystemExit(f"Wheel is missing required files: {', '.join(missing)}")
    if forbidden:
        raise SystemExit(f"Wheel contains deleted files: {', '.join(forbidden)}")
    if absent_from_source:
        raise SystemExit(
            "Wheel contains files absent from the source tree: "
            f"{', '.join(absent_from_source)}"
        )
    _validate_wheel_metadata(metadata)


def main() -> None:
    """Verify the newest wheel metadata, package, and prompt resources."""
    wheels = sorted(Path("dist").glob("docmind_ai_llm-*.whl"))
    if not wheels:
        raise SystemExit("No DocMind wheel found under dist/")
    wheel = wheels[-1].resolve()
    validate_wheel_contents(wheel)

    with tempfile.TemporaryDirectory(prefix="docmind-wheel-smoke-") as tmp:
        root = Path(tmp)
        venv = root / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                str(wheel),
            ],
            check=True,
        )
        code = f"""
from importlib.resources import files
from pathlib import Path
import src

target = Path({str(venv)!r}).resolve()
package_file = Path(src.__file__).resolve()
if not package_file.is_relative_to(target):
    raise SystemExit(f"imported source checkout instead of wheel: {{package_file}}")
template = files("src").joinpath(
    "prompting/templates/prompts/comprehensive-analysis.prompt.md"
)
body = template.read_text(encoding="utf-8")
if "comprehensive-analysis" not in body:
    raise SystemExit("representative packaged template could not be loaded")
print(f"wheel smoke ok: {{src.__version__}} {{template.name}}")
"""
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        subprocess.run(
            [str(python), "-I", "-c", code],
            cwd=root,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
