"""Process-wide owner for the active Chat coordinator."""

from __future__ import annotations

import atexit
import json
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from src.ui.vector_session import retire_session_runtime_resources

if TYPE_CHECKING:
    from src.agents.coordinator import MultiAgentCoordinator

_LOCK = threading.RLock()
_COORDINATOR: MultiAgentCoordinator | None = None
_RESOURCE_KEY: tuple[int, Path, int] | None = None

type ChatModelFailureStatus = Literal[
    "cache_missing", "local_path_incomplete", "initialization_failed"
]


class ChatModelUnavailableError(RuntimeError):
    """Raised with one sanitized configured-embedding failure state."""

    def __init__(
        self,
        status: ChatModelFailureStatus,
    ) -> None:
        """Create a sanitized failure carrying only its canonical UI state."""
        super().__init__("Configured Chat embedding is unavailable")
        self.status: ChatModelFailureStatus = status


@dataclass(frozen=True, slots=True)
class ChatModelReadiness:
    """Local-only readiness for the configured Chat embedding artifacts."""

    status: Literal["ready"] | ChatModelFailureStatus


def _nonempty_regular_file(path: Path) -> bool:
    """Return whether ``path`` resolves to a nonempty regular file."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _contained_nonempty_regular_file(path: Path, *, containment_root: Path) -> bool:
    """Return whether ``path`` is a nonempty file inside its storage boundary."""
    try:
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(containment_root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        return False
    return _nonempty_regular_file(resolved_path)


def _marian_tokenizer_artifacts_ready(
    root: Path,
    *,
    containment_root: Path,
) -> bool:
    """Validate Marian's paired SentencePiece payload and vocabulary."""

    def has_file(name: str) -> bool:
        return _contained_nonempty_regular_file(
            root / name,
            containment_root=containment_root,
        )

    tokenizer_config_path = root / "tokenizer_config.json"
    if not all(has_file(name) for name in ("source.spm", "target.spm", "vocab.json")):
        return False
    try:
        tokenizer_config = json.loads(
            tokenizer_config_path.resolve(strict=True).read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError, RuntimeError, UnicodeError):
        return False
    return not (
        isinstance(tokenizer_config, dict)
        and tokenizer_config.get("separate_vocabs") is True
        and not has_file("target_vocab.json")
    )


def _tokenizer_artifacts_ready(root: Path, *, containment_root: Path) -> bool:
    """Validate one supported Transformers tokenizer family in ``root``."""

    def has_file(name: str) -> bool:
        return _contained_nonempty_regular_file(
            root / name,
            containment_root=containment_root,
        )

    if not has_file("tokenizer_config.json"):
        return False
    if has_file("tokenizer.json") or has_file("vocab.txt"):
        return True
    if any(
        _contained_nonempty_regular_file(
            candidate,
            containment_root=containment_root,
        )
        for pattern in ("*.model", "*.tokenizer")
        for candidate in root.glob(pattern)
    ):
        return True
    if has_file("source.spm") or has_file("target.spm"):
        return _marian_tokenizer_artifacts_ready(
            root,
            containment_root=containment_root,
        )
    return has_file("vocab.json") and has_file("merges.txt")


def _resolve_model_shard(
    manifest_parent: Path,
    shard_name: object,
    *,
    containment_root: Path,
) -> Path | None:
    """Resolve one lexical child shard beneath its allowed storage root."""
    if not isinstance(shard_name, str) or not shard_name.strip():
        return None
    posix_path = PurePosixPath(shard_name)
    windows_path = PureWindowsPath(shard_name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        return None
    try:
        shard = (manifest_parent / Path(shard_name)).resolve(strict=True)
        shard.relative_to(containment_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return shard


def _sharded_weights_ready(index_path: Path, *, containment_root: Path) -> bool:
    """Validate every unique shard named by a Transformers weight index."""
    try:
        resolved_index_path = index_path.resolve(strict=True)
        manifest_parent = index_path.parent.resolve(strict=True)
        resolved_containment_root = containment_root.resolve(strict=True)
        resolved_index_path.relative_to(resolved_containment_root)
        manifest_parent.relative_to(resolved_containment_root)
    except (OSError, RuntimeError, ValueError):
        return False
    try:
        manifest = json.loads(resolved_index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    weight_map = manifest.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shards: set[Path] = set()
    for shard_name in weight_map.values():
        shard = _resolve_model_shard(
            manifest_parent,
            shard_name,
            containment_root=resolved_containment_root,
        )
        if shard is None:
            return False
        shards.add(shard)

    return all(_nonempty_regular_file(shard) for shard in shards)


def _local_model_artifacts_ready(
    model_path: Path,
    *,
    shard_containment_root: Path | None = None,
) -> bool:
    """Return whether a local SentenceTransformers directory has core assets."""
    roots = (model_path, model_path / "0_Transformer")
    containment_root = shard_containment_root or model_path
    for root in roots:
        config_path = root / "config.json"
        if not _contained_nonempty_regular_file(
            config_path,
            containment_root=containment_root,
        ):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(config, dict) or not config:
            continue
        if not _tokenizer_artifacts_ready(root, containment_root=containment_root):
            continue
        direct_weight_names = ("model.safetensors", "pytorch_model.bin")
        if any(
            _contained_nonempty_regular_file(
                root / name,
                containment_root=containment_root,
            )
            for name in direct_weight_names
        ):
            return True
        index_names = (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        if any(
            _sharded_weights_ready(
                root / name,
                containment_root=containment_root,
            )
            for name in index_names
        ):
            return True
    return False


def check_model_artifacts(
    *,
    model_name: str,
    model_revision: str | None,
    cache_folder: Path,
    local_model_path: Path | None,
) -> ChatModelReadiness:
    """Check the native Hugging Face cache without network or model imports."""
    if local_model_path is not None:
        if _local_model_artifacts_ready(Path(local_model_path).expanduser()):
            return ChatModelReadiness(status="ready")
        return ChatModelReadiness(status="local_path_incomplete")

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError

    try:
        snapshot_path = snapshot_download(
            repo_id=model_name,
            revision=model_revision,
            cache_dir=cache_folder.expanduser(),
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        return ChatModelReadiness(status="cache_missing")
    except (HFValidationError, OSError):
        return ChatModelReadiness(status="initialization_failed")
    if not _local_model_artifacts_ready(
        Path(snapshot_path),
        shard_containment_root=cache_folder.expanduser(),
    ):
        return ChatModelReadiness(status="cache_missing")
    return ChatModelReadiness(status="ready")


def _retire_session_resources() -> None:
    """Retire loop-bound routers and their vector clients before coordinator close."""
    retire_session_runtime_resources()


def get_coordinator(
    *,
    cache_version: int,
    checkpointer_path: Path,
    store: Any,
) -> MultiAgentCoordinator:
    """Return the coordinator for the current runtime generation.

    A settings generation change replaces and closes the previous coordinator,
    so provider, model, tool, and timeout changes cannot leave Chat on a stale
    graph.
    """
    global _COORDINATOR, _RESOURCE_KEY

    from src.agents.coordinator import MultiAgentCoordinator

    key = (int(cache_version), Path(checkpointer_path), id(store))
    previous: MultiAgentCoordinator | None
    with _LOCK:
        if _COORDINATOR is not None and key == _RESOURCE_KEY:
            return _COORDINATOR

        replacement = MultiAgentCoordinator(
            checkpointer_path=checkpointer_path,
            store=store,
        )
        previous = _COORDINATOR
        if previous is not None:
            # Keep replacement publication behind the runtime lock so another
            # session cannot register a new-generation router during retirement.
            _retire_session_resources()
        _COORDINATOR = replacement
        _RESOURCE_KEY = key
    # Coordinator close can wait for its bounded graph-runner grace. Session
    # resource retirement is fenced above; release the lock before this wait.
    if previous is not None:
        previous.close()
    return replacement


def invalidate_coordinator() -> None:
    """Detach and best-effort close the active coordinator without raising."""
    global _COORDINATOR, _RESOURCE_KEY

    with _LOCK:
        coordinator = _COORDINATOR
        try:
            _retire_session_resources()
        except Exception as exc:  # pragma: no cover - defensive cleanup boundary
            logger.warning(
                "Session runtime retirement failed (error_type={})",
                type(exc).__name__,
            )
        finally:
            _COORDINATOR = None
            _RESOURCE_KEY = None
    if coordinator is not None:
        try:
            coordinator.close()
        except Exception as exc:  # pragma: no cover - defensive cleanup boundary
            logger.warning(
                "Chat coordinator cleanup failed (error_type={})",
                type(exc).__name__,
            )


atexit.register(invalidate_coordinator)

__all__ = [
    "ChatModelReadiness",
    "ChatModelUnavailableError",
    "check_model_artifacts",
    "get_coordinator",
    "invalidate_coordinator",
]
