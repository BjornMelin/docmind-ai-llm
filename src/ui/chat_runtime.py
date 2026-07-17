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


def _resolve_model_shard(manifest_parent: Path, shard_name: object) -> Path | None:
    """Resolve one safe shard path beneath its manifest directory."""
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
        shard.relative_to(manifest_parent)
    except (OSError, RuntimeError, ValueError):
        return None
    return shard


def _sharded_weights_ready(index_path: Path) -> bool:
    """Validate every unique shard named by a Transformers weight index."""
    try:
        manifest = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    weight_map = manifest.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False

    try:
        manifest_parent = index_path.parent.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    shards: set[Path] = set()
    for shard_name in weight_map.values():
        shard = _resolve_model_shard(manifest_parent, shard_name)
        if shard is None:
            return False
        shards.add(shard)

    return all(_nonempty_regular_file(shard) for shard in shards)


def _local_model_artifacts_ready(model_path: Path) -> bool:
    """Return whether a local SentenceTransformers directory has core assets."""
    roots = (model_path, model_path / "0_Transformer")
    has_config = False
    for root in roots:
        config_path = root / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(config, dict) and config:
            has_config = True
            break
    direct_weight_names = ("model.safetensors", "pytorch_model.bin")
    index_names = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    has_weights = False
    for root in roots:
        if any(_nonempty_regular_file(root / name) for name in direct_weight_names):
            has_weights = True
            break
        if any(_sharded_weights_ready(root / name) for name in index_names):
            has_weights = True
            break
    return has_config and has_weights


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
    if not _local_model_artifacts_ready(Path(snapshot_path)):
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
