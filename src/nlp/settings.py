"""Typed spaCy settings (device + model selection).

This module intentionally avoids importing spaCy at import time. All spaCy
imports happen inside `src.nlp.spacy_service`.
"""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class SpacyDevice(StrEnum):
    """Device selection for spaCy/Thinc ops."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    APPLE = "apple"


type SpacyCacheKey = tuple[
    bool, str, str, int, tuple[str, ...], int, int, int
]  # (enabled, model, device, gpu_id, disable_pipes, batch_size, n_process, max_chars)


class SpacyNlpSettings(BaseModel):
    """Configuration for the centralized spaCy NLP runtime."""

    enabled: bool = Field(
        default=True, description="Enable spaCy enrichment during ingestion"
    )
    model: str = Field(
        default="en_core_web_sm",
        min_length=1,
        description="spaCy pipeline name or path (e.g., en_core_web_sm)",
    )
    device: SpacyDevice = Field(
        default=SpacyDevice.AUTO,
        description="Device preference for spaCy ops (cpu|cuda|apple|auto)",
    )
    gpu_id: int = Field(
        default=0,
        ge=0,
        le=128,
        description="CUDA GPU index when device is cuda/auto",
    )
    disable_pipes: list[str] = Field(
        default_factory=list,
        description="Pipeline components to disable on load (e.g., ['parser'])",
    )
    batch_size: int = Field(
        default=32, ge=1, le=4096, description="spaCy nlp.pipe batch size"
    )
    n_process: int = Field(
        default=1,
        ge=1,
        le=32,
        description="spaCy nlp.pipe worker processes (1 disables multiprocessing)",
    )
    max_characters: int = Field(
        default=200_000,
        ge=1,
        le=10_000_000,
        description="Skip enrichment for node texts longer than this cap",
    )

    def cache_key(self) -> SpacyCacheKey:
        """Return a stable, hashable key for caching a loaded pipeline."""
        pipes = tuple(sorted({p.strip() for p in self.disable_pipes if p.strip()}))
        return (
            bool(self.enabled),
            str(self.model),
            str(self.device),
            int(self.gpu_id),
            pipes,
            int(self.batch_size),
            int(self.n_process),
            int(self.max_characters),
        )

    @field_validator("disable_pipes", mode="before")
    @classmethod
    def _parse_disable_pipes(cls, value: object) -> list[str]:
        result: list[str] = []
        if value is None:
            return result

        parsed: object = value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return result
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw

        if isinstance(parsed, str):
            return [s.strip() for s in parsed.split(",") if s.strip()]
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return result


__all__ = ["SpacyCacheKey", "SpacyDevice", "SpacyNlpSettings"]
