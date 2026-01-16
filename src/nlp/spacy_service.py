"""Centralized spaCy runtime + typed enrichment outputs.

Design goals:
- One place where spaCy is imported and configured (device selection MUST happen
  before model loading).
- Fail-open by default: missing models fall back to `spacy.blank("en")`.
- Strict typing and schema-first outputs for downstream ingestion/UI.

Note: This module intentionally does not import Streamlit; Streamlit-level
caching is handled at UI boundaries via `st.cache_resource`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

from src.nlp.settings import SpacyCacheKey, SpacyDevice, SpacyNlpSettings

if TYPE_CHECKING:  # pragma: no cover
    from spacy.language import Language
    from spacy.tokens import Doc


class SentenceSpan(BaseModel):
    """A sentence span in character offsets."""

    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    text: str = Field(..., min_length=0)


class EntitySpan(BaseModel):
    """A named entity span in character offsets."""

    label: str = Field(..., min_length=1)
    text: str = Field(..., min_length=0)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    kb_id: str | None = Field(default=None, description="Knowledge base id (kb_id_)")
    ent_id: str | None = Field(default=None, description="Entity id (ent_id_)")


class SpacyEnrichment(BaseModel):
    """Typed enrichment payload produced from spaCy docs."""

    provider: str = Field(default="spacy")
    model: str = Field(..., min_length=1)
    sentences: list[SentenceSpan] = Field(default_factory=list)
    entities: list[EntitySpan] = Field(default_factory=list)


class SpacyModelLoadError(RuntimeError):
    """Raised when the caller requires GPU but spaCy cannot activate it."""


def _ensure_sentence_component(nlp: Language) -> None:
    """Ensure `doc.sents` is available via parser/senter/sentencizer."""
    try:
        if (
            nlp.has_pipe("parser")
            or nlp.has_pipe("senter")
            or nlp.has_pipe("sentencizer")
        ):
            return
        nlp.add_pipe("sentencizer")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("spaCy sentencizer init failed: {}", type(exc).__name__)


def _apply_device_preference(cfg: SpacyNlpSettings) -> None:
    """Apply device preference using spaCy top-level helpers.

    MUST be called before loading any pipeline/model.
    """
    from thinc.api import prefer_gpu, require_cpu, require_gpu

    device = cfg.device
    if device == SpacyDevice.CPU:
        require_cpu()
        return

    if device == SpacyDevice.CUDA:
        try:
            ok = require_gpu(int(cfg.gpu_id))
        except Exception:
            ok = False
        if not ok:
            raise SpacyModelLoadError(
                "SPACY_DEVICE=cuda requested, but no compatible GPU runtime is "
                "available. Install GPU deps (e.g., CuPy) and ensure CUDA is working."
            )
        return

    if device == SpacyDevice.APPLE:
        if sys.platform != "darwin":
            raise SpacyModelLoadError(
                "SPACY_DEVICE=apple requested, but this platform is not macOS."
            )
        try:
            ok = require_gpu(int(cfg.gpu_id))
        except Exception:
            ok = False
        if not ok:
            raise SpacyModelLoadError(
                "SPACY_DEVICE=apple requested, but spaCy could not activate Apple "
                "acceleration. Ensure `spacy[apple]==3.8.11` is installed."
            )
        return

    # AUTO
    try:
        ok = prefer_gpu(int(cfg.gpu_id))
    except Exception:
        ok = False
    if ok:
        return
    require_cpu()


@lru_cache(maxsize=8)
def _load_nlp_cached(cache_key: SpacyCacheKey) -> Language:
    """Load a spaCy pipeline with process-local caching."""
    cfg = SpacyNlpSettings.model_validate(
        {
            "enabled": cache_key[0],
            "model": cache_key[1],
            "device": cache_key[2],
            "gpu_id": cache_key[3],
            "disable_pipes": list(cache_key[4]),
            "batch_size": cache_key[5],
            "n_process": cache_key[6],
            "max_characters": cache_key[7],
        }
    )

    _apply_device_preference(cfg)

    import spacy

    disable = list({p.strip() for p in cfg.disable_pipes if p.strip()})
    try:
        nlp = spacy.load(cfg.model, disable=disable)
    except OSError as exc:
        logger.info(
            "spaCy model unavailable ({}); falling back to blank('en')", cfg.model
        )
        logger.debug("spaCy load error type: {}", type(exc).__name__)
        nlp = spacy.blank("en")

    _ensure_sentence_component(nlp)
    return nlp


class SpacyNlpService:
    """Central access point for spaCy loading and enrichment."""

    def __init__(self, cfg: SpacyNlpSettings) -> None:
        """Create a service instance for the given configuration."""
        self._cfg = cfg

    @property
    def cfg(self) -> SpacyNlpSettings:
        """Return the service configuration."""
        return self._cfg

    def load(self) -> Language:
        """Load (or reuse) the configured pipeline."""
        return _load_nlp_cached(self._cfg.cache_key())

    def enrich_doc(self, doc: Doc) -> SpacyEnrichment:
        """Extract typed sentences/entities from a spaCy doc."""
        sentences: list[SentenceSpan] = []
        try:
            for sent in doc.sents:
                sentences.append(
                    SentenceSpan(
                        start_char=int(sent.start_char),
                        end_char=int(sent.end_char),
                        text=str(sent.text),
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("spaCy sentence extraction failed: {}", type(exc).__name__)

        entities: list[EntitySpan] = []
        try:
            for ent in doc.ents:
                kb_id = str(ent.kb_id_) if getattr(ent, "kb_id_", "") else None
                ent_id = str(ent.ent_id_) if getattr(ent, "ent_id_", "") else None
                entities.append(
                    EntitySpan(
                        label=str(ent.label_),
                        text=str(ent.text),
                        start_char=int(ent.start_char),
                        end_char=int(ent.end_char),
                        kb_id=kb_id,
                        ent_id=ent_id,
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("spaCy entity extraction failed: {}", type(exc).__name__)

        return SpacyEnrichment(
            model=str(self._cfg.model),
            sentences=sentences,
            entities=entities,
        )

    def enrich_texts(self, texts: Sequence[str]) -> list[SpacyEnrichment]:
        """Enrich multiple texts via `nlp.pipe` (batch + optional multiprocessing)."""
        if not self._cfg.enabled:
            return [
                SpacyEnrichment(model=str(self._cfg.model), sentences=[], entities=[])
                for _ in texts
            ]

        capped: list[str] = []
        for text in texts:
            if len(text) > int(self._cfg.max_characters):
                capped.append("")
            else:
                capped.append(text)

        nlp = self.load()
        results: list[SpacyEnrichment] = []
        try:
            docs: Iterable[Doc] = nlp.pipe(
                capped,
                batch_size=int(self._cfg.batch_size),
                n_process=int(self._cfg.n_process),
            )
            for doc in docs:
                results.append(self.enrich_doc(doc))
        except Exception as exc:
            logger.warning(
                "spaCy enrichment failed (type={}): proceeding without NLP",
                type(exc).__name__,
            )
            logger.debug("spaCy enrichment error type: {}", type(exc).__name__)
            return [
                SpacyEnrichment(model=str(self._cfg.model), sentences=[], entities=[])
                for _ in texts
            ]

        return results


__all__ = [
    "EntitySpan",
    "SentenceSpan",
    "SpacyEnrichment",
    "SpacyModelLoadError",
    "SpacyNlpService",
]
