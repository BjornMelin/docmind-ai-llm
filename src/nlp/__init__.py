"""NLP subsystem (spaCy runtime + enrichment helpers)."""

from src.nlp.settings import SpacyDevice, SpacyNlpSettings
from src.nlp.spacy_service import (
    EntitySpan,
    SentenceSpan,
    SpacyEnrichment,
    SpacyModelLoadError,
    SpacyNlpService,
)

__all__ = [
    "EntitySpan",
    "SentenceSpan",
    "SpacyDevice",
    "SpacyEnrichment",
    "SpacyModelLoadError",
    "SpacyNlpService",
    "SpacyNlpSettings",
]
