"""spaCy-based NLP enrichment for LlamaIndex nodes.

This module defines a single transform component that:
- Loads/uses the centralized spaCy runtime (via SpacyNlpService)
- Adds typed enrichment payloads to node metadata

The transform is optional and should be wired into the ingestion pipeline
based on settings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent

from src.nlp.settings import SpacyNlpSettings
from src.nlp.spacy_service import SpacyNlpService


class SpacyNlpEnrichmentTransform(TransformComponent):
    """Enrich nodes with spaCy sentence + entity extraction."""

    cfg: SpacyNlpSettings
    service: SpacyNlpService
    metadata_key: str = "docmind_nlp"

    def __call__(self, nodes: Sequence[BaseNode], **_: Any) -> Sequence[BaseNode]:
        """Enrich nodes in-place and return them."""
        if not self.cfg.enabled or not nodes:
            return nodes

        texts = [n.get_content(metadata_mode=MetadataMode.NONE) for n in nodes]
        enrichments = self.service.enrich_texts(texts)

        for node, enrichment in zip(nodes, enrichments, strict=False):
            meta = node.metadata or {}
            meta[self.metadata_key] = enrichment.model_dump()
            node.metadata = meta

        return nodes


__all__ = ["SpacyNlpEnrichmentTransform"]
