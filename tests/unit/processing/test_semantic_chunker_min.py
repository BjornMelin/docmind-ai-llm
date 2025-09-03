"""Minimal unit tests for SemanticChunker parameter forwarding.

Focus on parameter forwarding and basic conversion only (library-first;
no reimplementation of chunking logic).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.processing.chunking.unstructured_chunker import (
    SemanticChunker,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_semantic_chunker_parameter_forwarding_defaults_async() -> None:
    """Chunker forwards default settings into chunk_by_title."""
    chunker = SemanticChunker()
    # Through async path with minimal valid elements and patched converter
    dummy = type("E", (), {"text": "A", "category": "NarrativeText", "metadata": {}})()
    with (
        patch(
            "src.processing.chunking.unstructured_chunker.SemanticChunker._convert_document_elements_to_unstructured",
            return_value=[dummy],
        ),
        patch(
            "src.processing.chunking.unstructured_chunker.chunk_by_title",
            return_value=[dummy],
        ) as mock_title,
    ):
        await chunker.chunk_elements_async([dummy])
        # Call happened with default parameters
        assert mock_title.called
        kwargs = mock_title.call_args.kwargs
        assert "max_characters" in kwargs
        assert kwargs["max_characters"] > 0
        assert "new_after_n_chars" in kwargs
        assert kwargs["new_after_n_chars"] > 0
        assert "combine_text_under_n_chars" in kwargs
        assert "multipage_sections" in kwargs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_semantic_chunker_parameter_validation_async() -> None:
    """Invalid relationships raise ValueError before invoking chunk_by_title."""
    chunker = SemanticChunker()
    elems = [type("E", (), {"text": "A", "category": "Title", "metadata": {}})()]
    params = chunker.default_parameters.model_copy()
    # Make invalid: combine_under >= new_after
    params.combine_text_under_n_chars = params.new_after_n_chars

    with pytest.raises(ValueError, match="combine_under < new_after < max_chars"):
        await chunker.chunk_elements_async(elems, parameters=params)
