"""Unit tests for multimodal image index lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.retrieval.image_index import (
    count_page_images_for_doc_id,
)

pytestmark = pytest.mark.unit


@dataclass
class _Count:
    count: int


class _Client:
    def __init__(self, count_value: int) -> None:
        self._count = int(count_value)
        self.count_calls = 0
        self.last_filter = None

    def count(
        self, *, collection_name: str, count_filter: Any = None, exact: bool = True
    ) -> _Count:
        del collection_name, exact
        self.count_calls += 1
        self.last_filter = count_filter
        return _Count(count=self._count)


def test_count_page_images_for_doc_id_best_effort() -> None:
    client = _Client(count_value=3)
    count = count_page_images_for_doc_id(client, "images", doc_id="doc-1")  # type: ignore[arg-type]
    assert count == 3
