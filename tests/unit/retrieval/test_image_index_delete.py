"""Unit tests for multimodal image index lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.retrieval.image_index import (
    count_page_images_for_doc_id,
    delete_page_images_for_doc_id,
)

pytestmark = pytest.mark.unit


@dataclass
class _Count:
    count: int


class _Client:
    def __init__(self, count_value: int) -> None:
        self._count = int(count_value)
        self.count_calls = 0
        self.delete_calls = 0
        self.last_filter = None

    def count(self, *, collection_name: str, count_filter=None, exact: bool = True):  # type: ignore[no-untyped-def]
        del collection_name, exact
        self.count_calls += 1
        self.last_filter = count_filter
        return _Count(count=self._count)

    def delete(self, *, collection_name: str, points_selector, wait: bool = True):  # type: ignore[no-untyped-def]
        del collection_name, wait
        self.delete_calls += 1
        self.last_filter = points_selector
        return None


def test_delete_page_images_for_doc_id_returns_prior_count() -> None:
    client = _Client(count_value=7)
    prior = delete_page_images_for_doc_id(client, "images", doc_id="doc-1")  # type: ignore[arg-type]
    assert prior == 7
    assert client.count_calls >= 1
    assert client.delete_calls == 1
    assert client.last_filter is not None


def test_delete_page_images_for_doc_id_with_no_existing_records() -> None:
    client = _Client(count_value=0)
    prior = delete_page_images_for_doc_id(client, "images", doc_id="doc-1")  # type: ignore[arg-type]
    assert prior == 0
    assert client.delete_calls == 1


def test_delete_page_images_for_doc_id_handles_delete_exception() -> None:
    class _FailingClient(_Client):
        def delete(  # type: ignore[override]
            self, *, collection_name: str, points_selector, wait: bool = True
        ):
            raise RuntimeError("delete failed")

    client = _FailingClient(count_value=4)
    prior = delete_page_images_for_doc_id(client, "images", doc_id="doc-1")  # type: ignore[arg-type]
    assert prior == 4


def test_count_page_images_for_doc_id_best_effort() -> None:
    client = _Client(count_value=3)
    count = count_page_images_for_doc_id(client, "images", doc_id="doc-1")  # type: ignore[arg-type]
    assert count == 3
