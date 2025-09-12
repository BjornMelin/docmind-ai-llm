from __future__ import annotations

import pytest

from src.eval.common.sorters import canonical_sort, round6


@pytest.mark.unit
def test_canonical_sort_tie_break_by_doc_id() -> None:
    items = [("b", 1.0), ("a", 1.0), ("c", 0.9)]
    out = canonical_sort(items)
    assert out == [("a", 1.0), ("b", 1.0), ("c", 0.9)]


@pytest.mark.unit
def test_round6() -> None:
    assert round6(0.1234564) == "0.123456"
    assert round6(0.1234566) == "0.123457"
