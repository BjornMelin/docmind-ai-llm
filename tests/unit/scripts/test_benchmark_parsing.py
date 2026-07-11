"""Tests for parser benchmark evidence integrity."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import benchmark_parsing

pytestmark = pytest.mark.unit


def test_benchmark_schema_tracks_the_local_model_contract() -> None:
    """Benchmark output uses the post-hard-cut evidence schema."""
    assert benchmark_parsing.BENCHMARK_SCHEMA_VERSION == 3


def test_benchmark_evidence_rejects_a_dirty_repository() -> None:
    """Release evidence must identify one exact committed source tree."""
    with pytest.raises(RuntimeError, match="requires a clean Git worktree"):
        benchmark_parsing._require_clean_repository({"dirty": True})

    benchmark_parsing._require_clean_repository({"dirty": False})


def test_content_validation_checks_expected_pages_and_tokens() -> None:
    """Known fixtures must retain their physical pages and expected content."""
    result = SimpleNamespace(
        page_count=3,
        pages=[
            SimpleNamespace(text_markdown="Physical page one"),
            SimpleNamespace(text_markdown="Physical page two"),
            SimpleNamespace(text_markdown="Physical page three"),
        ],
    )

    validation = benchmark_parsing._content_validation(
        "multipage_native.pdf",
        result,
    )

    assert validation == {
        "matched_token_count": 2,
        "minimum_pages": 3,
        "passed": True,
        "required_token_count": 2,
    }


def test_benchmark_validation_rejects_empty_fixture_output() -> None:
    """A hash alone cannot make empty parser output valid evidence."""
    result = {
        "content_validation": {"passed": False},
        "deterministic": True,
        "error": None,
        "framework": "docling",
        "page_count": 1,
        "repetition_output_hashes": ["hash"],
        "source_filename": "scanned.pdf",
        "text_char_count": 0,
    }

    with pytest.raises(RuntimeError, match="content validation failed"):
        benchmark_parsing._validate_benchmark_results(
            [result],
            expected_repeats=1,
        )


def test_benchmark_validation_accepts_repeated_deterministic_output() -> None:
    """Valid evidence requires every repetition hash and content proof."""
    result = {
        "content_validation": {"passed": True},
        "deterministic": True,
        "error": None,
        "framework": "docling",
        "page_count": 1,
        "repetition_output_hashes": ["same", "same"],
        "source_filename": "born_digital.pdf",
        "text_char_count": 80,
    }

    benchmark_parsing._validate_benchmark_results(
        [result],
        expected_repeats=2,
    )


def test_isolated_result_combiner_preserves_any_failed_repetition() -> None:
    """A later successful run cannot erase an earlier worker failure."""
    failed = {
        "deterministic_output_hash": "",
        "error": {
            "type": "BenchmarkWorkerError",
            "returncode": -11,
            "message": "private detail",
        },
        "source_filename": "born_digital.pdf",
    }
    successful = {
        "deterministic_output_hash": "hash",
        "error": None,
        "source_filename": "born_digital.pdf",
    }

    combined = benchmark_parsing._combine_isolated_results(
        "born_digital.pdf",
        [failed, successful],
    )

    assert combined["error"] == {
        "failed_repetition": 1,
        "returncode": -11,
        "type": "BenchmarkWorkerError",
    }
    assert combined["repetition_output_hashes"] == ["hash"]
    assert combined["deterministic"] is False
    assert "private detail" not in str(combined)


def test_summary_reports_the_maximum_repetition_latency() -> None:
    """Aggregate maximum latency must not collapse to a maximum of medians."""
    results = [
        {
            "content_validation": {"passed": True},
            "deterministic": True,
            "error": None,
            "latency_ms_max": 25.0,
            "latency_ms_median": 10.0,
            "rss_mb": 100.0,
        },
        {
            "content_validation": {"passed": True},
            "deterministic": True,
            "error": None,
            "latency_ms_max": 22.0,
            "latency_ms_median": 20.0,
            "rss_mb": 110.0,
        },
    ]

    summary = benchmark_parsing._summary(results)

    assert summary["latency_ms_median"] == 15.0
    assert summary["latency_ms_max"] == 25.0
