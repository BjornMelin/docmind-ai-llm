"""Tests for the ingestion pipeline builder."""

from __future__ import annotations

from types import SimpleNamespace

from src.models.processing import ProcessingStrategy
from src.processing.pipeline_builder import PipelineBuilder


def test_pipeline_builder_composes_factories(monkeypatch):
    """Builder should compose transforms, cache, and metadata deterministically."""
    recorded: dict[str, object] = {}

    def fake_ingestion_pipeline(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        "src.processing.pipeline_builder.IngestionPipeline",
        fake_ingestion_pipeline,
    )

    def factory_one(strategy: ProcessingStrategy):
        return [SimpleNamespace(label=f"{strategy.value}-a")]

    def factory_two(strategy: ProcessingStrategy):
        return [SimpleNamespace(label=f"{strategy.value}-b")]

    def cache_factory(strategy: ProcessingStrategy):
        return SimpleNamespace(kind="cache", strategy=strategy.value)

    def docstore_factory():
        return SimpleNamespace(kind="docstore")

    builder = PipelineBuilder(
        transform_factories=(factory_one, factory_two),
        cache_factory=cache_factory,
        docstore_factory=docstore_factory,
        metadata={"unit": "test"},
    )

    pipeline = builder.build(ProcessingStrategy.FAST)

    assert [t.label for t in pipeline.transformations] == [
        "fast-a",
        "fast-b",
    ]
    assert pipeline.cache.kind == "cache"
    assert pipeline.docstore.kind == "docstore"
    assert PipelineBuilder._pipeline_metadata[pipeline] == {
        "unit": "test",
        "strategy": "fast",
    }
    # Ensure factory outputs captured by stub
    assert recorded["cache"].strategy == "fast"
