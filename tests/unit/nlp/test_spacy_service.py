"""Unit tests for the centralized spaCy NLP subsystem."""

from __future__ import annotations

import pytest
from llama_index.core.schema import TextNode

from src.nlp.settings import SpacyDevice, SpacyNlpSettings
from src.nlp.spacy_service import SpacyModelLoadError, SpacyNlpService
from src.processing.nlp_enrichment import SpacyNlpEnrichmentTransform

pytestmark = [pytest.mark.unit, pytest.mark.requires_gpu]


def _clear_spacy_cache() -> None:
    from src.nlp import spacy_service as module

    module._load_nlp_cached.cache_clear()


def test_enrich_texts_blank_fallback_produces_schema() -> None:
    """Model missing -> blank('en') fallback -> stable schema."""
    _clear_spacy_cache()
    cfg = SpacyNlpSettings(model="__missing_model__", device=SpacyDevice.CPU)
    service = SpacyNlpService(cfg)

    out = service.enrich_texts(["Hello world. Second sentence."])

    assert len(out) == 1
    assert out[0].model == "__missing_model__"
    assert isinstance(out[0].sentences, list)
    assert isinstance(out[0].entities, list)
    assert out[0].sentences, "sentencizer should produce at least one sentence"


def test_cuda_device_fails_fast_when_require_gpu_false(monkeypatch) -> None:
    """SPACY_DEVICE=cuda must fail fast when spaCy can't activate GPU."""
    _clear_spacy_cache()
    import thinc.api as thinc_api

    monkeypatch.setattr(thinc_api, "require_gpu", lambda gpu_id=0: False)
    cfg = SpacyNlpSettings(model="__missing_cuda__", device=SpacyDevice.CUDA)
    service = SpacyNlpService(cfg)

    with pytest.raises(SpacyModelLoadError):
        service.load()


@pytest.mark.requires_gpu
def test_cuda_device_smoke_when_available() -> None:
    """Optional GPU smoke test (manual)."""
    _clear_spacy_cache()
    from thinc.api import prefer_gpu

    try:
        ok = prefer_gpu(0)
    except Exception:  # pragma: no cover - environment-dependent
        pytest.skip("GPU runtime not available")
    if not ok:
        pytest.skip("GPU runtime not available")

    cfg = SpacyNlpSettings(model="__missing_cuda__", device=SpacyDevice.CUDA)
    service = SpacyNlpService(cfg)
    _ = service.load()


def test_spacy_transform_writes_docmind_nlp_metadata() -> None:
    _clear_spacy_cache()
    cfg = SpacyNlpSettings(model="__missing_model__", device=SpacyDevice.CPU)
    service = SpacyNlpService(cfg)
    transform = SpacyNlpEnrichmentTransform(cfg=cfg, service=service)

    nodes = [TextNode(text="DocMind tests. One sentence.")]
    out = list(transform(nodes))
    assert out
    payload = out[0].metadata.get("docmind_nlp")
    assert isinstance(payload, dict)
    assert payload.get("provider") == "spacy"
    assert isinstance(payload.get("sentences"), list)


def test_enrich_texts_does_not_log_raw_exception_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure exceptions with user content are not emitted in logs."""
    _clear_spacy_cache()
    cfg = SpacyNlpSettings(model="__missing_model__", device=SpacyDevice.CPU)
    service = SpacyNlpService(cfg)

    pii_sentinel = "SSN 123-45-6789"

    class _FailingNLP:
        def pipe(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError(f"boom: {pii_sentinel}")

    monkeypatch.setattr(
        SpacyNlpService,
        "load",
        lambda _self: _FailingNLP(),  # type: ignore[return-value]
        raising=True,
    )

    from loguru import logger

    captured: list[str] = []
    token = logger.add(
        lambda message: captured.append(message.rstrip("\n")),
        format="{message}",
        level="DEBUG",
    )
    try:
        out = service.enrich_texts(["Hello world."])
    finally:
        logger.remove(token)

    assert len(out) == 1
    joined = "\n".join(captured)
    assert pii_sentinel not in joined
    assert "spaCy enrichment error type" in joined
