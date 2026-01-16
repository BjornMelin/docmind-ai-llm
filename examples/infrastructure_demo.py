#!/usr/bin/env python3
"""Demo of lightweight system/GPU info and the centralized spaCy NLP subsystem.

This example demonstrates:
1. Simple system info via psutil (no heavy deps)
2. Optional GPU stats via torch.cuda if available
3. Centralized spaCy loading + device selection (cpu|cuda|apple|auto)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.settings import SpacyNlpSettings
from src.nlp.spacy_service import SpacyModelLoadError, SpacyNlpService
from src.utils.monitoring import get_system_info

try:
    import torch
except ImportError:  # torch may be unavailable in some envs
    torch = None  # type: ignore


def demo_system_and_gpu():
    """Show basic system info and optional CUDA stats (if available)."""
    print("=== System/GPU Monitoring Demo ===")

    if info := get_system_info():
        print("System Info:")
        print(f"  CPU %: {info.get('cpu_percent')}%")
        print(f"  Mem %: {info.get('memory_percent')}%")
        if avg := info.get("load_average"):
            print(f"  Load Avg: {avg}")

    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        print("No CUDA GPU available - running on CPU")
        return

    try:
        device_index = 0
        name = torch.cuda.get_device_name(device_index)
        allocated_gb = torch.cuda.memory_allocated(device_index) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(device_index) / (1024**3)
        total_gb = torch.cuda.get_device_properties(device_index).total_memory / (
            1024**3
        )
        print("GPU Info:")
        print(f"  Device: {name}")
        print(f"  Memory Allocated: {allocated_gb:.2f} GB")
        print(f"  Memory Reserved:  {reserved_gb:.2f} GB")
        print(f"  Total Memory:     {total_gb:.2f} GB")
    except (RuntimeError, AssertionError) as e:  # keep demo resilient
        print(f"GPU info unavailable: {e}")


def demo_spacy_optimization():
    """Demonstrate centralized spaCy device selection + enrichment."""
    print("\n=== spaCy NLP Demo ===")

    gpu_id_str = os.getenv("SPACY_GPU_ID", "0")
    try:
        gpu_id = int(gpu_id_str)
    except ValueError:
        # Avoid crashing before validation; fallback to 0
        gpu_id = 0

    cfg = SpacyNlpSettings.model_validate(
        {
            "enabled": True,
            "model": os.getenv("SPACY_MODEL", "en_core_web_sm"),
            "device": os.getenv("SPACY_DEVICE", "auto"),
            "gpu_id": gpu_id,
            "disable_pipes": os.getenv("SPACY_DISABLE_PIPES", ""),
        }
    )
    service = SpacyNlpService(cfg)
    test_texts = [
        "This is the first document to process.",
        "Here's another example sentence for NLP analysis.",
        "DocMind centralizes spaCy usage for maintainability.",
        "Device selection happens before model loading.",
    ]

    print(f"Processing {len(test_texts)} texts...")

    try:
        out = service.enrich_texts(test_texts)

        print("\nProcessed documents:")
        for i, enrichment in enumerate(out, 1):
            print(
                f"  {i}. Sentences: {len(enrichment.sentences)}, "
                f"Entities: {len(enrichment.entities)}"
            )

        if out and out[0].entities:
            ent0 = out[0].entities[0]
            print(f"\nSample entity: {ent0.label} -> {ent0.text}")
    except SpacyModelLoadError as e:
        print(f"spaCy GPU activation failed: {e}")
        print("Tip: set SPACY_DEVICE=auto or install the correct GPU extras.")
    except Exception as e:
        print(f"spaCy processing error: {e}")
        print("Tip: install a model: uv run python -m spacy download en_core_web_sm")


def main():
    """Run the infrastructure demonstration."""
    print("DocMind AI - Infrastructure Optimization Demo")
    print("=" * 50)

    demo_system_and_gpu()
    demo_spacy_optimization()

    print("\n" + "=" * 50)
    print("✓ System + optional GPU info: minimal")
    print("✓ spaCy enrichment: sentences + entities")
    print("✓ Device selection: cpu|cuda|apple|auto")
    print("✓ No extra dependencies added")
    print("✓ KISS, DRY, YAGNI principles followed")


if __name__ == "__main__":
    main()
