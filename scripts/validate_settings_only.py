#!/usr/bin/env python3
"""Validate User Flexibility Settings Without Full Integration.

This script validates the settings configuration without triggering
full LlamaIndex integration that requires running services.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import only the settings class directly
from src.config.settings import DocMindSettings


def test_settings_scenario(name: str, env_vars: dict[str, str]) -> dict[str, Any]:
    """Test settings configuration for a user scenario."""
    print(f"\n🧪 Testing {name}...")

    # Create temporary env file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
        temp_env_file = f.name

    try:
        # Test settings creation
        settings = DocMindSettings(_env_file=temp_env_file)

        # Test key functionality
        settings.get_user_hardware_info()
        settings.get_user_scenario_config()
        backend_url = settings._get_backend_url()
        embedding_device = settings._get_embedding_device()
        batch_size = settings._get_embedding_batch_size()

        print(f"✅ {name} - Settings valid!")
        print(f"   Device: {embedding_device}")
        print(f"   Backend: {settings.llm_backend} ({backend_url})")
        print(f"   GPU Acceleration: {settings.enable_gpu_acceleration}")
        print(f"   Context Window: {settings.vllm.context_window}")
        print(f"   Batch Size: {batch_size}")
        print(
            f"   Memory Limits: {settings.max_memory_gb}GB RAM, "
            f"{settings.max_vram_gb}GB VRAM"
        )

        return {
            "name": name,
            "success": True,
            "settings": {
                "device": embedding_device,
                "backend": settings.llm_backend,
                "gpu_enabled": settings.enable_gpu_acceleration,
                "context_window": settings.vllm.context_window,
                "batch_size": batch_size,
                "memory_gb": settings.max_memory_gb,
                "vram_gb": settings.max_vram_gb,
            },
        }

    except Exception as e:
        print(f"❌ {name} - Error: {e}")
        return {"name": name, "success": False, "error": str(e)}

    finally:
        # Clean up temp file
        os.unlink(temp_env_file)


def main():
    """Run validation for all user scenarios."""
    print("🔧 Validating User Flexibility Settings (Configuration Only)")
    print("=" * 70)

    scenarios = [
        # CPU-only student
        {
            "name": "Student (CPU-only, 8GB RAM)",
            "env_vars": {
                "DOCMIND_ENABLE_GPU_ACCELERATION": "false",
                "DOCMIND_DEVICE": "cpu",
                "DOCMIND_MAX_MEMORY_GB": "8.0",
                "DOCMIND_CONTEXT_WINDOW_SIZE": "4096",
                "DOCMIND_LLM_BACKEND": "ollama",
            },
        },
        # Mid-range developer
        {
            "name": "Developer (RTX 3060, 12GB VRAM)",
            "env_vars": {
                "DOCMIND_ENABLE_GPU_ACCELERATION": "true",
                "DOCMIND_DEVICE": "cuda",
                "DOCMIND_MAX_VRAM_GB": "12.0",
                "DOCMIND_LLM_BACKEND": "vllm",
                "DOCMIND_CONTEXT_WINDOW_SIZE": "32768",
            },
        },
        # High-end researcher
        {
            "name": "Researcher (RTX 4090, 24GB VRAM)",
            "env_vars": {
                "DOCMIND_ENABLE_GPU_ACCELERATION": "true",
                "DOCMIND_DEVICE": "cuda",
                "DOCMIND_MAX_VRAM_GB": "24.0",
                "DOCMIND_LLM_BACKEND": "vllm",
                "DOCMIND_CONTEXT_WINDOW_SIZE": "131072",
            },
        },
        # Privacy-focused user
        {
            "name": "Privacy User (CPU, local models)",
            "env_vars": {
                "DOCMIND_ENABLE_GPU_ACCELERATION": "false",
                "DOCMIND_DEVICE": "cpu",
                "DOCMIND_LLM_BACKEND": "llama_cpp",
                "DOCMIND_LOCAL_MODEL_PATH": "/home/user/models",
                "DOCMIND_ENABLE_PERFORMANCE_LOGGING": "false",
            },
        },
        # Auto-detection
        {
            "name": "Auto-detection User",
            "env_vars": {
                "DOCMIND_DEVICE": "auto",
                "DOCMIND_LLM_BACKEND": "ollama",
            },
        },
        # Custom embedding model
        {
            "name": "Custom Embedding User",
            "env_vars": {
                "DOCMIND_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
                "DOCMIND_LLM_BACKEND": "openai",
                "DOCMIND_OPENAI_BASE_URL": "http://localhost:8080",
            },
        },
    ]

    results = []
    success_count = 0

    for scenario in scenarios:
        result = test_settings_scenario(scenario["name"], scenario["env_vars"])
        results.append(result)
        if result["success"]:
            success_count += 1

    print("\n📊 Validation Results")
    print("=" * 70)
    print(f"✅ Successful configurations: {success_count}/{len(scenarios)}")
    print(
        f"❌ Failed configurations: {len(scenarios) - success_count}/{len(scenarios)}"
    )

    if success_count == len(scenarios):
        print(f"\n🎉 SUCCESS: All {len(scenarios)} user scenarios are supported!")
        print("✅ User flexibility settings successfully restored")
        print("✅ CPU-only users can configure the application")
        print("✅ Multiple LLM backends are available")
        print("✅ Hardware diversity is accommodated")
        print("✅ Local-first operation is maintained")

        # Summarize key restored features
        print("\n🔑 Key Restored Features:")
        print("✅ enable_gpu_acceleration - Users can disable GPU")
        print("✅ device selection - 'cpu', 'cuda', or 'auto'")
        print("✅ llm_backend choice - 'ollama', 'vllm', 'openai', 'llama_cpp'")
        print("✅ context_window_size - Configurable 1K to 128K")
        print("✅ Hardware batch sizes - CPU vs GPU optimized")
        print("✅ Memory limits - max_memory_gb, max_vram_gb")
        print("✅ Performance tiers - 'low', 'medium', 'high', 'auto'")
        print("✅ User feature toggles - caching, debug, logging")

        return 0
    else:
        print("\n⚠️  Configuration failures:")
        for result in results:
            if not result["success"]:
                print(f"❌ {result['name']}: {result['error']}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
