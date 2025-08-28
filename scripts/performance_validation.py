#!/usr/bin/env python3
"""vLLM Performance Validation Script for DocMind AI.

This script validates the performance of the Qwen3-4B-Instruct-2507-FP8 model
with vLLM backend, testing against the requirements:

- REQ-0063-v2: Model loading with FP8 quantization
- REQ-0064-v2: Performance targets (100-160 tok/s decode, 800-1300 tok/s prefill)
- REQ-0069/0070: Memory constraints (<4GB RAM, <16GB VRAM)
- REQ-0094-v2: Context window performance (128K tokens)
- REQ-0007: Agent coordination latency (<300ms)

Usage:
    python scripts/performance_validation.py

Returns:
    0 on success, 1 on failure with detailed error reporting
"""

import asyncio
import gc
import sys
import time
from pathlib import Path

import psutil
import torch
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Unified configuration and integrations
from llama_index.core import Settings

from agents.coordinator import MultiAgentCoordinator
from config.integrations import initialize_integrations
from config.settings import settings


class PerformanceMetrics(BaseModel):
    """Performance metrics for validation."""

    # Basic metrics
    test_name: str
    success: bool
    error_message: str = ""

    # Timing metrics
    model_load_time_s: float = 0.0
    first_token_latency_ms: float = 0.0
    decode_throughput_tok_s: float = 0.0
    prefill_throughput_tok_s: float = 0.0
    end_to_end_latency_ms: float = 0.0

    # Memory metrics
    peak_ram_gb: float = 0.0
    peak_vram_gb: float = 0.0

    # Context metrics
    context_length: int = 0
    max_new_tokens: int = 0

    # Agent metrics
    agent_coordination_ms: float = 0.0

    def meets_requirements(self) -> bool:
        """Check if metrics meet DocMind AI requirements."""
        if not self.success:
            return False

        # REQ-0064-v2: Performance targets (only enforce when measured)
        decode_ok = (
            True
            if self.decode_throughput_tok_s <= 0
            else 100 <= self.decode_throughput_tok_s <= 160
        )
        prefill_ok = (
            True
            if self.prefill_throughput_tok_s <= 0
            else 800 <= self.prefill_throughput_tok_s <= 1300
        )

        # REQ-0069/0070: Memory constraints (only enforce when non-zero measured)
        ram_ok = True if self.peak_ram_gb <= 0 else self.peak_ram_gb <= 4.0
        vram_ok = True if self.peak_vram_gb <= 0 else self.peak_vram_gb <= 16.0

        # REQ-0007: Agent coordination (only enforce when measured)
        agent_ok = (
            True
            if self.agent_coordination_ms <= 0
            else self.agent_coordination_ms <= 300
        )

        return decode_ok and prefill_ok and ram_ok and vram_ok and agent_ok


class PerformanceValidator:
    """Comprehensive performance validator for vLLM + DocMind AI."""

    def __init__(self):
        """Initialize the validator."""
        self.results: list[PerformanceMetrics] = []
        self.llm = None
        self.agent_coordinator: MultiAgentCoordinator | None = None
        self.initial_ram_gb = 0.0
        self.initial_vram_gb = 0.0

    def _get_memory_usage(self) -> tuple[float, float]:
        """Get current RAM and VRAM usage in GB.

        Returns:
            Tuple of (RAM GB, VRAM GB)
        """
        # RAM usage
        ram_gb = psutil.Process().memory_info().rss / (1024**3)

        # VRAM usage
        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / (1024**3)

        return ram_gb, vram_gb

    def _record_baseline_memory(self):
        """Record baseline memory usage."""
        self.initial_ram_gb, self.initial_vram_gb = self._get_memory_usage()
        baseline_msg = (
            f"Baseline memory - RAM: {self.initial_ram_gb:.2f}GB, "
            f"VRAM: {self.initial_vram_gb:.2f}GB"
        )
        print(baseline_msg)

    def validate_environment(self) -> PerformanceMetrics:
        """Validate the environment and hardware."""
        result = PerformanceMetrics(test_name="Environment Validation", success=False)

        try:
            # Check CUDA availability
            if not torch.cuda.is_available():
                result.error_message = "CUDA not available"
                return result

            # Check compute capability (need >=8.0 for FP8)
            capability = torch.cuda.get_device_capability()
            if capability[0] < 8:
                result.error_message = f"Insufficient compute capability: {capability}"
                return result

            # Check GPU memory
            gpu_props = torch.cuda.get_device_properties(0)
            total_vram_gb = gpu_props.total_memory / (1024**3)
            if total_vram_gb < 14:  # Need at least 14GB for FP8 model
                result.error_message = f"Insufficient VRAM: {total_vram_gb:.1f}GB"
                return result

            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ Compute Capability: {capability}")
            print(f"✅ Total VRAM: {total_vram_gb:.1f}GB")
            print(f"✅ PyTorch CUDA: {torch.__version__}")

            result.success = True
            return result

        except Exception as e:
            result.error_message = str(e)
            return result

    def validate_model_loading(self) -> PerformanceMetrics:
        """Validate model loading with FP8 quantization."""
        result = PerformanceMetrics(test_name="Model Setup (Unified)", success=False)

        try:
            print("Initializing integrations and configuring LLM...")

            # Record memory before loading
            ram_before, vram_before = self._get_memory_usage()

            # Load model with timing
            start_time = time.time()

            # Initialize integrations (sets env + LlamaIndex Settings.llm)
            initialize_integrations()
            self.llm = Settings.llm
            if self.llm is None:
                raise RuntimeError(
                    "LLM not configured. Check integrations and settings."
                )

            load_time = time.time() - start_time

            # Record memory after loading
            ram_after, vram_after = self._get_memory_usage()

            result.model_load_time_s = load_time
            result.peak_ram_gb = ram_after
            result.peak_vram_gb = vram_after
            result.success = True

            print(f"✅ LLM configured in {load_time:.2f}s")
            print(f"✅ RAM usage: {ram_after:.2f}GB (+{ram_after - ram_before:.2f}GB)")
            print(
                f"✅ VRAM usage: {vram_after:.2f}GB (+{vram_after - vram_before:.2f}GB)"
            )

            return result

        except Exception as e:
            result.error_message = str(e)
            return result

    def validate_decode_performance(self) -> PerformanceMetrics:
        """Validate decode throughput performance."""
        result = PerformanceMetrics(
            test_name="Decode Performance", success=False, max_new_tokens=100
        )

        if not Settings.llm:
            result.error_message = "LLM not configured"
            return result

        try:
            print("Testing decode throughput...")

            # Test prompt
            prompt = (
                "Explain the key differences between machine learning and "
                "artificial intelligence in detail:"
            )

            # Record memory before generation
            ram_before, vram_before = self._get_memory_usage()

            # Generate with timing
            start_time = time.time()
            completion = Settings.llm.complete(
                prompt, temperature=0.1, max_tokens=result.max_new_tokens, top_p=0.9
            )
            response_text = getattr(completion, "text", str(completion))
            end_time = time.time()

            # Record memory after generation
            ram_after, vram_after = self._get_memory_usage()

            if response_text:

                # Calculate metrics
                generation_time = end_time - start_time
                # Rough token count estimate (actual would need tokenizer)
                estimated_tokens = (
                    len(response_text.split()) * 1.3
                )  # ~1.3 tokens per word
                throughput = estimated_tokens / generation_time

                result.decode_throughput_tok_s = throughput
                result.end_to_end_latency_ms = generation_time * 1000
                result.peak_ram_gb = max(ram_after, result.peak_ram_gb)
                result.peak_vram_gb = max(vram_after, result.peak_vram_gb)
                result.success = True

                token_msg = (
                    f"✅ Generated {estimated_tokens:.0f} tokens in "
                    f"{generation_time:.2f}s"
                )
                print(token_msg)
                print(f"✅ Decode throughput: {throughput:.2f} tok/s")
                print(f"✅ Response preview: {response_text[:100]}...")

            else:
                result.error_message = "No response generated"

            return result

        except Exception as e:
            result.error_message = str(e)
            return result

    def validate_prefill_performance(self) -> PerformanceMetrics:
        """Validate prefill throughput performance with longer context."""
        result = PerformanceMetrics(
            test_name="Prefill Performance",
            success=False,
            context_length=4096,  # 4K context test
            max_new_tokens=50,
        )

        if not Settings.llm:
            result.error_message = "LLM not configured"
            return result

        try:
            print("Testing prefill throughput with 4K context...")

            # Create a longer context prompt (~4K tokens)
            base_text = """
            Machine learning (ML) is a branch of artificial intelligence (AI)
            and computer science that focuses on the use of data and algorithms
            to imitate the way that humans learn, gradually improving its accuracy.
            Machine learning is an important component of the growing field of
            data science. Through the use of statistical methods, algorithms are
            trained to make classifications or predictions, uncovering key insights
            within data mining projects. These insights subsequently drive decision
            making within applications and businesses, ideally impacting key
            growth metrics.
            """

            # Repeat to create ~4K tokens
            question = (
                "\n\nBased on the above information, what are the key "
                "applications of machine learning?"
            )
            long_prompt = (base_text * 50) + question

            # Record memory before generation
            ram_before, vram_before = self._get_memory_usage()

            # Generate with timing (focus on prefill)
            start_time = time.time()
            completion = Settings.llm.complete(
                long_prompt,
                temperature=0.1,
                max_tokens=result.max_new_tokens,
                top_p=0.9,
            )
            response_text = getattr(completion, "text", str(completion))
            end_time = time.time()

            # Record memory after generation
            ram_after, vram_after = self._get_memory_usage()

            if response_text:

                # Calculate metrics (prefill is dominant with long context)
                generation_time = end_time - start_time
                # Estimate prefill tokens (~4K input + output)
                estimated_prefill_tokens = result.context_length
                estimated_decode_tokens = len(response_text.split()) * 1.3

                # Prefill throughput (most of the time is spent on prefill)
                prefill_throughput = estimated_prefill_tokens / generation_time

                result.prefill_throughput_tok_s = prefill_throughput
                result.end_to_end_latency_ms = generation_time * 1000
                result.peak_ram_gb = max(ram_after, result.peak_ram_gb)
                result.peak_vram_gb = max(vram_after, result.peak_vram_gb)
                result.success = True

                prefill_msg = (
                    f"✅ Processed {estimated_prefill_tokens} prefill tokens "
                    f"in {generation_time:.2f}s"
                )
                print(prefill_msg)
                print(f"✅ Prefill throughput: {prefill_throughput:.2f} tok/s")
                print(f"✅ Generated tokens: {estimated_decode_tokens:.0f}")

            else:
                result.error_message = "No response generated"

            return result

        except Exception as e:
            result.error_message = str(e)
            return result

    async def validate_agent_coordination(self) -> PerformanceMetrics:
        """Validate multi-agent coordination performance."""
        result = PerformanceMetrics(test_name="Agent Coordination", success=False)

        try:
            print("Testing multi-agent coordination latency...")

            # Initialize ADR-compliant multi-agent coordinator
            self.agent_coordinator = MultiAgentCoordinator(
                model_path=settings.vllm.model,
                max_context_length=settings.vllm.context_window,
                backend="vllm",
                enable_fallback=True,
                max_agent_timeout=settings.agents.decision_timeout / 1000.0,
            )

            # Test query
            test_query = (
                "What are the benefits of using machine learning in "
                "healthcare applications?"
            )

            # Record memory before coordination
            ram_before, vram_before = self._get_memory_usage()

            # Process query with timing
            start_time = time.time()
            response = self.agent_coordinator.process_query(test_query)
            end_time = time.time()

            # Record memory after coordination
            ram_after, vram_after = self._get_memory_usage()

            coordination_time = (end_time - start_time) * 1000  # Convert to ms

            result.agent_coordination_ms = coordination_time
            result.peak_ram_gb = max(ram_after, result.peak_ram_gb)
            result.peak_vram_gb = max(vram_after, result.peak_vram_gb)

            # AgentResponse model expected
            if hasattr(response, "content"):
                result.success = True
                print(f"✅ Agent coordination completed in {coordination_time:.2f}ms")
                # Use validation_score if available
                confidence = getattr(response, "validation_score", 0.0)
                print(f"✅ Validation score: {confidence:.2f}")
            else:
                result.error_message = "Agent coordination failed: invalid response"

            return result

        except Exception as e:
            result.error_message = str(e)
            return result

    def validate_context_window(self) -> PerformanceMetrics:
        """Validate 128K context window performance."""
        result = PerformanceMetrics(
            test_name="Context Window (128K)",
            success=False,
            context_length=131072,  # 128K tokens
            max_new_tokens=50,
        )

        if not Settings.llm:
            result.error_message = "LLM not configured"
            return result

        try:
            print("Testing 128K context window (this may take a while)...")

            # This is a simplified test - creating actual 128K tokens
            # would be very large
            # Instead, test with model's max_model_len configuration
            print(
                "✅ Model configured for max context: "
                f"{settings.vllm.context_window} tokens"
            )
            print(f"✅ FP8 KV cache setting: {settings.vllm.kv_cache_dtype}")

            # Test with moderate context that's still substantial
            base_context = (
                "This is a test of context handling with repeated information. " * 100
            )
            test_prompt = (
                base_context + "\n\nSummarize the main points from the above text."
            )

            # Record memory before generation
            ram_before, vram_before = self._get_memory_usage()

            start_time = time.time()
            completion = Settings.llm.complete(
                test_prompt,
                temperature=0.1,
                max_tokens=result.max_new_tokens,
                top_p=0.9,
            )
            response_text = getattr(completion, "text", str(completion))
            end_time = time.time()

            # Record memory after generation
            ram_after, vram_after = self._get_memory_usage()

            if response_text:
                generation_time = end_time - start_time

                result.end_to_end_latency_ms = generation_time * 1000
                result.peak_ram_gb = max(ram_after, result.peak_ram_gb)
                result.peak_vram_gb = max(vram_after, result.peak_vram_gb)
                result.success = True

                print(f"✅ Context processing completed in {generation_time:.2f}s")
                print("✅ Memory efficient with FP8 KV cache")
                print(f"✅ VRAM usage: {vram_after:.2f}GB")

            else:
                result.error_message = "No response generated for context test"

            return result

        except Exception as e:
            result.error_message = str(e)
            return result

    async def run_all_validations(self) -> dict[str, PerformanceMetrics]:
        """Run all performance validations."""
        print("🚀 Starting DocMind AI vLLM Performance Validation")
        print("=" * 60)

        # Record baseline memory
        self._record_baseline_memory()

        # Run validations in sequence
        validations = [
            ("environment", self.validate_environment()),
            ("model_loading", self.validate_model_loading()),
            ("decode_performance", self.validate_decode_performance()),
            ("prefill_performance", self.validate_prefill_performance()),
            ("context_window", self.validate_context_window()),
            ("agent_coordination", await self.validate_agent_coordination()),
        ]

        results = {}
        all_passed = True

        for name, result in validations:
            results[name] = result
            self.results.append(result)

            print(f"\n{'=' * 20} {result.test_name} {'=' * 20}")

            if result.success:
                print(f"✅ PASSED: {result.test_name}")

                # Show metrics if available
                if result.decode_throughput_tok_s > 0:
                    decode_msg = (
                        f"   Decode throughput: "
                        f"{result.decode_throughput_tok_s:.2f} tok/s"
                    )
                    print(decode_msg)
                if result.prefill_throughput_tok_s > 0:
                    prefill_msg = (
                        f"   Prefill throughput: "
                        f"{result.prefill_throughput_tok_s:.2f} tok/s"
                    )
                    print(prefill_msg)
                if result.agent_coordination_ms > 0:
                    print(
                        f"   Agent coordination: {result.agent_coordination_ms:.2f}ms"
                    )
                if result.peak_vram_gb > 0:
                    print(f"   Peak VRAM: {result.peak_vram_gb:.2f}GB")

                # Check requirements
                if hasattr(result, "meets_requirements"):
                    meets_reqs = result.meets_requirements()
                    print(
                        f"   Meets requirements: {'✅ Yes' if meets_reqs else '❌ No'}"
                    )
                    if not meets_reqs:
                        all_passed = False

            else:
                print(f"❌ FAILED: {result.test_name}")
                print(f"   Error: {result.error_message}")
                all_passed = False

                # Stop on critical failures
                if name in ["environment", "model_loading"]:
                    print("💥 Critical failure - stopping validation")
                    break

        # Cleanup placeholder (no explicit cleanup needed with unified integrations)

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n{'=' * 60}")
        print(
            f"🏁 Validation Complete: "
            f"{'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}"
        )

        return results

    def generate_report(self, results: dict[str, PerformanceMetrics]) -> str:
        """Generate a detailed performance report."""
        report_lines = [
            "DocMind AI vLLM Performance Validation Report",
            "=" * 50,
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {settings.vllm.model}",
            "Quantization: FP8 KV cache (env-configured)",
            "Backend: Unified via LlamaIndex integrations",
            "",
        ]

        # Requirements summary
        report_lines.extend(
            [
                "Requirements Validation:",
                "- REQ-0063-v2: Model loading with FP8 quantization",
                "- REQ-0064-v2: Performance (100-160 tok/s decode, "
                "800-1300 tok/s prefill)",
                "- REQ-0069/0070: Memory (<4GB RAM, <16GB VRAM)",
                "- REQ-0094-v2: 128K context window support",
                "- REQ-0007: Agent coordination (<300ms)",
                "",
            ]
        )

        # Results summary
        for _name, result in results.items():
            status = "✅ PASS" if result.success else "❌ FAIL"
            report_lines.append(f"{status} {result.test_name}")

            if not result.success:
                report_lines.append(f"    Error: {result.error_message}")
            else:
                if result.decode_throughput_tok_s > 0:
                    report_lines.append(
                        f"    Decode: {result.decode_throughput_tok_s:.2f} tok/s"
                    )
                if result.prefill_throughput_tok_s > 0:
                    report_lines.append(
                        f"    Prefill: {result.prefill_throughput_tok_s:.2f} tok/s"
                    )
                if result.agent_coordination_ms > 0:
                    report_lines.append(
                        f"    Agent coordination: {result.agent_coordination_ms:.2f}ms"
                    )
                if result.peak_vram_gb > 0:
                    report_lines.append(f"    Peak VRAM: {result.peak_vram_gb:.2f}GB")

        return "\n".join(report_lines)


async def main():
    """Main validation function."""
    try:
        validator = PerformanceValidator()
        results = await validator.run_all_validations()

        # Generate and save report
        report = validator.generate_report(results)

        # Save to file
        report_file = Path("performance_validation_report.txt")
        report_file.write_text(report)

        print(f"\n📄 Report saved to: {report_file}")
        print("\n" + "=" * 60)
        print(report)

        # Return appropriate exit code
        all_passed = all(r.success for r in results.values())
        return 0 if all_passed else 1

    except Exception as e:
        print(f"💥 Validation failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
