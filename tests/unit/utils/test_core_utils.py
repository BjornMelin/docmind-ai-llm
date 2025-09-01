"""Unit tests for src.utils.core utilities.

Covers:
- detect_hardware branches (CUDA available/unavailable)
- validate_startup_configuration success and failure paths
- verify_rrf_configuration property checks
- managed_gpu_operation cleanup behavior
- managed_async_qdrant_client lifecycle
- async_timer decorator logging and timing
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


@pytest.mark.unit
class TestDetectHardware:
    def test_detect_hardware_cuda_available(self):
        from src.utils import core

        with (
            patch.object(core.torch.cuda, "is_available", return_value=True),
            patch.object(core.torch.cuda, "get_device_name", return_value="RTX 4090"),
            patch.object(
                core.torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(total_memory=16 * 1024**3),  # 16GB
            ),
            patch.object(core.settings, "monitoring", SimpleNamespace(bytes_to_gb_divisor=1024**3)),
        ):
            info = core.detect_hardware()
        assert info["cuda_available"] is True
        assert info["gpu_name"] == "RTX 4090"
        # Rounded to 1 decimal place
        assert info["vram_total_gb"] == 16.0

    def test_detect_hardware_no_cuda(self):
        from src.utils import core

        with patch.object(core.torch.cuda, "is_available", return_value=False):
            info = core.detect_hardware()
        assert info["cuda_available"] is False
        assert info["gpu_name"] == "Unknown"
        assert info["vram_total_gb"] is None


@pytest.mark.unit
class TestValidateStartupConfiguration:
    def _settings(self, *, strategy="hybrid", alpha=70):
        return SimpleNamespace(
            database=SimpleNamespace(qdrant_url="http://localhost:6333"),
            enable_gpu_acceleration=False,
            retrieval=SimpleNamespace(strategy=strategy, rrf_alpha=alpha, rrf_k_constant=60),
        )

    def test_validate_startup_configuration_success(self):
        from src.utils import core

        mock_client = MagicMock()
        with (
            patch("qdrant_client.QdrantClient", return_value=mock_client),
            patch.object(core.torch.cuda, "is_available", return_value=False),
        ):
            result = core.validate_startup_configuration(self._settings())

        assert result["valid"] is True
        assert any("Qdrant connection successful" in s for s in result["info"])  # connectivity checked

        # client closed
        mock_client.get_collections.assert_called_once()
        mock_client.close.assert_called_once()

    def test_validate_startup_configuration_qdrant_failure_raises(self):
        from src.utils import core

        with patch("qdrant_client.QdrantClient", side_effect=ConnectionError("down")):
            with pytest.raises(RuntimeError, match="Critical configuration errors"):
                core.validate_startup_configuration(self._settings())


@pytest.mark.unit
class TestVerifyRRFConfiguration:
    def _settings(self, alpha):
        return SimpleNamespace(retrieval=SimpleNamespace(rrf_alpha=alpha, rrf_k_constant=60))

    def test_rrf_in_range(self):
        from src.utils.core import verify_rrf_configuration

        result = verify_rrf_configuration(self._settings(70))
        assert result["weights_correct"] is True
        assert result["alpha_in_range"] is True
        # 0.7 / (0.7 + 0.3) == 0.7
        assert result["computed_hybrid_alpha"] == pytest.approx(0.7, rel=1e-6)

    def test_rrf_out_of_range(self):
        from src.utils.core import verify_rrf_configuration

        result = verify_rrf_configuration(self._settings(5))
        assert result["alpha_in_range"] is False
        assert any("RRF alpha" in issue for issue in result["issues"])  # recommendation emitted


@pytest.mark.asyncio
@pytest.mark.unit
async def test_managed_gpu_operation_calls_cuda_cleanup():
    from src.utils import core

    with (
        patch.object(core.torch.cuda, "is_available", return_value=True),
        patch.object(core.torch.cuda, "synchronize") as sync,
        patch.object(core.torch.cuda, "empty_cache") as empty,
        patch("gc.collect") as gc_collect,
    ):
        async with core.managed_gpu_operation():
            pass
    sync.assert_called_once()
    empty.assert_called_once()
    gc_collect.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_managed_async_qdrant_client_lifecycle():
    from src.utils import core

    fake_client = SimpleNamespace(close=AsyncMock())
    with patch("src.utils.core.AsyncQdrantClient", return_value=fake_client) as ctor:
        async with core.managed_async_qdrant_client("http://qdrant") as client:
            assert client is fake_client
        ctor.assert_called_once_with(url="http://qdrant")
        fake_client.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_timer_decorator_times_and_logs():
    from src.utils import core

    @core.async_timer
    async def sample_sleep():
        return "ok"

    with (
        patch("time.perf_counter", side_effect=[0.0, 0.05]),
        patch("src.utils.core.logger") as mock_logger,
    ):
        out = await sample_sleep()
    assert out == "ok"
    mock_logger.info.assert_called()  # message emitted with duration
