"""Tests for TPU device detection and XLA mesh initialization."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.errors import CrucibleDependencyError, CrucibleDistributedError
from serve.tpu_setup import (
    TpuDeviceInfo,
    detect_tpu_availability,
    get_tpu_device_info,
    import_xla,
    init_xla_mesh,
    resolve_tpu_device,
)


def _make_xla_module(device: object = "xla:0", world_size: int = 4) -> MagicMock:
    """Build a mock torch_xla.core.xla_model module."""
    xla = MagicMock()
    xla.xla_device.return_value = device
    xla.xrt_world_size.return_value = world_size
    xla.runtime = SimpleNamespace(initialize_cache=MagicMock())
    return xla


class TestDetectTpuAvailability:
    """Tests for detect_tpu_availability()."""

    def test_returns_false_when_xla_not_installed(self) -> None:
        with patch("serve.tpu_setup._try_import_xla", return_value=None):
            assert detect_tpu_availability() is False

    def test_returns_true_when_device_available(self) -> None:
        mock_xla = _make_xla_module()
        with patch("serve.tpu_setup._try_import_xla", return_value=mock_xla):
            assert detect_tpu_availability() is True

    def test_returns_false_when_device_is_none(self) -> None:
        mock_xla = _make_xla_module(device=None)
        with patch("serve.tpu_setup._try_import_xla", return_value=mock_xla):
            assert detect_tpu_availability() is False

    def test_returns_false_on_xla_device_exception(self) -> None:
        mock_xla = MagicMock()
        mock_xla.xla_device.side_effect = RuntimeError("no TPU")
        with patch("serve.tpu_setup._try_import_xla", return_value=mock_xla):
            assert detect_tpu_availability() is False


class TestResolveTpuDevice:
    """Tests for resolve_tpu_device()."""

    def test_returns_device(self) -> None:
        xla = _make_xla_module(device="xla:0")
        assert resolve_tpu_device(xla) == "xla:0"

    def test_raises_on_none_device(self) -> None:
        xla = _make_xla_module(device=None)
        with pytest.raises(CrucibleDistributedError, match="No TPU device"):
            resolve_tpu_device(xla)

    def test_raises_on_exception(self) -> None:
        xla = MagicMock()
        xla.xla_device.side_effect = RuntimeError("fail")
        with pytest.raises(CrucibleDistributedError, match="Failed to resolve TPU"):
            resolve_tpu_device(xla)


class TestGetTpuDeviceInfo:
    """Tests for get_tpu_device_info()."""

    def test_returns_correct_info(self) -> None:
        xla = _make_xla_module(device="xla:0", world_size=8)
        info = get_tpu_device_info(xla)
        assert isinstance(info, TpuDeviceInfo)
        assert info.device_type == "tpu"
        assert info.num_devices == 8
        assert info.xla_device == "xla:0"

    def test_defaults_to_one_device_on_world_size_error(self) -> None:
        xla = _make_xla_module(device="xla:0")
        xla.xrt_world_size.side_effect = RuntimeError("not init")
        info = get_tpu_device_info(xla)
        assert info.num_devices == 1


class TestInitXlaMesh:
    """Tests for init_xla_mesh()."""

    def test_calls_initialize_cache(self) -> None:
        xla = _make_xla_module()
        init_xla_mesh(xla)
        xla.runtime.initialize_cache.assert_called_once_with(
            "/tmp/xla_cache", readonly=False,
        )

    def test_raises_on_failure(self) -> None:
        xla = MagicMock()
        xla.runtime.initialize_cache.side_effect = RuntimeError("fail")
        with pytest.raises(CrucibleDistributedError, match="Failed to initialize XLA mesh"):
            init_xla_mesh(xla)

    def test_skips_when_no_runtime(self) -> None:
        xla = MagicMock(spec=[])
        init_xla_mesh(xla)


class TestImportXla:
    """Tests for import_xla()."""

    def test_raises_when_not_available(self) -> None:
        with patch("serve.tpu_setup._try_import_xla", return_value=None):
            with pytest.raises(CrucibleDependencyError, match="torch_xla"):
                import_xla()

    def test_returns_module_when_available(self) -> None:
        mock_xla = _make_xla_module()
        with patch("serve.tpu_setup._try_import_xla", return_value=mock_xla):
            assert import_xla() is mock_xla
