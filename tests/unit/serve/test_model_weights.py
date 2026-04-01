"""Unit tests for model weight loading helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import CrucibleDependencyError, CrucibleServeError
from serve.model_weights import load_initial_weights


class _FakeTorch:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def load(self, path: str, map_location: object, **kwargs: object) -> object:
        _ = (path, map_location, kwargs)
        return self._payload

    def tensor(self, value: object, device: object) -> object:
        _ = device
        return value


class _FakeModel:
    def __init__(self) -> None:
        self.loaded_state: object | None = None

    def load_state_dict(self, state_dict: object) -> None:
        self.loaded_state = state_dict


class _FakeInitializer:
    def __init__(self, name: str, array: object) -> None:
        self.name = name
        self.array = array


class _FakeGraph:
    def __init__(self, initializers: list[_FakeInitializer]) -> None:
        self.initializer = initializers


class _FakeOnnxModel:
    def __init__(self, initializers: list[_FakeInitializer]) -> None:
        self.graph = _FakeGraph(initializers)


class _FakeNumpyHelper:
    @staticmethod
    def to_array(initializer: _FakeInitializer) -> object:
        return initializer.array


class _FakeOnnxModule:
    def __init__(self, initializers: list[_FakeInitializer]) -> None:
        self.numpy_helper = _FakeNumpyHelper()
        self._model = _FakeOnnxModel(initializers)

    def load(self, path: str) -> _FakeOnnxModel:
        _ = path
        return self._model


def test_load_initial_weights_raises_for_missing_path(tmp_path: Path) -> None:
    """Missing checkpoint path should raise a clear serve error."""
    with pytest.raises(CrucibleServeError):
        load_initial_weights(
            torch_module=_FakeTorch(payload={}),
            model=_FakeModel(),
            initial_weights_path=str(tmp_path / "missing.pt"),
            device="cpu",
        )

    assert True


def test_load_initial_weights_accepts_raw_state_dict(tmp_path: Path) -> None:
    """Raw state_dict payload should be forwarded to model loader."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    model = _FakeModel()

    load_initial_weights(
        torch_module=_FakeTorch(payload={"linear.weight": object()}),
        model=model,
        initial_weights_path=str(model_path),
        device="cpu",
    )

    assert isinstance(model.loaded_state, dict)


def test_load_initial_weights_accepts_nested_model_state_dict(tmp_path: Path) -> None:
    """Checkpoint payload with model_state_dict should be unpacked."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")
    model = _FakeModel()

    load_initial_weights(
        torch_module=_FakeTorch(payload={"model_state_dict": {"linear.bias": object()}}),
        model=model,
        initial_weights_path=str(model_path),
        device="cpu",
    )

    assert isinstance(model.loaded_state, dict)


def test_load_initial_weights_rejects_invalid_payload(tmp_path: Path) -> None:
    """Invalid checkpoint payload should raise serve error."""
    model_path = tmp_path / "model.pt"
    model_path.write_text("placeholder", encoding="utf-8")

    with pytest.raises(CrucibleServeError):
        load_initial_weights(
            torch_module=_FakeTorch(payload=["not", "a", "mapping"]),
            model=_FakeModel(),
            initial_weights_path=str(model_path),
            device="cpu",
        )

    assert True


def test_load_initial_weights_reads_onnx_initializers(tmp_path: Path, monkeypatch) -> None:
    """ONNX initializers should be converted and applied as model state."""
    model_path = tmp_path / "model.onnx"
    model_path.write_text("placeholder", encoding="utf-8")
    model = _FakeModel()
    onnx_module = _FakeOnnxModule(
        initializers=[
            _FakeInitializer("linear.weight", [[1.0, 2.0], [3.0, 4.0]]),
            _FakeInitializer("linear.bias", [0.1, 0.2]),
        ]
    )
    monkeypatch.setattr("serve.model_weights._import_onnx_optional", lambda: onnx_module)

    load_initial_weights(
        torch_module=_FakeTorch(payload={}),
        model=model,
        initial_weights_path=str(model_path),
        device="cpu",
    )

    assert isinstance(model.loaded_state, dict) and "linear.weight" in model.loaded_state


def test_load_initial_weights_onnx_requires_onnx_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """ONNX loading should raise dependency error when onnx is unavailable."""
    model_path = tmp_path / "model.onnx"
    model_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "serve.model_weights._import_onnx_optional",
        lambda: (_ for _ in ()).throw(CrucibleDependencyError("missing onnx")),
    )

    with pytest.raises(CrucibleDependencyError):
        load_initial_weights(
            torch_module=_FakeTorch(payload={}),
            model=_FakeModel(),
            initial_weights_path=str(model_path),
            device="cpu",
        )

    assert True
