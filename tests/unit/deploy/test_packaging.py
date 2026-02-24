"""Unit tests for deployment packaging."""

from __future__ import annotations

import hashlib
import os
import tempfile

import pytest

from core.errors import ForgeDeployError
from deploy.packaging import (
    PackageConfig,
    _compute_file_checksum,
    build_deployment_package,
)


def test_compute_file_checksum_correct() -> None:
    """Checksum should match SHA256 of file contents."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"test content for checksum")
        tmp.flush()
        tmp_path = tmp.name

    try:
        result = _compute_file_checksum(tmp_path)
        expected = hashlib.sha256(b"test content for checksum").hexdigest()
        assert result == expected
    finally:
        os.unlink(tmp_path)


def test_build_deployment_package_creates_manifest() -> None:
    """Package build should create manifest.json in output dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        with open(model_path, "wb") as f:
            f.write(b"fake model data")

        output_dir = os.path.join(tmpdir, "package")
        config = PackageConfig(
            model_path=model_path,
            output_dir=output_dir,
        )
        pkg = build_deployment_package(config)

        assert os.path.isfile(os.path.join(output_dir, "manifest.json"))
        assert pkg.package_path == output_dir
        assert pkg.checksum != ""
        assert pkg.config_path is None
        assert pkg.tokenizer_path is None


def test_build_deployment_package_rejects_missing_model() -> None:
    """Package build should raise ForgeDeployError for missing model."""
    config = PackageConfig(
        model_path="/nonexistent/model.onnx",
        output_dir="/tmp/out",
    )
    with pytest.raises(ForgeDeployError, match="Model file not found"):
        build_deployment_package(config)
