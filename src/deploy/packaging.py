"""Deployment package builder.

This module bundles model weights, config, tokenizer, and safety report
into a self-contained deployment package with a manifest and checksum.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from core.deployment_types import DeploymentPackage
from core.errors import ForgeDeployError


@dataclass(frozen=True)
class PackageConfig:
    """Configuration for building a deployment package.

    Attributes:
        model_path: Path to the model file to package.
        output_dir: Directory where the package is created.
        config_path: Optional path to a model config file.
        tokenizer_path: Optional path to a tokenizer file.
        safety_report_path: Optional path to a safety report.
    """

    model_path: str
    output_dir: str
    config_path: str | None = None
    tokenizer_path: str | None = None
    safety_report_path: str | None = None


def build_deployment_package(
    config: PackageConfig,
) -> DeploymentPackage:
    """Build a deployment package from the given configuration.

    Copies model and optional artifacts to output_dir, computes
    a SHA256 checksum, and writes a manifest.json.

    Args:
        config: Package build configuration.

    Returns:
        DeploymentPackage describing the built artifact.

    Raises:
        ForgeDeployError: If required files are missing.
    """
    if not os.path.isfile(config.model_path):
        raise ForgeDeployError(
            f"Model file not found: {config.model_path}"
        )

    pkg_dir = Path(config.output_dir)
    pkg_dir.mkdir(parents=True, exist_ok=True)

    dst_model = _copy_to_package(config.model_path, pkg_dir)
    dst_config = _copy_optional(config.config_path, pkg_dir)
    dst_tokenizer = _copy_optional(config.tokenizer_path, pkg_dir)
    dst_safety = _copy_optional(config.safety_report_path, pkg_dir)

    checksum = _compute_file_checksum(dst_model)

    manifest = _build_manifest(
        dst_model, dst_config, dst_tokenizer, dst_safety, checksum,
    )
    manifest_path = str(pkg_dir / "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return DeploymentPackage(
        package_path=str(pkg_dir),
        model_path=dst_model,
        config_path=dst_config,
        tokenizer_path=dst_tokenizer,
        safety_report_path=dst_safety,
        checksum=checksum,
    )


def _copy_to_package(src: str, pkg_dir: Path) -> str:
    """Copy a file into the package directory.

    Args:
        src: Source file path.
        pkg_dir: Package directory.

    Returns:
        Destination file path.
    """
    dst = str(pkg_dir / Path(src).name)
    shutil.copy2(src, dst)
    return dst


def _copy_optional(src: str | None, pkg_dir: Path) -> str | None:
    """Copy an optional file into the package directory.

    Args:
        src: Source file path or None.
        pkg_dir: Package directory.

    Returns:
        Destination path or None if src was None.

    Raises:
        ForgeDeployError: If the file does not exist.
    """
    if src is None:
        return None
    if not os.path.isfile(src):
        raise ForgeDeployError(f"File not found: {src}")
    return _copy_to_package(src, pkg_dir)


def _compute_file_checksum(file_path: str) -> str:
    """Compute SHA256 checksum of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hex-encoded SHA256 digest.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _build_manifest(
    model: str,
    config: str | None,
    tokenizer: str | None,
    safety: str | None,
    checksum: str,
) -> dict[str, str | None]:
    """Build the manifest dictionary for JSON serialization.

    Args:
        model: Model file path in the package.
        config: Config file path or None.
        tokenizer: Tokenizer file path or None.
        safety: Safety report path or None.
        checksum: SHA256 checksum of the model.

    Returns:
        Manifest dictionary.
    """
    return {
        "model_path": model,
        "config_path": config,
        "tokenizer_path": tokenizer,
        "safety_report_path": safety,
        "model_checksum_sha256": checksum,
    }
