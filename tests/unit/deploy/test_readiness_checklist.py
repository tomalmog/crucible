"""Unit tests for deployment readiness checklist."""

from __future__ import annotations

import os
import tempfile

from core.deployment_types import DeploymentChecklist
from deploy.readiness_checklist import format_checklist, run_readiness_checklist


def test_checklist_model_exists_check() -> None:
    """Checklist should detect whether the model file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        with open(model_path, "wb") as f:
            f.write(b"fake model data")

        checklist = run_readiness_checklist(model_path, tmpdir)
        items_dict = dict(checklist.items)
        assert items_dict["model_exists"] is True

    # Non-existent model
    checklist = run_readiness_checklist("/nonexistent/m.onnx", "/tmp")
    items_dict = dict(checklist.items)
    assert items_dict["model_exists"] is False


def test_checklist_format_output() -> None:
    """Formatted checklist should contain pass/fail indicators."""
    checklist = DeploymentChecklist(
        items=(("model_exists", True), ("config_exists", False)),
        all_passed=False,
    )
    lines = format_checklist(checklist)
    assert any("PASS" in line and "model_exists" in line for line in lines)
    assert any("FAIL" in line and "config_exists" in line for line in lines)
    assert any("SOME CHECKS FAILED" in line for line in lines)


def test_checklist_all_passed_when_valid() -> None:
    """Checklist should report all_passed when all gates pass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        with open(model_path, "wb") as f:
            f.write(b"fake model data")

        # Create config and tokenizer so those checks pass
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(tmpdir, "tokenizer.json"), "w") as f:
            f.write("{}")

        checklist = run_readiness_checklist(model_path, tmpdir)
        assert checklist.all_passed is True
