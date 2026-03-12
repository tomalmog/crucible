"""Unit tests for model deletion and safe path validation."""

from __future__ import annotations

from pathlib import Path

from store.model_registry import ModelRegistry, _safe_delete_local_path


def test_delete_model_with_local_files(tmp_path: Path) -> None:
    """delete_model with delete_local=True should remove safe files."""
    pulled = tmp_path / "pulled-models" / "my-model"
    pulled.mkdir(parents=True)
    (pulled / "weights.pt").write_text("fake")

    registry = ModelRegistry(tmp_path)
    registry.register_model("my-model", str(pulled))

    result = registry.delete_model("my-model", delete_local=True)
    assert len(result.local_paths_deleted) == 1
    assert not pulled.exists()


def test_safe_delete_under_pulled_models(tmp_path: Path) -> None:
    """_safe_delete_local_path should delete files under pulled-models."""
    target = tmp_path / "pulled-models" / "test" / "model.pt"
    target.parent.mkdir(parents=True)
    target.write_text("weights")

    ok, reason = _safe_delete_local_path(tmp_path, str(target))
    assert ok is True
    assert reason == ""
    assert not target.exists()


def test_safe_delete_under_runs(tmp_path: Path) -> None:
    """_safe_delete_local_path should delete files under runs."""
    target = tmp_path / "runs" / "run-001" / "checkpoint.pt"
    target.parent.mkdir(parents=True)
    target.write_text("checkpoint")

    ok, reason = _safe_delete_local_path(tmp_path, str(target))
    assert ok is True
    assert not target.exists()


def test_safe_delete_rejects_outside_path(tmp_path: Path) -> None:
    """_safe_delete_local_path should refuse paths outside safe dirs."""
    target = tmp_path / "other" / "model.pt"
    target.parent.mkdir(parents=True)
    target.write_text("secret")

    ok, reason = _safe_delete_local_path(tmp_path, str(target))
    assert ok is False
    assert "Skipped" in reason
    assert target.exists()


def test_safe_delete_rejects_traversal(tmp_path: Path) -> None:
    """_safe_delete_local_path should reject path traversal attacks."""
    target = tmp_path / "pulled-models" / ".." / "outside.pt"
    target_real = (tmp_path / "outside.pt")
    target_real.write_text("nope")

    ok, reason = _safe_delete_local_path(tmp_path, str(target))
    assert ok is False
    assert target_real.exists()


def test_safe_delete_handles_missing_path(tmp_path: Path) -> None:
    """_safe_delete_local_path should handle non-existent paths gracefully."""
    missing = tmp_path / "pulled-models" / "gone" / "model.pt"
    ok, reason = _safe_delete_local_path(tmp_path, str(missing))
    assert ok is False
    assert "does not exist" in reason


def test_safe_delete_directory(tmp_path: Path) -> None:
    """_safe_delete_local_path should delete directories with shutil.rmtree."""
    target_dir = tmp_path / "pulled-models" / "my-model"
    target_dir.mkdir(parents=True)
    (target_dir / "config.json").write_text("{}")
    (target_dir / "weights.pt").write_text("fake")

    ok, reason = _safe_delete_local_path(tmp_path, str(target_dir))
    assert ok is True
    assert not target_dir.exists()
