"""Unit tests for model deletion and safe path validation."""

from __future__ import annotations

from pathlib import Path

from store.model_registry import ModelRegistry, _safe_delete_local_path


def test_delete_model_removes_all_versions(tmp_path: Path) -> None:
    """delete_model should remove all versions and the group."""
    registry = ModelRegistry(tmp_path)
    registry.register_model("my-model", "/tmp/m1.pt")
    registry.register_model("my-model", "/tmp/m2.pt")
    assert len(registry.list_versions_for_model("my-model")) == 2

    result = registry.delete_model("my-model")
    assert result.versions_removed == 2
    assert registry.list_versions_for_model("my-model") == ()
    assert "my-model" not in registry.list_model_names()


def test_delete_model_does_not_affect_other_models(tmp_path: Path) -> None:
    """Deleting one model should leave others intact."""
    registry = ModelRegistry(tmp_path)
    registry.register_model("keep-me", "/tmp/keep.pt")
    registry.register_model("delete-me", "/tmp/del.pt")

    registry.delete_model("delete-me")
    assert "keep-me" in registry.list_model_names()
    assert "delete-me" not in registry.list_model_names()
    assert len(registry.list_versions_for_model("keep-me")) == 1


def test_delete_version_removes_single_version(tmp_path: Path) -> None:
    """delete_version should remove only the specified version."""
    registry = ModelRegistry(tmp_path)
    v1 = registry.register_model("my-model", "/tmp/m1.pt")
    v2 = registry.register_model("my-model", "/tmp/m2.pt")

    result = registry.delete_version("my-model", v1.version_id)
    assert result.versions_removed == 1
    remaining = registry.list_versions_for_model("my-model")
    assert len(remaining) == 1
    assert remaining[0].version_id == v2.version_id


def test_delete_version_updates_active_if_needed(tmp_path: Path) -> None:
    """Deleting the active version should promote the last remaining."""
    registry = ModelRegistry(tmp_path)
    v1 = registry.register_model("my-model", "/tmp/m1.pt")
    v2 = registry.register_model("my-model", "/tmp/m2.pt")
    assert registry.get_active_version_id_for_model("my-model") == v1.version_id

    registry.delete_version("my-model", v1.version_id)
    assert registry.get_active_version_id_for_model("my-model") == v2.version_id


def test_delete_last_version_removes_group(tmp_path: Path) -> None:
    """Deleting the only version should also remove the group and index entry."""
    registry = ModelRegistry(tmp_path)
    v1 = registry.register_model("my-model", "/tmp/m1.pt")

    registry.delete_version("my-model", v1.version_id)
    assert "my-model" not in registry.list_model_names()
    assert registry.list_versions_for_model("my-model") == ()


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


def test_delete_model_skips_unsafe_paths(tmp_path: Path) -> None:
    """Files outside safe zone should be skipped, not deleted."""
    unsafe_file = tmp_path / "outside" / "model.pt"
    unsafe_file.parent.mkdir(parents=True)
    unsafe_file.write_text("do not delete me")

    registry = ModelRegistry(tmp_path)
    registry.register_model("my-model", str(unsafe_file))

    result = registry.delete_model("my-model", delete_local=True)
    assert len(result.local_paths_skipped) == 1
    assert unsafe_file.exists()
    assert result.versions_removed == 1


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
