"""Unit tests for agent tarball bundler."""

from __future__ import annotations

import tarfile
from pathlib import Path
from unittest.mock import patch

from serve.agent_bundler import (
    _compute_src_hash,
    _should_include,
    build_agent_tarball,
)


def _create_fake_src(root: Path) -> Path:
    """Build a minimal src directory tree for bundling tests."""
    src = root / "src"
    for module, files in (
        ("core", ("__init__.py", "config.py")),
        ("serve", ("__init__.py", "training.py", "ssh_connection.py")),
        ("store", ("__init__.py",)),
    ):
        module_dir = src / module
        module_dir.mkdir(parents=True)
        for name in files:
            (module_dir / name).write_text(f"# {module}/{name}\n")
    return src


def _build_with_fake_src(tmp_path: Path) -> Path:
    """Patch _src_root and build a tarball, returning its path."""
    src = _create_fake_src(tmp_path)
    cache = tmp_path / "cache"
    with patch("serve.agent_bundler._src_root", return_value=src):
        return build_agent_tarball(cache_dir=cache)


def test_build_tarball_creates_file(tmp_path: Path) -> None:
    """Building a tarball should produce a .tar.gz file on disk."""
    tarball = _build_with_fake_src(tmp_path)
    assert tarball.exists()


def test_build_tarball_contains_entry_script(tmp_path: Path) -> None:
    """The tarball should include the generated entry-point script."""
    tarball = _build_with_fake_src(tmp_path)
    with tarfile.open(tarball, "r:gz") as tar:
        names = tar.getnames()
    assert "forge_agent_entry.py" in names


def test_build_tarball_includes_module_files(tmp_path: Path) -> None:
    """The tarball should include regular module source files."""
    tarball = _build_with_fake_src(tmp_path)
    with tarfile.open(tarball, "r:gz") as tar:
        names = tar.getnames()
    assert "core/config.py" in names


def test_build_tarball_excludes_host_only_files(tmp_path: Path) -> None:
    """Host-only serve files like ssh_connection.py must be excluded."""
    tarball = _build_with_fake_src(tmp_path)
    with tarfile.open(tarball, "r:gz") as tar:
        names = tar.getnames()
    assert "serve/ssh_connection.py" not in names


def test_build_tarball_excludes_pyc_files(tmp_path: Path) -> None:
    """Compiled .pyc files must not appear in the tarball."""
    src = _create_fake_src(tmp_path)
    (src / "core" / "config.pyc").write_bytes(b"\x00")
    cache = tmp_path / "cache"
    with patch("serve.agent_bundler._src_root", return_value=src):
        tarball = build_agent_tarball(cache_dir=cache)
    with tarfile.open(tarball, "r:gz") as tar:
        names = tar.getnames()
    assert "core/config.pyc" not in names


def test_build_tarball_cache_hit_returns_existing(tmp_path: Path) -> None:
    """A second build with identical content should return the cached path."""
    src = _create_fake_src(tmp_path)
    cache = tmp_path / "cache"
    with patch("serve.agent_bundler._src_root", return_value=src):
        first = build_agent_tarball(cache_dir=cache)
        second = build_agent_tarball(cache_dir=cache)
    assert first == second


def test_should_include_excludes_pycache() -> None:
    """__pycache__ directories must be excluded regardless of module."""
    assert _should_include(Path("core/__pycache__"), "core") is False


def test_should_include_allows_regular_serve_file() -> None:
    """Non-excluded serve files should be included."""
    assert _should_include(Path("serve/training.py"), "serve") is True


def test_compute_src_hash_deterministic(tmp_path: Path) -> None:
    """Hashing the same source tree twice should produce the same digest."""
    src = _create_fake_src(tmp_path)
    first = _compute_src_hash(src)
    second = _compute_src_hash(src)
    assert first == second
