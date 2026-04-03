"""Tests for HumanEval platform safety — preexec_fn must not be used on non-Linux
systems, and solution checking must handle timeouts/OS errors gracefully."""

from __future__ import annotations

import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call


# ── preexec_fn platform guard ────────────────────────────────────────────────


def test_preexec_fn_not_passed_on_non_linux(tmp_path: Path) -> None:
    """subprocess.run must NOT receive preexec_fn on non-Linux platforms."""
    from eval.benchmarks.humaneval import _check_solution

    captured_kwargs = {}

    def fake_run(cmd, **kwargs):
        captured_kwargs.update(kwargs)
        result = MagicMock()
        result.returncode = 0
        return result

    # A trivially correct solution
    code = "def add(a, b): return a + b"
    test_code = "def check(f): assert f(1, 2) == 3"
    entry_point = "add"

    with (
        patch("eval.benchmarks.humaneval.platform") as mock_platform,
        patch("eval.benchmarks.humaneval.subprocess.run", side_effect=fake_run),
    ):
        mock_platform.system.return_value = "Darwin"  # macOS
        _check_solution(code, test_code, entry_point)

    assert "preexec_fn" not in captured_kwargs, (
        "preexec_fn must not be passed to subprocess.run on non-Linux (e.g. macOS/Windows)"
    )


def test_preexec_fn_is_passed_on_linux(tmp_path: Path) -> None:
    """subprocess.run receives preexec_fn on Linux for memory sandboxing."""
    from eval.benchmarks.humaneval import _check_solution

    captured_kwargs = {}

    def fake_run(cmd, **kwargs):
        captured_kwargs.update(kwargs)
        result = MagicMock()
        result.returncode = 0
        return result

    code = "def f(): pass"
    test_code = "def check(f): pass"
    entry_point = "f"

    with (
        patch("eval.benchmarks.humaneval.platform") as mock_platform,
        patch("eval.benchmarks.humaneval.subprocess.run", side_effect=fake_run),
    ):
        mock_platform.system.return_value = "Linux"
        _check_solution(code, test_code, entry_point)

    assert "preexec_fn" in captured_kwargs, (
        "preexec_fn should be passed on Linux for memory limiting"
    )


def test_preexec_fn_not_passed_on_windows(tmp_path: Path) -> None:
    """subprocess.run must NOT receive preexec_fn on Windows."""
    from eval.benchmarks.humaneval import _check_solution

    captured_kwargs = {}

    def fake_run(cmd, **kwargs):
        captured_kwargs.update(kwargs)
        result = MagicMock()
        result.returncode = 0
        return result

    code = "def f(): pass"
    test_code = "def check(f): pass"
    entry_point = "f"

    with (
        patch("eval.benchmarks.humaneval.platform") as mock_platform,
        patch("eval.benchmarks.humaneval.subprocess.run", side_effect=fake_run),
    ):
        mock_platform.system.return_value = "Windows"
        _check_solution(code, test_code, entry_point)

    assert "preexec_fn" not in captured_kwargs


# ── Solution checking behavior ────────────────────────────────────────────────


def test_correct_solution_returns_true() -> None:
    """A correct solution that passes tests returns True."""
    from eval.benchmarks.humaneval import _check_solution

    # This will actually run Python, so keep it trivial
    code = "def add(a, b):\n    return a + b\n"
    test_code = "def check(add):\n    assert add(1, 2) == 3\n    assert add(0, 0) == 0\n"
    result = _check_solution(code, test_code, "add")
    assert result is True


def test_wrong_solution_returns_false() -> None:
    """A solution that fails tests returns False (not raises)."""
    from eval.benchmarks.humaneval import _check_solution

    code = "def add(a, b):\n    return a - b  # wrong!\n"
    test_code = "def check(add):\n    assert add(1, 2) == 3\n"
    result = _check_solution(code, test_code, "add")
    assert result is False


def test_solution_with_syntax_error_returns_false() -> None:
    """Syntactically broken generated code returns False, not raises."""
    from eval.benchmarks.humaneval import _check_solution

    code = "def add(a, b):\n    return a + b +\n"  # syntax error
    test_code = "def check(add): pass\n"
    result = _check_solution(code, test_code, "add")
    assert result is False


def test_solution_with_timeout_returns_false() -> None:
    """Infinite-loop solution times out and returns False."""
    from eval.benchmarks.humaneval import _check_solution

    code = "def f():\n    while True: pass\n"
    test_code = "def check(f):\n    f()\n"
    result = _check_solution(code, test_code, "f")
    assert result is False


def test_timeout_expired_is_caught_not_raised() -> None:
    """subprocess.TimeoutExpired is caught and returns False — never propagates."""
    from eval.benchmarks.humaneval import _check_solution

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("python", 10)

    with patch("eval.benchmarks.humaneval.subprocess.run", side_effect=fake_run):
        result = _check_solution("def f(): pass", "def check(f): pass", "f")

    assert result is False


def test_os_error_is_caught_not_raised() -> None:
    """OSError from subprocess is caught and returns False."""
    from eval.benchmarks.humaneval import _check_solution

    def fake_run(*args, **kwargs):
        raise OSError("No such file: python")

    with patch("eval.benchmarks.humaneval.subprocess.run", side_effect=fake_run):
        result = _check_solution("def f(): pass", "def check(f): pass", "f")

    assert result is False


def test_tmp_file_cleaned_up_after_success() -> None:
    """Temp file is deleted even when the solution check succeeds."""
    from eval.benchmarks.humaneval import _check_solution

    created_files = []

    original_named_temp = tempfile.NamedTemporaryFile

    def capture_tmp(*args, **kwargs):
        obj = original_named_temp(*args, **kwargs)
        created_files.append(Path(obj.name))
        return obj

    def fake_run(cmd, **kwargs):
        mock = MagicMock()
        mock.returncode = 0
        return mock

    with (
        patch("eval.benchmarks.humaneval.tempfile.NamedTemporaryFile", side_effect=capture_tmp),
        patch("eval.benchmarks.humaneval.subprocess.run", side_effect=fake_run),
        patch("eval.benchmarks.humaneval.platform") as mock_platform,
    ):
        mock_platform.system.return_value = "Darwin"
        _check_solution("def f(): pass", "def check(f): pass", "f")

    # All created temp files should have been cleaned up
    for path in created_files:
        assert not path.exists(), f"Temp file {path} was not cleaned up"


def test_tmp_file_cleaned_up_after_failure() -> None:
    """Temp file is deleted even when the solution check fails."""
    from eval.benchmarks.humaneval import _check_solution

    created_files = []
    original_named_temp = tempfile.NamedTemporaryFile

    def capture_tmp(*args, **kwargs):
        obj = original_named_temp(*args, **kwargs)
        created_files.append(Path(obj.name))
        return obj

    def fake_run(cmd, **kwargs):
        mock = MagicMock()
        mock.returncode = 1  # failure
        return mock

    with (
        patch("eval.benchmarks.humaneval.tempfile.NamedTemporaryFile", side_effect=capture_tmp),
        patch("eval.benchmarks.humaneval.subprocess.run", side_effect=fake_run),
        patch("eval.benchmarks.humaneval.platform") as mock_platform,
    ):
        mock_platform.system.return_value = "Darwin"
        _check_solution("def f(): pass", "def check(f): pass", "f")

    for path in created_files:
        assert not path.exists()


# ── Memory limit function ─────────────────────────────────────────────────────


def test_limit_subprocess_memory_does_not_raise_on_macos() -> None:
    """_limit_subprocess_memory must not raise on macOS (ValueError from setrlimit is caught)."""
    from eval.benchmarks.humaneval import _limit_subprocess_memory

    with patch("eval.benchmarks.humaneval.resource.setrlimit") as mock_setrlimit:
        mock_setrlimit.side_effect = ValueError("Not supported on macOS")
        # Must not raise
        _limit_subprocess_memory()


def test_limit_subprocess_memory_does_not_raise_on_os_error() -> None:
    """_limit_subprocess_memory handles OSError from setrlimit gracefully."""
    from eval.benchmarks.humaneval import _limit_subprocess_memory

    with patch("eval.benchmarks.humaneval.resource.setrlimit") as mock_setrlimit:
        mock_setrlimit.side_effect = OSError("Operation not permitted")
        _limit_subprocess_memory()
