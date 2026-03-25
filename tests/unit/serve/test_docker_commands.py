"""Tests for Docker CLI command builders and output parsers."""

from __future__ import annotations

import pytest

from core.errors import CrucibleDockerError
from serve.docker_commands import (
    build_docker_run_cmd,
    build_gpu_flags,
    parse_container_id,
    parse_docker_state,
)


class TestBuildGpuFlags:
    """Tests for build_gpu_flags()."""

    def test_empty_returns_all(self) -> None:
        """Empty GPU IDs should yield --gpus all."""
        assert build_gpu_flags("") == "--gpus all"

    def test_whitespace_returns_all(self) -> None:
        """Whitespace-only GPU IDs should yield --gpus all."""
        assert build_gpu_flags("  ") == "--gpus all"

    def test_single_device(self) -> None:
        """Single GPU device ID should be quoted correctly."""
        result = build_gpu_flags("0")
        assert "device=0" in result

    def test_multiple_devices(self) -> None:
        """Multiple GPU device IDs should be comma-separated."""
        result = build_gpu_flags("0,1")
        assert "device=0,1" in result


class TestBuildDockerRunCmd:
    """Tests for build_docker_run_cmd()."""

    def test_basic_command(self) -> None:
        """Basic docker run command contains image and -d flag."""
        result = build_docker_run_cmd(
            image="pytorch/pytorch:latest",
            gpu_flags="--gpus all",
            volumes=(("/host/data", "/data"),),
            workdir="/workspace",
            command="python train.py",
        )
        assert "docker run -d" in result
        assert "pytorch/pytorch:latest" in result
        assert "-v /host/data:/data" in result
        assert "-w /workspace" in result
        assert "python train.py" in result

    def test_multiple_volumes(self) -> None:
        """Multiple volume mounts appear in the command."""
        result = build_docker_run_cmd(
            image="img",
            gpu_flags="--gpus all",
            volumes=(("/a", "/b"), ("/c", "/d")),
            workdir="/w",
            command="echo hi",
        )
        assert "-v /a:/b" in result
        assert "-v /c:/d" in result


class TestParseContainerId:
    """Tests for parse_container_id()."""

    def test_valid_id(self) -> None:
        """Valid hex container ID should be parsed to 12 chars."""
        stdout = "abc123def456789\n"
        assert parse_container_id(stdout) == "abc123def456"

    def test_multiline_output(self) -> None:
        """Container ID should be extracted from the last line."""
        stdout = "some warning\nabc123def456\n"
        assert parse_container_id(stdout) == "abc123def456"

    def test_invalid_output_raises(self) -> None:
        """Non-hex output should raise CrucibleDockerError."""
        with pytest.raises(CrucibleDockerError):
            parse_container_id("Error: something went wrong")


class TestParseDockerState:
    """Tests for parse_docker_state()."""

    def test_running_state(self) -> None:
        """Docker 'running' should map to JobState 'running'."""
        assert parse_docker_state("running\n") == "running"

    def test_exited_state(self) -> None:
        """Docker 'exited' should map to JobState 'completed'."""
        assert parse_docker_state("exited") == "completed"

    def test_dead_state(self) -> None:
        """Docker 'dead' should map to JobState 'failed'."""
        assert parse_docker_state("dead") == "failed"

    def test_created_state(self) -> None:
        """Docker 'created' should map to JobState 'pending'."""
        assert parse_docker_state("created") == "pending"

    def test_unknown_state_raises(self) -> None:
        """Unrecognised Docker state should raise CrucibleDockerError."""
        with pytest.raises(CrucibleDockerError):
            parse_docker_state("nonsense")
