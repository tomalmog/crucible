"""Auto-provision a conda environment on remote clusters.

Ensures a ``forge`` conda env exists before job submission,
installing torch and other training dependencies if needed.

SSH ``exec_command`` runs a non-interactive, non-login shell that
does *not* source ``~/.bashrc``, so ``conda`` (a shell function)
is unavailable by default.  We source the conda init script
explicitly before every conda command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.errors import ForgeRemoteError

if TYPE_CHECKING:
    from serve.ssh_connection import SshSession

_ENV_NAME = "forge"
_PIP_PACKAGES = ("pyyaml", "matplotlib", "tokenizers")

# Shell snippet that sources conda's init script from common locations.
# We must NOT use ``eval "$(conda shell.bash hook)" || fallback`` because
# when conda is not on PATH the subshell produces empty output and
# ``eval ""`` exits 0, so the fallback never runs.  Instead we always
# scan for conda.sh in well-known paths.
_CONDA_INIT = (
    "for p in "
    "$HOME/miniconda3 $HOME/anaconda3 $HOME/miniforge3 "
    "/opt/conda /opt/miniconda3 /opt/anaconda3; do "
    'if [ -f "$p/etc/profile.d/conda.sh" ]; then '
    '. "$p/etc/profile.d/conda.sh"; break; fi; done'
)


def _conda_cmd(command: str) -> str:
    """Wrap *command* so conda is initialised in the shell first."""
    return f"{_CONDA_INIT} && {command}"


def ensure_remote_env(session: SshSession) -> None:
    """Ensure the ``forge`` conda env exists on the remote cluster.

    Checks ``conda env list`` for a ``forge`` entry and creates it
    if missing.  When the env exists, verifies that torch can see
    CUDA and reinstalls if needed.

    Args:
        session: Active SSH session to the cluster.

    Raises:
        ForgeRemoteError: If conda is unavailable or env creation fails.
    """
    if _env_exists(session):
        _ensure_torch_cuda(session)
        return

    print("FORGE_ENV_SETUP: forge conda env not found — creating...", flush=True)
    _create_env(session)
    print("FORGE_ENV_SETUP: forge conda env ready.", flush=True)


def _ensure_torch_cuda(session: SshSession) -> None:
    """Check that torch can see CUDA; reinstall if not."""
    stdout, _, code = session.execute(
        _conda_cmd(
            f"conda run -n {_ENV_NAME} python -c "
            "\"import torch; print(torch.cuda.is_available())\""
        ),
        timeout=30,
    )
    if code == 0 and "True" in stdout:
        return

    print(
        "FORGE_ENV_SETUP: torch cannot see CUDA — reinstalling...",
        flush=True,
    )
    cuda_tag = _detect_cuda_tag(session)
    torch_install = (
        f"torch --index-url https://download.pytorch.org/whl/{cuda_tag}"
    )
    _, stderr, code = session.execute(
        _conda_cmd(
            f"conda run -n {_ENV_NAME} pip install --force-reinstall"
            f" {torch_install}",
        ),
        timeout=600,
    )
    if code != 0:
        raise ForgeRemoteError(f"torch reinstall failed: {stderr.strip()}")
    print("FORGE_ENV_SETUP: torch reinstalled with CUDA support.", flush=True)


def _env_exists(session: SshSession) -> bool:
    """Return True if the ``forge`` conda env already exists."""
    stdout, _, code = session.execute(
        _conda_cmd("conda env list"), timeout=30,
    )
    if code != 0:
        raise ForgeRemoteError(
            "conda is not available on the remote cluster. "
            "Install Miniconda or ensure conda is on the PATH."
        )
    for line in stdout.splitlines():
        # Each line: "envname  /path/to/env" or "* /path"
        parts = line.split()
        if parts and parts[0] == _ENV_NAME:
            return True
    return False


def _detect_cuda_tag(session: SshSession) -> str:
    """Detect the cluster's CUDA version and return a pip index tag."""
    stdout, _, code = session.execute(
        "nvidia-smi --query-gpu=driver_version "
        "--format=csv,noheader 2>/dev/null | head -1",
        timeout=15,
    )
    if code != 0 or not stdout.strip():
        return "cu124"  # safe default

    # Parse CUDA version from nvidia-smi header instead
    stdout2, _, _ = session.execute(
        "nvidia-smi 2>/dev/null | head -3", timeout=15,
    )
    # Look for "CUDA Version: XX.Y"
    for line in stdout2.splitlines():
        if "CUDA Version" in line:
            import re
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", line)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                # Map to available PyTorch CUDA builds
                if major >= 13:
                    return "cu126"
                if major == 12 and minor >= 4:
                    return "cu124"
                if major == 12:
                    return "cu121"
                return "cu118"
    return "cu124"


def _create_env(session: SshSession) -> None:
    """Create the ``forge`` conda env and install dependencies."""
    _, stderr, code = session.execute(
        _conda_cmd(f"conda create -n {_ENV_NAME} python=3.11 -y"),
        timeout=300,
    )
    if code != 0:
        raise ForgeRemoteError(f"conda create failed: {stderr.strip()}")

    # Detect CUDA version and install matching torch build.
    cuda_tag = _detect_cuda_tag(session)
    torch_install = (
        f"torch --index-url https://download.pytorch.org/whl/{cuda_tag}"
    )
    print(f"FORGE_ENV_SETUP: Installing torch ({cuda_tag})...", flush=True)
    _, stderr, code = session.execute(
        _conda_cmd(
            f"conda run -n {_ENV_NAME} pip install {torch_install}",
        ),
        timeout=600,
    )
    if code != 0:
        raise ForgeRemoteError(f"torch install failed: {stderr.strip()}")

    pip_list = " ".join(_PIP_PACKAGES)
    _, stderr, code = session.execute(
        _conda_cmd(f"conda run -n {_ENV_NAME} pip install {pip_list}"),
        timeout=600,
    )
    if code != 0:
        raise ForgeRemoteError(f"pip install failed: {stderr.strip()}")
