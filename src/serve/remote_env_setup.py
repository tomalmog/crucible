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

ENV_NAME = "forge"
_PIP_PACKAGES = (
    "pyyaml", "numpy<2", "matplotlib", "tokenizers",
    "transformers", "accelerate", "safetensors",
)

# Shell snippet that sources conda's init script from common locations.
# We must NOT use ``eval "$(conda shell.bash hook)" || fallback`` because
# when conda is not on PATH the subshell produces empty output and
# ``eval ""`` exits 0, so the fallback never runs.  Instead we always
# scan for conda.sh in well-known paths.
CONDA_INIT = (
    "for p in "
    "$HOME/miniconda3 $HOME/anaconda3 $HOME/miniforge3 "
    "/opt/conda /opt/miniconda3 /opt/anaconda3; do "
    'if [ -f "$p/etc/profile.d/conda.sh" ]; then '
    '. "$p/etc/profile.d/conda.sh"; break; fi; done'
)


def conda_cmd(command: str) -> str:
    """Wrap *command* so conda is initialised in the shell first."""
    return f"{CONDA_INIT} && {command}"


def ensure_remote_env(session: SshSession) -> None:
    """Ensure the ``forge`` conda env exists on the remote cluster.

    Checks ``conda env list`` for a ``forge`` entry and creates it
    if missing.  If the existing env is broken (torch missing or
    install failures), removes it and rebuilds automatically.

    Args:
        session: Active SSH session to the cluster.

    Raises:
        ForgeRemoteError: If conda is unavailable or env creation
            fails after one automatic retry.
    """
    if _env_exists(session):
        try:
            _ensure_torch_installed(session)
            return
        except ForgeRemoteError:
            print(
                "FORGE_ENV_SETUP: forge env is broken — rebuilding...",
                flush=True,
            )
            _remove_env(session)

    print("FORGE_ENV_SETUP: creating forge conda env...", flush=True)
    try:
        _create_env(session)
    except ForgeRemoteError:
        # First attempt failed (e.g. stale partial env) — retry once
        print(
            "FORGE_ENV_SETUP: creation failed — cleaning up and retrying...",
            flush=True,
        )
        _remove_env(session)
        _create_env(session)
    print("FORGE_ENV_SETUP: forge conda env ready.", flush=True)


def reset_remote_env(session: SshSession) -> None:
    """Remove the ``forge`` conda env so it is rebuilt on next job.

    Safe to call even if the env does not exist.
    """
    if _env_exists(session):
        _remove_env(session)
        print("FORGE_ENV_SETUP: forge env removed.", flush=True)
    else:
        print("FORGE_ENV_SETUP: no forge env found — nothing to remove.", flush=True)


def _ensure_torch_installed(session: SshSession) -> None:
    """Check that torch is importable in the forge env."""
    check_script = "import torch; print('torch=' + torch.__version__)"
    stdout, _, code = session.execute(
        conda_cmd(
            f'conda run -n {ENV_NAME} python -c "{check_script}"'
        ),
        timeout=30,
    )
    if code == 0 and "torch=" in stdout:
        return

    print(
        "FORGE_ENV_SETUP: torch not found in forge env — installing...",
        flush=True,
    )
    cuda_tag = _detect_cuda_tag(session)
    torch_install = (
        f"torch --index-url https://download.pytorch.org/whl/{cuda_tag}"
    )
    _, stderr, code = session.execute(
        conda_cmd(
            f"conda run -n {ENV_NAME} pip install {torch_install}",
        ),
        timeout=600,
    )
    if code != 0:
        raise ForgeRemoteError(f"torch install failed: {stderr.strip()}")
    print("FORGE_ENV_SETUP: torch installed.", flush=True)


def _remove_env(session: SshSession) -> None:
    """Remove the ``forge`` conda env.  Ignores errors if already gone."""
    session.execute(
        conda_cmd(f"conda remove -n {ENV_NAME} --all -y"),
        timeout=120,
    )


def _env_exists(session: SshSession) -> bool:
    """Return True if the ``forge`` conda env already exists."""
    stdout, _, code = session.execute(
        conda_cmd("conda env list"), timeout=30,
    )
    if code != 0:
        raise ForgeRemoteError(
            "conda is not available on the remote cluster. "
            "Install Miniconda or ensure conda is on the PATH."
        )
    for line in stdout.splitlines():
        # Each line: "envname  /path/to/env" or "* /path"
        parts = line.split()
        if parts and parts[0] == ENV_NAME:
            return True
    return False


def _detect_cuda_tag(session: SshSession) -> str:
    """Detect the cluster's CUDA version and return a pip index tag.

    Tries nvidia-smi on the login node first; if unavailable (common
    on HPC clusters where GPUs are only on compute nodes), queries a
    compute node via ``srun``.
    """
    import re

    cuda_output = ""

    # Try login node first
    stdout, _, code = session.execute(
        "nvidia-smi 2>/dev/null | head -3", timeout=15,
    )
    if code == 0 and "CUDA Version" in stdout:
        cuda_output = stdout
    else:
        # Login node has no GPU — try querying a compute node
        stdout2, _, code2 = session.execute(
            "srun --gres=gpu:1 --time=00:01:00 --quiet "
            "nvidia-smi 2>/dev/null | head -3",
            timeout=90,
        )
        if code2 == 0 and "CUDA Version" in stdout2:
            cuda_output = stdout2

    for line in cuda_output.splitlines():
        if "CUDA Version" in line:
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", line)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                if major >= 13:
                    return "cu126"
                if major == 12 and minor >= 8:
                    return "cu126"
                if major == 12 and minor >= 6:
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
        conda_cmd(f"conda create -n {ENV_NAME} python=3.11 -y"),
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
        conda_cmd(
            f"conda run -n {ENV_NAME} pip install {torch_install}",
        ),
        timeout=600,
    )
    if code != 0:
        raise ForgeRemoteError(f"torch install failed: {stderr.strip()}")

    pip_list = " ".join(f"'{p}'" for p in _PIP_PACKAGES)
    _, stderr, code = session.execute(
        conda_cmd(f"conda run -n {ENV_NAME} pip install {pip_list}"),
        timeout=600,
    )
    if code != 0:
        raise ForgeRemoteError(f"pip install failed: {stderr.strip()}")
