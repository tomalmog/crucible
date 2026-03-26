"""Auto-provision a conda environment on remote clusters.

Ensures a ``crucible`` conda env exists before job submission,
installing torch and other training dependencies if needed.

SSH ``exec_command`` runs a non-interactive, non-login shell that
does *not* source ``~/.bashrc``, so ``conda`` (a shell function)
is unavailable by default.  We source the conda init script
explicitly before every conda command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.errors import CrucibleRemoteError

if TYPE_CHECKING:
    from serve.ssh_connection import SshSession

ENV_NAME = "crucible"
_PIP_PACKAGES = (
    "pyyaml", "numpy<2", "matplotlib", "tokenizers",
    "transformers", "accelerate", "safetensors", "datasets",
    "trl", "peft", "bitsandbytes",
)
# Mapping from pip package name to Python import name when they differ.
_IMPORT_NAMES: dict[str, str] = {
    "pyyaml": "yaml",
}

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

CONDA_ACTIVATE = f"{CONDA_INIT} && conda activate {ENV_NAME}"


def conda_cmd(command: str) -> str:
    """Wrap *command* so conda is initialised in the shell first."""
    return f"{CONDA_INIT} && {command}"


def ensure_remote_env(session: SshSession) -> None:
    """Ensure the ``crucible`` conda env exists on the remote cluster.

    Checks ``conda env list`` for a ``crucible`` entry and creates it
    if missing.  If the existing env is broken (torch missing or
    install failures), removes it and rebuilds automatically.

    Also removes the legacy ``forge`` env if present (from before the
    Forge → Crucible rename) to avoid confusion and free disk space.

    Args:
        session: Active SSH session to the cluster.

    Raises:
        CrucibleRemoteError: If conda is unavailable or env creation
            fails after one automatic retry.
    """
    try:
        env_names = _list_env_names(session)
    except CrucibleRemoteError:
        # Retry once — HPC login nodes can be slow on first access
        import time
        time.sleep(3)
        env_names = _list_env_names(session)
    if "forge" in env_names:
        _remove_legacy_forge_env(session)
    if ENV_NAME in env_names:
        try:
            _ensure_torch_installed(session)
            return
        except CrucibleRemoteError:
            print(
                "CRUCIBLE_ENV_SETUP: crucible env is broken — rebuilding...",
                flush=True,
            )
            _remove_env(session)

    print("CRUCIBLE_ENV_SETUP: creating crucible conda env...", flush=True)
    try:
        _create_env(session)
    except CrucibleRemoteError:
        # First attempt failed (e.g. stale partial env) — retry once
        print(
            "CRUCIBLE_ENV_SETUP: creation failed — cleaning up and retrying...",
            flush=True,
        )
        _remove_env(session)
        _create_env(session)
    print("CRUCIBLE_ENV_SETUP: crucible conda env ready.", flush=True)


def reset_remote_env(session: SshSession) -> None:
    """Remove the ``crucible`` conda env so it is rebuilt on next job.

    Also removes the legacy ``forge`` env if present (from before the
    Forge → Crucible rename).  Safe to call even if neither env exists.
    """
    env_names = _list_env_names(session)
    if "forge" in env_names:
        _remove_legacy_forge_env(session)
    if ENV_NAME in env_names:
        _remove_env(session)
        print("CRUCIBLE_ENV_SETUP: crucible env removed.", flush=True)
    else:
        print("CRUCIBLE_ENV_SETUP: no crucible env found — nothing to remove.", flush=True)


def _remove_legacy_forge_env(session: SshSession) -> None:
    """Remove the old ``forge`` conda env.

    Called only when ``_list_env_names`` already confirmed "forge"
    is present.  Frees disk space and avoids confusion after the
    Forge → Crucible rename.
    """
    print(
        "CRUCIBLE_ENV_SETUP: removing legacy 'forge' conda env...",
        flush=True,
    )
    session.execute(
        conda_cmd("conda remove -n forge --all -y"),
        timeout=120,
    )
    print(
        "CRUCIBLE_ENV_SETUP: legacy 'forge' env removed.",
        flush=True,
    )


def _ensure_torch_installed(session: SshSession) -> None:
    """Check that torch is importable in the crucible env.

    Only checks that torch can be imported — we cannot verify CUDA
    availability here because this runs on the login node which
    typically has no GPU.  CUDA compatibility is ensured at env
    creation time by matching the torch build to the detected CUDA
    version.
    """
    check_script = "import torch; print('torch=' + torch.__version__)"
    stdout, _, code = session.execute(
        conda_cmd(
            f'conda run -n {ENV_NAME} python -c "{check_script}"'
        ),
        timeout=60,
    )
    if code == 0 and "torch=" in stdout:
        # Verify required packages are present
        _ensure_packages_installed(session)
        return

    print(
        "CRUCIBLE_ENV_SETUP: torch not found in crucible env — installing...",
        flush=True,
    )
    cuda_tag = _detect_cuda_tag(session)
    torch_install = _torch_install_spec(cuda_tag)
    _, stderr, code = session.execute(
        conda_cmd(
            f"conda run -n {ENV_NAME} pip install {torch_install}",
        ),
        timeout=600,
    )
    if code != 0:
        raise CrucibleRemoteError(f"torch install failed: {stderr.strip()}")
    print("CRUCIBLE_ENV_SETUP: torch installed.", flush=True)
    _ensure_packages_installed(session)


def _ensure_packages_installed(session: SshSession) -> None:
    """Install any missing pip packages into the crucible env."""
    def _import_name(pkg: str) -> str:
        clean = pkg.split("<")[0].split(">")[0].strip()
        return _IMPORT_NAMES.get(clean, clean)

    imports = " ".join(
        f"__import__('{_import_name(p)}')" for p in _PIP_PACKAGES
    )
    _, _, code = session.execute(
        conda_cmd(
            f'conda run -n {ENV_NAME} python -c "{imports}"'
        ),
        timeout=60,
    )
    if code == 0:
        return
    print(
        "CRUCIBLE_ENV_SETUP: installing missing packages...",
        flush=True,
    )
    pip_list = " ".join(f"'{p}'" for p in _PIP_PACKAGES)
    _, stderr, code = session.execute(
        conda_cmd(f"conda run -n {ENV_NAME} pip install --only-binary :all: {pip_list}"),
        timeout=600,
    )
    if code != 0:
        raise CrucibleRemoteError(f"pip install failed: {stderr.strip()}")


def _remove_env(session: SshSession) -> None:
    """Remove the ``crucible`` conda env.  Ignores errors if already gone."""
    session.execute(
        conda_cmd(f"conda remove -n {ENV_NAME} --all -y"),
        timeout=120,
    )


def _list_env_names(session: SshSession) -> set[str]:
    """Return the set of conda env names on the remote cluster."""
    stdout, _, code = session.execute(
        conda_cmd("conda env list"), timeout=60,
    )
    if code != 0:
        raise CrucibleRemoteError(
            "conda is not available on the remote cluster. "
            "Install Miniconda or ensure conda is on the PATH."
        )
    names: set[str] = set()
    for line in stdout.splitlines():
        # Each line: "envname  /path/to/env" or "* /path"
        parts = line.split()
        if parts and parts[0] not in ("*", "#"):
            names.add(parts[0])
    return names


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
                # CUDA 12.8+ supports Blackwell (sm_120) — needs cu128
                if major >= 13 or (major == 12 and minor >= 8):
                    return "cu128"
                if major == 12 and minor >= 6:
                    return "cu126"
                if major == 12 and minor >= 4:
                    return "cu124"
                if major == 12:
                    return "cu121"
                return "cu118"
    return "cu124"


def _torch_install_spec(cuda_tag: str) -> str:
    """Return the pip install spec for torch given a CUDA tag.

    Tags like cu128 need the nightly index because stable PyTorch
    wheels don't yet include Blackwell (sm_120) kernels.
    """
    if cuda_tag in ("cu128", "cu129"):
        return (
            f"torch --index-url "
            f"https://download.pytorch.org/whl/nightly/{cuda_tag}"
        )
    return f"torch --index-url https://download.pytorch.org/whl/{cuda_tag}"


def _create_env(session: SshSession) -> None:
    """Create the ``crucible`` conda env and install dependencies."""
    # Accept conda ToS non-interactively (required since conda 25.x).
    session.execute(
        conda_cmd(
            "conda config --set solver libmamba 2>/dev/null; "
            "yes | conda tos accept --override-channels "
            "--channel https://repo.anaconda.com/pkgs/main 2>/dev/null; "
            "yes | conda tos accept --override-channels "
            "--channel https://repo.anaconda.com/pkgs/r 2>/dev/null"
        ),
        timeout=30,
    )
    _, stderr, code = session.execute(
        conda_cmd(f"conda create -n {ENV_NAME} python=3.11 -y"),
        timeout=300,
    )
    if code != 0:
        raise CrucibleRemoteError(f"conda create failed: {stderr.strip()}")

    # Detect CUDA version and install matching torch build.
    cuda_tag = _detect_cuda_tag(session)
    torch_install = _torch_install_spec(cuda_tag)
    print(f"CRUCIBLE_ENV_SETUP: Installing torch ({cuda_tag})...", flush=True)
    _, stderr, code = session.execute(
        conda_cmd(
            f"conda run -n {ENV_NAME} pip install {torch_install}",
        ),
        timeout=600,
    )
    if code != 0:
        raise CrucibleRemoteError(f"torch install failed: {stderr.strip()}")

    pip_list = " ".join(f"'{p}'" for p in _PIP_PACKAGES)
    _, stderr, code = session.execute(
        conda_cmd(f"conda run -n {ENV_NAME} pip install --only-binary :all: {pip_list}"),
        timeout=600,
    )
    if code != 0:
        raise CrucibleRemoteError(f"pip install failed: {stderr.strip()}")
