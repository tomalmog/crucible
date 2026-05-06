"""Shell helpers for Crucible's managed remote conda environment."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serve.ssh_connection import SshSession

ENV_NAME = "crucible"
_PIP_PACKAGES = (
    "pyyaml", "numpy<2", "matplotlib", "tokenizers",
    "transformers", "accelerate", "safetensors", "datasets",
    "trl", "peft", "bitsandbytes",
)
_IMPORT_CHECK = (
    "import torch, yaml, matplotlib, tokenizers, transformers, accelerate, "
    "safetensors, datasets, trl, peft, bitsandbytes"
)

# Shell snippet that sources conda's init script from common locations.
# We must NOT use ``eval "$(conda shell.bash hook)" || fallback`` because
# when conda is not on PATH the subshell produces empty output and
# ``eval ""`` exits 0, so the fallback never runs. Instead we always
# scan for conda.sh in well-known paths.
CONDA_INIT = (
    "for p in "
    "$HOME/miniconda3 $HOME/anaconda3 $HOME/miniforge3 "
    "/opt/conda /opt/miniconda3 /opt/anaconda3; do "
    'if [ -f "$p/etc/profile.d/conda.sh" ]; then '
    '. "$p/etc/profile.d/conda.sh"; break; fi; done'
)


def conda_cmd(command: str, storage_prefix: str = "") -> str:
    """Wrap *command* so conda is initialised in the shell first."""
    if storage_prefix:
        return f"{CONDA_INIT} && {storage_prefix} && {command}"
    return f"{CONDA_INIT} && {command}"


def managed_conda_exports(_remote_workspace: str, user: str) -> str:
    """Build shell exports for Crucible's node-local managed env and cache."""
    safe_user = user or "crucible"
    storage_root = shlex.quote(f"/var/tmp/{safe_user}-crucible-conda")
    envs_root = shlex.quote(f"/var/tmp/{safe_user}-crucible-conda/envs")
    pkgs_root = shlex.quote(f"/var/tmp/{safe_user}-crucible-conda/pkgs")
    return (
        f"mkdir -p {storage_root} {envs_root} {pkgs_root} && "
        f"export CONDA_ENVS_PATH={envs_root} && "
        f"export CONDA_PKGS_DIRS={pkgs_root}"
    )


def managed_conda_activate(remote_workspace: str, user: str) -> str:
    """Build the activation command for Crucible's managed conda env."""
    exports = managed_conda_exports(remote_workspace, user)
    return conda_cmd(f"conda activate {ENV_NAME}", exports)


def managed_conda_bootstrap(remote_workspace: str, user: str) -> str:
    """Build shell code that ensures the managed env exists on this node."""
    exports = managed_conda_exports(remote_workspace, user)
    import_check = shlex.quote(_IMPORT_CHECK)
    pip_packages = " ".join(f"'{pkg}'" for pkg in _PIP_PACKAGES)
    tos_accept = (
        "conda config --set solver libmamba 2>/dev/null; "
        "yes | conda tos accept --override-channels "
        "--channel https://repo.anaconda.com/pkgs/main 2>/dev/null; "
        "yes | conda tos accept --override-channels "
        "--channel https://repo.anaconda.com/pkgs/r 2>/dev/null"
    )
    return (
        f"{CONDA_INIT} && {exports} && "
        f"{tos_accept}; "
        f"if ! conda run -n {ENV_NAME} python -c {import_check} "
        ">/dev/null 2>&1; then "
        "echo 'CRUCIBLE_ENV_SETUP: preparing node-local crucible env...'; "
        f"if ! conda env list | awk '{{print $1}}' | grep -qx {ENV_NAME}; then "
        f"conda create -n {ENV_NAME} python=3.11 -y || exit 1; "
        "fi; "
        "cuda_tag=cu124; "
        "cuda_line=$(nvidia-smi 2>/dev/null | grep 'CUDA Version' | head -n 1 || true); "
        "if [ -n \"$cuda_line\" ]; then "
        "major=$(printf '%s' \"$cuda_line\" | "
        "sed -n 's/.*CUDA Version: *\\([0-9][0-9]*\\)\\.\\([0-9][0-9]*\\).*/\\1/p'); "
        "minor=$(printf '%s' \"$cuda_line\" | "
        "sed -n 's/.*CUDA Version: *\\([0-9][0-9]*\\)\\.\\([0-9][0-9]*\\).*/\\2/p'); "
        "if [ -n \"$major\" ] && [ -n \"$minor\" ]; then "
        "if [ \"$major\" -ge 13 ] || "
        "([ \"$major\" -eq 12 ] && [ \"$minor\" -ge 8 ]); then cuda_tag=cu128; "
        "elif [ \"$major\" -eq 12 ] && [ \"$minor\" -ge 6 ]; then cuda_tag=cu126; "
        "elif [ \"$major\" -eq 12 ] && [ \"$minor\" -ge 4 ]; then cuda_tag=cu124; "
        "elif [ \"$major\" -eq 12 ]; then cuda_tag=cu121; "
        "else cuda_tag=cu118; fi; fi; fi; "
        "echo \"CRUCIBLE_ENV_SETUP: installing torch ($cuda_tag)...\"; "
        f"if [ \"$cuda_tag\" = cu128 ] || [ \"$cuda_tag\" = cu129 ]; then "
        f"conda run -n {ENV_NAME} pip install torch "
        "--index-url https://download.pytorch.org/whl/nightly/$cuda_tag || exit 1; "
        "else "
        f"conda run -n {ENV_NAME} pip install torch "
        "--index-url https://download.pytorch.org/whl/$cuda_tag || exit 1; "
        "fi; "
        f"conda run -n {ENV_NAME} pip install --only-binary :all: {pip_packages} "
        "|| exit 1; "
        "fi"
    )


def managed_conda_command(session: SshSession, command: str) -> str:
    """Build a conda command for the active SSH session."""
    remote_workspace = session.resolve_path(session.cluster.remote_workspace)
    exports = managed_conda_exports(remote_workspace, session.cluster.user)
    return conda_cmd(command, exports)
