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
from serve.managed_conda_env import (
    ENV_NAME,
    managed_conda_command,
)

if TYPE_CHECKING:
    from serve.ssh_connection import SshSession

_PIP_PACKAGES = (
    "pyyaml", "numpy<2", "matplotlib", "tokenizers",
    "transformers", "accelerate", "safetensors", "datasets",
    "trl", "peft", "bitsandbytes",
)
# Extra packages only needed for eval jobs.  Installed on demand via
# ``ensure_eval_packages`` to avoid dependency conflicts with the base
# training stack.
_EVAL_PACKAGES = ("lm-eval",)
# Mapping from pip package name to Python import name when they differ.
_IMPORT_NAMES: dict[str, str] = {
    "pyyaml": "yaml",
    "lm-eval": "lm_eval",
}


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
        except CrucibleRemoteError as exc:
            if _is_env_runtime_error(str(exc)):
                raise
            print(
                "CRUCIBLE_ENV_SETUP: crucible env is broken — rebuilding...",
                flush=True,
            )
            _remove_env(session)

    print("CRUCIBLE_ENV_SETUP: creating crucible conda env...", flush=True)
    try:
        _create_env(session)
    except CrucibleRemoteError as exc:
        if _is_env_runtime_error(str(exc)):
            raise
        # First attempt failed (e.g. stale partial env) — retry once
        print(
            "CRUCIBLE_ENV_SETUP: creation failed — cleaning up and retrying...",
            flush=True,
        )
        _remove_env(session)
        _create_env(session)
    print("CRUCIBLE_ENV_SETUP: crucible conda env ready.", flush=True)


def ensure_eval_packages(session: SshSession) -> None:
    """Install eval-only packages (``lm-eval``) into the crucible env.

    Called by the eval submission path after the base env is ready.
    Separated from the base install because ``lm-eval`` has heavy
    dependency requirements that can conflict with training packages
    when installed together via ``--only-binary :all:``.
    """
    def _import_name(pkg: str) -> str:
        clean = pkg.split("<")[0].split(">")[0].strip()
        return _IMPORT_NAMES.get(clean, clean)

    imports = "; ".join(
        f"__import__('{_import_name(p)}')" for p in _EVAL_PACKAGES
    )
    _, _, code = session.execute(
        managed_conda_command(
            session,
            f'conda run -n {ENV_NAME} python -c "{imports}"'
        ),
        timeout=60,
    )
    if code == 0:
        return
    print("CRUCIBLE_ENV_SETUP: installing eval packages...", flush=True)
    pip_list = " ".join(f"'{p}'" for p in _EVAL_PACKAGES)
    # Pin numpy<2 and force pre-built wheels for compiled packages
    # (numpy, scikit-learn) to avoid building from source on clusters
    # with old GCC. lm-eval itself is pure Python so it's fine from sdist.
    _, stderr, code = session.execute(
        managed_conda_command(
            session,
            f"conda run -n {ENV_NAME} pip install "
            f"--only-binary numpy,scikit-learn "
            f"'numpy<2' {pip_list}"
        ),
        timeout=600,
    )
    if code != 0:
        raise CrucibleRemoteError(f"eval package install failed: {stderr.strip()}")
    print("CRUCIBLE_ENV_SETUP: eval packages ready.", flush=True)


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
        managed_conda_command(session, "conda remove -n forge --all -y"),
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
    stdout, stderr, code = session.execute(
        managed_conda_command(
            session,
            f'conda run -n {ENV_NAME} python -c "{check_script}"'
        ),
        timeout=60,
    )
    if code == 0 and "torch=" in stdout:
        # Verify required packages are present
        _ensure_packages_installed(session)
        return
    _raise_env_runtime_error(stdout, stderr)

    print(
        "CRUCIBLE_ENV_SETUP: torch not found in crucible env — installing...",
        flush=True,
    )
    cuda_tag = _detect_cuda_tag(session)
    torch_install = _torch_install_spec(cuda_tag)
    _, stderr, code = session.execute(
        managed_conda_command(
            session,
            f"conda run -n {ENV_NAME} pip install {torch_install}",
        ),
        timeout=600,
    )
    if code != 0:
        raise CrucibleRemoteError(f"torch install failed: {stderr.strip()}")
    print("CRUCIBLE_ENV_SETUP: torch installed.", flush=True)
    verify_stdout, verify_stderr, verify_code = session.execute(
        managed_conda_command(
            session,
            f'conda run -n {ENV_NAME} python -c "{check_script}"'
        ),
        timeout=60,
    )
    if verify_code != 0 or "torch=" not in verify_stdout:
        _raise_env_runtime_error(verify_stdout, verify_stderr)
        raise CrucibleRemoteError(
            "torch install finished but the crucible env still cannot import torch."
        )
    _ensure_packages_installed(session)


def _ensure_packages_installed(session: SshSession) -> None:
    """Install any missing pip packages into the crucible env."""
    def _import_name(pkg: str) -> str:
        clean = pkg.split("<")[0].split(">")[0].strip()
        return _IMPORT_NAMES.get(clean, clean)

    imports = "; ".join(
        f"__import__('{_import_name(p)}')" for p in _PIP_PACKAGES
    )
    _, _, code = session.execute(
        managed_conda_command(
            session,
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
        managed_conda_command(
            session,
            f"conda run -n {ENV_NAME} pip install --only-binary :all: {pip_list}",
        ),
        timeout=600,
    )
    if code != 0:
        raise CrucibleRemoteError(f"pip install failed: {stderr.strip()}")


def _remove_env(session: SshSession) -> None:
    """Remove the ``crucible`` conda env.  Ignores errors if already gone."""
    session.execute(
        managed_conda_command(session, f"conda remove -n {ENV_NAME} --all -y"),
        timeout=120,
    )


def _list_env_names(session: SshSession) -> set[str]:
    """Return the set of conda env names on the remote cluster."""
    stdout, _, code = session.execute(
        managed_conda_command(session, "conda env list"), timeout=60,
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
        managed_conda_command(
            session,
            "conda config --set solver libmamba 2>/dev/null; "
            "yes | conda tos accept --override-channels "
            "--channel https://repo.anaconda.com/pkgs/main 2>/dev/null; "
            "yes | conda tos accept --override-channels "
            "--channel https://repo.anaconda.com/pkgs/r 2>/dev/null"
        ),
        timeout=30,
    )
    _, stderr, code = session.execute(
        managed_conda_command(session, f"conda create -n {ENV_NAME} python=3.11 -y"),
        timeout=300,
    )
    if code != 0:
        raise CrucibleRemoteError(f"conda create failed: {stderr.strip()}")

    # Detect CUDA version and install matching torch build.
    cuda_tag = _detect_cuda_tag(session)
    torch_install = _torch_install_spec(cuda_tag)
    print(f"CRUCIBLE_ENV_SETUP: Installing torch ({cuda_tag})...", flush=True)
    _, stderr, code = session.execute(
        managed_conda_command(
            session,
            f"conda run -n {ENV_NAME} pip install {torch_install}",
        ),
        timeout=600,
    )
    if code != 0:
        raise CrucibleRemoteError(f"torch install failed: {stderr.strip()}")

    pip_list = " ".join(f"'{p}'" for p in _PIP_PACKAGES)
    _, stderr, code = session.execute(
        managed_conda_command(
            session,
            f"conda run -n {ENV_NAME} pip install --only-binary :all: {pip_list}",
        ),
        timeout=600,
    )
    if code != 0:
        raise CrucibleRemoteError(f"pip install failed: {stderr.strip()}")
    _ensure_torch_installed(session)


def _raise_env_runtime_error(stdout: str, stderr: str) -> None:
    """Raise a clearer error when the managed env exists but cannot execute."""
    details = "\n".join(part for part in (stderr.strip(), stdout.strip()) if part)
    if _is_env_runtime_error(details):
        raise CrucibleRemoteError(
            "The managed crucible env is on a noexec filesystem and cannot run. "
            "Move remote_workspace to an exec-enabled shared path or set "
            "cluster python_path to an executable shared environment."
        )


def _is_env_runtime_error(message: str) -> bool:
    """Return whether the error points to a non-executable env runtime."""
    lowered = message.lower()
    return "permission denied" in lowered or "failed to map segment" in lowered
