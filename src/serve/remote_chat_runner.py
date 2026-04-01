"""Run chat inference on a remote cluster via SSH.

Connects to the cluster, ensures the crucible conda env and source
bundle are present, then runs inference.  For Slurm clusters the
command is submitted via ``srun``; for SSH clusters the runner script
executes directly on the remote host.
"""

from __future__ import annotations

import base64
import hashlib
import shlex
import sys
from collections.abc import Generator
from pathlib import Path

from core.chat_types import ChatOptions
from core.errors import CrucibleRemoteError
from core.slurm_types import SlurmResourceConfig
from serve.agent_bundler import build_agent_tarball
from serve.remote_env_setup import CONDA_ACTIVATE, ensure_remote_env
from serve.ssh_connection import SshSession
from store.cluster_registry import load_cluster

_BUNDLE_DIR_NAME = ".crucible-chat-bundle"
_HASH_MARKER = ".bundle-hash"
_RUNNER_FILENAME = "_chat_runner.py"


def stream_remote_chat(
    data_root: Path,
    cluster_name: str,
    options: ChatOptions,
    resources: SlurmResourceConfig,
) -> Generator[str, None, None]:
    """Stream chat inference from a remote cluster model.

    Args:
        data_root: Root .crucible directory.
        cluster_name: Registered cluster name.
        options: Chat inference options (model_path, prompt, etc.).
        resources: Slurm resource allocation for the compute node.

    Yields:
        Stdout chunks from remote inference.
    """
    cluster = load_cluster(data_root, cluster_name)
    is_ssh = cluster.backend == "ssh"
    tarball = build_agent_tarball(
        cache_dir=data_root / "cache" / "agent-bundles",
    )

    with SshSession(cluster) as session:
        # Redirect setup output to stderr so it doesn't pollute
        # the chat data channel (stdout is the token stream).
        saved_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            if is_ssh:
                from serve.ssh_submit_helpers import provision_env
                env_activate = provision_env(session, cluster)
            else:
                ensure_remote_env(session)
                env_activate = ""
            bundle_dir = f"{cluster.remote_workspace}/{_BUNDLE_DIR_NAME}"
            _sync_bundle(session, tarball, bundle_dir)
            _upload_runner_script(session, bundle_dir, options)
        finally:
            sys.stdout = saved_stdout

        if is_ssh:
            command = _build_ssh_command(bundle_dir, env_activate)
        else:
            command = _build_srun_command(
                cluster.module_loads, bundle_dir, resources,
                default_partition=cluster.default_partition,
            )
        yield from session.stream_command(command, timeout=1800)


def _sync_bundle(
    session: SshSession,
    tarball: Path,
    bundle_dir: str,
) -> None:
    """Upload and extract the crucible bundle if the hash has changed."""
    local_hash = _file_hash(tarball)
    marker_path = f"{bundle_dir}/{_HASH_MARKER}"

    stdout, _, code = session.execute(
        f"cat {shlex.quote(marker_path)} 2>/dev/null", timeout=10,
    )
    if code == 0 and stdout.strip() == local_hash:
        return

    session.mkdir_p(bundle_dir)
    remote_tarball = f"{bundle_dir}/crucible-agent.tar.gz"
    session.upload(tarball, remote_tarball)
    _, stderr, code = session.execute(
        f"tar -xzf {shlex.quote(remote_tarball)} -C {shlex.quote(bundle_dir)}",
        timeout=60,
    )
    if code != 0:
        raise CrucibleRemoteError(f"Bundle extraction failed: {stderr.strip()}")
    session.upload_text(local_hash, marker_path)


def _file_hash(path: Path) -> str:
    """Compute a short SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def _upload_runner_script(
    session: SshSession,
    bundle_dir: str,
    options: ChatOptions,
) -> None:
    """Upload the chat runner script to the remote bundle directory."""
    # Resolve ~ so the compute node gets an absolute path.
    model_path = session.resolve_path(options.model_path)

    encoded = base64.b64encode(
        options.prompt.encode("utf-8"),
    ).decode("ascii")
    script = _RUNNER_TEMPLATE.format(
        encoded_prompt=encoded,
        model_path=model_path,
        max_new_tokens=options.max_new_tokens,
        temperature=options.temperature,
        top_k=options.top_k,
    )
    session.upload_text(script, f"{bundle_dir}/{_RUNNER_FILENAME}")


def _build_ssh_command(
    bundle_dir: str,
    env_activate: str = "",
) -> str:
    """Build a direct command for SSH clusters (no Slurm)."""
    parts: list[str] = []
    if env_activate:
        parts.append(env_activate)
    parts.append(f"cd {bundle_dir}")
    parts.append(f"python -u {_RUNNER_FILENAME}")
    return " && ".join(parts)


def _build_srun_command(
    module_loads: tuple[str, ...],
    bundle_dir: str,
    resources: SlurmResourceConfig,
    default_partition: str = "",
) -> str:
    """Build the srun command that runs inference on a compute node."""
    parts: list[str] = list(module_loads)
    parts.append(CONDA_ACTIVATE)
    parts.append(f"cd {bundle_dir}")

    srun = ["srun", "--nodes=1", "--ntasks=1"]
    gres = f"gpu:{resources.gpu_type}:1" if resources.gpu_type else "gpu:1"
    srun.append(f"--gres={gres}")
    partition = resources.partition or default_partition
    if partition:
        srun.append(f"--partition={partition}")
    if resources.memory:
        srun.append(f"--mem={resources.memory}")
    if resources.time_limit:
        srun.append(f"--time={resources.time_limit}")
    srun.append(f"python -u {_RUNNER_FILENAME}")

    parts.append(" ".join(srun))
    return " && ".join(parts)


# Runner script template uploaded to the remote cluster.
# Includes a stdout buffer that coalesces per-token writes into
# word-boundary chunks to reduce SSH packet overhead.
_RUNNER_TEMPLATE = """\
import sys, io, base64
sys.path.insert(0, ".")


class _ChunkBuffer(io.TextIOBase):
    \"\"\"Collects small writes, flushes on whitespace or threshold.\"\"\"

    _THRESHOLD = 48

    def __init__(self, target):
        self._target = target
        self._buf = ""

    def write(self, text):
        self._buf += text
        if len(self._buf) >= self._THRESHOLD or " " in text or "\\n" in text:
            self._target.write(self._buf)
            self._target.flush()
            self._buf = ""
        return len(text)

    def flush(self):
        if self._buf:
            self._target.write(self._buf)
            self._target.flush()
            self._buf = ""


sys.stdout = _ChunkBuffer(sys.stdout)

from core.chat_types import ChatOptions
from serve.chat_runner import run_chat

prompt = base64.b64decode("{encoded_prompt}").decode("utf-8")
opts = ChatOptions(
    model_path="{model_path}",
    prompt=prompt,
    max_new_tokens={max_new_tokens},
    temperature={temperature},
    top_k={top_k},
    stream=True,
)
run_chat(None, opts)
sys.stdout.flush()
"""
