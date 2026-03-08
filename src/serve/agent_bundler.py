"""Build a tarball of Forge modules for remote execution.

Bundles the minimal set of Forge source modules needed to run
training on a remote Slurm cluster, plus a generated entry script.
"""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
from pathlib import Path

from core.errors import ForgeRemoteError
from serve.agent_entry_script import ENTRY_SCRIPT

# Modules to include in the agent tarball
_BUNDLE_MODULES = (
    "core",
    "serve",
    "store",
    "ingest",
    "transforms",
)

# Files within serve/ that should NOT be bundled (they are host-only)
_SERVE_EXCLUDES = frozenset({
    "ssh_connection.py",
    "cluster_validator.py",
    "agent_bundler.py",
    "agent_entry_script.py",
    "slurm_script_gen.py",
    "remote_job_submitter.py",
    "remote_log_streamer.py",
    "remote_job_state.py",
    "remote_model_registry.py",
})


def _src_root() -> Path:
    """Locate the Forge src/ directory."""
    return Path(__file__).resolve().parent.parent


def _should_include(path: Path, module: str) -> bool:
    """Check if a file should be included in the bundle."""
    if module == "serve" and path.name in _SERVE_EXCLUDES:
        return False
    if path.name == "__pycache__":
        return False
    if path.suffix == ".pyc":
        return False
    return True


def _compute_src_hash(src_root: Path) -> str:
    """Compute a content hash of bundled source files for caching."""
    hasher = hashlib.sha256()
    # Include the entry script in the hash
    hasher.update(ENTRY_SCRIPT.encode("utf-8"))
    for module in _BUNDLE_MODULES:
        module_dir = src_root / module
        if not module_dir.is_dir():
            continue
        for py_file in sorted(module_dir.rglob("*.py")):
            if not _should_include(py_file, module):
                continue
            hasher.update(py_file.read_bytes())
    return hasher.hexdigest()[:16]


def build_agent_tarball(cache_dir: Path | None = None) -> Path:
    """Build a tarball containing Forge modules and entry script.

    Args:
        cache_dir: Directory to cache the tarball. If None, uses tempdir.

    Returns:
        Path to the generated .tar.gz file.
    """
    src_root = _src_root()
    content_hash = _compute_src_hash(src_root)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"forge-agent-{content_hash}.tar.gz"
        if cached.exists():
            return cached
        tarball_path = cached
    else:
        tmp = tempfile.mkdtemp(prefix="forge-agent-")
        tarball_path = Path(tmp) / "forge-agent.tar.gz"

    try:
        with tarfile.open(tarball_path, "w:gz") as tar:
            # Add source modules
            for module in _BUNDLE_MODULES:
                module_dir = src_root / module
                if not module_dir.is_dir():
                    continue
                for file_path in sorted(module_dir.rglob("*")):
                    if not file_path.is_file():
                        continue
                    if not _should_include(file_path, module):
                        continue
                    arcname = str(file_path.relative_to(src_root))
                    tar.add(str(file_path), arcname=arcname)

            # Add entry script
            _add_string_to_tar(
                tar, "forge_agent_entry.py", ENTRY_SCRIPT,
            )
    except Exception as error:
        raise ForgeRemoteError(
            f"Failed to build agent tarball: {error}"
        ) from error

    return tarball_path


def _add_string_to_tar(tar: tarfile.TarFile, name: str, content: str) -> None:
    """Add a string as a file to a tarball."""
    import io
    import time

    data = content.encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))
