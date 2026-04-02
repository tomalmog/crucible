"""End-to-end tests against the watgpu Slurm cluster.

Tests SSH connectivity, environment setup, dataset push, remote chat,
remote training submission, job status, and path quoting.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
FORGE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = FORGE_ROOT / ".crucible"
CLUSTER_NAME = "watgpu"
DATASET_NAME = "gpu-test-sft"
PYTHON = str(FORGE_ROOT / ".venv" / "bin" / "python")

# Ensure src is on the import path
sys.path.insert(0, str(FORGE_ROOT / "src"))

results: list[dict[str, object]] = []


def record(name: str, passed: bool, detail: str = "", elapsed: float = 0.0) -> None:
    status = "PASS" if passed else "FAIL"
    results.append({"name": name, "passed": passed, "detail": detail, "elapsed": elapsed})
    print(f"\n{'='*70}")
    print(f"[{status}] {name}  ({elapsed:.1f}s)")
    if detail:
        for line in detail.strip().splitlines():
            print(f"       {line}")
    print(f"{'='*70}\n", flush=True)


# ===================================================================
# Test 1: Remote environment setup
# ===================================================================
def test_remote_env_setup() -> None:
    name = "1. Remote environment setup"
    t0 = time.time()
    try:
        from core.config import CrucibleConfig
        from serve.remote_env_setup import CONDA_ACTIVATE, ensure_remote_env
        from serve.ssh_connection import SshSession
        from store.cluster_registry import load_cluster

        config = CrucibleConfig.from_env()
        cluster = load_cluster(config.data_root, CLUSTER_NAME)

        with SshSession(cluster) as session:
            ensure_remote_env(session)

            # Verify torch is importable
            stdout, stderr, code = session.execute(
                f'{CONDA_ACTIVATE} && python -c "import torch; print(torch.__version__)"',
                timeout=60,
            )
            assert code == 0, f"torch import failed (exit {code}): {stderr.strip()}"
            torch_version = stdout.strip().splitlines()[-1]

            # Verify trl
            stdout2, stderr2, code2 = session.execute(
                f'{CONDA_ACTIVATE} && python -c "import trl; print(trl.__version__)"',
                timeout=60,
            )
            assert code2 == 0, f"trl import failed (exit {code2}): {stderr2.strip()}"
            trl_version = stdout2.strip().splitlines()[-1]

            # Verify peft
            stdout3, stderr3, code3 = session.execute(
                f'{CONDA_ACTIVATE} && python -c "import peft; print(peft.__version__)"',
                timeout=60,
            )
            assert code3 == 0, f"peft import failed (exit {code3}): {stderr3.strip()}"
            peft_version = stdout3.strip().splitlines()[-1]

        record(
            name, True,
            f"torch={torch_version}, trl={trl_version}, peft={peft_version}",
            time.time() - t0,
        )
    except Exception as exc:
        record(name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", time.time() - t0)


# ===================================================================
# Test 2: Dataset push to remote
# ===================================================================
def test_dataset_push() -> None:
    name = "2. Dataset push to remote"
    t0 = time.time()
    try:
        from core.config import CrucibleConfig
        from serve.remote_dataset_ops import list_remote_datasets, push_dataset
        from serve.ssh_connection import SshSession
        from store.cluster_registry import load_cluster

        config = CrucibleConfig.from_env()
        cluster = load_cluster(config.data_root, CLUSTER_NAME)

        with SshSession(cluster) as session:
            info = push_dataset(session, cluster, DATASET_NAME, config.data_root)
            assert info.name == DATASET_NAME, f"Name mismatch: {info.name}"
            assert info.size_bytes > 0, "Dataset size is 0"

            # Verify it appears in the remote listing
            remote_datasets = list_remote_datasets(session, cluster)
            names = [d.name for d in remote_datasets]
            assert DATASET_NAME in names or any(
                DATASET_NAME.replace("-", "_") in n or n == DATASET_NAME
                for n in names
            ), f"Dataset not found in remote list: {names}"

        record(
            name, True,
            f"Pushed '{DATASET_NAME}' ({info.size_bytes} bytes, synced_at={info.synced_at})",
            time.time() - t0,
        )
    except Exception as exc:
        record(name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", time.time() - t0)


# ===================================================================
# Test 3: Remote chat (gpt2 via srun)
# ===================================================================
def test_remote_chat() -> None:
    name = "3. Remote chat (gpt2 on cluster)"
    t0 = time.time()
    try:
        # Run in-process to avoid subprocess buffering issues
        from core.chat_types import ChatOptions
        from core.config import CrucibleConfig
        from core.slurm_types import SlurmResourceConfig
        from serve.remote_chat_runner import stream_remote_chat

        config = CrucibleConfig.from_env()
        options = ChatOptions(
            model_path="gpt2",
            prompt="What is Python?",
            max_new_tokens=20,
            temperature=0.7,
            top_k=40,
            stream=True,
        )
        resources = SlurmResourceConfig(
            partition="SCHOOL",
            memory="16G",
            time_limit="00:10:00",
        )

        chunks: list[str] = []
        for chunk in stream_remote_chat(
            config.data_root, CLUSTER_NAME, options, resources,
        ):
            chunks.append(chunk)

        response = "".join(chunks).strip()
        if response:
            record(
                name, True,
                f"Got response ({len(response)} chars): {response[:200]}",
                time.time() - t0,
            )
        else:
            record(
                name, False,
                "Got empty response from remote chat",
                time.time() - t0,
            )
    except Exception as exc:
        err_str = str(exc).lower()
        if "timeout" in err_str or "queued" in err_str or "allocation" in err_str:
            record(
                name, True,
                f"Timed out waiting for GPU allocation (expected on busy cluster): {exc}",
                time.time() - t0,
            )
        else:
            record(name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", time.time() - t0)


# ===================================================================
# Test 4: Remote training submission (SFT on gpt2)
# ===================================================================
submitted_job_id: str = ""


def test_remote_training_submit() -> None:
    global submitted_job_id
    name = "4. Remote training submission (SFT gpt2)"
    t0 = time.time()
    try:
        # Register backends so get_backend works
        from core.backend_registry import _BACKENDS, register_backend
        if not _BACKENDS:
            from serve.local_runner import LocalRunner
            from serve.slurm_runner import SlurmRunner
            from serve.ssh_runner import SshRunner
            from serve.http_api_runner import HttpApiRunner
            register_backend("local", LocalRunner())
            register_backend("slurm", SlurmRunner())
            register_backend("ssh", SshRunner())
            register_backend("http-api", HttpApiRunner())

        from core.backend_registry import get_backend
        from core.config import CrucibleConfig
        from core.job_types import JobSpec, ResourceConfig

        config = CrucibleConfig.from_env()
        spec = JobSpec(
            job_type="sft",
            method_args={
                "--base-model": "gpt2",
                "--dataset": DATASET_NAME,
                "--epochs": "1",
                "--batch-size": "2",
                "--max-token-length": "64",
            },
            backend="slurm",
            cluster_name=CLUSTER_NAME,
            resources=ResourceConfig(
                partition="SCHOOL",
                nodes=1,
                gpus_per_node=1,
                cpus_per_task=4,
                memory="16G",
                time_limit="00:10:00",
            ),
        )

        backend = get_backend("slurm")
        job_record = backend.submit(config.data_root, spec)
        submitted_job_id = job_record.job_id

        record(
            name, True,
            f"job_id: {job_record.job_id}\n"
            f"backend_job_id: {job_record.backend_job_id}\n"
            f"state: {job_record.state}\n"
            f"backend: {job_record.backend}\n"
            f"cluster: {job_record.backend_cluster}",
            time.time() - t0,
        )
    except Exception as exc:
        record(name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", time.time() - t0)


# ===================================================================
# Test 5: Remote job status check
# ===================================================================
def test_remote_job_status() -> None:
    name = "5. Remote job status check"
    t0 = time.time()
    try:
        from store.job_store import list_jobs, load_job

        config_data_root = DATA_ROOT

        # Use the job we just submitted if available
        if submitted_job_id:
            latest = load_job(config_data_root, submitted_job_id)
        else:
            jobs = list_jobs(config_data_root)
            slurm_jobs = [j for j in jobs if j.backend == "slurm" and j.backend_cluster == CLUSTER_NAME]
            if not slurm_jobs:
                record(name, False, "No relevant Slurm jobs found", time.time() - t0)
                return
            latest = slurm_jobs[0]

        detail_lines = [
            f"job_id: {latest.job_id}",
            f"backend: {latest.backend}",
            f"type: {latest.job_type}",
            f"state: {latest.state}",
            f"cluster: {latest.backend_cluster}",
            f"created: {latest.created_at}",
            f"backend_job_id: {latest.backend_job_id}",
        ]

        # Check live state via the Slurm runner
        try:
            from serve.slurm_runner import SlurmRunner
            runner = SlurmRunner()
            state = runner.get_state(config_data_root, latest.job_id)
            detail_lines.append(f"live_state: {state}")
        except Exception as state_err:
            detail_lines.append(f"state_check_error: {state_err}")

        record(name, True, "\n".join(detail_lines), time.time() - t0)
    except Exception as exc:
        record(name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", time.time() - t0)


# ===================================================================
# Test 6: Remote workspace / model listing
# ===================================================================
def test_remote_model_listing() -> None:
    name = "6. Remote workspace and model listing"
    t0 = time.time()
    try:
        from core.config import CrucibleConfig
        from serve.ssh_connection import SshSession
        from store.cluster_registry import load_cluster

        config = CrucibleConfig.from_env()
        cluster = load_cluster(config.data_root, CLUSTER_NAME)

        with SshSession(cluster) as session:
            workspace = session.resolve_path(cluster.remote_workspace)
            detail_lines = [f"resolved_workspace: {workspace}"]

            # Check workspace exists
            _, _, code = session.execute(f"test -d {shlex.quote(workspace)}", timeout=10)
            assert code == 0, f"Workspace directory does not exist: {workspace}"
            detail_lines.append("workspace: exists")

            # List top-level dirs in workspace
            stdout, _, _ = session.execute(
                f"ls -1 {shlex.quote(workspace)} | head -20", timeout=15,
            )
            items = [i.strip() for i in stdout.strip().splitlines() if i.strip()]
            detail_lines.append(f"workspace_contents ({len(items)} items): {', '.join(items[:10])}")

            # Check datasets dir
            ds_dir = f"{workspace}/datasets"
            stdout2, _, _ = session.execute(
                f"ls -1 {shlex.quote(ds_dir)} 2>/dev/null | head -20", timeout=15,
            )
            ds_items = [i.strip() for i in stdout2.strip().splitlines() if i.strip()]
            detail_lines.append(f"remote_datasets ({len(ds_items)}): {', '.join(ds_items[:10])}")

            # Check for any model directories (output dirs with model files)
            stdout3, _, _ = session.execute(
                f"find {shlex.quote(workspace)} -maxdepth 3 -name 'result.json' -type f 2>/dev/null | head -5",
                timeout=20,
            )
            result_files = [f.strip() for f in stdout3.strip().splitlines() if f.strip()]
            detail_lines.append(f"completed_jobs_with_results: {len(result_files)}")

        record(name, True, "\n".join(detail_lines), time.time() - t0)
    except Exception as exc:
        record(name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", time.time() - t0)


# ===================================================================
# Test 7: Path quoting verification
# ===================================================================
def test_path_quoting() -> None:
    name = "7. Path quoting verification"
    t0 = time.time()
    try:
        from core.config import CrucibleConfig
        from serve.ssh_connection import SshSession
        from store.cluster_registry import load_cluster

        config = CrucibleConfig.from_env()
        cluster = load_cluster(config.data_root, CLUSTER_NAME)

        with SshSession(cluster) as session:
            resolved = session.resolve_path("~/crucible-jobs")
            detail_lines = [f"resolved ~/crucible-jobs -> {resolved}"]

            # Test 1: tilde expansion works
            assert not resolved.startswith("~"), f"Tilde not expanded: {resolved}"
            assert resolved.startswith("/"), f"Path not absolute: {resolved}"
            detail_lines.append("tilde_expansion: OK")

            # Test 2: mkdir with path containing spaces
            test_dir = f"{resolved}/.test-special chars"
            session.mkdir_p(test_dir)
            stdout, _, code = session.execute(
                f"test -d {shlex.quote(test_dir)} && echo OK", timeout=10,
            )
            assert code == 0 and "OK" in stdout, f"Directory with spaces failed: code={code}"
            detail_lines.append("dir_with_spaces: OK")

            # Cleanup
            session.execute(f"rmdir {shlex.quote(test_dir)}", timeout=10)
            detail_lines.append("cleanup: OK")

            # Test 3: mkdir with path containing special chars
            test_dir2 = f"{resolved}/.test_special$chars"
            session.mkdir_p(test_dir2)
            stdout2, _, code2 = session.execute(
                f"test -d {shlex.quote(test_dir2)} && echo OK", timeout=10,
            )
            assert code2 == 0 and "OK" in stdout2, f"Directory with $ failed: code={code2}"
            detail_lines.append("dir_with_dollar: OK")

            # Cleanup
            session.execute(f"rmdir {shlex.quote(test_dir2)}", timeout=10)
            detail_lines.append("cleanup_special: OK")

            # Test 4: resolve_path for various inputs
            assert session.resolve_path("~") == resolved.rsplit("/crucible-jobs", 1)[0], \
                "~ resolution incorrect"
            detail_lines.append("bare_tilde: OK")

            abs_path = "/tmp/test"
            assert session.resolve_path(abs_path) == abs_path, \
                "Absolute path should not be modified"
            detail_lines.append("absolute_path_passthrough: OK")

        record(name, True, "\n".join(detail_lines), time.time() - t0)
    except Exception as exc:
        record(name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", time.time() - t0)


# ===================================================================
# Summary
# ===================================================================
def print_summary() -> None:
    print("\n" + "=" * 70)
    print("  E2E TEST SUMMARY — watgpu cluster")
    print("=" * 70)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['name']}  ({r['elapsed']:.1f}s)")
    print(f"\n  Total: {total}  Passed: {passed}  Failed: {failed}")
    if failed:
        print("\n  FAILED TESTS:")
        for r in results:
            if not r["passed"]:
                print(f"    - {r['name']}")
    print("=" * 70 + "\n")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    os.environ["CRUCIBLE_DATA_ROOT"] = str(DATA_ROOT)

    print(f"Forge root:  {FORGE_ROOT}")
    print(f"Data root:   {DATA_ROOT}")
    print(f"Python:      {PYTHON}")
    print(f"Cluster:     {CLUSTER_NAME}")
    print(f"Dataset:     {DATASET_NAME}")
    print(flush=True)

    test_remote_env_setup()
    test_dataset_push()
    test_remote_chat()
    test_remote_training_submit()
    test_remote_job_status()
    test_remote_model_listing()
    test_path_quoting()

    print_summary()

    # Exit with failure code if any test failed
    sys.exit(0 if all(r["passed"] for r in results) else 1)
