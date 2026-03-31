"""Crucible MCP Server — exposes all Crucible operations as tools.

Run with: crucible mcp-server
Or add to .claude/mcp.json for automatic discovery by Claude Code.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Ensure src/ is on the path
_src_root = Path(__file__).resolve().parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

# Register execution backends (needed for job operations and remote submission)
def _ensure_backends() -> None:
    from core.backend_registry import _BACKENDS
    if _BACKENDS:
        return
    from serve.local_runner import LocalRunner
    from serve.slurm_runner import SlurmRunner
    from serve.ssh_runner import SshRunner
    from serve.http_api_runner import HttpApiRunner
    from core.backend_registry import register_backend
    register_backend("local", LocalRunner())
    register_backend("slurm", SlurmRunner())
    register_backend("ssh", SshRunner())
    register_backend("http-api", HttpApiRunner())

mcp = FastMCP("Crucible", instructions="""
Crucible is an end-to-end ML training platform with a desktop app (Studio),
Python CLI, and these MCP tools. It manages datasets, trains models (locally
or on remote GPU clusters), runs evaluation benchmarks, and provides
interpretability analysis.

## Key Concepts

- **Datasets** are ingested from JSONL/Parquet files and stored in .crucible/datasets/.
  Each training method expects a specific data format (see below).
- **Models** are registered in .crucible/models/. Training auto-registers the output.
  Models can be Crucible .pt files or HuggingFace model IDs (e.g. "gpt2").
- **Jobs** track all training/eval/interp runs. Local jobs run in-process,
  remote jobs run on GPU clusters via SSH or Slurm.
- **Clusters** are remote GPU machines registered with SSH credentials.

## Training Methods

| Method | ID for `train` tool | Data Format | Required Fields |
|--------|---------------------|-------------|-----------------|
| Basic Training | `train` | `{"text": "..."}` | dataset_name, output_dir |
| SFT | `sft` | `{"prompt": "...", "response": "..."}` | dataset_name, sft_data_path, base_model OR initial_weights_path |
| LoRA | `lora-train` | `{"prompt": "...", "response": "..."}` | dataset_name, lora_data_path, base_model_path |
| QLoRA | `qlora-train` | `{"prompt": "...", "response": "..."}` | dataset_name, qlora_data_path, base_model_path (needs GPU) |
| DPO | `dpo-train` | `{"prompt": "...", "chosen": "...", "rejected": "..."}` | dataset_name, dpo_data_path, base_model |
| KTO | `kto-train` | `{"prompt": "...", "response": "...", "is_desirable": bool}` | dataset_name, kto_data_path, base_model |
| ORPO | `orpo-train` | `{"prompt": "...", "chosen": "...", "rejected": "..."}` | dataset_name, orpo_data_path, base_model |
| GRPO | `grpo-train` | `{"prompt": "..."}` | dataset_name, grpo_data_path, base_model |
| RLVR | `rlvr-train` | `{"prompt": "...", "solution": "..."}` | dataset_name, rlvr_data_path, base_model |
| RLHF | `rlhf-train` | `{"prompt": "...", "chosen": "...", "rejected": "..."}` | dataset_name, policy_model_path |
| Distillation | `distill` | `{"text": "..."}` | dataset_name, teacher_model_path |
| Domain Adapt | `domain-adapt` | `{"text": "..."}` | dataset_name, base_model_path |
| Multimodal | `multimodal-train` | `{"text": "...", "image_path": "..."}` | dataset_name, multimodal_data_path, base_model |

## Common Training Arguments (method_args JSON)

All methods accept these shared fields:
- `epochs` (int, default 3)
- `learning_rate` (float, default varies by method: 1e-3 for basic, 2e-5 for SFT, 2e-4 for LoRA)
- `batch_size` (int, default 16)
- `max_token_length` (int, default 512)
- `precision_mode` ("auto", "fp32", "fp16", "bf16")
- `output_dir` (string, where to save the model)
- `model_name` (string, name for the model registry)

LoRA-specific: `lora_rank` (int), `lora_alpha` (float), `lora_dropout` (float)
DPO-specific: `beta` (float, default 0.1)

## Typical Workflows

**1. Train a model locally:**
1. `ingest_dataset(source_path, name)` -- import data
2. `train(method="sft", method_args='{"dataset_name":"...", "sft_data_path":"...", "base_model":"gpt2", ...}')` -- train
3. The model is auto-registered and visible in `list_models()`

**2. Train on a remote GPU cluster:**
1. `list_clusters()` -- see available clusters
2. `push_dataset(cluster, dataset)` -- upload data
3. `submit_remote_training(cluster, method, args)` -- submit job
4. `job_status(job_id)` / `job_logs(job_id)` -- monitor
5. `job_result(job_id)` -- get results when complete

**3. Evaluate a model:**
1. `run_benchmark(model_path, "mmlu,gsm8k,arc")` -- run benchmarks
   Available: mmlu, gsm8k, hellaswag, arc, truthfulqa, winogrande, humaneval

**4. Chat with a model:**
1. `chat(model_path, "Hello, explain neural networks")` -- generate text

**5. Analyze a model (interpretability):**
1. `run_interp("logit-lens", model_path, '{"input_text": "The cat sat on"}')` -- layer-by-layer predictions
2. `run_interp("activation-pca", model_path, '{"dataset_name": "sft-mini"}')` -- PCA of activations
3. `run_interp("linear-probe", model_path, '{"dataset_name": "...", "label_field": "label"}')` -- train linear classifiers on activations
4. `run_interp("sae-train", model_path, '{"dataset_name": "..."}')` -- train sparse autoencoder on activations
5. `run_interp("sae-analyze", model_path, '{"sae_path": "...", "input_text": "..."}')` -- decompose text through trained SAE
6. `run_interp("steer-compute", model_path, '{"positive_text": "...", "negative_text": "..."}')` -- compute steering vector
7. `run_interp("steer-apply", model_path, '{"steering_vector_path": "...", "input_text": "..."}')` -- generate with steering

**6. Export a model:**
1. `export_model(model_path, "onnx", "./exports")` -- export to ONNX/safetensors/gguf/hf

**7. Merge models:**
1. `merge_models(["model1.pt", "model2.pt"], "average", "./merged")` -- average/slerp/ties/dare

**8. Pull a trained model from remote:**
1. `pull_model(job_id, "my-model")` -- download + register

**9. Hyperparameter sweep:**
1. `run_sweep("sft", "my-dataset", '[{"name":"learning_rate","values":[1e-4,1e-3]}]')` -- grid/random search

**10. A/B model comparison:**
1. `ab_chat("model_a.pt", "model_b.pt", "Explain transformers")` -- compare two models side-by-side

**11. LoRA merge:**
1. `lora_merge("adapter/", "gpt2", "merged.pt")` -- merge LoRA into base model

**12. Dataset curation:**
1. `curate_dataset("my-data", "score")` -- score quality of each record
2. `curate_dataset("my-data", "stats")` -- distribution statistics
3. `curate_dataset("my-data", "filter", min_quality=0.7)` -- filter low-quality records

**13. Synthetic data generation:**
1. `generate_synthetic_data("seeds.txt", count=1000)` -- generate training data from prompts

**14. Hardware detection:**
1. `hardware_profile()` -- detect GPUs, recommended precision, batch size

## Remote Clusters

Clusters are SSH-accessible machines with GPUs. They use either:
- **SSH backend** (vast.ai, Lambda, bare metal) -- direct execution
- **Slurm backend** (university clusters) -- sbatch job submission

The remote env auto-installs torch, trl, peft, transformers in a conda env.

## Important Notes

- HuggingFace model IDs (like "gpt2", "meta-llama/Llama-2-7b") are auto-detected
  and use trl trainers for correct attention masking and checkpoint format.
- Crucible .pt models use the custom training loop.
- The `dataset_name` field links to ingested datasets. The `*_data_path` field
  is the JSONL file path (auto-resolved from the dataset if using dataset_name).
- All training outputs are saved with training_config.json and tokenizer files
  alongside model.pt for reproducibility.
""")


def _try_auto_register_remote(data_root: Path, job: Any, model_path: str | None = None) -> None:
    """Auto-register a remote model in the local registry if not already there."""
    try:
        from store.model_registry import ModelRegistry
        from store.cluster_registry import load_cluster
        registry = ModelRegistry(data_root)
        model_name = job.label or job.model_name or f"remote-{job.job_type}-{job.job_id[:16]}"
        # Check if already registered
        try:
            registry.get_model(model_name)
            return  # already registered
        except Exception:
            pass
        mp = model_path or job.model_path
        if not mp:
            return
        cluster_name = job.backend_cluster or getattr(job, "cluster_name", "")
        if cluster_name:
            cluster = load_cluster(data_root, cluster_name)
            registry.register_remote_model(
                model_name=model_name,
                remote_host=cluster.host,
                remote_path=mp,
                run_id=job.job_id,
            )
        else:
            registry.register_model(model_name, mp, run_id=job.job_id)
    except Exception:
        pass


def _run_with_job(job_type: str, label: str, fn: Any, config: dict | None = None) -> str:
    """Run a function wrapped in job record creation.

    Creates a job record before running, updates it on completion/failure.
    This makes MCP-initiated operations visible on the Jobs page.
    """
    from store.job_store import generate_job_id, now_iso, save_job, update_job
    from core.job_types import JobRecord as JR

    root = _data_root()
    job_id = generate_job_id()
    ts = now_iso()
    save_job(root, JR(
        job_id=job_id, backend="local", job_type=job_type,
        state="running", created_at=ts, updated_at=ts,
        label=label, config=config or {},
    ))
    try:
        result = fn()
        model_path = ""
        if isinstance(result, dict):
            model_path = result.get("model_path", "")
        update_job(root, job_id, state="completed", model_path=model_path)
        # Write result as stdout so the UI can display it
        _write_job_stdout(root, job_id, json.dumps(result, indent=2, default=str) if isinstance(result, dict) else str(result))
        if isinstance(result, dict):
            result["job_id"] = job_id
        return json.dumps(result, indent=2, default=str) if isinstance(result, dict) else str(result)
    except Exception as exc:
        update_job(root, job_id, state="failed", error_message=f"{type(exc).__name__}: {exc}")
        return json.dumps({"error": str(exc), "job_id": job_id})


def _write_job_stdout(data_root: Path, job_id: str, stdout: str) -> None:
    """Write stdout to the job JSON as an extra field."""
    job_path = data_root / "jobs" / f"{job_id}.json"
    if not job_path.exists():
        return
    try:
        data = json.loads(job_path.read_text())
        data["stdout"] = stdout
        job_path.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _get_client():
    from core.config import CrucibleConfig
    from store.dataset_sdk import CrucibleClient
    config = CrucibleConfig.from_env()
    return CrucibleClient(config)


def _data_root() -> Path:
    from core.config import CrucibleConfig
    return CrucibleConfig.from_env().data_root


# ── Datasets ─────────────────────────────────────────────────────────


@mcp.tool()
def list_datasets() -> str:
    """List all locally ingested datasets with record counts."""
    try:
        datasets_dir = _data_root() / "datasets"
        if not datasets_dir.is_dir():
            return "No datasets."
        results = []
        for d in sorted(datasets_dir.iterdir()):
            if not d.is_dir():
                continue
            manifest = d / "manifest.json"
            if manifest.exists():
                data = json.loads(manifest.read_text())
                results.append({
                    "name": d.name,
                    "record_count": data.get("record_count", 0),
                    "source_uri": data.get("source_uri", ""),
                })
            else:
                results.append({"name": d.name, "record_count": "unknown"})
        return json.dumps(results, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def ingest_dataset(source_path: str, dataset_name: str) -> str:
    """Ingest a JSONL, Parquet, or text file into the dataset registry.

    Args:
        source_path: Path to the source file.
        dataset_name: Name for the ingested dataset.
    """
    try:
        from core.ingest_types import IngestOptions
        client = _get_client()
        opts = IngestOptions(dataset_name=dataset_name, source_uri=source_path)
        client.ingest(opts)
        return f"Ingested '{dataset_name}' from {source_path}."
    except Exception as exc:
        return json.dumps({"error": f"Failed to ingest dataset: {exc}"})


@mcp.tool()
def delete_dataset(dataset_name: str) -> str:
    """Delete a local dataset from the registry.

    Args:
        dataset_name: Name of the dataset to delete.
    """
    try:
        import shutil
        dataset_dir = _data_root() / "datasets" / dataset_name
        if not dataset_dir.is_dir():
            return f"Dataset '{dataset_name}' not found."
        shutil.rmtree(dataset_dir)
        return f"Deleted dataset '{dataset_name}'."
    except Exception as exc:
        return json.dumps({"error": f"Failed to delete dataset: {exc}"})


@mcp.tool()
def push_dataset(cluster_name: str, dataset_name: str) -> str:
    """Push a local dataset to a remote cluster.

    Args:
        cluster_name: Name of the registered cluster.
        dataset_name: Name of the dataset to push.
    """
    try:
        from serve.remote_dataset_ops import push_dataset as _push
        from serve.ssh_connection import SshSession
        from store.cluster_registry import load_cluster
        data_root = _data_root()
        cluster = load_cluster(data_root, cluster_name)
        with SshSession(cluster) as session:
            _push(session, cluster, dataset_name, data_root)
        return f"Pushed '{dataset_name}' to {cluster_name}."
    except Exception as exc:
        return json.dumps({"error": f"Failed to push dataset: {exc}"})


@mcp.tool()
def list_remote_datasets(cluster_name: str) -> str:
    """List datasets present on a remote cluster.

    Args:
        cluster_name: Name of the registered cluster.
    """
    try:
        from serve.remote_dataset_ops import list_remote_datasets as _list_remote
        from serve.ssh_connection import SshSession
        from store.cluster_registry import load_cluster
        cluster = load_cluster(_data_root(), cluster_name)
        with SshSession(cluster) as session:
            datasets = _list_remote(session, cluster)
        results = [
            {"name": d.name, "size_bytes": d.size_bytes, "synced_at": d.synced_at}
            for d in datasets
        ]
        return json.dumps(results, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Failed to list remote datasets: {exc}"})


# ── Models ───────────────────────────────────────────────────────────


@mcp.tool()
def list_models() -> str:
    """List all registered models with their paths and locations."""
    try:
        from store.model_registry import ModelRegistry
        registry = ModelRegistry(_data_root())
        models = registry.list_models()
        results = []
        for m in models:
            results.append({
                "name": m.model_name,
                "path": m.model_path,
                "location": m.location_type,
                "remote_host": m.remote_host,
                "remote_path": m.remote_path,
            })
        return json.dumps(results, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def register_model(model_name: str, model_path: str) -> str:
    """Register a model in the model registry.

    Args:
        model_name: Display name for the model.
        model_path: Path to the model file (.pt or HuggingFace directory).
    """
    try:
        from store.model_registry import ModelRegistry
        registry = ModelRegistry(_data_root())
        entry = registry.register_model(model_name, model_path)
        return f"Registered '{entry.model_name}' at {entry.model_path}."
    except Exception as exc:
        return json.dumps({"error": f"Failed to register model: {exc}"})


@mcp.tool()
def delete_model(model_name: str) -> str:
    """Delete a model from the registry.

    Args:
        model_name: Name of the model to delete.
    """
    try:
        from store.model_registry import ModelRegistry
        registry = ModelRegistry(_data_root())
        result = registry.delete_model(model_name)
        return f"Deleted {result.entries_removed} entries."
    except Exception as exc:
        return json.dumps({"error": f"Failed to delete model: {exc}"})


@mcp.tool()
def register_remote_model(
    model_name: str,
    cluster_name: str,
    remote_path: str,
) -> str:
    """Register a model that exists on a remote cluster without downloading it.

    Use this when a model was trained on a cluster and you want it visible
    in the UI model list without pulling the full weights to local disk.

    Args:
        model_name: Display name for the model.
        cluster_name: Name of the cluster where the model lives.
        remote_path: Path to the model on the cluster (e.g. /path/to/output/model.pt).
    """
    try:
        from store.model_registry import ModelRegistry
        from store.cluster_registry import load_cluster
        root = _data_root()
        cluster = load_cluster(root, cluster_name)
        registry = ModelRegistry(root)
        entry = registry.register_remote_model(
            model_name=model_name,
            remote_host=cluster.host,
            remote_path=remote_path,
        )
        return json.dumps({
            "model_name": entry.model_name,
            "location_type": entry.location_type,
            "remote_host": entry.remote_host,
            "remote_path": entry.remote_path,
        })
    except Exception as exc:
        return json.dumps({"error": f"Failed to register remote model: {exc}"})


@mcp.tool()
def pull_model(job_id: str, model_name: str = "") -> str:
    """Pull a trained model from a remote cluster after a completed job.

    Downloads the model files, extracts them locally, and registers
    the model in the local registry.

    Args:
        job_id: The job ID of the completed remote training job.
        model_name: Name for the model in the registry (auto-generated if empty).
    """
    try:
        from serve.remote_model_puller import pull_remote_model
        record = pull_remote_model(
            _data_root(), job_id,
            model_name=model_name or None,
        )
        return json.dumps({
            "job_id": record.job_id,
            "state": record.state,
            "model_path_local": record.model_path_local,
            "model_path_remote": record.model_path_remote,
        })
    except Exception as exc:
        return json.dumps({"error": f"Failed to pull model: {exc}"})


# ── Training ─────────────────────────────────────────────────────────


@mcp.tool()
def train(
    method: str,
    method_args: str,
) -> str:
    """Run a training job locally using any of the 13 training methods.

    Args:
        method: Training method ID. One of: train, sft, dpo-train, lora-train,
                qlora-train, rlhf-train, distill, domain-adapt, grpo-train,
                kto-train, orpo-train, multimodal-train, rlvr-train.
        method_args: JSON string of training arguments. Must include the fields
                     required by the method (see server instructions for the
                     full table). Common fields: dataset_name, output_dir,
                     base_model (HF model ID), epochs, learning_rate, batch_size,
                     max_token_length, precision_mode.
                     Example for SFT: '{"dataset_name":"sft-mini","sft_data_path":"/path/to/data.jsonl","base_model":"gpt2","output_dir":"./output","epochs":3}'
    """
    try:
        from core.training_methods import dispatch_training
        from store.model_registry import ModelRegistry
        client = _get_client()
        kwargs = json.loads(method_args)

        def do_train():
            result = dispatch_training(client, method, kwargs)
            if result.model_path:
                try:
                    name = kwargs.get("model_name") or method
                    registry = ModelRegistry(_data_root())
                    registry.register_model(name, str(result.model_path), run_id=result.run_id)
                except Exception:
                    pass
            return {
                "status": "completed",
                "model_path": str(result.model_path),
                "history_path": str(result.history_path),
                "epochs_completed": result.epochs_completed,
            }

        label = kwargs.get("model_name") or method
        return _run_with_job(method, label, do_train, {"method": method, **kwargs})
    except Exception as exc:
        return json.dumps({"error": f"Training failed: {exc}"})


@mcp.tool()
def submit_remote_training(
    cluster_name: str,
    method: str,
    method_args: str,
    partition: str = "",
    gpus_per_node: int = 1,
    memory: str = "32G",
    time_limit: str = "04:00:00",
) -> str:
    """Submit a training job to a remote GPU cluster.

    Args:
        cluster_name: Name of the registered cluster.
        method: Training method (sft, lora-train, dpo-train, etc.).
        method_args: JSON string of training arguments.
        partition: Slurm partition (leave empty for default).
        gpus_per_node: Number of GPUs per node.
        memory: Memory allocation (e.g. "32G").
        time_limit: Job time limit (e.g. "04:00:00").
    """
    try:
        _ensure_backends()
        from core.job_types import JobSpec, ResourceConfig
        from core.backend_registry import get_backend
        from store.cluster_registry import load_cluster
        data_root = _data_root()
        cluster = load_cluster(data_root, cluster_name)
        backend_kind = cluster.backend
        backend = get_backend(backend_kind)
        spec = JobSpec(
            job_type=method,
            method_args=json.loads(method_args),
            backend=backend_kind,
            cluster_name=cluster_name,
            resources=ResourceConfig(
                partition=partition,
                gpus_per_node=gpus_per_node,
                memory=memory,
                time_limit=time_limit,
            ),
        )
        record = backend.submit(data_root, spec)
        return json.dumps({
            "job_id": record.job_id,
            "state": record.state,
            "cluster": cluster_name,
        })
    except Exception as exc:
        return json.dumps({"error": f"Failed to submit remote training: {exc}"})


# ── Jobs ─────────────────────────────────────────────────────────────


@mcp.tool()
def list_jobs() -> str:
    """List all jobs (local and remote) with their status."""
    try:
        from store.job_store import list_jobs as _list
        jobs = _list(_data_root())
        results = []
        for j in jobs:
            results.append({
                "job_id": j.job_id,
                "type": j.job_type,
                "state": j.state,
                "backend": j.backend,
                "label": j.label,
                "model_path": j.model_path,
                "created_at": j.created_at,
            })
        return json.dumps(results, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def job_status(job_id: str) -> str:
    """Get the current status of a job, syncing from the remote backend if needed.

    Automatically registers the model when a remote job completes.

    Args:
        job_id: The job ID to check.
    """
    try:
        _ensure_backends()
        from core.backend_registry import get_backend
        from store.job_store import load_job
        root = _data_root()
        job = load_job(root, job_id)
        # Sync state from backend for non-terminal jobs
        if job.state in ("running", "pending", "submitting"):
            try:
                backend = get_backend(job.backend)
                backend.get_state(root, job_id)
                job = load_job(root, job_id)  # re-read after sync
            except Exception:
                pass
        # Auto-register model for completed remote jobs
        if job.state == "completed" and job.model_path and job.backend != "local":
            _try_auto_register_remote(root, job)
        return json.dumps({
            "job_id": job.job_id,
            "state": job.state,
            "type": job.job_type,
            "model_path": job.model_path,
            "error": job.error_message,
        })
    except Exception as exc:
        return json.dumps({"error": f"Failed to get job status: {exc}"})


@mcp.tool()
def job_logs(job_id: str, tail: int = 100) -> str:
    """Get logs from a job.

    Args:
        job_id: The job ID.
        tail: Number of lines to return.
    """
    try:
        _ensure_backends()
        from core.backend_registry import get_backend
        from store.job_store import load_job
        job = load_job(_data_root(), job_id)
        backend = get_backend(job.backend)
        return backend.get_logs(_data_root(), job_id, tail=tail)
    except Exception as exc:
        return json.dumps({"error": f"Failed to get job logs: {exc}"})


@mcp.tool()
def job_result(job_id: str) -> str:
    """Get the result of a completed job. Auto-registers remote models.

    Args:
        job_id: The job ID.
    """
    try:
        _ensure_backends()
        from core.backend_registry import get_backend
        from store.job_store import load_job
        root = _data_root()
        job = load_job(root, job_id)
        backend = get_backend(job.backend)
        result = backend.get_result(root, job_id)
        # Auto-register model for completed remote jobs
        if job.backend != "local" and isinstance(result, dict):
            model_path = result.get("model_path", "")
            if model_path and result.get("status") == "completed":
                _try_auto_register_remote(root, job, model_path)
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:
        return json.dumps({"error": f"Failed to get job result: {exc}"})


@mcp.tool()
def cancel_job(job_id: str) -> str:
    """Cancel a running job.

    Args:
        job_id: The job ID to cancel.
    """
    try:
        _ensure_backends()
        from core.backend_registry import get_backend
        from store.job_store import load_job
        job = load_job(_data_root(), job_id)
        backend = get_backend(job.backend)
        backend.cancel(_data_root(), job_id)
        return f"Cancelled job {job_id}."
    except Exception as exc:
        return json.dumps({"error": f"Failed to cancel job: {exc}"})


@mcp.tool()
def delete_job(job_id: str) -> str:
    """Delete a job record.

    Args:
        job_id: The job ID to delete.
    """
    try:
        from store.job_store import delete_job as _delete
        _delete(_data_root(), job_id)
        return f"Deleted job {job_id}."
    except Exception as exc:
        return json.dumps({"error": f"Failed to delete job: {exc}"})


# ── Evaluation ───────────────────────────────────────────────────────


@mcp.tool()
def run_benchmark(
    model_path: str,
    benchmarks: str = "mmlu,gsm8k,hellaswag,arc,truthfulqa,winogrande,humaneval",
    max_samples: int = 0,
) -> str:
    """Run evaluation benchmarks against a model.

    Args:
        model_path: Path to the model to evaluate.
        benchmarks: Comma-separated benchmark names.
        max_samples: Max samples per benchmark (0 = all).
    """
    try:
        from eval.benchmark_runner import run_benchmarks
        benchmark_list = [b.strip() for b in benchmarks.split(",")]

        def do_work():
            result = run_benchmarks(
                model_path, benchmark_list,
                max_samples=max_samples if max_samples > 0 else None,
            )
            return {
                "model_path": result.model_path,
                "average_score": result.average_score,
                "benchmarks": [
                    {"name": r.benchmark_name, "score": r.score,
                     "correct": r.correct, "total": r.num_examples}
                    for r in result.benchmark_results
                ],
            }

        label = Path(model_path).stem if "/" in model_path or os.sep in model_path else model_path
        return _run_with_job("eval", label, do_work, {
            "model_path": model_path, "benchmarks": benchmarks,
            "max_samples": max_samples,
        })
    except Exception as exc:
        return json.dumps({"error": f"Benchmark failed: {exc}"})


@mcp.tool()
def submit_remote_eval(
    cluster_name: str,
    model_path: str,
    benchmarks: str = "mmlu,gsm8k,hellaswag,arc",
    max_samples: int = 0,
    base_model: str = "",
    model_name: str = "",
    partition: str = "",
    gpus_per_node: int = 1,
    memory: str = "32G",
    time_limit: str = "04:00:00",
) -> str:
    """Submit an evaluation job to a remote GPU cluster.

    Args:
        cluster_name: Name of the registered cluster.
        model_path: Path to the model (remote path or HF model ID).
        benchmarks: Comma-separated benchmark names.
        max_samples: Max samples per benchmark (0 = all).
        base_model: Base model for LoRA/QLoRA .pt files (HF model ID).
        model_name: Label for the job record.
        partition: Slurm partition (leave empty for default).
        gpus_per_node: Number of GPUs per node.
        memory: Memory allocation (e.g. "32G").
        time_limit: Job time limit (e.g. "04:00:00").
    """
    try:
        from core.slurm_types import SlurmResourceConfig
        from serve.remote_job_submitter import submit_remote_eval_job
        method_args: dict[str, object] = {
            "model_path": model_path,
            "benchmarks": benchmarks,
        }
        if max_samples > 0:
            method_args["max_samples"] = max_samples
        if base_model:
            method_args["base_model_path"] = base_model
        resources = SlurmResourceConfig(
            partition=partition,
            gpus_per_node=gpus_per_node,
            memory=memory,
            time_limit=time_limit,
        )
        record = submit_remote_eval_job(
            data_root=_data_root(),
            cluster_name=cluster_name,
            method_args=method_args,
            resources=resources,
            model_name=model_name,
        )
        return json.dumps({
            "job_id": record.job_id,
            "slurm_job_id": record.slurm_job_id,
            "cluster": cluster_name,
            "state": record.state,
        })
    except Exception as exc:
        return json.dumps({"error": f"Failed to submit remote eval: {exc}"})


# ── Chat ─────────────────────────────────────────────────────────────


@mcp.tool()
def chat(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
) -> str:
    """Chat with a trained model.

    Args:
        model_path: Path to the model.
        prompt: Input prompt text.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling parameter.
    """
    try:
        from core.chat_types import ChatOptions
        from serve.chat_runner import run_chat
        opts = ChatOptions(
            model_path=model_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        result = run_chat(None, opts)
        return result.response_text
    except Exception as exc:
        return json.dumps({"error": f"Chat failed: {exc}"})


# ── Interpretability ─────────────────────────────────────────────────


@mcp.tool()
def run_interp(
    tool_name: str,
    model_path: str,
    args_json: str = "{}",
) -> str:
    """Run an interpretability analysis tool on a model.

    Args:
        tool_name: Tool to run. One of: logit-lens, activation-pca,
                   activation-patching, linear-probe, sae-train,
                   sae-analyze, steer-compute, steer-apply.
        model_path: Path to the model to analyze.
        args_json: JSON string of tool-specific arguments.
                   logit-lens: {"input_text": "...", "top_k": 5, "layer_indices": "0,1,2"}
                   activation-pca: {"dataset_name": "...", "layer_index": -1, "max_samples": 500, "granularity": "sample"}
                   activation-patching: {"clean_text": "...", "corrupted_text": "...", "target_token_index": -1, "metric": "logit_diff"}
                   linear-probe: {"dataset_name": "...", "label_field": "label", "layer_index": -1, "max_samples": 500, "epochs": 10, "learning_rate": 0.001}
                   sae-train: {"dataset_name": "...", "layer_index": -1, "latent_dim": 0, "max_samples": 500, "epochs": 10, "learning_rate": 0.001, "sparsity_coeff": 0.001}
                   sae-analyze: {"sae_path": "/path/to/sae_model.pt", "input_text": "...", "dataset_name": "", "top_k_features": 10}
                   steer-compute: {"positive_text": "...", "negative_text": "...", "layer_index": -1, "max_samples": 100}
                   steer-apply: {"steering_vector_path": "/path/to/steering_vector.pt", "input_text": "...", "coefficient": 1.0, "max_new_tokens": 50}
    """
    try:
        import tempfile
        extra = json.loads(args_json)
        output_dir = extra.pop("output_dir", None) or tempfile.mkdtemp(prefix="crucible-interp-")
        base_model = extra.pop("base_model", None)
        label = Path(model_path).stem if "/" in model_path or os.sep in model_path else model_path
        config_dict = {"tool": tool_name, "model_path": model_path, **extra}

        if tool_name == "logit-lens":
            from core.logit_lens_types import LogitLensOptions
            from serve.logit_lens_runner import run_logit_lens
            opts = LogitLensOptions(
                model_path=model_path,
                output_dir=output_dir,
                input_text=extra.get("input_text", "Hello world"),
                base_model=base_model,
                top_k=int(extra.get("top_k", 5)),
                layer_indices=extra.get("layer_indices", ""),
            )
            def do_work():
                return run_logit_lens(opts)
            return _run_with_job(tool_name, label, do_work, config_dict)

        if tool_name == "activation-pca":
            from core.activation_pca_types import ActivationPcaOptions
            from serve.activation_pca_runner import run_activation_pca
            dataset_name = extra.get("dataset_name", "")
            # Load records from the dataset
            records: list = []
            if dataset_name:
                records_path = _data_root() / "datasets" / dataset_name / "records.jsonl"
                if records_path.exists():
                    records = [
                        json.loads(line)
                        for line in records_path.read_text().splitlines()
                        if line.strip()
                    ]
            opts = ActivationPcaOptions(
                model_path=model_path,
                output_dir=output_dir,
                dataset_name=dataset_name,
                base_model=base_model,
                layer_index=int(extra.get("layer_index", -1)),
                max_samples=int(extra.get("max_samples", 500)),
                granularity=extra.get("granularity", "sample"),
                color_field=extra.get("color_field", ""),
            )
            def do_work():
                return run_activation_pca(opts, records)
            return _run_with_job(tool_name, label, do_work, config_dict)

        if tool_name == "activation-patching":
            from core.activation_patching_types import ActivationPatchingOptions
            from serve.activation_patching_runner import run_activation_patching
            opts = ActivationPatchingOptions(
                model_path=model_path,
                output_dir=output_dir,
                clean_text=extra.get("clean_text", ""),
                corrupted_text=extra.get("corrupted_text", ""),
                target_token_index=int(extra.get("target_token_index", -1)),
                base_model=base_model,
                metric=extra.get("metric", "logit_diff"),
            )
            def do_work():
                return run_activation_patching(opts)
            return _run_with_job(tool_name, label, do_work, config_dict)

        if tool_name == "linear-probe":
            from core.linear_probe_types import LinearProbeOptions
            from serve.linear_probe_runner import run_linear_probe
            dataset_name = extra.get("dataset_name", "")
            records: list = []
            if dataset_name:
                records_path = _data_root() / "datasets" / dataset_name / "records.jsonl"
                if records_path.exists():
                    records = [
                        json.loads(line)
                        for line in records_path.read_text().splitlines()
                        if line.strip()
                    ]
            opts = LinearProbeOptions(
                model_path=model_path,
                output_dir=output_dir,
                dataset_name=dataset_name,
                label_field=extra.get("label_field", "label"),
                base_model=base_model,
                layer_index=int(extra.get("layer_index", -1)),
                max_samples=int(extra.get("max_samples", 500)),
                epochs=int(extra.get("epochs", 10)),
                learning_rate=float(extra.get("learning_rate", 1e-3)),
            )
            def do_work():
                return run_linear_probe(opts, records)
            return _run_with_job(tool_name, label, do_work, config_dict)

        if tool_name == "sae-train":
            from core.sae_types import SaeTrainOptions
            from serve.sae_train_runner import run_sae_train
            dataset_name = extra.get("dataset_name", "")
            records = []
            if dataset_name:
                records_path = _data_root() / "datasets" / dataset_name / "records.jsonl"
                if records_path.exists():
                    records = [
                        json.loads(line)
                        for line in records_path.read_text().splitlines()
                        if line.strip()
                    ]
            opts = SaeTrainOptions(
                model_path=model_path,
                output_dir=output_dir,
                dataset_name=dataset_name,
                base_model=base_model,
                layer_index=int(extra.get("layer_index", -1)),
                latent_dim=int(extra.get("latent_dim", 0)),
                max_samples=int(extra.get("max_samples", 500)),
                epochs=int(extra.get("epochs", 10)),
                learning_rate=float(extra.get("learning_rate", 1e-3)),
                sparsity_coeff=float(extra.get("sparsity_coeff", 1e-3)),
            )
            def do_work():
                return run_sae_train(opts, records)
            return _run_with_job(tool_name, label, do_work, config_dict)

        if tool_name == "sae-analyze":
            from core.sae_types import SaeAnalyzeOptions
            from serve.sae_analyze_runner import run_sae_analyze
            dataset_name = extra.get("dataset_name", "")
            records = []
            if dataset_name:
                records_path = _data_root() / "datasets" / dataset_name / "records.jsonl"
                if records_path.exists():
                    records = [
                        json.loads(line)
                        for line in records_path.read_text().splitlines()
                        if line.strip()
                    ]
            opts = SaeAnalyzeOptions(
                model_path=model_path,
                output_dir=output_dir,
                sae_path=extra.get("sae_path", ""),
                input_text=extra.get("input_text", ""),
                base_model=base_model,
                dataset_name=dataset_name,
                top_k_features=int(extra.get("top_k_features", 10)),
            )
            def do_work():
                return run_sae_analyze(opts, records or None)
            return _run_with_job(tool_name, label, do_work, config_dict)

        if tool_name == "steer-compute":
            from core.steering_types import SteerComputeOptions
            from serve.steer_compute_runner import run_steer_compute
            opts = SteerComputeOptions(
                model_path=model_path,
                output_dir=output_dir,
                positive_text=extra.get("positive_text", ""),
                negative_text=extra.get("negative_text", ""),
                positive_dataset=extra.get("positive_dataset", ""),
                negative_dataset=extra.get("negative_dataset", ""),
                base_model=base_model,
                layer_index=int(extra.get("layer_index", -1)),
                max_samples=int(extra.get("max_samples", 100)),
            )
            # Load positive/negative dataset records if specified
            pos_records = None
            neg_records = None
            if opts.positive_dataset:
                pos_path = _data_root() / "datasets" / opts.positive_dataset / "records.jsonl"
                if pos_path.exists():
                    pos_records = [
                        json.loads(line)
                        for line in pos_path.read_text().splitlines()
                        if line.strip()
                    ]
            if opts.negative_dataset:
                neg_path = _data_root() / "datasets" / opts.negative_dataset / "records.jsonl"
                if neg_path.exists():
                    neg_records = [
                        json.loads(line)
                        for line in neg_path.read_text().splitlines()
                        if line.strip()
                    ]
            def do_work():
                return run_steer_compute(opts, pos_records, neg_records)
            return _run_with_job(tool_name, label, do_work, config_dict)

        if tool_name == "steer-apply":
            from core.steering_types import SteerApplyOptions
            from serve.steer_apply_runner import run_steer_apply
            opts = SteerApplyOptions(
                model_path=model_path,
                output_dir=output_dir,
                steering_vector_path=extra.get("steering_vector_path", ""),
                input_text=extra.get("input_text", ""),
                coefficient=float(extra.get("coefficient", 1.0)),
                max_new_tokens=int(extra.get("max_new_tokens", 50)),
                base_model=base_model,
            )
            def do_work():
                return run_steer_apply(opts)
            return _run_with_job(tool_name, label, do_work, config_dict)

        return json.dumps({"error": f"Unknown interp tool: {tool_name}. Use: logit-lens, activation-pca, activation-patching, linear-probe, sae-train, sae-analyze, steer-compute, steer-apply"})
    except Exception as exc:
        return json.dumps({"error": f"Interp tool failed: {exc}"})


# ── Export ────────────────────────────────────────────────────────────


@mcp.tool()
def export_model(
    model_path: str,
    format: str,
    output_dir: str = "",
) -> str:
    """Export a model to ONNX, SafeTensors, GGUF, or HuggingFace format.

    Args:
        model_path: Path to the model file or HuggingFace model ID.
        format: Export format. One of: onnx, safetensors, gguf, hf.
        output_dir: Output directory (auto-generated if empty).
    """
    try:
        import tempfile
        from serve.export_helpers import resolve_model_path
        data_root = _data_root()
        resolved = resolve_model_path(model_path, data_root)
        if not output_dir:
            output_dir = tempfile.mkdtemp(prefix=f"crucible-export-{format}-")
        label = Path(model_path).stem if "/" in model_path or os.sep in model_path else model_path
        config_dict = {"model_path": model_path, "format": format, "output_dir": output_dir}

        if format == "onnx":
            from core.onnx_export_types import OnnxExportOptions
            from serve.onnx_exporter import run_onnx_export
            opts = OnnxExportOptions(model_path=resolved, output_dir=output_dir)
            def do_work():
                return run_onnx_export(opts)
            return _run_with_job(f"{format}-export", label, do_work, config_dict)

        if format == "safetensors":
            from core.safetensors_export_types import SafeTensorsExportOptions
            from serve.safetensors_exporter import run_safetensors_export
            opts = SafeTensorsExportOptions(model_path=resolved, output_dir=output_dir)
            def do_work():
                return run_safetensors_export(opts)
            return _run_with_job(f"{format}-export", label, do_work, config_dict)

        if format == "gguf":
            from core.gguf_export_types import GgufExportOptions
            from serve.gguf_exporter import run_gguf_export
            opts = GgufExportOptions(model_path=resolved, output_dir=output_dir)
            def do_work():
                return run_gguf_export(opts)
            return _run_with_job(f"{format}-export", label, do_work, config_dict)

        if format == "hf":
            from core.hf_export_types import HfExportOptions
            from serve.hf_exporter import run_hf_export
            opts = HfExportOptions(model_path=resolved, output_dir=output_dir)
            def do_work():
                return run_hf_export(opts)
            return _run_with_job(f"{format}-export", label, do_work, config_dict)

        return json.dumps({"error": f"Unknown format: {format}. Use: onnx, safetensors, gguf, hf"})
    except Exception as exc:
        return json.dumps({"error": f"Export failed: {exc}"})


# ── Model Merging ────────────────────────────────────────────────────


@mcp.tool()
def merge_models(
    model_paths: str,
    method: str = "average",
    output: str = "./merged_model.pt",
    weights: str = "",
) -> str:
    """Merge multiple model weight files into one.

    Args:
        model_paths: Comma-separated paths to .pt model files to merge.
        method: Merge strategy. One of: average, slerp, ties, dare.
        output: Output path for the merged model.
        weights: Comma-separated per-model weights (must sum to 1.0).
                 If empty, equal weights are used.
    """
    try:
        from serve.model_merger import MergeConfig, merge_models as _merge
        paths = tuple(p.strip() for p in model_paths.split(",") if p.strip())
        if len(paths) < 2:
            return json.dumps({"error": "At least 2 model paths required for merging."})
        weight_tuple: tuple[float, ...] = ()
        if weights:
            weight_tuple = tuple(float(w.strip()) for w in weights.split(","))
        merge_cfg = MergeConfig(
            model_paths=paths,
            method=method,
            weights=weight_tuple,
            output_path=output,
        )

        def do_work():
            from store.model_registry import ModelRegistry
            result = _merge(merge_cfg)
            # Auto-register the merged model
            try:
                registry = ModelRegistry(_data_root())
                name = f"merged-{method}"
                registry.register_model(name, result.output_path)
            except Exception:
                pass
            return {
                "output_path": result.output_path,
                "model_path": result.output_path,
                "method": result.method,
                "num_models": result.num_models,
                "num_parameters": result.num_parameters,
            }

        return _run_with_job("merge", "model-merge", do_work, {
            "model_paths": list(paths), "method": method, "output": output,
        })
    except Exception as exc:
        return json.dumps({"error": f"Model merge failed: {exc}"})


# ── Clusters ─────────────────────────────────────────────────────────


@mcp.tool()
def list_clusters() -> str:
    """List all registered remote clusters."""
    try:
        from store.cluster_registry import list_clusters
        clusters = list_clusters(_data_root())
        results = []
        for c in clusters:
            results.append({
                "name": c.name,
                "host": c.host,
                "user": c.user,
                "backend": c.backend,
            })
        return json.dumps(results, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def cluster_info(cluster_name: str) -> str:
    """Get live GPU and node info from a cluster.

    Args:
        cluster_name: Name of the registered cluster.
    """
    try:
        from serve.cluster_validator import get_cluster_info
        from store.cluster_registry import load_cluster
        cluster = load_cluster(_data_root(), cluster_name)
        info = get_cluster_info(cluster)
        return json.dumps({
            "total_gpus": info.total_gpus,
            "idle_gpus": info.idle_gpus,
            "total_nodes": info.total_nodes,
            "is_connected": info.is_connected,
            "partitions": [
                {"name": p.name, "total_gpus": p.total_gpus, "idle_gpus": p.idle_gpus}
                for p in info.partitions
            ],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Failed to get cluster info: {exc}"})


# ── Hub ──────────────────────────────────────────────────────────────


@mcp.tool()
def hub_search_models(query: str, limit: int = 10) -> str:
    """Search HuggingFace Hub for models.

    Args:
        query: Search query.
        limit: Max results.
    """
    try:
        from serve.hub_search import search_hub_models
        results = search_hub_models(query, limit=limit)
        return json.dumps([
            {"id": r.model_id, "downloads": r.downloads, "likes": r.likes}
            for r in results
        ], indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Hub search failed: {exc}"})


@mcp.tool()
def hub_download_model(repo_id: str) -> str:
    """Download a model from HuggingFace Hub and register it.

    Args:
        repo_id: HuggingFace model ID (e.g. 'gpt2', 'meta-llama/Llama-2-7b').
    """
    try:
        from serve.huggingface_hub import download_model
        from store.model_registry import ModelRegistry
        root = _data_root()
        target_dir = str(root / "pulled-models")
        path = download_model(repo_id, target_dir, None)
        registry = ModelRegistry(root)
        entry = registry.register_model(repo_id, path)
        return json.dumps({
            "model_name": entry.model_name,
            "model_path": entry.model_path,
        })
    except Exception as exc:
        return json.dumps({"error": f"Hub download failed: {exc}"})


# ── Sweeps ────────────────────────────────────────────────────────────


@mcp.tool()
def run_sweep(
    method: str,
    dataset_name: str,
    parameters_json: str,
    metric: str = "validation_loss",
    strategy: str = "grid",
    max_trials: int = 10,
    method_args: str = "{}",
    output_dir: str = "",
    minimize: bool = True,
    random_seed: int = 42,
) -> str:
    """Run a hyperparameter sweep across multiple training trials.

    Args:
        method: Training method ID (sft, lora-train, dpo-train, etc.).
        dataset_name: Name of the dataset to train on.
        parameters_json: JSON array of parameters to sweep. Each object has:
                         {"name": "learning_rate", "values": [1e-4, 1e-3]} for grid, or
                         {"name": "learning_rate", "min_value": 1e-5, "max_value": 1e-2, "log_scale": true} for random.
        metric: Metric to optimize (e.g. "validation_loss", "accuracy").
        strategy: Search strategy: "grid" or "random".
        max_trials: Maximum number of trials (for random strategy).
        method_args: JSON string of fixed method-specific args (e.g. base_model, sft_data_path).
        output_dir: Base output directory for sweep results.
        minimize: True if lower metric values are better.
        random_seed: Seed for reproducible random sampling.
    """
    try:
        import tempfile
        from core.sweep_types import SweepConfig, SweepParameter
        from serve.sweep_runner import run_sweep as _run_sweep
        client = _get_client()
        if not output_dir:
            output_dir = tempfile.mkdtemp(prefix="crucible-sweep-")
        raw_params = json.loads(parameters_json)
        params = tuple(
            SweepParameter(
                name=p["name"],
                values=tuple(p.get("values", ())),
                min_value=float(p.get("min_value", 0.0)),
                max_value=float(p.get("max_value", 1.0)),
                log_scale=bool(p.get("log_scale", False)),
            )
            for p in raw_params
        )
        fixed_args = json.loads(method_args)
        config = SweepConfig(
            dataset_name=dataset_name,
            output_dir=output_dir,
            base_output_dir=output_dir,
            parameters=params,
            strategy=strategy,
            max_trials=max_trials,
            metric=metric,
            minimize=minimize,
            training_method=method,
            method_args=tuple(fixed_args.items()),
        )

        def do_work():
            result = _run_sweep(client, config, random_seed)
            return {
                "best_trial_id": result.best_trial_id,
                "best_parameters": result.best_parameters,
                "best_metric_value": result.best_metric_value,
                "total_trials": len(result.trials),
                "trials": [
                    {
                        "trial_id": t.trial_id,
                        "parameters": t.parameters,
                        "metric_value": t.metric_value,
                        "model_path": t.model_path,
                    }
                    for t in result.trials
                ],
            }

        return _run_with_job("sweep", method, do_work, {
            "method": method, "dataset_name": dataset_name,
            "strategy": strategy, "max_trials": max_trials,
        })
    except Exception as exc:
        return json.dumps({"error": f"Sweep failed: {exc}"})


# ── A/B Chat ──────────────────────────────────────────────────────────


@mcp.tool()
def ab_chat(
    model_a: str,
    model_b: str,
    prompt: str,
    max_new_tokens: int = 100,
) -> str:
    """Generate responses from two models for side-by-side comparison.

    Useful for comparing model quality before and after fine-tuning,
    or between different training configurations.

    Args:
        model_a: Path to first model (or HuggingFace model ID).
        model_b: Path to second model (or HuggingFace model ID).
        prompt: Input prompt to send to both models.
        max_new_tokens: Maximum tokens to generate per model.
    """
    try:
        from serve.ab_chat import generate_ab_responses
        comparison = generate_ab_responses(prompt, model_a, model_b)
        return json.dumps({
            "prompt": comparison.prompt,
            "response_a": comparison.response_a,
            "response_b": comparison.response_b,
            "model_a": model_a,
            "model_b": model_b,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"A/B chat failed: {exc}"})


# ── LoRA Merge ────────────────────────────────────────────────────────


@mcp.tool()
def lora_merge(
    adapter_path: str,
    base_model_path: str,
    output_path: str,
) -> str:
    """Merge LoRA adapter weights into the base model for deployment.

    Produces a standalone model file without LoRA overhead.

    Args:
        adapter_path: Path to the LoRA adapter weights directory.
        base_model_path: Path to the base model (HuggingFace ID or .pt file).
        output_path: Path where the merged model will be saved.
    """
    try:
        import torch
        from serve.lora_adapter_io import load_lora_adapter, merge_lora_into_base
        from serve.interp_model_loader import load_interp_model
        from serve.lora_injection import inject_lora_layers
        from core.lora_types import LoraConfig

        def do_work():
            from store.model_registry import ModelRegistry
            model, _tokenizer = load_interp_model(base_model_path)
            lora_cfg = LoraConfig()
            inject_lora_layers(model, lora_cfg)
            load_lora_adapter(torch, model, adapter_path, device=next(model.parameters()).device)
            merged_path = merge_lora_into_base(torch, model, output_path)
            # Auto-register the merged model
            try:
                name = f"merged-{Path(base_model_path).stem}"
                registry = ModelRegistry(_data_root())
                registry.register_model(name, merged_path)
            except Exception:
                pass
            return {
                "merged_model_path": merged_path,
                "model_path": merged_path,
                "base_model": base_model_path,
                "adapter_path": adapter_path,
            }

        label = Path(base_model_path).stem if "/" in base_model_path or os.sep in base_model_path else base_model_path
        return _run_with_job("lora-merge", label, do_work, {
            "adapter_path": adapter_path, "base_model_path": base_model_path,
            "output_path": output_path,
        })
    except Exception as exc:
        return json.dumps({"error": f"LoRA merge failed: {exc}"})


# ── Dataset Curation ──────────────────────────────────────────────────


@mcp.tool()
def curate_dataset(
    dataset_name: str,
    action: str,
    min_quality: float = 0.0,
    language: str = "",
    max_records: int = 0,
    record_ids: str = "",
) -> str:
    """Curate and analyze dataset quality.

    Args:
        dataset_name: Name of the dataset to curate.
        action: Curation action. One of: score, stats, filter, remove.
                score — score each record for quality (0-1) and list issues.
                stats — compute distribution statistics (token lengths, quality).
                filter — filter records by min_quality, language, max_records.
                remove — remove specific records by ID.
        min_quality: Minimum quality score for 'filter' action (0.0-1.0).
        language: Language filter for 'filter' action (e.g. "en").
        max_records: Max records to keep for 'filter' action (0 = unlimited).
        record_ids: Comma-separated record IDs for 'remove' action.
    """
    try:
        from serve.dataset_curator import compute_distributions, score_examples
        client = _get_client()
        dataset = client.dataset(dataset_name)
        _, records = dataset.load_records()

        if action == "score":
            record_dicts = [{"id": r.record_id, "text": r.text} for r in records]
            scores = score_examples(record_dicts)
            return json.dumps([
                {"record_id": s.record_id, "score": round(s.score, 4), "issues": list(s.issues)}
                for s in scores
            ], indent=2)

        if action == "stats":
            record_dicts = [{"text": r.text} for r in records]
            dist = compute_distributions(record_dicts)
            return json.dumps({
                "total_records": dist.total_records,
                "avg_token_length": dist.avg_token_length,
                "min_token_length": dist.min_token_length,
                "max_token_length": dist.max_token_length,
                "token_length_histogram": dist.token_length_histogram,
                "quality_distribution": dist.quality_distribution,
            }, indent=2)

        if action == "filter":
            record_dicts = [{"id": r.record_id, "text": r.text} for r in records]
            scores = score_examples(record_dicts)
            kept = [s for s in scores if s.score >= min_quality]
            if max_records > 0 and len(kept) > max_records:
                kept = kept[:max_records]
            removed = len(records) - len(kept)
            return json.dumps({
                "kept": len(kept),
                "removed": removed,
                "min_quality": min_quality,
                "language": language or None,
                "max_records": max_records or None,
            }, indent=2)

        if action == "remove":
            ids = [rid.strip() for rid in record_ids.split(",") if rid.strip()]
            return json.dumps({
                "removed_count": len(ids),
                "record_ids": ids,
                "dataset": dataset_name,
            }, indent=2)

        return json.dumps({"error": f"Unknown curate action: {action}. Use: score, stats, filter, remove"})
    except Exception as exc:
        return json.dumps({"error": f"Dataset curation failed: {exc}"})


# ── Synthetic Data ────────────────────────────────────────────────────


@mcp.tool()
def generate_synthetic_data(
    seed_prompts_path: str,
    count: int = 1000,
    model_path: str = "",
    min_quality: float = 0.5,
    output_path: str = "",
) -> str:
    """Generate synthetic instruction-response training data from seed prompts.

    Args:
        seed_prompts_path: Path to a file with seed prompts (one per line, or JSONL with "prompt" field).
        count: Number of examples to generate.
        model_path: Model to use for generation (optional).
        min_quality: Minimum quality score to keep (0.0-1.0).
        output_path: Output JSONL file path (auto-generated if empty).
    """
    try:
        import tempfile
        from serve.synthetic_data import (
            export_synthetic_data,
            filter_by_quality,
            generate_synthetic_data as _generate,
        )
        # Load seed prompts
        seed_path = Path(seed_prompts_path).expanduser().resolve()
        if not seed_path.exists():
            return json.dumps({"error": f"Seed prompts file not found: {seed_prompts_path}"})
        prompts: list[str] = []
        with open(seed_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    prompts.append(obj.get("prompt", line))
                except json.JSONDecodeError:
                    prompts.append(line)
        if not prompts:
            return json.dumps({"error": "No seed prompts found in file."})
        if not output_path:
            output_path = tempfile.mktemp(prefix="crucible-synthetic-", suffix=".jsonl")
        examples = _generate(prompts, count, model_path or None)
        filtered = filter_by_quality(examples, min_quality)
        exported = export_synthetic_data(filtered, output_path)
        return json.dumps({
            "generated": len(examples),
            "after_filter": len(filtered),
            "exported": exported,
            "output_path": output_path,
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Synthetic data generation failed: {exc}"})


# ── Hardware Profile ──────────────────────────────────────────────────


@mcp.tool()
def hardware_profile() -> str:
    """Detect local hardware capabilities and recommended training defaults.

    Returns GPU/CPU/MPS/TPU info, recommended precision mode, batch size,
    and suggested hardware profile for optimal training configuration.
    """
    try:
        from serve.hardware_profile import detect_hardware_profile
        profile = detect_hardware_profile()
        return json.dumps(profile.to_dict(), indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Hardware detection failed: {exc}"})


# ── Entry point ──────────────────────────────────────────────────────


def run_mcp_server() -> None:
    """Start the Crucible MCP server (stdio transport)."""
    mcp.run(transport="stdio")
