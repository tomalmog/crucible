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

**6. Export a model:**
1. `export_model(model_path, "onnx", "./exports")` -- export to ONNX/safetensors/gguf/hf

**7. Merge models:**
1. `merge_models(["model1.pt", "model2.pt"], "average", "./merged")` -- average/slerp/ties/dare

**8. Pull a trained model from remote:**
1. `pull_model(job_id, "my-model")` -- download + register

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
        result = dispatch_training(client, method, kwargs)
        # Auto-register the trained model
        if result.model_path:
            try:
                name = kwargs.get("model_name") or method
                registry = ModelRegistry(_data_root())
                registry.register_model(name, str(result.model_path), run_id=result.run_id)
            except Exception:
                pass
        return json.dumps({
            "status": "completed",
            "model_path": str(result.model_path),
            "history_path": str(result.history_path),
            "epochs_completed": result.epochs_completed,
        })
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
    """Get the current status of a job.

    Args:
        job_id: The job ID to check.
    """
    try:
        _ensure_backends()
        from store.job_store import load_job
        job = load_job(_data_root(), job_id)
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
    """Get the result of a completed job.

    Args:
        job_id: The job ID.
    """
    try:
        _ensure_backends()
        from core.backend_registry import get_backend
        from store.job_store import load_job
        job = load_job(_data_root(), job_id)
        backend = get_backend(job.backend)
        result = backend.get_result(_data_root(), job_id)
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
        result = run_benchmarks(
            model_path, benchmark_list,
            max_samples=max_samples if max_samples > 0 else None,
        )
        return json.dumps({
            "model_path": result.model_path,
            "average_score": result.average_score,
            "benchmarks": [
                {"name": r.benchmark_name, "score": r.score,
                 "correct": r.correct, "total": r.num_examples}
                for r in result.benchmark_results
            ],
        }, indent=2)
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
                   activation-patching.
        model_path: Path to the model to analyze.
        args_json: JSON string of tool-specific arguments.
                   logit-lens: {"input_text": "...", "top_k": 5, "layer_indices": "0,1,2"}
                   activation-pca: {"dataset_name": "...", "layer_index": -1, "max_samples": 500, "granularity": "sample"}
                   activation-patching: {"clean_text": "...", "corrupted_text": "...", "target_token_index": -1, "metric": "logit_diff"}
    """
    try:
        import tempfile
        extra = json.loads(args_json)
        output_dir = extra.pop("output_dir", None) or tempfile.mkdtemp(prefix="crucible-interp-")
        base_model = extra.pop("base_model", None)

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
            result = run_logit_lens(opts)
            return json.dumps(result, indent=2, default=str)

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
            result = run_activation_pca(opts, records)
            return json.dumps(result, indent=2, default=str)

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
            result = run_activation_patching(opts)
            return json.dumps(result, indent=2, default=str)

        return json.dumps({"error": f"Unknown interp tool: {tool_name}. Use: logit-lens, activation-pca, activation-patching"})
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

        if format == "onnx":
            from core.onnx_export_types import OnnxExportOptions
            from serve.onnx_exporter import run_onnx_export
            opts = OnnxExportOptions(model_path=resolved, output_dir=output_dir)
            result = run_onnx_export(opts)
            return json.dumps(result, indent=2, default=str)

        if format == "safetensors":
            from core.safetensors_export_types import SafeTensorsExportOptions
            from serve.safetensors_exporter import run_safetensors_export
            opts = SafeTensorsExportOptions(model_path=resolved, output_dir=output_dir)
            result = run_safetensors_export(opts)
            return json.dumps(result, indent=2, default=str)

        if format == "gguf":
            from core.gguf_export_types import GgufExportOptions
            from serve.gguf_exporter import run_gguf_export
            opts = GgufExportOptions(model_path=resolved, output_dir=output_dir)
            result = run_gguf_export(opts)
            return json.dumps(result, indent=2, default=str)

        if format == "hf":
            from core.hf_export_types import HfExportOptions
            from serve.hf_exporter import run_hf_export
            opts = HfExportOptions(model_path=resolved, output_dir=output_dir)
            result = run_hf_export(opts)
            return json.dumps(result, indent=2, default=str)

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
        config = MergeConfig(
            model_paths=paths,
            method=method,
            weights=weight_tuple,
            output_path=output,
        )
        result = _merge(config)
        return json.dumps({
            "output_path": result.output_path,
            "method": result.method,
            "num_models": result.num_models,
            "num_parameters": result.num_parameters,
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
        from serve.hub_download import download_and_register_model
        entry = download_and_register_model(_data_root(), repo_id)
        return f"Downloaded and registered '{entry.model_name}' at {entry.model_path}."
    except Exception as exc:
        return json.dumps({"error": f"Hub download failed: {exc}"})


# ── Entry point ──────────────────────────────────────────────────────


def run_mcp_server() -> None:
    """Start the Crucible MCP server (stdio transport)."""
    mcp.run(transport="stdio")
