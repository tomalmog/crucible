"""String constant containing the remote agent entry point script.

This is embedded into the agent tarball as crucible_agent_entry.py
and executed on the remote Slurm node.
"""

ENTRY_SCRIPT = '''"""Crucible remote agent entry point.

Reads a JSON training config, sets up sys.path, and dispatches training.
"""

import os
import signal
import sys

# Immediate output to confirm Python started
print("CRUCIBLE_AGENT: Python process started", flush=True)

import argparse
import json
import resource


def _log_memory(label: str) -> None:
    """Print current RSS memory usage."""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KB
    if sys.platform == "darwin":
        rss_mb = rss_kb / 1024 / 1024
    else:
        rss_mb = rss_kb / 1024
    print(f"CRUCIBLE_AGENT: [{label}] RSS={rss_mb:.0f}MB", flush=True)


def _signal_handler(signum, frame):
    """Log signal before exit (catches SIGTERM from Slurm)."""
    name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    print(f"CRUCIBLE_AGENT_ERROR: Received signal {name} ({signum})", flush=True)
    _log_memory("on-signal")
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGUSR1, _signal_handler)

# Add the extracted bundle directory to sys.path
bundle_dir = os.path.dirname(os.path.abspath(__file__))
if bundle_dir not in sys.path:
    sys.path.insert(0, bundle_dir)

# Methods that require dataset records via the snapshot store
_RECORD_BASED_METHODS = frozenset({"train", "distill", "domain-adapt"})


def _ensure_output_dir(method_args: dict) -> None:
    """Set output_dir to ./output if not already specified."""
    if not method_args.get("output_dir"):
        out = os.path.join(os.getcwd(), "output")
        os.makedirs(out, exist_ok=True)
        method_args["output_dir"] = out


def _ensure_dataset_name(method_args: dict, method: str) -> None:
    """Set dataset_name appropriately for the training method.

    Record-based methods (train, distill, domain-adapt) need a real
    dataset_name for snapshot store lookups.  All other methods need
    dataset_name present (it is a required dataclass field) but set
    to empty string so the SDK uses the data path field directly
    instead of attempting a dataset store lookup.
    """
    if method in _RECORD_BASED_METHODS:
        if not method_args.get("dataset_name"):
            method_args["dataset_name"] = "remote-dataset"
    else:
        method_args["dataset_name"] = ""


def _hydrate_nested_configs(method_args: dict, method: str) -> None:
    """Convert nested dict values to proper dataclass instances.

    JSON serialization flattens dataclass objects to dicts. This
    reconstructs them so dispatch_training can build the Options.
    Currently only RLHF has nested configs (reward_config, ppo_config).
    """
    if method == "rlhf-train":
        from core.rlhf_types import PpoConfig, RewardModelConfig
        rc = method_args.get("reward_config")
        if isinstance(rc, dict):
            method_args["reward_config"] = RewardModelConfig(**rc)
        pc = method_args.get("ppo_config")
        if isinstance(pc, dict):
            method_args["ppo_config"] = PpoConfig(**pc)


def _ingest_raw_data(client, method_args: dict) -> None:
    """Ingest a raw JSONL file into a crucible dataset for record-based methods.

    If ``raw_data_path`` is present in *method_args*, ingest it as a
    temporary dataset so methods like train/distill/domain-adapt can
    load records from the snapshot store.
    """
    raw_path = method_args.pop("raw_data_path", None)
    if not raw_path:
        return
    from core.ingest_types import IngestOptions
    ds_name = method_args.get("dataset_name", "remote-dataset")
    opts = IngestOptions(dataset_name=ds_name, source_uri=raw_path)
    print(f"CRUCIBLE_AGENT: Ingesting {raw_path} as dataset '{ds_name}'...")
    client.ingest(opts)
    method_args["dataset_name"] = ds_name
    print(f"CRUCIBLE_AGENT: Ingestion complete.")


class _SimpleRecord:
    """Lightweight record for reading raw JSONL when the dataset store
    is not available (e.g. on a remote cluster)."""
    __slots__ = ("content", "metadata")

    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}


def _read_data_as_records(path: str) -> list:
    """Read a JSONL or Parquet file and convert to simple record objects."""
    if path.endswith(".parquet"):
        return _read_parquet_as_records(path)
    return _read_jsonl_as_records(path)


def _read_parquet_as_records(path: str) -> list:
    """Read a Parquet file and convert to simple record objects."""
    _TEXT_KEYS = ("text", "input", "content", "prompt", "instruction")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("CRUCIBLE_AGENT: pyarrow not available, trying pandas...", flush=True)
        import pandas as pd
        df = pd.read_parquet(path)
        rows = df.to_dict(orient="records")
        return _rows_to_records(rows, _TEXT_KEYS)
    table = pq.read_table(path)
    rows = table.to_pylist()
    return _rows_to_records(rows, _TEXT_KEYS)


def _read_jsonl_as_records(path: str) -> list:
    """Read a JSONL file and convert to simple record objects."""
    _TEXT_KEYS = ("text", "input", "content", "prompt", "instruction")
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = ""
            for k in _TEXT_KEYS:
                val = row.get(k, "")
                if val and isinstance(val, str):
                    content = val
                    break
            # For SFT/chat data, also append the response if present
            response = row.get("response", "") or row.get("chosen", "")
            if response and isinstance(response, str):
                content = (content + chr(10) + response) if content else response
            meta = {k: v for k, v in row.items() if k not in (*_TEXT_KEYS, "response", "chosen", "rejected")}
            records.append(_SimpleRecord(content=content, metadata=meta))
    return records


def _rows_to_records(rows: list, text_keys: tuple) -> list:
    """Convert a list of dicts to _SimpleRecord objects."""
    records = []
    for row in rows:
        content = ""
        for k in text_keys:
            val = row.get(k, "")
            if val and isinstance(val, str):
                content = val
                break
        response = row.get("response", "") or row.get("chosen", "")
        if response and isinstance(response, str):
            content = (content + chr(10) + response) if content else response
        meta = {k: v for k, v in row.items() if k not in (*text_keys, "response", "chosen", "rejected")}
        records.append(_SimpleRecord(content=content, metadata=meta))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Crucible remote agent")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    method = config.get("method", "train")
    method_args = config.get("method_args", {})
    output_path = config.get("result_output", "result.json")

    _log_memory("before-imports")
    print("CRUCIBLE_AGENT: Loading crucible modules...", flush=True)
    from core.config import CrucibleConfig
    from core.training_methods import dispatch_training
    from store.dataset_sdk import CrucibleClient
    _log_memory("after-imports")

    print("CRUCIBLE_AGENT: Initializing client...", flush=True)
    crucible_config = CrucibleConfig.from_env()
    print(f"CRUCIBLE_AGENT: data_root={crucible_config.data_root}", flush=True)
    client = CrucibleClient(crucible_config)
    _log_memory("after-client-init")
    print("CRUCIBLE_AGENT: Client ready", flush=True)

    # Pre-flight: import torch before dispatch to isolate import crashes
    print("CRUCIBLE_AGENT: Importing torch...", flush=True)
    import torch
    print(f"CRUCIBLE_AGENT: torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CRUCIBLE_AGENT: GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB", flush=True)
    _log_memory("after-torch")

    # --- Eval jobs: run benchmarks directly, skip training setup ---
    if method == "eval":
        print("CRUCIBLE_AGENT: Running evaluation...", flush=True)
        from eval.benchmark_runner import run_benchmarks, AVAILABLE_BENCHMARKS
        benchmarks_str = method_args.get("benchmarks", "")
        benchmarks = (
            [b.strip() for b in benchmarks_str.split(",") if b.strip()]
            if benchmarks_str else list(AVAILABLE_BENCHMARKS)
        )
        model_path = method_args.get("model_path", "")
        base_model_path = method_args.get("base_model_path") or None
        max_samples_val = method_args.get("max_samples")
        max_samples = int(max_samples_val) if max_samples_val else None
        try:
            eval_result = run_benchmarks(
                model_path, benchmarks, base_model_path,
                max_samples=max_samples,
            )
            result_data = {
                "status": "completed",
                "job_type": "eval",
                "model_path": eval_result.model_path,
                "average_score": eval_result.average_score,
                "benchmarks": [
                    {"name": r.benchmark_name, "score": r.score,
                     "num_examples": r.num_examples, "correct": r.correct}
                    for r in eval_result.benchmark_results
                ],
            }
            if eval_result.base_results:
                result_data["base_benchmarks"] = [
                    {"name": r.benchmark_name, "score": r.score,
                     "num_examples": r.num_examples, "correct": r.correct}
                    for r in eval_result.base_results
                ]
        except Exception as exc:
            import traceback
            result_data = {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=2)
        if result_data["status"] == "failed":
            print(f"CRUCIBLE_AGENT_ERROR: {result_data['error']}", file=sys.stderr)
            sys.exit(1)
        avg = result_data.get("average_score", "N/A")
        print(f"CRUCIBLE_AGENT: Evaluation complete. Average score: {avg}")
        print("CRUCIBLE_AGENT_COMPLETE")
        return

    # --- Interpretability jobs: run analysis, write result.json ---
    def _dispatch_logit_lens(ma):
        from core.logit_lens_types import LogitLensOptions
        from serve.logit_lens_runner import run_logit_lens
        opts = LogitLensOptions(**{
            k: v for k, v in ma.items()
            if k in LogitLensOptions.__dataclass_fields__
        })
        return run_logit_lens(opts)

    def _dispatch_activation_pca(ma):
        from core.activation_pca_types import ActivationPcaOptions
        from serve.activation_pca_runner import run_activation_pca
        opts = ActivationPcaOptions(**{
            k: v for k, v in ma.items()
            if k in ActivationPcaOptions.__dataclass_fields__
        })
        raw_path = ma.get("raw_data_path", "")
        records = []
        if raw_path and os.path.isfile(raw_path):
            print(f"CRUCIBLE_AGENT: Reading records from {raw_path}", flush=True)
            records = _read_data_as_records(raw_path)
            print(f"CRUCIBLE_AGENT: Loaded {len(records)} records", flush=True)
        return run_activation_pca(opts, records)

    def _dispatch_activation_patch(ma):
        from core.activation_patching_types import ActivationPatchingOptions
        from serve.activation_patching_runner import run_activation_patching
        opts = ActivationPatchingOptions(**{
            k: v for k, v in ma.items()
            if k in ActivationPatchingOptions.__dataclass_fields__
        })
        return run_activation_patching(opts)

    def _dispatch_linear_probe(ma):
        from core.linear_probe_types import LinearProbeOptions
        from serve.linear_probe_runner import run_linear_probe
        opts = LinearProbeOptions(**{
            k: v for k, v in ma.items()
            if k in LinearProbeOptions.__dataclass_fields__
        })
        raw_path = ma.get("raw_data_path", "")
        records = []
        if raw_path and os.path.isfile(raw_path):
            print(f"CRUCIBLE_AGENT: Reading records from {raw_path}", flush=True)
            records = _read_data_as_records(raw_path)
            print(f"CRUCIBLE_AGENT: Loaded {len(records)} records", flush=True)
        return run_linear_probe(opts, records)

    def _dispatch_sae_train(ma):
        from core.sae_types import SaeTrainOptions
        from serve.sae_train_runner import run_sae_train
        opts = SaeTrainOptions(**{
            k: v for k, v in ma.items()
            if k in SaeTrainOptions.__dataclass_fields__
        })
        raw_path = ma.get("raw_data_path", "")
        records = []
        if raw_path and os.path.isfile(raw_path):
            print(f"CRUCIBLE_AGENT: Reading records from {raw_path}", flush=True)
            records = _read_data_as_records(raw_path)
            print(f"CRUCIBLE_AGENT: Loaded {len(records)} records", flush=True)
        return run_sae_train(opts, records)

    def _dispatch_sae_analyze(ma):
        from core.sae_types import SaeAnalyzeOptions
        from serve.sae_analyze_runner import run_sae_analyze
        opts = SaeAnalyzeOptions(**{
            k: v for k, v in ma.items()
            if k in SaeAnalyzeOptions.__dataclass_fields__
        })
        records = None
        raw_path = ma.get("raw_data_path", "")
        if raw_path and os.path.isfile(raw_path):
            print(f"CRUCIBLE_AGENT: Reading records from {raw_path}", flush=True)
            records = _read_data_as_records(raw_path)
            print(f"CRUCIBLE_AGENT: Loaded {len(records)} records for feature associations", flush=True)
        return run_sae_analyze(opts, records)

    def _dispatch_steer_compute(ma):
        from core.steering_types import SteerComputeOptions
        from serve.steer_compute_runner import run_steer_compute
        opts = SteerComputeOptions(**{
            k: v for k, v in ma.items()
            if k in SteerComputeOptions.__dataclass_fields__
        })
        pos_records = None
        neg_records = None
        pos_path = ma.get("positive_raw_data_path", "")
        neg_path = ma.get("negative_raw_data_path", "")
        if pos_path and os.path.isfile(pos_path):
            pos_records = _read_data_as_records(pos_path)
            print(f"CRUCIBLE_AGENT: Loaded {len(pos_records)} positive records", flush=True)
        if neg_path and os.path.isfile(neg_path):
            neg_records = _read_data_as_records(neg_path)
            print(f"CRUCIBLE_AGENT: Loaded {len(neg_records)} negative records", flush=True)
        return run_steer_compute(opts, pos_records, neg_records)

    def _dispatch_steer_apply(ma):
        from core.steering_types import SteerApplyOptions
        from serve.steer_apply_runner import run_steer_apply
        opts = SteerApplyOptions(**{
            k: v for k, v in ma.items()
            if k in SteerApplyOptions.__dataclass_fields__
        })
        return run_steer_apply(opts)

    _INTERP_DISPATCH = {
        "logit-lens": _dispatch_logit_lens,
        "activation-pca": _dispatch_activation_pca,
        "activation-patch": _dispatch_activation_patch,
        "linear-probe": _dispatch_linear_probe,
        "sae-train": _dispatch_sae_train,
        "sae-analyze": _dispatch_sae_analyze,
        "steer-compute": _dispatch_steer_compute,
        "steer-apply": _dispatch_steer_apply,
    }

    if method in _INTERP_DISPATCH:
        print(f"CRUCIBLE_AGENT: Running interpretability ({method})...", flush=True)
        _ensure_output_dir(method_args)
        try:
            interp_result = _INTERP_DISPATCH[method](method_args)
            result_data = {
                "status": "completed",
                "job_type": method,
                **interp_result,
            }
        except Exception as exc:
            import traceback
            result_data = {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        with open(output_path, "w") as f:
            json.dump(result_data, f, indent=2)
        if result_data["status"] == "failed":
            print(f"CRUCIBLE_AGENT_ERROR: {result_data['error']}", file=sys.stderr)
            sys.exit(1)
        print(f"CRUCIBLE_AGENT: Interpretability ({method}) complete.")
        print("CRUCIBLE_AGENT_COMPLETE")
        return

    # --- Training jobs: set up output dir, dataset, and dispatch ---
    _ensure_output_dir(method_args)
    _ensure_dataset_name(method_args, method)
    _hydrate_nested_configs(method_args, method)

    if method in _RECORD_BASED_METHODS:
        _ingest_raw_data(client, method_args)

    print(f"CRUCIBLE_AGENT: Dispatching {method}...", flush=True)
    print(f"CRUCIBLE_AGENT: Config keys: {list(method_args.keys())}", flush=True)
    _log_memory("before-dispatch")

    try:
        result = dispatch_training(client, method, method_args)
        result_data = {
            "status": "completed",
            "model_path": getattr(result, "model_path", ""),
            "history_path": getattr(result, "history_path", ""),
            "epochs_completed": getattr(result, "epochs_completed", 0),
            "run_id": getattr(result, "run_id", ""),
            "result": str(result),
        }
        # Embed training history into result.json so the UI can render
        # loss curves without a separate SSH fetch.
        hp = getattr(result, "history_path", "")
        if hp:
            try:
                with open(hp) as hf:
                    result_data["training_history"] = json.load(hf)
            except Exception:
                pass
    except Exception as exc:
        import traceback
        result_data = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)

    if result_data["status"] == "failed":
        print(f"CRUCIBLE_AGENT_ERROR: {result_data['error']}", file=sys.stderr)
        sys.exit(1)
    print("CRUCIBLE_AGENT_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"CRUCIBLE_AGENT_ERROR: Unhandled {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        _log_memory("on-crash")
        raise
'''
