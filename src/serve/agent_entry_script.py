"""String constant containing the remote agent entry point script.

This is embedded into the agent tarball as forge_agent_entry.py
and executed on the remote Slurm node.
"""

ENTRY_SCRIPT = '''"""Forge remote agent entry point.

Reads a JSON training config, sets up sys.path, and dispatches training.
"""

import os
import signal
import sys

# Immediate output to confirm Python started
print("FORGE_AGENT: Python process started", flush=True)

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
    print(f"FORGE_AGENT: [{label}] RSS={rss_mb:.0f}MB", flush=True)


def _signal_handler(signum, frame):
    """Log signal before exit (catches SIGTERM from Slurm)."""
    name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    print(f"FORGE_AGENT_ERROR: Received signal {name} ({signum})", flush=True)
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
    """Ingest a raw JSONL file into a forge dataset for record-based methods.

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
    print(f"FORGE_AGENT: Ingesting {raw_path} as dataset '{ds_name}'...")
    client.ingest(opts)
    method_args["dataset_name"] = ds_name
    print(f"FORGE_AGENT: Ingestion complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge remote agent")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    method = config.get("method", "train")
    method_args = config.get("method_args", {})
    output_path = config.get("result_output", "result.json")

    _log_memory("before-imports")
    print("FORGE_AGENT: Loading forge modules...", flush=True)
    from core.config import ForgeConfig
    from core.training_methods import dispatch_training
    from store.dataset_sdk import ForgeClient
    _log_memory("after-imports")

    print("FORGE_AGENT: Initializing client...", flush=True)
    forge_config = ForgeConfig.from_env()
    print(f"FORGE_AGENT: data_root={forge_config.data_root}", flush=True)
    client = ForgeClient(forge_config)
    _log_memory("after-client-init")
    print("FORGE_AGENT: Client ready", flush=True)

    # Ensure output_dir and dataset_name are set
    _ensure_output_dir(method_args)
    _ensure_dataset_name(method_args, method)
    _hydrate_nested_configs(method_args, method)

    # For record-based methods, ingest raw data if provided
    if method in _RECORD_BASED_METHODS:
        _ingest_raw_data(client, method_args)

    print(f"FORGE_AGENT: Dispatching {method}...", flush=True)
    print(f"FORGE_AGENT: Config keys: {list(method_args.keys())}", flush=True)
    _log_memory("before-dispatch")

    # Pre-flight: import torch before dispatch to isolate import crashes
    print("FORGE_AGENT: Importing torch...", flush=True)
    import torch
    print(f"FORGE_AGENT: torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"FORGE_AGENT: GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB", flush=True)
    _log_memory("after-torch")

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
        print(f"FORGE_AGENT_ERROR: {result_data['error']}", file=sys.stderr)
        sys.exit(1)
    print("FORGE_AGENT_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"FORGE_AGENT_ERROR: Unhandled {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        _log_memory("on-crash")
        raise
'''
