"""Verification check implementations for Crucible."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from core.chat_types import ChatOptions
from core.errors import CrucibleVerificationError
from core.types import IngestOptions, TrainingOptions, TrainingRunResult
from core.verification_types import VerificationMode, VerificationRuntime
from store.dataset_sdk import CrucibleClient

CheckCallable = Callable[[VerificationRuntime], str]
CheckRow = tuple[str, str, CheckCallable]


def build_runtime(client: CrucibleClient, source_path: str) -> VerificationRuntime:
    """Build runtime state used by verification checks."""
    data_root = Path(tempfile.mkdtemp(prefix="crucible-verify-")).resolve()
    resolved_source = _resolve_source_path(source_path)
    run_spec_path = _write_runtime_run_spec(data_root, resolved_source)
    return VerificationRuntime(
        client=client.with_data_root(str(data_root)),
        data_root=data_root,
        source_path=resolved_source,
        dataset_name="verify_demo",
        run_spec_path=run_spec_path,
        train_output_dir=data_root / "outputs" / "train" / "verify_demo",
        export_output_dir=data_root / "outputs" / "exports" / "verify_demo",
        hooks_output_dir=data_root / "outputs" / "train" / "verify_hooks",
        hooks_marker_path=data_root / "outputs" / "hooks_marker.txt",
    )


def build_checks(mode: VerificationMode) -> tuple[CheckRow, ...]:
    """Build ordered check list for one verification mode."""
    checks: list[CheckRow] = [
        ("V001", "Hardware Profile", check_hardware_profile),
        ("V002", "Ingest Dataset", check_ingest_dataset),
        ("V003", "Run-Spec Execution", check_run_spec),
        ("V004", "Training Artifacts", check_training_artifacts),
        ("V005", "Lifecycle + Lineage", check_lifecycle_lineage),
        ("V006", "Export Training Data", check_export_training),
        ("V007", "Chat Inference", check_chat),
    ]
    if mode == "full":
        checks.append(("V008", "Hooks Extension", check_hooks_extension))
    return tuple(checks)


def check_hardware_profile(runtime: VerificationRuntime) -> str:
    """Validate hardware profile surface."""
    profile = runtime.client.hardware_profile()
    required = {"accelerator", "gpu_count", "recommended_precision_mode"}
    missing = sorted(required - set(profile))
    if missing:
        raise CrucibleVerificationError(
            f"Hardware profile missing required fields: {', '.join(missing)}."
        )
    return (
        f"accelerator={profile['accelerator']} "
        f"gpu_count={profile['gpu_count']} "
        f"precision={profile['recommended_precision_mode']}"
    )


def check_ingest_dataset(runtime: VerificationRuntime) -> str:
    """Verify ingest creates a dataset with records."""
    dataset_name = runtime.client.ingest(
        IngestOptions(
            dataset_name=runtime.dataset_name,
            source_uri=str(runtime.source_path),
            quality_model="perplexity",
        )
    )
    _, records = runtime.client.dataset(runtime.dataset_name).load_records()
    if not records:
        raise CrucibleVerificationError("Ingest produced zero records.")
    return f"dataset={dataset_name} records={len(records)}"


def check_run_spec(runtime: VerificationRuntime) -> str:
    """Verify run-spec execution path."""
    output_lines = runtime.client.run_spec(str(runtime.run_spec_path))
    if len(output_lines) < 2:
        raise CrucibleVerificationError("Run-spec execution returned insufficient output lines.")
    return f"output_lines={len(output_lines)}"


def check_training_artifacts(runtime: VerificationRuntime) -> str:
    """Verify training output artifact contract."""
    result = runtime.client.train(
        TrainingOptions(
            dataset_name=runtime.dataset_name,
            output_dir=str(runtime.train_output_dir),
            epochs=1,
            batch_size=2,
            max_token_length=64,
            validation_split=0.2,
            checkpoint_every_epochs=1,
            max_checkpoint_files=2,
            progress_log_interval_steps=1,
        )
    )
    runtime.train_result = result
    _verify_training_output_files(result)
    return f"run_id={result.run_id} artifacts={runtime.train_output_dir}"


def check_lifecycle_lineage(runtime: VerificationRuntime) -> str:
    """Verify lifecycle state and lineage edge materialization."""
    if runtime.train_result is None or runtime.train_result.run_id is None:
        raise CrucibleVerificationError("Training result missing; cannot verify lifecycle.")
    run_id = runtime.train_result.run_id
    run_record = runtime.client.get_training_run(run_id)
    if run_record.state != "completed":
        raise CrucibleVerificationError(f"Expected run state completed, got {run_record.state}.")
    lineage = runtime.client.get_lineage_graph()
    _verify_lineage_produced_edge(lineage, run_id, runtime.train_result.model_path)
    return f"run_state={run_record.state} run_id={run_id}"


def check_export_training(runtime: VerificationRuntime) -> str:
    """Verify training export shard manifest creation."""
    manifest_path = runtime.client.dataset(runtime.dataset_name).export_training(
        output_dir=str(runtime.export_output_dir),
        shard_size=50,
        include_metadata=True,
    )
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise CrucibleVerificationError(f"Training export manifest missing at {manifest_path}.")
    return f"manifest={manifest_path}"


def check_chat(runtime: VerificationRuntime) -> str:
    """Verify chat inference path with trained model."""
    if runtime.train_result is None:
        raise CrucibleVerificationError("Training result missing; cannot run chat verification.")
    response = runtime.client.chat(
        ChatOptions(
            dataset_name=runtime.dataset_name,
            model_path=runtime.train_result.model_path,
            prompt="hello",
            max_new_tokens=12,
            max_token_length=64,
            temperature=0.7,
            top_k=20,
        )
    )
    preview = response.response_text.strip()
    if not preview:
        raise CrucibleVerificationError("Chat response was empty.")
    runtime.chat_preview = preview[:80]
    return f"response_preview={runtime.chat_preview}"


def check_hooks_extension(runtime: VerificationRuntime) -> str:
    """Verify hooks file extension point during training."""
    hooks_path = _write_runtime_hooks_file(runtime)
    runtime.client.train(
        TrainingOptions(
            dataset_name=runtime.dataset_name,
            output_dir=str(runtime.hooks_output_dir),
            epochs=1,
            batch_size=2,
            max_token_length=64,
            validation_split=0.2,
            hooks_path=str(hooks_path),
        )
    )
    if not runtime.hooks_marker_path.exists():
        raise CrucibleVerificationError("Hooks marker file was not created by on_run_end hook.")
    return f"hooks_marker={runtime.hooks_marker_path}"


def _resolve_source_path(source_path: str) -> Path:
    source = Path(source_path).expanduser()
    if not source.is_absolute():
        source = Path.cwd() / source
    source = source.resolve()
    if not source.exists():
        raise CrucibleVerificationError(
            f"Verification source path does not exist at {source}. "
            "Provide --source with an existing file or directory."
        )
    return source


def _write_runtime_run_spec(data_root: Path, source_path: Path) -> Path:
    spec_path = data_root / "verification_run_spec.yaml"
    yaml_body = (
        "version: 1\n"
        "defaults:\n"
        f"  data_root: {data_root.as_posix()}\n"
        "  dataset: verify_spec\n"
        "steps:\n"
        "  - command: ingest\n"
        f"    source: {source_path.as_posix()}\n"
        "    quality_model: perplexity\n"
        "  - command: hardware-profile\n"
    )
    spec_path.write_text(yaml_body, encoding="utf-8")
    return spec_path


def _verify_training_output_files(result: TrainingRunResult) -> None:
    required_paths: list[str | None] = [
        result.model_path,
        result.history_path,
        result.artifact_contract_path,
        str(Path(result.model_path).parent / "training_config.json"),
        str(Path(result.model_path).parent / "tokenizer_vocab.json"),
        str(Path(result.model_path).parent / "reproducibility_bundle.json"),
    ]
    missing = [path for path in required_paths if path is None or not Path(path).exists()]
    if missing:
        missing_rows = ", ".join(path if path is not None else "<none>" for path in missing)
        raise CrucibleVerificationError(
            f"Training artifact verification failed. Missing files: {missing_rows}."
        )
    if not result.run_id:
        raise CrucibleVerificationError("Training result missing run_id.")


def _verify_lineage_produced_edge(
    lineage_payload: dict[str, object],
    run_id: str,
    model_path: str,
) -> None:
    edges = lineage_payload.get("edges")
    if not isinstance(edges, list):
        raise CrucibleVerificationError("Lineage payload missing edges list.")
    expected_edge = {"from": f"run:{run_id}", "to": f"model:{model_path}", "type": "produced"}
    if expected_edge not in edges:
        raise CrucibleVerificationError("Lineage graph missing produced model edge.")


def _write_runtime_hooks_file(runtime: VerificationRuntime) -> Path:
    hooks_path = runtime.data_root / "verification_hooks.py"
    marker = runtime.hooks_marker_path.as_posix()
    body = (
        "from pathlib import Path\n"
        "def on_run_end(context, result):\n"
        f"    Path('{marker}').write_text('hook-ok', encoding='utf-8')\n"
        "    _ = context\n"
        "    _ = result\n"
    )
    hooks_path.write_text(body, encoding="utf-8")
    return hooks_path
