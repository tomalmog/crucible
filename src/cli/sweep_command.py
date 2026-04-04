"""Sweep command wiring for Crucible CLI.

This module isolates hyperparameter sweep command parser and execution
logic, mapping CLI arguments to SweepConfig for sweep orchestration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from core.errors import CrucibleDependencyError, CrucibleSweepError
from core.sweep_types import SweepConfig, SweepParameter
from core.training_methods import ALL_TRAINING_METHODS
from serve.sweep_analysis import format_sweep_report, format_sweep_report_json
from serve.sweep_runner import run_sweep
from store.dataset_sdk import CrucibleClient


def add_sweep_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register sweep subcommand.

    Args:
        subparsers: Argparse subparsers object.
    """
    parser = subparsers.add_parser(
        "sweep",
        help="Run hyperparameter sweep over training configurations",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory for sweep trials",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="YAML file with sweep parameter definitions",
    )
    parser.add_argument(
        "--params",
        default=None,
        help='Inline JSON sweep parameters. Format: \'{"parameters": [{"name": "learning_rate", "values": [0.001, 0.01]}]}\'',
    )
    parser.add_argument(
        "--strategy",
        default="grid",
        choices=["grid", "random"],
        help="Sweep strategy (grid or random)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=10,
        help="Maximum number of trials for random search",
    )
    parser.add_argument(
        "--metric",
        default="validation_loss",
        help="Metric name to optimize",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        default=False,
        help="Maximize metric instead of minimizing",
    )
    parser.add_argument(
        "--method",
        default="train",
        choices=list(ALL_TRAINING_METHODS),
        help="Training method to use for each trial",
    )
    parser.add_argument(
        "--method-args",
        default=None,
        help="JSON string with fixed method-specific arguments (e.g. base_model, sft_data_path)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON",
    )
    # Remote cluster flags
    parser.add_argument(
        "--cluster",
        default=None,
        help="Submit sweep to a remote cluster instead of running locally",
    )
    parser.add_argument("--partition", default="", help="Slurm partition")
    parser.add_argument(
        "--gpus-per-node", type=int, default=1, help="GPUs per node",
    )
    parser.add_argument("--gpu-type", default="", help="GPU type (e.g. a100)")
    parser.add_argument(
        "--cpus-per-task", type=int, default=4, help="CPUs per task",
    )
    parser.add_argument("--memory", default="32G", help="Memory limit")
    parser.add_argument(
        "--time-limit", default="12:00:00", help="Wall-clock limit",
    )


def run_sweep_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Handle sweep command invocation.

    Routes to remote cluster submission when ``--cluster`` is provided,
    otherwise runs locally.

    Args:
        client: SDK client for training.
        args: Parsed CLI args.

    Returns:
        Exit code.
    """
    import json as _json
    config = _build_sweep_config(args)
    if getattr(args, "cluster", None):
        return _run_remote_sweep(client, config, args)
    result = run_sweep(client, config, random_seed=42)
    if getattr(args, "json", False):
        print(_json.dumps(format_sweep_report_json(result)))
    else:
        report = format_sweep_report(result)
        print(report)
    return 0


def _run_remote_sweep(
    client: CrucibleClient,
    config: SweepConfig,
    args: argparse.Namespace,
) -> int:
    """Generate parameter combos locally and submit as a remote sweep.

    Uses the unified backend dispatch system so sweeps run on the
    cluster via Slurm job arrays (or SSH, depending on backend).

    Args:
        client: SDK client (used for data_root).
        config: Validated SweepConfig with parameters and strategy.
        args: CLI args containing cluster and resource flags.

    Returns:
        Exit code.
    """
    from core.backend_registry import get_backend
    from core.job_types import JobSpec, ResourceConfig
    from core.training_methods import TRAINING_METHOD_LABELS
    from serve.sweep_parameter_generator import generate_sweep_parameters
    from store.cluster_registry import load_cluster

    param_combos = generate_sweep_parameters(config, random_seed=42)
    method_label = TRAINING_METHOD_LABELS.get(
        config.training_method, config.training_method,
    )
    print(
        f"Sweep: {len(param_combos)} trials | method={method_label} "
        f"| strategy={config.strategy} | cluster={args.cluster}",
        flush=True,
    )

    trial_configs: list[dict[str, object]] = []
    for params in param_combos:
        tc: dict[str, object] = {"dataset_name": config.dataset_name}
        # Apply fixed method-specific args (e.g. base_model_path)
        for key, value in config.method_args:
            tc[key] = value
        # Apply swept parameters
        tc.update(params)
        trial_configs.append(tc)

    data_root = client._config.data_root
    cluster = load_cluster(data_root, args.cluster)
    resources = ResourceConfig(
        partition=getattr(args, "partition", ""),
        gpus_per_node=getattr(args, "gpus_per_node", 1),
        cpus_per_task=getattr(args, "cpus_per_task", 4),
        memory=getattr(args, "memory", "32G"),
        time_limit=getattr(args, "time_limit", "12:00:00"),
        gpu_type=getattr(args, "gpu_type", ""),
    )
    spec = JobSpec(
        job_type=config.training_method,
        backend=cluster.backend,
        cluster_name=args.cluster,
        resources=resources,
        is_sweep=True,
        sweep_trials=tuple(trial_configs),
    )
    backend = get_backend(spec.backend)
    record = backend.submit(data_root, spec)
    print(
        f"Submitted remote sweep {record.job_id} "
        f"({len(trial_configs)} trials, state={record.state})",
    )
    return 0


def _build_sweep_config(args: argparse.Namespace) -> SweepConfig:
    """Build SweepConfig from CLI arguments and config source.

    Supports YAML file (--config-file) or inline JSON (--params).

    Args:
        args: Parsed CLI arguments.

    Returns:
        Validated SweepConfig.

    Raises:
        CrucibleSweepError: If config is invalid or missing.
    """
    if args.params:
        parameters = _parse_inline_params(args.params)
    elif args.config_file:
        parameters = _load_parameters_from_yaml(args.config_file)
    else:
        raise CrucibleSweepError(
            "Sweep requires --config-file or --params. "
            "Provide parameter definitions via one of these flags."
        )
    method_args: tuple[tuple[str, str], ...] = ()
    if args.method_args:
        import json as _json
        try:
            raw_args = _json.loads(args.method_args)
        except _json.JSONDecodeError as error:
            raise CrucibleSweepError(
                f"Failed to parse --method-args JSON: {error}."
            ) from error
        if not isinstance(raw_args, dict):
            raise CrucibleSweepError("--method-args must be a JSON object.")
        method_args = tuple(
            (k.lstrip("-").replace("-", "_"), v)
            for k, v in raw_args.items()
        )
    return SweepConfig(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        base_output_dir=args.output_dir,
        parameters=parameters,
        strategy=args.strategy,
        max_trials=args.max_trials,
        metric=args.metric,
        minimize=not args.maximize,
        training_method=args.method,
        method_args=method_args,
    )


def _parse_inline_params(params_json: str) -> tuple[SweepParameter, ...]:
    """Parse sweep parameters from inline JSON string.

    Expected format: {"parameters": [{"name": "learning_rate", "values": [0.001, 0.01]}]}

    Args:
        params_json: JSON string with parameter definitions.

    Returns:
        Tuple of SweepParameter instances.
    """
    import json as _json
    try:
        raw = _json.loads(params_json)
    except _json.JSONDecodeError as error:
        raise CrucibleSweepError(
            f"Failed to parse --params JSON: {error}."
        ) from error
    return _parse_parameters(raw)


def _load_parameters_from_yaml(config_path: str) -> tuple[SweepParameter, ...]:
    """Load sweep parameters from a YAML config file.

    Args:
        config_path: Path to YAML sweep config.

    Returns:
        Tuple of SweepParameter instances.

    Raises:
        CrucibleDependencyError: If PyYAML is unavailable.
        CrucibleSweepError: If file is invalid.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as error:
        raise CrucibleDependencyError(
            "Sweep config requires PyYAML. Install with 'pip install pyyaml==6.0.2'."
        ) from error
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise CrucibleSweepError(
            f"Sweep config file not found at {path}."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CrucibleSweepError(
            f"Failed to parse sweep config: {error}."
        ) from error
    return _parse_parameters(raw)


def _parse_parameters(raw: Any) -> tuple[SweepParameter, ...]:
    """Parse SweepParameter instances from raw YAML data.

    Args:
        raw: Parsed YAML dictionary.

    Returns:
        Tuple of SweepParameter instances.

    Raises:
        CrucibleSweepError: If data structure is invalid.
    """
    if not isinstance(raw, dict):
        raise CrucibleSweepError("Sweep config must be a YAML mapping.")
    params_list = raw.get("parameters", [])
    if not isinstance(params_list, list) or not params_list:
        raise CrucibleSweepError(
            "Sweep config must contain a non-empty 'parameters' list."
        )
    results: list[SweepParameter] = []
    for entry in params_list:
        results.append(_parse_single_parameter(entry))
    return tuple(results)


def _parse_single_parameter(entry: Any) -> SweepParameter:
    """Parse one SweepParameter from a YAML entry.

    Args:
        entry: Dictionary from YAML parameters list.

    Returns:
        SweepParameter instance.

    Raises:
        CrucibleSweepError: If entry is malformed.
    """
    if not isinstance(entry, dict) or "name" not in entry:
        raise CrucibleSweepError(
            "Each sweep parameter must be a mapping with a 'name' field."
        )
    values = tuple(float(v) for v in entry.get("values", []))
    return SweepParameter(
        name=entry["name"],
        values=values,
        min_value=float(entry.get("min_value", 0.0)),
        max_value=float(entry.get("max_value", 1.0)),
        log_scale=bool(entry.get("log_scale", False)),
    )
