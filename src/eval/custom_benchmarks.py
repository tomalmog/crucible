"""Custom benchmark management.

Creates, lists, and deletes user-defined evaluation benchmarks
stored as lm-eval YAML task configs with local JSONL data.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CustomBenchmarkInfo:
    name: str
    entries: int
    created_at: str


def create_custom_benchmark(
    data_root: Path,
    name: str,
    source_path: str,
) -> CustomBenchmarkInfo:
    """Create a custom benchmark from a JSONL file with prompt/response fields."""
    from core.constants import sanitize_remote_name
    safe_name = sanitize_remote_name(name)
    if not safe_name or safe_name != name:
        raise ValueError(f"Invalid benchmark name: {name!r}")

    bench_dir = data_root / "benchmarks" / name
    bench_dir.mkdir(parents=True, exist_ok=True)

    # Copy and validate the source JSONL
    source = Path(source_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    entries = 0
    data_path = bench_dir / "data.jsonl"
    with open(source, encoding="utf-8") as src, open(data_path, "w", encoding="utf-8") as dst:
        for line_num, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" not in row or "response" not in row:
                raise ValueError(
                    f"Line {line_num}: missing 'prompt' or 'response' field. "
                    "Each line must have both fields."
                )
            dst.write(line + "\n")
            entries += 1

    if entries == 0:
        shutil.rmtree(bench_dir)
        raise ValueError("Source file has no valid entries")

    # Generate lm-eval YAML task config
    yaml_content = f"""task: {name}
dataset_path: json
dataset_kwargs:
  data_files: {data_path}
test_split: train
output_type: loglikelihood
doc_to_text: "{{{{prompt}}}} "
doc_to_target: "{{{{response}}}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
num_fewshot: 0
"""
    (bench_dir / "task.yaml").write_text(yaml_content, encoding="utf-8")

    # Save metadata
    now = datetime.now(timezone.utc).isoformat()
    meta = {"name": name, "entries": entries, "created_at": now}
    (bench_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return CustomBenchmarkInfo(name=name, entries=entries, created_at=now)


def list_custom_benchmarks(data_root: Path) -> list[CustomBenchmarkInfo]:
    """List all custom benchmarks."""
    bench_root = data_root / "benchmarks"
    if not bench_root.exists():
        return []
    results = []
    for d in sorted(bench_root.iterdir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        results.append(CustomBenchmarkInfo(
            name=meta["name"],
            entries=meta.get("entries", 0),
            created_at=meta.get("created_at", ""),
        ))
    return results


def delete_custom_benchmark(data_root: Path, name: str) -> None:
    """Delete a custom benchmark."""
    bench_dir = data_root / "benchmarks" / name
    if not bench_dir.exists():
        raise FileNotFoundError(f"Benchmark '{name}' not found")
    shutil.rmtree(bench_dir)


def get_benchmarks_include_path(data_root: Path) -> str | None:
    """Return the benchmarks directory path if it exists and has entries."""
    bench_root = data_root / "benchmarks"
    if bench_root.exists() and any(bench_root.iterdir()):
        return str(bench_root)
    return None
