"""Benchmark registry management.

Manages user benchmark registries: lm-eval references (built-in and
user-added) and custom benchmarks with local JSONL data. Seeds
default benchmarks on first access.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkInfo:
    """Registry entry for any benchmark (lm-eval or custom)."""

    name: str
    display_name: str
    type: str  # "lm-eval" or "custom"
    entries: int
    description: str
    created_at: str


# ── Default benchmarks seeded on first access ────────────────────────

_DEFAULTS: list[dict[str, object]] = [
    {"name": "mmlu", "display_name": "MMLU", "entries": 14042, "description": "Massive Multitask Language Understanding — 57 subjects from STEM to humanities."},
    {"name": "hellaswag", "display_name": "HellaSwag", "entries": 10042, "description": "Commonsense reasoning — choose the most plausible sentence continuation."},
    {"name": "arc", "display_name": "ARC Challenge", "entries": 1172, "description": "AI2 Reasoning Challenge (hard set) — grade-school science questions."},
    {"name": "arc_easy", "display_name": "ARC Easy", "entries": 2376, "description": "AI2 Reasoning Challenge (easy set) — simpler science questions."},
    {"name": "winogrande", "display_name": "WinoGrande", "entries": 1267, "description": "Pronoun resolution requiring commonsense reasoning."},
    {"name": "truthfulqa", "display_name": "TruthfulQA", "entries": 817, "description": "Tests whether models generate truthful answers to tricky questions."},
    {"name": "gsm8k", "display_name": "GSM8K", "entries": 1319, "description": "Grade school math word problems requiring multi-step reasoning."},
    {"name": "math", "display_name": "MATH", "entries": 5000, "description": "Competition-level mathematics from AMC, AIME, and Olympiad problems."},
    {"name": "bbh", "display_name": "BBH", "entries": 6511, "description": "Big-Bench Hard — 23 challenging reasoning tasks."},
    {"name": "humaneval", "display_name": "HumanEval", "entries": 164, "description": "Python code generation — complete functions and pass unit tests."},
    {"name": "mbpp", "display_name": "MBPP", "entries": 500, "description": "Mostly Basic Python Problems — simpler code generation tasks."},
    {"name": "boolq", "display_name": "BoolQ", "entries": 3270, "description": "Yes/no reading comprehension questions from Wikipedia passages."},
    {"name": "piqa", "display_name": "PIQA", "entries": 1838, "description": "Physical Intuition QA — commonsense about physical world interactions."},
    {"name": "openbookqa", "display_name": "OpenBookQA", "entries": 500, "description": "Science QA requiring reasoning with provided science facts."},
]


def _ensure_defaults(data_root: Path) -> None:
    """Seed default benchmarks if the registry is empty or missing."""
    bench_root = data_root / "benchmarks"
    if bench_root.exists() and any(bench_root.iterdir()):
        return
    bench_root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    for d in _DEFAULTS:
        meta = {
            "name": d["name"],
            "display_name": d["display_name"],
            "type": "lm-eval",
            "entries": d["entries"],
            "description": d["description"],
            "created_at": now,
        }
        entry_dir = bench_root / str(d["name"])
        entry_dir.mkdir(exist_ok=True)
        (entry_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ── CRUD operations ──────────────────────────────────────────────────


def list_benchmarks(data_root: Path) -> list[BenchmarkInfo]:
    """List all registered benchmarks (lm-eval + custom)."""
    _ensure_defaults(data_root)
    bench_root = data_root / "benchmarks"
    if not bench_root.exists():
        return []
    results = []
    for d in sorted(bench_root.iterdir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        results.append(BenchmarkInfo(
            name=meta["name"],
            display_name=meta.get("display_name", meta["name"]),
            type=meta.get("type", "custom"),
            entries=meta.get("entries", 0),
            description=meta.get("description", ""),
            created_at=meta.get("created_at", ""),
        ))
    return results


def add_lm_eval_benchmark(
    data_root: Path,
    name: str,
    display_name: str = "",
    description: str = "",
) -> BenchmarkInfo:
    """Add an lm-eval task reference to the registry (instant, no data fetch)."""
    from core.constants import sanitize_remote_name
    safe = sanitize_remote_name(name)
    if not safe:
        raise ValueError(f"Invalid benchmark name: {name!r}")

    bench_dir = data_root / "benchmarks" / safe
    if bench_dir.exists():
        raise ValueError(f"Benchmark '{name}' already exists in registry")

    bench_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "name": name,
        "display_name": display_name or name,
        "type": "lm-eval",
        "entries": 0,
        "description": description or f"lm-eval task: {name}",
        "created_at": now,
    }
    (bench_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return BenchmarkInfo(**meta)


def resolve_entry_count(data_root: Path, name: str) -> int:
    """Fetch the entry count for an lm-eval task and update meta.json."""
    from eval.benchmark_runner import _TASK_NAME_MAP
    task_name = _TASK_NAME_MAP.get(name, name)
    count = _get_lm_eval_entry_count(task_name)
    if count > 0:
        meta_path = data_root / "benchmarks" / name / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["entries"] = count
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return count


def _get_lm_eval_entry_count(name: str) -> int:
    """Load an lm-eval task to determine total entry count across all subtasks."""
    try:
        from lm_eval.tasks import TaskManager
        tm = TaskManager()
        task_dict = tm.load_task_or_group(name)
        tasks = _collect_tasks(task_dict)
        total = 0
        for task_obj in tasks:
            if hasattr(task_obj, "has_test_docs") and task_obj.has_test_docs():
                total += len(list(task_obj.test_docs()))
            elif hasattr(task_obj, "has_validation_docs") and task_obj.has_validation_docs():
                total += len(list(task_obj.validation_docs()))
        return total
    except Exception:
        return 0


def _collect_tasks(task_dict: dict) -> list:
    """Recursively collect all task objects from a task dict."""
    tasks: list = []
    for _, val in task_dict.items():
        if isinstance(val, dict):
            tasks.extend(_collect_tasks(val))
        elif hasattr(val, "has_test_docs") or hasattr(val, "has_validation_docs"):
            tasks.append(val)
    return tasks


def create_custom_benchmark(
    data_root: Path,
    name: str,
    source_path: str,
) -> BenchmarkInfo:
    """Create a custom benchmark from a JSONL file with prompt/response fields."""
    from core.constants import sanitize_remote_name
    safe_name = sanitize_remote_name(name)
    if not safe_name or safe_name != name:
        raise ValueError(f"Invalid benchmark name: {name!r}")

    bench_dir = data_root / "benchmarks" / name
    bench_dir.mkdir(parents=True, exist_ok=True)

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

    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "name": name,
        "display_name": name,
        "type": "custom",
        "entries": entries,
        "description": "Custom benchmark",
        "created_at": now,
    }
    (bench_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return BenchmarkInfo(**meta)


def delete_benchmark(data_root: Path, name: str) -> None:
    """Delete any benchmark (lm-eval reference or custom)."""
    bench_dir = data_root / "benchmarks" / name
    if not bench_dir.exists():
        raise FileNotFoundError(f"Benchmark '{name}' not found")
    shutil.rmtree(bench_dir)


def sample_custom_benchmark(
    data_root: Path,
    name: str,
    offset: int = 0,
    limit: int = 25,
) -> list[dict[str, str]]:
    """Read a page of entries from a custom benchmark's data file."""
    data_path = data_root / "benchmarks" / name / "data.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Benchmark '{name}' data not found")
    rows: list[dict[str, str]] = []
    skipped = 0
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skipped < offset:
                skipped += 1
                continue
            if len(rows) >= limit:
                break
            row = json.loads(line)
            rows.append({"prompt": row.get("prompt", ""), "response": row.get("response", "")})
    return rows


def search_lm_eval_tasks(query: str, limit: int = 20) -> list[dict[str, str]]:
    """Search lm-eval task names matching a query string."""
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    q = query.lower()
    matches = [t for t in sorted(tm.all_tasks) if q in t.lower()]
    return [{"name": t} for t in matches[:limit]]


def get_benchmarks_include_path(data_root: Path) -> str | None:
    """Return the benchmarks directory path if custom tasks exist."""
    bench_root = data_root / "benchmarks"
    if not bench_root.exists():
        return None
    for d in bench_root.iterdir():
        if (d / "task.yaml").exists():
            return str(bench_root)
    return None
