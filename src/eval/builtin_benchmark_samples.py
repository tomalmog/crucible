"""Load sample entries from lm-eval benchmarks via HuggingFace datasets.

Bypasses lm-eval's TaskManager for speed — loads directly from HF
datasets using stored dataset paths. Falls back to TaskManager for
unknown tasks.
"""

from __future__ import annotations

# Known dataset configs for fast direct loading.
# Maps our benchmark name → HF dataset load args + column names.
_KNOWN_DATASETS: dict[str, dict[str, str | None]] = {
    "mmlu": {"path": "cais/mmlu", "name": "all", "split": "test", "prompt_col": "question", "response_col": "choices"},
    "hellaswag": {"path": "Rowan/hellaswag", "name": None, "split": "validation", "prompt_col": "ctx", "response_col": "endings"},
    "arc": {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "test", "prompt_col": "question", "response_col": "choices"},
    "arc_easy": {"path": "allenai/ai2_arc", "name": "ARC-Easy", "split": "test", "prompt_col": "question", "response_col": "choices"},
    "winogrande": {"path": "allenai/winogrande", "name": "winogrande_xl", "split": "validation", "prompt_col": "sentence", "response_col": "option1"},
    "truthfulqa": {"path": "truthfulqa/truthful_qa", "name": "multiple_choice", "split": "validation", "prompt_col": "question", "response_col": "mc1_targets"},
    "gsm8k": {"path": "openai/gsm8k", "name": "main", "split": "test", "prompt_col": "question", "response_col": "answer"},
    "math": {"path": "EleutherAI/hendrycks_math", "name": "algebra", "split": "test", "prompt_col": "problem", "response_col": "solution"},
    "bbh": {"path": "lukaemon/bbh", "name": "boolean_expressions", "split": "test", "prompt_col": "input", "response_col": "target"},
    "humaneval": {"path": "openai/openai_humaneval", "name": None, "split": "test", "prompt_col": "prompt", "response_col": "canonical_solution"},
    "mbpp": {"path": "google-research-datasets/mbpp", "name": "full", "split": "test", "prompt_col": "text", "response_col": "code"},
    "boolq": {"path": "google/boolq", "name": None, "split": "validation", "prompt_col": "question", "response_col": "answer"},
    "piqa": {"path": "piqa", "name": "plain_text", "split": "validation", "prompt_col": "goal", "response_col": "sol1"},
    "openbookqa": {"path": "allenai/openbookqa", "name": "main", "split": "test", "prompt_col": "question_stem", "response_col": "choices"},
}


def sample_builtin_benchmark(
    name: str, offset: int = 0, limit: int = 25,
) -> dict:
    """Load a page of samples from an lm-eval benchmark.

    Uses direct HuggingFace dataset loading for known benchmarks (fast),
    falls back to lm-eval TaskManager for unknown ones.
    """
    from eval.benchmark_runner import _TASK_NAME_MAP
    task_name = _TASK_NAME_MAP.get(name, name)

    # Fast path: known datasets — load directly from HF
    info = _KNOWN_DATASETS.get(name)
    if info:
        return _load_direct(info, offset, limit)

    # Slow fallback: use lm-eval TaskManager
    return _load_via_task_manager(task_name, offset, limit)


def _load_direct(
    info: dict[str, str | None], offset: int, limit: int,
) -> dict:
    """Load samples directly from HuggingFace datasets (fast path)."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
        kwargs: dict = {"path": info["path"], "split": info["split"]}
        if info.get("name"):
            kwargs["name"] = info["name"]
        ds = load_dataset(**kwargs)
    except Exception as exc:
        return {"rows": [], "total": 0, "error": str(exc)}

    total = len(ds)
    prompt_col = info.get("prompt_col", "question")
    response_col = info.get("response_col", "answer")

    rows = []
    end = min(offset + limit, total)
    for i in range(offset, end):
        row = ds[i]
        p = row.get(prompt_col, "")
        r = row.get(response_col, "")
        rows.append({
            "prompt": str(p) if not isinstance(p, str) else p,
            "response": str(r) if not isinstance(r, str) else r,
        })

    return {"rows": rows, "total": total}


def _load_via_task_manager(
    task_name: str, offset: int, limit: int,
) -> dict:
    """Load samples via lm-eval TaskManager (slow fallback)."""
    try:
        from lm_eval.tasks import TaskManager
        tm = TaskManager()
        task_dict = tm.load_task_or_group(task_name)
    except Exception as exc:
        return {"rows": [], "total": 0, "error": str(exc)}

    task_obj = _find_first_task(task_dict)
    if task_obj is None:
        return {"rows": [], "total": 0}

    rows: list[dict[str, str]] = []
    skipped = 0
    for doc in _iter_eval_docs(task_obj):
        if skipped < offset:
            skipped += 1
            continue
        if len(rows) >= limit:
            break
        rows.append({
            "prompt": _extract_prompt(task_obj, doc),
            "response": _extract_target(task_obj, doc),
        })

    # Get total from dataset metadata
    total = skipped + len(rows)
    ds = getattr(task_obj, "dataset", None)
    if ds is not None:
        cfg = task_obj.config
        split = cfg.test_split or cfg.validation_split
        if split and split in ds:
            total = ds[split].num_rows

    return {"rows": rows, "total": total}


def _find_first_task(task_dict: dict) -> object | None:
    for _, val in task_dict.items():
        if isinstance(val, dict):
            result = _find_first_task(val)
            if result is not None:
                return result
        elif hasattr(val, "has_test_docs") or hasattr(val, "has_validation_docs"):
            return val
    return None


def _iter_eval_docs(task_obj: object):
    if hasattr(task_obj, "has_test_docs") and task_obj.has_test_docs():
        yield from task_obj.test_docs()
    elif hasattr(task_obj, "has_validation_docs") and task_obj.has_validation_docs():
        yield from task_obj.validation_docs()


def _extract_prompt(task_obj: object, doc: dict) -> str:
    try:
        text = task_obj.doc_to_text(doc)
        return str(text) if text is not None else ""
    except Exception:
        for key in ("question", "prompt", "input", "ctx", "sentence", "goal", "problem", "text"):
            if key in doc and isinstance(doc[key], str):
                return doc[key]
        return str(doc)[:200]


def _extract_target(task_obj: object, doc: dict) -> str:
    try:
        target = task_obj.doc_to_target(doc)
        if isinstance(target, (list, tuple)):
            return ", ".join(str(t) for t in target)
        return str(target) if target is not None else ""
    except Exception:
        for key in ("answer", "response", "output", "target", "solution", "label"):
            if key in doc:
                return str(doc[key])
        return ""
