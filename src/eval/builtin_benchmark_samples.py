"""Load sample entries from any lm-eval benchmark via HuggingFace datasets."""

from __future__ import annotations


def sample_builtin_benchmark(
    name: str, offset: int = 0, limit: int = 25,
) -> dict:
    """Load samples from any lm-eval task.

    Uses the task object's own dataset and doc_to_text/doc_to_target
    to extract prompt/response pairs. Returns ``{"rows": [...], "total": N}``.
    """
    try:
        from lm_eval.tasks import TaskManager
        from eval.benchmark_runner import _TASK_NAME_MAP
        task_name = _TASK_NAME_MAP.get(name, name)
        tm = TaskManager()
        task_dict = tm.load_task_or_group(task_name)
    except Exception as exc:
        return {"rows": [], "total": 0, "error": str(exc)}

    # Collect all task objects (flatten groups)
    tasks = _collect_all_tasks(task_dict)
    if not tasks:
        return {"rows": [], "total": 0}

    # Gather all docs across all subtasks with their parent task reference
    all_docs: list[tuple[object, dict]] = []
    for task_obj in tasks:
        for doc in _get_eval_docs(task_obj):
            all_docs.append((task_obj, doc))

    total = len(all_docs)
    rows = []
    end = min(offset + limit, total)
    for i in range(offset, end):
        task_obj, doc = all_docs[i]
        prompt = _extract_prompt(task_obj, doc)
        response = _extract_target(task_obj, doc)
        rows.append({"prompt": prompt, "response": response})

    return {"rows": rows, "total": total}


def _collect_all_tasks(task_dict: dict) -> list:
    """Recursively collect all task objects from a task dict, flattening groups."""
    tasks: list = []
    for _, val in task_dict.items():
        if isinstance(val, dict):
            tasks.extend(_collect_all_tasks(val))
        elif hasattr(val, "has_test_docs") or hasattr(val, "has_validation_docs"):
            tasks.append(val)
    return tasks


def _get_eval_docs(task_obj: object) -> list:
    """Get the evaluation documents from a task."""
    if hasattr(task_obj, "has_test_docs") and task_obj.has_test_docs():
        return list(task_obj.test_docs())
    if hasattr(task_obj, "has_validation_docs") and task_obj.has_validation_docs():
        return list(task_obj.validation_docs())
    return []


def _extract_prompt(task_obj: object, doc: dict) -> str:
    """Extract the prompt text from a document using the task's doc_to_text."""
    try:
        text = task_obj.doc_to_text(doc)
        return str(text) if text is not None else ""
    except Exception:
        # Fallback: try common field names
        for key in ("question", "prompt", "input", "ctx", "sentence", "goal", "problem", "text"):
            if key in doc:
                val = doc[key]
                return str(val) if not isinstance(val, str) else val
        return str(doc)[:200]


def _extract_target(task_obj: object, doc: dict) -> str:
    """Extract the target/answer from a document using the task's doc_to_target."""
    try:
        target = task_obj.doc_to_target(doc)
        if isinstance(target, (list, tuple)):
            return ", ".join(str(t) for t in target)
        return str(target) if target is not None else ""
    except Exception:
        for key in ("answer", "response", "output", "target", "solution", "label"):
            if key in doc:
                val = doc[key]
                return str(val) if not isinstance(val, str) else val
        return ""
