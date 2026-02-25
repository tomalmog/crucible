"""Human-in-the-loop annotation for preference data.

This module provides labeling interfaces for RLHF/DPO data
with optional LLM-assisted pre-labeling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnnotationTask:
    """One annotation task for human labeling.

    Attributes:
        task_id: Unique task identifier.
        prompt: The prompt to evaluate.
        response_a: First response option.
        response_b: Second response option.
        pre_label: LLM-suggested label (optional).
        human_label: Human-assigned label.
    """

    task_id: str
    prompt: str
    response_a: str
    response_b: str
    pre_label: str = ""
    human_label: str = ""


@dataclass
class AnnotationSession:
    """A batch of annotation tasks.

    Attributes:
        session_id: Session identifier.
        tasks: List of annotation tasks.
        completed: Number of completed annotations.
    """

    session_id: str
    tasks: list[AnnotationTask] = field(default_factory=list)
    completed: int = 0


def create_annotation_tasks(
    data_path: str,
) -> list[AnnotationTask]:
    """Create annotation tasks from a preference data file."""
    path = Path(data_path)
    if not path.exists():
        return []
    tasks: list[AnnotationTask] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tasks.append(AnnotationTask(
                task_id=f"task-{i}",
                prompt=obj.get("prompt", ""),
                response_a=obj.get("response_a", obj.get("chosen", "")),
                response_b=obj.get("response_b", obj.get("rejected", "")),
            ))
    return tasks


def export_annotations(
    tasks: list[AnnotationTask],
    output_path: str,
) -> int:
    """Export completed annotations as DPO training data."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for t in tasks:
            if not t.human_label:
                continue
            if t.human_label == "a":
                entry = {"prompt": t.prompt, "chosen": t.response_a, "rejected": t.response_b}
            elif t.human_label == "b":
                entry = {"prompt": t.prompt, "chosen": t.response_b, "rejected": t.response_a}
            else:
                continue
            fh.write(json.dumps(entry) + "\n")
            count += 1
    return count
