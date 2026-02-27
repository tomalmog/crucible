"""ARC benchmark implementation.

AI2 Reasoning Challenge — tests science exam question answering
with the Challenge split. Uses multiple-choice scoring by comparing
sequence loss across answer completions.
"""

from __future__ import annotations

from eval.benchmark_runner import BenchmarkResult
from eval.benchmarks._model_loader import compute_sequence_loss, load_eval_model


def run_arc(model_path: str) -> BenchmarkResult:
    """Run ARC-Challenge benchmark against a model.

    For each question, computes sequence loss for each answer
    completion and picks the one with lowest loss.
    """
    eval_model = load_eval_model(model_path)
    examples = _load_arc_examples()
    correct = 0
    for example in examples:
        prompt = f"Question: {example['question']}\nAnswer:"
        losses = [
            compute_sequence_loss(eval_model, prompt + " " + choice)
            for choice in example["choices"]
        ]
        predicted = int(min(range(len(losses)), key=lambda i: losses[i]))
        if predicted == example["answer_idx"]:
            correct += 1
    total = max(len(examples), 1)
    score = round((correct / total) * 100, 2)
    return BenchmarkResult(
        benchmark_name="arc",
        score=score,
        num_examples=len(examples),
        correct=correct,
        details={"split": "challenge", "format": "multiple_choice"},
    )


def _load_arc_examples() -> list[dict[str, object]]:
    """Load ARC-Challenge examples from HuggingFace datasets."""
    from core.errors import ForgeDependencyError
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as error:
        raise ForgeDependencyError(
            "ARC benchmark requires the datasets package. "
            "Install with: pip install datasets"
        ) from error
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    examples = []
    for r in ds:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        answer_key = r["answerKey"]
        answer_idx = labels.index(answer_key) if answer_key in labels else 0
        examples.append({
            "question": r["question"],
            "choices": texts,
            "answer_idx": answer_idx,
        })
    return examples
