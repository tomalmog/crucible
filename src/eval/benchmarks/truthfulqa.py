"""TruthfulQA benchmark implementation.

TruthfulQA — evaluates model tendency to generate truthful
answers vs common misconceptions. Uses MC1 (multiple-choice)
scoring where the model must select the single correct answer.
"""

from __future__ import annotations

from eval.benchmark_runner import BenchmarkResult
from eval.benchmarks._model_loader import EvalModel, compute_completion_loss, load_eval_model


def run_truthfulqa(
    model_path: str,
    *,
    max_samples: int | None = None,
    eval_model: EvalModel | None = None,
) -> BenchmarkResult:
    """Run TruthfulQA MC1 benchmark against a model.

    For each question, computes sequence loss for each answer
    option and picks the one with the lowest loss.
    """
    if eval_model is None:
        eval_model = load_eval_model(model_path)
    examples = _load_truthfulqa_examples()
    if max_samples:
        examples = examples[:max_samples]
    correct = 0
    total = len(examples)
    for idx, example in enumerate(examples):
        prompt = f"Question: {example['question']}\nAnswer: "
        choices = example["choices"]
        losses = [
            compute_completion_loss(eval_model, prompt, str(c))
            for c in choices
        ]
        predicted = int(min(range(len(losses)), key=lambda i: losses[i]))
        if predicted == example["correct_idx"]:
            correct += 1
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(
                f"  truthfulqa: {idx + 1}/{total} examples, "
                f"{correct} correct",
                flush=True,
            )
    total = max(total, 1)
    score = round((correct / total) * 100, 2)
    return BenchmarkResult(
        benchmark_name="truthfulqa",
        score=score,
        num_examples=len(examples),
        correct=correct,
        details={"format": "multiple_choice", "metric": "mc1_accuracy"},
    )


def _load_truthfulqa_examples() -> list[dict[str, object]]:
    """Load TruthfulQA MC1 examples from HuggingFace datasets."""
    from core.errors import CrucibleDependencyError
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as error:
        raise CrucibleDependencyError(
            "TruthfulQA benchmark requires the datasets package. "
            "Install with: pip install datasets"
        ) from error
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    examples = []
    for r in ds:
        mc1_targets = r["mc1_targets"]
        choices = mc1_targets["choices"]
        labels = mc1_targets["labels"]
        correct_idx = labels.index(1) if 1 in labels else 0
        examples.append({
            "question": r["question"],
            "choices": choices,
            "correct_idx": correct_idx,
        })
    return examples
