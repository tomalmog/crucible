"""WinoGrande benchmark implementation.

WinoGrande — evaluates commonsense reasoning through
pronoun resolution tasks. Each example has a sentence with
a blank and two candidate completions.
"""

from __future__ import annotations

from eval.benchmark_runner import BenchmarkResult
from eval.benchmarks._model_loader import EvalModel, compute_completion_loss, load_eval_model


def run_winogrande(
    model_path: str,
    *,
    max_samples: int | None = None,
    eval_model: EvalModel | None = None,
) -> BenchmarkResult:
    """Run WinoGrande benchmark against a model.

    For each example, computes sequence loss for each option
    filling the blank and picks the lower-loss completion.
    """
    if eval_model is None:
        eval_model = load_eval_model(model_path)
    examples = _load_winogrande_examples()
    if max_samples:
        examples = examples[:max_samples]
    correct = 0
    total = len(examples)
    for idx, example in enumerate(examples):
        sentence = example["sentence"]
        blank_idx = sentence.find("_")
        prefix = sentence[:blank_idx] if blank_idx >= 0 else ""
        suffix = sentence[blank_idx + 1:] if blank_idx >= 0 else ""
        completion1 = str(example["option1"]) + suffix
        completion2 = str(example["option2"]) + suffix
        loss1 = compute_completion_loss(eval_model, prefix, completion1)
        loss2 = compute_completion_loss(eval_model, prefix, completion2)
        predicted = 1 if loss1 <= loss2 else 2
        if predicted == example["answer"]:
            correct += 1
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(
                f"  winogrande: {idx + 1}/{total} examples, "
                f"{correct} correct",
                flush=True,
            )
    total = max(total, 1)
    score = round((correct / total) * 100, 2)
    return BenchmarkResult(
        benchmark_name="winogrande",
        score=score,
        num_examples=len(examples),
        correct=correct,
        details={"format": "binary_choice", "metric": "accuracy"},
    )


def _load_winogrande_examples() -> list[dict[str, object]]:
    """Load WinoGrande examples from HuggingFace datasets."""
    from core.errors import CrucibleDependencyError
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as error:
        raise CrucibleDependencyError(
            "WinoGrande benchmark requires the datasets package. "
            "Install with: pip install datasets"
        ) from error
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    return [{"sentence": r["sentence"], "option1": r["option1"],
             "option2": r["option2"], "answer": int(r["answer"])} for r in ds]
