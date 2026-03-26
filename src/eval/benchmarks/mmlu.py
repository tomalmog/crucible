"""MMLU benchmark implementation.

Massive Multitask Language Understanding — tests knowledge across
57 subjects including STEM, humanities, social sciences, and more.
Uses multiple-choice scoring by comparing logits for A/B/C/D tokens.
"""

from __future__ import annotations

from eval.benchmark_runner import BenchmarkResult
from eval.benchmarks._model_loader import EvalModel, compute_logits, load_eval_model


def run_mmlu(
    model_path: str,
    *,
    max_samples: int | None = None,
    eval_model: EvalModel | None = None,
) -> BenchmarkResult:
    """Run MMLU benchmark against a model.

    Loads the MMLU dataset, formats each question as a multiple-choice
    prompt, and scores by comparing model logits for answer tokens.
    """
    if eval_model is None:
        eval_model = load_eval_model(model_path)
    examples = _load_mmlu_examples()
    if max_samples:
        examples = examples[:max_samples]
    correct = 0
    total = len(examples)
    for idx, example in enumerate(examples):
        prompt = _format_prompt(example)
        predicted = _score_choices(eval_model, prompt)
        if predicted == example["answer"]:
            correct += 1
        if (idx + 1) % 500 == 0 or idx + 1 == total:
            print(
                f"  mmlu: {idx + 1}/{total} questions, "
                f"{correct} correct",
                flush=True,
            )
    total = max(total, 1)
    score = round((correct / total) * 100, 2)
    return BenchmarkResult(
        benchmark_name="mmlu",
        score=score,
        num_examples=len(examples),
        correct=correct,
        details={"subjects": 57, "format": "multiple_choice"},
    )


def _load_mmlu_examples() -> list[dict[str, object]]:
    """Load MMLU examples from HuggingFace datasets."""
    from core.errors import CrucibleDependencyError
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as error:
        raise CrucibleDependencyError(
            "MMLU benchmark requires the datasets package. "
            "Install with: pip install datasets"
        ) from error
    ds = load_dataset("cais/mmlu", "all", split="test")
    return [{"question": r["question"], "choices": r["choices"],
             "answer": r["answer"], "subject": r["subject"]} for r in ds]


def _format_prompt(example: dict[str, object]) -> str:
    """Format an MMLU example as a multiple-choice prompt."""
    choices = example["choices"]
    labels = ["A", "B", "C", "D"]
    options = "\n".join(f"{labels[i]}. {choices[i]}" for i in range(len(choices)))
    return f"Question: {example['question']}\n{options}\nAnswer:"


def _score_choices(eval_model: EvalModel, prompt: str) -> int:
    """Return the index of the highest-scoring choice token."""
    logits = compute_logits(eval_model, prompt)
    choice_tokens = ["A", "B", "C", "D"]
    scores = []
    for token in choice_tokens:
        token_ids = eval_model.tokenizer.encode(token, 10)
        tid = token_ids[-1] if token_ids else 0
        scores.append(float(logits[tid].item()))
    return int(max(range(len(scores)), key=lambda i: scores[i]))
