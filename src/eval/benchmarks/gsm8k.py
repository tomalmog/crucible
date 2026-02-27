"""GSM8K benchmark implementation.

Grade School Math 8K — tests mathematical reasoning with
grade school math word problems. Scores by exact-match of the
final numeric answer extracted from model generation.
"""

from __future__ import annotations

import re

from eval.benchmark_runner import BenchmarkResult
from eval.benchmarks._model_loader import generate_text, load_eval_model


def run_gsm8k(model_path: str) -> BenchmarkResult:
    """Run GSM8K benchmark against a model.

    Generates a chain-of-thought response for each problem,
    extracts the final number, and checks exact match.
    """
    eval_model = load_eval_model(model_path)
    examples = _load_gsm8k_examples()
    correct = 0
    for example in examples:
        prompt = f"Question: {example['question']}\nAnswer: Let's solve step by step.\n"
        response = generate_text(eval_model, prompt, max_new_tokens=128)
        predicted = _extract_final_number(response)
        expected = _extract_final_number(str(example["answer"]))
        if predicted is not None and expected is not None:
            if abs(predicted - expected) < 1e-6:
                correct += 1
    total = max(len(examples), 1)
    score = round((correct / total) * 100, 2)
    return BenchmarkResult(
        benchmark_name="gsm8k",
        score=score,
        num_examples=len(examples),
        correct=correct,
        details={"format": "chain_of_thought", "metric": "exact_match"},
    )


def _extract_final_number(text: str) -> float | None:
    """Extract the last number from text, handling GSM8K #### format."""
    marker_match = re.search(r"####\s*([\d,.-]+)", text)
    if marker_match:
        return _parse_number(marker_match.group(1))
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return _parse_number(numbers[-1])
    return None


def _parse_number(text: str) -> float | None:
    """Parse a number string, stripping commas."""
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _load_gsm8k_examples() -> list[dict[str, object]]:
    """Load GSM8K examples from HuggingFace datasets."""
    from core.errors import ForgeDependencyError
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as error:
        raise ForgeDependencyError(
            "GSM8K benchmark requires the datasets package. "
            "Install with: pip install datasets"
        ) from error
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [{"question": r["question"], "answer": r["answer"]} for r in ds]
