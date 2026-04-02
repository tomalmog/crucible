"""HumanEval benchmark implementation.

HumanEval — evaluates code generation capability with
164 hand-crafted Python programming problems. Generates
function completions and runs test cases in isolated subprocesses.
"""

from __future__ import annotations

import resource
import subprocess
import tempfile
from pathlib import Path

from eval.benchmark_runner import BenchmarkResult
from eval.benchmarks._model_loader import EvalModel, generate_text, load_eval_model


_EXEC_TIMEOUT_SECONDS = 10
_EXEC_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024  # 512 MB


def run_humaneval(
    model_path: str,
    *,
    max_samples: int | None = None,
    eval_model: EvalModel | None = None,
) -> BenchmarkResult:
    """Run HumanEval benchmark against a model.

    Generates function body completions for each problem and
    executes test cases in sandboxed subprocesses (pass@1).
    """
    if eval_model is None:
        eval_model = load_eval_model(model_path)
    examples = _load_humaneval_examples()
    if max_samples:
        examples = examples[:max_samples]
    correct = 0
    total = len(examples)
    for idx, example in enumerate(examples):
        prompt = str(example["prompt"])
        completion = generate_text(eval_model, prompt, max_new_tokens=256)
        full_code = prompt + completion
        test_code = str(example["test"])
        entry_point = str(example["entry_point"])
        if _check_solution(full_code, test_code, entry_point):
            correct += 1
        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(
                f"  humaneval: {idx + 1}/{total} problems, "
                f"{correct} passed",
                flush=True,
            )
    total = max(total, 1)
    score = round((correct / total) * 100, 2)
    return BenchmarkResult(
        benchmark_name="humaneval",
        score=score,
        num_examples=len(examples),
        correct=correct,
        details={"language": "python", "metric": "pass@1"},
    )


def _limit_subprocess_memory() -> None:
    """Cap virtual memory so generated code cannot OOM the parent process."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_EXEC_MEMORY_LIMIT_BYTES, _EXEC_MEMORY_LIMIT_BYTES))
    except (ValueError, OSError):
        # macOS does not support lowering RLIMIT_AS — skip memory limit
        pass


def _check_solution(code: str, test_code: str, entry_point: str) -> bool:
    """Execute generated code with test cases in an isolated subprocess."""
    script = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write(script)
            tmp_path = Path(tmp.name)
        result = subprocess.run(
            ["python", str(tmp_path)],
            capture_output=True,
            timeout=_EXEC_TIMEOUT_SECONDS,
            preexec_fn=_limit_subprocess_memory,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _load_humaneval_examples() -> list[dict[str, object]]:
    """Load HumanEval examples from HuggingFace datasets."""
    from core.errors import CrucibleDependencyError
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as error:
        raise CrucibleDependencyError(
            "HumanEval benchmark requires the datasets package. "
            "Install with: pip install datasets"
        ) from error
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [{"prompt": r["prompt"], "test": r["test"],
             "entry_point": r["entry_point"]} for r in ds]
