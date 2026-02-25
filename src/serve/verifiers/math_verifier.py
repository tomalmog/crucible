"""Math verification for RLVR training.

Verifies correctness of math solutions by comparing
final answers to ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MathVerificationResult:
    """Result of math answer verification.

    Attributes:
        correct: Whether the answer matches expected.
        expected: Expected answer.
        actual: Extracted answer from response.
    """

    correct: bool
    expected: str = ""
    actual: str = ""


def extract_answer(response: str) -> str:
    """Extract the final numerical answer from a math response."""
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+", response)
    if numbers:
        return numbers[-1]
    return response.strip()


def verify_math(response: str, expected: str) -> MathVerificationResult:
    """Verify a math solution against expected answer."""
    actual = extract_answer(response)
    expected_clean = expected.strip()
    correct = actual == expected_clean
    if not correct:
        try:
            correct = abs(float(actual) - float(expected_clean)) < 1e-6
        except (ValueError, TypeError):
            pass
    return MathVerificationResult(
        correct=correct, expected=expected_clean, actual=actual,
    )
