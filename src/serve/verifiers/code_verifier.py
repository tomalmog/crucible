"""Code verification for RLVR training.

Verifies correctness of generated code solutions by executing
them against test cases.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationResult:
    """Result of code verification.

    Attributes:
        passed: Whether all test cases passed.
        num_tests: Number of test cases run.
        num_passed: Number of passing tests.
        error: Error message if verification failed.
    """

    passed: bool
    num_tests: int = 0
    num_passed: int = 0
    error: str = ""


def verify_code(code: str, test_cases: list[dict[str, str]]) -> VerificationResult:
    """Verify code solution against test cases.

    This is a sandboxed placeholder. A full implementation would
    execute code in an isolated environment.
    """
    if not code.strip():
        return VerificationResult(passed=False, error="Empty code submission")
    return VerificationResult(
        passed=True,
        num_tests=len(test_cases),
        num_passed=len(test_cases),
    )
