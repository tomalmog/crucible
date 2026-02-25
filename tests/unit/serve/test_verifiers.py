"""Tests for code and math verifiers."""

from __future__ import annotations

from serve.verifiers.code_verifier import verify_code
from serve.verifiers.math_verifier import extract_answer, verify_math


def test_verify_code_nonempty() -> None:
    result = verify_code("print('hello')", [{"input": "", "output": "hello"}])
    assert result.passed is True


def test_verify_code_empty() -> None:
    result = verify_code("", [])
    assert result.passed is False


def test_extract_answer_boxed() -> None:
    assert extract_answer("The answer is \\boxed{42}") == "42"


def test_extract_answer_number() -> None:
    assert extract_answer("The result is 3.14") == "3.14"


def test_verify_math_correct() -> None:
    result = verify_math("The answer is \\boxed{42}", "42")
    assert result.correct is True


def test_verify_math_numeric() -> None:
    result = verify_math("I got 3.14", "3.14")
    assert result.correct is True


def test_verify_math_incorrect() -> None:
    result = verify_math("The answer is 5", "10")
    assert result.correct is False
