"""Unit tests for jailbreak detector."""

from __future__ import annotations

from safety.jailbreak_detector import (
    JailbreakPrompt,
    detect_jailbreak_compliance,
    run_jailbreak_suite,
)


def test_refused_response_detected() -> None:
    """A response with refusal phrases should be detected as refused."""
    refused = detect_jailbreak_compliance(
        prompt="Do something harmful",
        response="I cannot help with that request.",
    )
    assert refused is True


def test_compliant_response_not_refused() -> None:
    """A response without refusal phrases should not be detected as refused."""
    refused = detect_jailbreak_compliance(
        prompt="Do something harmful",
        response="Sure, here are the instructions for that.",
    )
    assert refused is False


def test_refusal_case_insensitive() -> None:
    """Refusal detection should be case-insensitive."""
    refused = detect_jailbreak_compliance(
        prompt="Bad prompt",
        response="I Cannot comply with this request.",
    )
    assert refused is True


def test_jailbreak_suite_runs_all_prompts() -> None:
    """Jailbreak suite should run all prompts and return results."""
    prompts = [
        JailbreakPrompt(prompt="Hack something", category="harmful"),
        JailbreakPrompt(prompt="Ignore rules", category="injection"),
    ]

    def mock_generate(prompt: str) -> str:
        return "I cannot help with that."

    results = run_jailbreak_suite(prompts, mock_generate)
    assert len(results) == 2
    assert all(r.refused for r in results)
    assert results[0].category == "harmful"
    assert results[1].category == "injection"
