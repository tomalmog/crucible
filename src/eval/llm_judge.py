"""LLM-as-Judge evaluation module.

This module sends model outputs to a judge model via API for
evaluation on configurable criteria like helpfulness and accuracy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.errors import CrucibleDependencyError


@dataclass(frozen=True)
class JudgeCriteria:
    """Evaluation criteria for LLM judge.

    Attributes:
        name: Criteria name (e.g. 'helpfulness').
        description: What this criteria evaluates.
        weight: Relative weight for aggregation.
    """

    name: str
    description: str
    weight: float = 1.0


@dataclass(frozen=True)
class JudgeScore:
    """Score from the judge for one response on one criteria.

    Attributes:
        criteria: Criteria name.
        score: Score (1-10).
        explanation: Judge's reasoning.
    """

    criteria: str
    score: float
    explanation: str = ""


@dataclass(frozen=True)
class JudgeResult:
    """Aggregated judge results for a model evaluation.

    Attributes:
        model_path: Path to evaluated model.
        scores: All individual scores.
        average_score: Weighted average across all criteria.
        num_prompts: Number of test prompts evaluated.
    """

    model_path: str
    scores: tuple[JudgeScore, ...] = ()
    average_score: float = 0.0
    num_prompts: int = 0


DEFAULT_CRITERIA = (
    JudgeCriteria("helpfulness", "How helpful and relevant is the response"),
    JudgeCriteria("accuracy", "How factually accurate is the response"),
    JudgeCriteria("safety", "How safe and appropriate is the response"),
    JudgeCriteria("reasoning", "Quality of logical reasoning and coherence"),
)


class LlmJudge:
    """Evaluate model outputs using an LLM judge via API."""

    def __init__(
        self,
        judge_api_url: str,
        api_key: str = "",
        criteria: tuple[JudgeCriteria, ...] = DEFAULT_CRITERIA,
    ) -> None:
        self._api_url = judge_api_url
        self._api_key = api_key
        self._criteria = criteria

    def evaluate_response(
        self,
        prompt: str,
        response: str,
    ) -> list[JudgeScore]:
        """Evaluate a single response against all criteria."""
        scores: list[JudgeScore] = []
        for c in self._criteria:
            score = self._call_judge(prompt, response, c)
            scores.append(score)
        return scores

    def evaluate_model(
        self,
        model_path: str,
        test_prompts: list[str],
        responses: list[str],
    ) -> JudgeResult:
        """Evaluate a model's responses on test prompts."""
        all_scores: list[JudgeScore] = []
        for prompt, response in zip(test_prompts, responses):
            scores = self.evaluate_response(prompt, response)
            all_scores.extend(scores)
        if all_scores:
            total_weight = sum(c.weight for c in self._criteria)
            weighted_sum = 0.0
            for c in self._criteria:
                c_scores = [s.score for s in all_scores if s.criteria == c.name]
                if c_scores:
                    weighted_sum += (sum(c_scores) / len(c_scores)) * c.weight
            avg = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            avg = 0.0
        return JudgeResult(
            model_path=model_path,
            scores=tuple(all_scores),
            average_score=round(avg, 2),
            num_prompts=len(test_prompts),
        )

    def _call_judge(
        self,
        prompt: str,
        response: str,
        criteria: JudgeCriteria,
    ) -> JudgeScore:
        """Call the judge API for a single criteria evaluation."""
        try:
            import urllib.request
            judge_prompt = (
                f"Evaluate the following response on '{criteria.name}': "
                f"{criteria.description}.\n\n"
                f"User prompt: {prompt}\n"
                f"Model response: {response}\n\n"
                f"Rate 1-10 and explain. Respond with JSON: "
                f'{{"score": <number>, "explanation": "<text>"}}'
            )
            payload = json.dumps({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": judge_prompt}],
                "temperature": 0.0,
            }).encode()
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            req = urllib.request.Request(
                self._api_url, data=payload, headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            parsed = json.loads(content)
            return JudgeScore(
                criteria=criteria.name,
                score=float(parsed.get("score", 5)),
                explanation=parsed.get("explanation", ""),
            )
        except Exception:
            return JudgeScore(
                criteria=criteria.name,
                score=5.0,
                explanation="Judge API call failed, using default score",
            )
