"""Safety evaluation typed models.

Frozen dataclasses for toxicity scoring, safety gating,
alignment reporting, and red-team evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyEvalConfig:
    """Configuration for a safety evaluation run.

    Attributes:
        model_path: Path to the model weights file.
        eval_data_path: Path to evaluation data.
        output_dir: Directory for safety report output.
        toxicity_threshold: Score above which text is flagged.
        max_samples: Maximum number of samples to evaluate.
    """

    model_path: str
    eval_data_path: str
    output_dir: str
    toxicity_threshold: float = 0.5
    max_samples: int = 100


@dataclass(frozen=True)
class ToxicityScore:
    """Toxicity score for a single text sample.

    Attributes:
        text: The evaluated text.
        score: Toxicity score in [0, 1].
        flagged: Whether the score exceeds the threshold.
    """

    text: str
    score: float
    flagged: bool


@dataclass(frozen=True)
class SafetyReport:
    """Aggregated safety evaluation report.

    Attributes:
        total_samples: Number of texts evaluated.
        flagged_count: Number of texts flagged as toxic.
        mean_score: Mean toxicity score across all samples.
        max_score: Maximum toxicity score observed.
        scores: Individual toxicity scores per sample.
    """

    total_samples: int
    flagged_count: int
    mean_score: float
    max_score: float
    scores: tuple[ToxicityScore, ...]


@dataclass(frozen=True)
class SafetyGateResult:
    """Result of a pre-deploy safety gate check.

    Attributes:
        passed: Whether the gate check passed.
        report: Underlying safety report.
        threshold: Toxicity threshold used for the gate.
        gate_name: Identifier for this gate check.
    """

    passed: bool
    report: SafetyReport
    threshold: float
    gate_name: str


@dataclass(frozen=True)
class RedTeamResult:
    """Result of a red-team evaluation suite.

    Attributes:
        suite_name: Name of the red-team suite.
        total_prompts: Number of adversarial prompts tested.
        failures: Number of prompts where the model failed to refuse.
        failure_rate: Fraction of prompts that were not refused.
    """

    suite_name: str
    total_prompts: int
    failures: int
    failure_rate: float
