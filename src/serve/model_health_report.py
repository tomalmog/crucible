"""Report builders for model health suites."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from serve.model_health_assessment import assess_health_result
from serve.model_health_suite import (
    ModelHealthCheckCommand,
    ModelHealthCheckOptions,
    health_suite_title,
)


@dataclass(frozen=True)
class HealthCheckRunReport:
    """Report entry for one health check inside a suite."""

    check_id: str
    label: str
    why: str
    status: str
    summary: str
    implication: str
    recommended_action: str
    severity: str
    output_dir: str
    result: dict[str, object] | None = None
    error: str = ""


@dataclass(frozen=True)
class ModelHealthSuiteRunReport:
    """Single output payload for a model health suite."""

    status: str
    job_type: str
    workflow: str
    suite_id: str
    suite_title: str
    overall_result: str
    model_path: str
    dataset_name: str
    probe_text: str
    clean_text: str
    corrupted_text: str
    label_field: str
    max_samples: int
    checks: tuple[HealthCheckRunReport, ...]
    plain_english_summary: str
    report_path: str


def completed_check_report(
    command: ModelHealthCheckCommand,
    result: dict[str, object],
) -> HealthCheckRunReport:
    """Build a report entry for a successful health check."""
    assessment = assess_health_result(command.tool_name, result)
    return HealthCheckRunReport(
        check_id=command.tool_name,
        label=str(command.config["healthCheckLabel"]),
        why=str(command.config["healthCheckReason"]),
        status="completed",
        summary=assessment["summary"],
        implication=assessment["implication"],
        recommended_action=assessment["recommended_action"],
        severity=assessment["severity"],
        output_dir=_str_arg(command.args, "output_dir"),
        result=result,
    )


def failed_check_report(command: ModelHealthCheckCommand, error: Exception) -> HealthCheckRunReport:
    """Build a report entry for a failed health check."""
    return HealthCheckRunReport(
        check_id=command.tool_name,
        label=str(command.config["healthCheckLabel"]),
        why=str(command.config["healthCheckReason"]),
        status="failed",
        summary=f"This check failed before producing a usable result: {error}",
        implication="The assessment is incomplete, so promotion confidence should be reduced.",
        recommended_action="Fix the failed diagnostic input or runtime issue, then rerun the health check.",
        severity="critical",
        output_dir=_str_arg(command.args, "output_dir"),
        error=str(error),
    )


def build_suite_report(
    suite_id: str,
    options: ModelHealthCheckOptions,
    check_reports: tuple[HealthCheckRunReport, ...],
) -> dict[str, object]:
    """Build, write, and return the suite-level health report."""
    path = report_path_for(options.output_dir)
    report = ModelHealthSuiteRunReport(
        status=_suite_status(check_reports),
        job_type="model-health-check",
        workflow="model-health-check",
        suite_id=suite_id,
        suite_title=health_suite_title(suite_id),
        overall_result=_overall_result(check_reports),
        model_path=options.model_path,
        dataset_name=options.dataset_name,
        probe_text=options.probe_text,
        clean_text=options.clean_text,
        corrupted_text=options.corrupted_text,
        label_field=options.label_field,
        max_samples=options.max_samples,
        checks=check_reports,
        plain_english_summary=_plain_english_summary(check_reports),
        report_path=str(path),
    )
    payload = _json_ready(report)
    _write_report(path, payload)
    return payload


def report_path_for(output_dir: str) -> Path:
    """Return the single report artifact path for a suite."""
    return Path(output_dir).expanduser().resolve() / "model_health_report.json"


def _plain_english_summary(reports: tuple[HealthCheckRunReport, ...]) -> str:
    completed = [report for report in reports if report.status == "completed"]
    failed = [report for report in reports if report.status != "completed"]
    review_items = [
        report for report in reports
        if report.severity in ("critical", "warning", "review")
    ]
    lead = f"Completed {len(completed)} of {len(reports)} model-health checks."
    if failed:
        names = ", ".join(report.label for report in failed)
        return f"{lead} Resolve failed checks before promotion: {names}."
    if review_items:
        names = ", ".join(report.label for report in review_items[:3])
        return f"{lead} Promotion review should focus on: {names}."
    return f"{lead} No blocking health-check findings were detected; continue with eval and product review."


def _suite_status(reports: tuple[HealthCheckRunReport, ...]) -> str:
    if any(report.status != "completed" for report in reports):
        return "completed_with_errors"
    return "completed"


def _overall_result(reports: tuple[HealthCheckRunReport, ...]) -> str:
    if any(report.severity == "critical" for report in reports):
        return "Do not promote yet"
    if any(report.severity in ("warning", "review") for report in reports):
        return "Review before promotion"
    return "Ready for promotion review"


def _json_ready(report: ModelHealthSuiteRunReport) -> dict[str, object]:
    return cast(dict[str, object], json.loads(json.dumps(asdict(report))))


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _str_arg(args: dict[str, object], key: str) -> str:
    value = args.get(key)
    return value if isinstance(value, str) else ""
