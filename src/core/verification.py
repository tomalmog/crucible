"""Verification workflow orchestration and report formatting."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable

from core.verification_checks import build_checks, build_runtime
from core.verification_types import (
    VerificationCheckResult,
    VerificationMode,
    VerificationOptions,
    VerificationReport,
    VerificationRuntime,
    VerificationStatus,
)
from store.dataset_sdk import CrucibleClient

__all__ = [
    "VerificationCheckResult",
    "VerificationMode",
    "VerificationOptions",
    "VerificationReport",
    "run_verification",
    "render_verification_report",
    "save_verification_report",
]


def run_verification(client: CrucibleClient, options: VerificationOptions) -> VerificationReport:
    """Run verification checks and return structured report."""
    runtime = build_runtime(client, options.source_path)
    checks = build_checks(options.mode)
    results = _run_checks(runtime, checks, options.fail_fast)
    artifacts_kept = _should_keep_artifacts(options.keep_artifacts, results)
    if not artifacts_kept:
        shutil.rmtree(runtime.data_root, ignore_errors=True)
    return VerificationReport(
        mode=options.mode,
        runtime_data_root=str(runtime.data_root),
        artifacts_kept=artifacts_kept,
        checks=tuple(results),
    )


def _run_checks(
    runtime: VerificationRuntime,
    checks: tuple[tuple[str, str, Callable[[VerificationRuntime], str]], ...],
    fail_fast: bool,
) -> list[VerificationCheckResult]:
    results: list[VerificationCheckResult] = []
    for check_id, title, check_fn in checks:
        started_at = time.monotonic()
        status, details = _run_single_check(check_fn, runtime)
        results.append(
            VerificationCheckResult(
                check_id=check_id,
                title=title,
                status=status,
                details=details,
                duration_seconds=round(time.monotonic() - started_at, 3),
            )
        )
        if status == "failed" and fail_fast:
            break
    return results


def _run_single_check(
    check_fn: Callable[[VerificationRuntime], str],
    runtime: VerificationRuntime,
) -> tuple[VerificationStatus, str]:
    try:
        details = str(check_fn(runtime))
        return "passed", details
    except Exception as error:
        return "failed", str(error)


def _should_keep_artifacts(
    keep_requested: bool,
    results: list[VerificationCheckResult],
) -> bool:
    if keep_requested:
        return True
    return any(row.status == "failed" for row in results)


def render_verification_report(report: VerificationReport) -> str:
    """Render report into stable multi-line text for CLI output."""
    lines = [
        f"mode={report.mode}",
        f"runtime_data_root={report.runtime_data_root}",
        f"artifacts_kept={str(report.artifacts_kept).lower()}",
    ]
    for row in report.checks:
        lines.append(
            f"[{row.status.upper()}] {row.check_id} {row.title} "
            f"({row.duration_seconds:.3f}s) :: {row.details}"
        )
    lines.append(f"passed={report.passed_count}")
    lines.append(f"failed={report.failed_count}")
    return "\n".join(lines)


def save_verification_report(report: VerificationReport) -> Path:
    """Persist report JSON into runtime data root for debugging."""
    report_path = Path(report.runtime_data_root) / "verification_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": report.mode,
        "runtime_data_root": report.runtime_data_root,
        "artifacts_kept": report.artifacts_kept,
        "checks": [
            {
                "check_id": row.check_id,
                "title": row.title,
                "status": row.status,
                "details": row.details,
                "duration_seconds": row.duration_seconds,
            }
            for row in report.checks
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report_path
