"""Unit tests for verify CLI command wiring."""

from __future__ import annotations

from dataclasses import dataclass

from cli.main import main
from core.verification import VerificationCheckResult, VerificationReport


@dataclass(frozen=True)
class _FakeReportPayload:
    report: VerificationReport


def _build_report(failed_count: int) -> VerificationReport:
    status = "failed" if failed_count > 0 else "passed"
    return VerificationReport(
        mode="quick",
        runtime_data_root="/tmp/crucible-verify",
        artifacts_kept=failed_count > 0,
        checks=(
            VerificationCheckResult(
                check_id="V001",
                title="check",
                status=status,
                details="ok",
                duration_seconds=0.01,
            ),
        ),
    )


def test_cli_verify_returns_zero_when_all_checks_pass(
    monkeypatch,
    capsys,
) -> None:
    """Verify command should exit zero on fully passing report."""
    fake_payload = _FakeReportPayload(report=_build_report(failed_count=0))
    monkeypatch.setattr(
        "cli.verify_command.run_verification",
        lambda client, options: fake_payload.report,
    )
    monkeypatch.setattr(
        "cli.verify_command.save_verification_report",
        lambda report: "/tmp/crucible-verify/verification_report.json",
    )

    exit_code = main(["verify"])
    output = capsys.readouterr().out

    assert exit_code == 0 and "report_path=" in output


def test_cli_verify_returns_one_when_any_check_fails(
    monkeypatch,
    capsys,
) -> None:
    """Verify command should exit one when report has failed checks."""
    fake_payload = _FakeReportPayload(report=_build_report(failed_count=1))
    monkeypatch.setattr(
        "cli.verify_command.run_verification",
        lambda client, options: fake_payload.report,
    )
    monkeypatch.setattr(
        "cli.verify_command.save_verification_report",
        lambda report: "/tmp/crucible-verify/verification_report.json",
    )

    exit_code = main(["verify"])
    _ = capsys.readouterr()

    assert exit_code == 1


def test_cli_verify_handles_input_validation_failures_without_traceback(
    capsys,
) -> None:
    """Verify command should print a friendly error for invalid source paths."""
    exit_code = main(["verify", "--source", "/tmp/crucible-verify-missing-source"])
    output = capsys.readouterr().out.strip()

    assert exit_code == 1 and output.startswith("verification_error=")
