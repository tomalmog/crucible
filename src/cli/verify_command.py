"""Verification command wiring for Forge CLI."""

from __future__ import annotations

import argparse
from typing import cast

from core.errors import ForgeVerificationError
from core.verification import (
    VerificationMode,
    VerificationOptions,
    render_verification_report,
    run_verification,
    save_verification_report,
)
from store.dataset_sdk import ForgeClient


def add_verify_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register verify subcommand."""
    parser = subparsers.add_parser(
        "verify",
        help="Run automated end-to-end verification checks",
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="Verification mode",
    )
    parser.add_argument(
        "--source",
        default="tests/fixtures/raw_valid",
        help="Source path used for ingest checks",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep runtime verification data root even when all checks pass",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed check",
    )


def run_verify_command(client: ForgeClient, args: argparse.Namespace) -> int:
    """Execute verification workflow and print check report."""
    options = VerificationOptions(
        mode=cast(VerificationMode, args.mode),
        source_path=args.source,
        keep_artifacts=args.keep_artifacts,
        fail_fast=args.fail_fast,
    )
    try:
        report = run_verification(client, options)
    except ForgeVerificationError as error:
        print(f"verification_error={error}")
        return 1
    report_path = save_verification_report(report)
    print(render_verification_report(report))
    print(f"report_path={report_path}")
    return 0 if report.failed_count == 0 else 1
