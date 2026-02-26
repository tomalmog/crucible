"""CLI command for hardware profile detection."""

from __future__ import annotations

import argparse

from serve.hardware_profile import detect_hardware_profile


def add_hardware_profile_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register hardware-profile subcommand."""
    subparsers.add_parser(
        "hardware-profile",
        help="Detect accelerator hardware and suggested training defaults",
    )


def run_hardware_profile_command() -> int:
    """Print detected hardware profile as key=value rows."""
    profile = detect_hardware_profile().to_dict()
    for key in sorted(profile.keys()):
        print(f"{key}={profile[key]}")
    return 0
