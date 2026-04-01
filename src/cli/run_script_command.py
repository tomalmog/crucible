"""The ``crucible run-script`` command — execute a user training script."""

from __future__ import annotations

import argparse
import subprocess
import sys

from store.dataset_sdk import CrucibleClient


def add_run_script_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "run-script",
        help="Execute a Python training script with job tracking",
    )
    parser.add_argument("script", help="Path to the Python script to run")
    parser.add_argument("--model-name", default=None, help="Name for model registry")


def run_run_script_command(client: CrucibleClient, args: argparse.Namespace) -> int:
    """Execute a user training script as a subprocess with job tracking.

    The script runs in a subprocess with PYTHONPATH set so it can
    import crucible_sdk. The job tracking wrapper in main.py handles
    creating the job record and capturing stdout/stderr.
    """
    from pathlib import Path

    script_path = Path(args.script).expanduser().resolve()
    if not script_path.exists():
        print(f"Script not found: {script_path}", file=sys.stderr)
        return 1

    # Run the script with PYTHONPATH including src/ for crucible_sdk imports
    import os
    src_dir = str(Path(__file__).resolve().parent.parent)
    existing_pp = os.environ.get("PYTHONPATH", "")
    combined_pp = f"{src_dir}:{existing_pp}" if existing_pp else src_dir
    env = {**os.environ, "PYTHONPATH": combined_pp}

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=True,
        text=True,
    )

    # Print captured output so TeeWriter in main.py captures it into the job
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    if proc.returncode != 0:
        # Extract the last meaningful error line for the job error_message
        error_lines = [l for l in (proc.stderr or "").strip().splitlines() if l.strip()]
        error_msg = error_lines[-1] if error_lines else f"Script exited with code {proc.returncode}"
        raise RuntimeError(error_msg)

    return 0
