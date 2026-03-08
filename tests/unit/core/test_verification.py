"""Unit tests for verification workflow helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.chat_types import ChatOptions, ChatResult
from core.types import IngestOptions, TrainingOptions, TrainingRunResult
from core.verification import (
    VerificationOptions,
    render_verification_report,
    run_verification,
)


@dataclass(frozen=True)
class _FakeRunRecord:
    run_id: str
    state: str
    artifact_contract_path: str | None


class _FakeDataset:
    def __init__(self, client: "_FakeClient", dataset_name: str) -> None:
        self._client = client
        self._dataset_name = dataset_name

    def load_records(self) -> tuple[object, list[dict[str, str]]]:
        return (None, [{"text": "record"}])

    def export_training(
        self,
        output_dir: str,
        shard_size: int = 1000,
        include_metadata: bool = False,
    ) -> str:
        _ = shard_size
        _ = include_metadata
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_path = output_path / "training_manifest.json"
        manifest_path.write_text('{"ok": true}\n', encoding="utf-8")
        return str(manifest_path)


class _FakeClient:
    def __init__(self) -> None:
        self.data_root = Path.cwd()
        self.train_run_id = "run-verify-123"
        self.train_model_path = ""
        self.run_record = _FakeRunRecord(
            run_id=self.train_run_id,
            state="completed",
            artifact_contract_path=None,
        )

    def with_data_root(self, data_root: str) -> "_FakeClient":
        self.data_root = Path(data_root).resolve()
        return self

    def ingest(self, options: IngestOptions) -> str:
        _ = options
        return options.dataset_name

    def dataset(self, dataset_name: str) -> _FakeDataset:
        _ = dataset_name
        return _FakeDataset(self, dataset_name)

    def train(self, options: TrainingOptions) -> TrainingRunResult:
        output_dir = Path(options.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pt"
        history_path = output_dir / "history.json"
        contract_path = output_dir / "training_artifacts_manifest.json"
        config_path = output_dir / "training_config.json"
        tokenizer_path = output_dir / "tokenizer_vocab.json"
        repro_path = output_dir / "reproducibility_bundle.json"
        model_path.write_text("weights", encoding="utf-8")
        history_path.write_text('{"epochs": [], "batch_losses": []}\n', encoding="utf-8")
        config_path.write_text('{"ok": true}\n', encoding="utf-8")
        tokenizer_path.write_text('{"<pad>": 0}\n', encoding="utf-8")
        repro_path.write_text('{"ok": true}\n', encoding="utf-8")
        contract_path.write_text('{"ok": true}\n', encoding="utf-8")
        self.train_model_path = str(model_path)
        self.run_record = _FakeRunRecord(
            run_id=self.train_run_id,
            state="completed",
            artifact_contract_path=str(contract_path),
        )
        return TrainingRunResult(
            model_path=str(model_path),
            history_path=str(history_path),
            plot_path=None,
            epochs_completed=1,
            run_id=self.train_run_id,
            artifact_contract_path=str(contract_path),
        )

    def chat(self, options: ChatOptions) -> ChatResult:
        _ = options
        return ChatResult(response_text="verification response")

    def run_spec(self, spec_file: str) -> tuple[str, ...]:
        if not Path(spec_file).exists():
            return ()
        return ("a", "b", "c", "d")

    def hardware_profile(self) -> dict[str, object]:
        return {
            "accelerator": "cpu",
            "gpu_count": 0,
            "recommended_precision_mode": "fp32",
        }

    def list_training_runs(self) -> tuple[str, ...]:
        return (self.train_run_id,)

    def get_training_run(self, run_id: str) -> _FakeRunRecord:
        _ = run_id
        return self.run_record

    def get_lineage_graph(self) -> dict[str, object]:
        return {
            "runs": {
                self.train_run_id: {
                    "model_path": self.train_model_path,
                }
            },
            "edges": [
                {
                    "from": f"run:{self.train_run_id}",
                    "to": f"model:{self.train_model_path}",
                    "type": "produced",
                }
            ],
        }


def test_run_verification_quick_reports_all_checks_pass() -> None:
    """Quick mode should return report with no failed checks for healthy client."""
    report = run_verification(
        client=_FakeClient(),
        options=VerificationOptions(
            mode="quick",
            source_path="tests/fixtures/raw_valid",
            keep_artifacts=False,
            fail_fast=False,
        ),
    )
    rendered = render_verification_report(report)
    report_path = Path(report.runtime_data_root) / "verification_report.json"
    if report.artifacts_kept:
        report_path.write_text(json.dumps({"kept": True}) + "\n", encoding="utf-8")

    assert report.failed_count == 0 and "passed=" in rendered
